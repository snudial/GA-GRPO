"""Common record schemas for rubric-episode analysis and agentic OPSD training.

Design notes
------------
* Plain stdlib ``dataclasses`` are used instead of pydantic so that every offline
  analysis script runs unchanged inside the AppWorld venv, the BFCL venv and the
  training venv (the three do not share a pydantic major version).
* Every record carries an explicit ``to_dict`` / ``from_dict`` pair. JSONL is the
  wire format described in the implementation plan section 15.
* Enum *values* are English snake_case (stable dict keys, safe for parquet column
  values). Korean display labels used in the plan are available through
  ``EpisodeType.korean`` and are additionally serialised as ``type_ko`` so that a
  report can be written in either language without a lookup table.

Nothing in this module imports torch. It must stay importable from a plain
CPython with no ML stack installed.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Iterable, Iterator, Mapping, Sequence

__all__ = [
    "RubricState",
    "RubricKind",
    "CostVerdict",
    "EpisodeType",
    "EpisodeCloseReason",
    "ClassificationStatus",
    "CoverageClass",
    "EvaluationStatus",
    "EvaluationPhase",
    "SCHEMA_VERSION",
    "EpisodeRole",
    "MaskMode",
    "FailureKind",
    "ToolCall",
    "StepRecord",
    "RolloutRecord",
    "RubricSpec",
    "RubricEvaluation",
    "RubricStateVector",
    "CostEvidence",
    "Episode",
    "RolloutOutcome",
    "read_jsonl",
    "write_jsonl",
    "to_jsonable",
]


SCHEMA_VERSION = "2.0"
"""Bumped when a record's meaning changes.

2.0: episode classification split into ``episode_type`` (the plan's three types,
``None`` unless confidently labelled) and ``classification_status`` (why it was
labelled or abstained). 1.0 folded abstentions into extra episode types, which let
an undecided case be read as a recovery.
"""


# --------------------------------------------------------------------------- #
# Enumerations
# --------------------------------------------------------------------------- #


class RubricState(str, Enum):
    """State of a single rubric at a single step (plan section 4.1)."""

    PENDING = "PENDING"
    """Required outcome has not happened yet. Not (yet) an error."""

    SATISFIED = "SATISFIED"
    """Requirement holds in the current world state."""

    VIOLATED = "VIOLATED"
    """A forbidden state occurred, or a previously satisfied condition broke."""

    UNKNOWN = "UNKNOWN"
    """Evaluator failed or the observation was insufficient. Never closes an episode."""


class RubricKind(str, Enum):
    """Whether the rubric is a positive goal or a safety/collateral constraint.

    The mapping to AppWorld's evaluator vocabulary is:
      * ``GOAL``   <- ``no_op_fail``  (doing nothing fails the test)
      * ``SAFETY`` <- ``no_op_pass``  (doing nothing passes the test)
    """

    GOAL = "goal"
    SAFETY = "safety"


class CostVerdict(str, Enum):
    """External-cost verdict for a recovered episode (plan section 4.3)."""

    NONE = "none"
    """No observable external footprint. -> clean recovery."""

    EXTERNAL = "external"
    """A real state-changing side effect or irreversible mutation occurred."""

    UNKNOWN = "unknown"
    """No evidence either way. Never silently coerced to ``NONE``."""


class EpisodeType(str, Enum):
    """The plan's three episode types (section 4.3), and only those.

    An episode carries a type **only** when it was confidently labelled; anything
    undecided is an abstention recorded in :class:`ClassificationStatus`, not a
    fourth type. Adding types for the undecided cases would let them be read as
    recoveries by any consumer that groups on this field.
    """

    CLEAN_RECOVERY = "clean_recovery"
    COSTLY_RECOVERY = "costly_recovery"
    UNRESOLVED_FAILURE = "unresolved_failure"

    @property
    def korean(self) -> str:
        return _EPISODE_TYPE_KO[self]

    @property
    def is_recovered(self) -> bool:
        return self is not EpisodeType.UNRESOLVED_FAILURE


_EPISODE_TYPE_KO: dict[EpisodeType, str] = {
    EpisodeType.CLEAN_RECOVERY: "완전 복구형",
    EpisodeType.COSTLY_RECOVERY: "비용 복구형",
    EpisodeType.UNRESOLVED_FAILURE: "실패 잔존형",
}


class ClassificationStatus(str, Enum):
    """Whether an episode was labelled, and if not, why not.

    Every episode gets one. Only ``LABELED`` episodes carry an ``episode_type`` and
    enter the headline three-way statistics; the rest are kept, counted and
    reported so the denominator is always visible.
    """

    LABELED = "labeled"

    ABSTAINED_UNKNOWN_STATE = "abstained_unknown_state"
    """The closure could not be attributed: observations were missing, or both the
    requirement and the world moved."""

    ABSTAINED_UNCOVERED_EFFECT = "abstained_uncovered_effect"
    """An effect occurred that no rubric watches, so the episode boundary cannot be
    trusted."""

    ABSTAINED_COST_UNKNOWN = "abstained_cost_unknown"
    """Recovered, but a relevant cost dimension was unobservable."""

    EXCLUDED_SUPERSEDED = "excluded_superseded"
    """Closed because the requirement moved onto the unrepaired world."""

    EXCLUDED_UNSCOREABLE = "excluded_unscoreable"
    """The benchmark entry itself cannot be scored."""

    @property
    def is_labeled(self) -> bool:
        return self is ClassificationStatus.LABELED

    @property
    def is_abstention(self) -> bool:
        return self.value.startswith("abstained_")

    @property
    def is_exclusion(self) -> bool:
        return self.value.startswith("excluded_")

    @property
    def korean(self) -> str:
        return _CLASSIFICATION_STATUS_KO[self]


_CLASSIFICATION_STATUS_KO: dict[ClassificationStatus, str] = {
    ClassificationStatus.LABELED: "분류됨",
    ClassificationStatus.ABSTAINED_UNKNOWN_STATE: "판정 보류: 상태 미상",
    ClassificationStatus.ABSTAINED_UNCOVERED_EFFECT: "판정 보류: 미관측 효과",
    ClassificationStatus.ABSTAINED_COST_UNKNOWN: "판정 보류: 비용 미상",
    ClassificationStatus.EXCLUDED_SUPERSEDED: "제외: 목표 변경",
    ClassificationStatus.EXCLUDED_UNSCOREABLE: "제외: 채점 불가",
}


class CoverageClass(str, Enum):
    """How well a rubric, effect or boundary can be decided (stage 5.8)."""

    DECIDABLE = "decidable"
    """Turn-local evidence exists; an episode boundary here is trustworthy."""

    TERMINAL_ONLY = "terminal_only"
    """Only the terminal evaluator can judge it, so no boundary claim is made."""

    UNCOVERED = "uncovered"
    """Nothing observes it. Never silently treated as satisfied."""


class EvaluationStatus(str, Enum):
    VALID = "valid"
    UNKNOWN = "unknown"
    INVALID = "invalid"


class EvaluationPhase(str, Enum):
    """When in a step a rubric was evaluated."""

    PRE_ACTION = "pre_action"
    POST_ACTION = "post_action"
    TERMINAL = "terminal"


class EpisodeCloseReason(str, Enum):
    """Why an episode stopped being violated.

    The distinction exists because a rubric whose expectation advances with the
    benchmark's turns can stop being violated for two entirely different reasons:
    the policy repaired the world, or the next turn's requirement happened to
    match the broken world. Only the first is recovery.
    """

    RECOVERED_BY_ACTION = "RECOVERED_BY_ACTION"
    """The observed world changed and now satisfies the rubric."""

    SUPERSEDED_BY_TARGET = "SUPERSEDED_BY_TARGET"
    """The observed world did not change; the expected value moved onto it."""

    TERMINAL_UNRESOLVED = "TERMINAL_UNRESOLVED"
    """Still violated when the rollout ended."""

    UNKNOWN = "UNKNOWN"
    """Closed, but not attributable: either the evaluation carried no
    expected/actual values, or both the requirement and the world moved and the
    benchmark's granularity cannot say which caused the closure. Never counted as
    recovery."""


_CLOSE_REASON_KO: dict[EpisodeCloseReason, str] = {
    EpisodeCloseReason.RECOVERED_BY_ACTION: "행동에 의한 복구",
    EpisodeCloseReason.SUPERSEDED_BY_TARGET: "목표 변경으로 소멸",
    EpisodeCloseReason.TERMINAL_UNRESOLVED: "미해결 종료",
    EpisodeCloseReason.UNKNOWN: "판정 불가",
}


class EpisodeRole(str, Enum):
    """Per-token analysis role (plan section 12.2). Never changes the loss."""

    NORMAL = "NORMAL"
    OPENING_ACTION = "OPENING_ACTION"
    ACTIVE_SUFFIX = "ACTIVE_SUFFIX"
    RECOVERY_ACTION = "RECOVERY_ACTION"
    POST_RECOVERY = "POST_RECOVERY"
    UNRESOLVED_SUFFIX = "UNRESOLVED_SUFFIX"


class MaskMode(str, Enum):
    """The three loss masks compared in the paired audit (plan section 10)."""

    PREFIX_TO_FIRST_VIOLATION = "prefix_to_first_violation"
    FULL_ROLLOUT = "full_rollout"
    POST_FIRST_VIOLATION = "post_first_violation"


class FailureKind(str, Enum):
    """Why a rollout failed, when it failed."""

    NONE = "none"
    LOCALIZED_VIOLATION = "localized_violation"
    """At least one hard rubric transitioned into VIOLATED at an identifiable step."""

    UNLOCALIZED_OMISSION = "unlocalized_omission"
    """PENDING -> terminal PENDING. No action can be blamed (plan section 4.2)."""

    ABORTED = "aborted"
    """Context limit, parser crash, environment crash. Excluded from the 3-way rates."""


# --------------------------------------------------------------------------- #
# Serialisation helpers
# --------------------------------------------------------------------------- #


def _enum_to_value(obj: Any) -> Any:
    if isinstance(obj, Enum):
        return obj.value
    if isinstance(obj, dict):
        return {k: _enum_to_value(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_enum_to_value(v) for v in obj]
    return obj


to_jsonable = _enum_to_value
"""Recursively replace enums with their values. Public alias for other modules."""


@dataclass
class _Record:
    """Mixin providing uniform dict/JSON conversion."""

    def to_dict(self) -> dict[str, Any]:
        return _enum_to_value(asdict(self))

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True)


# --------------------------------------------------------------------------- #
# Raw rollout records (plan section 5.1)
# --------------------------------------------------------------------------- #


@dataclass
class ToolCall(_Record):
    tool: str
    arguments: dict[str, Any] = field(default_factory=dict)
    raw_text: str | None = None
    parse_ok: bool = True
    parse_error: str | None = None

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ToolCall":
        return cls(
            tool=data["tool"],
            arguments=dict(data.get("arguments") or {}),
            raw_text=data.get("raw_text"),
            parse_ok=bool(data.get("parse_ok", True)),
            parse_error=data.get("parse_error"),
        )


@dataclass
class StepRecord(_Record):
    """One assistant action plus the observation it produced.

    ``assistant_token_span`` is a half-open ``[start, end)`` interval into the
    *student* rollout token sequence. It is the only anchor the mask builder uses,
    so it must be produced by the same tokenizer pass that builds the model input.
    """

    step: int
    turn: int
    assistant_text: str = ""
    parsed_calls: list[ToolCall] = field(default_factory=list)
    raw_observation: str = ""
    observation_role: str = "user"
    tool_execution_ok: bool | None = None
    """Whether the tool/API call itself succeeded. Deliberately kept separate from
    rubric success -- plan section 6.2. Never used as an episode label."""

    state_before_hash: str | None = None
    state_after_hash: str | None = None
    state_delta: dict[str, Any] = field(default_factory=dict)
    evaluator_evidence: dict[str, Any] = field(default_factory=dict)
    """Raw per-step evaluator evidence.

    CrEST needs the verifier verdict at every completed turn, not only the
    terminal session verdict.  Keeping it on the step also makes the reward
    derivation auditable without teaching the benchmark-agnostic manager BFCL
    semantics.
    """
    assistant_char_span: tuple[int, int] | None = None
    assistant_token_span: tuple[int, int] | None = None
    history_before: str | None = None

    @property
    def num_action_tokens(self) -> int:
        if self.assistant_token_span is None:
            return 0
        start, end = self.assistant_token_span
        return max(0, end - start)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "StepRecord":
        span = data.get("assistant_token_span")
        char_span = data.get("assistant_char_span")
        return cls(
            step=int(data["step"]),
            turn=int(data["turn"]),
            assistant_text=data.get("assistant_text", ""),
            parsed_calls=[ToolCall.from_dict(c) for c in data.get("parsed_calls") or []],
            raw_observation=data.get("raw_observation", ""),
            observation_role=data.get("observation_role", "user"),
            tool_execution_ok=data.get("tool_execution_ok"),
            state_before_hash=data.get("state_before_hash"),
            state_after_hash=data.get("state_after_hash"),
            state_delta=dict(data.get("state_delta") or {}),
            evaluator_evidence=dict(data.get("evaluator_evidence") or {}),
            assistant_char_span=tuple(char_span) if char_span else None,  # type: ignore[arg-type]
            assistant_token_span=tuple(span) if span else None,  # type: ignore[arg-type]
            history_before=data.get("history_before"),
        )


@dataclass
class RolloutRecord(_Record):
    """A single on-policy trajectory. One per (task, seed, sample)."""

    benchmark: str
    task_id: str
    rollout_id: str
    steps: list[StepRecord] = field(default_factory=list)
    policy_step: int = 0
    model_checksum: str | None = None
    seed: int | None = None
    terminated: bool = False
    truncated: bool = False
    abort_reason: str | None = None
    split: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def aborted(self) -> bool:
        return self.abort_reason is not None or (self.truncated and not self.terminated)

    def step_by_index(self, step: int) -> StepRecord | None:
        for record in self.steps:
            if record.step == step:
                return record
        return None

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "RolloutRecord":
        return cls(
            benchmark=data["benchmark"],
            task_id=data["task_id"],
            rollout_id=data["rollout_id"],
            steps=[StepRecord.from_dict(s) for s in data.get("steps") or []],
            policy_step=int(data.get("policy_step", 0)),
            model_checksum=data.get("model_checksum"),
            seed=data.get("seed"),
            terminated=bool(data.get("terminated", False)),
            truncated=bool(data.get("truncated", False)),
            abort_reason=data.get("abort_reason"),
            split=data.get("split"),
            metadata=dict(data.get("metadata") or {}),
        )


# --------------------------------------------------------------------------- #
# Rubric records (plan section 5.2)
# --------------------------------------------------------------------------- #


@dataclass
class RubricSpec(_Record):
    """Declaration of one rubric for one task."""

    rubric_id: str
    kind: RubricKind
    hard: bool
    source: str
    initial_state: RubricState
    natural_language: str = ""
    predicate_version: str | None = None
    desired_postcondition: str | None = None
    forbidden_postconditions: list[str] = field(default_factory=list)
    turn: int | None = None
    """BFCL rubrics are anchored to a specific user turn; AppWorld rubrics are not."""

    def __post_init__(self) -> None:
        self.kind = RubricKind(self.kind)
        self.initial_state = RubricState(self.initial_state)
        if self.kind is RubricKind.GOAL and self.initial_state is RubricState.VIOLATED:
            raise ValueError(
                f"goal rubric {self.rubric_id!r} cannot start VIOLATED; "
                "an unmet positive goal starts PENDING (plan section 4.1)"
            )

    @classmethod
    def default_initial_state(cls, kind: RubricKind) -> RubricState:
        return RubricState.PENDING if kind is RubricKind.GOAL else RubricState.SATISFIED

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "RubricSpec":
        kind = RubricKind(data["kind"])
        return cls(
            rubric_id=data["rubric_id"],
            kind=kind,
            hard=bool(data.get("hard", True)),
            source=data.get("source", ""),
            initial_state=RubricState(data.get("initial_state") or cls.default_initial_state(kind)),
            natural_language=data.get("natural_language", ""),
            predicate_version=data.get("predicate_version"),
            desired_postcondition=data.get("desired_postcondition"),
            forbidden_postconditions=list(data.get("forbidden_postconditions") or []),
            turn=data.get("turn"),
        )


@dataclass
class RubricEvaluation(_Record):
    """Result of running one rubric predicate against one snapshot."""

    rubric_id: str
    passed: bool | None
    """``None`` means the predicate could not be evaluated -> UNKNOWN."""

    trace: str = ""
    error: str | None = None

    target_revision: int | None = None
    """Which revision of this rubric's *expectation* was checked. Increments only
    when the expected value actually changes, so two turns that want the same
    thing share a revision. ``None`` for rubrics with a fixed expectation
    (AppWorld, and BFCL's action-level rubrics)."""

    expected_value: str | None = None
    """Canonical short form of what the rubric required at this step."""

    actual_value: str | None = None
    """Canonical short form of what was observed. Together with
    ``expected_value`` this is what lets an episode closure be attributed to the
    policy rather than to a moved goalpost."""

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "RubricEvaluation":
        return cls(
            rubric_id=data["rubric_id"],
            passed=data.get("passed"),
            trace=data.get("trace", ""),
            error=data.get("error"),
            target_revision=data.get("target_revision"),
            expected_value=data.get("expected_value"),
            actual_value=data.get("actual_value"),
        )


@dataclass
class RubricStateVector(_Record):
    """Rubric states observed after a given step (plan section 5.2).

    ``step == -1`` denotes the pre-rollout initial vector.
    """

    rollout_id: str
    step: int
    rubrics: dict[str, RubricState] = field(default_factory=dict)
    turn: int | None = None
    evaluation_state_hash: str | None = None
    evaluator_trace: str | None = None

    target_revisions: dict[str, int] = field(default_factory=dict)
    expected_values: dict[str, str] = field(default_factory=dict)
    actual_values: dict[str, str] = field(default_factory=dict)
    """Per-rubric attribution data, present only for rubrics whose expectation can
    move (BFCL ``state:*``). Empty for AppWorld, whose requirements are fixed for
    the whole task."""

    def __post_init__(self) -> None:
        self.rubrics = {k: RubricState(v) for k, v in self.rubrics.items()}

    def observation(self, rubric_id: str) -> tuple[int | None, str | None, str | None]:
        """``(target_revision, expected_value, actual_value)`` for one rubric."""
        return (
            self.target_revisions.get(rubric_id),
            self.expected_values.get(rubric_id),
            self.actual_values.get(rubric_id),
        )

    @classmethod
    def from_evaluations(
        cls,
        rollout_id: str,
        step: int,
        states: Mapping[str, "RubricState"],
        evaluations: Mapping[str, "RubricEvaluation"],
        *,
        turn: int | None = None,
        evaluation_state_hash: str | None = None,
    ) -> "RubricStateVector":
        """Build a vector, lifting attribution data out of the raw evaluations."""
        return cls(
            rollout_id=rollout_id,
            step=step,
            rubrics=dict(states),
            turn=turn,
            evaluation_state_hash=evaluation_state_hash,
            target_revisions={
                rid: e.target_revision
                for rid, e in evaluations.items()
                if e.target_revision is not None
            },
            expected_values={
                rid: e.expected_value
                for rid, e in evaluations.items()
                if e.expected_value is not None
            },
            actual_values={
                rid: e.actual_value
                for rid, e in evaluations.items()
                if e.actual_value is not None
            },
        )

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "RubricStateVector":
        return cls(
            rollout_id=data["rollout_id"],
            step=int(data["step"]),
            rubrics={k: RubricState(v) for k, v in (data.get("rubrics") or {}).items()},
            turn=data.get("turn"),
            evaluation_state_hash=data.get("evaluation_state_hash"),
            evaluator_trace=data.get("evaluator_trace"),
            target_revisions=dict(data.get("target_revisions") or {}),
            expected_values=dict(data.get("expected_values") or {}),
            actual_values=dict(data.get("actual_values") or {}),
        )


# --------------------------------------------------------------------------- #
# Episode records (plan section 5.3)
# --------------------------------------------------------------------------- #


@dataclass
class CostEvidence(_Record):
    """Why an episode was (or was not) charged an external cost."""

    verdict: CostVerdict = CostVerdict.UNKNOWN
    reasons: list[str] = field(default_factory=list)
    process_overhead_steps: int = 0
    process_overhead_tokens: int = 0
    process_overhead_calls: int = 0
    """Extra retries / read-only calls / tokens. Recorded but never counted as
    external cost (plan section 4.3)."""

    def __post_init__(self) -> None:
        self.verdict = CostVerdict(self.verdict)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "CostEvidence":
        return cls(
            verdict=CostVerdict(data.get("verdict", CostVerdict.UNKNOWN)),
            reasons=list(data.get("reasons") or []),
            process_overhead_steps=int(data.get("process_overhead_steps", 0)),
            process_overhead_tokens=int(data.get("process_overhead_tokens", 0)),
            process_overhead_calls=int(data.get("process_overhead_calls", 0)),
        )


@dataclass
class Episode(_Record):
    """One open->(close) interval of a single rubric."""

    rollout_id: str
    episode_id: str
    rubric_id: str
    open_step: int
    open_turn: int | None = None
    open_action_token_span: tuple[int, int] | None = None
    close_step: int | None = None
    close_turn: int | None = None
    close_action_token_span: tuple[int, int] | None = None
    manifestation_steps: list[int] = field(default_factory=list)
    reopen_index: int = 0
    cost: CostEvidence = field(default_factory=CostEvidence)
    rubric_kind: RubricKind = RubricKind.SAFETY
    hard: bool = True

    close_reason: EpisodeCloseReason = EpisodeCloseReason.UNKNOWN
    open_target_revision: int | None = None
    open_expected_value: str | None = None
    open_actual_value: str | None = None
    close_target_revision: int | None = None
    close_actual_value: str | None = None
    """What the rubric wanted and what was there, at both ends of the episode.
    Preserved so a reader can re-derive ``close_reason`` instead of trusting it."""

    cost_vector: Any = None
    """:class:`bfcl_env.cost_evidence.CostEvidenceVector`, when the benchmark has
    a rule registry. Typed loosely to keep this module import-free."""

    coverage_class: CoverageClass = CoverageClass.DECIDABLE
    unscoreable: bool = False
    open_action_ref: str | None = None
    close_action_ref: str | None = None
    evidence_refs: list[str] = field(default_factory=list)

    # ---- derived ---------------------------------------------------------- #

    @property
    def resolved(self) -> bool:
        return self.close_step is not None

    @property
    def latency_steps(self) -> int | None:
        if self.close_step is None:
            return None
        return self.close_step - self.open_step

    @property
    def latency_turns(self) -> int | None:
        if self.close_turn is None or self.open_turn is None:
            return None
        return self.close_turn - self.open_turn

    @property
    def num_manifestations(self) -> int:
        return len(self.manifestation_steps)

    @property
    def recovered_by_action(self) -> bool:
        """Only this counts towards a recovery statistic."""
        return (
            self.resolved
            and self.close_reason is EpisodeCloseReason.RECOVERED_BY_ACTION
        )

    @property
    def classification_status(self) -> ClassificationStatus:
        """Why this episode is (or is not) labelled.

        Order matters: an exclusion outranks an abstention, and an abstention
        outranks a label. The first rule that applies wins, so an episode can never
        be labelled on the strength of evidence a later rule would have rejected.
        """
        if self.close_reason is EpisodeCloseReason.SUPERSEDED_BY_TARGET:
            return ClassificationStatus.EXCLUDED_SUPERSEDED
        if self.unscoreable:
            return ClassificationStatus.EXCLUDED_UNSCOREABLE
        if not self.resolved:
            # An unresolved episode needs no cost verdict: it is a failure that
            # persisted, whatever it cost along the way.
            return (
                ClassificationStatus.ABSTAINED_UNCOVERED_EFFECT
                if self.coverage_class is CoverageClass.UNCOVERED
                else ClassificationStatus.LABELED
            )
        if self.close_reason is EpisodeCloseReason.UNKNOWN:
            return ClassificationStatus.ABSTAINED_UNKNOWN_STATE
        if self.coverage_class is CoverageClass.UNCOVERED:
            return ClassificationStatus.ABSTAINED_UNCOVERED_EFFECT
        if self._has_registry_vector:
            if self.cost_vector.is_undecided:
                return ClassificationStatus.ABSTAINED_COST_UNKNOWN
            return ClassificationStatus.LABELED
        if self.cost.verdict is CostVerdict.UNKNOWN:
            return ClassificationStatus.ABSTAINED_COST_UNKNOWN
        return ClassificationStatus.LABELED

    @property
    def _has_registry_vector(self) -> bool:
        """A vector without relevant dimensions means the benchmark has no cost
        registry; fall back to the ledger rather than abstaining on everything."""
        return (
            self.cost_vector is not None
            and bool(getattr(self.cost_vector, "relevant_dimensions", None))
        )

    @property
    def episode_type(self) -> EpisodeType | None:
        """One of the plan's three types, or ``None`` when not confidently labelled."""
        if not self.classification_status.is_labeled:
            return None
        if not self.resolved:
            return EpisodeType.UNRESOLVED_FAILURE
        if self._has_registry_vector:
            return (
                EpisodeType.COSTLY_RECOVERY
                if self.cost_vector.has_external_cost
                else EpisodeType.CLEAN_RECOVERY
            )
        return (
            EpisodeType.COSTLY_RECOVERY
            if self.cost.verdict is CostVerdict.EXTERNAL
            else EpisodeType.CLEAN_RECOVERY
        )

    @property
    def type(self) -> EpisodeType | None:
        """Backwards-compatible alias for :attr:`episode_type`."""
        return self.episode_type

    def to_dict(self) -> dict[str, Any]:
        data = super().to_dict()
        data["latency_steps"] = self.latency_steps
        data["latency_turns"] = self.latency_turns
        data["num_manifestations"] = self.num_manifestations
        data["resolved"] = self.resolved
        data["recovered_by_action"] = self.recovered_by_action
        status = self.classification_status
        episode_type = self.episode_type
        data["classification_status"] = status.value
        data["classification_status_ko"] = status.korean
        data["type"] = episode_type.value if episode_type else None
        data["type_ko"] = episode_type.korean if episode_type else None
        data["close_reason_ko"] = _CLOSE_REASON_KO[self.close_reason]
        data["coverage_class"] = self.coverage_class.value
        data["cost_vector"] = (
            self.cost_vector.to_dict() if self.cost_vector is not None else None
        )
        # ``external_cost`` mirrors the plan's section 5.3 field name.
        data["external_cost"] = self.cost.verdict.value
        return data

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Episode":
        open_span = data.get("open_action_token_span")
        close_span = data.get("close_action_token_span")
        return cls(
            rollout_id=data["rollout_id"],
            episode_id=data["episode_id"],
            rubric_id=data["rubric_id"],
            open_step=int(data["open_step"]),
            open_turn=data.get("open_turn"),
            open_action_token_span=tuple(open_span) if open_span else None,  # type: ignore[arg-type]
            close_step=data.get("close_step"),
            close_turn=data.get("close_turn"),
            close_action_token_span=tuple(close_span) if close_span else None,  # type: ignore[arg-type]
            manifestation_steps=list(data.get("manifestation_steps") or []),
            reopen_index=int(data.get("reopen_index", 0)),
            cost=CostEvidence.from_dict(data.get("cost") or {}),
            rubric_kind=RubricKind(data.get("rubric_kind", RubricKind.SAFETY)),
            hard=bool(data.get("hard", True)),
            close_reason=EpisodeCloseReason(
                data.get("close_reason", EpisodeCloseReason.UNKNOWN)
            ),
            open_target_revision=data.get("open_target_revision"),
            open_expected_value=data.get("open_expected_value"),
            open_actual_value=data.get("open_actual_value"),
            close_target_revision=data.get("close_target_revision"),
            close_actual_value=data.get("close_actual_value"),
            cost_vector=_cost_vector_from(data.get("cost_vector")),
            coverage_class=CoverageClass(data.get("coverage_class", "decidable")),
            unscoreable=bool(data.get("unscoreable", False)),
            open_action_ref=data.get("open_action_ref"),
            close_action_ref=data.get("close_action_ref"),
            evidence_refs=list(data.get("evidence_refs") or []),
        )


def _cost_vector_from(payload: Mapping[str, Any] | None):
    """Rebuild a cost vector without importing the module at load time."""
    if not payload:
        return None
    from bfcl_env.cost_evidence import CostEvidenceVector  # noqa: PLC0415

    return CostEvidenceVector.from_dict(payload)


@dataclass
class RolloutOutcome(_Record):
    """Rollout-level classification (plan section 3.1).

    The four classes are mutually exclusive. ``aborted`` rollouts are dropped from
    the denominator of the other three and reported separately.
    """

    rollout_id: str
    task_id: str
    benchmark: str
    terminal_success: bool = False
    any_hard_violation: bool = False
    aborted: bool = False
    failure_kind: FailureKind = FailureKind.NONE
    first_violation_step: int | None = None
    first_violation_turn: int | None = None
    first_violation_rubric_id: str | None = None
    num_episodes: int = 0
    num_recovered_episodes: int = 0
    """Closed *by the policy's action*. Superseded episodes are counted separately
    so they can never inflate a recovery rate."""

    num_superseded_episodes: int = 0
    num_unattributed_episodes: int = 0
    num_labeled_episodes: int = 0
    num_abstained_episodes: int = 0
    num_unresolved_episodes: int = 0
    """recovered + superseded + unattributed + unresolved == num_episodes."""
    num_reopens: int = 0
    num_manifestations: int = 0
    max_concurrent_active: int = 0
    num_action_tokens: int = 0
    num_steps: int = 0
    num_turns: int = 0
    coverage_warning: bool = False
    uncovered_mutations: list[str] = field(default_factory=list)
    unknown_rubrics: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.failure_kind = FailureKind(self.failure_kind)

    @property
    def clean_success(self) -> bool:
        return (not self.aborted) and self.terminal_success and not self.any_hard_violation

    @property
    def recovered_success(self) -> bool:
        return (not self.aborted) and self.terminal_success and self.any_hard_violation

    @property
    def terminal_failure(self) -> bool:
        return (not self.aborted) and not self.terminal_success

    @property
    def rollout_class(self) -> str:
        if self.aborted:
            return "aborted_or_unknown"
        if self.clean_success:
            return "clean_success"
        if self.recovered_success:
            return "recovered_success"
        return "terminal_failure"

    @property
    def has_localized_first_violation(self) -> bool:
        return self.first_violation_step is not None

    def to_dict(self) -> dict[str, Any]:
        data = super().to_dict()
        data["clean_success"] = self.clean_success
        data["recovered_success"] = self.recovered_success
        data["terminal_failure"] = self.terminal_failure
        data["rollout_class"] = self.rollout_class
        data["has_localized_first_violation"] = self.has_localized_first_violation
        return data

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "RolloutOutcome":
        known = {
            "rollout_id", "task_id", "benchmark", "terminal_success", "any_hard_violation",
            "aborted", "failure_kind", "first_violation_step", "first_violation_turn",
            "first_violation_rubric_id", "num_episodes", "num_recovered_episodes",
            "num_superseded_episodes", "num_unattributed_episodes",
            "num_labeled_episodes", "num_abstained_episodes",
            "num_unresolved_episodes",
            "num_reopens", "num_manifestations",
            "max_concurrent_active", "num_action_tokens", "num_steps", "num_turns",
            "coverage_warning", "uncovered_mutations", "unknown_rubrics",
        }
        return cls(**{k: v for k, v in data.items() if k in known})


# --------------------------------------------------------------------------- #
# JSONL IO
# --------------------------------------------------------------------------- #


def read_jsonl(path: str) -> Iterator[dict[str, Any]]:
    """Yield one dict per non-empty line."""
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(path: str, records: Iterable[Any]) -> int:
    """Write records (dicts or ``_Record`` instances) as JSONL. Returns the count."""
    count = 0
    with open(path, "w", encoding="utf-8") as handle:
        for record in records:
            payload = record.to_dict() if hasattr(record, "to_dict") else record
            handle.write(json.dumps(_enum_to_value(payload), ensure_ascii=False, sort_keys=True))
            handle.write("\n")
            count += 1
    return count
