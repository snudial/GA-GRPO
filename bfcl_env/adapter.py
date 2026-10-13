"""BFCL v3 multi-turn: task loading, rubric compilation, comparison (plan section 7).

Everything in this module is pure data. ``bfcl_eval`` is imported lazily and only
where genuinely required (function docs and ground-truth execution), so the rubric
logic is testable from fixtures with no benchmark installed.

What the official checker does, and what this reuses
----------------------------------------------------
``bfcl_eval/eval_checker/multi_turn_eval/multi_turn_checker.py`` executes each
turn, then

* ``_compare_instances`` iterates ``vars(ground_truth_instance)``, **skips any
  attribute starting with ``_``**, and compares with ``!=`` (so each simulator's
  own ``__eq__`` decides, e.g. ``File.__eq__`` compares name and content only),
* ``_is_subsequence_unordered`` requires the current turn's ground-truth execution
  responses to appear, multiset-style and order-independently, among the model's
  responses **accumulated across all turns so far**,
* and it **returns at the first mismatch**.

This module reproduces the two comparisons exactly and leaves the early return to
the runner mode (plan section 7.2): ``official_stop`` reproduces the official
score, ``recovery_continue`` keeps the same trajectory going so recovery is
observable at all.

One deliberate deviation from the plan's section 7.1 naming
-----------------------------------------------------------
The plan writes state rubrics as ``state:u:{class}:{attribute}``, one family per
user turn. With a per-turn id, a rubric can only ever be checked at the end of its
own turn -- comparing the world against turn ``u``'s ground truth after turn
``u+1`` is meaningless, since turn ``u+1`` legitimately changes it. That makes a
state violation permanently unresolvable, so every state mismatch would classify
as ``unresolved_failure`` and ``cross_turn_recovery`` (plan section 7.2) could
never be measured.

So the turn index moves from the id into the *evaluation*: the rubric is
``state:{class}:{attribute}``, its expectation advances to the current turn's
ground-truth snapshot, and it is evaluated only at turn boundaries. A model that
falls behind at turn 1 and catches up at turn 2 then produces a real closed
episode, which is exactly ``cross_turn_recovery``. Per-turn reporting is preserved
because every state vector carries its ``turn``.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import re
from collections import Counter
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from bfcl_env.schemas import RubricEvaluation, RubricKind, RubricSpec, RubricState

__all__ = [
    "BFCL_CATEGORIES",
    "RUNNER_MODES",
    "ACTION_RUBRICS",
    "EXECUTION_ERROR_PREFIX",
    "EXECUTION_OUTCOME_POLICY_VERSION",
    "PARSER_POLICY_VERSION",
    "ExecutionOutcomeKind",
    "ExecutionOutcome",
    "normalize_execution_outcome",
    "is_execution_failure",
    "ActionParseKind",
    "ActionParse",
    "classify_raw_action",
    "canonical_value",
    "target_revisions",
    "build_expectations",
    "BFCLTask",
    "TurnSnapshot",
    "GroundTruthTrace",
    "bfcl_data_root",
    "TaskLoadReport",
    "load_task_report",
    "load_tasks",
    "load_task",
    "compile_rubrics",
    "snapshot_instances",
    "compare_states",
    "responses_contain",
    "classify_action_errors",
    "state_rubric_id",
    "response_rubric_id",
    "no_tool_call_rubric_id",
]

BFCL_CATEGORIES = (
    "multi_turn_base",
    "multi_turn_miss_func",
    "multi_turn_miss_param",
    "multi_turn_long_context",
)

RUNNER_MODES = ("official_stop", "recovery_continue")

EXECUTION_ERROR_PREFIX = "Error during execution:"
"""``execute_multi_turn_func_call`` swallows every exception and appends this
prefix instead of raising, so a failed call is only visible in the result string."""

EXECUTION_OUTCOME_POLICY_VERSION = "bfcl_dual_channel_failure_v1"
"""Recorded in the manifest. The ``top_level_nonempty_error_key`` rule below was
established by auditing all 129 multi-turn function docs at the pinned BFCL
revision: none declares ``error`` as a legitimate response property. Re-run that
audit before trusting this rule against a different BFCL revision."""

PARSER_POLICY_VERSION = "bfcl_python_call_attempt_v1"
"""Recorded in the manifest. Encodes the tri-state split below, which is specific
to BFCL's Python-style call format."""


class ExecutionOutcomeKind(str, Enum):
    """How one executed call ended."""

    SUCCESS = "SUCCESS"
    TOOL_ERROR_RESPONSE = "TOOL_ERROR_RESPONSE"
    """The simulator returned ``{"error": ...}`` normally. This is the *common*
    failure channel -- 167 such return sites across the eight simulator modules."""

    TOOL_EXCEPTION = "TOOL_EXCEPTION"
    """The call raised and the helper swallowed it into a result string."""

    INFRASTRUCTURE_ERROR = "INFRASTRUCTURE_ERROR"
    """The harness itself failed. Never a policy error; it aborts the rollout."""


@dataclass
class ExecutionOutcome:
    """Normalised verdict for one execution result string.

    Centralised deliberately: scattering "is this an error?" tests through the
    runtime is how one of the two failure channels gets forgotten.
    """

    kind: ExecutionOutcomeKind
    detection_rule: str
    raw_result: str = ""
    message: str = ""

    @property
    def failed(self) -> bool:
        return self.kind is not ExecutionOutcomeKind.SUCCESS

    @property
    def is_tool_failure(self) -> bool:
        """A failure the *policy* caused and could recover from."""
        return self.kind in (
            ExecutionOutcomeKind.TOOL_ERROR_RESPONSE,
            ExecutionOutcomeKind.TOOL_EXCEPTION,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind.value,
            "detection_rule": self.detection_rule,
            "message": self.message,
            "policy_version": EXECUTION_OUTCOME_POLICY_VERSION,
        }


def normalize_execution_outcome(raw_result: str) -> ExecutionOutcome:
    """Classify one result string from ``execute_multi_turn_func_call``.

    The simulators use two failure channels and only one of them raises:

    * a Python exception, swallowed into ``"Error during execution: ..."``,
    * a normal return of ``{"error": "..."}``, e.g. ``cd`` into a missing
      directory.

    Detecting only the first would miss most real BFCL execution errors.
    """
    if raw_result is None:
        return ExecutionOutcome(
            ExecutionOutcomeKind.SUCCESS, "empty_result", "", ""
        )
    if raw_result.startswith(EXECUTION_ERROR_PREFIX):
        return ExecutionOutcome(
            ExecutionOutcomeKind.TOOL_EXCEPTION,
            "execution_error_prefix",
            raw_result,
            raw_result[len(EXECUTION_ERROR_PREFIX):].strip(),
        )

    stripped = raw_result.strip()
    if stripped.startswith("{"):
        try:
            payload = json.loads(stripped)
        except (json.JSONDecodeError, ValueError):
            payload = None
        if isinstance(payload, dict) and payload.get("error"):
            return ExecutionOutcome(
                ExecutionOutcomeKind.TOOL_ERROR_RESPONSE,
                "top_level_nonempty_error_key",
                raw_result,
                str(payload["error"]),
            )

    return ExecutionOutcome(ExecutionOutcomeKind.SUCCESS, "no_error_signal", raw_result, "")


def is_execution_failure(response: str) -> bool:
    """Thin wrapper over :func:`normalize_execution_outcome`, kept for call sites
    that only need the boolean."""
    return normalize_execution_outcome(response).failed


ACTION_RUBRICS: dict[str, str] = {
    "action:no_parse_error": "every assistant action must decode into a tool-call list",
    "action:no_unknown_tool": "every call must name a function that is currently available",
    "action:no_invalid_arguments": "every call must supply its required arguments",
    "action:no_execution_error": "no call may raise inside the simulator",
}
"""Action-level rubrics. These are the only deterministic, sub-turn errors the plan
allows to open an episode (section 7.1); everything else waits for a turn boundary."""

_CALL_NAME_RE = re.compile(r"^\s*([A-Za-z_]\w*)\s*\(")


def state_rubric_id(class_name: str, attribute: str) -> str:
    return f"state:{class_name}:{attribute}"


def response_rubric_id(turn: int) -> str:
    return f"response:turn{turn}"


def no_tool_call_rubric_id(turn: int) -> str:
    return f"no_tool_call:turn{turn}"


# --------------------------------------------------------------------------- #
# action parsing: three outcomes, not two
# --------------------------------------------------------------------------- #

_CALL_ATTEMPT_RE = re.compile(r"[A-Za-z_]\w*\(")
"""Evidence that the model tried to emit a call. No whitespace is allowed before
the paren, so ordinary prose containing a parenthesis is not misread as one."""


class ActionParseKind(str, Enum):
    """What an assistant action turned out to be.

    The official decoder raises identically on ``"[]"``, on ``""`` and on plain
    prose, and ``BaseHandler`` treats all three as "the turn is over". But plan
    section 7.1 wants a genuine parser failure to open an episode, so the two must
    be told apart -- which the exception alone cannot do.
    """

    TOOL_CALL = "TOOL_CALL"
    END_TURN = "END_TURN"
    PARSE_ERROR = "PARSE_ERROR"


@dataclass
class ActionParse:
    kind: ActionParseKind
    calls: list[str] = field(default_factory=list)
    error: str | None = None
    policy_version: str = PARSER_POLICY_VERSION

    @property
    def parse_ok(self) -> bool:
        return self.kind is not ActionParseKind.PARSE_ERROR


def classify_raw_action(raw_text: str, decoder=None) -> ActionParse:
    """Decode one assistant action into exactly one of three outcomes.

    ``decoder`` defaults to BFCL's ``default_decode_execute_prompting``; it is
    injectable so the tri-state rule can be tested without the benchmark.
    """
    if decoder is None:
        from bfcl_eval.model_handler.utils import (  # noqa: PLC0415
            default_decode_execute_prompting,
        )

        decoder = default_decode_execute_prompting

    try:
        calls = list(decoder(raw_text))
    except Exception as error:  # noqa: BLE001 - a bad action is data, not a crash
        if _CALL_ATTEMPT_RE.search(raw_text or ""):
            return ActionParse(
                ActionParseKind.PARSE_ERROR, [], f"{type(error).__name__}: {error}"
            )
        return ActionParse(ActionParseKind.END_TURN, [])

    if not calls:
        return ActionParse(ActionParseKind.END_TURN, [])
    return ActionParse(ActionParseKind.TOOL_CALL, calls)


# --------------------------------------------------------------------------- #
# expectation revisions
# --------------------------------------------------------------------------- #


def canonical_value(value: Any, *, max_repr: int = 200) -> str:
    """A short, exact identity for an arbitrary simulator attribute.

    Short values keep their ``repr`` so a report is readable; long ones collapse to
    a digest. Either way two canonical forms are equal iff the underlying reprs
    are, which is what the recovery/superseded attribution relies on.
    """
    text = repr(value)
    if len(text) <= max_repr:
        return text
    return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()[:32]


def target_revisions(
    expected_by_turn: Sequence[str],
) -> list[int]:
    """Map each turn's expected value to a revision index.

    The revision increments only when the expectation actually *changes*, so two
    consecutive turns that want the same thing share a revision and a closure
    between them is still attributable to the policy.
    """
    revisions: list[int] = []
    current = 0
    previous: str | None = None
    for index, expected in enumerate(expected_by_turn):
        if index > 0 and expected != previous:
            current += 1
        revisions.append(current)
        previous = expected
    return revisions


# --------------------------------------------------------------------------- #
# task loading
# --------------------------------------------------------------------------- #


def bfcl_data_root(root: str | os.PathLike[str] | None = None) -> Path:
    """Locate ``bfcl_eval/data``.

    Order: explicit argument, ``BFCL_DATA_ROOT``, then ``bfcl_eval`` itself.
    """
    for candidate in (root, os.environ.get("BFCL_DATA_ROOT")):
        if candidate:
            path = Path(candidate)
            if (path / "possible_answer").is_dir():
                return path
            if (path / "data" / "possible_answer").is_dir():
                return path / "data"
    try:
        from bfcl_eval.constants.eval_config import PROMPT_PATH  # noqa: PLC0415
    except ImportError as error:
        raise FileNotFoundError(
            "BFCL data root not found; set BFCL_DATA_ROOT to the directory that "
            "contains possible_answer/"
        ) from error
    return Path(PROMPT_PATH)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    """BFCL ships JSONL under a ``.json`` name, sometimes with no trailing newline."""
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


@dataclass
class BFCLTask:
    """One multi-turn entry joined with its ground truth."""

    task_id: str
    category: str
    question: list[list[dict[str, str]]]
    initial_config: dict[str, Any]
    involved_classes: list[str]
    ground_truth: list[list[str]]
    missed_function: dict[int, list[str]] = field(default_factory=dict)
    """``{turn_index: [function_name, ...]}``. Only ``multi_turn_miss_func`` has it;
    the named turn carries an empty user message and the functions are revealed
    there."""

    @property
    def num_turns(self) -> int:
        return len(self.question)

    @property
    def long_context(self) -> bool:
        return "long_context" in self.category

    def user_messages(self, turn: int) -> list[dict[str, str]]:
        return self.question[turn] if 0 <= turn < len(self.question) else []

    def ground_truth_calls(self, turn: int) -> list[str]:
        return self.ground_truth[turn] if 0 <= turn < len(self.ground_truth) else []

    def is_holdout_turn(self, turn: int) -> bool:
        return turn in self.missed_function

    def revealed_functions(self, turn: int) -> list[str]:
        """Held-out function names available from ``turn`` onwards."""
        names: list[str] = []
        for holdout_turn, function_names in sorted(self.missed_function.items()):
            if holdout_turn <= turn:
                names.extend(function_names)
        return names

    @classmethod
    def from_entries(
        cls, prompt: Mapping[str, Any], answer: Mapping[str, Any], category: str
    ) -> "BFCLTask":
        if prompt["id"] != answer["id"]:
            raise ValueError(
                f"prompt/answer id mismatch: {prompt['id']!r} vs {answer['id']!r}"
            )
        missed = {
            int(turn): list(names)
            for turn, names in (prompt.get("missed_function") or {}).items()
        }
        return cls(
            task_id=prompt["id"],
            category=category,
            question=[list(turn) for turn in prompt["question"]],
            initial_config=copy.deepcopy(prompt["initial_config"]),
            involved_classes=list(prompt["involved_classes"]),
            ground_truth=[list(turn) for turn in answer["ground_truth"]],
            missed_function=missed,
        )


@dataclass
class TaskLoadReport:
    """Loaded tasks plus whatever had to be excluded, and why.

    Exclusions are never silent: a dropped task changes the denominator of every
    rate this pipeline reports, so the reason travels with the data.
    """

    category: str
    tasks: dict[str, BFCLTask] = field(default_factory=dict)
    skipped: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "category": self.category,
            "num_tasks": len(self.tasks),
            "num_skipped": len(self.skipped),
            "skipped": dict(self.skipped),
        }


def load_task_report(
    category: str,
    root: str | os.PathLike[str] | None = None,
    *,
    on_malformed: str = "skip",
) -> TaskLoadReport:
    """Load one category, joined with its ``possible_answer`` ground truth.

    A handful of upstream entries declare a different number of ground-truth turns
    than user turns (``multi_turn_base_154`` at the pinned commit). The official
    runner scores those ``multi_turn:force_terminated`` for every model, so they
    are unscoreable rather than merely hard. ``on_malformed``:

    * ``"skip"`` (default) excludes them and records the reason,
    * ``"raise"`` refuses to load the category at all,
    * ``"keep"`` returns them untouched, for anyone deliberately studying them.
    """
    if category not in BFCL_CATEGORIES:
        raise ValueError(
            f"category {category!r} is not one of the four multi-turn categories "
            f"this pipeline covers: {BFCL_CATEGORIES}"
        )
    if on_malformed not in ("skip", "raise", "keep"):
        raise ValueError("on_malformed must be 'skip', 'raise' or 'keep'")

    data_root = bfcl_data_root(root)
    # The CrEST/T3RL author-lineage commit calls this dataset BFCL_v4 while the
    # pinned compatibility checkout calls it BFCL_v3. Read the prefix from the
    # selected runtime instead of silently mixing v1.3 data with the newer
    # handler/evaluator.
    from bfcl_eval.constants.category_mapping import VERSION_PREFIX  # noqa: PLC0415

    prompts = _read_jsonl(data_root / f"{VERSION_PREFIX}_{category}.json")
    answers = {row["id"]: row for row in _read_jsonl(
        data_root / "possible_answer" / f"{VERSION_PREFIX}_{category}.json"
    )}

    report = TaskLoadReport(category=category)
    for prompt in prompts:
        answer = answers.get(prompt["id"])
        if answer is None:
            raise ValueError(f"no ground truth for {prompt['id']!r} in {category}")
        task = BFCLTask.from_entries(prompt, answer, category)

        if task.num_turns != len(task.ground_truth) and on_malformed != "keep":
            reason = (
                f"{task.num_turns} user turns but {len(task.ground_truth)} "
                "ground-truth turns; the official runner reports force_terminated"
            )
            if on_malformed == "raise":
                raise ValueError(f"{task.task_id}: {reason}")
            report.skipped[task.task_id] = reason
            continue

        report.tasks[task.task_id] = task
    return report


def load_tasks(
    category: str,
    root: str | os.PathLike[str] | None = None,
    *,
    on_malformed: str = "skip",
) -> dict[str, BFCLTask]:
    """Scoreable tasks of one category. See :func:`load_task_report`."""
    return load_task_report(category, root, on_malformed=on_malformed).tasks


def load_task(
    task_id: str, category: str | None = None, root: str | os.PathLike[str] | None = None
) -> BFCLTask:
    """Load a single task. ``category`` defaults to the prefix of ``task_id``."""
    category = category or task_id.rsplit("_", 1)[0]
    tasks = load_tasks(category, root, on_malformed="keep")
    if task_id not in tasks:
        raise KeyError(f"task {task_id!r} not found in {category}")
    return tasks[task_id]


# --------------------------------------------------------------------------- #
# state snapshots
# --------------------------------------------------------------------------- #


def snapshot_instances(instances: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """Deep-copy the public attributes of each simulator instance.

    Private (``_``-prefixed) attributes are dropped, matching the official
    ``_compare_instances``. Deep copies are taken because the same instance keeps
    mutating as later turns execute, and a snapshot that aliased it would silently
    compare a turn against itself.
    """
    snapshot: dict[str, dict[str, Any]] = {}
    for class_name, instance in instances.items():
        snapshot[class_name] = {
            attribute: copy.deepcopy(value)
            for attribute, value in vars(instance).items()
            if not attribute.startswith("_")
        }
    return snapshot


@dataclass
class TurnSnapshot:
    """Ground truth after one turn."""

    turn: int
    state: dict[str, dict[str, Any]]
    responses: list[str]
    calls: list[str]

    @property
    def is_empty_turn(self) -> bool:
        return not self.calls


@dataclass
class GroundTruthTrace:
    """The reference trajectory, executed once per session at reset."""

    task_id: str
    turns: list[TurnSnapshot] = field(default_factory=list)
    initial_state: dict[str, dict[str, Any]] = field(default_factory=dict)

    def snapshot(self, turn: int) -> TurnSnapshot | None:
        for entry in self.turns:
            if entry.turn == turn:
                return entry
        return None

    def state_attributes(self) -> list[tuple[str, str]]:
        """Every ``(class, attribute)`` pair the comparison can ever look at."""
        pairs: set[tuple[str, str]] = set()
        for state in [self.initial_state, *[t.state for t in self.turns]]:
            for class_name, attributes in state.items():
                for attribute in attributes:
                    pairs.add((class_name, attribute))
        return sorted(pairs)


def build_expectations(
    trace: "GroundTruthTrace",
) -> dict[tuple[str, str], list[tuple[int, str]]]:
    """Per ``(class, attribute)``, the ``(target_revision, expected_value)`` per turn.

    This is what turns a moving requirement into something an episode closure can
    be attributed against: two turns wanting the same value share a revision, so a
    repair across them still counts as recovery, while a turn that moves the goal
    onto an already-broken world does not.
    """
    pairs = trace.state_attributes()
    expectations: dict[tuple[str, str], list[tuple[int, str]]] = {}
    for class_name, attribute in pairs:
        per_turn: list[str] = []
        for snapshot in trace.turns:
            attributes = snapshot.state.get(class_name, {})
            per_turn.append(
                canonical_value(attributes[attribute])
                if attribute in attributes
                else "<absent>"
            )
        expectations[(class_name, attribute)] = list(
            zip(target_revisions(per_turn), per_turn)
        )
    return expectations


# --------------------------------------------------------------------------- #
# comparison
# --------------------------------------------------------------------------- #


def compare_states(
    model_state: Mapping[str, Mapping[str, Any]],
    ground_truth_state: Mapping[str, Mapping[str, Any]],
) -> dict[tuple[str, str], bool]:
    """Per-attribute equality, using each simulator class's own ``__eq__``.

    An attribute present in the ground truth but missing from the model snapshot
    counts as a mismatch, not as a missing verdict: the classes are constructed
    from the same ``initial_config``, so a genuinely absent attribute means the
    model's world diverged structurally.
    """
    verdicts: dict[tuple[str, str], bool] = {}
    for class_name, attributes in ground_truth_state.items():
        model_attributes = model_state.get(class_name, {})
        for attribute, expected in attributes.items():
            if attribute not in model_attributes:
                verdicts[(class_name, attribute)] = False
                continue
            try:
                verdicts[(class_name, attribute)] = not (
                    model_attributes[attribute] != expected
                )
            except Exception:  # noqa: BLE001 - a comparison that explodes is a mismatch
                verdicts[(class_name, attribute)] = False
    return verdicts


def responses_contain(
    accumulated: Sequence[str], required: Sequence[str]
) -> tuple[bool, list[str]]:
    """Multiset containment, order-independent -- the official ``_is_subsequence_unordered``.

    Returns ``(ok, missing)``.
    """
    available = Counter(accumulated)
    missing: list[str] = []
    for item in required:
        if available.get(item, 0) > 0:
            available[item] -= 1
        else:
            missing.append(item)
    return (not missing), missing


# --------------------------------------------------------------------------- #
# action-level deterministic errors (plan section 7.1)
# --------------------------------------------------------------------------- #


def _call_name(call: str) -> str | None:
    match = _CALL_NAME_RE.match(call)
    return match.group(1) if match else None


def classify_action_errors(
    *,
    parse_ok: bool,
    decoded_calls: Sequence[str],
    available_tools: Mapping[str, Mapping[str, Any]],
    execution_results: Sequence[str],
) -> dict[str, RubricEvaluation]:
    """Verdicts for the four action rubrics, for one executed step.

    Only deterministic, unambiguous errors are judged here. A call that runs fine
    but achieves the wrong thing is *not* an action-level error -- that is what the
    turn-boundary state comparison is for. Judging it per action would flag a
    perfectly normal multi-step plan as an error at every intermediate step
    (plan section 7.1).
    """
    verdicts = {
        rubric_id: RubricEvaluation(rubric_id=rubric_id, passed=True)
        for rubric_id in ACTION_RUBRICS
    }

    if not parse_ok:
        verdicts["action:no_parse_error"] = RubricEvaluation(
            "action:no_parse_error", passed=False, trace="action did not decode"
        )
        # Nothing ran, so the remaining three have no evidence this step.
        for rubric_id in ("action:no_unknown_tool", "action:no_invalid_arguments",
                          "action:no_execution_error"):
            verdicts[rubric_id] = RubricEvaluation(
                rubric_id, passed=None, error="action did not decode"
            )
        return verdicts

    if not decoded_calls:
        # A turn-ending "no more calls" step. It cannot violate an action rubric,
        # and asserting that it satisfies them would wrongly close open episodes.
        for rubric_id in ACTION_RUBRICS:
            verdicts[rubric_id] = RubricEvaluation(
                rubric_id, passed=None, error="no calls in this action"
            )
        return verdicts

    unknown: list[str] = []
    invalid: list[str] = []
    for call in decoded_calls:
        name = _call_name(call)
        if name is None:
            invalid.append(call)
            continue
        document = available_tools.get(name)
        if document is None:
            unknown.append(name)
            continue
        required = (document.get("parameters") or {}).get("required") or []
        for parameter in required:
            if not re.search(rf"[(,]\s*{re.escape(parameter)}\s*=", call):
                invalid.append(f"{name}: missing required argument {parameter!r}")

    if unknown:
        verdicts["action:no_unknown_tool"] = RubricEvaluation(
            "action:no_unknown_tool", passed=False,
            trace="unknown tools: " + ", ".join(sorted(set(unknown))),
        )
    if invalid:
        verdicts["action:no_invalid_arguments"] = RubricEvaluation(
            "action:no_invalid_arguments", passed=False, trace="; ".join(invalid[:5])
        )

    failures = [r for r in execution_results if is_execution_failure(r)]
    if failures:
        verdicts["action:no_execution_error"] = RubricEvaluation(
            "action:no_execution_error", passed=False, trace="; ".join(failures[:5])
        )

    return verdicts


# --------------------------------------------------------------------------- #
# rubric compilation (plan section 7.1)
# --------------------------------------------------------------------------- #


def compile_rubrics(task: BFCLTask, trace: GroundTruthTrace) -> list[RubricSpec]:
    """Compile one task's hard rubrics from its executed ground truth.

    Three families:

    * ``action:*`` -- deterministic per-action safety rubrics. They open on a bad
      call and close on the next clean one, which is what makes same-turn recovery
      observable.
    * ``state:{class}:{attribute}`` -- safety rubrics compared at turn boundaries
      against the current turn's ground-truth snapshot (see the module docstring
      for why the turn index is not in the id).
    * ``response:turn{u}`` -- goal rubrics. Unmet is ``PENDING``, never
      ``VIOLATED``: a response that has not appeared yet is an omission, not a
      forbidden state (plan section 4.1). A turn whose ground truth calls nothing
      instead gets ``no_tool_call:turn{u}``.
    """
    specs: list[RubricSpec] = []

    for rubric_id, description in ACTION_RUBRICS.items():
        specs.append(
            RubricSpec(
                rubric_id=rubric_id,
                kind=RubricKind.SAFETY,
                hard=True,
                source="bfcl_action_level",
                initial_state=RubricState.SATISFIED,
                natural_language=description,
                forbidden_postconditions=[f"not ({description})"],
            )
        )

    for class_name, attribute in trace.state_attributes():
        specs.append(
            RubricSpec(
                rubric_id=state_rubric_id(class_name, attribute),
                kind=RubricKind.SAFETY,
                hard=True,
                source="bfcl_state_check",
                initial_state=RubricState.SATISFIED,
                natural_language=(
                    f"{class_name}.{attribute} must match the reference state for the "
                    "current turn"
                ),
                forbidden_postconditions=[
                    f"{class_name}.{attribute} differs from the reference state"
                ],
            )
        )

    for snapshot in trace.turns:
        if snapshot.is_empty_turn:
            specs.append(
                RubricSpec(
                    rubric_id=no_tool_call_rubric_id(snapshot.turn),
                    kind=RubricKind.SAFETY,
                    hard=True,
                    source="bfcl_empty_ground_truth_turn",
                    initial_state=RubricState.SATISFIED,
                    natural_language=(
                        f"turn {snapshot.turn} requires no tool call; the model must "
                        "ask for the missing information instead"
                    ),
                    forbidden_postconditions=[
                        f"a tool executed during turn {snapshot.turn}"
                    ],
                    turn=snapshot.turn,
                )
            )
        if snapshot.responses:
            specs.append(
                RubricSpec(
                    rubric_id=response_rubric_id(snapshot.turn),
                    kind=RubricKind.GOAL,
                    hard=True,
                    source="bfcl_response_check",
                    initial_state=RubricState.PENDING,
                    natural_language=(
                        f"the reference execution responses of turn {snapshot.turn} must "
                        "appear among the responses produced so far"
                    ),
                    desired_postcondition=(
                        f"turn {snapshot.turn} reference responses are all present"
                    ),
                    turn=snapshot.turn,
                )
            )

    return specs


def available_tool_documents(
    task: BFCLTask, turn: int, all_documents: Mapping[str, Mapping[str, Any]]
) -> dict[str, Mapping[str, Any]]:
    """Tool docs visible at ``turn``.

    ``multi_turn_miss_func`` removes the held-out functions from the tool list and
    reveals them at the holdout turn, so the set grows mid-rollout. Calling a
    not-yet-revealed function is an ``action:no_unknown_tool`` violation, which is
    the behaviour the category is designed to elicit.
    """
    held_out = {
        name
        for holdout_turn, names in task.missed_function.items()
        if holdout_turn > turn
        for name in names
    }
    return {name: doc for name, doc in all_documents.items() if name not in held_out}


def load_tool_documents(
    involved_classes: Iterable[str],
    root: str | os.PathLike[str] | None = None,
    *,
    category: str = "multi_turn_base",
) -> dict[str, dict[str, Any]]:
    """Officially preprocessed function docs, keyed by function name."""
    try:
        from bfcl_eval.constants.executable_backend_config import (  # noqa: PLC0415
            MULTI_TURN_FUNC_DOC_FILE_MAPPING,
        )
    except ImportError:
        # BFCL v1.3 kept this mapping in category_mapping.
        from bfcl_eval.constants.category_mapping import (  # type: ignore[no-redef]  # noqa: PLC0415
            MULTI_TURN_FUNC_DOC_FILE_MAPPING,
        )

    data_root = bfcl_data_root(root)
    ordered_documents: list[dict[str, Any]] = []
    for class_name in involved_classes:
        filename = MULTI_TURN_FUNC_DOC_FILE_MAPPING.get(class_name)
        if filename is None:
            raise KeyError(f"no function-doc file mapped for class {class_name!r}")
        ordered_documents.extend(
            _read_jsonl(data_root / "multi_turn_func_doc" / filename)
        )

    try:
        from bfcl_eval.model_handler.utils import (  # noqa: PLC0415
            func_doc_language_specific_pre_processing,
        )
    except ImportError:
        # The author-lineage Qwen handler passes the stored documents through
        # unchanged; v1.3 appended a language hint here.
        processed = copy.deepcopy(ordered_documents)
    else:
        processed = func_doc_language_specific_pre_processing(
            copy.deepcopy(ordered_documents), category
        )
    return {document["name"]: document for document in processed}
