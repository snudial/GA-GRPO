"""Transport-agnostic rollout DTOs and interfaces (ADR 0001).

Three collaborators, none of which knows the others' implementation:

* :class:`GenerationBackend` -- turns a prompt into student action text. vLLM in
  production, a scripted fake in tests. It does **not** compute the loss: OPSD
  needs a student forward, a privileged teacher forward and a backward over the
  same action tokens, which an inference server cannot provide.
* :class:`EnvironmentBackend` -- creates a session, steps it, closes it. Satisfied
  in-process by :class:`bfcl_env.environment_service.LocalEnvironmentBackend`
  and over the wire by ``HttpEnvironmentBackend``.
* :class:`EnvironmentRuntime` -- the benchmark itself. The only place where
  AppWorld and BFCL differ.

Everything here is JSON-serialisable so the identical objects cross an HTTP
boundary or don't, without the caller caring which.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Mapping, Protocol, Sequence, runtime_checkable

from bfcl_env.cost_ledger import MutationEvent
from bfcl_env.schemas import (
    RubricEvaluation,
    RubricSpec,
    RubricState,
    ToolCall,
    to_jsonable,
)

__all__ = [
    "PROTOCOL_VERSION",
    "StructuredAction",
    "PolicyVersion",
    "TaskRequest",
    "SessionInfo",
    "StepResult",
    "Generation",
    "HealthInfo",
    "EnvironmentError_",
    "SessionNotFound",
    "SequenceConflict",
    "SessionExpired",
    "InfrastructureError",
    "GenerationBackend",
    "EnvironmentBackend",
    "EnvironmentRuntime",
    "RuntimeReset",
    "RuntimeStep",
]

PROTOCOL_VERSION = "1"
"""Bumped when a field's meaning changes. Reported by ``health()`` and checked by
the client so a stale service is caught before a run, not after it."""


# --------------------------------------------------------------------------- #
# errors
# --------------------------------------------------------------------------- #


class EnvironmentError_(RuntimeError):
    """Base class for environment-service failures.

    Named with a trailing underscore to avoid shadowing the builtin ``OSError``
    alias ``EnvironmentError``.
    """


class SessionNotFound(EnvironmentError_):
    """Unknown or already-closed session id."""


class SessionExpired(EnvironmentError_):
    """The session outlived its TTL and was reclaimed."""


class InfrastructureError(EnvironmentError_):
    """The service cannot serve this request at all.

    Raised by a runtime for a failure of the *harness*: a corrupted session, a
    database that will not load, a crashed evaluator process. It becomes a 5xx so
    the client retries and, if it persists, the whole run stops.

    A failure of the *task* is never this. A tool that returns an error, an API
    that rejects an argument, a wrong password, an insufficient balance -- all of
    those are ordinary results the policy is expected to react to, and they come
    back as an HTTP 200 ``StepResult``. Turning them into HTTP errors would make
    the environment refuse to report exactly the events this project studies.

    A contained failure -- the runtime raised, but the service is healthy -- is
    also not this: it is reported as ``StepResult.environment_error``, which aborts
    that one rollout without implicating the service.
    """


class SequenceConflict(EnvironmentError_):
    """A ``sequence_no`` was reused with different action bytes, or skipped.

    A retry must resend the *same* action. Anything else is a client bug, and
    executing it would silently place a second order.
    """


# --------------------------------------------------------------------------- #
# small value objects
# --------------------------------------------------------------------------- #


@dataclass
class _Dto:
    def to_dict(self) -> dict[str, Any]:
        return to_jsonable(asdict(self))


@dataclass
class StructuredAction(_Dto):
    """One assistant action: the raw text plus whatever parsed out of it.

    ``raw_text`` is authoritative. It is what the student actually emitted and what
    the teacher will be asked to score, so it travels with the action even when
    parsing failed.
    """

    raw_text: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    parse_ok: bool = True
    parse_error: str | None = None
    parse_kind: str | None = None
    """Benchmark-declared parse outcome, e.g. ``TOOL_CALL`` / ``END_TURN`` /
    ``PARSE_ERROR``. ``parse_ok`` alone cannot distinguish "the model finished its
    turn" from "the model emitted a broken call", and those are different events."""

    def fingerprint(self) -> str:
        """Identity used to detect a retry carrying different bytes."""
        import hashlib

        return hashlib.sha256(self.raw_text.encode("utf-8")).hexdigest()

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "StructuredAction":
        return cls(
            raw_text=data["raw_text"],
            tool_calls=[ToolCall.from_dict(c) for c in data.get("tool_calls") or []],
            parse_ok=bool(data.get("parse_ok", True)),
            parse_error=data.get("parse_error"),
            parse_kind=data.get("parse_kind"),
        )


@dataclass
class PolicyVersion(_Dto):
    """Which policy produced a rollout (plan 9.3).

    A batch whose ``policy_step`` does not match the trainer's current step is
    dropped, not trained on.
    """

    policy_step: int
    model_checksum: str | None = None
    adapter_checksum: str | None = None
    """LoRA adapter state; ``None`` under full fine-tuning or a fixed base."""

    def matches(self, other: "PolicyVersion") -> bool:
        if self.policy_step != other.policy_step:
            return False
        for mine, theirs in (
            (self.model_checksum, other.model_checksum),
            (self.adapter_checksum, other.adapter_checksum),
        ):
            if mine is not None and theirs is not None and mine != theirs:
                return False
        return True

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "PolicyVersion":
        return cls(
            policy_step=int(data["policy_step"]),
            model_checksum=data.get("model_checksum"),
            adapter_checksum=data.get("adapter_checksum"),
        )


@dataclass
class TaskRequest(_Dto):
    """One (task, seed) the manager wants a rollout for."""

    task_id: str
    seed: int
    rollout_id: str
    benchmark: str = ""
    split: str | None = None
    max_steps: int = 30
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "TaskRequest":
        return cls(
            task_id=data["task_id"],
            seed=int(data["seed"]),
            rollout_id=data["rollout_id"],
            benchmark=data.get("benchmark", ""),
            split=data.get("split"),
            max_steps=int(data.get("max_steps", 30)),
            metadata=dict(data.get("metadata") or {}),
        )


@dataclass
class Generation(_Dto):
    """What the generation backend produced for one prompt."""

    text: str
    token_ids: list[int] = field(default_factory=list)
    finish_reason: str = "stop"
    policy_version: PolicyVersion | None = None

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Generation":
        version = data.get("policy_version")
        return cls(
            text=data["text"],
            token_ids=list(data.get("token_ids") or []),
            finish_reason=data.get("finish_reason", "stop"),
            policy_version=PolicyVersion.from_dict(version) if version else None,
        )


# --------------------------------------------------------------------------- #
# session + step
# --------------------------------------------------------------------------- #


@dataclass
class SessionInfo(_Dto):
    """Everything the manager needs to start prompting, returned by ``create_session``.

    The rubric specs come back here so the manager can build the teacher's
    privileged packet without a second round trip, and so a mismatch between the
    service's compiler and the trainer's expectations surfaces immediately.
    """

    session_id: str
    task_id: str
    rollout_id: str
    seed: int
    benchmark: str
    system_prompt: str
    task_prompt: str
    rubric_specs: list[RubricSpec] = field(default_factory=list)
    initial_rubric_states: dict[str, RubricState] = field(default_factory=dict)
    initial_state_hash: str = ""
    max_steps: int = 30
    expires_at: float | None = None
    """Unix timestamp. The manager must close or finish before this."""

    backend_version: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    volatile_metadata_keys: list[str] = field(default_factory=list)
    """Metadata keys whose value changes every run (a generated experiment name, a
    wall-clock stamp). The manager files these separately so a stored rollout stays
    byte-identical across replays of the same trajectory. The runtime declares
    them; nothing downstream needs to know which benchmark they came from."""

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "SessionInfo":
        return cls(
            session_id=data["session_id"],
            task_id=data["task_id"],
            rollout_id=data["rollout_id"],
            seed=int(data["seed"]),
            benchmark=data["benchmark"],
            system_prompt=data.get("system_prompt", ""),
            task_prompt=data.get("task_prompt", ""),
            rubric_specs=[RubricSpec.from_dict(s) for s in data.get("rubric_specs") or []],
            initial_rubric_states={
                k: RubricState(v) for k, v in (data.get("initial_rubric_states") or {}).items()
            },
            initial_state_hash=data.get("initial_state_hash", ""),
            max_steps=int(data.get("max_steps", 30)),
            expires_at=data.get("expires_at"),
            backend_version=dict(data.get("backend_version") or {}),
            metadata=dict(data.get("metadata") or {}),
            volatile_metadata_keys=list(data.get("volatile_metadata_keys") or []),
        )


@dataclass
class StepResult(_Dto):
    """The complete record of one action, produced atomically with its evaluation.

    ``rubric_evaluations`` (raw verdicts) is the source of truth: the offline
    compiler recomputes ``rubric_states`` from it and must agree.
    ``rubric_states`` is here so the manager can stop early and pick visualisation
    samples without a second call that could observe a different world.
    """

    session_id: str
    sequence_no: int
    observation: str
    parsed_action: StructuredAction
    observation_role: str | None = None
    """Chat role for ``observation``. Tool output and a new user turn are not
    interchangeable in model-specific chat templates."""
    observation_messages: list[dict[str, str]] = field(default_factory=list)
    """Exact ordered chat messages produced by one environment step.

    BFCL parallel calls require one separate ``role=tool`` message per execution
    result. ``observation`` remains the joined audit text for compatibility.
    """
    system_prompt_update: str | None = None
    """Replacement for the first system message before the next generation."""
    state_before_hash: str = ""
    state_after_hash: str = ""
    state_delta: dict[str, Any] = field(default_factory=dict)
    rubric_states: dict[str, RubricState] = field(default_factory=dict)
    rubric_evaluations: dict[str, RubricEvaluation] = field(default_factory=dict)
    evaluator_evidence: dict[str, Any] = field(default_factory=dict)
    mutations: list[MutationEvent] = field(default_factory=list)
    tool_execution_ok: bool | None = None
    """Did the call run. Never a rubric verdict (plan 6.2)."""

    terminated: bool = False
    """The policy ended the episode."""

    truncated: bool = False
    """A budget ran out."""

    environment_error: str | None = None
    """The world itself failed. Only this makes a rollout aborted (plan 3.1)."""

    replayed: bool = False
    """This result was served from the idempotency cache, not re-executed."""

    turn: int = 0

    @property
    def finished(self) -> bool:
        return self.terminated or self.truncated or self.environment_error is not None

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "StepResult":
        return cls(
            session_id=data["session_id"],
            sequence_no=int(data["sequence_no"]),
            observation=data.get("observation", ""),
            observation_role=data.get("observation_role"),
            observation_messages=[
                {"role": str(message["role"]), "content": str(message["content"])}
                for message in (data.get("observation_messages") or [])
            ],
            system_prompt_update=data.get("system_prompt_update"),
            parsed_action=StructuredAction.from_dict(data["parsed_action"]),
            state_before_hash=data.get("state_before_hash", ""),
            state_after_hash=data.get("state_after_hash", ""),
            state_delta=dict(data.get("state_delta") or {}),
            rubric_states={
                k: RubricState(v) for k, v in (data.get("rubric_states") or {}).items()
            },
            rubric_evaluations={
                k: RubricEvaluation.from_dict({"rubric_id": k, **v})
                for k, v in (data.get("rubric_evaluations") or {}).items()
            },
            evaluator_evidence=dict(data.get("evaluator_evidence") or {}),
            mutations=[MutationEvent(**m) for m in data.get("mutations") or []],
            tool_execution_ok=data.get("tool_execution_ok"),
            terminated=bool(data.get("terminated", False)),
            truncated=bool(data.get("truncated", False)),
            environment_error=data.get("environment_error"),
            replayed=bool(data.get("replayed", False)),
            turn=int(data.get("turn", 0)),
        )


@dataclass
class HealthInfo(_Dto):
    """What a service reports so a mismatched deployment fails fast."""

    ok: bool
    benchmark: str
    protocol_version: str = PROTOCOL_VERSION
    benchmark_commit: str | None = None
    rubric_compiler_version: str | None = None
    num_open_sessions: int = 0
    detail: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "HealthInfo":
        return cls(
            ok=bool(data["ok"]),
            benchmark=data.get("benchmark", ""),
            protocol_version=data.get("protocol_version", PROTOCOL_VERSION),
            benchmark_commit=data.get("benchmark_commit"),
            rubric_compiler_version=data.get("rubric_compiler_version"),
            num_open_sessions=int(data.get("num_open_sessions", 0)),
            detail=dict(data.get("detail") or {}),
        )


# --------------------------------------------------------------------------- #
# interfaces
# --------------------------------------------------------------------------- #


@runtime_checkable
class GenerationBackend(Protocol):
    """Produces student actions. vLLM in production."""

    def policy_version(self) -> PolicyVersion: ...

    def generate(
        self,
        prompts: Sequence[Sequence[Mapping[str, str]]],
        *,
        max_new_tokens: int,
        temperature: float = 1.0,
        seed: int | None = None,
        stop: Sequence[str] | None = None,
    ) -> list[Generation]:
        """One generation per prompt, in the same order."""


@runtime_checkable
class EnvironmentBackend(Protocol):
    """Session-oriented environment access (ADR 0001 section 3)."""

    benchmark: str

    def create_session(self, task_id: str, seed: int, rollout_id: str) -> SessionInfo: ...

    def step(
        self, session_id: str, sequence_no: int, action: StructuredAction
    ) -> StepResult:
        """Execute one action.

        Idempotent on ``(session_id, sequence_no)``: a resend of the same action
        returns the stored result with ``replayed=True`` and does not touch the
        world. A resend with different bytes raises :class:`SequenceConflict`.
        """

    def close_session(self, session_id: str) -> None: ...

    def health(self) -> HealthInfo: ...


# --------------------------------------------------------------------------- #
# runtime (the only benchmark-specific surface)
# --------------------------------------------------------------------------- #


@dataclass
class RuntimeReset:
    """What a runtime returns when a task is instantiated."""

    handle: Any
    """Opaque per-session state. Only the runtime interprets it."""

    system_prompt: str
    task_prompt: str
    rubric_specs: list[RubricSpec]
    state_hash: str = ""
    max_steps: int = 30
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RuntimeStep:
    """What a runtime returns for one executed action.

    Raw evaluator verdicts only: turning verdicts into rubric *states* needs the
    session's history, which :class:`LocalEnvironmentBackend` owns via a
    :class:`bfcl_env.rubric_state_machine.RubricTracker` per rubric.
    """

    observation: str
    observation_role: str | None = None
    observation_messages: list[dict[str, str]] = field(default_factory=list)
    system_prompt_update: str | None = None
    rubric_evaluations: dict[str, RubricEvaluation] = field(default_factory=dict)
    evaluator_evidence: dict[str, Any] = field(default_factory=dict)
    state_hash: str = ""
    state_delta: dict[str, Any] = field(default_factory=dict)
    mutations: list[MutationEvent] = field(default_factory=list)
    tool_execution_ok: bool | None = None
    terminated: bool = False
    turn: int = 0


@runtime_checkable
class EnvironmentRuntime(Protocol):
    """A benchmark. The only place AppWorld and BFCL differ (ADR 0001)."""

    benchmark: str

    def reset(self, task_id: str, seed: int) -> RuntimeReset: ...

    def parse_action(self, raw_text: str) -> StructuredAction:
        """Parse model output. A failure returns ``parse_ok=False``, not an exception:
        an unparseable action is a policy error to be scored, not a crash."""

    def execute(self, handle: Any, action: StructuredAction) -> RuntimeStep: ...

    def close(self, handle: Any) -> None: ...

    def describe(self) -> dict[str, Any]:
        """Version information for ``health()``."""
