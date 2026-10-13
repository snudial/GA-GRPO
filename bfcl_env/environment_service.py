"""Session bookkeeping shared by every environment transport (ADR 0001 section 3).

``LocalEnvironmentBackend`` wraps an :class:`~bfcl_env.rollout_protocol.EnvironmentRuntime`
and adds everything that must behave identically in-process and behind HTTP:

* one isolated session per rollout, with an explicit TTL and close,
* requests within a session serialised by a lock, so two workers cannot interleave
  actions against the same world,
* **idempotent** ``step`` on ``(session_id, sequence_no)``: a retry after a network
  timeout replays the stored result instead of placing a second order,
* rubric *states* derived from raw verdicts by a per-session
  :class:`~bfcl_env.rubric_state_machine.RubricTracker`, so the online states and
  the offline recompilation agree by construction,
* environment failure reported as ``environment_error``, never as a task failure.

The AppWorld HTTP service is this class plus a thin JSON transport; the BFCL
backend is this class used directly.
"""

from __future__ import annotations

import threading
import time
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Callable

from bfcl_env.rollout_protocol import (
    EnvironmentRuntime,
    HealthInfo,
    InfrastructureError,
    SequenceConflict,
    SessionExpired,
    SessionInfo,
    SessionNotFound,
    StepResult,
    StructuredAction,
)
from bfcl_env.rubric_state_machine import RubricTracker
from bfcl_env.schemas import RubricEvaluation, RubricState

__all__ = ["LocalEnvironmentBackend", "SessionLimitExceeded"]


class SessionLimitExceeded(RuntimeError):
    """The service is already holding ``max_sessions`` live worlds."""


@dataclass
class _Session:
    info: SessionInfo
    handle: Any
    trackers: dict[str, RubricTracker]
    max_steps: int
    ttl_seconds: float
    created_at: float
    lock: threading.RLock = field(default_factory=threading.RLock)
    next_sequence_no: int = 0
    steps_taken: int = 0
    finished: bool = False
    last_used_at: float = 0.0
    results: "OrderedDict[int, StepResult]" = field(default_factory=OrderedDict)
    fingerprints: dict[int, str] = field(default_factory=dict)
    lowest_cached_sequence_no: int = 0
    last_state_hash: str = ""
    """Tracked explicitly: the previous StepResult may have been evicted from the
    bounded idempotency cache."""

    @property
    def expires_at(self) -> float:
        return self.created_at + self.ttl_seconds

    def is_expired(self, now: float) -> bool:
        return now > self.expires_at


class LocalEnvironmentBackend:
    """In-process :class:`~bfcl_env.rollout_protocol.EnvironmentBackend`.

    Args:
        runtime: the benchmark.
        ttl_seconds: how long a session may live before it is reclaimed. A crashed
            trainer must not pin a world forever.
        idempotency_window: how many recent ``sequence_no`` results are retained
            for replay. A retry beyond this window is a :class:`SequenceConflict`
            rather than a silent re-execution.
        max_sessions: refuse rather than thrash when worlds are expensive.
        clock: injectable for tests; defaults to :func:`time.time`.
    """

    def __init__(
        self,
        runtime: EnvironmentRuntime,
        *,
        ttl_seconds: float = 3600.0,
        idempotency_window: int = 4,
        max_sessions: int = 64,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.runtime = runtime
        self.benchmark = runtime.benchmark
        self.ttl_seconds = ttl_seconds
        self.idempotency_window = max(1, idempotency_window)
        self.max_sessions = max_sessions
        self._clock = clock
        self._sessions: dict[str, _Session] = {}
        self._closed_ids: set[str] = set()
        self._registry_lock = threading.RLock()

    # ---- session lifecycle ------------------------------------------------- #

    def create_session(self, task_id: str, seed: int, rollout_id: str) -> SessionInfo:
        now = self._clock()
        self.reap_expired()

        with self._registry_lock:
            if len(self._sessions) >= self.max_sessions:
                raise SessionLimitExceeded(
                    f"{len(self._sessions)} sessions already open (max {self.max_sessions})"
                )

        reset = self.runtime.reset(task_id, seed)
        trackers = {spec.rubric_id: RubricTracker(spec) for spec in reset.rubric_specs}
        session_id = f"{self.benchmark}:{task_id}:{rollout_id}:{uuid.uuid4().hex[:8]}"

        info = SessionInfo(
            session_id=session_id,
            task_id=task_id,
            rollout_id=rollout_id,
            seed=seed,
            benchmark=self.benchmark,
            system_prompt=reset.system_prompt,
            task_prompt=reset.task_prompt,
            rubric_specs=list(reset.rubric_specs),
            initial_rubric_states={rid: t.state for rid, t in trackers.items()},
            initial_state_hash=reset.state_hash,
            max_steps=reset.max_steps,
            expires_at=now + self.ttl_seconds,
            backend_version=self.runtime.describe(),
            metadata=dict(reset.metadata),
            volatile_metadata_keys=list(reset.metadata.get("_volatile_keys") or []),
        )

        session = _Session(
            info=info,
            handle=reset.handle,
            trackers=trackers,
            max_steps=reset.max_steps,
            ttl_seconds=self.ttl_seconds,
            created_at=now,
            last_used_at=now,
            last_state_hash=reset.state_hash,
        )
        with self._registry_lock:
            self._sessions[session_id] = session
        return info

    def close_session(self, session_id: str) -> None:
        """Idempotent for a session this backend already closed."""
        with self._registry_lock:
            session = self._sessions.pop(session_id, None)
            if session is None:
                if session_id in self._closed_ids:
                    return
                raise SessionNotFound(f"unknown session {session_id!r}")
            self._closed_ids.add(session_id)

        with session.lock:
            try:
                self.runtime.close(session.handle)
            except Exception:  # noqa: BLE001 - closing must not mask the rollout result
                pass

    def reap_expired(self) -> list[str]:
        """Close sessions past their TTL. Returns the ids reclaimed."""
        now = self._clock()
        with self._registry_lock:
            stale = [sid for sid, s in self._sessions.items() if s.is_expired(now)]
        for session_id in stale:
            try:
                self.close_session(session_id)
            except SessionNotFound:
                continue
        return stale

    @property
    def num_open_sessions(self) -> int:
        with self._registry_lock:
            return len(self._sessions)

    def health(self) -> HealthInfo:
        description = self.runtime.describe()
        return HealthInfo(
            ok=True,
            benchmark=self.benchmark,
            benchmark_commit=description.get("commit"),
            rubric_compiler_version=description.get("rubric_compiler_version"),
            num_open_sessions=self.num_open_sessions,
            detail=description,
        )

    # ---- stepping ---------------------------------------------------------- #

    def _resolve(self, session_id: str) -> _Session:
        with self._registry_lock:
            session = self._sessions.get(session_id)
            closed = session_id in self._closed_ids
        if session is None:
            if closed:
                raise SessionNotFound(f"session {session_id!r} is closed")
            raise SessionNotFound(f"unknown session {session_id!r}")
        if session.is_expired(self._clock()):
            self.close_session(session_id)
            raise SessionExpired(
                f"session {session_id!r} expired at {session.expires_at:.0f}"
            )
        return session

    def step(
        self, session_id: str, sequence_no: int, action: StructuredAction
    ) -> StepResult:
        session = self._resolve(session_id)
        fingerprint = action.fingerprint()

        with session.lock:
            cached = session.results.get(sequence_no)
            if cached is not None:
                if session.fingerprints.get(sequence_no) != fingerprint:
                    raise SequenceConflict(
                        f"sequence_no {sequence_no} of session {session_id!r} was already "
                        "executed with a different action; a retry must resend the same "
                        "bytes"
                    )
                replay = StepResult.from_dict(cached.to_dict())
                replay.replayed = True
                return replay

            if sequence_no < session.lowest_cached_sequence_no:
                raise SequenceConflict(
                    f"sequence_no {sequence_no} is older than the idempotency window "
                    f"(oldest retained: {session.lowest_cached_sequence_no})"
                )
            if sequence_no != session.next_sequence_no:
                raise SequenceConflict(
                    f"expected sequence_no {session.next_sequence_no}, got {sequence_no}; "
                    "steps may not be skipped or reordered"
                )
            if session.finished:
                raise SequenceConflict(
                    f"session {session_id!r} already finished; open a new session"
                )

            result = self._execute(session, sequence_no, action)
            self._remember(session, sequence_no, fingerprint, result)
            session.next_sequence_no = sequence_no + 1
            if result.finished:
                session.finished = True
            return result

    def _execute(
        self, session: _Session, sequence_no: int, action: StructuredAction
    ) -> StepResult:
        state_before = session.last_state_hash

        if action.parse_kind is None:
            # Parsing is benchmark-specific, so it belongs on the service side: a
            # remote caller has no way to run the runtime's decoder. The action's
            # identity is its raw text, which the idempotency fingerprint already
            # uses, so parsing here cannot change what a retry matches.
            try:
                action = self.runtime.parse_action(action.raw_text)
            except InfrastructureError:
                raise
            except Exception as error:  # noqa: BLE001 - a parser crash is data
                session.last_used_at = self._clock()
                return StepResult(
                    session_id=session.info.session_id,
                    sequence_no=sequence_no,
                    observation="",
                    parsed_action=action,
                    state_before_hash=state_before,
                    state_after_hash=state_before,
                    rubric_states={rid: t.state for rid, t in session.trackers.items()},
                    environment_error=f"parser crashed: {type(error).__name__}: {error}",
                    turn=session.steps_taken,
                )

        try:
            step = self.runtime.execute(session.handle, action)
        except InfrastructureError:
            # The service itself is unwell. Propagate so it surfaces as a 5xx and
            # the client retries, rather than silently aborting one rollout.
            raise
        except Exception as error:  # noqa: BLE001 - a crashed world is data, not a raise
            session.last_used_at = self._clock()
            return StepResult(
                session_id=session.info.session_id,
                sequence_no=sequence_no,
                observation="",
                parsed_action=action,
                state_before_hash=state_before,
                state_after_hash=state_before,
                rubric_states={rid: t.state for rid, t in session.trackers.items()},
                environment_error=f"{type(error).__name__}: {error}",
                turn=session.steps_taken,
            )

        # Raw verdicts -> states, through the same machine the offline compiler uses.
        evaluations: dict[str, RubricEvaluation] = {}
        states: dict[str, RubricState] = {}
        for rubric_id, tracker in session.trackers.items():
            evaluation = step.rubric_evaluations.get(rubric_id)
            if evaluation is None:
                evaluation = RubricEvaluation(
                    rubric_id=rubric_id, passed=None, error="no_verdict_this_step"
                )
            evaluations[rubric_id] = evaluation
            states[rubric_id] = tracker.update(evaluation.passed)

        session.steps_taken += 1
        session.last_used_at = self._clock()
        session.last_state_hash = step.state_hash
        truncated = not step.terminated and session.steps_taken >= session.max_steps

        return StepResult(
            session_id=session.info.session_id,
            sequence_no=sequence_no,
            observation=step.observation,
            observation_role=step.observation_role,
            observation_messages=list(step.observation_messages),
            system_prompt_update=step.system_prompt_update,
            parsed_action=action,
            state_before_hash=state_before,
            state_after_hash=step.state_hash,
            state_delta=dict(step.state_delta),
            rubric_states=states,
            rubric_evaluations=evaluations,
            evaluator_evidence=dict(step.evaluator_evidence),
            mutations=list(step.mutations),
            tool_execution_ok=step.tool_execution_ok,
            terminated=step.terminated,
            truncated=truncated,
            turn=step.turn,
        )

    def _remember(
        self,
        session: _Session,
        sequence_no: int,
        fingerprint: str,
        result: StepResult,
    ) -> None:
        session.results[sequence_no] = result
        session.fingerprints[sequence_no] = fingerprint
        while len(session.results) > self.idempotency_window:
            evicted, _ = session.results.popitem(last=False)
            session.fingerprints.pop(evicted, None)
            session.lowest_cached_sequence_no = evicted + 1
