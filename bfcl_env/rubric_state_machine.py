"""Deterministic rubric state machine and episode interval extraction.

This module is intentionally free of any benchmark, torch or environment import.
It answers exactly two questions:

1. Given the previous state of a rubric and a fresh evaluator verdict, what is the
   new state?  -> :func:`derive_state`
2. Given the full state sequence of one rubric, what episodes does it contain?
   -> :func:`extract_episode_intervals`

Transition table (plan section 4.2)::

    PENDING   -> SATISFIED : goal reached, not an episode
    SATISFIED -> VIOLATED  : episode opens
    PENDING   -> VIOLATED  : explicit forbidden state, episode opens
    VIOLATED  -> VIOLATED  : additional manifestation of the same episode
    VIOLATED  -> SATISFIED : episode closes
    VIOLATED  -> UNKNOWN   : held open, never closed by UNKNOWN
    terminal VIOLATED      : unresolved failure

An UNKNOWN run is transparent: the machine compares against the last *known*
state, so ``VIOLATED -> UNKNOWN -> VIOLATED`` stays one episode and
``VIOLATED -> UNKNOWN -> SATISFIED`` closes at the SATISFIED step.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Mapping, Sequence

from bfcl_env.schemas import (
    EpisodeCloseReason,
    RubricEvaluation,
    RubricKind,
    RubricSpec,
    RubricState,
    RubricStateVector,
)

__all__ = [
    "INITIAL_STEP",
    "RubricObservation",
    "EpisodeInterval",
    "attribute_close_reason",
    "derive_state",
    "RubricTracker",
    "build_state_sequence",
    "extract_episode_intervals",
    "extract_all_episode_intervals",
    "first_violation_step",
]

INITIAL_STEP = -1
"""Step index reserved for the pre-rollout state vector."""


# --------------------------------------------------------------------------- #
# Single-rubric transition
# --------------------------------------------------------------------------- #


def derive_state(
    kind: RubricKind,
    passed: bool | None,
    *,
    last_known_state: RubricState,
    ever_satisfied: bool,
) -> RubricState:
    """Map one evaluator verdict to a rubric state.

    Args:
        kind: ``GOAL`` (AppWorld ``no_op_fail``) or ``SAFETY`` (``no_op_pass``).
        passed: evaluator verdict; ``None`` when the predicate could not run.
        last_known_state: most recent non-UNKNOWN state of this rubric.
        ever_satisfied: whether this rubric has been SATISFIED at any earlier step.

    A failing *goal* is only a violation once the goal has actually been reached
    before, or once it is currently in a violated run. Before that it is a plain
    ``PENDING`` -- an unmet positive goal is not an error (plan section 4.1).
    """
    if passed is None:
        return RubricState.UNKNOWN
    if passed:
        return RubricState.SATISFIED
    if kind is RubricKind.SAFETY:
        return RubricState.VIOLATED
    # GOAL and currently failing.
    if ever_satisfied or last_known_state is RubricState.VIOLATED:
        return RubricState.VIOLATED
    return RubricState.PENDING


@dataclass
class RubricTracker:
    """Streaming per-rubric state, usable while a rollout is still running."""

    spec: RubricSpec
    state: RubricState = RubricState.PENDING
    last_known_state: RubricState = RubricState.PENDING
    ever_satisfied: bool = False

    def __post_init__(self) -> None:
        self.state = self.spec.initial_state
        self.last_known_state = self.spec.initial_state
        self.ever_satisfied = self.spec.initial_state is RubricState.SATISFIED

    def update(self, passed: bool | None) -> RubricState:
        new_state = derive_state(
            self.spec.kind,
            passed,
            last_known_state=self.last_known_state,
            ever_satisfied=self.ever_satisfied,
        )
        self.state = new_state
        if new_state is not RubricState.UNKNOWN:
            self.last_known_state = new_state
            if new_state is RubricState.SATISFIED:
                self.ever_satisfied = True
        return new_state


def build_state_sequence(
    rollout_id: str,
    specs: Sequence[RubricSpec],
    evaluations_per_step: Sequence[tuple[int, Mapping[str, RubricEvaluation]]],
    *,
    turns_per_step: Mapping[int, int] | None = None,
    include_initial: bool = True,
) -> list[RubricStateVector]:
    """Turn per-step evaluator verdicts into the per-step rubric state vectors.

    Args:
        rollout_id: identifier copied into every emitted vector.
        specs: the rubric declarations for this task.
        evaluations_per_step: ``[(step, {rubric_id: RubricEvaluation})]`` in
            execution order. A rubric absent from a step's mapping is treated as
            UNKNOWN for that step -- the evaluator produced no verdict.
        turns_per_step: optional ``step -> turn`` mapping stored on each vector.
        include_initial: emit the pre-rollout vector at ``step = -1``.
    """
    trackers = {spec.rubric_id: RubricTracker(spec) for spec in specs}
    vectors: list[RubricStateVector] = []

    if include_initial:
        vectors.append(
            RubricStateVector(
                rollout_id=rollout_id,
                step=INITIAL_STEP,
                rubrics={rid: t.state for rid, t in trackers.items()},
                turn=None,
            )
        )

    for step, evaluations in evaluations_per_step:
        states: dict[str, RubricState] = {}
        for rubric_id, tracker in trackers.items():
            evaluation = evaluations.get(rubric_id)
            passed = evaluation.passed if evaluation is not None else None
            states[rubric_id] = tracker.update(passed)
        # from_evaluations carries target_revision / expected / actual through, so
        # the offline recompilation can attribute closures exactly as the live
        # runtime would.
        vectors.append(
            RubricStateVector.from_evaluations(
                rollout_id,
                step,
                states,
                evaluations,
                turn=None if turns_per_step is None else turns_per_step.get(step),
            )
        )
    return vectors


# --------------------------------------------------------------------------- #
# Episode extraction
# --------------------------------------------------------------------------- #


@dataclass
class RubricObservation:
    """What one rubric wanted and saw at one step.

    Only rubrics whose expectation can move between turns carry these (BFCL
    ``state:*``). When they are absent, a closure cannot be attributed and is
    reported as ``UNKNOWN`` rather than assumed to be recovery.
    """

    target_revision: int | None = None
    expected_value: str | None = None
    actual_value: str | None = None

    @property
    def attributable(self) -> bool:
        return self.actual_value is not None


def attribute_close_reason(
    opened: RubricObservation,
    closed: RubricObservation,
    *,
    moving_target: bool = True,
) -> EpisodeCloseReason:
    """Decide whether the policy fixed the world or the requirement moved.

    ``moving_target=False`` means this rubric's expectation is fixed for the whole
    task -- AppWorld's evaluator predicates, and BFCL's action-level rubrics. There
    the only thing that *can* change is the world, and the world only changes by
    agent action, so a closure is a recovery by construction.

    For a rubric whose expectation advances with the benchmark's turns:

    ==============================  ==============  =====================
    expectation between open/close  world changed?  verdict
    ==============================  ==============  =====================
    unchanged (same revision)       yes             RECOVERED_BY_ACTION
    unchanged (same revision)       no              UNKNOWN
    changed                         no              SUPERSEDED_BY_TARGET
    changed                         yes             UNKNOWN
    ==============================  ==============  =====================

    The last row is deliberately not resolved. Once both the requirement and the
    world have moved, attribution would need to know whether the world moved
    *towards* the requirement or the requirement moved *onto* the world -- and at
    the granularity these benchmarks expose (BFCL bundles an entire filesystem
    into one ``root`` attribute) that is not decidable. A model doing legitimate
    unrelated work is enough to change ``root``, which would let a moved goalpost
    masquerade as a repair.

    So it is reported as unattributed and excluded from every recovery statistic.
    Under-counting recovery is recoverable; inflating it is the failure this whole
    mechanism exists to prevent.
    """
    if not moving_target:
        return EpisodeCloseReason.RECOVERED_BY_ACTION
    if not (opened.attributable and closed.attributable):
        return EpisodeCloseReason.UNKNOWN

    target_moved = _target_moved(opened, closed)
    world_moved = closed.actual_value != opened.actual_value

    if not target_moved:
        if not world_moved:
            # Neither the requirement nor the world changed, yet the verdict
            # flipped. The evaluation itself must have changed; do not guess.
            return EpisodeCloseReason.UNKNOWN
        return EpisodeCloseReason.RECOVERED_BY_ACTION
    if not world_moved:
        return EpisodeCloseReason.SUPERSEDED_BY_TARGET
    return EpisodeCloseReason.UNKNOWN


def _target_moved(opened: RubricObservation, closed: RubricObservation) -> bool:
    """Did the rubric's expectation change between the two observations?"""
    if opened.target_revision is not None and closed.target_revision is not None:
        return closed.target_revision != opened.target_revision
    if opened.expected_value is not None and closed.expected_value is not None:
        return closed.expected_value != opened.expected_value
    return True


@dataclass
class EpisodeInterval:
    """A single open->(close) interval, expressed only in step indices."""

    rubric_id: str
    open_step: int
    close_step: int | None = None
    manifestation_steps: list[int] = field(default_factory=list)
    reopen_index: int = 0
    close_reason: EpisodeCloseReason = EpisodeCloseReason.TERMINAL_UNRESOLVED
    open_observation: RubricObservation = field(default_factory=RubricObservation)
    close_observation: RubricObservation = field(default_factory=RubricObservation)

    @property
    def resolved(self) -> bool:
        return self.close_step is not None

    @property
    def recovered_by_action(self) -> bool:
        return self.close_reason is EpisodeCloseReason.RECOVERED_BY_ACTION


def _iter_states(
    states: Iterable[tuple[int, RubricState]] | Mapping[int, RubricState],
) -> list[tuple[int, RubricState]]:
    if isinstance(states, Mapping):
        return sorted(states.items())
    return sorted(states, key=lambda item: item[0])


def extract_episode_intervals(
    rubric_id: str,
    states: Iterable[tuple[int, RubricState]] | Mapping[int, RubricState],
    *,
    initial_state: RubricState | None = None,
    observations: Mapping[int, RubricObservation] | None = None,
) -> list[EpisodeInterval]:
    """Extract every episode of one rubric from its ordered state sequence.

    ``states`` may include the ``step = -1`` initial entry; if it does not,
    ``initial_state`` supplies the state the rollout started from. An episode is
    never opened at ``INITIAL_STEP``: a rubric that is already VIOLATED before the
    policy acts is a property of the task, not of the trajectory.

    ``observations`` supplies the expected/actual values needed to tell a genuine
    recovery from a requirement that moved onto the broken world. Passing it at all
    declares this rubric to have a *moving* expectation; omitting it declares a
    fixed one, where a closure is a recovery by construction.
    """
    lookup = observations or {}
    moving_target = bool(lookup)
    ordered = _iter_states(states)
    episodes: list[EpisodeInterval] = []
    if not ordered:
        return episodes

    if ordered[0][0] == INITIAL_STEP:
        start_state = ordered[0][1]
        ordered = ordered[1:]
    else:
        start_state = initial_state if initial_state is not None else RubricState.PENDING

    if start_state is RubricState.VIOLATED:
        # A rubric already violated before the policy acted cannot be blamed on any
        # action. Refuse rather than silently attributing it to the first step.
        raise ValueError(
            f"rubric {rubric_id!r} starts VIOLATED; compile it as PENDING (goal) or "
            "SATISFIED (safety) so that episodes stay attributable to actions"
        )

    open_episode: EpisodeInterval | None = None
    reopen_index = 0

    for step, state in ordered:
        if state is RubricState.UNKNOWN:
            # Transparent: neither opens, closes nor manifests anything.
            continue

        if state is RubricState.VIOLATED:
            if open_episode is None:
                open_episode = EpisodeInterval(
                    rubric_id=rubric_id,
                    open_step=step,
                    manifestation_steps=[step],
                    reopen_index=reopen_index,
                    open_observation=lookup.get(step, RubricObservation()),
                )
            else:
                open_episode.manifestation_steps.append(step)
        else:  # SATISFIED or PENDING
            if open_episode is not None:
                if state is RubricState.SATISFIED:
                    open_episode.close_step = step
                    open_episode.close_observation = lookup.get(
                        step, RubricObservation()
                    )
                    open_episode.close_reason = attribute_close_reason(
                        open_episode.open_observation,
                        open_episode.close_observation,
                        moving_target=moving_target,
                    )
                    episodes.append(open_episode)
                    open_episode = None
                    reopen_index += 1
                else:
                    # VIOLATED -> PENDING should be unreachable for a compiled
                    # rubric (a violated goal that is not yet met stays VIOLATED).
                    # Keep the episode open rather than inventing a recovery.
                    pass

    if open_episode is not None:
        open_episode.close_reason = EpisodeCloseReason.TERMINAL_UNRESOLVED
        episodes.append(open_episode)

    return episodes


def extract_all_episode_intervals(
    vectors: Sequence[RubricStateVector],
    specs: Sequence[RubricSpec] | None = None,
    *,
    hard_only: bool = True,
) -> list[EpisodeInterval]:
    """Extract episodes for every rubric present in a rollout's state vectors.

    Results are ordered by ``(open_step, rubric_id, reopen_index)`` so that the
    first element is the earliest episode in the rollout.
    """
    spec_by_id = {spec.rubric_id: spec for spec in (specs or [])}
    per_rubric: dict[str, list[tuple[int, RubricState]]] = {}
    observations: dict[str, dict[int, RubricObservation]] = {}
    for vector in sorted(vectors, key=lambda v: v.step):
        for rubric_id, state in vector.rubrics.items():
            per_rubric.setdefault(rubric_id, []).append((vector.step, state))
            revision, expected, actual = vector.observation(rubric_id)
            if revision is not None or expected is not None or actual is not None:
                observations.setdefault(rubric_id, {})[vector.step] = RubricObservation(
                    target_revision=revision,
                    expected_value=expected,
                    actual_value=actual,
                )

    episodes: list[EpisodeInterval] = []
    for rubric_id, states in per_rubric.items():
        spec = spec_by_id.get(rubric_id)
        if hard_only and spec is not None and not spec.hard:
            continue
        initial = spec.initial_state if spec is not None else None
        episodes.extend(
            extract_episode_intervals(
                rubric_id, states, initial_state=initial,
                observations=observations.get(rubric_id),
            )
        )

    episodes.sort(key=lambda e: (e.open_step, e.rubric_id, e.reopen_index))
    return episodes


def first_violation_step(episodes: Sequence[EpisodeInterval]) -> int | None:
    """Earliest ``open_step`` across episodes, i.e. ``t*`` in plan section 10."""
    if not episodes:
        return None
    return min(episode.open_step for episode in episodes)
