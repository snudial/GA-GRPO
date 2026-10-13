"""Join rollouts, rubric state vectors and mutation logs into episodes + outcomes.

This is the only place where the three-way rollout classification of plan
section 3.1 is decided, and the only place that assigns ``t*``
(``first_violation_step``). Everything downstream -- statistics, the three loss
masks, the HTML overlays -- reads the records produced here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

from bfcl_env.cost_evidence import evaluate_cost_vector
from bfcl_env.cost_ledger import CostLedger, MutationEvent
from bfcl_env.rubric_state_machine import (
    INITIAL_STEP,
    EpisodeInterval,
    extract_all_episode_intervals,
)
from bfcl_env.schemas import (
    ClassificationStatus,
    CoverageClass,
    Episode,
    EpisodeCloseReason,
    FailureKind,
    RolloutOutcome,
    RolloutRecord,
    RubricKind,
    RubricSpec,
    RubricState,
    RubricStateVector,
)

__all__ = ["ClassifiedRollout", "classify_rollout", "terminal_vector"]


@dataclass
class ClassifiedRollout:
    """Everything derived from one rollout."""

    outcome: RolloutOutcome
    episodes: list[Episode] = field(default_factory=list)

    @property
    def first_violation_step(self) -> int | None:
        return self.outcome.first_violation_step


def terminal_vector(vectors: Sequence[RubricStateVector]) -> RubricStateVector | None:
    """Last non-initial state vector, i.e. the state the rollout ended in."""
    candidates = [v for v in vectors if v.step != INITIAL_STEP]
    if not candidates:
        return None
    return max(candidates, key=lambda v: v.step)


def _turn_of_step(rollout: RolloutRecord, step: int) -> int | None:
    record = rollout.step_by_index(step)
    return None if record is None else record.turn


def _span_of_step(rollout: RolloutRecord, step: int) -> tuple[int, int] | None:
    record = rollout.step_by_index(step)
    return None if record is None else record.assistant_token_span


def _max_concurrent_active(intervals: Sequence[EpisodeInterval], last_step: int) -> int:
    """Largest number of distinct rubrics violated at the same time."""
    if not intervals:
        return 0
    best = 0
    steps = sorted({i.open_step for i in intervals} | {last_step})
    for step in steps:
        active = 0
        for interval in intervals:
            if interval.open_step > step:
                continue
            if interval.close_step is not None and step >= interval.close_step:
                continue
            active += 1
        best = max(best, active)
    return best


def classify_rollout(
    rollout: RolloutRecord,
    specs: Sequence[RubricSpec],
    vectors: Sequence[RubricStateVector],
    *,
    mutations: Sequence[MutationEvent] = (),
    cost_ledger: CostLedger | None = None,
    coverage_warning: bool = False,
    uncovered_mutations: Sequence[str] = (),
    unscoreable: bool = False,
) -> ClassifiedRollout:
    """Classify one rollout end to end.

    Only ``hard`` rubrics take part in episode extraction and in the success
    decision; soft rubrics are carried in the specs for reporting but never open an
    episode (plan section 3.1).
    """
    hard_specs = [spec for spec in specs if spec.hard]
    hard_ids = {spec.rubric_id for spec in hard_specs}
    spec_by_id = {spec.rubric_id: spec for spec in specs}

    ledger = cost_ledger or CostLedger()
    ledger.token_counts_by_step.update(
        {s.step: s.num_action_tokens for s in rollout.steps}
    )

    intervals = extract_all_episode_intervals(vectors, hard_specs, hard_only=True)
    intervals = [i for i in intervals if i.rubric_id in hard_ids]

    last_step = max((s.step for s in rollout.steps), default=INITIAL_STEP)

    episodes: list[Episode] = []
    seen_per_rubric: dict[str, int] = {}
    for interval in intervals:
        index = seen_per_rubric.get(interval.rubric_id, 0)
        seen_per_rubric[interval.rubric_id] = index + 1
        spec = spec_by_id.get(interval.rubric_id)
        cost = ledger.assess(
            mutations,
            interval.open_step,
            interval.close_step,
            last_step=last_step,
        )
        window = ledger.events_in_window(
            mutations, interval.open_step, interval.close_step, last_step=last_step
        )
        cost_vector = evaluate_cost_vector(
            rollout.benchmark,
            window,
            process_overhead_turns=len(
                {
                    s.turn for s in rollout.steps
                    if interval.open_step <= s.step
                    <= (interval.close_step if interval.close_step is not None else last_step)
                }
            ),
            process_overhead_tokens=cost.process_overhead_tokens,
        )
        episodes.append(
            Episode(
                rollout_id=rollout.rollout_id,
                episode_id=f"{rollout.rollout_id}:{interval.rubric_id}:{interval.reopen_index}",
                rubric_id=interval.rubric_id,
                open_step=interval.open_step,
                open_turn=_turn_of_step(rollout, interval.open_step),
                open_action_token_span=_span_of_step(rollout, interval.open_step),
                close_step=interval.close_step,
                close_turn=(
                    None if interval.close_step is None
                    else _turn_of_step(rollout, interval.close_step)
                ),
                close_action_token_span=(
                    None if interval.close_step is None
                    else _span_of_step(rollout, interval.close_step)
                ),
                manifestation_steps=list(interval.manifestation_steps),
                reopen_index=interval.reopen_index,
                close_reason=interval.close_reason,
                open_target_revision=interval.open_observation.target_revision,
                open_expected_value=interval.open_observation.expected_value,
                open_actual_value=interval.open_observation.actual_value,
                close_target_revision=interval.close_observation.target_revision,
                close_actual_value=interval.close_observation.actual_value,
                cost=cost,
                rubric_kind=spec.kind if spec is not None else RubricKind.SAFETY,
                hard=True,
                cost_vector=cost_vector,
                coverage_class=(
                    CoverageClass.UNCOVERED if coverage_warning else CoverageClass.DECIDABLE
                ),
                unscoreable=unscoreable,
                open_action_ref=f"actions/{interval.open_step}",
                close_action_ref=(
                    None if interval.close_step is None
                    else f"actions/{interval.close_step}"
                ),
                evidence_refs=list(cost_vector.evidence_refs),
            )
        )

    # ---- terminal decision -------------------------------------------------- #
    terminal = terminal_vector(vectors)
    unknown_rubrics: list[str] = []
    terminal_success = False
    if terminal is not None:
        terminal_states = {
            rid: state for rid, state in terminal.rubrics.items() if rid in hard_ids
        }
        unknown_rubrics = sorted(
            rid for rid, state in terminal_states.items() if state is RubricState.UNKNOWN
        )
        terminal_success = bool(terminal_states) and all(
            state is RubricState.SATISFIED for state in terminal_states.values()
        )

    aborted = rollout.aborted or terminal is None or bool(unknown_rubrics)

    if aborted:
        failure_kind = FailureKind.ABORTED
    elif terminal_success:
        failure_kind = FailureKind.NONE
    elif episodes:
        failure_kind = FailureKind.LOCALIZED_VIOLATION
    else:
        # Nothing ever became VIOLATED, yet the task failed: pure omission. No
        # action can be blamed, so t* stays None (plan section 4.2).
        failure_kind = FailureKind.UNLOCALIZED_OMISSION

    first_violation = min((e.open_step for e in episodes), default=None)
    first_violation_rubric = None
    if first_violation is not None:
        opening = sorted(
            (e for e in episodes if e.open_step == first_violation),
            key=lambda e: e.rubric_id,
        )
        first_violation_rubric = opening[0].rubric_id

    num_reopens = sum(max(0, count - 1) for count in seen_per_rubric.values())

    outcome = RolloutOutcome(
        rollout_id=rollout.rollout_id,
        task_id=rollout.task_id,
        benchmark=rollout.benchmark,
        terminal_success=terminal_success and not aborted,
        any_hard_violation=bool(episodes),
        aborted=aborted,
        failure_kind=failure_kind,
        first_violation_step=first_violation,
        first_violation_turn=(
            None if first_violation is None else _turn_of_step(rollout, first_violation)
        ),
        first_violation_rubric_id=first_violation_rubric,
        num_episodes=len(episodes),
        num_recovered_episodes=sum(1 for e in episodes if e.recovered_by_action),
        num_labeled_episodes=sum(
            1 for e in episodes if e.classification_status.is_labeled
        ),
        num_abstained_episodes=sum(
            1 for e in episodes if e.classification_status.is_abstention
        ),
        num_superseded_episodes=sum(
            1 for e in episodes if e.close_reason is EpisodeCloseReason.SUPERSEDED_BY_TARGET
        ),
        num_unattributed_episodes=sum(
            1 for e in episodes
            if e.resolved and e.close_reason is EpisodeCloseReason.UNKNOWN
        ),
        num_unresolved_episodes=sum(1 for e in episodes if not e.resolved),
        num_reopens=num_reopens,
        num_manifestations=sum(e.num_manifestations for e in episodes),
        max_concurrent_active=_max_concurrent_active(intervals, last_step),
        num_action_tokens=sum(s.num_action_tokens for s in rollout.steps),
        num_steps=len(rollout.steps),
        num_turns=len({s.turn for s in rollout.steps}),
        coverage_warning=coverage_warning,
        uncovered_mutations=list(uncovered_mutations),
        unknown_rubrics=unknown_rubrics,
    )

    return ClassifiedRollout(outcome=outcome, episodes=episodes)
