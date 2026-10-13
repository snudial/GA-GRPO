"""External-cost accounting for recovered episodes (plan section 4.3).

The distinction this module enforces is:

* **external cost** -- a real state-changing side effect happened (a payment, an
  order, a message, a booking), or an observable footprint survived the undo
  (cancellation fee, refund delay, duplicate notification), or an irreversible
  mutation occurred that the evaluator does not allow.
* **process overhead** -- extra model tokens, extra read-only API calls, retries,
  extra turns. Recorded, never counted as external cost. Folding these in would
  make nearly every recovery "costly" and destroy the contrast.

When the environment gives no evidence either way the verdict is
``CostVerdict.UNKNOWN``. It is never silently coerced to ``NONE``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Sequence

from bfcl_env.schemas import CostEvidence, CostVerdict

__all__ = ["MutationEvent", "CostLedger"]


@dataclass
class MutationEvent:
    """One environment-changing (or explicitly read-only) call made by the policy.

    Adapters emit these; the ledger never inspects tool names itself.

    Attributes:
        step: rollout step the call was made in.
        tool: tool / API name, for the audit trail only.
        mutating: ``True`` if the call changed persistent state, ``False`` if it is
            read-only, ``None`` if the environment cannot say. ``None`` forces
            UNKNOWN -- an unclassifiable call is not evidence of innocence.
        reversible: whether the mutation was fully undone by a later call.
        footprint: whether an observable residue survived the undo (fee, delay,
            duplicate notification, audit-log entry the evaluator counts).
        target: the object touched, e.g. ``"orders:order_123"``.
        detail: free-form note copied into ``CostEvidence.reasons``.
    """

    step: int
    tool: str
    mutating: bool | None
    reversible: bool | None = None
    footprint: bool | None = None
    target: str = ""
    detail: str = ""

    @property
    def is_read_only(self) -> bool:
        return self.mutating is False

    def reason(self) -> str:
        if self.detail:
            return self.detail
        label = self.target or self.tool
        return f"{self.tool}:{label}@step{self.step}"


@dataclass
class CostLedger:
    """Decides the external-cost verdict for one episode window.

    Args:
        default_when_unspecified: verdict used when a mutating call happened inside
            the episode window but the adapter did not report whether it was
            reversible or left a footprint.

            * AppWorld passes ``CostVerdict.EXTERNAL``: a mutating API against the
              app database is by construction a real side effect.
            * BFCL passes ``CostVerdict.UNKNOWN``: the simulators do not model
              reversibility or persistence, so the honest answer is "no evidence".
        token_counts_by_step: optional ``step -> action token count`` used to fill
            in ``process_overhead_tokens``.
    """

    default_when_unspecified: CostVerdict = CostVerdict.UNKNOWN
    token_counts_by_step: dict[int, int] = field(default_factory=dict)

    def events_in_window(
        self,
        events: Iterable[MutationEvent],
        open_step: int,
        close_step: int | None,
        *,
        last_step: int | None = None,
    ) -> list[MutationEvent]:
        """Events attributable to an episode.

        The window is ``[open_step, close_step]`` inclusive: the opening action is
        usually the harmful mutation itself, and the closing action may be the undo
        that leaves the footprint. An unresolved episode runs to ``last_step``.
        """
        end = close_step if close_step is not None else last_step
        selected = []
        for event in events:
            if event.step < open_step:
                continue
            if end is not None and event.step > end:
                continue
            selected.append(event)
        return selected

    def assess(
        self,
        events: Sequence[MutationEvent],
        open_step: int,
        close_step: int | None,
        *,
        last_step: int | None = None,
    ) -> CostEvidence:
        """Produce the ``CostEvidence`` for one episode window."""
        window = self.events_in_window(events, open_step, close_step, last_step=last_step)

        read_only = [e for e in window if e.is_read_only]
        unclassified = [e for e in window if e.mutating is None]
        mutating = [e for e in window if e.mutating is True]

        overhead_steps = len({e.step for e in window})
        evidence = CostEvidence(
            verdict=CostVerdict.UNKNOWN,
            reasons=[],
            process_overhead_steps=overhead_steps,
            process_overhead_calls=len(read_only),
            process_overhead_tokens=sum(
                self.token_counts_by_step.get(step, 0) for step in {e.step for e in window}
            ),
        )

        if unclassified:
            evidence.verdict = CostVerdict.UNKNOWN
            evidence.reasons = [f"unclassified_call:{e.reason()}" for e in unclassified[:8]]
            return evidence

        if not mutating:
            # Nothing persistent happened between open and close: the model noticed
            # and corrected using read-only calls / retries only.
            evidence.verdict = CostVerdict.NONE
            if read_only:
                evidence.reasons = ["read_only_recovery"]
            return evidence

        costly = [
            e for e in mutating if e.reversible is False or e.footprint is True
        ]
        if costly:
            evidence.verdict = CostVerdict.EXTERNAL
            evidence.reasons = [f"irreversible_or_footprint:{e.reason()}" for e in costly[:8]]
            return evidence

        clean = [e for e in mutating if e.reversible is True and e.footprint is False]
        if len(clean) == len(mutating):
            evidence.verdict = CostVerdict.NONE
            evidence.reasons = ["fully_reverted_without_footprint"]
            return evidence

        # A mutation happened but reversibility/footprint were not reported.
        evidence.verdict = self.default_when_unspecified
        evidence.reasons = [f"unspecified_mutation:{e.reason()}" for e in mutating[:8]]
        return evidence
