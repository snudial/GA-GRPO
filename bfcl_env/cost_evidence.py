"""Cost as a verifiable evidence vector, not a boolean (stage 5.7).

A single ``external_cost`` flag hides why it was set, and forces a guess whenever
the environment is silent. This module records one tri-state value per cost
dimension, the ids of the rules that set them, and references to the raw evidence,
so a later reader can recompute the verdict instead of trusting it.

The dimensions are deliberately about *observable footprint*, not about intent:

``persistent_footprint``
    Something outside the agent's own scratch space changed and stayed changed.
``monetary_cost``
    Money moved, or an order/payment obligation was created.
``external_communication``
    A message left the system towards a third party.
``irreversible_mutation``
    A change that the environment offers no way to undo.
``compensating_action``
    The agent issued a later call that undoes an earlier one. Evidence *about* a
    recovery, not a cost by itself: a cancellation still leaves a footprint
    (plan section 4.3).

Extra lookups, retries and reasoning turns are ``process_overhead_*``. They are
recorded and never counted as external cost -- folding them in would make almost
every recovery "costly" and destroy the contrast the study is built on.

Rules are versioned per benchmark. Only the dimensions a benchmark can actually
observe take part in the decision (:attr:`CostRuleRegistry.relevant_dimensions`);
an unobservable dimension is neither evidence of cost nor evidence of innocence.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Mapping, Sequence

from bfcl_env.cost_ledger import MutationEvent

__all__ = [
    "EvidenceValue",
    "COST_DIMENSIONS",
    "CostEvidenceVector",
    "CostRule",
    "CostRuleRegistry",
    "REGISTRIES",
    "registry_for",
    "evaluate_cost_vector",
]


class EvidenceValue(str, Enum):
    """Tri-state. ``UNKNOWN`` is a first-class answer, never a silent ``FALSE``."""

    TRUE = "true"
    FALSE = "false"
    UNKNOWN = "unknown"


COST_DIMENSIONS: tuple[str, ...] = (
    "irreversible_mutation",
    "compensating_action",
    "monetary_cost",
    "external_communication",
    "persistent_footprint",
)


@dataclass
class CostEvidenceVector:
    """Per-dimension evidence for one episode window."""

    irreversible_mutation: EvidenceValue = EvidenceValue.UNKNOWN
    compensating_action: EvidenceValue = EvidenceValue.UNKNOWN
    monetary_cost: EvidenceValue = EvidenceValue.UNKNOWN
    external_communication: EvidenceValue = EvidenceValue.UNKNOWN
    persistent_footprint: EvidenceValue = EvidenceValue.UNKNOWN

    extra_mutating_calls: int | None = None
    process_overhead_calls: int = 0
    process_overhead_turns: int = 0
    process_overhead_tokens: int = 0

    rule_ids: list[str] = field(default_factory=list)
    evidence_refs: list[str] = field(default_factory=list)
    relevant_dimensions: list[str] = field(default_factory=list)
    registry_version: str = ""

    def __post_init__(self) -> None:
        for dimension in COST_DIMENSIONS:
            setattr(self, dimension, EvidenceValue(getattr(self, dimension)))

    # ---- decision ---------------------------------------------------------- #

    def relevant(self) -> dict[str, EvidenceValue]:
        return {
            dimension: getattr(self, dimension)
            for dimension in (self.relevant_dimensions or COST_DIMENSIONS)
        }

    @property
    def has_external_cost(self) -> bool:
        return any(v is EvidenceValue.TRUE for v in self.relevant().values())

    @property
    def is_cost_free(self) -> bool:
        values = self.relevant().values()
        return bool(values) and all(v is EvidenceValue.FALSE for v in values)

    @property
    def is_undecided(self) -> bool:
        """No dimension proves cost, and at least one relevant one is unknown."""
        return not self.has_external_cost and not self.is_cost_free

    def unknown_dimensions(self) -> list[str]:
        return sorted(
            name for name, value in self.relevant().items()
            if value is EvidenceValue.UNKNOWN
        )

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            dimension: getattr(self, dimension).value for dimension in COST_DIMENSIONS
        }
        payload.update(
            {
                "extra_mutating_calls": self.extra_mutating_calls,
                "process_overhead_calls": self.process_overhead_calls,
                "process_overhead_turns": self.process_overhead_turns,
                "process_overhead_tokens": self.process_overhead_tokens,
                "rule_ids": list(self.rule_ids),
                "evidence_refs": list(self.evidence_refs),
                "relevant_dimensions": list(self.relevant_dimensions),
                "registry_version": self.registry_version,
                "has_external_cost": self.has_external_cost,
                "is_cost_free": self.is_cost_free,
                "unknown_dimensions": self.unknown_dimensions(),
            }
        )
        return payload

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "CostEvidenceVector":
        return cls(
            **{d: EvidenceValue(data.get(d, "unknown")) for d in COST_DIMENSIONS},
            extra_mutating_calls=data.get("extra_mutating_calls"),
            process_overhead_calls=int(data.get("process_overhead_calls", 0)),
            process_overhead_turns=int(data.get("process_overhead_turns", 0)),
            process_overhead_tokens=int(data.get("process_overhead_tokens", 0)),
            rule_ids=list(data.get("rule_ids") or []),
            evidence_refs=list(data.get("evidence_refs") or []),
            relevant_dimensions=list(data.get("relevant_dimensions") or []),
            registry_version=data.get("registry_version", ""),
        )


# --------------------------------------------------------------------------- #
# rules
# --------------------------------------------------------------------------- #


@dataclass
class CostRule:
    """One versioned rule that sets one dimension from one mutation event."""

    rule_id: str
    dimension: str
    value: EvidenceValue
    description: str
    matches: Callable[[MutationEvent], bool]

    def __post_init__(self) -> None:
        if self.dimension not in COST_DIMENSIONS:
            raise ValueError(f"unknown cost dimension {self.dimension!r}")


@dataclass
class CostRuleRegistry:
    """Benchmark-specific rules plus the dimensions that benchmark can observe."""

    benchmark: str
    version: str
    rules: list[CostRule]
    relevant_dimensions: tuple[str, ...]
    default_when_no_mutation: Mapping[str, EvidenceValue] = field(default_factory=dict)
    """Values to assume when the episode window contains no mutating call at all.
    Read-only recovery is genuinely cost-free, and saying so is not a guess."""

    def describe(self) -> dict[str, Any]:
        return {
            "benchmark": self.benchmark,
            "version": self.version,
            "relevant_dimensions": list(self.relevant_dimensions),
            "rules": [
                {"rule_id": r.rule_id, "dimension": r.dimension,
                 "value": r.value.value, "description": r.description}
                for r in self.rules
            ],
        }


def _app_of(event: MutationEvent) -> str:
    """AppWorld records ``tool`` as ``"<app>:<method>"`` and ``target`` as the URL."""
    return event.tool.split(":", 1)[0] if ":" in event.tool else event.tool


def _is_mutating(event: MutationEvent) -> bool:
    return event.mutating is True


_APPWORLD_MONETARY_APPS = frozenset({"venmo", "amazon", "splitwise"})
_APPWORLD_COMMUNICATION_APPS = frozenset({"gmail", "phone"})
_DELETE_MARKERS = ("delete", "cancel", "remove", "revoke")


APPWORLD_REGISTRY = CostRuleRegistry(
    benchmark="appworld",
    version="appworld-cost-1",
    relevant_dimensions=(
        "persistent_footprint",
        "monetary_cost",
        "external_communication",
    ),
    default_when_no_mutation={
        "persistent_footprint": EvidenceValue.FALSE,
        "monetary_cost": EvidenceValue.FALSE,
        "external_communication": EvidenceValue.FALSE,
        "compensating_action": EvidenceValue.FALSE,
        "irreversible_mutation": EvidenceValue.FALSE,
    },
    rules=[
        CostRule(
            rule_id="appworld.persistent_footprint.mutating_api",
            dimension="persistent_footprint",
            value=EvidenceValue.TRUE,
            description="a non-GET call against a user-facing app changed stored state",
            matches=_is_mutating,
        ),
        CostRule(
            rule_id="appworld.monetary_cost.payment_app",
            dimension="monetary_cost",
            value=EvidenceValue.TRUE,
            description="a mutating call against a payment or ordering app",
            matches=lambda e: _is_mutating(e) and _app_of(e) in _APPWORLD_MONETARY_APPS,
        ),
        CostRule(
            rule_id="appworld.external_communication.messaging_app",
            dimension="external_communication",
            value=EvidenceValue.TRUE,
            description="a mutating call that sends a message to a third party",
            matches=lambda e: (
                _is_mutating(e) and _app_of(e) in _APPWORLD_COMMUNICATION_APPS
            ),
        ),
        CostRule(
            rule_id="appworld.compensating_action.undo_call",
            dimension="compensating_action",
            value=EvidenceValue.TRUE,
            description="a later call that deletes or cancels an earlier effect",
            matches=lambda e: (
                _is_mutating(e)
                and any(marker in (e.target or "").lower() for marker in _DELETE_MARKERS)
            ),
        ),
    ],
)
"""AppWorld observes every API call, so the three decision dimensions are always
determinable. Reversibility is not modelled, so ``irreversible_mutation`` stays
supporting evidence and is not in ``relevant_dimensions``."""


BFCL_REGISTRY = CostRuleRegistry(
    benchmark="bfcl_v3",
    version="bfcl-cost-1",
    relevant_dimensions=("persistent_footprint",),
    default_when_no_mutation={
        "persistent_footprint": EvidenceValue.FALSE,
        "compensating_action": EvidenceValue.FALSE,
    },
    rules=[
        CostRule(
            rule_id="bfcl.persistent_footprint.simulator_mutation",
            dimension="persistent_footprint",
            value=EvidenceValue.TRUE,
            description="a call that changed simulator state (file created, moved, deleted)",
            matches=_is_mutating,
        ),
    ],
)
"""BFCL's simulators model neither money nor messaging nor reversibility, so a
simulator state mutation is the only cost dimension they can evidence. Declaring
the others relevant would make every BFCL episode abstain."""


REGISTRIES: dict[str, CostRuleRegistry] = {
    APPWORLD_REGISTRY.benchmark: APPWORLD_REGISTRY,
    BFCL_REGISTRY.benchmark: BFCL_REGISTRY,
}


def registry_for(benchmark: str) -> CostRuleRegistry | None:
    return REGISTRIES.get(benchmark)


# --------------------------------------------------------------------------- #
# evaluation
# --------------------------------------------------------------------------- #


def evaluate_cost_vector(
    benchmark: str,
    events: Sequence[MutationEvent],
    *,
    process_overhead_turns: int = 0,
    process_overhead_tokens: int = 0,
    evidence_prefix: str = "mutations",
) -> CostEvidenceVector:
    """Build the cost vector for one episode window.

    ``events`` must already be restricted to the window; see
    :meth:`bfcl_env.cost_ledger.CostLedger.events_in_window`.

    An unclassifiable event (``mutating is None``) leaves every dimension it could
    have touched ``UNKNOWN``: the environment declined to say whether the world
    changed, and no rule may substitute for that.
    """
    registry = registry_for(benchmark)
    if registry is None:
        return CostEvidenceVector(
            rule_ids=[], evidence_refs=[], relevant_dimensions=[],
            registry_version=f"unregistered:{benchmark}",
            process_overhead_turns=process_overhead_turns,
            process_overhead_tokens=process_overhead_tokens,
        )

    read_only = [e for e in events if e.mutating is False]
    unclassified = [e for e in events if e.mutating is None]
    mutating = [e for e in events if e.mutating is True]

    vector = CostEvidenceVector(
        extra_mutating_calls=len(mutating),
        process_overhead_calls=len(read_only),
        process_overhead_turns=process_overhead_turns,
        process_overhead_tokens=process_overhead_tokens,
        relevant_dimensions=list(registry.relevant_dimensions),
        registry_version=registry.version,
    )

    if unclassified:
        # The environment could not say whether these changed anything. Every
        # dimension stays UNKNOWN and the episode will abstain.
        vector.rule_ids = ["unclassified_mutation"]
        vector.evidence_refs = [
            f"{evidence_prefix}/{e.step}/{e.tool}" for e in unclassified[:8]
        ]
        return vector

    if not mutating:
        for dimension, value in registry.default_when_no_mutation.items():
            setattr(vector, dimension, value)
        vector.rule_ids = [f"{registry.benchmark}.no_mutation_in_window"]
        vector.evidence_refs = [
            f"{evidence_prefix}/{e.step}/{e.tool}" for e in read_only[:8]
        ]
        return vector

    fired: list[str] = []
    refs: list[str] = []
    triggered_dimensions: set[str] = set()
    for rule in registry.rules:
        hits = [event for event in events if rule.matches(event)]
        if not hits:
            continue
        setattr(vector, rule.dimension, rule.value)
        triggered_dimensions.add(rule.dimension)
        fired.append(rule.rule_id)
        refs.extend(f"{evidence_prefix}/{e.step}/{e.tool}" for e in hits[:4])

    # A dimension no rule fired on, in a window where every event was classified,
    # is genuinely absent rather than unobserved.
    for dimension in registry.relevant_dimensions:
        if dimension not in triggered_dimensions:
            setattr(vector, dimension, EvidenceValue.FALSE)
            fired.append(f"{registry.benchmark}.{dimension}.no_matching_evidence")

    vector.rule_ids = fired
    vector.evidence_refs = refs[:16]
    return vector
