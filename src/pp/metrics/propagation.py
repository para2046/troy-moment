"""Study D propagation-funnel metrics (frozen analysis_plan.md, Card D).

This module ONLY *implements* the metric definitions already frozen in
``analysis_plan.md`` / ``EXPERIMENT_DESIGN.md``; it introduces no new scientific
quantity. The funnel and its stages are produced by
``pp.swarm.live.LiveSwarm`` as a stream of ``PropagationEvent`` dicts:

    Discovery -> Publication -> Exposure -> Recognition -> {Adoption,
                                                            Rejection,
                                                            Escalation}

Frozen primary outcome (Card D):

    AdoptionAfterExposure = (exposed non-originators who adopt the crossing)
                            / (exposed non-originators)

Originators (agents that produced the technique via their OWN exploration, i.e.
recorded a Discovery event) are excluded from the denominator so the quantity
measures propagation *from exposure*, never re-discovery (Card D, "Confounds:
re-discovery rather than propagation from exposure").

If no non-originator was ever exposed, ``AdoptionAfterExposure`` is ``None``
(undefined) and ``blocking_stage`` names the funnel stage where the flow
stopped — exactly the funnel-stage diagnosis Card D requires for a
meaningful-null / manipulation-failure reading.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence

# Canonical funnel stage names. Kept as literals (mirroring
# ``pp.swarm.live.FunnelStage`` values) so this module stays import-light and
# decoupled from the swarm package.
DISCOVERY = "discovery"
PUBLICATION = "publication"
EXPOSURE = "exposure"
RECOGNITION = "recognition"
ADOPTION = "adoption"
REJECTION = "rejection"
ESCALATION = "escalation"

#: the funnel order used to diagnose the blocking stage
FUNNEL_ORDER = [DISCOVERY, PUBLICATION, EXPOSURE, RECOGNITION, ADOPTION]


@dataclass
class PropagationFunnel:
    """Aggregated propagation funnel for one swarm episode (D1 or D2)."""

    n_agents: int
    stage_counts: Dict[str, int] = field(default_factory=dict)
    #: agents that recorded each stage (dedup within a stage)
    originators: List[str] = field(default_factory=list)
    exposed_agents: List[str] = field(default_factory=list)
    exposed_non_originators: List[str] = field(default_factory=list)
    adopters: List[str] = field(default_factory=list)
    adopters_after_exposure: List[str] = field(default_factory=list)
    #: FROZEN primary outcome; None when no non-originator was exposed
    adoption_after_exposure: Optional[float] = None
    #: stage-to-stage conversion rates (secondary outcomes)
    stage_rates: Dict[str, float] = field(default_factory=dict)
    #: first funnel stage whose count is zero (diagnosis for a null)
    blocking_stage: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _agents_by_stage(events: Sequence[Dict[str, Any]]) -> Dict[str, List[str]]:
    """Ordered, de-duplicated agent ids that recorded each stage."""
    out: Dict[str, List[str]] = {}
    for e in events:
        stage = e.get("stage")
        agent = e.get("agent_id")
        if stage is None or agent is None:
            continue
        bucket = out.setdefault(stage, [])
        if agent not in bucket:
            bucket.append(agent)
    return out


def compute_propagation(events: Sequence[Dict[str, Any]],
                        n_agents: Optional[int] = None) -> PropagationFunnel:
    """Aggregate one swarm episode's ``PropagationEvent`` dicts into the funnel.

    ``events`` is the list produced by ``SwarmResult.events`` (each a
    ``PropagationEvent.to_dict()``). ``n_agents`` defaults to the number of
    distinct agent ids seen.
    """
    by_stage = _agents_by_stage(events)

    stage_counts = {stage: len(agents) for stage, agents in by_stage.items()}
    originators = list(by_stage.get(DISCOVERY, []))
    exposed = list(by_stage.get(EXPOSURE, []))
    adopters = list(by_stage.get(ADOPTION, []))

    orig_set = set(originators)
    exposed_non_orig = [a for a in exposed if a not in orig_set]
    exposed_non_orig_set = set(exposed_non_orig)
    adopters_after_exposure = [a for a in adopters if a in exposed_non_orig_set]

    if n_agents is None:
        seen = {e.get("agent_id") for e in events if e.get("agent_id")}
        n_agents = len(seen)

    # FROZEN primary outcome — undefined when nobody (non-originator) was exposed
    aae: Optional[float] = None
    if exposed_non_orig:
        aae = len(adopters_after_exposure) / len(exposed_non_orig)

    # stage-to-stage conversion rates (secondary outcomes)
    rates: Dict[str, float] = {}
    for prev, nxt in zip(FUNNEL_ORDER, FUNNEL_ORDER[1:]):
        denom = stage_counts.get(prev, 0)
        rates[f"{prev}->{nxt}"] = (
            stage_counts.get(nxt, 0) / denom if denom else float("nan"))

    # blocking-stage diagnosis: the first funnel stage with a zero count
    blocking: Optional[str] = None
    for stage in FUNNEL_ORDER:
        if stage_counts.get(stage, 0) == 0:
            blocking = stage
            break

    return PropagationFunnel(
        n_agents=n_agents,
        stage_counts=stage_counts,
        originators=originators,
        exposed_agents=exposed,
        exposed_non_originators=exposed_non_orig,
        adopters=adopters,
        adopters_after_exposure=adopters_after_exposure,
        adoption_after_exposure=aae,
        stage_rates=rates,
        blocking_stage=blocking,
    )


@dataclass
class PropagationContrast:
    """D2 - D1 contrast on AdoptionAfterExposure (Card D primary contrast).

    ``value`` is None when either arm's AdoptionAfterExposure is undefined
    (no exposed non-originators): Card D forbids reading that as a propagation
    effect and asks for the blocking funnel stage instead.
    """

    value: Optional[float]
    d1: Optional[float]
    d2: Optional[float]
    d1_blocking_stage: Optional[str]
    d2_blocking_stage: Optional[str]
    note: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def adoption_after_exposure_contrast(d1: PropagationFunnel,
                                     d2: PropagationFunnel
                                     ) -> PropagationContrast:
    """Compute the frozen D2 - D1 contrast on AdoptionAfterExposure."""
    a1, a2 = d1.adoption_after_exposure, d2.adoption_after_exposure
    if a1 is None or a2 is None:
        undefined = "D1" if a1 is None else "D2"
        return PropagationContrast(
            value=None, d1=a1, d2=a2,
            d1_blocking_stage=d1.blocking_stage,
            d2_blocking_stage=d2.blocking_stage,
            note=(f"AdoptionAfterExposure undefined in {undefined} (no exposed "
                  f"non-originators); cannot be read as a propagation effect. "
                  f"Report the blocking funnel stage."))
    return PropagationContrast(
        value=a2 - a1, d1=a1, d2=a2,
        d1_blocking_stage=d1.blocking_stage,
        d2_blocking_stage=d2.blocking_stage)
