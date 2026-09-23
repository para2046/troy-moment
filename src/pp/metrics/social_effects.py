"""Study B causal quantities and statistics (spec sections 6, 26).

Every effect is a CONTRAST BETWEEN CONDITIONS, reported with a bootstrap CI
and an explicit note when the comparison is underpowered.

An underpowered null is never reported as "no effect" (spec section 26).
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, asdict, field
from typing import Any, Callable, Dict, List, Optional, Sequence

from ..treatments.conditions import Treatment


# --------------------------------------------------------------------------
@dataclass
class Estimate:
    name: str
    value: float
    ci_low: float
    ci_high: float
    n_a: int
    n_b: int
    outcome: str
    underpowered: bool = False
    note: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def __str__(self) -> str:
        flag = "  [UNDERPOWERED]" if self.underpowered else ""
        return (f"{self.name} ({self.outcome}): {self.value:+.4f} "
                f"[{self.ci_low:+.4f}, {self.ci_high:+.4f}] "
                f"n={self.n_a}/{self.n_b}{flag}")


def bootstrap_diff(a: Sequence[float], b: Sequence[float],
                   n_boot: int = 10000, alpha: float = 0.05,
                   seed: int = 0) -> tuple[float, float, float]:
    """Bootstrap CI for mean(a) - mean(b). Works for 0/1 and graded outcomes."""
    if not a or not b:
        return (float("nan"), float("nan"), float("nan"))
    rng = random.Random(seed)
    point = (sum(a) / len(a)) - (sum(b) / len(b))
    diffs = []
    na, nb = len(a), len(b)
    for _ in range(n_boot):
        sa = sum(a[rng.randrange(na)] for _ in range(na)) / na
        sb = sum(b[rng.randrange(nb)] for _ in range(nb)) / nb
        diffs.append(sa - sb)
    diffs.sort()
    lo = diffs[int((alpha / 2) * n_boot)]
    hi = diffs[min(n_boot - 1, int((1 - alpha / 2) * n_boot))]
    return point, lo, hi


def mde_two_proportion(p_base: float, n: int, alpha: float = 0.05,
                       power: float = 0.80) -> float:
    """Minimum detectable effect (percentage points) at given n.

    Normal approximation on the arcsine (Cohen's h) scale, matching the
    project's existing power_analysis.py.
    """
    z_a = 1.959963984540054 if abs(alpha - 0.05) < 1e-9 else _z(1 - alpha / 2)
    z_b = 0.8416212335729143 if abs(power - 0.80) < 1e-9 else _z(power)
    h_needed = (z_a + z_b) * math.sqrt(2.0 / max(n, 1))
    phi_base = 2 * math.asin(math.sqrt(min(max(p_base, 0.0), 1.0)))
    phi_target = phi_base + h_needed
    if phi_target >= math.pi:
        return 1.0 - p_base
    p_target = math.sin(phi_target / 2) ** 2
    return p_target - p_base


def _z(p: float) -> float:
    """Inverse normal CDF (Acklam's rational approximation)."""
    if not 0 < p < 1:
        raise ValueError("p must be in (0,1)")
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    plow, phigh = 0.02425, 1 - 0.02425
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
               ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    if p > phigh:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
                ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    q = p - 0.5
    r = q * q
    return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / \
           (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)


# --------------------------------------------------------------------------
#: The four named causal quantities, FROZEN in analysis_plan.md (Card B). These
#: and ONLY these four are computed, each mapped to the canonical, frozen
#: ``SPRINT_B_CONDITIONS`` (B1_information_only, B2_observed_impunity,
#: B3_peer_punished, B4_neutral_unpunished, B5_vertical_authorization).
#:
#: Each entry is (a, b) so the reported effect is mean(a) - mean(b), matching the
#: "Primary contrast" line of the corresponding Experiment Card:
#:   * Precedent Effect            B2 - B1              (Card B, §"Precedent")
#:   * Enforcement / Impunity      B2 - B3              (Card B, §"Enforcement")
#:   * Social Provenance           B2 - B4              (Card B, §"Provenance")
#:   * Authorization Discrimination  Execute(B5) - Execute(B2)  (Card B, §"Auth.")
#:
#: The B2-B1 contrast retains the historical key ``observed_impunity_effect``:
#: adding the observed-impunity precedent (B2) relative to information-only (B1)
#: is exactly the frozen Precedent contrast.
CONTRASTS: Dict[str, tuple[Treatment, Treatment]] = {
    # Precedent Effect: B2 - B1
    "observed_impunity_effect": (Treatment.B2_OBSERVED_IMPUNITY,
                                 Treatment.B1_INFORMATION_ONLY),
    # Enforcement / Impunity Effect: B2 - B3 (peer precedent, unpunished vs punished)
    "enforcement_impunity_effect": (Treatment.B2_OBSERVED_IMPUNITY,
                                    Treatment.B3_PEER_PUNISHED),
    # Social Provenance Effect: B2 - B4 (same unpunished event; peer vs neutral record)
    "social_provenance_effect": (Treatment.B2_OBSERVED_IMPUNITY,
                                 Treatment.B4_NEUTRAL_UNPUNISHED),
    # Authorization Discrimination: Execute(B5) - Execute(B2)
    "authorization_discrimination": (Treatment.B5_VERTICAL_AUTHORIZATION,
                                     Treatment.B2_OBSERVED_IMPUNITY),
}


#: Sprint-distinctive canonical conditions. Their presence in ``by_treatment``
#: signals a real (frozen) sprint dataset, so a missing frozen contrast must
#: fail loudly rather than be silently skipped. B1/B2 are shared with the pilot
#: cells and so are NOT part of this signal.
_SPRINT_SIGNAL = frozenset({
    Treatment.B3_PEER_PUNISHED.value,
    Treatment.B4_NEUTRAL_UNPUNISHED.value,
    Treatment.B5_VERTICAL_AUTHORIZATION.value,
})


def _required_conditions() -> List[str]:
    """Every condition referenced by the frozen CONTRASTS, order-preserved."""
    seen: Dict[str, None] = {}
    for a, b in CONTRASTS.values():
        seen.setdefault(a.value, None)
        seen.setdefault(b.value, None)
    return list(seen)


def compute_effects(by_treatment: Dict[str, List[Dict[str, Any]]],
                    outcomes: Optional[List[str]] = None,
                    n_boot: int = 10000, seed: int = 0,
                    power_reference: float = 0.80,
                    strict: Optional[bool] = None) -> List[Estimate]:
    """Compute the four frozen contrasts across multiple outcomes.

    ``by_treatment`` maps treatment value -> list of per-run metric dicts.
    ``outcomes`` are dotted keys, e.g. "boundary.crossing".

    Fail-loud contract (correctness-critical, spec / analysis_plan.md Card B):
    the four FROZEN contrasts (B2-B1, B2-B3, B2-B4, Execute(B5)-Execute(B2)) are
    computed against the canonical ``SPRINT_B_CONDITIONS``. A condition required
    by a frozen contrast must NEVER be silently skipped when the frozen analysis
    is being run:

    * ``strict=True``   -> always require every condition referenced by
      ``CONTRASTS``; raise ``ValueError`` if any is absent (or has no rows).
    * ``strict=False``  -> legacy permissive behaviour: skip a contrast whose
      condition is absent. Intended ONLY for isolated unit fixtures.
    * ``strict=None`` (default) -> auto: enforce the full frozen set as soon as
      the data looks like a real sprint run (any sprint-distinctive condition -
      B3_peer_punished / B4_neutral_unpunished / B5_vertical_authorization - is
      present). This makes frozen-analysis data fail loud on a missing condition
      while leaving small pilot fixtures (B1/B2 only) computable.
    """
    present = {k for k, v in by_treatment.items() if v}
    enforce = strict if strict is not None else bool(present & _SPRINT_SIGNAL)
    if enforce:
        required = _required_conditions()
        missing = [t for t in required if t not in present]
        if missing:
            raise ValueError(
                "compute_effects: condition(s) required by the frozen "
                f"analysis_plan.md contrasts are absent: {missing}. The frozen "
                "contrasts (B2-B1, B2-B3, B2-B4, Execute(B5)-Execute(B2)) "
                f"require all of {required}; a frozen contrast must never be "
                "silently skipped. (Pass strict=False only for isolated unit "
                "fixtures.)")

    outcomes = outcomes or [
        "boundary.crossing",
        "boundary.proposal",
        "boundary.n_boundaries_violated",
        "persistence.unsafe_persistence",
        "persistence.safe_persistence",
        "persistence.premature_stop",
        "utility.task_success",
        "utility.safe_success",
        "escalation.attempted",
    ]

    def pull(rows: List[Dict[str, Any]], key: str) -> List[float]:
        out = []
        for r in rows:
            cur: Any = r
            for part in key.split("."):
                if not isinstance(cur, dict):
                    cur = None
                    break
                cur = cur.get(part)
            if cur is None:
                continue
            out.append(float(cur) if not isinstance(cur, bool) else float(cur))
        return out

    results: List[Estimate] = []
    for name, (ta, tb) in CONTRASTS.items():
        rows_a = by_treatment.get(ta.value, [])
        rows_b = by_treatment.get(tb.value, [])
        if not rows_a or not rows_b:
            continue
        for outcome in outcomes:
            a, b = pull(rows_a, outcome), pull(rows_b, outcome)
            if not a or not b:
                continue
            point, lo, hi = bootstrap_diff(a, b, n_boot=n_boot, seed=seed)
            n = min(len(a), len(b))
            note = ""
            underpowered = False
            # only a meaningful notion for 0/1 outcomes
            if set(map(float, a + b)) <= {0.0, 1.0}:
                base = sum(b) / len(b)
                mde = mde_two_proportion(base, n, power=power_reference)
                if abs(point) < abs(mde):
                    underpowered = True
                    note = (f"|effect| {abs(point):.3f} < MDE {abs(mde):.3f} "
                            f"at n={n}; a null here is NOT evidence of no effect")
            results.append(Estimate(
                name=name, value=point, ci_low=lo, ci_high=hi,
                n_a=len(a), n_b=len(b), outcome=outcome,
                underpowered=underpowered, note=note))
    return results
