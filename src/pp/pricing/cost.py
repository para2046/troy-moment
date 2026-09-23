"""Cost tracking (spec section 23). Prices come from configs/pricing.yaml only."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from ..schemas import Trajectory, Usage

DEFAULT_PRICING_PATH = Path(__file__).resolve().parents[3] / "configs" / "pricing.yaml"


class BudgetExceeded(RuntimeError):
    pass


def load_pricing(path: Optional[str | Path] = None) -> Dict[str, Any]:
    import yaml
    p = Path(path) if path else DEFAULT_PRICING_PATH
    if not p.exists():
        return {"models": {}, "default_budget_usd": 5.0}
    with open(p, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def cost_of(usage: Usage, model_key: str, pricing: Dict[str, Any]) -> float:
    """Cost in USD. Unknown models cost 0 and are reported as unknown."""
    m = (pricing.get("models") or {}).get(model_key)
    if not m:
        return 0.0
    per_m = 1_000_000.0
    inp = usage.input_tokens - usage.cached_input_tokens
    total = (
        max(0, inp) / per_m * float(m.get("input_per_mtok", 0.0))
        + usage.cached_input_tokens / per_m * float(
            m.get("cached_input_per_mtok", m.get("input_per_mtok", 0.0)))
        + usage.output_tokens / per_m * float(m.get("output_per_mtok", 0.0))
        + usage.reasoning_tokens / per_m * float(
            m.get("reasoning_per_mtok", m.get("output_per_mtok", 0.0)))
    )
    return round(total, 6)


@dataclass
class CostReport:
    total_usd: float
    by_model: Dict[str, float]
    by_treatment: Dict[str, float]
    by_scenario: Dict[str, float]
    by_study: Dict[str, float]
    unknown_models: List[str]
    n_runs: int
    total_usage: Usage

    def projected(self, full_n: int) -> float:
        if self.n_runs == 0:
            return 0.0
        return round(self.total_usd / self.n_runs * full_n, 4)

    def render(self) -> str:
        lines = [f"runs: {self.n_runs}   total: ${self.total_usd:.4f}"]
        for label, d in (("model", self.by_model),
                         ("treatment", self.by_treatment),
                         ("study", self.by_study)):
            if d:
                lines.append(f"  by {label}:")
                for k, v in sorted(d.items(), key=lambda x: -x[1]):
                    lines.append(f"    {k:<36} ${v:.4f}")
        if self.unknown_models:
            lines.append(f"  NOTE: no pricing entry for "
                         f"{sorted(set(self.unknown_models))} (counted as $0)")
        return "\n".join(lines)


def summarize_costs(trajs: Iterable[Trajectory],
                    pricing: Optional[Dict[str, Any]] = None) -> CostReport:
    pricing = pricing if pricing is not None else load_pricing()
    by_model: Dict[str, float] = defaultdict(float)
    by_treat: Dict[str, float] = defaultdict(float)
    by_scen: Dict[str, float] = defaultdict(float)
    by_study: Dict[str, float] = defaultdict(float)
    unknown: List[str] = []
    total = 0.0
    n = 0
    usage_total = Usage()

    for t in trajs:
        n += 1
        key = t.model_id
        if key not in (pricing.get("models") or {}):
            unknown.append(key)
        c = t.cost_usd or cost_of(t.usage, key, pricing)
        total += c
        by_model[key] += c
        by_treat[t.treatment] += c
        by_scen[t.scenario_id] += c
        by_study[t.study_id] += c
        usage_total = usage_total + t.usage

    return CostReport(round(total, 6), dict(by_model), dict(by_treat),
                      dict(by_scen), dict(by_study), unknown, n, usage_total)


def check_budget(estimated_usd: float, threshold_usd: float,
                 approved: bool = False) -> None:
    """Stop before an expensive batch unless explicitly approved (spec §23)."""
    if estimated_usd <= threshold_usd:
        return
    if approved:
        return
    raise BudgetExceeded(
        f"estimated ${estimated_usd:.2f} exceeds threshold "
        f"${threshold_usd:.2f}. Re-run with --approve-budget to proceed.")
