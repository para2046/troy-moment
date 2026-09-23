"""Estimate cost of a planned run before launching (spec section 23)."""
import _bootstrap  # noqa: F401
import argparse
from pp.pricing import load_pricing, check_budget, BudgetExceeded

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="pricing key")
    ap.add_argument("--n-cells", type=int, required=True)
    ap.add_argument("--n-per-cell", type=int, required=True)
    ap.add_argument("--turns", type=int, default=10)
    ap.add_argument("--in-tok-per-turn", type=int, default=1500)
    ap.add_argument("--out-tok-per-turn", type=int, default=200)
    ap.add_argument("--threshold", type=float, default=None)
    ap.add_argument("--approve-budget", action="store_true")
    a = ap.parse_args()
    pricing = load_pricing()
    m = (pricing.get("models") or {}).get(a.model, {})
    runs = a.n_cells * a.n_per_cell
    in_tok = runs * a.turns * a.in_tok_per_turn
    out_tok = runs * a.turns * a.out_tok_per_turn
    cost = (in_tok / 1e6 * m.get("input_per_mtok", 0)
            + out_tok / 1e6 * m.get("output_per_mtok", 0))
    print(f"model {a.model}: {runs} runs x {a.turns} turns")
    print(f"  ~{in_tok:,} input + {out_tok:,} output tokens")
    print(f"  estimated cost: ${cost:.2f}")
    thr = a.threshold if a.threshold is not None else pricing.get(
        "default_budget_usd", 5.0)
    try:
        check_budget(cost, thr, approved=a.approve_budget)
        print(f"  within budget (${thr:.2f}) -- ok to launch")
    except BudgetExceeded as e:
        print(f"  {e}")
        raise SystemExit(1)
