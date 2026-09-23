"""Validate stimulus invariance (spec section 12). Exit non-zero on failure.

'Do not start real experiments until stimulus validation passes.'
"""
import _bootstrap  # noqa: F401
import argparse
import json
import sys
from pathlib import Path

from pp.treatments import StimulusSpec, validate_stimulus_set

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", default=str(
        ROOT / "data/generated_stimuli/stimuli.spec.json"))
    args = ap.parse_args()

    p = Path(args.spec)
    if not p.exists():
        print(f"spec not found: {p}\nrun build_stimuli.py first", file=sys.stderr)
        return 2
    d = json.loads(p.read_text(encoding="utf-8"))
    spec = StimulusSpec(
        stimulus_id=d["stimulus_id"],
        technical_block=d["technical_block"],
        impunity_block=d["impunity_block"],
        explicit_grant_block=d["explicit_grant_block"],
        vertical_block=d["vertical_block"],
        irrelevant_block=d["irrelevant_block"],
    )
    rep = validate_stimulus_set(spec)

    print(f"stimulus: {rep.stimulus_id}")
    print(f"technical-block hashes unique: "
          f"{len(set(rep.technical_block_hashes.values()))} "
          f"(want 1 across invariant group)")
    print("char counts:")
    for k, v in sorted(rep.char_counts.items()):
        print(f"    {k:<32} {v:>4}")
    print("key contrasts (similarity, added text):")
    for k in ("B1_information_only->B2_observed_impunity",
              "B3b_non_agent_source->B2_observed_impunity"):
        if k in rep.similarity:
            print(f"    {k}: sim={rep.similarity[k]}")
            print(f"       + {rep.diffs.get(k, '')[:120]}")
    if rep.warnings:
        print("WARNINGS:")
        for w in rep.warnings:
            print(f"    - {w}")
    if rep.ok:
        print("\nVALIDATION PASSED")
        return 0
    print("\nVALIDATION FAILED:")
    for fmsg in rep.failures:
        print(f"    - {fmsg}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
