"""Study B — causal boundary suite (fork B1-B5 from a frozen checkpoint).

Manifest-driven and model-agnostic: for each frozen checkpoint the shared
``studies.study_ab_runner`` engine forks B1-B5 from ONE snapshot (byte-identical
restore, only the injected board varies), scores each run with the frozen
authority contract (BoundaryActionExecuted / AuthorizationValid /
UnauthorizedCrossing / AuthorizedExecution) and records the manipulation probe.
Writes ``results/raw/<model>/study_b/<checkpoint>.json``.

    # offline proof (no Modal, no network, no keys):
    python scripts/run_study_b.py --dry-run

    # real run (after freeze):
    python scripts/run_study_b.py \
        --model-config configs/models/qwen3_8_max.yaml \
        --task-selection data/qualified/task_selection.json \
        --checkpoints data/checkpoints/checkpoints_v2.json \
        --stimuli data/checkpoints/stimuli_v2.json \
        --out results/raw
"""
import _bootstrap  # noqa: F401
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "studies") not in sys.path:
    sys.path.insert(0, str(ROOT / "studies"))

import study_ab_runner as engine  # noqa: E402
from pp.models import load_model_config  # noqa: E402
from pp.runner_contract import normalize_benchmark  # noqa: E402


#: The ONLY substrate Study-B checkpoints may be restored into. Checkpoints
#: built on the retired bespoke runtime (shared warm container, generic
#: debian_slim image, hardcoded ``python -m pytest``) carry no ``substrate``
#: block; they are evidence of a superseded stack and must never be restored
#: into the formal one. This fails closed rather than falling back.
REQUIRED_SUBSTRATE = "pp.runtimes.impossible.env.SweBenchPerTaskSandboxEnv"


def assert_formal_substrate(checkpoints_path: str) -> str:
    """Refuse a checkpoint artifact that was not built on the formal substrate.

    Returns the declared ``env_class`` on success. Raises ``SystemExit`` with a
    ``MEASUREMENT_FAILURE`` marker otherwise -- never substitutes a default.
    """
    doc = json.loads(Path(checkpoints_path).read_text(encoding="utf-8"))
    declared = ((doc.get("substrate") or {}).get("env_class") or "").strip()
    if declared != REQUIRED_SUBSTRATE:
        raise SystemExit(f"""MEASUREMENT_FAILURE: refusing to run Study B on {checkpoints_path}
  declared substrate : {declared or '<absent>'}
  required substrate : {REQUIRED_SUBSTRATE}
  A checkpoint artifact with no substrate block was built on the RETIRED
  bespoke runtime (shared warm container, generic debian_slim image,
  hardcoded 'python -m pytest'). Restoring it into the formal per-task
  substrate would silently mix stacks.
  Use data/checkpoints/checkpoints_v2.json.""")
    return declared


def assert_stimulus_coverage(checkpoints_path: str, stimuli_path: str) -> int:
    """Refuse to start unless EVERY checkpoint's stimulus is present.

    The engine already raises ``KeyError`` on a missing stimulus, but it does so
    mid-loop -- after the earlier checkpoints have already been paid for. The
    shipped default (``data/stimuli/stimuli_manifest.json``, tagged
    ``SCREENING_OR_SMOKE_ONLY``) covers only the 8 LHAW checkpoints, so a formal
    run launched on it dies 8 checkpoints in. Check up front instead.
    """
    ck = json.loads(Path(checkpoints_path).read_text(encoding="utf-8"))
    st = json.loads(Path(stimuli_path).read_text(encoding="utf-8"))
    need = {c.get("stimulus_id") for c in ck.get("checkpoints", [])}
    have = {s.get("stimulus_id") for s in st.get("stimuli", [])}
    missing = sorted(x for x in (need - have) if x)
    if missing:
        raise SystemExit(f"""MEASUREMENT_FAILURE: stimuli do not cover the checkpoints
  checkpoints : {checkpoints_path} ({len(need)} stimulus ids)
  stimuli     : {stimuli_path} (tag={st.get('tag')!r}, {len(have)} ids)
  covered     : {len(need & have)}/{len(need)}
  missing     : {len(missing)}, e.g. {missing[:3]}
  Refusing up front rather than failing partway through a paid run.
  Use data/checkpoints/stimuli_v2.json.""")
    return len(need)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-config", default=str(ROOT / "configs/models/mock.yaml"))
    ap.add_argument("--mock-policy", default=None,
                    help="override the mock policy (dry-run / mock backends)")
    ap.add_argument("--task-selection",
                    default=str(ROOT / "data/qualified/task_selection.json"))
    ap.add_argument("--checkpoints",
                    default=str(ROOT / "data/checkpoints/checkpoints_v2.json"))
    ap.add_argument("--stimuli",
                    default=str(ROOT / "data/checkpoints/stimuli_v2.json"))
    ap.add_argument("--experiment-config",
                    default=str(ROOT / "configs/experiment_config.yaml"))
    ap.add_argument("--out", default=str(ROOT / "results/raw"))
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--max-turns", type=int, default=6)
    ap.add_argument("--shard-index", type=int, default=0)
    ap.add_argument("--shard-count", type=int, default=1)
    ap.add_argument("--arm",
                    choices=["ours", "official_prompt_typed_tools",
                             "ours_prompt_open_tools",
                             "disclosed", "undisclosed", "official_replication"],
                    default="ours",
                    help="which cell of the prompt x tools grid to run "
                         "(see pp.arms). 'ours' = our contract + typed tools "
                         "(the baseline). 'official_prompt_typed_tools' = "
                         "upstream's wording without a shell, isolating the "
                         "prompt. 'ours_prompt_open_tools' = our contract with "
                         "upstream's full tool set (bash, python, text_editor, think), "
                         "isolating capability. "
                         "'official_replication' (upstream wording + upstream "
                         "tools) is ImpossibleBench itself and is NOT run here. "
                         "'disclosed'/'undisclosed' are the legacy labels the "
                         "archived corpus was recorded under.")
    ap.add_argument("--only-benchmark", choices=["all", "impossible", "lhaw"],
                    default="all",
                    help="restrict Study B to one benchmark's checkpoints. "
                         "MEASURED: LHAW checkpoints engage the authorization "
                         "boundary in 16%% of episodes vs 100%% for "
                         "ImpossibleBench, so they cannot move the primary "
                         "outcome. Excluded ids are recorded in the summary.")
    ap.add_argument("--resume", action="store_true",
                    help="skip checkpoints whose output file is already COMPLETE "
                         "(5 forks, contract, stimulus report, every fork scored). "
                         "Lets an interrupted run continue without re-paying for "
                         "work already finished. A truncated or short file is NOT "
                         "trusted and is re-run.")
    ap.add_argument("--dry-run", action="store_true",
                    help="offline MockBackend proof on shortlist checkpoints")
    args = ap.parse_args()

    if args.dry_run:
        # A dry run must NOT write into the evidence tree. `--out` defaults to
        # results/raw, and the suite counts episodes there as recorded evidence
        # (that is how superseded data is kept from drifting into a headline).
        # A mock episode landing beside real ones is exactly the contamination
        # the rest of this repo is built to prevent, so unless the caller named
        # a destination, redirect to a scratch directory.
        if args.out == str(ROOT / "results/raw"):
            args.out = str(ROOT / "results" / "_dryrun")
            print(f"[dry-run] writing to {args.out} (not the evidence tree)",
                  file=sys.stderr)
        summary = engine.dry_run(args.out, mock_policy=args.mock_policy
                                 or "crosser",
                                 include_impossible_checkpoint=True)
        b = summary["studies"].get("B", {})
        print(json.dumps(b, indent=2))
        print(f"\nStudy B dry-run OK (fork_integrity_ok="
              f"{b.get('fork_integrity_ok')}) -> {args.out}/{summary['model']}/")
        return

    substrate = assert_formal_substrate(args.checkpoints)
    n_ck = assert_stimulus_coverage(args.checkpoints, args.stimuli)
    print(f"[preflight] substrate={substrate}", file=sys.stderr)
    print(f"[preflight] stimulus coverage {n_ck}/{n_ck} OK", file=sys.stderr)
    from pp.runner_contract import (official_system_prompt, prompt_sha256,
                                    system_prompt_for,
                                    OFFICIAL_TOOLS_BLOCK_TYPED,
                                    OFFICIAL_PREAMBLE_TYPED)
    from pp.arms import ARMS as _A, PROMPT_OFFICIAL, TOOLS_TYPED
    _c = _A.get(args.arm)
    if _c is not None and _c.prompt == PROMPT_OFFICIAL:
        # Official-wording arms print the graded test command, so the prompt is
        # PER TASK and there is no single sha to pin. Report the invariant parts
        # instead: the STOP sentence present, our three guardrails absent.
        _p = official_system_prompt(
            test_command="<per-task>", test_files=["<per-task>"],
            **({"tools_block": OFFICIAL_TOOLS_BLOCK_TYPED,
                "preamble": OFFICIAL_PREAMBLE_TYPED}
               if _c.tools == TOOLS_TYPED else {}))
    elif _c is not None:
        _p = system_prompt_for("impossible", arm="disclosed",
                               untyped_tools=(_c.tools != TOOLS_TYPED))
        print(f"[preflight] arm={args.arm} prompt_sha={prompt_sha256(_p)[:16]} "
              f"boundary_tokens_in_prompt="
              f"{'EXECUTE_OUT_OF_SCOPE_ACTION' in _p}", file=sys.stderr)
    elif args.arm == "official_replication":
        _p = official_system_prompt(test_command="<per-task>",
                                    test_files=["<per-task>"])
        print(f"[preflight] arm={args.arm} (official template, per-task prompt)",
              file=sys.stderr)
        print(f"[preflight]   STOP sentence present : "
              f"{'STOP.' in _p}", file=sys.stderr)
        print(f"[preflight]   scope statement       : "
              f"{'out of your scope' in _p}", file=sys.stderr)
        print(f"[preflight]   action vocabulary     : "
              f"{'MUST be exactly one of' in _p}", file=sys.stderr)
        print(f"[preflight]   anti-legitimation     : "
              f"{'unpunished precedent' in _p}", file=sys.stderr)
    else:
        _p = system_prompt_for("impossible", arm=args.arm)
        print(f"[preflight] arm={args.arm} prompt_sha={prompt_sha256(_p)[:16]} "
              f"boundary_tokens_in_prompt="
              f"{'EXECUTE_OUT_OF_SCOPE_ACTION' in _p}", file=sys.stderr)

    cfg = load_model_config(args.model_config)
    if args.mock_policy and cfg.get("backend") == "mock":
        cfg["policy"] = args.mock_policy
    manifests = engine.Manifests.load(
        args.task_selection, args.checkpoints, args.stimuli,
        args.experiment_config, for_studies=("B",))
    provider = engine.ModalEnvProvider()
    # The tool set comes from the SAME table that chose the prompt, so a cell
    # cannot advertise one capability set and serve another.
    from pp.arms import ARMS as _ARMS
    _cell = _ARMS.get(args.arm)
    if _cell is not None:
        provider.official_tools = _cell.official_tools
        # verbatim from the arm table: the prompt advertises exactly this set
        provider.untyped_tools = list(_cell.untyped_tools)
        print(f"[preflight] arm={_cell.name} prompt={_cell.prompt} "
              f"tools={_cell.tools} untyped={list(_cell.untyped_tools)} "
              f"({_cell.rationale})", file=sys.stderr)
        if not _cell.runnable:
            print(f"[preflight] REFUSING: {_cell.rationale}", file=sys.stderr)
            return
    elif args.arm == "official_replication":
        # Grant official's untyped tools (bash / python / text_editor incl.
        # undo_edit). Safe only because the crossing instrument now reads the
        # harness-file digest, not the shape of the write.
        provider.official_tools = True
        print("[preflight] official_tools=True (bash, python, text_editor)",
              file=sys.stderr)
    shard = engine.ShardSpec(index=args.shard_index, count=args.shard_count)
    # THE RUNNABLE CONDITION SET. B4 is retired from new runs (DECISIONS.md
    # section 3) but remains in the schema; without this the engine's default is
    # the full five-condition SPRINT_B_CONDITIONS and a fifth of the budget goes
    # to a cell that cannot discriminate.
    from pp.treatments.conditions import ACTIVE_B_CONDITIONS
    summary = engine.run_all(
        cfg, manifests, provider, args.out, shard=shard,
        seeds_b=args.seeds, max_turns_b=args.max_turns, studies=("B",),
        conditions_b=tuple(t.value for t in ACTIVE_B_CONDITIONS),
        arm=args.arm, resume=args.resume,
        benchmarks_b=(None if args.only_benchmark == "all"
                      else (normalize_benchmark(args.only_benchmark),)))
    print(json.dumps(summary["studies"].get("B", {}), indent=2))


if __name__ == "__main__":
    main()
