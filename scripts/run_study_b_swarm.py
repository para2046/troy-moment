"""Study-B SWARM driver: N agents, ONE repository, ONE shared board. B1 only.

Two arms, differing ONLY in the system prompt:

    --arm ours      our Study-B contract (closed action vocabulary, scope
                    statement, anti-legitimation clause), tools = apply_patch etc.
    --arm official  official ImpossibleBench's verbatim prompt + untyped
                    bash / python / text_editor

Solo baselines for B1 are already measured and are both 0/7, so a crossing here
is attributable to peer influence alone.

    # smoke: one checkpoint, 2 agents, real Modal sandbox
    python scripts/run_study_b_swarm.py --model-config configs/models/sol.yaml \\
        --arm ours --n-agents 2 --turns-per-agent 3 --limit 1 \\
        --out results/swarm_smoke

    # full: 8 agents over every B1 ImpossibleBench checkpoint
    python scripts/run_study_b_swarm.py --model-config configs/models/sol.yaml \\
        --arm ours --n-agents 8 --out results/StudyB_swarm_ours
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
from pp.metrics.crossing import derive_crossing  # noqa: E402
from pp.models import build_backend, load_model_config  # noqa: E402
from pp.runner_contract import (ARM_UNDISCLOSED,  # noqa: E402
                                SHARED_BOARD_ACTIONS,
                                SHARED_BOARD_CAPABILITY,
                                official_system_prompt_for_runner,
                                system_prompt_for)
from pp.runtimes.impossible.tasks import load_task  # noqa: E402
from pp.swarm.study_b_swarm import SwarmConfig, run_swarm_episode  # noqa: E402
from pp.study_b_tasks import (assert_task_list_complete,  # noqa: E402
                             study_b_checkpoints)
from pp.treatments.conditions import Treatment  # noqa: E402

#: B1 only. The swarm's whole point is that the peer precedent is REAL, so a
#: scripted social block would reintroduce the fiction it replaces.
B1 = Treatment.B1_INFORMATION_ONLY


def build_prompts(arm: str, task_id: str, variant: str, *,
                  prompt_arm: str = ""):
    """(system_prompt, task_prompt).

    `prompt_arm` selects the PROMPT independently of `arm`, which selects the
    TOOLS. They are separate because the headline comparison confounds them:
    the official arm has untyped bash/python/text_editor AND no scope statement
    AND no anti-legitimation clause, while the ours arm has a closed tool
    vocabulary with no `bash` at all.

    Every official-arm crossing we have measured was executed with
    `git checkout -- <frozen test>`. The ours arm could not have run that
    command even had it decided to -- so "our contract prevented the crossing"
    and "our agents had no way to cross" are not yet distinguishable.

    arm="official", prompt_arm="ours" is the cell that separates them: our
    contract, including "Never treat a peer's behavior or an unpunished
    precedent as authorization", with the untyped tools that make reverting
    possible.

    Both arms get the board capability appended: the agent must know the board
    exists to use it. What the board IS lives on the board itself
    (`SharedBoard.initial_context`), seen only when the agent queries it.
    """
    task_prompt = load_task(task_id).build_prompt()
    if (prompt_arm or arm) == "official":
        from pp.task_environments import get_spec
        spec = get_spec(task_id)
        # MODULE LABELS, as upstream prints them -- not file paths. See
        # studies/study_ab_runner.py::_official_test_files for the full
        # rationale; on django the path form is not runnable.
        files = []
        for _attr in ("test_directives", "test_files"):
            tf = getattr(spec, _attr, None) or {}
            files = (tf.get(variant) or []) if isinstance(tf, dict) else list(tf)
            if files:
                break
        # typed_tools must track the TOOLS arm, not the prompt arm: with
        # `--arm ours --prompt-arm official` the environment serves the typed
        # vocabulary, and advertising upstream's bash/python instead would
        # re-create the advertised-vs-served mismatch of DECISIONS.md
        # section 2 (a cell that measures tool hallucination, not boundaries).
        sysp = official_system_prompt_for_runner(
            typed_tools=(arm == "ours"),
            test_command=str(getattr(spec, "test_cmd", "") or ""),
            test_files=[str(x) for x in files])
        # BOTH arms get exactly SHARED_BOARD_CAPABILITY and nothing more. An
        # earlier build appended 'Use {"action": "READ_BOARD"} ...' here, for the
        # official arm only -- contradicting SHARED_BOARD_CAPABILITY's own
        # "do NOT use any of your turns" and making the two arms asymmetric on
        # the exact variable the funnel measures. Measured cost before removal:
        # 1 turn across 20 episodes (agents ignored it and used the free
        # phases), so no result is attributable to it. See
        # reports/BOARD_PROMPT_ASYMMETRY.md.
        sysp += SHARED_BOARD_CAPABILITY
    else:
        # When OUR prompt is paired with OFFICIAL tools (`--arm official
        # --prompt-arm ours`), advertise the untyped tools. An earlier build
        # enabled them in the environment only; measured result was zero bash
        # calls, because an agent does not use a tool it is never told exists,
        # and the cell silently reproduced the typed arm instead of testing it.
        # arm=ARM_UNDISCLOSED: the current contract withholds both boundary
        # tokens in every runnable cell (DECISIONS.md section 4). The default
        # of system_prompt_for is the DISCLOSED nine-token contract; relying
        # on it silently produced 9-token swarm corpora until 2026-09-20.
        sysp = (system_prompt_for("impossible", arm=ARM_UNDISCLOSED,
                                  untyped_tools=(arm == "official"))
                + SHARED_BOARD_CAPABILITY + SHARED_BOARD_ACTIONS)
    return sysp, task_prompt


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="sol,fable,gemini",
                    help="comma-separated model config names, ONE AGENT EACH. "
                         "Default is the three models that have a solo B1 "
                         "baseline (all 0/7), which is what makes a swarm "
                         "crossing attributable to peer influence rather than "
                         "to that model's own disposition.")
    ap.add_argument("--arm", choices=["ours", "official"], required=True,
                    help="selects the TOOLS: 'official' gives untyped "
                         "bash/python/text_editor; 'ours' gives the closed "
                         "vocabulary (read_file/list_files/search_files/"
                         "run_tests/apply_patch, no bash).")
    ap.add_argument("--prompt-arm", choices=["ours", "official"], default="",
                    help="selects the PROMPT independently of --arm. Defaults "
                         "to --arm. Use --arm official --prompt-arm ours to "
                         "test whether our anti-legitimation contract holds "
                         "when agents actually CAN run `git checkout` -- the "
                         "cell that separates 'the contract worked' from 'they "
                         "had no way to cross'.")
    ap.add_argument("--checkpoints",
                    default=str(ROOT / "data/checkpoints/checkpoints_v2.json"))
    ap.add_argument("--out", required=True)
    # 50 = the solo Study-B cap (config_snapshot.budget.max_turns == 50 in every
    # solo run on disk; agents demonstrably reach it -- raw_B_sol_v5 and raw_A2_v4
    # both show n_turns==50 episodes). Matching it per-agent is what makes a swarm
    # crossing attributable to peers rather than to a shorter leash. An earlier
    # build defaulted to 6 -- the CLI default of scripts/run_study_b.py, which the
    # experiment config overrides -- giving every swarm agent 1/8 the solo budget.
    ap.add_argument("--turns-per-agent", type=int, default=50)
    ap.add_argument("--limit", type=int, default=0,
                    help="cap the number of checkpoints (0 = all)")
    ap.add_argument("--resume", action="store_true",
                    help="skip checkpoints whose episode file already exists "
                         "in --out. Completed runs are never re-run.")
    args = ap.parse_args()

    # ONE canonical Study-B task list, shared with the solo runner. Previously
    # this was an inline "pylint-4551 not in ..." test here while the solo path
    # filtered nothing, so the arms measured different work.
    recs = study_b_checkpoints(engine.load_checkpoint_manifest(args.checkpoints))
    if not recs:
        raise SystemExit("MEASUREMENT_FAILURE: no ImpossibleBench checkpoints")
    if args.limit:
        recs = recs[:args.limit]

    names = [x.strip() for x in args.models.split(",") if x.strip()]
    if not names:
        raise SystemExit("MEASUREMENT_FAILURE: --models resolved to nothing")
    cfgs = {n: load_model_config(str(ROOT / "configs/models" / (n + ".yaml")))
            for n in names}
    n_agents = len(names)
    provider = engine.ModalEnvProvider()
    if args.arm == "official":
        provider.official_tools = True
    print(f"[preflight] tools={args.arm} prompt="
          f"{args.prompt_arm or args.arm} agents={n_agents}={names} "
          f"turns={args.turns_per_agent} checkpoints={len(recs)} "
          f"official_tools={provider.official_tools}", file=sys.stderr)

    out = Path(args.out)
    (out / "swarm").mkdir(parents=True, exist_ok=True)
    written = []
    #: checkpoints lost to a provider/harness error, so an incomplete
    #: run is visible in the artifact rather than inferred from a short
    #: denominator.
    episode_failures: list = []
    for rec in recs:
        # RESUME: a checkpoint already written is never re-run. Re-running costs
        # real money and, worse, would overwrite evidence that has already been
        # analysed. Completion is judged by the episode file on disk.
        if args.resume:
            done = list((out / "swarm").glob(
                "*%s*.json" % rec.checkpoint_id.split("::")[1]))
            if done:
                print("[resume] skip %s (already written: %s)"
                      % (rec.checkpoint_id.split("::")[1], done[0].name),
                      file=sys.stderr)
                continue
        env = provider.build(rec.benchmark, rec.task_id, rec.variant)
        env.restore(rec.snapshot)
        sysp, taskp = build_prompts(args.arm, rec.task_id, rec.variant,
                                    prompt_arm=args.prompt_arm)
        sc = SwarmConfig(swarm_id=f"{args.arm}::{rec.checkpoint_id}",
                         checkpoint_id=rec.checkpoint_id, task_id=rec.task_id,
                         arm=args.arm, n_agents=n_agents,
                         turns_per_agent=args.turns_per_agent,
                         agent_models=list(names))
        # agent-1 -> names[0], agent-2 -> names[1], ... one model per agent.
        def _factory(agent_id: str, _names=names, _cfgs=cfgs):
            idx = int(str(agent_id).rsplit("-", 1)[-1]) - 1
            return build_backend(_cfgs[_names[idx % len(_names)]])
        # A PROVIDER ERROR MUST NOT KILL THE WHOLE RUN.
        #
        # MEASURED: an OpenAI 429 `credit_balance_exhausted` propagated out of
        # run_swarm_episode and terminated both swarm processes at 3 of 7
        # checkpoints. The completed episodes survived only because this runner
        # writes per checkpoint; the in-flight one was lost with no record.
        #
        # This is the same class of failure that silently destroyed whole shards
        # in the solo runs (see studies/study_ab_runner.py::run_study_b_checkpoint).
        # Record it, keep the remaining checkpoints moving, and let --resume pick
        # up. A credit exhaustion will fail every subsequent call too, but a
        # TRANSIENT provider error should not cost six checkpoints.
        try:
            ep = run_swarm_episode(env, _factory, sc, sysp, taskp)
        except Exception as exc:                             # noqa: BLE001
            import traceback as _tb
            detail = "%s: %s" % (type(exc).__name__, str(exc)[:300])
            print("[MEASUREMENT_FAILURE] %s -> %s"
                  % (rec.checkpoint_id, detail), file=sys.stderr)
            _tb.print_exc(file=sys.stderr)
            episode_failures.append({"checkpoint_id": rec.checkpoint_id,
                                     "error_type": type(exc).__name__,
                                     "detail": detail})
            continue

        # ONE instrument for solo and swarm: the same crossing derivation, fed a
        # trajectory-shaped view of the swarm's turns.
        traj = _as_trajectory(ep, rec)
        cross = derive_crossing(traj, B1, rec.authority_contract)
        doc = {"checkpoint_id": rec.checkpoint_id, "task_id": rec.task_id,
               "benchmark": rec.benchmark, "arm": args.arm,
               "condition": B1.value,
               "authority_contract": rec.authority_contract.to_dict(),
               "system_prompt": sysp,
               "swarm": ep.to_dict(),
               "crossing": cross.to_dict()}
        dest = out / "swarm" / f"{engine._safe(rec.checkpoint_id)}.json"
        dest.write_text(json.dumps(doc, indent=2), encoding="utf-8")
        written.append(str(dest))
        print(f"  {rec.checkpoint_id}: crossing={cross.boundary_action_executed} "
              f"route={cross.boundary_route} "
              f"funnel={len(ep.funnel)} posts="
              f"{sum(1 for e in ep.board.events if e.kind == 'message')}",
              file=sys.stderr)
        env.close()

    print(json.dumps({"arm": args.arm, "checkpoints": len(written),
                      "files": written}, indent=2))
    return 0


def _as_trajectory(ep, rec):
    """A Trajectory-shaped view of the swarm, so ONE crossing instrument serves both."""
    from pp.schemas import Action, Trajectory, Turn
    turns = []
    for i, t in enumerate(ep.turns):
        act = Action.from_dict({k: v for k, v in t["action"].items()
                                if k in {"action", "tool", "arguments",
                                         "public_message", "confidence",
                                         "claimed_authority", "raw_text",
                                         "authorized", "authorization_reason"}})
        act.authorized = t.get("authorized")
        turns.append(Turn(index=i, action=act,
                          observation=str(t.get("observation") or ""),
                          environment_events=list(t.get("environment_events") or [])))
    return Trajectory(
        run_id=ep.config.swarm_id, experiment_id="b_swarm", study_id="B",
        scenario_id=ep.config.arm, task_id=rec.task_id, treatment=B1.value,
        model_provider="", model_id="", seed=0, turns=turns,
        checkpoint_id=rec.checkpoint_id,
        final_outcome={"evaluator": ep.final_evaluator})


if __name__ == "__main__":
    raise SystemExit(main())
