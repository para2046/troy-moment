"""Drive period-correct canonical replays for the Impossible requalification.

Execution is delegated entirely to ``agent-env``'s published interface
(``pp.task_environments.run_canonical``, discovered through
:mod:`pp.requal.env_adapter`). This module only *schedules* executions and
stores the records; it builds no environment of its own and never touches the
shared warm container.

Validation design (Human-Principal decision, 2026-09-12)
--------------------------------------------------------
**2 observations per sample, escalate on disagreement** -- replacing "3 replays
+ an order-swapped pass":

1. one canonical replay, in canonical (alphabetical) task order;
2. one **order-swapped** pass -- a genuinely different preceding-task order,
   with the ``pytest-dev`` repo deliberately moved out of its canonical
   position, since editable-installing the pytest repo was the vector of the
   original contamination. This is simultaneously the second observation and the
   direct test of whether the isolation fix holds;
3. if the two AGREE on the model-visible semantic failure type -> STABLE:
   ``n_observations=2, agreement=true, escalated=false``;
4. if they DISAGREE -> escalate that sample to >=3 further replays, report it as
   an instability finding with **all** observations. No averaging, no silent
   majority vote, no discarded minority observation.

**Rationale, stated plainly rather than implied.** The original defect *was*
single-observation evidence: one run per task, where the observation turned out
to be a function of which task ran before it. Two observations is the minimum
that can detect instability at all; one cannot, by construction. Moving from 3
to 2 is a bounded reduction in assurance -- a low-rate intermittent flake could
pass -- accepted in exchange for wall-clock now that per-execution
``modal.Sandbox`` isolation from immutable prebuilt per-instance images makes
cross-task contamination architecturally impossible. 2 and 3 observations are
**not** equivalent and are not presented as such.

**Asymmetry between the paired variants, by design.** Study-A Q2 evidence covers
both paired variants (60 task-variant samples). The full 2-observation stability
treatment is applied to the **30 conflicting** samples, where *semantic failure
stability* is the criterion. The **30 original** samples evidence *original
solvability* -- a different claim that does not turn on failure-type stability --
so **1 observation each** is sufficient there. Baseline: 30x2 + 30x1 = **90
executions**, plus escalations.
"""
from __future__ import annotations

import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from . import env_adapter

REPO = Path(__file__).resolve().parents[3]
OUT_ROOT = REPO / "results" / "requal_impossible"

VARIANTS = ("original", "conflicting")

#: Observations per variant under the 2-observation design.
OBSERVATIONS = {"conflicting": 2, "original": 1}

#: Further replays a disagreeing sample is escalated to.
ESCALATION_REPLAYS = 3

#: The repo whose editable install was the contamination vector. The
#: order-swapped pass must not leave it in its canonical position.
CONTAMINATION_VECTOR_PREFIX = "pytest-dev__"

_lock = threading.Lock()
_thread_last: Dict[int, str] = {}

#: concurrency telemetry
_inflight = 0
_peak_inflight = 0
_throttle_events: List[Dict[str, Any]] = []

_THROTTLE_MARKERS = ("rate limit", "too many requests", "429", "resource",
                     "quota", "capacity", "sandbox creation", "concurrency")


def _path(task: str, variant: str, obs: int) -> Path:
    return OUT_ROOT / task / f"{variant}.obs{obs}.json"


def _log(msg: str) -> None:
    with _lock:
        print(msg, flush=True)


# ---------------------------------------------------------------------------
# ordering
# ---------------------------------------------------------------------------
def canonical_order(tasks: Sequence[str]) -> List[str]:
    """Canonical (alphabetical) order -- the order the pre-fix runs used."""
    return sorted(tasks)


def swapped_order(tasks: Sequence[str]) -> List[str]:
    """A genuinely different order, with the contamination vector relocated.

    Reversing alphabetical order already moves every task, and it moves the
    ``pytest-dev`` tasks from the back of the canonical order to the front. The
    relocation is then asserted rather than assumed.
    """
    canon = canonical_order(tasks)
    swapped = list(reversed(canon))
    for t in canon:
        if t.startswith(CONTAMINATION_VECTOR_PREFIX):
            assert canon.index(t) != swapped.index(t), (
                f"order-swapped pass left the contamination vector {t} in its "
                "canonical position")
    return swapped


def order_evidence(tasks: Sequence[str]) -> Dict[str, Any]:
    canon, swap = canonical_order(tasks), swapped_order(tasks)
    vec = [t for t in canon if t.startswith(CONTAMINATION_VECTOR_PREFIX)]
    return {
        "canonical_order": canon,
        "swapped_order": swap,
        "n_positions_changed": sum(1 for i, t in enumerate(canon)
                                   if swap[i] != t),
        "contamination_vector_tasks": vec,
        "contamination_vector_positions": {
            t: {"canonical": canon.index(t), "swapped": swap.index(t)}
            for t in vec},
        "contamination_vector_relocated": all(
            canon.index(t) != swap.index(t) for t in vec),
        "note": ("agent-env gives every execution a FRESH modal.Sandbox from an "
                 "immutable per-instance image, so there is no shared state for "
                 "order to act through. The swapped pass therefore tests the "
                 "isolation claim: a semantic type that differed across "
                 "predecessors would falsify it."),
    }


# ---------------------------------------------------------------------------
# one execution
# ---------------------------------------------------------------------------
def one(task: str, variant: str, obs: int, *, order_label: str,
        force: bool = False) -> Dict[str, Any]:
    """One canonical execution, cached on disk."""
    global _inflight, _peak_inflight
    p = _path(task, variant, obs)
    if p.exists() and not force:
        rec = json.loads(p.read_text(encoding="utf-8"))
        rec["reused_from_cache"] = True
        # Re-classify rather than trust: a taxonomy correction must reach every
        # observation without re-paying for a Modal sandbox.
        return env_adapter.reclassify(rec)
    tid = threading.get_ident()
    preceding = _thread_last.get(tid)
    with _lock:
        _inflight += 1
        _peak_inflight = max(_peak_inflight, _inflight)
    t0 = time.time()
    try:
        rec = env_adapter.run_canonical(task, variant=variant, rep=obs,
                                        preceding_task=preceding,
                                        session=f"q2_requal_{order_label}")
    except env_adapter.EnvLayerUnavailable as exc:
        with _lock:
            _inflight -= 1
        return env_adapter.pending_record(task, variant, str(exc))
    except Exception as exc:                                   # noqa: BLE001
        msg = f"{type(exc).__name__}: {exc}"
        if any(m in msg.lower() for m in _THROTTLE_MARKERS):
            with _lock:
                _throttle_events.append({"task": task, "variant": variant,
                                         "obs": obs, "error": msg[:300]})
        # A harness exception is a MEASUREMENT FAILURE, never evidence.
        rec = env_adapter.normalize_record(
            {"harness_error": msg, "ok": False},
            task_id=task, variant=variant, rep=obs,
            preceding_task=preceding, env_layer="pp.task_environments")
    finally:
        with _lock:
            _inflight = max(0, _inflight - 1)
    rec["wall_seconds"] = round(time.time() - t0, 2)
    rec["observation_index"] = obs
    rec["order_label"] = order_label
    rec["preceding_execution_in_worker"] = preceding
    _thread_last[tid] = f"{task}::{variant}"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(rec, indent=2, default=str) + "\n", encoding="utf-8")
    _log(f"  {task:34s} {variant:12s} obs{obs}/{order_label:10s} "
         f"rc={rec.get('rc')} {rec.get('semantic_failure_type')} "
         f"subst={rec.get('substantive_failure_evidence')} "
         f"{rec.get('wall_seconds')}s  after={preceding}")
    return rec


# ---------------------------------------------------------------------------
# the run
# ---------------------------------------------------------------------------
def run_all(tasks: Sequence[str], *, workers: int = 8, force: bool = False,
            budget_seconds: Optional[float] = None,
            harvest_study: Optional[str] = "A",
            execute: bool = True,
            escalate: bool = True,
            ) -> Tuple[Dict[str, Dict[str, List[Dict[str, Any]]]], Dict[str, Any]]:
    """Run the 2-observation design over ``tasks``.

    Returns ``({task: {variant: [records]}}, telemetry)``.
    """
    global _peak_inflight
    _peak_inflight = 0
    _throttle_events.clear()
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    assert workers <= 12, ("concurrency above 12 requires telling the "
                           "orchestrator: the Modal workspace is shared with "
                           "agent-bcheck")

    out: Dict[str, Dict[str, List[Dict[str, Any]]]] = {
        t: {v: [] for v in VARIANTS} for t in tasks}

    canon, swap = canonical_order(tasks), swapped_order(tasks)
    # phase 1: canonical order -- conflicting obs1 + original obs1
    jobs: List[Tuple[str, str, int, str]] = []
    for t in canon:
        jobs.append((t, "conflicting", 1, "canonical"))
    for t in canon:
        jobs.append((t, "original", 1, "canonical"))
    # phase 2: order-swapped -- conflicting obs2 only
    for t in swap:
        jobs.append((t, "conflicting", 2, "order_swapped"))

    t_start = time.time()
    n_exec = _dispatch(jobs, out, workers=workers, force=force,
                       execute=execute, budget_seconds=budget_seconds,
                       t_start=t_start)

    # ---- escalation: samples whose two observations DISAGREE ---------------
    escalations: List[Dict[str, Any]] = []
    if escalate and execute:
        esc_jobs: List[Tuple[str, str, int, str]] = []
        for t in canon:
            recs = [r for r in out[t]["conflicting"]
                    if r.get("semantic_failure_type") is not None]
            types = sorted({r["semantic_failure_type"] for r in recs})
            if len(recs) >= 2 and len(types) > 1:
                escalations.append({"task_id": t, "variant": "conflicting",
                                    "disagreeing_types": types})
                for k in range(3, 3 + ESCALATION_REPLAYS):
                    esc_jobs.append((t, "conflicting", k, "escalation"))
        if esc_jobs:
            _log(f"  [escalation] {len(escalations)} sample(s) disagreed; "
                 f"running {len(esc_jobs)} further replays")
            n_exec += _dispatch(esc_jobs, out, workers=workers, force=force,
                                execute=execute, budget_seconds=None,
                                t_start=t_start)

    wall = time.time() - t_start

    # ---- agent-env's own published executions: INDEPENDENT CORROBORATION ---
    # Deliberately NOT merged into the decision observations. The Human
    # Principal fixed the design at 2 observations per sample with escalation on
    # disagreement; silently adding agent-env's replays would change that design
    # and would pool executions run under a different session/gate. They are
    # recorded as an independent cross-check instead.
    corroboration: Dict[str, Dict[str, Any]] = {}
    if harvest_study:
        harvested = env_adapter.harvest_published(harvest_study)
        for task, pv in harvested.items():
            if task not in out:
                continue
            for variant, recs in pv.items():
                mine = {r.get("semantic_failure_type")
                        for r in out.get(task, {}).get(variant, [])
                        if r.get("semantic_failure_type")}
                theirs = sorted({r.get("semantic_failure_type") for r in recs
                                 if r.get("semantic_failure_type")})
                corroboration.setdefault(task, {})[variant] = {
                    "n_independent_executions": len(recs),
                    "semantic_types": theirs,
                    "our_semantic_types": sorted(mine),
                    "agrees_with_ours": bool(mine) and bool(theirs)
                    and set(theirs) == mine,
                    "source": "results/study_env_validation/ (agent-env, read-only)",
                    "used_in_decision": False,
                    "logs": [r.get("log") for r in recs],
                }

    for task in out:
        for variant in out[task]:
            out[task][variant].sort(
                key=lambda r: (str(r.get("order_label")),
                               r.get("observation_index") or r.get("rep") or 0))

    # Wall-clock stats are taken from every stored execution record (including
    # ones reused from an earlier pass), because they were genuinely measured.
    walls = [r.get("wall_seconds") for pv in out.values() for recs in pv.values()
             for r in recs if r.get("wall_seconds")]
    telemetry = {
        "design": "2_observations_escalate_on_disagreement",
        "observations_per_variant": dict(OBSERVATIONS),
        "escalation_replays_on_disagreement": ESCALATION_REPLAYS,
        "n_task_variant_samples": len(tasks) * len(VARIANTS),
        "baseline_executions_planned": len(tasks) * (
            OBSERVATIONS["conflicting"] + OBSERVATIONS["original"]),
        "executions_run_this_pass": n_exec,
        "executions_reused_from_cache": sum(
            1 for pv in out.values() for recs in pv.values() for r in recs
            if r.get("reused_from_cache")),
        "wall_seconds_this_pass": round(wall, 1),
        "mean_seconds_per_execution": (round(sum(walls) / len(walls), 1)
                                       if walls else None),
        "max_seconds_per_execution": (round(max(walls), 1) if walls else None),
        "min_seconds_per_execution": (round(min(walls), 1) if walls else None),
        "workers_requested": workers,
        "peak_concurrency_this_pass": _peak_inflight,
        "throttling_events": list(_throttle_events),
        "throttling_observed": bool(_throttle_events),
        "escalated_samples": escalations,
        "n_escalated_samples": len(escalations),
        "order_evidence": order_evidence(tasks),
        "modal_workspace_shared_with": "agent-bcheck",
        "measured_pass": _measured_pass_telemetry(tasks),
        "independent_corroboration": corroboration,
    }
    return out, telemetry


def _measured_pass_telemetry(tasks: Sequence[str]) -> Dict[str, Any]:
    """Reconstruct the wall-clock of the pass that actually executed.

    Every execution writes its own record file, so the pass window is
    ``[min(mtime) - duration, max(mtime)]`` and peak concurrency is the maximum
    number of overlapping ``[finish - wall_seconds, finish]`` intervals. Both
    are measured from the stored records rather than asserted.
    """
    recs = []
    for t in tasks:
        d = OUT_ROOT / t
        if not d.is_dir():
            continue
        for f in d.glob("*.json"):
            try:
                r = json.loads(f.read_text(encoding="utf-8"))
            except Exception:                                  # noqa: BLE001
                continue
            w = r.get("wall_seconds")
            if not w:
                continue
            end = f.stat().st_mtime
            recs.append((end - float(w), end, float(w)))
    if not recs:
        return {"n_executions": 0}
    starts = [a for a, _, _ in recs]
    ends = [b for _, b, _ in recs]
    events = sorted([(a, 1) for a, _, _ in recs] + [(b, -1) for _, b, _ in recs])
    cur = peak = 0
    for _, delta in events:
        cur += delta
        peak = max(peak, cur)
    walls = [w for _, _, w in recs]
    total = max(ends) - min(starts)
    return {
        "n_executions": len(recs),
        "wall_seconds_total": round(total, 1),
        "wall_minutes_total": round(total / 60.0, 1),
        "mean_seconds_per_execution": round(sum(walls) / len(walls), 1),
        "median_seconds_per_execution": round(
            sorted(walls)[len(walls) // 2], 1),
        "max_seconds_per_execution": round(max(walls), 1),
        "min_seconds_per_execution": round(min(walls), 1),
        "sum_execution_seconds": round(sum(walls), 1),
        "peak_concurrency_achieved": peak,
        "effective_parallelism": round(sum(walls) / total, 2) if total else None,
        "method": ("measured from the per-execution record files: interval "
                   "[mtime - wall_seconds, mtime] per execution; peak "
                   "concurrency = max overlapping intervals"),
    }


def _dispatch(jobs, out, *, workers, force, execute, budget_seconds, t_start
              ) -> int:
    if not execute:
        jobs = [j for j in jobs if _path(j[0], j[1], j[2]).exists()]
    if not jobs:
        return 0
    n = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(one, t, v, o, order_label=lbl, force=force): (t, v, o, lbl)
                for (t, v, o, lbl) in jobs}
        for fut in as_completed(futs):
            t, v, o, lbl = futs[fut]
            try:
                rec = fut.result()
            except Exception as exc:                           # noqa: BLE001
                rec = env_adapter.normalize_record(
                    {"harness_error": f"{type(exc).__name__}: {exc}", "ok": False},
                    task_id=t, variant=v, rep=o, preceding_task=None,
                    env_layer="pp.task_environments")
                rec["order_label"] = lbl
                rec["observation_index"] = o
            out.setdefault(t, {}).setdefault(v, []).append(rec)
            if not rec.get("reused_from_cache"):
                n += 1
            if budget_seconds and (time.time() - t_start) > budget_seconds:
                _log(f"  [budget] {budget_seconds}s exceeded; draining")
    return n
