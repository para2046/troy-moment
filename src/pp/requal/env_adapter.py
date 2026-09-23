"""Thin adapter over ``agent-env``'s period-correct per-task environment layer.

``agent-env`` owns the environment layer entirely (``scripts/build_task_environments.py``,
``data/frozen/task_environments.json``, ``reports/PER_TASK_ENVIRONMENTS.md``,
``src/pp/runtimes/impossible/**``). This module **discovers** that interface and
calls it. It deliberately contains no environment construction of its own.

Contract expected (per the orchestrator brief; adapted at runtime to whatever is
actually published):

    build_env(task_id)                  -> env/image identity record
    run_canonical(task_id, variant=..., ...) -> record with keys such as
        env_id / image, python_version, install_result, patch_apply_result,
        argv, cwd, stdout, stderr, returncode, semantic_failure_type

HARD RULE: if the period-correct layer is not available for a task, this adapter
returns ``PENDING_ENVIRONMENT_BUILD``. It **never** falls back to the shared warm
container -- that container is the defect being removed, and evidence produced in
it is order-dependent by construction.
"""
from __future__ import annotations

import importlib
import inspect
import json
import re
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

REPO = Path(__file__).resolve().parents[3]

#: Where agent-env publishes its programmatic interface + record schema.
ENV_REPORT = REPO / "reports" / "PER_TASK_ENVIRONMENTS.md"
ENV_SPEC_REPORT = REPO / "reports" / "PER_TASK_ENVIRONMENT_SPEC.md"
ENV_RECORDS = REPO / "data" / "frozen" / "task_environments.json"
ENV_VALIDATION_DIR = REPO / "results" / "study_env_validation"

#: Candidate module paths, in preference order. The first one exposing a
#: callable ``run_canonical`` wins.
CANDIDATE_MODULES: List[str] = [
    "pp.task_environments",          # <- the interface agent-env published
    "pp.runtimes.impossible.per_task_env",
    "pp.runtimes.impossible.task_environments",
    "pp.runtimes.impossible.environments",
    "pp.runtimes.impossible.env_builder",
    "pp.runtimes.impossible.period_correct",
    "pp.runtimes.impossible",
]

#: Module-level function names we will accept for each role.
_BUILD_NAMES = ("build_env", "build_task_env", "build_environment", "build")
_RUN_NAMES = ("run_canonical", "run_canonical_replay", "run_canonical_probe",
              "canonical_run", "run")

PENDING = "PENDING_ENVIRONMENT_BUILD"

#: Modules that MUST NOT be used as the environment layer: the shared warm
#: container path. Using it would reintroduce the order dependence.
FORBIDDEN_FALLBACKS = (
    "pp.runtimes.impossible.modal_app.ImpossibleSweRuntime",
    "pp.runtimes.impossible.env.SweBenchModalEnv",
)


class EnvLayerUnavailable(RuntimeError):
    """The period-correct per-task environment layer is not published yet."""


class ForbiddenFallbackUsed(RuntimeError):
    """A FORBIDDEN_FALLBACKS component was used where it is not permitted.

    ``measurement_status`` is carried so a caller records a MEASUREMENT_FAILURE
    rather than a scientific outcome: an episode that ran on the forbidden
    substrate produced no valid measurement at all.
    """

    measurement_status = "MEASUREMENT_FAILURE"


# ---------------------------------------------------------------------------
# FORBIDDEN_FALLBACKS enforcement -- for the FORMAL path, not only requal
# ---------------------------------------------------------------------------
# The requalification path honoured ``FORBIDDEN_FALLBACKS`` by never discovering
# the warm container. The FORMAL path *was* the forbidden fallback
# (``reports/EXECUTION_EQUIVALENCE_V2.md`` §3), because nothing ever checked it.
# These two helpers are that check, and they are called from
# ``studies.study_ab_runner.ModalEnvProvider.build``.
def qualname_of(obj: Any) -> str:
    """``module.QualName`` for a class, an instance, or a dotted string."""
    if isinstance(obj, str):
        return obj
    cls = obj if isinstance(obj, type) else type(obj)
    return f"{cls.__module__}.{cls.__qualname__}"


def forbidden_fallback_match(obj: Any) -> Optional[str]:
    """The FORBIDDEN_FALLBACKS entry ``obj`` matches, if any.

    Matches on the class path AND on the declared ``execution_substrate``, so an
    env that merely *drives* the warm container is caught even if its own class
    name is new.
    """
    names = {qualname_of(obj)}
    subst = getattr(obj, "execution_substrate", "")
    if isinstance(subst, str) and subst:
        names.add(subst)
    for forbidden in FORBIDDEN_FALLBACKS:
        leaf = forbidden.rsplit(".", 1)[-1]
        for name in names:
            if name == forbidden or name.rsplit(".", 1)[-1] == leaf:
                return forbidden
    return None


def enforce_formal_substrate(env: Any, *, context: str = "") -> Dict[str, Any]:
    """Raise unless ``env`` is a legitimate FORMAL execution substrate.

    Four independent conditions, every one of which the pre-fix wiring failed:

    1. the env's class / declared substrate is not in ``FORBIDDEN_FALLBACKS``;
    2. it does NOT drive a shared warm runtime;
    3. it DOES open a per-execution sandbox;
    4. it declares itself the formal substrate.

    Returns the facts it checked (for recording); raises
    :class:`ForbiddenFallbackUsed` otherwise. There is no permissive mode and no
    fallback: a substrate that fails this check is a MEASUREMENT_FAILURE, never a
    run that is scored anyway.
    """
    facts = {
        "context": context,
        "env_class": qualname_of(env),
        "execution_substrate": getattr(env, "execution_substrate", ""),
        "env_drives_shared_runtime": bool(
            getattr(env, "env_drives_shared_runtime", True)),
        "env_opens_per_task_sandbox": bool(
            getattr(env, "env_opens_per_task_sandbox", False)),
        "is_formal_substrate": bool(getattr(env, "is_formal_substrate", False)),
        "forbidden_fallback_match": forbidden_fallback_match(env),
    }
    problems: List[str] = []
    if facts["forbidden_fallback_match"]:
        problems.append(
            f"{facts['env_class']} matches FORBIDDEN_FALLBACKS entry "
            f"{facts['forbidden_fallback_match']!r}")
    if facts["env_drives_shared_runtime"]:
        problems.append("env drives a SHARED WARM runtime "
                        "(env_drives_shared_runtime=True)")
    if not facts["env_opens_per_task_sandbox"]:
        problems.append("env does not open a per-execution sandbox "
                        "(env_opens_per_task_sandbox=False)")
    if not facts["is_formal_substrate"]:
        problems.append("env does not declare itself the formal substrate")
    facts["ok"] = not problems
    facts["problems"] = problems
    if problems:
        raise ForbiddenFallbackUsed(
            "refusing to run the FORMAL path on this substrate"
            + (f" [{context}]" if context else "") + ": " + "; ".join(problems)
            + ". The formal Impossible substrate is "
            "pp.runtimes.impossible.env.SweBenchPerTaskSandboxEnv (fresh "
            "per-execution modal.Sandbox from the benchmark's own per-instance "
            "image, evaluated with the repo's own test_cmd). Evidence produced "
            "in the shared warm container is order-dependent by construction.")
    return facts


# ---------------------------------------------------------------------------
# discovery
# ---------------------------------------------------------------------------
def _module_path_from_report() -> Optional[str]:
    """Read the module path agent-env documents, if the report exists."""
    for report in (ENV_REPORT, ENV_SPEC_REPORT):
        if not report.exists():
            continue
        text = report.read_text(encoding="utf-8", errors="replace")
        # `from pp.x.y import build_env, run_canonical`
        m = re.search(r"from\s+([\w.]+)\s+import\s+[^\n]*run_canonical", text)
        if m:
            return m.group(1)
        m = re.search(r"import\s+([\w.]+)[^\n]*\n[^\n]*run_canonical", text)
        if m:
            return m.group(1)
        m = re.search(r"`([\w.]+)\.run_canonical\(", text)
        if m:
            return m.group(1)
    return None


def _pick(mod, names) -> Optional[Callable]:
    for n in names:
        fn = getattr(mod, n, None)
        if callable(fn):
            return fn
    return None


def discover() -> Dict[str, Any]:
    """Locate the published env layer. Never raises."""
    tried: List[Dict[str, str]] = []
    candidates = []
    documented = _module_path_from_report()
    if documented:
        candidates.append(documented)
    candidates += [m for m in CANDIDATE_MODULES if m != documented]

    for path in candidates:
        try:
            mod = importlib.import_module(path)
        except Exception as exc:                               # noqa: BLE001
            tried.append({"module": path, "error": f"{type(exc).__name__}: {exc}"})
            continue
        run = _pick(mod, _RUN_NAMES)
        build = _pick(mod, _BUILD_NAMES)
        if run is None:
            tried.append({"module": path, "error": "no run_canonical-like callable"})
            continue
        return {
            "available": True,
            "module": path,
            "documented_module": documented,
            "run_canonical": run,
            "run_canonical_name": run.__name__,
            "run_canonical_signature": str(inspect.signature(run)),
            "build_env": build,
            "build_env_name": getattr(build, "__name__", None),
            "tried": tried,
        }
    return {
        "available": False,
        "module": None,
        "documented_module": documented,
        "run_canonical": None,
        "build_env": None,
        "tried": tried,
        "report_present": ENV_REPORT.exists(),
        "records_present": ENV_RECORDS.exists(),
    }


def env_records() -> Dict[str, Any]:
    """agent-env's frozen per-task environment records, if published."""
    if not ENV_RECORDS.exists():
        return {}
    try:
        doc = json.loads(ENV_RECORDS.read_text(encoding="utf-8"))
    except Exception:                                          # noqa: BLE001
        return {}
    # Accept either {task_id: record} or {"environments": [ {task_id: ...} ]}.
    if isinstance(doc, dict):
        for key in ("environments", "tasks", "records", "task_environments"):
            v = doc.get(key)
            if isinstance(v, dict):
                return v
            if isinstance(v, list):
                out = {}
                for r in v:
                    tid = r.get("task_id") or r.get("source_task") or r.get("instance_id")
                    if tid:
                        out[_bare(tid)] = r
                return out
        if all(isinstance(v, dict) for v in doc.values()):
            return {_bare(k): v for k, v in doc.items()}
    return {}


def _bare(task_id: str) -> str:
    """``impossible::astropy__astropy-12907`` -> ``astropy__astropy-12907``."""
    return task_id.split("::")[-1]


def supported_task_ids() -> List[str]:
    """Task ids for which a period-correct environment is actually available."""
    d = discover()
    ids = set()
    recs = env_records()
    for tid, rec in recs.items():
        # Only count an environment that agent-env reports as usable.
        ok = rec.get("build_ok", rec.get("ok", rec.get("built", True)))
        if ok is not False:
            ids.add(_bare(tid))
    if d["available"]:
        lister = None
        for n in ("supported_task_ids", "available_task_ids", "task_ids"):
            lister = getattr(importlib.import_module(d["module"]), n, None)
            if callable(lister):
                break
            lister = None
        if lister is not None:
            try:
                ids.update(_bare(t) for t in lister())
            except Exception:                                  # noqa: BLE001
                pass
    return sorted(ids)


# ---------------------------------------------------------------------------
# invocation
# ---------------------------------------------------------------------------
def _call_flexible(fn: Callable, task_id: str, **kwargs) -> Any:
    """Call ``fn(task_id, ...)`` passing only kwargs its signature accepts."""
    try:
        sig = inspect.signature(fn)
        accepted = set(sig.parameters)
        has_var_kw = any(p.kind is inspect.Parameter.VAR_KEYWORD
                         for p in sig.parameters.values())
    except (TypeError, ValueError):
        accepted, has_var_kw = set(), True
    passed = kwargs if has_var_kw else {k: v for k, v in kwargs.items()
                                        if k in accepted}
    return fn(task_id, **passed)


def run_canonical(task_id: str, *, variant: str, rep: int = 1,
                  preceding_task: Optional[str] = None,
                  **extra) -> Dict[str, Any]:
    """One period-correct canonical replay. Returns a normalized record.

    ``preceding_task`` is forwarded when the env layer accepts it, so that
    order-swapped replays can be requested; with truly per-task environments it
    is expected to be a no-op, which is itself the thing being validated.
    """
    d = discover()
    if not d["available"]:
        raise EnvLayerUnavailable(
            "period-correct per-task environment layer not published yet "
            f"(tried: {[t['module'] for t in d['tried']]}); refusing to fall "
            "back to the shared warm container")
    raw = _call_flexible(d["run_canonical"], _bare(task_id), variant=variant,
                         rep=rep, replay=rep, preceding_task=preceding_task,
                         study="A", session=extra.pop("session", "q2_requal"),
                         **extra)
    return normalize_record(raw, task_id=_bare(task_id), variant=variant,
                            rep=rep, preceding_task=preceding_task,
                            env_layer=d["module"])


def build_env(task_id: str, **extra) -> Dict[str, Any]:
    d = discover()
    if not d["available"] or d["build_env"] is None:
        raise EnvLayerUnavailable("build_env not published yet")
    return _call_flexible(d["build_env"], _bare(task_id), **extra)


# ---------------------------------------------------------------------------
# record normalization
# ---------------------------------------------------------------------------
_ALIASES = {
    "returncode": ("returncode", "rc", "exit_code", "returncode_pytest"),
    "stdout": ("stdout", "stdout_tail", "out"),
    "stderr": ("stderr", "stderr_tail", "err"),
    "summary": ("summary", "evaluator_summary", "pytest_summary"),
    "argv": ("argv", "cmd", "command", "args"),
    "cwd": ("cwd", "workdir", "working_dir"),
    "env_id": ("env_id", "image", "image_tag", "image_id", "environment_id",
               "env_identity", "env"),
    "python_version": ("python_version", "python", "py_version", "interpreter"),
    "install_result": ("install_result", "pip_install_result", "install",
                       "pip_install_e_ok", "install_ok"),
    "patch_apply_result": ("patch_apply_result", "patch_result",
                           "test_patch_applied", "patch_apply"),
    "tests_pass": ("tests_pass", "passed", "tests_passed"),
    "harness_error": ("harness_error", "error", "exception"),
    "timed_out": ("timed_out", "timeout"),
    "semantic_failure_type": ("semantic_failure_type", "semantic_type",
                              "model_visible_category"),
    "observation": ("observation", "obs"),
    "toolchain": ("toolchain", "toolchain_versions", "pins", "constraints"),
}


def _get(raw: Dict[str, Any], key: str, default=None):
    for a in _ALIASES.get(key, (key,)):
        if a in raw:
            return raw[a]
    return default


#: Keys in an env-layer record that carry a PRIOR SELECTION STATUS. agent-env
#: records ``study_a_status`` / ``prior_status`` for its own bookkeeping; they
#: must not be reachable from our decision path, so they are stripped here --
#: before the record ever enters the evidence structure -- and the strip is
#: recorded so nothing is silently dropped.
PRIOR_STATUS_KEYS_IN_ENV_RECORDS = (
    "study_a_status", "study_b_status", "prior_status", "status",
    "shortlist_bucket", "tier", "selection_status",
)


def _sanitize(raw: Any) -> Any:
    """Deep-copy an env-layer record with prior-status fields removed."""
    if isinstance(raw, dict):
        return {k: _sanitize(v) for k, v in raw.items()
                if k not in PRIOR_STATUS_KEYS_IN_ENV_RECORDS}
    if isinstance(raw, list):
        return [_sanitize(v) for v in raw]
    if isinstance(raw, str) and raw.strip().lower() in (
            "primary", "reserve", "exclude"):
        return "<prior-status value stripped: unreachable from the decision path>"
    return raw


#: Cap on the stdout/stderr text retained per replay in our artifact. The full
#: text lives in agent-env's own per-execution logs; we keep a generous tail so
#: the evidence is auditable without duplicating ~100 KB per replay.
STREAM_TAIL = 4000


def normalize_record(raw: Any, *, task_id: str, variant: str, rep: int,
                     preceding_task: Optional[str],
                     env_layer: Optional[str],
                     source: str = "run_canonical") -> Dict[str, Any]:
    """Normalize an env-layer record into the replay record we store.

    Classification is done on the **model-visible observation** -- the string
    ``SweBenchModalEnv.step`` actually shows the agent -- not on stdout. stdout /
    stderr / argv / cwd / rc are recorded for diagnosis.
    """
    from .taxonomy import classify_observation

    if not isinstance(raw, dict):
        raw = {"value": raw}
    raw = _sanitize(raw)

    rc = _get(raw, "returncode")
    stdout = _get(raw, "stdout") or ""
    stderr = _get(raw, "stderr") or ""
    summary = _get(raw, "summary")
    mvo = raw.get("model_visible_observation") or _get(raw, "observation")
    tests_pass = _get(raw, "tests_pass")
    if tests_pass is None:
        tests_pass = raw.get("passed")
    cls = classify_observation(
        returncode=rc,
        summary=summary,
        stdout=stdout,
        stderr=stderr,
        tests_pass=tests_pass,
        harness_error=_get(raw, "harness_error") or raw.get("error"),
        timed_out=bool(_get(raw, "timed_out", False)),
        runner_invoked=bool(raw.get("ok", True)) and not raw.get("error"),
        model_visible_observation=mvo,
    )
    declared = (raw.get("semantic_failure_type")
                or raw.get("model_visible_semantic_type"))
    env_obj = raw.get("environment") or {}
    tc = raw.get("toolchain") or {}
    return {
        "task_id": task_id,
        "variant": variant,
        "rep": rep,
        "session": raw.get("session"),
        "preceding_task_in_env": preceding_task,
        "env_layer_module": env_layer,
        "record_source": source,
        # ---- environment / toolchain identity -----------------------------
        "env_id": env_obj.get("image") or _get(raw, "env_id"),
        "env_spec_source": env_obj.get("spec_source"),
        "swebench_version": env_obj.get("swebench_version"),
        "base_commit": env_obj.get("base_commit"),
        "python_version": (tc.get("python_observed")
                           or env_obj.get("python_declared")
                           or _get(raw, "python_version")),
        "python_declared": env_obj.get("python_declared"),
        "toolchain": tc or _get(raw, "toolchain"),
        "install_result": raw.get("install_result") or _get(raw, "install_result"),
        "patch_apply_result": raw.get("patch_apply") or _get(raw, "patch_apply_result"),
        # ---- per-replay execution record (required by the brief) ----------
        "argv": _get(raw, "argv"),
        "cwd": _get(raw, "cwd"),
        "test_command": raw.get("test_command"),
        "test_directives_normalized": raw.get("test_directives"),
        "stdout": stdout[-STREAM_TAIL:],
        "stderr": stderr[-STREAM_TAIL:],
        "stdout_len": len(stdout),
        "stderr_len": len(stderr),
        "summary": summary,
        "model_visible_observation": mvo,
        "rc": rc,                                  # DIAGNOSTIC ONLY
        "rc_is_diagnostic_only": True,
        # ---- semantic classification (ours, independently recomputed) -----
        "semantic_failure_type": cls["semantic_failure_type"],
        "semantic_failure_type_declared_by_env_layer": declared,
        "semantic_type_agrees_with_env_layer": (
            None if declared is None else declared == cls["semantic_failure_type"]),
        "evaluator_semantic_type_declared_by_env_layer": raw.get(
            "evaluator_semantic_type"),
        "target_statuses": raw.get("target_statuses"),
        "collection_result": raw.get("collection_result"),
        "measurement_failure": cls["measurement_failure"],
        "substantive_failure_evidence": cls["substantive_failure_evidence"],
        "substantive_failure_evidence_inclusive": cls[
            "substantive_failure_evidence_inclusive"],
        "matched_evidence": cls["matched_evidence"],
        "prior_status_stripped_from_env_record": True,
    }


# ---------------------------------------------------------------------------
# harvest: consume agent-env's already-published per-execution records
# ---------------------------------------------------------------------------
def harvest_published(study: str = "A") -> Dict[str, Dict[str, List[Dict[str, Any]]]]:
    """Read agent-env's per-execution records for ``study`` (READ-ONLY).

    Returns ``{task_id: {variant: [normalized records]}}``. Reusing agent-env's
    executions instead of re-running them avoids duplicated Modal spend and
    keeps a single source of execution truth. ``results/study_env_validation/**``
    is owned by agent-env and is never written here.

    Study-A and Study-B records are NEVER pooled: only ``study`` is read.
    """
    out: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
    root = ENV_VALIDATION_DIR / study
    if not root.is_dir():
        return out
    for task_dir in sorted(root.iterdir()):
        if not task_dir.is_dir():
            continue
        for f in sorted(task_dir.glob("*.json")):
            if f.name == "assessment.json":
                continue
            try:
                raw = json.loads(f.read_text(encoding="utf-8"))
            except Exception:                                  # noqa: BLE001
                continue
            if not isinstance(raw, dict) or raw.get("study") != study:
                continue
            variant = raw.get("variant") or "conflicting"
            rec = normalize_record(
                raw, task_id=_bare(raw.get("instance_id") or task_dir.name),
                variant=variant, rep=raw.get("replay"),
                preceding_task=None, env_layer="pp.task_environments",
                source=f"harvested:{f.relative_to(REPO).as_posix()}")
            rec["log"] = f.relative_to(REPO).as_posix()
            out.setdefault(_bare(task_dir.name), {}).setdefault(
                variant, []).append(rec)
    return out


def reclassify(rec: Dict[str, Any]) -> Dict[str, Any]:
    """Re-run OUR classifier over a cached replay record.

    Cached records are re-classified rather than trusted, so that a correction
    to :mod:`pp.requal.taxonomy` is applied to every observation without
    re-executing (and re-paying for) a single Modal sandbox. The recorded
    ``argv``/``cwd``/``stdout``/``stderr``/``rc`` are untouched.
    """
    from .taxonomy import classify_observation

    if rec.get("record_status") == PENDING:
        return rec
    cls = classify_observation(
        returncode=rec.get("rc"),
        summary=rec.get("summary"),
        stdout=rec.get("stdout"),
        stderr=rec.get("stderr"),
        tests_pass=None,
        harness_error=None,
        timed_out=False,
        runner_invoked=True,
        model_visible_observation=rec.get("model_visible_observation"),
    )
    declared = rec.get("semantic_failure_type_declared_by_env_layer")
    rec["semantic_failure_type"] = cls["semantic_failure_type"]
    rec["measurement_failure"] = cls["measurement_failure"]
    rec["substantive_failure_evidence"] = cls["substantive_failure_evidence"]
    rec["substantive_failure_evidence_inclusive"] = cls[
        "substantive_failure_evidence_inclusive"]
    rec["matched_evidence"] = cls["matched_evidence"]
    rec["semantic_type_agrees_with_env_layer"] = (
        None if declared is None else declared == cls["semantic_failure_type"])
    rec["reclassified"] = True
    return rec


def pending_record(task_id: str, variant: str, why: str) -> Dict[str, Any]:
    """A placeholder for a candidate whose environment is not built yet.

    Deliberately NOT a semantic failure type: an unbuilt environment produced no
    observation at all, so it can neither satisfy nor refute any criterion.
    """
    return {
        "task_id": _bare(task_id),
        "variant": variant,
        "rep": None,
        "record_status": PENDING,
        "why": why,
        "semantic_failure_type": None,
        "measurement_failure": None,
        "substantive_failure_evidence": False,
        "rc": None,
        "rc_is_diagnostic_only": True,
    }
