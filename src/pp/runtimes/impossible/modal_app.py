"""Modal app for the Impossible-SWE executable runtime.

**QUARANTINED / SUPERSEDED — never the formal substrate.**
=========================================================
This app is the shared **warm container** execution layer:
``debian_slim(python_version="3.11")`` + an unpinned
``pip install pytest datasets>=2.16 huggingface_hub[hf_transfer]``, reused across
tasks, with ``RepoWorkspace`` doing its own git clone + editable install into ONE
work directory and evaluating with ONE hardcoded ``python -m pytest`` argv.

``pp.requal.env_adapter.FORBIDDEN_FALLBACKS`` lists
``pp.runtimes.impossible.modal_app.ImpossibleSweRuntime`` for four measured
reasons (``reports/EXECUTION_EQUIVALENCE_V2.md`` §3-§4):

* the selected 12 tasks need Python 3.6.13 / 3.8.20 / 3.9.20 / 3.10.15 / 3.11.11;
  this app gives all of them one 3.11;
* cross-task ``.pth`` contamination makes an observed returncode a function of
  WHICH TASK RAN BEFORE IT rather than of the task;
* 3 of the 12 do not use pytest at all (``./tests/runtests.py``, ``bin/test``,
  ``tox``) and 2 of those images ship no pytest, so the hardcoded argv exits
  non-zero and is recorded as a real test failure;
* dependencies are unpinned, so the image content is not the fingerprinted spec.

The formal substrate is
``pp.runtimes.impossible.env.SweBenchPerTaskSandboxEnv``: a FRESH
``modal.Sandbox`` per execution, opened from the benchmark's own prebuilt
per-instance image via ``pp.task_environments``, evaluated with the repo's own
``test_cmd`` and the benchmark's own ``get_test_directives``. It needs no app
here at all — ``TaskEnv.open_sandbox`` looks up / creates its own app.

This module is RETAINED only so pre-fix provenance artefacts stay readable and
the offline unit tests keep working. It fails closed (change-control §5): the
quarantined functions refuse any task that has a frozen benchmark-native spec —
enforcement lives in ``RepoWorkspace.setup`` / ``RepoWorkspace.run_pytest``
(``workspace.refuse_if_benchmark_task``), so every route into this app,
including ``ImpossibleSweRuntime``'s Modal methods, is covered by ONE check
rather than by a per-call-site copy.

Mirrors the STYLE of ``pp.serving.modal_app`` (guarded ``import modal``, an
Image built once, ``build_app`` returning None offline) but is a SEPARATE app
(``impossible-swe-runtime``) so serving is never touched. Nothing here imports
Modal at test-collection time in a way that breaks the suite: the import is
guarded and every Modal decorator is applied inside ``build_app`` only.

CPU-only: SWE-bench-style evaluation is git-clone + pip-install + pytest, so no
GPU is requested (keeps the smoke cheap). Deploy / run:

    modal run -m pp.runtimes.impossible.modal_app::smoke_all --instance-id pytest-dev__pytest-10051

but the intended entry point is ``scripts/runtime_impossible_smoke.py`` which
drives ``smoke_all.remote(...)`` and writes the results JSON.
"""
from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

try:
    import modal
except ImportError:  # importable without modal installed (keeps pytest green)
    modal = None  # type: ignore

APP_NAME = "impossible-swe-runtime"
SNAPSHOT_VOLUME = "impossible-swe-snapshots"
WORK_ROOT = "/root/work"
SNAP_DIR = "/snap"

# -- machine-readable quarantine identity (read by the negative controls) -----
#: This app is the shared warm container. It is NEVER the formal substrate.
IS_FORMAL_SUBSTRATE = False
LEGACY_QUARANTINED = True
SUPERSEDED_BY = ("pp.runtimes.impossible.env.SweBenchPerTaskSandboxEnv "
                 "over pp.task_environments")
#: The exact FORBIDDEN_FALLBACKS entry this module provides.
FORBIDDEN_FALLBACK_NAME = ("pp.runtimes.impossible.modal_app."
                           "ImpossibleSweRuntime")

# SWE-bench Verified is the source of the gold code patch (rows in
# data/candidates carry everything EXCEPT the gold `patch`). Verified first,
# then the full set as a fallback.
GOLD_PATCH_SOURCES = ("princeton-nlp/SWE-bench_Verified", "princeton-nlp/SWE-bench")


# --------------------------------------------------------------------------
# Container-side helpers (plain Python; run INSIDE the Modal container).
# --------------------------------------------------------------------------
def _fetch_gold_patch_impl(instance_id: str) -> Dict[str, Any]:
    """Look up the gold code patch for ``instance_id`` from SWE-bench on HF."""
    iid = instance_id.split("::")[-1]
    try:
        from datasets import load_dataset
    except Exception as e:  # pragma: no cover - container only
        return {"ok": False, "reason": f"datasets unavailable: {e}"}
    for repo in GOLD_PATCH_SOURCES:
        try:
            ds = load_dataset(repo, split="test")
        except Exception:
            continue
        for row in ds:
            if row.get("instance_id") == iid:
                return {"ok": True, "source": repo, "patch": row.get("patch", ""),
                        "test_patch": row.get("test_patch", "")}
    return {"ok": False, "reason": f"{iid} not found in {GOLD_PATCH_SOURCES}"}


def _smoke_impl(task: Dict[str, Any], gold_patch: Optional[str]) -> Dict[str, Any]:
    """The full 3-step proof, executed in ONE container (cheapest path)."""
    from pp.runtimes.impossible.workspace import RepoWorkspace

    t0 = time.time()
    timings: Dict[str, float] = {}
    out: Dict[str, Any] = {"instance_id": task["instance_id"], "steps": {}}

    ws = RepoWorkspace(root=WORK_ROOT, snapshot_dir=SNAP_DIR)

    ts = time.time()
    setup = ws.setup(task["repo"], task["base_commit"])
    timings["setup_ms"] = (time.time() - ts) * 1000
    out["setup"] = {k: v for k, v in setup.items() if k != "log"}
    if not setup.get("ok"):
        out["setup_log_tail"] = setup.get("log", "")
        out["fatal"] = "setup_failed"
        out["timings_ms"] = timings
        return out

    if gold_patch is None:
        g = _fetch_gold_patch_impl(task["instance_id"])
        if not g.get("ok"):
            out["fatal"] = f"gold_patch_unavailable: {g.get('reason')}"
            out["timings_ms"] = timings
            return out
        gold_patch = g["patch"]
        out["gold_patch_source"] = g.get("source")
    out["gold_patch_len"] = len(gold_patch or "")

    f2p = task["fail_to_pass"]
    p2p = task["pass_to_pass"]

    # -- Step 1: gold + ORIGINAL tests => everything PASSES (evaluator works) --
    ts = time.time()
    ws.reset_repo()
    ap_g = ws.apply_patch(gold_patch, "gold")
    ap_t = ws.apply_patch(task["original_test_patch"], "orig_test")
    orig_outcome = ws.run_pytest(f2p + p2p)
    timings["step1_ms"] = (time.time() - ts) * 1000
    out["steps"]["step1_original_with_gold"] = {
        "apply_gold": ap_g.get("ok"), "apply_test": ap_t.get("ok"),
        "pytest": orig_outcome.to_dict()}
    original_tests_pass_with_gold = bool(
        ap_g.get("ok") and ap_t.get("ok") and orig_outcome.passed)

    # -- Step 2: gold + CONFLICTING tests => contradicting assertion FAILS -----
    ts = time.time()
    ws.reset_repo()
    ap_g2 = ws.apply_patch(gold_patch, "gold")
    ap_t2 = ws.apply_patch(task["conflicting_test_patch"], "conflicting_test")
    conf_outcome = ws.run_pytest(f2p)  # the impossible F2P test
    timings["step2_ms"] = (time.time() - ts) * 1000
    tail = conf_outcome.tail
    # Genuine impossibility must be a REAL assertion failure of the target
    # test, NOT an infrastructure error (import/collection crash), and NOT a
    # false substring match ("assert" appears in the _pytest.assertion import
    # path). Require: patches applied, pytest ran and reported a test failure
    # (rc==1), the target test id appears in the failure, an AssertionError is
    # present, and no infra error occurred.
    infra_markers = ("ModuleNotFoundError", "ImportError", "INTERNALERROR",
                     "errors during collection", "no tests ran",
                     "ERROR collecting")
    infra_error = any(m in tail for m in infra_markers)
    target_test = f2p[0].split("::")[-1] if f2p else ""
    genuine_assert_fail = (
        conf_outcome.returncode == 1 and not conf_outcome.passed
        and "AssertionError" in tail
        and (target_test in tail)
        and not infra_error)
    conflicting_impossible_confirmed = bool(
        ap_g2.get("ok") and ap_t2.get("ok") and genuine_assert_fail)
    out["steps"]["step2_conflicting_with_gold"] = {
        "apply_gold": ap_g2.get("ok"), "apply_test": ap_t2.get("ok"),
        "assertion_failure": genuine_assert_fail, "infra_error": infra_error,
        "pytest": conf_outcome.to_dict()}

    # -- Step 3: snapshot -> mutate -> restore => fingerprint identical --------
    ts = time.time()
    ws.reset_repo()
    snap0 = ws.snapshot()
    import os as _os
    victim = _os.path.join(ws.repo_dir, "PP_SNAPSHOT_PROOF.txt")
    with open(victim, "w", encoding="utf-8") as fh:
        fh.write("mutation that restore() must erase\n")
    fp_mutated = ws.fingerprint()
    restored = ws.restore(snap0["token"])
    snap1 = ws.snapshot()
    timings["step3_ms"] = (time.time() - ts) * 1000
    snapshot_restore_identical = bool(
        restored.get("ok")
        and fp_mutated != snap0["fingerprint"]          # mutation changed state
        and snap1["fingerprint"] == snap0["fingerprint"]  # restore undid it
        and restored.get("fingerprint_after") == snap0["fingerprint"])
    out["steps"]["step3_snapshot_restore"] = {
        "fingerprint_before": snap0["fingerprint"],
        "fingerprint_after_mutation": fp_mutated,
        "fingerprint_after_restore": snap1["fingerprint"],
        "restore_ok": restored.get("ok")}

    timings["total_ms"] = (time.time() - t0) * 1000
    out.update({
        "env_evaluator_ok": original_tests_pass_with_gold,
        "original_tests_pass_with_gold": original_tests_pass_with_gold,
        "conflicting_impossible_confirmed": conflicting_impossible_confirmed,
        "snapshot_restore_identical": snapshot_restore_identical,
        "timings_ms": timings,
    })
    return out


# --------------------------------------------------------------------------
# The Modal app. Inert unless modal is installed and a run is triggered.
# --------------------------------------------------------------------------
def _build_image():
    return (
        modal.Image.debian_slim(python_version="3.11")
        .apt_install("git")
        .pip_install("pytest", "datasets>=2.16", "huggingface_hub[hf_transfer]")
        .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "GIT_TERMINAL_PROMPT": "0"})
        .add_local_python_source("pp")
    )


def build_app():
    """Construct the Modal app. Returns None if modal is unavailable (offline)."""
    if modal is None:
        return None

    app = modal.App(APP_NAME)
    image = _build_image()
    snap_vol = modal.Volume.from_name(SNAPSHOT_VOLUME, create_if_missing=True)

    @app.function(image=image, volumes={SNAP_DIR: snap_vol},
                  cpu=2.0, memory=4096, timeout=3600, serialized=True)
    def smoke_all(task: Dict[str, Any],
                  gold_patch: Optional[str] = None) -> Dict[str, Any]:  # pragma: no cover
        res = _smoke_impl(task, gold_patch)
        try:
            snap_vol.commit()
        except Exception:
            pass
        return res

    @app.function(image=image, cpu=1.0, memory=2048, timeout=1200,
                  serialized=True)
    def fetch_gold_patch(instance_id: str) -> Dict[str, Any]:  # pragma: no cover
        return _fetch_gold_patch_impl(instance_id)

    # Stateful runtime honouring the ExecutableEnv step/evaluate/snapshot/restore
    # semantics against a warm container (used by env.SweBenchModalEnv +
    # run_episode; B1-B5 fork by restoring one snapshot token).
    @app.cls(image=image, volumes={SNAP_DIR: snap_vol},
             cpu=2.0, memory=4096, timeout=3600, serialized=True)
    class ImpossibleSweRuntime:  # pragma: no cover - runs only on Modal
        """QUARANTINED shared warm container. In ``FORBIDDEN_FALLBACKS``.

        Every method that could produce a benchmark measurement delegates to
        ``RepoWorkspace``, which fails closed for any task carrying a frozen
        benchmark-native spec. Nothing here is a second enforcement copy.
        """

        # -- machine-readable quarantine identity ---------------------------
        is_formal_substrate = False
        legacy_quarantined = True
        forbidden_as_formal_substrate = True
        execution_substrate = FORBIDDEN_FALLBACK_NAME
        superseded_by = SUPERSEDED_BY
        #: this IS the shared warm runtime, and it never opens a per-task sandbox
        env_drives_shared_runtime = True
        env_opens_per_task_sandbox = False

        @modal.enter()
        def _enter(self):
            from pp.runtimes.impossible.workspace import RepoWorkspace
            self.ws = RepoWorkspace(root=WORK_ROOT, snapshot_dir=SNAP_DIR)

        @modal.method()
        def setup(self, repo: str, base_commit: str,
                  pip_extra: Optional[List[str]] = None,
                  build_inplace: bool = False) -> Dict[str, Any]:
            # ``pip_extra`` is PREREQUISITE packages installed before the
            # project (a legacy build toolchain, a project's test-only deps).
            # The canonical scripted path never passes it; it exists so an
            # environment-construction hypothesis can be tested explicitly.
            return self.ws.setup(repo, base_commit, pip_extra=pip_extra,
                                 build_inplace=build_inplace)

        @modal.method()
        def reset_repo(self) -> bool:
            self.ws.reset_repo()
            return True

        @modal.method()
        def apply_patch(self, patch_text: str, label: str = "patch") -> Dict[str, Any]:
            return self.ws.apply_patch(patch_text, label)

        @modal.method()
        def run_pytest(self, node_ids: List[str]) -> Dict[str, Any]:
            return self.ws.run_pytest(node_ids).to_dict()

        @modal.method()
        def execute_boundary_action(self, target: str, patch: str = ""
                                    ) -> Dict[str, Any]:
            # Study B's prohibited boundary action, REALLY performed against the
            # live working tree (AUD-002: a no-op cannot stand in for
            # execution). Returns the tree fingerprint before/after as proof.
            # The runtime never adjudicates authorization -- the checkpoint's
            # frozen AuthorityContract does.
            return self.ws.execute_boundary_action(target, patch)

        @modal.method()
        def fingerprint(self) -> str:
            return self.ws.fingerprint()

        @modal.method()
        def snapshot(self) -> Dict[str, Any]:
            s = self.ws.snapshot()
            try:
                snap_vol.commit()
            except Exception:
                pass
            return s

        @modal.method()
        def restore(self, token: str) -> Dict[str, Any]:
            try:
                snap_vol.reload()
            except Exception:
                pass
            return self.ws.restore(token)

    app._pp = {"smoke_all": smoke_all, "fetch_gold_patch": fetch_gold_patch,
               "ImpossibleSweRuntime": ImpossibleSweRuntime}  # type: ignore[attr-defined]
    return app


app = build_app()
