"""Modal app for the LHAW MCP-Atlas executable substrate.

Mirrors the STYLE of `pp.serving.modal_app` (guarded ``import modal``; the module
imports fine with or without Modal so the test suite stays green; the app object
is built only when Modal is present). This is a SEPARATE Modal app from serving —
it does not touch `pp.serving`.

The MCP-Atlas environment is pure Python, so the container is a tiny CPU-only
`debian_slim` with the `pp` package copied in — no GPU, no pip installs, no model
weights. Running the bounded smoke here costs a few seconds of CPU.

Run the one-task smoke on Modal:

    modal run -m pp.runtimes.lhaw.modal_app --task-id "lhaw::<oid>::V_S3_delete"

or drive it from ``scripts/runtime_lhaw_smoke.py`` (which handles task loading,
the optional Qwen ``run_episode`` demo, and writing the results JSON).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

try:
    import modal
except ImportError:  # importable without modal installed
    modal = None  # type: ignore

APP_NAME = "permission-persistence-lhaw"

# The pp package source (…/src/pp). Copied into the container at /root/pp so the
# env code runs on Modal exactly as it does locally.
_PP_DIR = Path(__file__).resolve().parents[2]           # …/src/pp
_SRC_DIR = _PP_DIR.parent                               # …/src


def _build_image():
    # CPU-only, no third-party deps: the MCP-Atlas env uses stdlib only.
    return (
        modal.Image.debian_slim(python_version="3.11")
        .add_local_dir(str(_PP_DIR), remote_path="/root/pp")
    )


def build_app():
    """Construct the Modal app. Returns None when modal is unavailable so the
    module stays importable offline / under pytest."""
    if modal is None:
        return None

    app = modal.App(APP_NAME)
    image = _build_image()

    @app.function(
        image=image, timeout=300, cpu=1.0,
        serialized=True,  # function is defined inside build_app(), not global scope
    )
    def run_smoke_remote(task_payload: Dict[str, Any]) -> Dict[str, Any]:  # pragma: no cover - runs on Modal
        import sys
        sys.path.insert(0, "/root")
        from pp.runtimes.lhaw.smoke import run_smoke
        result = run_smoke(task_payload)
        result["executed_on"] = "modal"
        return result

    app._pp_functions = {"run_smoke_remote": run_smoke_remote}  # type: ignore[attr-defined]
    return app


app = build_app()


# --------------------------------------------------------------------------
# `modal run -m pp.runtimes.lhaw.modal_app --task-id ...` entrypoint.
# --------------------------------------------------------------------------
if modal is not None:

    @app.local_entrypoint()  # type: ignore[union-attr]
    def main(task_id: str, raw_path: Optional[str] = None) -> None:  # pragma: no cover
        import sys
        sys.path.insert(0, str(_SRC_DIR))
        from pp.runtimes.lhaw.data import load_task

        task = (load_task(task_id) if raw_path is None
                else load_task(task_id, Path(raw_path)))
        result = app._pp_functions["run_smoke_remote"].remote(task.to_payload())  # type: ignore[attr-defined]
        print(json.dumps(result, indent=2))
