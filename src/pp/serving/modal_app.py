"""Modal serving app for the open-weight models (spec section 7).

Each open-weight model is served as an OpenAI-compatible vLLM endpoint on
Modal GPUs, so it plugs straight into ``OpenAICompatibleBackend`` with only a
``base_url`` change. Model choices, GPU types, and parallelism are DATA
(configs/serving/*.yaml), never code.

Deploy (needs a Modal account + `modal token new`):

    modal deploy -m pp.serving.modal_app

Then each model is reachable at its Modal web URL; point the matching
configs/models/*.yaml ``base_url`` at it.

Nothing here runs without Modal + a GPU; the wiring, the config plumbing, and
the client path are all testable locally against a stub server (see
scripts/fake_vllm_server.py and tests/test_serving.py).
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List

try:
    import modal
except ImportError:  # importable without modal installed
    modal = None  # type: ignore

import yaml

# --------------------------------------------------------------------------
# Serving configuration is loaded from YAML so adding a model is a data edit.
# --------------------------------------------------------------------------
_CONFIG_DIR = Path(__file__).resolve().parents[3] / "configs" / "serving"


def load_serving_configs(config_dir: Path = _CONFIG_DIR) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    if not config_dir.exists():
        return out
    for p in sorted(config_dir.glob("*.yaml")):
        with open(p, encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        out[cfg.get("name", p.stem)] = cfg
    return out


SERVING = load_serving_configs()


# --------------------------------------------------------------------------
# The Modal app. Everything below is inert unless `modal` is installed and a
# deploy is triggered; import never fails.
# --------------------------------------------------------------------------
def _build_image(vllm_version: str = "0.6.3",
                 transformers_version: str = "4.52.4"):
    # transformers is pinned deliberately: vLLM 0.9.x bundles its own `aimv2`
    # config, and transformers >=4.54 also registers one, so an unpinned
    # (latest) transformers crashes vLLM at import with a duplicate-key
    # ValueError. Pin to a version that pairs with the vLLM release (4.52.x
    # still supports Qwen3, which landed in 4.51).
    pkgs = [f"vllm=={vllm_version}", "huggingface_hub[hf_transfer]", "hf-transfer"]
    if transformers_version:
        pkgs.append(f"transformers=={transformers_version}")
    return (
        modal.Image.debian_slim(python_version="3.11")
        .pip_install(*pkgs)
        .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "VLLM_USE_V1": "1"})
    )


def _register_model(app, name: str, cfg: Dict[str, Any]):
    """Attach one vLLM OpenAI-compatible server function to the app.

    Returns the decorated function object. One function per model so they scale
    and bill independently — 'do not assume all three fit on one machine'.
    """
    gpu = cfg.get("gpu", "H100")
    n_gpu = int(cfg.get("n_gpu", 1))
    gpu_spec = gpu if n_gpu == 1 else f"{gpu}:{n_gpu}"
    hf_id = cfg["hf_model_id"]
    served_name = cfg.get("served_model_name", cfg["model_id"])
    max_len = int(cfg.get("max_model_len", 32768))
    port = int(cfg.get("port", 8000))
    scaledown = int(cfg.get("scaledown_window_s", 600))
    timeout = int(cfg.get("startup_timeout_s", 1800))
    concurrency = int(cfg.get("max_concurrent_requests", 32))
    secret_name = cfg.get("hf_secret", "huggingface")
    extra_args: List[str] = list(cfg.get("extra_vllm_args", []))

    image = _build_image(cfg.get("vllm_version", "0.6.3"),
                         cfg.get("transformers_version", "4.52.4"))
    volume = modal.Volume.from_name(
        f"vllm-cache-{name}", create_if_missing=True)

    @app.function(
        name=f"serve_{name}",
        image=image,
        gpu=gpu_spec,
        volumes={"/root/.cache/huggingface": volume},
        secrets=[modal.Secret.from_name(secret_name)]
        if secret_name else [],
        timeout=timeout,
        scaledown_window=scaledown,
        min_containers=int(cfg.get("min_containers", 0)),
        serialized=True,  # required: functions are built dynamically from config
    )
    @modal.concurrent(max_inputs=concurrency)
    @modal.web_server(port=port, startup_timeout=timeout)
    def serve() -> None:  # pragma: no cover - runs only on Modal GPUs
        import subprocess

        cmd = [
            "vllm", "serve", hf_id,
            "--served-model-name", served_name,
            "--host", "0.0.0.0",
            "--port", str(port),
            "--max-model-len", str(max_len),
            "--tensor-parallel-size", str(n_gpu),
        ]
        # optional served API key so the endpoint is not fully open
        api_key = os.environ.get("VLLM_API_KEY")
        if api_key:
            cmd += ["--api-key", api_key]
        cmd += extra_args
        subprocess.Popen(" ".join(cmd), shell=True)

    return serve


def build_app():
    """Construct the Modal app from serving configs.

    Called at import time only when modal is present; returns None otherwise so
    the module stays importable in test/offline environments.
    """
    if modal is None:
        return None
    app = modal.App("permission-persistence-serving")
    app._pp_functions = {}  # type: ignore[attr-defined]
    for name, cfg in SERVING.items():
        if not cfg.get("hf_model_id"):
            continue
        if cfg.get("placeholder"):
            # repo id / GPU sizing not verified; never auto-deploy (a stray
            # 8xH200 node is expensive). Set placeholder:false once confirmed.
            continue
        app._pp_functions[name] = _register_model(app, name, cfg)  # type: ignore
    return app


# module-level app for `modal deploy -m pp.serving.modal_app`
app = build_app()
