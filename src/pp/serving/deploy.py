"""Helpers to point model configs at deployed Modal endpoints.

Modal web URLs follow a stable pattern once an app is deployed:

    https://<workspace>--<app-name>-<function-name>.modal.run

This resolves a served model's URL and rewrites the matching
configs/models/*.yaml ``base_url`` so the experiment code (which only knows
``base_url``) needs no change.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import yaml

APP_NAME = "permission-persistence-serving"
MODELS_DIR = Path(__file__).resolve().parents[3] / "configs" / "models"


def modal_url(workspace: str, name: str, app_name: str = APP_NAME) -> str:
    """The web URL of the ``serve_<name>`` function's endpoint, /v1 suffixed.

    Modal renders web-endpoint URLs with underscores converted to hyphens and
    lowercased, so ``serve_qwen3_32b`` becomes ``serve-qwen3-32b`` in the host.
    """
    fn = f"serve_{name}".replace("_", "-")
    host = f"{workspace}--{app_name}-{fn}".replace("_", "-").lower()
    return f"https://{host}.modal.run/v1"


def point_config_at_modal(model_config_path: str | Path, workspace: str,
                          serving_name: str,
                          app_name: str = APP_NAME) -> Dict[str, Any]:
    """Rewrite a model config's base_url to the Modal endpoint. Returns it."""
    p = Path(model_config_path)
    cfg = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    cfg["base_url"] = modal_url(workspace, serving_name, app_name)
    cfg["backend"] = "openai_compatible"
    p.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True),
                 encoding="utf-8")
    return cfg


def resolve_base_url(workspace: Optional[str], serving_name: str,
                     local_port: Optional[int] = None,
                     app_name: str = APP_NAME) -> str:
    """Pick a base_url: local stub if a port is given, else the Modal endpoint."""
    if local_port is not None:
        return f"http://127.0.0.1:{local_port}/v1"
    if not workspace:
        raise ValueError("need either local_port or a Modal workspace")
    return modal_url(workspace, serving_name, app_name)
