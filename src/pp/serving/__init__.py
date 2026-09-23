"""Modal serving layer for open-weight models."""
from .deploy import modal_url, point_config_at_modal, resolve_base_url, APP_NAME
try:
    from .modal_app import load_serving_configs, SERVING
except Exception:  # modal_app is import-safe, but be defensive
    load_serving_configs = None
    SERVING = {}
__all__ = ["modal_url", "point_config_at_modal", "resolve_base_url",
           "APP_NAME", "load_serving_configs", "SERVING"]
