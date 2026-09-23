"""Backend registry. Model IDs live in configs/models/*.yaml, never in code."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from .base import BaseModelBackend, GenerationResult, ModelMetadata, ActionParseError, parse_action
from .mock import MockBackend, POLICIES
from .api_backends import (
    OpenAIBackend,
    AnthropicBackend,
    GeminiBackend,
    OpenAICompatibleBackend,
)

BACKENDS: Dict[str, type] = {
    "mock": MockBackend,
    "openai": OpenAIBackend,
    "anthropic": AnthropicBackend,
    "gemini": GeminiBackend,
    "google": GeminiBackend,
    "openai_compatible": OpenAICompatibleBackend,
}


def build_backend(config: Dict[str, Any]) -> BaseModelBackend:
    """Build a backend from a config dict.

    Required key: ``backend``. Everything else is passed through, so adding a
    provider never requires touching experiment logic.
    """
    cfg = dict(config)
    backend = cfg.pop("backend", None)
    if backend is None:
        raise ValueError("model config must specify 'backend'")
    if backend not in BACKENDS:
        raise ValueError(f"unknown backend {backend!r}; known: {sorted(BACKENDS)}")
    cfg.pop("name", None)
    cfg.pop("pricing_key", None)
    cfg.pop("notes", None)
    return BACKENDS[backend](**cfg)


def load_model_config(path: str | Path) -> Dict[str, Any]:
    import yaml
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


__all__ = [
    "BaseModelBackend", "GenerationResult", "ModelMetadata",
    "ActionParseError", "parse_action",
    "MockBackend", "POLICIES",
    "OpenAIBackend", "AnthropicBackend", "GeminiBackend", "OpenAICompatibleBackend",
    "BACKENDS", "build_backend", "load_model_config",
]
