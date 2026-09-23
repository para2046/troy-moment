"""Per-model concurrency control for the Modal Shared Endpoints.

Standalone module: it deliberately imports nothing from the model core
(``base``/``api_backends``/``__init__``) so the trajectory runner can bound the
number of concurrent calls per shared endpoint without touching the frozen
adapter code.

Why this exists
---------------
Each Modal Shared Endpoint enforces a HARD concurrency cap (16 for
glm-5.3-flash / qwen3.8-max / kimi-k3). Hitting that cap means 429s and dropped
trajectories. We therefore run a *runtime* semaphore strictly BELOW the hard cap
(default 12) to leave headroom for retries and other tenants. A configuration
that would meet or exceed the hard cap is rejected at construction time rather
than silently oversubscribing the endpoint.

The limiter is thread-based because the Study-A/B trajectory runners fan out
concurrent generate() calls across a thread pool. Acquire either as a context
manager (preferred)::

    with get_limiter(cfg).slot():
        backend.generate(...)

or explicitly with ``acquire()`` / ``release()``.

Fallback selection (Kimi -> Gemma) also lives here so the runner has a single,
config-driven place to ask "which endpoint should I actually call?".
"""
from __future__ import annotations

import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterator, Optional, Tuple

# Workspace-resolved facts (see configs/models/{glm,qwen3_8_max,kimi,gemma}.yaml).
SHARED_ENDPOINT_HARD_CAP = 16      # per-endpoint HARD cap enforced by the workspace
DEFAULT_CONCURRENCY_LIMIT = 12     # runtime semaphore, strictly below the hard cap


class ConcurrencyConfigError(ValueError):
    """A limiter was configured in a way that would violate the hard cap."""


class ConcurrencyError(RuntimeError):
    """A runtime concurrency invariant was violated (e.g. over-release)."""


class ConcurrencyTimeout(TimeoutError):
    """A slot could not be acquired within the requested timeout."""


class FallbackUnavailable(RuntimeError):
    """A fallback config was requested but none is configured/resolvable."""


class ConcurrencyLimiter:
    """A bounded per-model concurrency limiter.

    Parameters
    ----------
    limit:
        Max simultaneous in-flight calls this process will make to the endpoint.
        Must be >= 1 and strictly below ``hard_cap``.
    hard_cap:
        The endpoint's HARD cap. Purely a guard rail; the limiter never issues
        more than ``limit`` slots, but any attempt to configure ``limit`` at or
        above ``hard_cap`` (or to request more than ``hard_cap`` slots at once)
        raises so the misconfiguration is caught loudly.
    name:
        Human-readable model/endpoint name, used in error messages.
    """

    def __init__(self, limit: int = DEFAULT_CONCURRENCY_LIMIT,
                 hard_cap: int = SHARED_ENDPOINT_HARD_CAP,
                 name: str = "model") -> None:
        limit = int(limit)
        hard_cap = int(hard_cap)
        if hard_cap < 1:
            raise ConcurrencyConfigError(f"{name}: hard_cap must be >= 1, got {hard_cap}")
        if limit < 1:
            raise ConcurrencyConfigError(
                f"{name}: concurrency_limit must be >= 1, got {limit}")
        if limit >= hard_cap:
            raise ConcurrencyConfigError(
                f"{name}: concurrency_limit ({limit}) must be strictly below the "
                f"shared-endpoint hard_cap ({hard_cap})")
        self.name = name
        self.limit = limit
        self.hard_cap = hard_cap
        self._sem = threading.BoundedSemaphore(limit)
        self._lock = threading.Lock()
        self._in_use = 0

    # -- introspection ------------------------------------------------------
    @property
    def in_use(self) -> int:
        with self._lock:
            return self._in_use

    @property
    def available(self) -> int:
        with self._lock:
            return self.limit - self._in_use

    def check_capacity(self, n: int) -> None:
        """Raise if a caller asks for more concurrency than the hard cap allows.

        Guards callers that try to size their own fan-out from the hard cap
        instead of the runtime limit.
        """
        if n > self.hard_cap:
            raise ConcurrencyError(
                f"{self.name}: requested {n} concurrent calls exceeds hard_cap "
                f"({self.hard_cap})")

    # -- acquire / release --------------------------------------------------
    def acquire(self, blocking: bool = True,
                timeout: Optional[float] = None) -> bool:
        acquired = self._sem.acquire(blocking, timeout)
        if acquired:
            with self._lock:
                self._in_use += 1
        return acquired

    def release(self) -> None:
        with self._lock:
            if self._in_use <= 0:
                raise ConcurrencyError(
                    f"{self.name}: release() without a matching acquire()")
            self._in_use -= 1
        self._sem.release()

    @contextmanager
    def slot(self, timeout: Optional[float] = None) -> Iterator["ConcurrencyLimiter"]:
        """Context manager that holds one concurrency slot for its body."""
        if not self.acquire(timeout=timeout):
            raise ConcurrencyTimeout(
                f"{self.name}: could not acquire a slot within {timeout}s "
                f"(limit={self.limit})")
        try:
            yield self
        finally:
            self.release()

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return (f"ConcurrencyLimiter(name={self.name!r}, limit={self.limit}, "
                f"hard_cap={self.hard_cap}, in_use={self.in_use})")


# --------------------------------------------------------------------------
# Config-driven construction + shared registry
# --------------------------------------------------------------------------
def limiter_from_config(config: Dict[str, Any]) -> ConcurrencyLimiter:
    """Build a limiter from a model config dict.

    Reads ``concurrency_limit`` (default 12) and ``hard_cap`` (default 16).
    Non-shared-endpoint configs (no ``hard_cap``) still get a limiter using the
    defaults so the runner can treat every backend uniformly.
    """
    name = config.get("name") or config.get("model_id") or "model"
    limit = int(config.get("concurrency_limit", DEFAULT_CONCURRENCY_LIMIT))
    hard_cap = int(config.get("hard_cap", SHARED_ENDPOINT_HARD_CAP))
    return ConcurrencyLimiter(limit=limit, hard_cap=hard_cap, name=name)


_REGISTRY: Dict[str, ConcurrencyLimiter] = {}
_REGISTRY_LOCK = threading.Lock()


def endpoint_key(config: Dict[str, Any]) -> str:
    """Identity a limiter is shared under.

    Callers hitting the *same* shared endpoint must share one semaphore, so the
    key is the endpoint base_url when present (all shared-endpoint configs point
    at the same Modal deployment), else the model name.
    """
    return (config.get("base_url")
            or config.get("name")
            or config.get("model_id")
            or "model")


def get_limiter(config: Dict[str, Any]) -> ConcurrencyLimiter:
    """Return the process-wide limiter for a config, creating it once.

    All configs that resolve to the same ``endpoint_key`` share a single
    limiter, so the runtime never exceeds the endpoint's real capacity even when
    several model configs point at the same Modal Shared Endpoint.
    """
    key = endpoint_key(config)
    with _REGISTRY_LOCK:
        limiter = _REGISTRY.get(key)
        if limiter is None:
            limiter = limiter_from_config(config)
            _REGISTRY[key] = limiter
        return limiter


def reset_registry() -> None:
    """Clear the shared registry (test hook)."""
    with _REGISTRY_LOCK:
        _REGISTRY.clear()


# --------------------------------------------------------------------------
# Kimi -> Gemma fallback (config-driven)
# --------------------------------------------------------------------------
def _default_config_dir() -> Path:
    # src/pp/models/concurrency.py -> repo_root/configs/models
    return Path(__file__).resolve().parents[3] / "configs" / "models"


def _load_yaml(path: Path) -> Dict[str, Any]:
    import yaml
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_fallback_config(primary_config: Dict[str, Any],
                         config_dir: Optional[Path] = None) -> Dict[str, Any]:
    """Resolve the fallback model config for a primary config.

    The primary config names its fallback by model id via ``fallback_model``
    (e.g. kimi.yaml -> ``gemma-4-31b``). We locate the config file whose
    ``model_id`` (or ``name``) matches and load it. Raises
    ``FallbackUnavailable`` if no fallback is configured or resolvable.
    """
    fb_id = primary_config.get("fallback_model")
    if not fb_id:
        raise FallbackUnavailable(
            f"{primary_config.get('name', 'model')}: no fallback_model configured")
    cfg_dir = Path(config_dir) if config_dir is not None else _default_config_dir()
    if not cfg_dir.is_dir():
        raise FallbackUnavailable(f"config dir not found: {cfg_dir}")
    for path in sorted(cfg_dir.glob("*.yaml")):
        try:
            cfg = _load_yaml(path)
        except Exception:  # noqa: BLE001 - a malformed sibling shouldn't crash resolution
            continue
        if cfg.get("model_id") == fb_id or cfg.get("name") == fb_id:
            return cfg
    raise FallbackUnavailable(
        f"fallback model {fb_id!r} not found under {cfg_dir}")


def select_active_config(primary_config: Dict[str, Any],
                         endpoint_available: bool = True,
                         config_dir: Optional[Path] = None) -> Tuple[Dict[str, Any], bool]:
    """Choose which model config the runner should actually call.

    Returns ``(config, used_fallback)``. If ``endpoint_available`` is True the
    primary config is returned unchanged. Otherwise the configured fallback
    (Kimi -> Gemma-4-31B) is loaded and returned; if no fallback exists this
    raises ``FallbackUnavailable`` (never silently keeps calling a dead endpoint).
    """
    if endpoint_available:
        return primary_config, False
    return load_fallback_config(primary_config, config_dir=config_dir), True


__all__ = [
    "SHARED_ENDPOINT_HARD_CAP",
    "DEFAULT_CONCURRENCY_LIMIT",
    "ConcurrencyConfigError",
    "ConcurrencyError",
    "ConcurrencyTimeout",
    "FallbackUnavailable",
    "ConcurrencyLimiter",
    "limiter_from_config",
    "endpoint_key",
    "get_limiter",
    "reset_registry",
    "load_fallback_config",
    "select_active_config",
]
