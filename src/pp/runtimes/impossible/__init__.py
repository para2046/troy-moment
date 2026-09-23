"""Impossible-SWE executable runtime (Modal-backed).

Import-safe with NO Modal installed: only ``modal_app`` touches Modal, and its
import is guarded. This ``__init__`` exposes the Modal-free pieces (task loader,
workspace logic, the ExecutableEnv implementation); import ``modal_app``
explicitly when a Modal run is actually needed.
"""
from __future__ import annotations

from .tasks import SweTask, load_task, DEFAULT_RAW_DIR
from .workspace import RepoWorkspace, PytestOutcome
from .env import SweBenchModalEnv

__all__ = [
    "SweTask", "load_task", "DEFAULT_RAW_DIR",
    "RepoWorkspace", "PytestOutcome",
    "SweBenchModalEnv",
]
