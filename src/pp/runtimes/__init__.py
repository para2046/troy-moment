"""Executable benchmark runtimes (Modal-backed).

Shared, model-agnostic contract lives in `base`. Benchmark-specific
implementations live in `impossible/` and `lhaw/` (each owned by one worker).
The contract is authored once to avoid concurrent edits to shared runtime code.
"""
from .base import (
    ExecutableEnv,
    StepResult,
    Trace,
    TraceStep,
    SnapshotHandle,
    run_episode,
)

__all__ = ["ExecutableEnv", "StepResult", "Trace", "TraceStep",
           "SnapshotHandle", "run_episode"]
