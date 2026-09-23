"""Boundary checkpoints — the Study A -> Study B handoff."""
from .generate import (extract_checkpoint, synthetic_checkpoint,
                       save_checkpoints, load_checkpoints)
__all__ = ["extract_checkpoint", "synthetic_checkpoint",
           "save_checkpoints", "load_checkpoints"]
