"""LHAW loader (ScaleAI/lhaw) — paired well-specified / underspecified prompts.

Each LHAW row is a task from which ONE specification dimension (Goal, Constraint,
Input, or Context) has been removed by a stated removal strategy, producing an
underspecified variant with a known ambiguity class (outcome-critical /
divergent / benign).

Design rules:
* Offline-safe. If ``datasets`` is unavailable or the fetch fails, fall back to
  a local JSONL cache (config ``cache_path``). Never hard-fail at import time
  and never at call time when a cache is provided.
* Prompts come from the HF row only. TAC / SWE-Bench Pro / MCP-Atlas are NOT
  fetched here.
* ``authorization_confounded`` is set True when the removed dimension is a
  Constraint — removing a constraint can look like implicit permission, which
  would confound a boundary/authorization read.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from ..schemas import SpecificationType, Solvability, TaskInstance

HF_REPO = "ScaleAI/lhaw"

#: dimensions LHAW removes; a removed Constraint is authorization-confounded.
DIMENSIONS = ("Goal", "Constraint", "Input", "Context")
AMBIGUITY_CLASSES = ("outcome-critical", "divergent", "benign")


def _norm_dimension(value: Any) -> str:
    s = str(value or "").strip().lower()
    for d in DIMENSIONS:
        if d.lower() in s:
            return d
    return str(value or "unknown")


@dataclass
class LHAWTask:
    """One LHAW paired task (original + underspecified)."""

    task_id: str
    original_prompt: str
    underspecified_prompt: str
    original_task: str
    removed_dimension: str
    removal_strategy: str
    ambiguity_class: str
    authorization_confounded: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_task_instance(self) -> TaskInstance:
        """The UNDERSPECIFIED variant as a repo TaskInstance.

        The paired original prompt and all LHAW dimensions travel in metadata
        so a Study A run can recover the pairing.
        """
        return TaskInstance(
            task_id=self.task_id,
            source="lhaw::ScaleAI/lhaw",
            prompt=self.underspecified_prompt,
            specification_type=SpecificationType.UNDERSPECIFIED,
            solvability=Solvability.UNDERSPECIFIED,
            expected_boundary=(
                "removed constraint may be misread as implicit permission"
                if self.authorization_confounded else None),
            metadata={
                "benchmark": "lhaw",
                "original_prompt": self.original_prompt,
                "original_task": self.original_task,
                "removed_dimension": self.removed_dimension,
                "removal_strategy": self.removal_strategy,
                "ambiguity_class": self.ambiguity_class,
                "authorization_confounded": self.authorization_confounded,
                "paired": True,
                **self.metadata,
            },
        )

    def original_task_instance(self) -> TaskInstance:
        """The WELL-SPECIFIED original as a repo TaskInstance."""
        return TaskInstance(
            task_id=f"{self.task_id}::original",
            source="lhaw::ScaleAI/lhaw",
            prompt=self.original_prompt,
            specification_type=SpecificationType.WELL_SPECIFIED,
            solvability=Solvability.SOLVABLE,
            metadata={"benchmark": "lhaw", "paired": True,
                      "removed_dimension": self.removed_dimension},
        )


def _first(row: Dict[str, Any], *keys: str, default: Any = "") -> Any:
    for k in keys:
        if k in row and row[k] not in (None, ""):
            return row[k]
    return default


def _record_from_row(row: Dict[str, Any], idx: int) -> LHAWTask:
    removed = _norm_dimension(
        _first(row, "removed_dimension", "dimension", "removed", "axis"))
    task_id = str(_first(row, "task_id", "id", "uid", default=f"lhaw-{idx}"))
    return LHAWTask(
        task_id=f"lhaw::{task_id}",
        original_prompt=str(_first(row, "original_prompt", "original",
                                   "well_specified_prompt", "full_prompt")),
        underspecified_prompt=str(_first(row, "underspecified_prompt",
                                         "underspecified", "ambiguous_prompt",
                                         "prompt")),
        original_task=str(_first(row, "original_task", "task", "task_name",
                                 default=task_id)),
        removed_dimension=removed,
        removal_strategy=str(_first(row, "removal_strategy", "strategy",
                                    "method", default="unspecified")),
        ambiguity_class=str(_first(row, "ambiguity_class", "ambiguity",
                                   "class", default="unspecified")),
        authorization_confounded=(removed == "Constraint"),
        metadata={k: v for k, v in row.items()
                  if k not in {"original_prompt", "underspecified_prompt"}},
    )


def _iter_jsonl(path: str | Path) -> Iterable[Dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_lhaw(cache_path: Optional[str | Path] = None,
              hf_repo: str = HF_REPO,
              split: str = "test",
              limit: Optional[int] = None,
              prefer_cache: bool = False) -> List[LHAWTask]:
    """Load LHAW paired tasks.

    Resolution order: a local JSONL ``cache_path`` (if given and either
    ``prefer_cache`` or the HF library is unavailable), otherwise the HF
    dataset. Offline with a cache never raises; offline without a cache raises
    a clear error only at call time.
    """
    rows: Optional[Iterable[Dict[str, Any]]] = None

    def _load_hf() -> Optional[Iterable[Dict[str, Any]]]:
        try:
            from datasets import load_dataset
        except ImportError:
            return None
        try:
            ds = load_dataset(hf_repo, split=split)
        except Exception:
            return None
        return list(ds)

    if cache_path and prefer_cache:
        rows = list(_iter_jsonl(cache_path))
    else:
        rows = _load_hf()
        if rows is None and cache_path:
            rows = list(_iter_jsonl(cache_path))

    if rows is None:
        raise RuntimeError(
            "LHAW unavailable: install `datasets` for the HF fetch or pass "
            "cache_path to a local JSONL cache (offline).")

    out = [_record_from_row(r, i) for i, r in enumerate(rows)]
    if limit is not None:
        out = out[:limit]
    return out
