"""ImpossibleBench loader (safety-research/impossiblebench).

Exposes ORIGINAL vs CONFLICTING task pairs. The original is a genuinely
solvable task; the conflicting variant embeds a specification conflict so that
no authorized action sequence satisfies it — a model that "succeeds" must have
crossed a boundary (e.g. edited the test / gamed the checker).

Offline-safe: prefers a local JSONL cache when given, otherwise the HF dataset;
never hard-fails at import and never at call time when a cache is provided.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from ..schemas import SpecificationType, Solvability, TaskInstance

HF_REPO = "safety-research/impossiblebench"


@dataclass
class ImpossibleTask:
    """One ImpossibleBench pair (original solvable + conflicting impossible)."""

    task_id: str
    original_prompt: str
    conflicting_prompt: str
    variant: str = "conflicting"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def original_task_instance(self) -> TaskInstance:
        return TaskInstance(
            task_id=f"{self.task_id}::original",
            source="impossiblebench::safety-research/impossiblebench",
            prompt=self.original_prompt,
            specification_type=SpecificationType.WELL_SPECIFIED,
            solvability=Solvability.SOLVABLE,
            success_condition={"public_tests_pass": True},
            metadata={"benchmark": "impossiblebench", "pair": "original",
                      "variant": self.variant, "paired": True, **self.metadata},
        )

    def conflicting_task_instance(self) -> TaskInstance:
        return TaskInstance(
            task_id=f"{self.task_id}::conflicting",
            source="impossiblebench::safety-research/impossiblebench",
            prompt=self.conflicting_prompt,
            specification_type=SpecificationType.UNDERSPECIFIED,
            solvability=Solvability.IMPOSSIBLE,
            expected_boundary=("specification conflict: no authorized action "
                               "sequence satisfies the task"),
            success_condition={"public_tests_pass": True},
            metadata={"benchmark": "impossiblebench", "pair": "conflicting",
                      "variant": self.variant, "paired": True, **self.metadata},
        )

    def to_task_instances(self) -> Tuple[TaskInstance, TaskInstance]:
        return self.original_task_instance(), self.conflicting_task_instance()


def _first(row: Dict[str, Any], *keys: str, default: Any = "") -> Any:
    for k in keys:
        if k in row and row[k] not in (None, ""):
            return row[k]
    return default


def _record_from_row(row: Dict[str, Any], idx: int) -> ImpossibleTask:
    task_id = str(_first(row, "task_id", "id", "uid", "instance_id",
                         default=f"impossible-{idx}"))
    return ImpossibleTask(
        task_id=f"impossiblebench::{task_id}",
        original_prompt=str(_first(row, "original_prompt", "original",
                                   "solvable_prompt", "prompt")),
        conflicting_prompt=str(_first(row, "conflicting_prompt", "conflicting",
                                      "impossible_prompt", "modified_prompt")),
        variant=str(_first(row, "variant", "type", "category",
                           default="conflicting")),
        metadata={k: v for k, v in row.items()
                  if k not in {"original_prompt", "conflicting_prompt"}},
    )


def _iter_jsonl(path: str | Path) -> Iterable[Dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_impossiblebench(cache_path: Optional[str | Path] = None,
                         hf_repo: str = HF_REPO,
                         split: str = "test",
                         limit: Optional[int] = None,
                         prefer_cache: bool = False) -> List[ImpossibleTask]:
    """Load ImpossibleBench ORIGINAL/CONFLICTING pairs.

    Local JSONL ``cache_path`` is used when given and either ``prefer_cache``
    or the HF library is unavailable; otherwise the HF dataset is fetched.
    Offline with a cache never raises.
    """
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
        rows: Optional[Iterable[Dict[str, Any]]] = list(_iter_jsonl(cache_path))
    else:
        rows = _load_hf()
        if rows is None and cache_path:
            rows = list(_iter_jsonl(cache_path))

    if rows is None:
        raise RuntimeError(
            "ImpossibleBench unavailable: install `datasets` for the HF fetch "
            "or pass cache_path to a local JSONL cache (offline).")

    out = [_record_from_row(r, i) for i, r in enumerate(rows)]
    if limit is not None:
        out = out[:limit]
    return out
