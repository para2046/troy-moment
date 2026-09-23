"""LHAW task loading (dataset-agnostic reader for the MCP-Atlas subset).

The raw LHAW release (`data/candidates/lhaw_raw/test.json`) pairs, for each
variant, a fully specified `original_prompt` with an `underspecified_prompt`
plus `expected_questions` and `terminal_states`. This module reads that file
(read-only) and normalises one variant into an `LHAWTask` the environment and
oracle consume. No Modal, no network — pure stdlib so it imports offline.
"""
from __future__ import annotations

import ast
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# repo-root-relative default; overridable for Modal (data shipped as an arg).
_DEFAULT_RAW = (
    Path(__file__).resolve().parents[4]
    / "data" / "candidates" / "lhaw_raw" / "test.json"
)

VARIANT_ORIGINAL = "original"
VARIANT_UNDERSPECIFIED = "underspecified"


@dataclass
class LHAWTask:
    """One LHAW variant, benchmark-neutral for the runner/oracle."""

    task_id: str            # e.g. "lhaw::<oid>::V_S3_delete"
    variant_id: str         # raw variant_id, e.g. "<oid>::V_S3_delete"
    dataset: str            # "MCP-Atlas"
    original_task: str
    original_prompt: str
    underspecified_prompt: str
    information_dimension: List[str] = field(default_factory=list)
    ambiguity_class: str = ""
    removed_segments: List[Dict[str, Any]] = field(default_factory=list)
    expected_questions: List[Dict[str, Any]] = field(default_factory=list)
    #: parsed observed-outcome checkpoint vectors (from the original task)
    terminal_states: List[Tuple[int, ...]] = field(default_factory=list)

    # -- derived signals the oracle relies on -----------------------------
    def prompt_for(self, variant: str) -> str:
        if variant == VARIANT_ORIGINAL:
            return self.original_prompt
        if variant == VARIANT_UNDERSPECIFIED:
            return self.underspecified_prompt
        raise ValueError(f"unknown variant {variant!r}")

    def expected_question_segments(self) -> List[str]:
        return [str(q.get("segment_id")) for q in self.expected_questions
                if q.get("segment_id") is not None]

    def all_expected_questions(self) -> List[str]:
        out: List[str] = []
        for q in self.expected_questions:
            out.extend(q.get("questions", []) or [])
        return out

    def distinct_completion_vectors(self) -> List[Tuple[int, ...]]:
        """Non-trivial (not all-zero) observed terminal vectors, de-duped."""
        seen: List[Tuple[int, ...]] = []
        for v in self.terminal_states:
            if any(v) and v not in seen:
                seen.append(v)
        return seen

    def is_outcome_critical(self) -> bool:
        return self.ambiguity_class == "outcome-critical"

    def is_target_ambiguous(self) -> bool:
        """True when the observed outcomes diverge (>=2 distinct completion
        vectors) OR the task is labelled outcome-critical: committing to a
        single target without clarifying then risks the WRONG target."""
        return self.is_outcome_critical() or len(self.distinct_completion_vectors()) >= 2

    def to_payload(self) -> Dict[str, Any]:
        """Plain-dict form so a single task can be shipped to Modal as an arg
        (avoids mounting the whole 1MB release into the container)."""
        return {
            "task_id": self.task_id,
            "variant_id": self.variant_id,
            "dataset": self.dataset,
            "original_task": self.original_task,
            "original_prompt": self.original_prompt,
            "underspecified_prompt": self.underspecified_prompt,
            "information_dimension": self.information_dimension,
            "ambiguity_class": self.ambiguity_class,
            "removed_segments": self.removed_segments,
            "expected_questions": self.expected_questions,
            "terminal_states": [list(v) for v in self.terminal_states],
        }

    @staticmethod
    def from_payload(d: Dict[str, Any]) -> "LHAWTask":
        return LHAWTask(
            task_id=d["task_id"],
            variant_id=d["variant_id"],
            dataset=d["dataset"],
            original_task=d["original_task"],
            original_prompt=d["original_prompt"],
            underspecified_prompt=d["underspecified_prompt"],
            information_dimension=list(d.get("information_dimension", [])),
            ambiguity_class=d.get("ambiguity_class", ""),
            removed_segments=list(d.get("removed_segments", [])),
            expected_questions=list(d.get("expected_questions", [])),
            terminal_states=[tuple(v) for v in d.get("terminal_states", [])],
        )


def _parse_terminal_states(raw: Any) -> List[Tuple[int, ...]]:
    """`terminal_states` ships as a stringified list of tuples; be tolerant."""
    if raw is None:
        return []
    if isinstance(raw, str):
        raw = raw.strip()
        if not raw:
            return []
        try:
            raw = ast.literal_eval(raw)
        except (ValueError, SyntaxError):
            return []
    out: List[Tuple[int, ...]] = []
    for v in raw or []:
        try:
            out.append(tuple(int(x) for x in v))
        except (TypeError, ValueError):
            continue
    return out


def _strip_prefix(task_id: str) -> str:
    """`lhaw::<oid>::V_S3_delete` -> `<oid>::V_S3_delete` (raw variant_id)."""
    return task_id[len("lhaw::"):] if task_id.startswith("lhaw::") else task_id


def _row_to_task(row: Dict[str, Any]) -> LHAWTask:
    variant_id = row["variant_id"]
    return LHAWTask(
        task_id=f"lhaw::{variant_id}",
        variant_id=variant_id,
        dataset=row.get("dataset", ""),
        original_task=row.get("original_task", ""),
        original_prompt=row.get("original_prompt", ""),
        underspecified_prompt=row.get("underspecified_prompt", ""),
        information_dimension=list(row.get("information_dimension", []))
        if isinstance(row.get("information_dimension"), list)
        else [row.get("information_dimension")] if row.get("information_dimension") else [],
        ambiguity_class=row.get("ambiguity_class", ""),
        removed_segments=list(row.get("removed_segments", []) or []),
        expected_questions=list(row.get("expected_questions", []) or []),
        terminal_states=_parse_terminal_states(row.get("terminal_states")),
    )


def _load_raw(raw_path: Path = _DEFAULT_RAW) -> List[Dict[str, Any]]:
    with open(raw_path, encoding="utf-8") as f:
        return json.load(f)


def load_task(task_id: str, raw_path: Path = _DEFAULT_RAW) -> LHAWTask:
    """Load one LHAW variant by task_id (with or without the `lhaw::` prefix)."""
    want = _strip_prefix(task_id)
    for row in _load_raw(raw_path):
        if row.get("variant_id") == want:
            return _row_to_task(row)
    raise KeyError(f"LHAW task {task_id!r} not found in {raw_path}")


def list_mcp_atlas_tasks(raw_path: Path = _DEFAULT_RAW) -> List[str]:
    """All MCP-Atlas variant task_ids available in the release."""
    return [f"lhaw::{r['variant_id']}" for r in _load_raw(raw_path)
            if r.get("dataset") == "MCP-Atlas"]
