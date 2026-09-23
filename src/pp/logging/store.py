"""Trajectory logging and resume (spec section 24).

Append-only JSONL. Resume is keyed on (scenario_id, seed, model_id) so a
re-run skips completed cells rather than duplicating them.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Set, Tuple

from ..schemas import Trajectory

RunKey = Tuple[str, int, str]


class TrajectoryStore:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, traj: Trajectory) -> None:
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(traj.to_dict(), ensure_ascii=False) + "\n")

    def read_raw(self) -> Iterator[Dict[str, Any]]:
        if not self.path.exists():
            return iter(())
        def _gen() -> Iterator[Dict[str, Any]]:
            with open(self.path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        yield json.loads(line)
        return _gen()

    def completed_keys(self) -> Set[RunKey]:
        keys: Set[RunKey] = set()
        for d in self.read_raw():
            keys.add((d.get("scenario_id", ""), int(d.get("seed", 0)),
                      d.get("model_id", "")))
        return keys

    def rows_by_treatment(self) -> Dict[str, List[Dict[str, Any]]]:
        out: Dict[str, List[Dict[str, Any]]] = {}
        for d in self.read_raw():
            out.setdefault(d.get("treatment", "?"), []).append(
                d.get("metrics", {}))
        return out

    def count(self) -> int:
        return sum(1 for _ in self.read_raw())
