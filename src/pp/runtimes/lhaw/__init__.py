"""LHAW executable runtime (MCP-Atlas substrate).

This subpackage implements the shared `pp.runtimes.base.ExecutableEnv` contract
for the **MCP-Atlas** family of LHAW tasks (tool-use / API invocation). MCP-Atlas
was chosen over SWE-Bench-Pro and TheAgentCompany because it is the cheapest
substrate to execute on Modal (a tool-calling conversation state is pure,
serializable Python — no per-task repo Docker build, no enterprise service
stack) while still yielding enough shortlisted primary tasks (6 primary + 5
reserve MCP-Atlas rows in data/qualified/shortlist.json). See
`chosen_substrate_reason` in results/runtime/lhaw_smoke.json.

The environment state is a small JSON document, which is exactly what makes
`snapshot()`/`restore()` provably byte-identical for Study-B forking. `import
modal` is only ever touched inside `modal_app.py` and is guarded there, so this
package (and the whole test suite) imports fine with or without Modal.
"""
from __future__ import annotations

from .data import LHAWTask, load_task, list_mcp_atlas_tasks
from .oracle import Verdict, VerdictLabel, adjudicate
from .env import LHAWMCPAtlasEnv

__all__ = [
    "LHAWTask",
    "load_task",
    "list_mcp_atlas_tasks",
    "Verdict",
    "VerdictLabel",
    "adjudicate",
    "LHAWMCPAtlasEnv",
]
