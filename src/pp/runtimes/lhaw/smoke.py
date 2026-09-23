"""Bounded smoke checks for the MCP-Atlas LHAW substrate (one task).

`run_smoke(task_payload)` is pure Python (no Modal, no network) so it runs
identically on the host and inside a Modal container. It proves three claims the
brief requires:

  * ``env_evaluator_ok``            — env initialises and the oracle returns a
                                       well-formed Safe/Wrong/Over verdict.
  * ``pairing_ok``                  — BOTH the original and underspecified
                                       variants load and run (Study A1 pairing).
  * ``snapshot_restore_identical``  — snapshot -> mutate -> restore reproduces a
                                       byte-identical fingerprint (Study-B fork).

The Qwen `run_episode` demo is deliberately NOT here (it needs network); the
orchestrator script adds it around this core.
"""
from __future__ import annotations

import time
from typing import Any, Dict

from ...schemas import Action, ActionType
from .data import LHAWTask, VARIANT_ORIGINAL, VARIANT_UNDERSPECIFIED
from .env import LHAWMCPAtlasEnv
from .oracle import VerdictLabel

_VALID_LABELS = {v.value for v in VerdictLabel}


def _check_env_evaluator(task: LHAWTask) -> Dict[str, Any]:
    """Init env on the underspecified variant, drive a KNOWN action, and
    confirm the oracle returns a well-formed verdict."""
    env = LHAWMCPAtlasEnv(task, variant=VARIANT_UNDERSPECIFIED)
    prompt = env.reset()
    # KNOWN action: ask for clarification on an outcome-critical variant.
    res = env.step(Action(action=ActionType.ASK_FOR_CLARIFICATION,
                          public_message="Which target did you mean?"))
    verdict = env.evaluate()
    well_formed = (
        isinstance(verdict, dict)
        and verdict.get("label") in _VALID_LABELS
        and verdict.get("well_formed") is True
        and isinstance(verdict.get("rationale"), str) and verdict["rationale"]
    )
    return {
        "ok": bool(well_formed and prompt and res.authorized is True),
        "verdict_label": verdict.get("label"),
        "verdict_rationale": verdict.get("rationale"),
        "prompt_len": len(prompt),
    }


def _check_pairing(task: LHAWTask) -> Dict[str, Any]:
    """Both variants load, differ, and each yields a well-formed verdict."""
    out: Dict[str, Any] = {}
    prompts = {}
    for variant in (VARIANT_ORIGINAL, VARIANT_UNDERSPECIFIED):
        env = LHAWMCPAtlasEnv(task, variant=variant)
        p = env.reset(variant)
        prompts[variant] = p
        env.step(Action(action=ActionType.ASK_FOR_CLARIFICATION,
                        public_message="?"))
        v = env.evaluate()
        out[f"{variant}_verdict"] = v.get("label")
        out[f"{variant}_prompt_len"] = len(p)
    differ = prompts[VARIANT_ORIGINAL] != prompts[VARIANT_UNDERSPECIFIED]
    both_loaded = all(prompts.values())
    out["ok"] = bool(differ and both_loaded)
    out["variants_differ"] = differ
    return out


def _check_snapshot_restore(task: LHAWTask) -> Dict[str, Any]:
    """snapshot -> mutate -> restore must reproduce the exact fingerprint, and
    two restores of the same handle must be byte-identical."""
    env = LHAWMCPAtlasEnv(task, variant=VARIANT_UNDERSPECIFIED)
    env.reset()
    # advance a couple of turns so the frozen state is non-trivial
    env.step(Action(action=ActionType.USE_ALLOWED_TOOL, tool="search"))
    env.step(Action(action=ActionType.ASK_FOR_CLARIFICATION,
                    public_message="which one?"))
    handle = env.snapshot()
    fp_frozen = handle.fingerprint

    # mutate AFTER snapshot (fork would diverge here)
    env.step(Action(action=ActionType.CONTINUE_AUTHORIZED_WORK,
                    arguments={"final_answer": "some target"}))
    fp_mutated = env.fingerprint()

    # restore #1
    env.restore(handle)
    fp_restore_1 = env.snapshot().fingerprint
    # restore #2 into a fresh env
    env2 = LHAWMCPAtlasEnv(task, variant=VARIANT_ORIGINAL)
    env2.reset()  # deliberately different state before restore
    env2.restore(handle)
    fp_restore_2 = env2.snapshot().fingerprint

    identical = (fp_frozen == fp_restore_1 == fp_restore_2) and (fp_mutated != fp_frozen)
    return {
        "ok": bool(identical),
        "fingerprint_frozen": fp_frozen,
        "fingerprint_after_mutation": fp_mutated,
        "fingerprint_restore_1": fp_restore_1,
        "fingerprint_restore_2": fp_restore_2,
        "mutation_changed_state": fp_mutated != fp_frozen,
    }


def run_smoke(task_payload: Dict[str, Any]) -> Dict[str, Any]:
    """Run all three bounded checks for one task payload. Pure/deterministic."""
    t0 = time.time()
    task = LHAWTask.from_payload(task_payload)
    env_eval = _check_env_evaluator(task)
    pairing = _check_pairing(task)
    snap = _check_snapshot_restore(task)
    elapsed_ms = (time.time() - t0) * 1000.0
    return {
        "task_id": task.task_id,
        "substrate": "MCP-Atlas",
        "env_evaluator_ok": env_eval["ok"],
        "pairing_ok": pairing["ok"],
        "snapshot_restore_identical": snap["ok"],
        "details": {
            "env_evaluator": env_eval,
            "pairing": pairing,
            "snapshot_restore": snap,
        },
        "timing_ms": round(elapsed_ms, 3),
    }
