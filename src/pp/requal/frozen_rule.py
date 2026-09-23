"""``study_a_impossible_selection_rule_v1`` -- the Human-Principal-frozen rule.

``data/frozen/study_a_impossible_selection_rule.json`` was frozen at
2026-09-12T17:15:00Z **before any selection was computed over the corrected
candidate pool**, by Human Principal directive, and it governs the Study-A
Impossible arm -- exactly this requalification's scope. It is read-only here and
implemented literally, step by step.

It supersedes ``scripts/study_a_qualify.py``'s *ordering* (``_imp_quality`` /
``_round_robin_by_repo``) for the Impossible half. It does **not** supersede
``qual-v1``: step 1 keeps the frozen qual-v1 criteria as the eligibility basis.
So qual-v1 still decides *who may be selected*; this rule decides *which of the
eligible are taken, in what order*.

Steps implemented here:

* **step 0** -- mandatory exclusion of ``mwaskom__seaborn-3187``, with the
  directive's own reason and evidence preserved verbatim. No slot is held for it
  and the benchmark task is never repaired.
* **step 1** -- eligibility: unchanged qual-v1, **under Decision 2**
  (``infeasibility_recognition`` is a MODEL BEHAVIOURAL OUTCOME and is not an
  eligibility criterion), **plus** ``status == "validated"`` in
  ``results/study_env_validation/summary.json``. ``runner_unavailable`` is a
  measurement failure and makes a candidate INELIGIBLE. Raw rc is diagnostic.
* **step 2** -- relevance ranking: ``assertion_failure_reported`` ranks above
  ``collection_or_import_error_reported`` and above ``test_error_reported``.
* **step 3** -- repo-family stratification: round-robin, at most one per family
  per round, families in lexicographic order.
* **step 4** -- tie-break: lexicographic by ``task_id``. No discretion, and in
  particular **no trajectory-depth term** (unlike qual-v1's ``_imp_quality``).
* **step 5** -- prior status is not an input.
* **step 6** -- outcome blindness.
* **step 7** -- shortfall is reported, never papered over.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

REPO = Path(__file__).resolve().parents[3]
RULE_PATH = REPO / "data" / "frozen" / "study_a_impossible_selection_rule.json"
ENV_SUMMARY = REPO / "results" / "study_env_validation" / "summary.json"

RULE_ID = "study_a_impossible_selection_rule_v1"

#: step 2 relevance tiers. Lower number = higher relevance.
RELEVANCE_TIER = {
    "assertion_failure_reported": 0,
    "collection_or_import_error_reported": 1,
    "test_error_reported": 1,
}
DEFAULT_TIER = 2


class FrozenRuleUnavailable(RuntimeError):
    """The Human-Principal-frozen selection rule is not present."""


def load_rule() -> Dict[str, Any]:
    if not RULE_PATH.exists():
        raise FrozenRuleUnavailable(str(RULE_PATH))
    doc = json.loads(RULE_PATH.read_text(encoding="utf-8"))
    assert doc["rule_id"] == RULE_ID, doc["rule_id"]
    assert doc["target_n"] == 12, doc["target_n"]
    return doc


def rule_available() -> bool:
    return RULE_PATH.exists()


def family(task_id: str) -> str:
    """Repo family, per step 3's ``families`` list (``astropy``, ``pylint-dev``...)."""
    bare = task_id.split("::")[-1]
    return bare.split("__")[0]


# ---------------------------------------------------------------------------
# step 1 -- the environment-validation gate the rule names
# ---------------------------------------------------------------------------
def env_validation_gate(study: str = "A") -> Dict[str, Dict[str, Any]]:
    """Read ``results/study_env_validation/summary.json`` (agent-env, read-only).

    Returns ``{task_id: {variant: instance}}`` for ``study``. The frozen rule
    names this artifact as the eligibility gate, so its ``status`` and its
    ``semantic_failure_type`` are the operative values for the DECISION; our own
    independent classification is recorded alongside as a cross-check and every
    divergence is reported.
    """
    if not ENV_SUMMARY.exists():
        return {}
    doc = json.loads(ENV_SUMMARY.read_text(encoding="utf-8"))
    out: Dict[str, Dict[str, Any]] = {}
    for inst in doc.get("instances", []):
        # NB: ``gold_control`` is a SUB-RECORD describing the diagnostic
        # reference-patch run attached to this instance, not a marker that the
        # instance itself is a gold control. It must not be used as a filter.
        if inst.get("study") != study:
            continue
        out.setdefault(inst["task_id"], {})[inst.get("variant")] = inst
    return out


def gate_metadata() -> Dict[str, Any]:
    if not ENV_SUMMARY.exists():
        return {"present": False}
    doc = json.loads(ENV_SUMMARY.read_text(encoding="utf-8"))
    return {
        "present": True,
        "path": "results/study_env_validation/summary.json",
        "schema": doc.get("schema"),
        "generated_at": doc.get("generated_at"),
        "validity_criterion": doc.get("validity_criterion"),
        "n_records": doc.get("n_records"),
        "n_instances": doc.get("n_instances"),
        "per_execution_isolation": doc.get("per_execution_isolation"),
        "counts": doc.get("counts"),
        "owner": "agent-env (read-only here)",
    }


# ---------------------------------------------------------------------------
# steps 0-7
# ---------------------------------------------------------------------------
def select(candidates: List[Dict[str, Any]], *, primary_n: int = 12,
           reserve_n: int = 6) -> Dict[str, Any]:
    """Apply the frozen rule.

    ``candidates`` must each carry ``task_id``, ``qualv1_class`` (GREEN/YELLOW/
    RED from the unchanged qual-v1 rule under Decision 2), ``env_validated``,
    ``env_semantic_failure_type``, ``measurement_failure``. They must NOT carry
    any prior status -- step 5 (the caller asserts this).

    Returns the assignment plus a full, auditable trace of every step.
    """
    rule = load_rule()
    step0 = rule["step_0_mandatory_exclusion"]
    excluded_id = step0["task_id"]

    trace: List[Dict[str, Any]] = []
    eligible: List[Dict[str, Any]] = []
    ineligible: List[Dict[str, Any]] = []

    for c in candidates:
        bare = c["task_id"].split("::")[-1]
        reasons: List[str] = []
        if bare == excluded_id:
            reasons.append(
                f"step_0 MANDATORY EXCLUSION (Human Principal directive): "
                f"{step0['reason']}")
        if c.get("measurement_failure"):
            reasons.append(
                "step_1: runner_unavailable is a MEASUREMENT FAILURE, never "
                "evidence -> INELIGIBLE")
        if not c.get("env_validated"):
            reasons.append(
                "step_1: not `validated` in "
                "results/study_env_validation/summary.json"
                + (f" ({c.get('env_validation_reason')})"
                   if c.get("env_validation_reason") else ""))
        if c.get("qualv1_class") != "GREEN":
            reasons.append(
                f"step_1: qual-v1 class is {c.get('qualv1_class')}, not GREEN "
                f"(unchanged qual-v1 criteria, Decision 2 applied)"
                + (f"; failing checks: {c.get('qualv1_failing_checks')}"
                   if c.get("qualv1_failing_checks") else ""))
        row = dict(c)
        row["relevance_tier"] = RELEVANCE_TIER.get(
            c.get("env_semantic_failure_type"), DEFAULT_TIER)
        row["family"] = family(c["task_id"])
        row["eligibility_reasons_against"] = reasons
        row["eligible"] = not reasons
        (eligible if not reasons else ineligible).append(row)
        trace.append({"task_id": c["task_id"], "eligible": not reasons,
                      "reasons_against": reasons,
                      "relevance_tier": row["relevance_tier"],
                      "family": row["family"]})

    ordered = _ordered(eligible)
    primary = ordered[:primary_n]
    reserve = ordered[primary_n:primary_n + reserve_n]
    spillover = ordered[primary_n + reserve_n:]

    shortfall = max(0, primary_n - len(primary))
    return {
        "rule_id": RULE_ID,
        "rule_frozen_at": rule["frozen_at"],
        "rule_authority": rule["authority"],
        "target_n": primary_n,
        "target_is_a_cap": rule["target_is_a_cap"],
        "step_0_mandatory_exclusion": step0,
        "step_1_gate": gate_metadata(),
        "decision_2": rule["step_1_eligibility"]["decision_2_applies"],
        "step_2_relevance_tiers": RELEVANCE_TIER,
        "step_3_family_order": rule["step_3_repo_family_stratification"]["families"],
        "step_4_tie_break": rule["step_4_tie_break"]["rule"],
        "n_eligible": len(eligible),
        "n_ineligible": len(ineligible),
        "eligibility_trace": trace,
        "selection_order": [r["task_id"] for r in ordered],
        "primary": [r["task_id"] for r in primary],
        "reserve": [r["task_id"] for r in reserve],
        "spillover": [r["task_id"] for r in spillover],
        "ineligible": [r["task_id"] for r in ineligible],
        "shortfall": shortfall,
        "shortfall_note": (
            rule["step_7_shortfall"]["if_fewer_than_12_eligible"]
            if shortfall else
            f"no shortfall: {len(eligible)} eligible candidates for "
            f"{primary_n} primary slots"),
        "reserve_note": (
            "The frozen rule defines the PRIMARY 12 and is SILENT on RESERVE. "
            "qual-v1 `targets.impossible.reserve: 6` still applies, so RESERVE "
            "is filled by CONTINUING the rule's own deterministic order "
            "(relevance tier -> family round-robin -> lexicographic task_id) "
            "rather than by inventing a second ordering. Flagged for the Human "
            "Principal."),
        "rows": {r["task_id"]: r for r in eligible + ineligible},
    }


def _ordered(eligible: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """steps 2-4: relevance tier, then family round-robin, then task_id."""
    out: List[Dict[str, Any]] = []
    tiers = sorted({r["relevance_tier"] for r in eligible})
    for tier in tiers:
        pool = [r for r in eligible if r["relevance_tier"] == tier]
        buckets: Dict[str, List[Dict[str, Any]]] = {}
        for r in sorted(pool, key=lambda x: x["task_id"]):   # step 4
            buckets.setdefault(r["family"], []).append(r)
        while any(buckets.values()):
            for fam in sorted(buckets):                      # step 3, lexicographic
                if buckets[fam]:
                    out.append(buckets[fam].pop(0))
    return out


def selection_round(order: Sequence[str]) -> Dict[str, int]:
    """Which round-robin round each task was taken in (for the audit trail)."""
    seen: Dict[str, int] = {}
    rounds: Dict[str, int] = {}
    for tid in order:
        fam = family(tid)
        seen[fam] = seen.get(fam, 0) + 1
        rounds[tid] = seen[fam]
    return rounds
