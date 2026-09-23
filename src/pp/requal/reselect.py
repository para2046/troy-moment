"""Regenerate PRIMARY / RESERVE / EXCLUDE for the Study-A **Impossible** half.

The selection algorithm is the frozen one: ``_round_robin_by_repo`` /
``_imp_quality`` are imported from ``scripts/study_a_qualify.py`` unchanged, and
the targets come from ``configs/qualification_criteria.yaml``
(``impossible: {primary: 12, reserve: 6}``).

Prior status is not an input. It is not even present in the structures this
module sees: :func:`reselect_impossible` asserts
:func:`pp.requal.criteria.assert_no_forbidden_decision_inputs` over every
candidate *before* any status is assigned, and the caller reads
``data/qualified/study_a_selection.json`` only afterwards to build the comparison
report.

**LHAW is untouched.** LHAW does not use the pytest/SWE runtime. Its rows are
carried through byte-identically from the frozen selection and
:func:`carry_lhaw_rows` verifies that with a per-row hash.
"""
from __future__ import annotations

import collections
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from . import frozen_rule as fr
from .criteria import (
    CRITERIA_VERSION,
    assert_no_forbidden_decision_inputs,
    qualv1_rule_module,
)

#: Full step-by-step trace of the most recent frozen-rule application, so the
#: driver can embed it in the artifact without threading it through a return
#: type that the qual-v1 fallback path does not share.
LAST_FROZEN_RULE_RESULT: Dict[str, Any] = {}

REPO = Path(__file__).resolve().parents[3]
FROZEN_SELECTION = REPO / "data" / "qualified" / "study_a_selection.json"

TRAJECTORY_MEANINGFUL_ACTIONS_MIN = 4
TARGETS = {"primary": 12, "reserve": 6}

PENDING_STATUS = "PENDING_ENVIRONMENT_BUILD"


class LhawMutated(AssertionError):
    """A LHAW row was modified. LHAW is out of scope for this requalification."""


# ---------------------------------------------------------------------------
# LHAW carry-through
# ---------------------------------------------------------------------------
def _row_hash(row: Dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(row, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def carry_lhaw_rows(frozen_doc: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], Dict[str, str]]:
    """Copy the LHAW rows through unchanged and return them with their hashes."""
    rows = [r for r in frozen_doc["tasks"] if r.get("benchmark") == "lhaw"]
    out = [json.loads(json.dumps(r)) for r in rows]           # deep copy
    hashes = {r["task_id"]: _row_hash(r) for r in rows}
    for r in out:
        if _row_hash(r) != hashes[r["task_id"]]:              # pragma: no cover
            raise LhawMutated(r["task_id"])
    return out, hashes


def assert_lhaw_unchanged(new_rows: List[Dict[str, Any]],
                          frozen_hashes: Dict[str, str]) -> None:
    """Fail if any LHAW row in the regenerated selection differs."""
    lhaw = [r for r in new_rows if r.get("benchmark") == "lhaw"]
    if len(lhaw) != len(frozen_hashes):
        raise LhawMutated(
            f"LHAW row count changed: {len(lhaw)} != {len(frozen_hashes)}")
    for r in lhaw:
        tid = r.get("task_id")
        if tid not in frozen_hashes:
            raise LhawMutated(f"unknown LHAW row {tid}")
        if _row_hash(r) != frozen_hashes[tid]:
            raise LhawMutated(
                f"LHAW row {tid} was modified; LHAW does not use the "
                "pytest/SWE runtime and is out of scope for this "
                "requalification")


# ---------------------------------------------------------------------------
# reselection
# ---------------------------------------------------------------------------
def reselect_impossible(candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Assign PRIMARY / RESERVE / EXCLUDE to the Impossible candidates.

    ``candidates`` are dicts with ``task_id``, ``repo``, ``class``, ``red_flags``,
    ``evidence`` (the rule's own evidence dict) and ``measured``. They must NOT
    carry any status field -- that is asserted, not assumed.
    """
    rule = qualv1_rule_module()
    for c in candidates:
        assert_no_forbidden_decision_inputs(
            c, where=f"reselection candidate {c.get('task_id')}")

    out = [dict(c) for c in candidates]

    if fr.rule_available():
        return _reselect_under_frozen_rule(out)

    # Candidates with no valid period-correct observation cannot be evaluated at
    # all: primary_freeze.primary_requires_observed_executable_validation means
    # they can never be PRIMARY, and assigning them RESERVE/EXCLUDE would be a
    # decision taken on absent evidence. They are held as PENDING.
    pending = [c for c in out if not c.get("measured")]
    pool = [c for c in out if c.get("measured")]

    green = [c for c in pool if c["class"] == "GREEN"]
    yellow = [c for c in pool if c["class"] == "YELLOW"]
    red = [c for c in pool if c["class"] == "RED"]

    p_target, r_target = TARGETS["primary"], TARGETS["reserve"]

    primary, green_rest = rule._round_robin_by_repo(green, p_target)
    res_green, green_rest2 = rule._round_robin_by_repo(green_rest, r_target)
    reserve = list(res_green)
    remaining = r_target - len(reserve)
    yellow_sorted = sorted(yellow, key=rule._imp_quality)
    reserve += yellow_sorted[:max(0, remaining)]
    yellow_rest = yellow_sorted[max(0, remaining):]

    primary_ids = {c["task_id"] for c in primary}
    reserve_ids = {c["task_id"] for c in reserve}

    for c in primary:
        c["status"] = "PRIMARY"
        c["reason_code"] = "primary_green"
        c["inclusion_reason"] = (
            f"GREEN under {CRITERIA_VERSION} on RECOMPUTED period-correct "
            f"evidence: observed executable validation, stable evaluator "
            f"(model-visible semantic failure type stable across replays), "
            f"verified infeasibility manifesting in-env + >=1 MEANINGFUL "
            f"failure-feedback cycle carrying substantive failure evidence, "
            f"nontrivial trajectory>={TRAJECTORY_MEANINGFUL_ACTIONS_MIN}; "
            f"selected as primary (repo-diversified)."
        )
    for c in reserve:
        c["status"] = "RESERVE"
        if c["class"] == "GREEN":
            c["reason_code"] = "reserve_green_spillover"
            c["inclusion_reason"] = (
                f"GREEN under {CRITERIA_VERSION} on recomputed evidence but "
                f"primary capacity ({p_target}) met; held as reserve (fully "
                f"valid spillover)."
            )
        else:
            c["reason_code"] = "reserve_yellow"
            ch = c.get("checks", {})
            ev = c.get("evidence", {})
            if not ev.get("executed"):
                weak = ("no valid period-correct observation for both variants "
                        "-> reserve at most")
            elif not ch.get("meaningful_failure_feedback_cycle", True):
                weak = ("no MEANINGFUL failure-feedback cycle: the evaluator "
                        "returned no substantive failure evidence under the "
                        "period-correct environment")
            elif not ch.get("nontrivial_trajectory", True):
                weak = (f"shallow trajectory (<{TRAJECTORY_MEANINGFUL_ACTIONS_MIN} "
                        f"meaningful actions)")
            else:
                weak = "one non-fatal adequacy weakness"
            c["inclusion_reason"] = (
                f"YELLOW under {CRITERIA_VERSION} on recomputed evidence: "
                f"scientifically valid with one non-fatal weakness ({weak}); "
                f"reserve."
            )
    for c in green_rest2:
        if c["task_id"] in primary_ids or c["task_id"] in reserve_ids:
            continue
        c["status"] = "EXCLUDE"
        c["reason_code"] = "capacity_met_green"
        c["exclusion_reason"] = (
            f"GREEN under {CRITERIA_VERSION} on recomputed evidence but "
            f"primary+reserve capacity (primary {p_target} / reserve "
            f"{r_target}) already met with repo coverage; not activated."
        )
    for c in yellow_rest:
        c["status"] = "EXCLUDE"
        c["reason_code"] = "capacity_met_yellow"
        c["exclusion_reason"] = (
            f"YELLOW under {CRITERIA_VERSION} on recomputed evidence (valid "
            f"reserve candidate) but reserve capacity ({r_target}) already met "
            f"by stronger candidates; not activated."
        )
    for c in red:
        c["status"] = "EXCLUDE"
        c["reason_code"] = "red:" + ";".join(c["red_flags"])
        c["exclusion_reason"] = (
            f"RED under {CRITERIA_VERSION} on recomputed evidence: "
            f"{', '.join(c['red_flags'])}."
        )
    for c in pending:
        c["status"] = PENDING_STATUS
        c["reason_code"] = "pending_environment_build"
        c["exclusion_reason"] = None
        c["inclusion_reason"] = None
        c["pending_reason"] = (
            "no valid observation in a period-correct per-task environment yet; "
            "the environment layer has not published this task. Under "
            "primary_freeze.primary_requires_observed_executable_validation a "
            "candidate without observed executable validation can never be "
            "PRIMARY, and assigning RESERVE/EXCLUDE on absent evidence would be "
            "a decision without measurement. Held as PENDING, not decided."
        )

    for c in out:
        c.setdefault("status", None)
        c.setdefault("inclusion_reason", None)
        c.setdefault("exclusion_reason", None)
        c.setdefault("reason_code", None)
        c["criteria_version"] = CRITERIA_VERSION
        assert c["status"] is not None, c["task_id"]
    return out


def _reselect_under_frozen_rule(out: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Assign statuses using ``study_a_impossible_selection_rule_v1``.

    qual-v1 still decides *eligibility* (step 1); the frozen rule decides the
    *order and the cap*. Prior status is not consulted here or anywhere before
    the comparison report.
    """
    inputs = []
    for c in out:
        rec = c.get("recomputed_evidence") or {}
        failing = [k for k, v in (c.get("checks") or {}).items()
                   if k != "capability_floor_suspected" and v is not True]
        inputs.append({
            "task_id": c["task_id"],
            "qualv1_class": c.get("class"),
            "qualv1_failing_checks": failing,
            "env_validated": bool(rec.get("env_gate_validated")),
            "env_validation_reason": rec.get("env_gate_reason"),
            "env_semantic_failure_type": rec.get(
                "authoritative_semantic_failure_type"),
            "measurement_failure": rec.get("measurement_failure_status") ==
            "MEASUREMENT_FAILURE_ALL_REPLAYS",
            "measured": bool(c.get("measured")),
        })
    for i in inputs:
        assert_no_forbidden_decision_inputs(
            i, where=f"frozen-rule input {i['task_id']}")

    res = fr.select(inputs, primary_n=fr.load_rule()["target_n"],
                    reserve_n=TARGETS["reserve"])
    rounds = fr.selection_round(res["selection_order"])
    by_id = {c["task_id"]: c for c in out}
    step0 = res["step_0_mandatory_exclusion"]

    for tid in res["primary"]:
        c = by_id[tid]
        c["status"] = "PRIMARY"
        c["reason_code"] = "primary_frozen_rule"
        c["inclusion_reason"] = (
            f"Eligible under {CRITERIA_VERSION} (unchanged criteria, Decision 2 "
            f"applied: infeasibility_recognition withheld as a model "
            f"behavioural outcome) AND `validated` in "
            f"results/study_env_validation/summary.json; selected by "
            f"{fr.RULE_ID} step 2 relevance tier "
            f"{res['rows'][tid]['relevance_tier']} "
            f"(`{res['rows'][tid]['env_semantic_failure_type']}`), step 3 "
            f"repo-family round-robin round {rounds[tid]} "
            f"(family `{res['rows'][tid]['family']}`), step 4 lexicographic "
            f"tie-break.")
    for tid in res["reserve"]:
        c = by_id[tid]
        c["status"] = "RESERVE"
        c["reason_code"] = "reserve_frozen_rule_order"
        c["inclusion_reason"] = (
            f"Eligible under {CRITERIA_VERSION} but the frozen rule's PRIMARY "
            f"cap of {res['target_n']} was met earlier in its own order "
            f"(tier {res['rows'][tid]['relevance_tier']}, family "
            f"`{res['rows'][tid]['family']}`, round {rounds[tid]}); held as "
            f"reserve. {res['reserve_note']}")
    for tid in res["spillover"]:
        c = by_id[tid]
        c["status"] = "EXCLUDE"
        c["reason_code"] = "capacity_met_eligible"
        c["exclusion_reason"] = (
            f"Eligible under {CRITERIA_VERSION} + the environment-validation "
            f"gate, but primary ({res['target_n']}) and reserve "
            f"({TARGETS['reserve']}) capacity were both met earlier in "
            f"{fr.RULE_ID}'s order (tier {res['rows'][tid]['relevance_tier']}, "
            f"family `{res['rows'][tid]['family']}`, round {rounds[tid]}); not "
            f"activated. `target_is_a_cap`: {res['target_is_a_cap']}")
    for tid in res["ineligible"]:
        c = by_id[tid]
        c["status"] = "EXCLUDE"
        row = res["rows"][tid]
        mandatory = tid.split("::")[-1] == step0["task_id"]
        c["reason_code"] = ("step_0_mandatory_exclusion" if mandatory
                            else "ineligible_frozen_rule_step_1")
        c["exclusion_reason"] = "; ".join(row["eligibility_reasons_against"])
        if mandatory:
            c["mandatory_exclusion"] = {
                "authority": "Human Principal directive "
                             f"({fr.RULE_ID} step 0)",
                "reason": step0["reason"],
                "evidence": step0["evidence"],
                "do_not_repair": step0["do_not_repair"],
                "replacement": step0["replacement"],
            }

    for c in out:
        c.setdefault("status", None)
        c.setdefault("inclusion_reason", None)
        c.setdefault("exclusion_reason", None)
        c.setdefault("reason_code", None)
        c["criteria_version"] = CRITERIA_VERSION
        c["selection_rule_id"] = fr.RULE_ID
        c["frozen_rule_row"] = res["rows"].get(c["task_id"])
        c["frozen_rule_round"] = rounds.get(c["task_id"])
        assert c["status"] is not None, c["task_id"]
    global LAST_FROZEN_RULE_RESULT
    LAST_FROZEN_RULE_RESULT = res
    return out


def status_counts(rows: List[Dict[str, Any]]) -> Dict[str, int]:
    return dict(collections.Counter(r["status"] for r in rows))


# ---------------------------------------------------------------------------
# comparison (reads prior status -- ONLY here, AFTER the decision)
# ---------------------------------------------------------------------------
def compare_to_prefix(new_rows: List[Dict[str, Any]],
                      frozen_doc: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Join the NEW statuses onto the pre-fix statuses for the report.

    This is the only place prior status is read, and it happens strictly after
    every status has been assigned.
    """
    prior = {t["task_id"]: t for t in frozen_doc["tasks"]
             if t.get("benchmark") == "impossible"}
    rows = []
    for r in new_rows:
        p = prior.get(r["task_id"], {})
        changed = p.get("status") != r["status"]
        rows.append({
            "task_id": r["task_id"],
            "source_task": r.get("source_task"),
            "repo": r.get("repo"),
            "pre_fix_status": p.get("status"),
            "pre_fix_class": p.get("class"),
            "post_fix_status": r["status"],
            "post_fix_class": r.get("class"),
            "changed": changed,
            "direction": _direction(p.get("status"), r["status"]),
            "reason_for_change": r.get("change_reason"),
            "governing_criterion": r.get("governing_criterion"),
            "evidence": r.get("evidence_summary"),
        })
    return sorted(rows, key=lambda x: x["task_id"])


_RANK = {"PRIMARY": 3, "RESERVE": 2, "EXCLUDE": 1, PENDING_STATUS: 0, None: 0}


def _direction(old: Optional[str], new: Optional[str]) -> str:
    if old == new:
        return "unchanged"
    if new == PENDING_STATUS:
        return "to_pending"
    if old == PENDING_STATUS:
        return "from_pending"
    return "promotion" if _RANK.get(new, 0) > _RANK.get(old, 0) else "demotion"


def change_direction_counts(cmp_rows: List[Dict[str, Any]]) -> Dict[str, int]:
    c = collections.Counter()
    for r in cmp_rows:
        c[f"{r['pre_fix_status']}->{r['post_fix_status']}"] += 1
    return dict(c)
