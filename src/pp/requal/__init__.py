"""Study-A Impossible **re**-qualification (agent-q2).

Rerun of executable Q2 qualification for the FULL Study-A Impossible candidate
pool (30 candidates) under the corrected runtime + period-correct per-task
environments, followed by reapplication of the FROZEN ``qual-v1`` criteria.

Design invariants enforced by this package (and by
``tests/test_impossible_requalification.py``):

1. **Prior status confers nothing.** A candidate's pre-fix
   PRIMARY/RESERVE/EXCLUDE status is *not reachable* from the decision path.
   :mod:`pp.requal.criteria` and :mod:`pp.requal.reselect` refuse to accept any
   mapping carrying a prior-status key.
2. **No treatment outcome ever enters selection.** Asserted by construction; the
   forbidden-input guard names every treatment-result artifact.
3. **Raw rc is diagnostic only.** Every criterion is evaluated against the
   model-visible *semantic failure type* (:mod:`pp.requal.taxonomy`). ``rc`` is
   recorded per replay and never read by a criterion.
4. **``runner_unavailable`` is a MEASUREMENT FAILURE, never evidence.** It can
   neither satisfy nor refute a criterion; it marks the observation invalid.
5. **The rule code is the frozen qual-v1 code.** Classification and selection
   are performed by importing ``scripts/study_a_qualify.py`` unchanged, so only
   the *evidence* differs between the pre-fix and post-fix runs, never the rule.
6. **LHAW is out of scope.** LHAW does not use the pytest/SWE runtime; its rows
   are carried through byte-identically and this package never mutates them.
"""

from .taxonomy import (  # noqa: F401
    SEMANTIC_FAILURE_TYPES,
    MEASUREMENT_FAILURE_TYPES,
    SUBSTANTIVE_EVIDENCE_TYPES,
    classify_observation,
    is_measurement_failure,
    is_substantive_failure_evidence,
)

__all__ = [
    "SEMANTIC_FAILURE_TYPES",
    "MEASUREMENT_FAILURE_TYPES",
    "SUBSTANTIVE_EVIDENCE_TYPES",
    "classify_observation",
    "is_measurement_failure",
    "is_substantive_failure_evidence",
]
