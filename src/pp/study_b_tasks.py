"""THE canonical Study-B task list. One definition, every runner.

WHY THIS FILE EXISTS
--------------------
Study-B arms silently drifted onto different task sets, which made some
cross-arm comparisons rest on different work:

    StudyB_official            7 checkpoints, INCLUDES pylint, MISSING django
    raw_B_sol_v5_disclosed     7 checkpoints, INCLUDES pylint, MISSING django
    raw_B_disclosed            7 checkpoints, EXCLUDES pylint, HAS django
    raw_B_sol_v5_undisclosed   8 checkpoints, INCLUDES pylint, HAS django

Two independent causes:

1. `scripts/run_study_b_swarm.py` filtered pylint out with an inline string
   test, while the solo path (`scripts/run_study_b.py` ->
   `studies/study_ab_runner.py`) filtered NOTHING -- so the solo arms ran a
   checkpoint we had already documented as unusable.
2. Shard dropout removed whichever checkpoint a dead shard owned, with no
   record (see reports/DJANGO_TEST_COMMAND_DEFECT.md and the missing s1).

Cause 1 is fixed here: both runners import this list. Cause 2 needs the
completeness assertion at the bottom of this module -- a filter cannot detect a
shard that died.

EXCLUSIONS ARE NAMED AND JUSTIFIED, never silent.
"""
from __future__ import annotations

from typing import Any, Dict, Iterable, List

#: Checkpoints removed from Study B, with the reason. A checkpoint may ONLY be
#: dropped for a defect in the checkpoint itself -- never because a model did
#: badly on it, and never to make a number look better.
STUDY_B_EXCLUDED: Dict[str, str] = {
    "pylint-dev__pylint-4551":
        "UNUSABLE UPSTREAM. Both variants fail at pytest COLLECTION with "
        "ImportError: cannot import name 'get_annotation' from "
        "'pylint.pyreverse.utils' -> PP_TEST_RC=2, so no test in the module can "
        "ever run and `tests_pass` can never be True. Measured identically for "
        "`original` and `conflicting`; the test patch applies cleanly, so the "
        "defect is the commit, not our harness. Full evidence: "
        "reports/BROKEN_CHECKPOINT_PYLINT_4551.md",
}

#: The task list every Study-B arm must use. Derived, not hand-typed, so it
#: cannot drift from the manifest.
STUDY_B_N_CHECKPOINTS: int = 7


def is_impossible(rec: Any) -> bool:
    """True for an ImpossibleBench checkpoint record."""
    return getattr(rec, "benchmark", None) in ("impossiblebench", "impossible")


def excluded_reason(checkpoint_id: str) -> str:
    """The documented reason this checkpoint is out, or "" if it is in."""
    for key, reason in STUDY_B_EXCLUDED.items():
        if key in str(checkpoint_id):
            return reason
    return ""


def study_b_checkpoints(recs: Iterable[Any], *, verbose: bool = True) -> List[Any]:
    """Filter a checkpoint manifest down to THE Study-B task list.

    Every exclusion is printed with its reason -- a silently shorter list is how
    two arms end up measuring different work and nobody notices.
    """
    import sys
    kept, dropped = [], []
    for rec in recs:
        if not is_impossible(rec):
            continue
        why = excluded_reason(rec.checkpoint_id)
        (dropped if why else kept).append((rec, why))
    if verbose:
        for rec, why in dropped:
            print("[study_b_tasks] EXCLUDED %s: %s"
                  % (rec.checkpoint_id.split("::")[1], why.split(".")[0]),
                  file=sys.stderr)
        print("[study_b_tasks] task list = %d checkpoint(s): %s"
              % (len(kept), ", ".join(r.checkpoint_id.split("::")[1]
                                      for r, _ in kept)), file=sys.stderr)
    return [r for r, _ in kept]


def assert_task_list_complete(written_checkpoint_ids: Iterable[str],
                              *, arm: str = "") -> None:
    """FAIL CLOSED if an arm did not actually produce the whole task list.

    This is the check whose absence let `StudyB_official` be reported as
    complete while shard s1 (django-10554) had vanished without a log, a
    manifest, or an abort record. A filter cannot catch that -- only comparing
    what was WRITTEN against what was OWED can.
    """
    got = {str(c).split("::")[1] if "::" in str(c) else str(c)
           for c in written_checkpoint_ids}
    if len(got) != STUDY_B_N_CHECKPOINTS:
        raise SystemExit(
            "MEASUREMENT_FAILURE: arm %r wrote %d checkpoint(s), owes %d. "
            "Present: %s. A missing checkpoint is NOT a null result -- it is an "
            "incomplete run, and reporting a rate over the wrong denominator "
            "misstates the finding."
            % (arm or "?", len(got), STUDY_B_N_CHECKPOINTS, sorted(got)))


# ---------------------------------------------------------------------------
# THE ANALYSIS SET -- distinct from the RUN set above, and smaller on purpose.
# ---------------------------------------------------------------------------
#
# `study_b_checkpoints()` returns the 7 checkpoints a NEW run must produce.
# But the arms already on disk do not all cover those 7:
#
#     StudyB_official          astropy matplotlib xarray pylint pytest sklearn sphinx
#     raw_B_sol_v5_disclosed   astropy matplotlib xarray pylint pytest sklearn sphinx
#     raw_B_disclosed (x3)     astropy django matplotlib xarray pytest sklearn sphinx
#     raw_B_undisclosed (x3)   astropy django matplotlib xarray pytest sklearn sphinx
#
# The INTERSECTION of every arm is 6 checkpoints. Comparing arms on anything
# wider means comparing them on different work: `official` has no django, the
# 3-model arms have no pylint, and pylint is unusable anyway.
#
# Using the intersection is the conservative choice -- it discards real data
# (django episodes that exist for the ours arms) rather than imputing anything
# for the arm that lacks it. Report it as n=6 checkpoints, never as n=7.
STUDY_B_ANALYSIS_SET: tuple = (
    "astropy__astropy-12907",
    "matplotlib__matplotlib-20859",
    "pydata__xarray-2905",
    "pytest-dev__pytest-10051",
    "scikit-learn__scikit-learn-10297",
    "sphinx-doc__sphinx-10323",
)

#: Why each RUN-set checkpoint is absent from the ANALYSIS set.
STUDY_B_ANALYSIS_OMITTED: Dict[str, str] = {
    "django__django-10554":
        "absent from StudyB_official -- shard s1 vanished with no log, manifest "
        "or abort record, so the official arm has zero django episodes. Also "
        "carries a live prompt defect in that arm (non-runnable test command; "
        "see reports/DJANGO_TEST_COMMAND_DEFECT.md). Restore to the analysis "
        "set once the official arm is re-run on it.",
    "pylint-dev__pylint-4551":
        "UNUSABLE UPSTREAM (see STUDY_B_EXCLUDED) and absent from the 3-model "
        "arms, so it fails both tests for inclusion.",
}


def in_analysis_set(checkpoint_id: str) -> bool:
    """True if this checkpoint belongs to the cross-arm comparison set."""
    return any(k in str(checkpoint_id) for k in STUDY_B_ANALYSIS_SET)
