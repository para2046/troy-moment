"""Provenance-aware authorization: the deterministic ground truth (Study B) and
the mitigation check (Study C)."""
from .contract import (
    AuthorityContract,
    LEGITIMATE_SOURCES,
    authorization_valid,
)
from .mitigation import (
    ProvenanceClass,
    MitigationDecision,
    classify_source,
    check_provenance,
)

__all__ = ["AuthorityContract", "LEGITIMATE_SOURCES", "authorization_valid",
           "ProvenanceClass", "MitigationDecision", "classify_source",
           "check_provenance"]
