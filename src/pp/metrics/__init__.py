"""Metrics. Persistence, boundary, board, escalation, utility, mediators."""
from .persistence import compute_persistence, PersistenceMetrics
from .authorization import (
    compute_boundary, compute_board, compute_escalation,
    compute_non_action, compute_utility, compute_mediators,
    BoundaryMetrics, BoardMetrics, EscalationMetrics,
    NonActionMetrics, UtilityMetrics, MediatorFlags,
)
from .social_effects import (
    compute_effects, bootstrap_diff, mde_two_proportion, CONTRASTS, Estimate,
)
from .crossing import derive_crossing, CrossingDerivation
from .mitigation_metrics import (
    false_permission_reduction, legitimate_authority_retention,
    crossing_counts, compute_mitigation_metrics,
    MitigationMetric, MitigationReport,
)
from .propagation import (
    compute_propagation, adoption_after_exposure_contrast,
    PropagationFunnel, PropagationContrast,
)
__all__ = [
    "compute_persistence", "PersistenceMetrics",
    "compute_boundary", "compute_board", "compute_escalation",
    "compute_non_action", "compute_utility", "compute_mediators",
    "BoundaryMetrics", "BoardMetrics", "EscalationMetrics",
    "NonActionMetrics", "UtilityMetrics", "MediatorFlags",
    "compute_effects", "bootstrap_diff", "mde_two_proportion",
    "CONTRASTS", "Estimate",
    "derive_crossing", "CrossingDerivation",
    "false_permission_reduction", "legitimate_authority_retention",
    "crossing_counts", "compute_mitigation_metrics",
    "MitigationMetric", "MitigationReport",
    "compute_propagation", "adoption_after_exposure_contrast",
    "PropagationFunnel", "PropagationContrast",
]
