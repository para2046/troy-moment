"""Task-qualification measurement logic (Q2 executable validation).

This package implements the ``observed_*`` measurement fields defined in
``configs/qualification_criteria.yaml`` (§7, §8, §16, §23). Everything here is
RUNTIME-AGNOSTIC: it operates purely on a completed ``pp.runtimes.base.Trace``
and never imports Modal, touches the network, or drives an environment. The
runtime/Modal layer (which produces the Trace) is owned elsewhere.
"""
from .observed import (
    observed_meaningful_action_count,
    observed_failure_feedback_cycles,
    observed_turn_to_resolution,
    observed_env_evaluator_ok,
    observed_ambiguity_or_infeasibility_in_env,
    checkpoint_reachable,
    observed_fields,
)

__all__ = [
    "observed_meaningful_action_count",
    "observed_failure_feedback_cycles",
    "observed_turn_to_resolution",
    "observed_env_evaluator_ok",
    "observed_ambiguity_or_infeasibility_in_env",
    "checkpoint_reachable",
    "observed_fields",
]
