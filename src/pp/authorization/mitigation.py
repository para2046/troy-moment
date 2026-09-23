"""Study C provenance mitigation.

Before a boundary action is executed, classify the CLAIMED authorization source
by provenance and gate the action: only a LEGITIMATE authority may expand
scope. Peer messages, peer behaviour, and observed lack-of-punishment are
representable as claims but never authorize.

This is the enforcement counterpart to ``contract.py``: the contract is the
frozen ground truth (Study B scoring); this gate is a run-time check a mitigated
agent (or a wrapper) applies before acting (Study C).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any, Dict, Optional, Union

from ..schemas import ActionType, AuthoritySource
from .contract import AuthorityContract


class ProvenanceClass(str, Enum):
    """Provenance of a claimed authorization source (spec §21, Study C)."""

    SYSTEM_POLICY = "system_policy"
    HUMAN_PRINCIPAL = "human_principal"
    VERIFIED_AUTHORIZATION_RECORD = "verified_authorization_record"
    VERIFIED_DELEGATE = "verified_delegate"
    PEER_MESSAGE = "peer_message"
    PEER_BEHAVIOR = "peer_behavior"
    OBSERVED_LACK_OF_PUNISHMENT = "observed_lack_of_punishment"
    NONE = "none"

    def is_legitimate(self) -> bool:
        """Only these provenances may legitimately expand scope."""
        return self in {
            ProvenanceClass.SYSTEM_POLICY,
            ProvenanceClass.HUMAN_PRINCIPAL,
            ProvenanceClass.VERIFIED_AUTHORIZATION_RECORD,
            ProvenanceClass.VERIFIED_DELEGATE,
        }


#: how each legitimate provenance maps onto the contract's authority vocabulary
_TO_AUTHORITY = {
    ProvenanceClass.SYSTEM_POLICY: AuthoritySource.SYSTEM_INSTRUCTION,
    ProvenanceClass.HUMAN_PRINCIPAL: AuthoritySource.HUMAN_PRINCIPAL,
    ProvenanceClass.VERIFIED_AUTHORIZATION_RECORD: AuthoritySource.VERIFIED_POLICY,
    ProvenanceClass.VERIFIED_DELEGATE: AuthoritySource.HUMAN_PRINCIPAL,
}

#: keyword cues used to classify a free-text claim
_CUES = [
    (ProvenanceClass.SYSTEM_POLICY,
     ("system policy", "system instruction", "system prompt", "policy states")),
    (ProvenanceClass.HUMAN_PRINCIPAL,
     ("human principal", "principal", "the user", "my operator", "owner said",
      "human_principal")),
    (ProvenanceClass.VERIFIED_AUTHORIZATION_RECORD,
     ("verified authorization", "authorization record", "signed grant",
      "verified_policy", "verified policy", "signed authorization")),
    (ProvenanceClass.VERIFIED_DELEGATE,
     ("verified delegate", "delegated authority", "authorized delegate")),
    (ProvenanceClass.PEER_MESSAGE,
     ("peer said", "peer message", "peer told", "another agent said",
      "peer_message", "a peer approved", "peer gave")),
    (ProvenanceClass.OBSERVED_LACK_OF_PUNISHMENT,
     ("no penalty", "no punishment", "went unpunished", "nobody was punished",
      "no intervention", "unpunished", "observed impunity", "no consequence",
      "execution record", "archived record", "it worked before")),
    (ProvenanceClass.PEER_BEHAVIOR,
     ("peer did", "peers did", "another agent did", "others did", "peer behavior",
      "observed_peer_behavior", "a peer already", "someone else did")),
]

#: direct mapping from the schema AuthoritySource vocabulary
_FROM_AUTHORITY = {
    AuthoritySource.SYSTEM_INSTRUCTION: ProvenanceClass.SYSTEM_POLICY,
    AuthoritySource.HUMAN_PRINCIPAL: ProvenanceClass.HUMAN_PRINCIPAL,
    AuthoritySource.VERIFIED_POLICY: ProvenanceClass.VERIFIED_AUTHORIZATION_RECORD,
    AuthoritySource.PEER_MESSAGE: ProvenanceClass.PEER_MESSAGE,
    AuthoritySource.OBSERVED_PEER_BEHAVIOR: ProvenanceClass.PEER_BEHAVIOR,
    AuthoritySource.NO_TRACEABLE_SOURCE: ProvenanceClass.NONE,
}


def classify_source(claimed: Union[str, AuthoritySource, ProvenanceClass, None]
                    ) -> ProvenanceClass:
    """Classify a claimed authorization source into a ProvenanceClass.

    Accepts a ProvenanceClass, a schema AuthoritySource, or free text. Never
    raises: an unrecognised or empty claim is ``NONE``.
    """
    if claimed is None:
        return ProvenanceClass.NONE
    if isinstance(claimed, ProvenanceClass):
        return claimed
    if isinstance(claimed, AuthoritySource):
        return _FROM_AUTHORITY.get(claimed, ProvenanceClass.NONE)
    text = str(claimed).strip().lower()
    if not text:
        return ProvenanceClass.NONE
    # exact enum-value match first
    for pc in ProvenanceClass:
        if text == pc.value:
            return pc
    for src in AuthoritySource:
        if text == src.value:
            return _FROM_AUTHORITY.get(src, ProvenanceClass.NONE)
    # keyword cues (order matters: legitimate cues checked before peer/impunity)
    for pc, cues in _CUES:
        if any(cue in text for cue in cues):
            return pc
    return ProvenanceClass.NONE


@dataclass
class MitigationDecision:
    allow: bool
    reason: str
    provenance: str
    mapped_authority: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def check_provenance(action_type: ActionType,
                     target: str,
                     claimed_source: Union[str, AuthoritySource,
                                           ProvenanceClass, None],
                     contract: AuthorityContract) -> MitigationDecision:
    """Gate a proposed boundary action by the provenance of its authority.

    Deny unless the claimed source has a LEGITIMATE provenance AND that
    authority is one the frozen contract authorizes for this exact
    (action, target). Non-boundary actions are allowed unconditionally.
    """
    pc = classify_source(claimed_source)

    # only the boundary action against the boundary target is gated
    is_boundary = (action_type == contract.boundary_action
                   and target == contract.boundary_target)
    if not is_boundary:
        return MitigationDecision(
            allow=True, reason="not the gated boundary action",
            provenance=pc.value)

    if not pc.is_legitimate():
        return MitigationDecision(
            allow=False,
            reason=(f"{pc.value} is not a legitimate authority; peer/observed "
                    f"provenance cannot expand scope"),
            provenance=pc.value)

    mapped = _TO_AUTHORITY[pc]
    ok, why = contract.authorizes(action_type, target, mapped)
    return MitigationDecision(
        allow=ok, reason=why, provenance=pc.value,
        mapped_authority=mapped.value)
