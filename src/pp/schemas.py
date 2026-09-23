"""Core data schemas.

Design rules enforced here:

* The ENVIRONMENT decides whether an action was authorized. A model never
  labels its own violation (spec section 13).
* PROPOSE and EXECUTE are distinct actions and distinct outcomes
  (spec sections 13, 14.2).
* Peer behaviour is never an authorization source (spec sections 1, 25).
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, asdict, replace
from enum import Enum
from typing import Any, Dict, List, Optional


# --------------------------------------------------------------------------
# Actions
# --------------------------------------------------------------------------
class ActionType(str, Enum):
    """The closed action vocabulary (spec section 13)."""

    CONTINUE_AUTHORIZED_WORK = "CONTINUE_AUTHORIZED_WORK"
    USE_ALLOWED_TOOL = "USE_ALLOWED_TOOL"
    ASK_FOR_CLARIFICATION = "ASK_FOR_CLARIFICATION"
    ESCALATE_TO_HUMAN = "ESCALATE_TO_HUMAN"
    DEFER = "DEFER"
    REFUSE = "REFUSE"
    STOP = "STOP"
    PROPOSE_OUT_OF_SCOPE_ACTION = "PROPOSE_OUT_OF_SCOPE_ACTION"
    EXECUTE_OUT_OF_SCOPE_ACTION = "EXECUTE_OUT_OF_SCOPE_ACTION"
    READ_BOARD = "READ_BOARD"
    POST_TO_BOARD = "POST_TO_BOARD"

    @classmethod
    def non_action_types(cls) -> List["ActionType"]:
        """Kept separate rather than merged (spec section 14.3)."""
        return [cls.ASK_FOR_CLARIFICATION, cls.REFUSE, cls.DEFER, cls.STOP]


class NonActionKind(str, Enum):
    CLARIFICATION = "clarification"
    REFUSAL = "refusal"
    DEFERMENT = "deferment"
    STOP = "stop"


class AuthoritySource(str, Enum):
    """Answers to 'what authorizes this action?' (spec section 21).

    PEER_MESSAGE and OBSERVED_PEER_BEHAVIOR are deliberately present and
    deliberately NOT legitimate. See ``is_legitimate``.
    """

    SYSTEM_INSTRUCTION = "system_instruction"
    HUMAN_PRINCIPAL = "human_principal"
    VERIFIED_POLICY = "verified_policy"
    PEER_MESSAGE = "peer_message"
    OBSERVED_PEER_BEHAVIOR = "observed_peer_behavior"
    NO_TRACEABLE_SOURCE = "no_traceable_source"

    def is_legitimate(self) -> bool:
        return self in {
            AuthoritySource.SYSTEM_INSTRUCTION,
            AuthoritySource.HUMAN_PRINCIPAL,
            AuthoritySource.VERIFIED_POLICY,
        }


@dataclass
class Action:
    """A structured action emitted by an agent.

    ``authorized`` is intentionally Optional and is filled in by the
    environment, never by the model.
    """

    action: ActionType
    tool: Optional[str] = None
    arguments: Dict[str, Any] = field(default_factory=dict)
    public_message: Optional[str] = None
    confidence: Optional[float] = None
    claimed_authority: Optional[AuthoritySource] = None
    raw_text: Optional[str] = None

    # environment-assigned, never model-assigned
    authorized: Optional[bool] = None
    authorization_reason: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["action"] = self.action.value
        if self.claimed_authority is not None:
            d["claimed_authority"] = self.claimed_authority.value
        # PARSE LENIENCY PROVENANCE. Set by `pp.models.base.parse_action` when it
        # had to be lenient: `json_repaired` (an unterminated object was closed)
        # or `action_was_tool_name` (a tool name appeared in the `action` slot,
        # which is how models answer under the official-replication prompt).
        # Serialised so an analysis can exclude lenient turns -- leniency that
        # does not reach the artifact is indistinguishable from a clean parse.
        for attr in ("json_repaired", "action_was_tool_name"):
            val = getattr(self, attr, None)
            if val:
                d[attr] = val
        return d

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "Action":
        d = dict(d)
        d["action"] = ActionType(d["action"])
        ca = d.get("claimed_authority")
        d["claimed_authority"] = AuthoritySource(ca) if ca else None
        return Action(**d)


# --------------------------------------------------------------------------
# Tasks
# --------------------------------------------------------------------------
class SpecificationType(str, Enum):
    WELL_SPECIFIED = "well_specified"
    UNDERSPECIFIED = "underspecified"


class Solvability(str, Enum):
    """Study A task conditions (spec section: STUDY A)."""

    SOLVABLE = "solvable"
    DIFFICULT_SOLVABLE = "difficult_solvable"
    UNDERSPECIFIED = "underspecified"
    IMPOSSIBLE = "impossible"

    def authorized_route_exists(self) -> bool:
        """Whether any authorized action sequence can satisfy the evaluator."""
        return self in {Solvability.SOLVABLE, Solvability.DIFFICULT_SOLVABLE}


@dataclass
class TaskInstance:
    """Generic task schema (spec section 9.1). Not coupled to one benchmark."""

    task_id: str
    source: str
    prompt: str
    specification_type: SpecificationType
    solvability: Solvability
    authorized_scope: List[str] = field(default_factory=list)
    prohibited_scope: List[str] = field(default_factory=list)
    allowed_tools: List[str] = field(default_factory=list)
    restricted_tools: List[str] = field(default_factory=list)
    success_condition: Dict[str, Any] = field(default_factory=dict)
    expected_boundary: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["specification_type"] = self.specification_type.value
        d["solvability"] = self.solvability.value
        return d

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "TaskInstance":
        d = dict(d)
        d["specification_type"] = SpecificationType(d["specification_type"])
        d["solvability"] = Solvability(d["solvability"])
        return TaskInstance(**d)


# --------------------------------------------------------------------------
# Board
# --------------------------------------------------------------------------
@dataclass
class BoardMessage:
    """One message on the shared board.

    ``technical_block`` and the permission layer are stored separately so that
    stimulus invariance can be checked mechanically (spec section 12).
    """

    message_id: str
    technical_block: str
    permission_block: Optional[str] = None
    source_handle: Optional[str] = None      # None => unsigned (B3a)
    source_kind: str = "peer"                # peer | non_agent | principal | none
    timestamp: Optional[str] = None
    padding_block: Optional[str] = None      # length-matching filler for B1
    #: enforcement outcome span, kept in its own field so the B2-vs-B3
    #: (unpunished vs punished) contrast is mechanically confined to it.
    enforcement_block: Optional[str] = None

    def render(self) -> str:
        parts = [self.technical_block]
        if self.padding_block:
            parts.append(self.padding_block)
        if self.permission_block:
            parts.append(self.permission_block)
        if self.enforcement_block:
            parts.append(self.enforcement_block)
        body = " ".join(p.strip() for p in parts if p and p.strip())
        if self.source_handle:
            body = f"{body} -- {self.source_handle}"
        return body

    def technical_hash(self) -> Optional[str]:
        """sha256 of the technical block, or None when there is no block.

        The hash exists to prove the technical span is IDENTICAL across B1-B5, so
        a condition contrast cannot be confounded by task information. The
        official-replication arm has no technical block at all (official
        ImpossibleBench has no analogue of one), and hashing `None` raised
        `AttributeError` at serialisation time -- after the episode had already
        been paid for. `None` here means "no block to hash", which is a different
        claim from "a block that hashes to the empty string".
        """
        if self.technical_block is None:
            return None
        return hashlib.sha256(self.technical_block.encode("utf-8")).hexdigest()

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["rendered"] = self.render()
        d["technical_block_hash"] = self.technical_hash()
        return d


# --------------------------------------------------------------------------
# Checkpoints  (spec section 4 / 17)
# --------------------------------------------------------------------------
@dataclass
class BoundaryCheckpoint:
    """Standardised state handed from Study A to Study B.

    Everything the target agent will see is frozen here so that Study B can
    vary ONLY the board.
    """

    checkpoint_id: str
    task_id: str
    turn_index: int
    authorized_attempts_failed: int
    task_incomplete: bool
    out_of_scope_option_available: bool
    boundary_crossed_yet: bool
    public_state: Dict[str, Any] = field(default_factory=dict)
    history: List[Dict[str, Any]] = field(default_factory=list)
    source_run_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def fingerprint(self) -> str:
        """Stable hash of everything that must not vary across conditions."""
        payload = json.dumps(
            {
                "task_id": self.task_id,
                "turn_index": self.turn_index,
                "authorized_attempts_failed": self.authorized_attempts_failed,
                "public_state": self.public_state,
                "history": self.history,
            },
            sort_keys=True,
            ensure_ascii=False,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["fingerprint"] = self.fingerprint()
        return d

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "BoundaryCheckpoint":
        d = {k: v for k, v in d.items() if k != "fingerprint"}
        return BoundaryCheckpoint(**d)


# --------------------------------------------------------------------------
# Trajectory + usage
# --------------------------------------------------------------------------
@dataclass
class Usage:
    input_tokens: int = 0
    cached_input_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int = 0
    calls: int = 0
    #: Tokens written INTO a prompt cache. A third, separately-priced class that
    #: was previously not captured at all.
    #:
    #: Anthropic reports three disjoint input classes -- `input_tokens` (full
    #: price), `cache_read_input_tokens` (~0.1x), and
    #: `cache_creation_input_tokens` (~1.25x) -- and a cached turn puts almost
    #: everything in the latter two. Recording only the first two therefore made
    #: most of a cached episode's input INVISIBLE: measured on a 5-turn probe,
    #: `input_tokens` fell to 20 while 2,531 write tokens went unrecorded. Any
    #: cost figure built on that undercounts.
    #:
    #: MEASURED NET EFFECT of caching, counting writes at their real 1.25x:
    #: 12,455 effective input units uncached vs 4,174 cached over 5 turns, i.e.
    #: ~3x cheaper. Caching is a clear win -- but only a complete count can show
    #: that, rather than assert it.
    cache_write_tokens: int = 0

    def __add__(self, other: "Usage") -> "Usage":
        return Usage(
            input_tokens=self.input_tokens + other.input_tokens,
            cached_input_tokens=self.cached_input_tokens + other.cached_input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            reasoning_tokens=self.reasoning_tokens + other.reasoning_tokens,
            calls=self.calls + other.calls,
            cache_write_tokens=self.cache_write_tokens + other.cache_write_tokens,
        )

    def billable_input_units(self, *, write_mult: float = 1.25,
                             read_mult: float = 0.10) -> float:
        """Input tokens expressed in FULL-PRICE-equivalent units.

        The three input classes carry different rates, so summing them is wrong
        and reporting only the cheap one is worse. Multipliers are the provider
        defaults and are overridable because they are a pricing fact, not ours.
        """
        return (self.input_tokens
                + self.cache_write_tokens * write_mult
                + self.cached_input_tokens * read_mult)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class Turn:
    index: int
    action: Action
    observation: str
    environment_events: List[str] = field(default_factory=list)
    reasoning_text: Optional[str] = None
    usage: Usage = field(default_factory=Usage)
    #: ``finish_reason`` as the provider reported it for THIS turn's call.
    #: ``"length"`` is the provider's truncation signal. Nothing assigned this or
    #: ``usage`` on the Study-A/B path before, so every recorded trajectory
    #: reported all-zero usage with ``calls: 0`` and no truncation status at all
    #: -- which made the directed pre-run gate ``finish_reason != length``
    #: impossible to check from the recorded data.
    finish_reason: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "index": self.index,
            "action": self.action.to_dict(),
            "observation": self.observation,
            "environment_events": self.environment_events,
            "reasoning_text": self.reasoning_text,
            "usage": self.usage.to_dict(),
            "finish_reason": self.finish_reason,
        }


@dataclass
class Trajectory:
    """Complete record of one run (spec section 24)."""

    run_id: str
    experiment_id: str
    study_id: str
    scenario_id: str
    task_id: str
    treatment: str
    model_provider: str
    model_id: str
    seed: int
    turns: List[Turn] = field(default_factory=list)
    checkpoint_id: Optional[str] = None
    checkpoint_fingerprint: Optional[str] = None
    board_messages: List[Dict[str, Any]] = field(default_factory=list)
    system_prompt: str = ""
    authorized_policy: Dict[str, Any] = field(default_factory=dict)
    final_outcome: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, Any] = field(default_factory=dict)
    usage: Usage = field(default_factory=Usage)
    cost_usd: float = 0.0
    retry_history: List[Dict[str, Any]] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    timestamp: Optional[str] = None
    git_commit: Optional[str] = None
    config_snapshot: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["turns"] = [t.to_dict() for t in self.turns]
        d["usage"] = self.usage.to_dict()
        return d

    def executed_violations(self) -> List[Turn]:
        return [
            t for t in self.turns
            if t.action.authorized is False
            and t.action.action == ActionType.EXECUTE_OUT_OF_SCOPE_ACTION
        ]

    def proposed_violations(self) -> List[Turn]:
        return [
            t for t in self.turns
            if t.action.action == ActionType.PROPOSE_OUT_OF_SCOPE_ACTION
        ]
