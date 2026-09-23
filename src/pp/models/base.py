"""Model backend abstraction (spec section 8).

No model IDs appear in experiment logic; they come from config.
"""
from __future__ import annotations

import os

import json
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ..schemas import Action, ActionType, AuthoritySource, Usage


@dataclass
class GenerationResult:
    text: str
    usage: Usage = field(default_factory=Usage)
    raw: Optional[Dict[str, Any]] = None
    reasoning_text: Optional[str] = None


@dataclass
class ModelMetadata:
    provider: str
    model_id: str
    snapshot: Optional[str] = None
    #: None means "no sampling temperature was sent" -- required by providers
    #: whose reasoning models reject any non-default temperature.
    temperature: Optional[float] = 0.0
    reasoning_setting: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "provider": self.provider,
            "model_id": self.model_id,
            "snapshot": self.snapshot,
            "temperature": self.temperature,
            "reasoning_setting": self.reasoning_setting,
            **({"extra": self.extra} if self.extra else {}),
        }


class BaseModelBackend(ABC):
    """All backends normalise to this interface."""

    def __init__(self, model_id: str, temperature: Optional[float] = 0.0,
                 reasoning_setting: Optional[str] = None, **kwargs: Any) -> None:
        self.model_id = model_id
        self.temperature = temperature
        self.reasoning_setting = reasoning_setting
        self.extra = kwargs
        self._usage = Usage()

    # -- required -----------------------------------------------------------
    @abstractmethod
    def generate(self, system: str, messages: List[Dict[str, str]],
                 max_tokens: int = 1024) -> GenerationResult:
        ...

    @property
    @abstractmethod
    def provider(self) -> str:
        ...

    # -- shared -------------------------------------------------------------
    def generate_structured(self, system: str, messages: List[Dict[str, str]],
                            max_tokens: int = 1024) -> tuple[Action, GenerationResult]:
        """Generate and parse into a structured Action.

        Parse failure is surfaced, never silently coerced into a benign action.
        A per-model ``max_tokens`` (config) overrides the caller default so a
        reasoning model gets enough room to emit its <think> trace *and* the
        action JSON without truncation.
        """
        max_tokens = int(self.extra.get("max_tokens", max_tokens))
        result = self.generate(system, messages, max_tokens=max_tokens)
        try:
            action = parse_action(result.text)
        except ActionParseError as e:
            # Re-raise carrying the response, so the runner can record the halted
            # call's usage and finish_reason. Without this the telemetry of every
            # UNPARSEABLE call is lost -- and an unparseable call is exactly what a
            # truncated one produces, so the truncation record would read clean.
            e.result = result
            raise
        action.raw_text = result.text
        if result.reasoning_text is None:
            # populate the mediator trace from an inline <think> block
            _, reasoning = split_reasoning(result.text)
            result.reasoning_text = reasoning
        return action, result

    def get_usage(self) -> Usage:
        return self._usage

    def get_model_metadata(self) -> ModelMetadata:
        return ModelMetadata(
            provider=self.provider,
            model_id=self.model_id,
            snapshot=self.extra.get("snapshot"),
            temperature=self.temperature,
            reasoning_setting=self.reasoning_setting,
        )

    def _record(self, usage: Usage) -> None:
        self._usage = self._usage + usage
        _charge_process_budget(usage)

#: Each provider signals "I stopped because I ran out of output budget" in its own
#: field with its own vocabulary. Reading one string missed two of the three.
#:
#:   OpenAI / vLLM   raw["finish_reason"] == "length"
#:   Anthropic       raw["stop_reason"]   == "max_tokens"
#:   Gemini          raw["finish_reason"] == "MAX_TOKENS"
_TRUNCATED_VALUES = {
    "finish_reason": {"length", "MAX_TOKENS"},
    "stop_reason": {"max_tokens"},
}

#: Values that look abnormal but are NOT truncation, with why. Recorded so a run is
#: not scored on a false alarm.
BENIGN_FINISH_SIGNALS = {
    # MEASURED on gemini-3.8-flash with our structured-action prompt: the response
    # carries a complete, valid action in part[0] and an EMPTY part[1], and the API
    # labels the whole response MALFORMED_FUNCTION_CALL. 5 of 6 healthy responses
    # were flagged this way. Cause: the prompt names run_tests/apply_patch in prose
    # without declaring them through the provider's native `tools` parameter, so its
    # function-calling path half-triggers on a tool it was never given.
    "MALFORMED_FUNCTION_CALL":
        "complete valid content plus a trailing empty part; not truncation",
}


def truncation_from_raw(raw: Optional[Dict[str, Any]]
                        ) -> tuple[Optional[bool], str]:
    """``(was_truncated, signal)`` from a ``GenerationResult.raw``, per provider.

    Returns ``(None, ...)`` when the response carries no recognised signal at all --
    which is NOT the same as "not truncated" and must not be recorded as such.

    Deliberately does NOT treat "anything other than a clean stop" as truncation:
    see ``BENIGN_FINISH_SIGNALS``.
    """
    if not raw:
        return None, "no raw payload"
    for key, bad in _TRUNCATED_VALUES.items():
        if key in raw:
            val = raw.get(key)
            if val is None:
                return None, f"{key} absent"
            s = str(val)
            if s in bad:
                return True, f"{key}={s}"
            if s in BENIGN_FINISH_SIGNALS:
                return False, f"{key}={s} (benign: {BENIGN_FINISH_SIGNALS[s]})"
            return False, f"{key}={s}"
    return None, f"no truncation field in {sorted(raw)}"


class ActionParseError(ValueError):
    """Raised when a model response cannot be parsed into an Action.

    Carries the ``GenerationResult`` that failed to parse, when one exists.

    WHY IT MUST. `run_episode` binds `gen` from `generate_structured`'s RETURN
    value, so when this is raised the call's telemetry is unreachable and was being
    thrown away -- precisely for the turns that matter most. A GLM episode that
    burns its whole `max_tokens` on reasoning and returns empty content reported
    `usage.calls: 0` and `any_turn_truncated: false`, i.e. the truncation detector
    read as "nothing was truncated" on a run where 4 of 5 calls were truncated and
    ~86% of the output tokens spent went unrecorded. A false negative in the very
    field added to detect the failure.
    """

    def __init__(self, message: str, *, result: Any = None) -> None:
        super().__init__(message)
        #: the GenerationResult whose text could not be parsed, if available
        self.result = result


_THINK_BLOCK = re.compile(r"<think>(.*?)</think>", re.S | re.I)
_OPEN_THINK = re.compile(r"<think>", re.I)


def split_reasoning(text: str) -> tuple[str, Optional[str]]:
    """Separate a reasoning model's <think>...</think> trace from its answer.

    Returns (answer_text, reasoning_text). Reasoning is a *mediator* we log but
    keep out of the action JSON search — a reasoning model (e.g. Qwen3) routinely
    writes example braces inside <think>, which would otherwise corrupt a greedy
    JSON match. A truncated, unclosed <think> (reasoning ran past max_tokens
    before emitting the answer) leaves no answer text; that surfaces downstream
    as a genuine parse failure rather than being silently coerced.
    """
    if not text:
        return text or "", None
    blocks = _THINK_BLOCK.findall(text)
    reasoning = "\n".join(b.strip() for b in blocks).strip() or None
    answer = _THINK_BLOCK.sub("", text)
    # an unclosed <think> means everything after it is truncated reasoning
    open_m = _OPEN_THINK.search(answer)
    if open_m:
        trailing = answer[open_m.end():].strip()
        reasoning = ((reasoning + "\n" + trailing).strip() if reasoning
                     else trailing or reasoning)
        answer = answer[:open_m.start()]
    return answer.strip(), reasoning


def _extract_json_object(text: str) -> Optional[str]:
    """Return the last top-level balanced {...} object in text, or None.

    Scans for balanced braces (ignoring braces inside strings) and keeps the
    last complete object, which for an action response is the action itself
    even if earlier prose contained braces.
    """
    last = None
    depth = 0
    start = -1
    in_str = False
    esc = False
    for i, ch in enumerate(text):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and start >= 0:
                    last = text[start:i + 1]
    return last


#: Tool names that models emit in the `action` slot under the official-
#: replication prompt. Mapped to the canonical tool name; the action
#: becomes USE_ALLOWED_TOOL. Disjoint from every ActionType value, so this
#: can never shadow a real action token.
_TOOL_NAME_ALIASES = {
    "bash": "bash", "shell": "bash", "sh": "bash",
    "python": "python", "python3": "python",
    "text_editor": "text_editor", "editor": "text_editor",
    "str_replace": "text_editor", "replace": "text_editor",
    "view": "text_editor", "create": "text_editor",
    "insert": "text_editor", "undo_edit": "text_editor",
    "run_tests": "run_tests", "pytest": "run_tests",
    # NOTE the five OURS-arm tool names are deliberately ABSENT.
    #
    # This table exists to recover an OFFICIAL-arm response that puts a TOOL name
    # in the `action` slot, and its justification was that tool names are
    # disjoint from the action vocabulary. That is true of `ActionType`, but NOT
    # of the out-of-vocabulary verbs that `tests/test_incident_controls.py` pins
    # from recorded pre-fix evidence. `"apply_patch": "apply_patch"` made the OOV
    # verb `APPLY_PATCH` lowercase onto an alias, so
    #
    #   {"action":"APPLY_PATCH","tool":"shell","arguments":{"command":"ls"}}
    #
    # was silently rewritten to USE_ALLOWED_TOOL with tool='shell' -- a tool not
    # in IMPOSSIBLE_TOOLS -- instead of raising. That defeated two pinned
    # controls, including the one asserting a parse halt is never scored as a
    # behaviour. The other five OOV verbs (RUN_COMMAND, RUN_TOOL,
    # INSPECT_REPOSITORY, RESPOND, FINAL_RESPONSE) still raise correctly.
    #
    # The ours arm instructs USE_ALLOWED_TOOL + tool explicitly, so these five
    # entries bought nothing and cost the control. Scope the table to the
    # official-arm names its comment actually cites.
}


def _repair_unclosed_braces(text: str, max_missing: int = 3) -> Optional[str]:
    """Close an object the model left unterminated. Returns None if not that case.

    MEASURED DEFECT this recovers. Under the official-replication prompt, sol
    emits long shell commands and sometimes closes the inner `arguments` object
    but not the outer one:

        {"action":"bash","tool":"bash","arguments":{"cmd":"sed -n '1,220p' ..."}
                                                  ^ opens 2              closes to 1 ^

    `finish_reason` is `stop` and the content is complete apart from one brace --
    it is a formatting slip, not truncation. 11 of 35 StudyB_official episodes
    (31%) halted at turn 1 or 2 on exactly this.

    DELIBERATELY NARROW. This appends closing braces and nothing else. It does
    not guess at missing commas, unescaped quotes, or an absent `action` field,
    and it refuses when:
      * the text does not start with `{` (so prose-with-braces is untouched),
      * a string is left open (the content itself is incomplete -- that IS
        truncation and must stay an error),
      * more than `max_missing` braces are absent (a large imbalance means the
        response is malformed in some way this cannot reason about).

    Every repair is recorded on the Action as `json_repaired`, so a run can be
    re-analysed with repaired turns excluded. It is not silent.
    """
    s = text.strip()
    if not s.startswith("{"):
        return None
    depth = 0
    in_str = False
    esc = False
    for ch in s:
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth < 0:
                return None          # more closes than opens: not our case
    if in_str:
        return None                  # unterminated string -> genuine truncation
    if not (0 < depth <= max_missing):
        return None
    return s + ("}" * depth)


def parse_action(text: str) -> Action:
    """Parse a model response into an Action.

    Deliberately strict. A response that cannot be parsed raises, so that
    unparseable output is logged as an error rather than silently becoming
    a STOP (which would corrupt the premature-stop metric).
    """
    if text is None:
        raise ActionParseError("empty response")
    answer, _reasoning = split_reasoning(text)
    block = _extract_json_object(answer)
    _repaired = False
    if block is None:
        # One narrowly-defined recovery: an object the model left unterminated.
        # See `_repair_unclosed_braces` -- appends closing braces only, refuses on
        # an open string (real truncation), and is recorded on the Action.
        block = _repair_unclosed_braces(answer)
        _repaired = block is not None
    if block is None:
        raise ActionParseError(
            f"no JSON object found in response: {answer[:200]!r}")
    try:
        payload = json.loads(block)
    except json.JSONDecodeError as e:
        raise ActionParseError(f"invalid JSON: {e}; text={block[:200]!r}") from e

    raw_action = payload.get("action")
    if raw_action is None:
        raise ActionParseError("response JSON has no 'action' field")
    # TOOL NAME IN THE ACTION SLOT.
    #
    # Under the official-replication arm the prompt is official ImpossibleBench's
    # own, whose "TOOLS AVAILABLE" block presents `bash` / `python` /
    # `text_editor` as the things the agent DOES. Models therefore answer
    # `{"action": "bash", "cmd": ...}` rather than
    # `{"action": "USE_ALLOWED_TOOL", "tool": "bash", ...}`. MEASURED: 20 of 20
    # episodes halted at turn 0 with `unknown action 'bash'` / `'replace'`, and
    # sol repeated it on 3 of 3 direct retries -- it is the model's stable reading
    # of that prompt, not a formatting slip.
    #
    # This is a TRANSPORT normalisation, not a coercion: the tool names are known,
    # disjoint from the action vocabulary, and the resulting action is exactly
    # what the model asked for. It is NOT the banned "coerce an unrecognised
    # action into a benign one" -- an action that matches no tool and no
    # ActionType still raises. The rewrite is recorded on the Action so it is
    # visible in the trajectory rather than silent.
    _norm = str(raw_action).strip()
    _tool_alias = _TOOL_NAME_ALIASES.get(_norm.lower())
    if _tool_alias is not None:
        payload.setdefault("tool", _tool_alias)
        payload["_action_was_tool_name"] = _norm
        _norm = ActionType.USE_ALLOWED_TOOL.value
    try:
        action_type = ActionType(_norm.upper())
    except ValueError as e:
        raise ActionParseError(f"unknown action {raw_action!r}") from e

    claimed = payload.get("claimed_authority")
    claimed_authority = None
    if claimed:
        try:
            claimed_authority = AuthoritySource(str(claimed).strip().lower())
        except ValueError:
            claimed_authority = AuthoritySource.NO_TRACEABLE_SOURCE

    conf = payload.get("confidence")
    try:
        confidence = float(conf) if conf is not None else None
    except (TypeError, ValueError):
        confidence = None

    act = Action(
        action=action_type,
        tool=payload.get("tool"),
        arguments=payload.get("arguments") or {},
        public_message=payload.get("public_message"),
        confidence=confidence,
        claimed_authority=claimed_authority,
    )
    # PROVENANCE of any leniency applied while parsing. Both are recorded so a
    # run can be re-analysed with lenient turns excluded; neither is silent.
    if _repaired:
        act.json_repaired = True
    if payload.get("_action_was_tool_name"):
        act.action_was_tool_name = str(payload["_action_was_tool_name"])
    return act


class SpendBudgetExceeded(RuntimeError):
    """A process-wide token budget was exhausted. Fails the run, never silently.

    WHY THIS EXISTS. A Study-A/B launch fans out to 24+ concurrent shards, each
    resending the whole episode history every turn, and nothing bounded the total.
    A run therefore had no upper limit on spend other than finishing -- and if a
    parameter was wrong (as an inherited `--max-turns` once was), the money was
    gone before the defect surfaced in the artifacts. Prices in
    `configs/pricing.yaml` are also placeholders, so a dollar cap could not be
    trusted; TOKENS are ground truth, so the cap is denominated in tokens.
    """


#: Per-process cumulative token ceiling, from PP_TOKEN_BUDGET (input+output).
#: Unset or 0 means no ceiling -- the previous behaviour, so nothing changes for
#: callers that do not opt in. Per PROCESS, not per episode: with N shards the
#: effective total is N x this, which the launcher must account for.
_BUDGET_ENV = "PP_TOKEN_BUDGET"
_spent_tokens = 0


def process_tokens_spent() -> int:
    """Total input+output tokens recorded by every backend in this process."""
    return _spent_tokens


def _charge_process_budget(usage: Usage) -> None:
    """Charge the ceiling in COST-WEIGHTED units, not raw token echoes.

    The first version charged `input_tokens + output_tokens` at full weight. That
    is wrong in two different directions at once, because the providers do not
    agree on what `input_tokens` means:

    * OpenAI / Google report a TOTAL that INCLUDES the cached portion. With a
      measured 86-91% cache hit rate on `sol`, charging the total counts ~7x the
      tokens actually billed at full price -- so the ceiling fired on shards whose
      real spend was a fraction of it (observed: 9 aborts at ~800k "tokens").
    * Anthropic reports three DISJOINT classes, and the old counter silently
      omitted `cache_write_tokens` entirely -- undercharging exactly where the
      1.25x premium applies.

    A ceiling that misrepresents cost by ~7x in one direction and undercounts in
    another is not a safety net; it is noise that kills valid runs. So the charge
    now mirrors `Usage.billable_input_units`: cached reads at 0.10, cache writes
    at 1.25, fresh input and output at 1.0.
    """
    global _spent_tokens
    fresh = int(getattr(usage, "input_tokens", 0) or 0)
    cached = int(getattr(usage, "cached_input_tokens", 0) or 0)
    written = int(getattr(usage, "cache_write_tokens", 0) or 0)
    # OpenAI/Google nest `cached` INSIDE `input_tokens`; Anthropic keeps them
    # disjoint. Subtracting when nested and not otherwise is the only way to avoid
    # double-charging one provider or under-charging the other.
    if cached and cached <= fresh:
        fresh -= cached                       # nested (OpenAI / Google)
    _spent_tokens += int(fresh
                         + cached * 0.10
                         + written * 1.25
                         + (getattr(usage, "output_tokens", 0) or 0))
    raw = os.environ.get(_BUDGET_ENV)
    if not raw:
        return
    try:
        cap = int(raw)
    except ValueError:
        raise SpendBudgetExceeded(
            f"{_BUDGET_ENV}={raw!r} is not an integer; refusing to run with an "
            "unreadable spend ceiling")
    if cap > 0 and _spent_tokens > cap:
        raise SpendBudgetExceeded(
            f"process token budget exhausted: {_spent_tokens:,} > {cap:,} "
            f"({_BUDGET_ENV}). Raise the ceiling deliberately or reduce scope; "
            "this aborts rather than continuing to spend.")
