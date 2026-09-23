"""Model-agnostic executable-runtime contract (shared; authored once).

Architecture (per user directive):

    runner -> model adapter -> action -> Modal environment -> observation

The runner (`run_episode`) is benchmark- AND model-agnostic: it drives ANY
`ExecutableEnv` with ANY `pp.models` backend (Qwen via Modal for screening; the
closed frontier models for full runs) so every model shares one
runner/environment/logging path.

The hard Study-B requirement is `snapshot()` / `restore()`: B1–B5 must fork from
the EXACT same frozen checkpoint. An env implementation MUST guarantee that
`restore(h)` reproduces byte-identical task state (files, evaluator state, turn
index) so the only thing that differs across B1–B5 is the injected board
stimulus. This module defines the contract; Modal-backed implementations live in
`impossible/` and `lhaw/`. No Modal import here — this stays import-safe offline.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable

from ..schemas import Action, Usage

#: Opaque, restorable handle to a frozen environment state (e.g. a content hash
#: plus a Modal Volume path or a serialized blob). Equality of the `token` means
#: the same restorable state; `fingerprint` is a stable content hash used to
#: PROVE two restores are identical (Study-B fork integrity).
@dataclass
class SnapshotHandle:
    token: str
    fingerprint: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StepResult:
    observation: str
    events: List[str] = field(default_factory=list)
    done: bool = False
    success: bool = False
    #: env's authorization verdict for the action, if the env adjudicates it
    authorized: Optional[bool] = None


@dataclass
class TraceStep:
    index: int
    action: Action
    observation: str
    events: List[str] = field(default_factory=list)
    authorized: Optional[bool] = None
    #: Per-call model telemetry, carried so the trajectory can record it.
    #: Previously `run_episode` discarded the whole ``GenerationResult`` except a
    #: cost read that could never work, so every recorded trajectory showed
    #: ``usage`` all-zero WITH ``calls: 0`` -- the tell that the field was a
    #: dataclass default nothing had ever assigned, not a provider that returned
    #: no usage.
    usage: Optional[Usage] = None
    #: ``finish_reason`` as the provider reported it. ``"length"`` is the
    #: provider's own truncation signal. Truncation is ALREADY caught by strict
    #: action parsing (a truncated response is not parseable, so the episode is
    #: marked measurement-invalid), so this is not the safety net -- it is what
    #: lets an invalid episode be attributed to TRUNCATION rather than to
    #: malformed model output. Those have different causes and different fixes.
    finish_reason: Optional[str] = None
    #: The model's reasoning / thinking text for THIS turn, as the provider
    #: returned it. `schemas.Turn` has always declared and serialised this field,
    #: but on the Study-A/B path nothing ever assigned it: `run_episode` had
    #: `gen.reasoning_text` in hand and dropped it, so every recorded trajectory
    #: carried `"reasoning_text": null` on every turn. Same defect class as the
    #: `usage`/`finish_reason` drop above -- a field that exists, is written out,
    #: and is never populated reads as "the provider returned no reasoning"
    #: rather than "the harness discarded it".
    #:
    #: Reasoning is a MEDIATOR, not an outcome: no metric may branch on it. It is
    #: recorded so that a crossing can be read alongside the reasoning that
    #: preceded it. Coverage is per provider and is NOT uniform -- see
    #: `reports/REASONING_CAPTURE.md`; absence here is not evidence of absence of
    #: reasoning.
    reasoning_text: Optional[str] = None


@dataclass
class Trace:
    """Everything downstream qualification/metrics needs from one episode."""
    task_id: str
    benchmark: str
    steps: List[TraceStep] = field(default_factory=list)
    evaluator_result: Optional[Dict[str, Any]] = None   # e.g. {"tests_pass": bool}
    final_state: Dict[str, Any] = field(default_factory=dict)
    parse_errors: List[str] = field(default_factory=list)
    timings_ms: Dict[str, float] = field(default_factory=dict)
    #: Accumulated across every model call in the episode. `calls` is the check
    #: that it was actually populated: an all-zero usage WITH `calls == 0` means
    #: nothing assigned it, whereas `calls > 0` with zero tokens means the
    #: provider genuinely returned no usage. Those are different faults.
    usage: Usage = field(default_factory=Usage)
    #: turn indices whose response the provider reported as `finish_reason=length`
    truncated_turns: List[int] = field(default_factory=list)
    #: one entry per turn that HALTED on an unparseable response, with that call's
    #: finish_reason, output_tokens and reasoning/content sizes. These calls produce
    #: no TraceStep, so without this the tokens they consumed and the reason they
    #: failed leave no trace at all -- and a ceiling-exhausted call with empty
    #: content is exactly such a turn.
    halted_turns: List[Dict[str, Any]] = field(default_factory=list)
    #: one entry per model call: the provider's raw finish signal and how it
    #: was classified. Kept because a provider can report an abnormal signal
    #: on a perfectly good response, and a bare boolean would hide that.
    finish_signals: List[Dict[str, Any]] = field(default_factory=list)
    #: episode budget state: the ceilings in force, any clamp applied, and
    #: whether the episode ended because a ceiling bound rather than because
    #: the model or the env ended it. `stopped_on` absent means it did not.
    budget: Dict[str, Any] = field(default_factory=dict)
    #: NOT derived here. `Usage` carries no price, and the roster's Modal
    #: endpoints are GPU/container-metered with no per-token rate, so cost is
    #: computed at analysis time from configs/pricing.yaml against `usage`.
    #: This previously read `gen.usage.cost_usd` -- a field `Usage` does not have
    #: -- so `getattr(..., 0.0)` silently returned 0.0 on every call and every
    #: recorded trajectory reported `cost_usd: 0.0`.
    cost_usd: float = 0.0
    tag: str = "SCREENING_OR_SMOKE_ONLY"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id, "benchmark": self.benchmark,
            "steps": [{"index": s.index, "action": s.action.action.value,
                       "tool": s.action.tool, "observation": s.observation,
                       "events": s.events, "authorized": s.authorized,
                       "usage": s.usage.to_dict() if s.usage else None,
                       "finish_reason": s.finish_reason}
                      for s in self.steps],
            "evaluator_result": self.evaluator_result,
            "final_state": self.final_state, "parse_errors": self.parse_errors,
            "timings_ms": self.timings_ms, "cost_usd": self.cost_usd,
            "usage": self.usage.to_dict(),
            "truncated_turns": list(self.truncated_turns),
            "halted_turns": list(self.halted_turns),
            "finish_signals": list(self.finish_signals),
            "budget": dict(self.budget),
            "tag": self.tag,
        }


@runtime_checkable
class ExecutableEnv(Protocol):
    """The contract every benchmark runtime implements (Modal-backed).

    Implementations own HOW state lives (a Modal container + Volume, a repo
    workdir, MCP servers), but MUST honour these semantics exactly.
    """

    task_id: str
    benchmark: str

    def reset(self) -> str:
        """Initialise task state; return the initial observation/prompt."""
        ...

    def step(self, action: Action) -> StepResult:
        """Apply one action; return the observation. Deterministic given state."""
        ...

    def evaluate(self) -> Dict[str, Any]:
        """Run the deterministic evaluator/oracle against current state."""
        ...

    def public_state(self) -> Dict[str, Any]:
        """State safe to expose (never hidden CoT)."""
        ...

    def snapshot(self) -> SnapshotHandle:
        """Freeze current state to a restorable handle (Study-B checkpoint)."""
        ...

    def restore(self, handle: SnapshotHandle) -> None:
        """Restore to a snapshot's EXACT state. Must be byte-identical: two
        restores of the same handle produce the same `snapshot().fingerprint`."""
        ...


#: Hard ceilings per episode, directed. Whichever binds first stops the episode.
#: `MAX_TURNS_CEILING` caps trajectory length; `OUTPUT_TOKEN_BUDGET` caps total
#: OUTPUT tokens across the episode's model calls.
#:
#: Why a token budget exists at all: GLM expands its reasoning to fill whatever
#: `max_tokens` it is given. Measured on these prompts -- one call returning
#: 65,536/65,536 output tokens, ~240,000 characters of reasoning and **zero**
#: characters of content, and a Study-B smoke where 4 of 5 calls did the same. A
#: per-call ceiling bounds one call; without an episode budget, 8 such calls cost
#: ~524k output tokens and produce no measurement at all. 128k admits roughly two
#: worst-case calls and then stops.
#: ALIGNED WITH THE OFFICIAL RELEASE. ImpossibleBench runs its SWE-bench split at
#: `message_limit=100`, which is ~50 agent turns (one assistant + one tool message
#: per turn), and `max_attempts=10`. Under v2 this ceiling was 10, so the agent had
#: a fifth of the official budget AND no way to read a file -- two deficits that
#: multiply rather than add.
MAX_TURNS_CEILING = 50

#: Episode output-token budget. Official sets NO token limit; this is retained as a
#: genuine backstop rather than a binding constraint, which the measured roster
#: supports: per-call output is ~772 (sol), ~660 (fable) and 43-57 (gemini-3.8-flash),
#: so 50 turns costs roughly 39k / 33k / 2.5k -- 3x to 50x under this budget. It
#: bound only for models that expand reasoning to fill whatever ceiling they are
#: given (GLM-5.3 returned 65,536/65,536 output tokens with ZERO content), and those
#: are no longer on the roster. A `budget.ceilings_conflict` note is still recorded
#: whenever per_call x turns could exceed it, so the tension stays visible.
OUTPUT_TOKEN_BUDGET = 128_000


def run_episode(env: ExecutableEnv, backend: Any, *, system_prompt: str,
                max_turns: int = 12,
                from_snapshot: Optional[SnapshotHandle] = None,
                extra_user_prefix: str = "",
                output_token_budget: int = OUTPUT_TOKEN_BUDGET,
                max_turns_ceiling: int = MAX_TURNS_CEILING) -> Trace:
    """Drive one episode: model-agnostic (`backend` is any pp.models backend
    exposing `generate_structured(system, messages) -> (Action, GenerationResult)`).

    For Study-B forks, pass `from_snapshot` to restore the frozen checkpoint and
    `extra_user_prefix` to inject that condition's board stimulus; the underlying
    task state is identical across B1–B5 by construction.
    """
    from ..models.base import (ActionParseError,  # local import: keep base light
                               truncation_from_raw)

    trace = Trace(task_id=env.task_id, benchmark=env.benchmark)
    if from_snapshot is not None:
        env.restore(from_snapshot)
        obs = extra_user_prefix or env.public_state().get("prompt", "")
    else:
        obs = env.reset()
        if extra_user_prefix:
            obs = extra_user_prefix + "\n" + obs
    messages = [{"role": "user", "content": obs}]
    # the turn ceiling BINDS: a caller asking for more gets the ceiling, and
    # the clamp is recorded rather than applied silently.
    effective_turns = min(int(max_turns), int(max_turns_ceiling))
    if effective_turns != max_turns:
        trace.budget["max_turns_requested"] = int(max_turns)
        trace.budget["max_turns_clamped_to"] = effective_turns
    trace.budget["max_turns"] = effective_turns
    trace.budget["output_token_budget"] = int(output_token_budget)
    # CONSISTENCY: the two ceilings can contradict each other. With a per-call
    # `max_tokens` of 65,536 (GLM's configured ceiling), two calls already
    # exceed a 128k episode budget, so the budget would bind at turn 2 and the
    # multi-turn failure-feedback loop A2 measures could never happen. Record
    # the worst case and the turn at which the budget can first bind, so the
    # conflict is visible in the artifact instead of silently truncating
    # every episode to two turns.
    per_call = int(getattr(backend, "extra", {}).get("max_tokens", 0) or 0)
    if per_call > 0:
        trace.budget["per_call_max_tokens"] = per_call
        trace.budget["worst_case_output_tokens"] = per_call * effective_turns
        turns_affordable = max(1, int(output_token_budget) // per_call)
        trace.budget["turns_affordable_at_worst_case"] = turns_affordable
        if per_call * effective_turns > int(output_token_budget):
            trace.budget["ceilings_conflict"] = (
                f"per-call {per_call} x {effective_turns} turns = "
                f"{per_call * effective_turns} exceeds the episode budget "
                f"{int(output_token_budget)}; the budget can bind as early as "
                f"turn {turns_affordable}, so the turn ceiling may never be "
                f"reached. Lower per-call max_tokens to "
                f"{int(output_token_budget) // effective_turns} to fit "
                f"{effective_turns} turns.")
    for i in range(effective_turns):
        # BUDGET GATE, checked before spending another call. Recorded as
        # `budget_exhausted` -- NEVER as the model choosing to stop, because
        # persistence length is a Study-A outcome and a harness-imposed stop
        # scored as a model stop would corrupt it.
        if trace.usage.output_tokens >= int(output_token_budget):
            trace.budget["stopped_on"] = "output_token_budget"
            trace.budget["stopped_at_turn"] = i
            trace.budget["output_tokens_at_stop"] = trace.usage.output_tokens
            break
        try:
            action, gen = backend.generate_structured(system_prompt, messages)
        except ActionParseError as e:
            trace.parse_errors.append(f"turn {i}: {e}")
            # Record the HALTED call's telemetry before breaking. `ActionParseError`
            # carries the GenerationResult precisely so this is possible: an
            # unparseable response is exactly what a truncated one produces, so
            # dropping it here made the truncation record report clean on runs where
            # most calls were truncated (observed: usage.calls 0 and
            # any_turn_truncated false while 4 of 5 calls hit the ceiling).
            failed = getattr(e, "result", None)
            if failed is not None:
                fu = getattr(failed, "usage", None)
                if fu is not None:
                    trace.usage = trace.usage + fu
                fr = (getattr(failed, "raw", None) or {}).get("finish_reason")
                trace.halted_turns.append({
                    "turn": i, "finish_reason": fr,
                    "output_tokens": getattr(fu, "output_tokens", None),
                    "reasoning_chars": len(getattr(failed, "reasoning_text", "") or ""),
                    "content_chars": len(getattr(failed, "text", "") or ""),
                    "error": str(e)[:300],
                })
                t_h, sig_h = truncation_from_raw(getattr(failed, "raw", None))
                if t_h:
                    trace.truncated_turns.append(i)
                trace.halted_turns[-1]["truncation_signal"] = sig_h
                trace.finish_signals.append({"turn": i, "signal": sig_h,
                                             "truncated": t_h, "halted": True})
            break  # never silently coerce to STOP
        # -- carry the per-call telemetry ------------------------------------
        # `gen` is a GenerationResult: text / usage: Usage / raw: dict /
        # reasoning_text. Everything except a broken cost read used to be
        # discarded here, which is why every recorded trajectory showed usage
        # all-zero WITH `calls: 0`.
        step_usage = getattr(gen, "usage", None)
        raw = getattr(gen, "raw", None) or {}
        finish_reason = raw.get("finish_reason")
        if step_usage is not None:
            trace.usage = trace.usage + step_usage
        # Truncation is decided PER PROVIDER: OpenAI/vLLM say
        # finish_reason=length, Anthropic says stop_reason=max_tokens, Gemini says
        # finish_reason=MAX_TOKENS. Reading one string detected one of the three.
        # And "anything other than a clean stop" is NOT truncation: gemini-3.8-flash
        # returns MALFORMED_FUNCTION_CALL on complete, valid responses (a trailing
        # empty part), which that rule would misread as failure 5 times in 6.
        truncated, signal = truncation_from_raw(raw)
        if truncated:
            trace.truncated_turns.append(i)
        trace.finish_signals.append({"turn": i, "signal": signal,
                                     "truncated": truncated})
        res = env.step(action)
        trace.steps.append(TraceStep(index=i, action=action,
                                     observation=res.observation,
                                     events=res.events, authorized=res.authorized,
                                     usage=step_usage,
                                     finish_reason=finish_reason,
                                     reasoning_text=getattr(
                                         gen, "reasoning_text", None)))
        messages.append({"role": "assistant",
                         "content": action.raw_text or action.action.value})
        messages.append({"role": "user", "content": res.observation})
        if res.done:
            break
    trace.evaluator_result = env.evaluate()
    trace.final_state = env.public_state()
    # cost_usd is intentionally left at 0.0: see Trace.cost_usd -- it is
    # computed at analysis time from configs/pricing.yaml against trace.usage,
    # because Usage carries no price and the Modal endpoints are GPU-metered.
    return trace
