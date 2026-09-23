"""API backends. All model IDs come from config, never from experiment logic.

Imports of provider SDKs are deferred so the package (and the whole test
suite) works with no SDKs and no API keys installed.
"""
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from ..schemas import Usage
from .base import BaseModelBackend, GenerationResult


class _MissingSDK(RuntimeError):
    pass


#: ``reasoning_setting`` sentinel meaning "send NO reasoning parameter, so the
#: provider's own default applies". Distinct from ``None``/absent only in that
#: it is an explicit, recorded choice rather than an oversight.
PROVIDER_DEFAULT = "provider_default"


def _request_timeout(backend) -> Optional[float]:
    """Per-request wall-clock guard, in seconds, from the model config.

    A large ``max_tokens`` removes TRUNCATION but does not bound WALL-CLOCK: a
    GLM probe at ``max_tokens=65536`` was observed not returning after ~25
    minutes, while its gate call at the same ceiling returned in ~10s. Without a
    per-request timeout one hung call stalls an entire run shard indefinitely.
    This is a liveness guard, NOT an output limit -- it never truncates a
    response, it fails the call so the runner can record and retry it.

    ``request_timeout_s: null`` disables it explicitly.
    """
    if "request_timeout_s" not in backend.extra:
        return None
    v = backend.extra.get("request_timeout_s")
    return None if v is None else float(v)


class OpenAIBackend(BaseModelBackend):
    """OpenAI Responses/Chat-compatible backend.

    Two output-budget details are provider contracts, not preferences, and are
    therefore DATA from the model config (spec section 7):

    * ``max_tokens_param`` -- the request field carrying the output ceiling.
      Defaults to ``"max_tokens"``, which is what a local vLLM / SGLang server
      accepts. OpenAI *reasoning* models reject ``max_tokens`` outright
      ("Unsupported parameter: 'max_tokens' is not supported with this model.
      Use 'max_completion_tokens' instead.") and bill reasoning INSIDE that
      budget, so those configs set ``max_tokens_param: max_completion_tokens``.
    * ``temperature: null`` -- omit the field entirely. OpenAI reasoning models
      reject any non-default temperature ("Unsupported value: 'temperature'
      does not support 0.0 with this model. Only the default (1) value is
      supported."), so the only valid request is one that does not set it.

    ``reasoning_setting`` maps to ``reasoning_effort``; the ``provider_default``
    sentinel (or no value) sends nothing, leaving the provider's own default.
    """

    #: request field carrying the output ceiling when the config is silent.
    #: ``max_tokens`` keeps every existing local/vLLM config working unchanged.
    DEFAULT_MAX_TOKENS_PARAM = "max_tokens"

    def __init__(self, model_id: str, api_key_env: str = "OPENAI_API_KEY",
                 base_url: Optional[str] = None, **kwargs: Any) -> None:
        super().__init__(model_id=model_id, **kwargs)
        self.api_key_env = api_key_env
        self.base_url = base_url
        self._client = None

    @property
    def provider(self) -> str:
        return "openai"

    def _get_client(self):
        if self._client is None:
            try:
                from openai import OpenAI
            except ImportError as e:
                raise _MissingSDK("pip install openai") from e
            key = os.environ.get(self.api_key_env)
            if not key:
                raise RuntimeError(f"{self.api_key_env} is not set")
            self._client = OpenAI(api_key=key, base_url=self.base_url)
        return self._client

    def _resolve_api_model(self) -> str:
        """The value sent as the API ``model`` param. Base = the model_id."""
        return self.model_id

    def max_tokens_param(self) -> str:
        """Name of the request field that carries the output ceiling."""
        return str(self.extra.get("max_tokens_param")
                   or self.DEFAULT_MAX_TOKENS_PARAM)

    def uses_responses_api(self) -> bool:
        """Whether to send this model through ``/v1/responses``.

        DATA from the model config (``use_responses_api: true``), default False.

        WHY this exists: on ``/v1/chat/completions`` OpenAI returns **no reasoning
        content at all** for its reasoning models -- only a token count. Measured
        on `gpt-5.6-sol`: `reasoning_tokens=25`, `reasoning_text=None`. The
        Responses API with `reasoning={"summary": ...}` DOES return summarised
        reasoning (measured: 367 chars at `summary="auto"`, 399 at
        `"detailed"`). Reasoning is the mediator this study reads a crossing
        against, so "the provider withholds it" was a real gap, not a preference.

        Local vLLM / SGLang servers expose chat-completions only, so this must
        stay opt-in per config rather than becoming the default.
        """
        return bool(self.extra.get("use_responses_api"))

    def _reasoning_summary_setting(self) -> str:
        """``auto`` unless the config overrides. Only used on the Responses API."""
        return str(self.extra.get("reasoning_summary") or "auto")

    def generate(self, system: str, messages: List[Dict[str, str]],
                 max_tokens: int = 1024) -> GenerationResult:
        if self.uses_responses_api():
            return self._generate_via_responses(system, messages, max_tokens)
        client = self._get_client()
        create_kwargs: Dict[str, Any] = {
            "model": self._resolve_api_model(),
            "messages": [{"role": "system", "content": system}, *messages],
            # per provider contract: max_tokens (local/vLLM) vs
            # max_completion_tokens (OpenAI reasoning models).
            self.max_tokens_param(): max_tokens,
        }
        # temperature: null in the config means OMIT the field. Reasoning models
        # that only accept their default temperature require that.
        if self.temperature is not None:
            create_kwargs["temperature"] = self.temperature
        # reasoning depth. PROVIDER_DEFAULT (or unset) sends nothing.
        if self.reasoning_setting and self.reasoning_setting != PROVIDER_DEFAULT:
            create_kwargs["reasoning_effort"] = self.reasoning_setting
        # provider-specific passthrough (e.g. vLLM chat_template_kwargs to turn
        # off a reasoning model's <think> mode). DATA from the model config.
        extra_body = self.extra.get("extra_body")
        if extra_body:
            create_kwargs["extra_body"] = extra_body
        _to = _request_timeout(self)
        if _to is not None:
            create_kwargs["timeout"] = _to
        resp = client.chat.completions.create(**create_kwargs)
        u = getattr(resp, "usage", None)
        details = getattr(u, "completion_tokens_details", None) if u else None
        usage = Usage(
            input_tokens=getattr(u, "prompt_tokens", 0) or 0,
            output_tokens=getattr(u, "completion_tokens", 0) or 0,
            cached_input_tokens=getattr(
                getattr(u, "prompt_tokens_details", None), "cached_tokens", 0) or 0,
            reasoning_tokens=getattr(details, "reasoning_tokens", 0) or 0,
            calls=1,
        )
        self._record(usage)
        choice = resp.choices[0]
        msg = choice.message
        # vLLM may split reasoning into a separate field when a reasoning parser
        # is enabled; capture it as the mediator trace if present.
        reasoning = getattr(msg, "reasoning_content", None)
        return GenerationResult(
            text=msg.content or "", usage=usage, reasoning_text=reasoning,
            # truncation evidence: finish_reason == "length" means the output
            # budget was exhausted (reasoning models spend it on reasoning too).
            raw={"finish_reason": getattr(choice, "finish_reason", None),
                 "max_tokens_param": self.max_tokens_param(),
                 "max_tokens_sent": max_tokens},
        )

    # -- Responses API transport -------------------------------------------
    #: Responses API `incomplete_details.reason` values that mean the output
    #: ceiling was hit. Normalised to the chat vocabulary (`"length"`) so
    #: `pp.models.base.truncation_from_raw` keeps working unchanged -- the
    #: truncation detector must not need a fourth provider dialect.
    _RESPONSES_TRUNCATION_REASONS = ("max_output_tokens",)

    def _generate_via_responses(self, system: str,
                                messages: List[Dict[str, str]],
                                max_tokens: int) -> GenerationResult:
        """Same measurement, different transport. See `uses_responses_api`."""
        client = self._get_client()
        kwargs: Dict[str, Any] = {
            "model": self._resolve_api_model(),
            # system prompt travels as `instructions`, NOT as a message: the
            # contract sha is computed over this exact string either way.
            "instructions": system,
            "input": [{"role": m["role"], "content": m["content"]}
                      for m in messages],
            "max_output_tokens": max_tokens,
        }
        if self.temperature is not None:
            kwargs["temperature"] = self.temperature
        reasoning: Dict[str, Any] = {"summary": self._reasoning_summary_setting()}
        if self.reasoning_setting and self.reasoning_setting != PROVIDER_DEFAULT:
            reasoning["effort"] = self.reasoning_setting
        kwargs["reasoning"] = reasoning
        _to = _request_timeout(self)
        if _to is not None:
            kwargs["timeout"] = _to

        resp = client.responses.create(**kwargs)

        u = getattr(resp, "usage", None)
        det = getattr(u, "output_tokens_details", None) if u else None
        usage = Usage(
            input_tokens=getattr(u, "input_tokens", 0) or 0,
            output_tokens=getattr(u, "output_tokens", 0) or 0,
            cached_input_tokens=getattr(
                getattr(u, "input_tokens_details", None), "cached_tokens", 0) or 0,
            reasoning_tokens=getattr(det, "reasoning_tokens", 0) or 0,
            calls=1,
        )
        self._record(usage)

        # Reasoning summaries arrive as `reasoning` items, each with a list of
        # summary parts. They are NEVER part of `output_text`, so unlike Gemini
        # there is no risk of thinking leaking into the action payload.
        chunks: List[str] = []
        for item in (getattr(resp, "output", None) or []):
            if getattr(item, "type", "") != "reasoning":
                continue
            for s in (getattr(item, "summary", None) or []):
                t = getattr(s, "text", "") or ""
                if t:
                    chunks.append(t)
        reasoning_text = "\n".join(chunks).strip() or None

        text = getattr(resp, "output_text", None) or ""
        status = getattr(resp, "status", None)
        inc = getattr(resp, "incomplete_details", None)
        inc_reason = getattr(inc, "reason", None) if inc else None
        # Normalise to the chat vocabulary so truncation detection is unchanged.
        if inc_reason in self._RESPONSES_TRUNCATION_REASONS:
            finish = "length"
        elif status == "completed":
            finish = "stop"
        else:
            finish = status or None
        return GenerationResult(
            text=text, usage=usage, reasoning_text=reasoning_text,
            raw={"finish_reason": finish,
                 "transport": "responses",
                 "responses_status": status,
                 "incomplete_reason": inc_reason,
                 "reasoning_summary": self._reasoning_summary_setting(),
                 "max_tokens_param": "max_output_tokens",
                 "max_tokens_sent": max_tokens},
        )


class AnthropicBackend(BaseModelBackend):
    """Anthropic Messages backend.

    ``max_tokens`` on this provider is a JOINT budget: it includes thinking
    tokens as well as the visible answer, so a reasoning model's ceiling has to
    cover both.

    One further provider contract matters for LARGE ceilings. This provider's
    SDK refuses a NON-streaming request whose ``max_tokens`` implies a
    generation that could outlive the 10-minute non-streaming timeout, raising
    client-side before anything reaches the wire::

        ValueError: Streaming is required for operations that may take longer
        than 10 minutes.

    The guard is ``3600 * max_tokens / 128000 > 600`` (``_base_client.
    _calculate_nonstreaming_timeout``), i.e. it trips at ``max_tokens >=
    21334``. It is a CLIENT-SIDE TIMEOUT heuristic, not the model's output
    limit -- the Models API reports ``max_tokens: 128000`` for Fable 5.1 -- and
    the provider's own documented remedy is to stream. So this backend streams
    automatically whenever the configured ceiling exceeds that threshold and
    reassembles the final ``Message`` (identical content blocks, usage and
    ``stop_reason``), which is what lets a large, headroom-first ceiling be used
    without either clamping it to the timeout heuristic or risking a hang.
    """

    #: Highest ``max_tokens`` this provider's SDK accepts on a NON-streaming
    #: request: ``floor(128000 * 600 / 3600)``. Above it, stream instead.
    NONSTREAMING_MAX_TOKENS = 21333

    #: PROMPT CACHING. This provider does NOT cache implicitly -- caching happens
    #: only where an explicit `cache_control` breakpoint is placed. OpenAI caches
    #: automatically and Google caches partially, so this backend was the only one
    #: paying full price for re-sent history.
    #:
    #: MEASURED over two hours of Study-A/B running, `cached_input_tokens` vs
    #: `input_tokens`:
    #:
    #:     sol    (OpenAI)     5,958,557 / 6,673,713   = 89.3% cached
    #:     gemini (Google)     4,938,641 / 13,369,379  = 36.9% cached
    #:     fable  (Anthropic)          0 / 1,316,050   =  0.0% cached
    #:
    #: An agent episode resends the ENTIRE history every turn, so uncached input
    #: grows quadratically in turn count: a 20-turn episode pays for the same
    #: prefix ~20 times. Raising the turn ceiling 6 -> 50 multiplied that.
    #:
    #: Two breakpoints are placed: the system prompt (byte-identical for every
    #: turn of every episode on a benchmark, so it is the single most reusable
    #: span in the run), and the end of the current message list (so the NEXT
    #: turn's request reads this turn's prefix from cache). A cache WRITE costs
    #: more than a plain input token and a READ costs far less, so this is a clear
    #: win at any episode length above a couple of turns -- and a no-op below the
    #: provider's minimum cacheable length, which simply does not cache.
    #:
    #: This changes BILLING ONLY. The bytes the model receives are identical, so
    #: no prompt hash, contract sha, or measured behaviour is affected.
    CACHE_CONTROL: Dict[str, str] = {"type": "ephemeral"}

    def _caching_enabled(self) -> bool:
        """DATA from the model config; on by default. `prompt_caching: false` opts out."""
        v = self.extra.get("prompt_caching")
        return True if v is None else bool(v)

    def _cacheable_system(self, system: str):
        """The system prompt as a cache-marked block list.

        Returned unchanged (a bare string) when caching is disabled, so the
        opt-out path sends byte-identical requests to the pre-caching behaviour.
        """
        if not self._caching_enabled() or not system:
            return system
        return [{"type": "text", "text": system,
                 "cache_control": dict(self.CACHE_CONTROL)}]

    def _cache_marked_messages(self, messages: List[Dict[str, Any]]):
        """Copy of ``messages`` with a cache breakpoint at the end of the list.

        Only the LAST message is rewritten, and only from `str` content to a
        single text block -- an equivalent representation of the same bytes. A
        message whose content is already a block list is left alone rather than
        guessed at, since placing a breakpoint inside a structured payload we did
        not build risks changing what the model sees.
        """
        if not self._caching_enabled() or not messages:
            return messages
        out = [dict(m) for m in messages]
        last = out[-1]
        content = last.get("content")
        if not isinstance(content, str) or not content:
            return messages
        last["content"] = [{"type": "text", "text": content,
                            "cache_control": dict(self.CACHE_CONTROL)}]
        return out

    def __init__(self, model_id: str, api_key_env: str = "ANTHROPIC_API_KEY",
                 **kwargs: Any) -> None:
        super().__init__(model_id=model_id, **kwargs)
        self.api_key_env = api_key_env
        self._client = None

    @property
    def provider(self) -> str:
        return "anthropic"

    def _get_client(self):
        if self._client is None:
            try:
                import anthropic
            except ImportError as e:
                raise _MissingSDK("pip install anthropic") from e
            key = os.environ.get(self.api_key_env)
            if not key:
                raise RuntimeError(f"{self.api_key_env} is not set")
            self._client = anthropic.Anthropic(api_key=key)
        return self._client

    def _thinking_param(self) -> Optional[Dict[str, Any]]:
        """Build the ``thinking`` request field from the model config.

        Three cases, because the provider contract is model-dependent:

        * ``reasoning_setting`` unset or ``provider_default`` -> omit the field.
          On current models (Fable 5.x, Opus 5) thinking is ON by default, so
          omitting it is the provider default rather than "thinking off".
        * a digit string (e.g. ``"8000"``) -> the LEGACY fixed-budget form
          ``{"type": "enabled", "budget_tokens": N}``. Kept for pre-4.6 models;
          ``budget_tokens`` must stay below ``max_tokens`` because Anthropic's
          ``max_tokens`` INCLUDES thinking tokens. Rejected (400) by Fable 5.x.
        * any other string (i.e. ``"adaptive"``) -> ``{"type": <value>}``, the
          only on-mode current models accept.

        ``reasoning_display`` maps to ``thinking.display``. It controls
        VISIBILITY only -- thinking happens and is billed identically under
        every setting -- so ``summarized`` makes reasoning OBSERVABLE without
        changing its depth.
        """
        setting = self.reasoning_setting
        if not setting or setting == PROVIDER_DEFAULT:
            return None
        setting = str(setting)
        if setting.isdigit():
            thinking: Dict[str, Any] = {"type": "enabled",
                                        "budget_tokens": int(setting)}
        else:
            thinking = {"type": setting}
        display = self.extra.get("reasoning_display")
        if display:
            thinking["display"] = str(display)
        return thinking

    def generate(self, system: str, messages: List[Dict[str, str]],
                 max_tokens: int = 1024) -> GenerationResult:
        client = self._get_client()
        # NOTE: no `temperature`. The Messages API of current models removed the
        # sampling parameters outright -- the SDK does not even accept the
        # keyword -- so passing it through raised TypeError for every Anthropic
        # model. Depth is controlled by thinking/effort instead; a config that
        # wants to record its decoding choice uses `temperature: null`.
        kwargs: Dict[str, Any] = dict(
            model=self.extra.get("served_model_name") or self.model_id,
            system=self._cacheable_system(system),
            messages=self._cache_marked_messages(messages),
            max_tokens=max_tokens,
        )
        thinking = self._thinking_param()
        if thinking:
            kwargs["thinking"] = thinking
        effort = self.extra.get("reasoning_effort")
        if effort:
            kwargs["output_config"] = {"effort": str(effort)}
        # Large ceilings MUST stream on this provider (see class docstring).
        streamed = int(max_tokens) > self.NONSTREAMING_MAX_TOKENS
        if streamed:
            _to = _request_timeout(self)
            if _to is not None:
                kwargs["timeout"] = _to
            with client.messages.stream(**kwargs) as stream:
                resp = stream.get_final_message()
        else:
            _to = _request_timeout(self)
            if _to is not None:
                kwargs["timeout"] = _to
            resp = client.messages.create(**kwargs)
        text, reasoning = "", None
        thinking_blocks = 0
        for block in resp.content:
            btype = getattr(block, "type", None)
            if btype == "text":
                text += block.text
            elif btype == "thinking":
                thinking_blocks += 1
                reasoning = (reasoning or "") + getattr(block, "thinking", "")
        u = resp.usage
        usage = Usage(
            input_tokens=getattr(u, "input_tokens", 0) or 0,
            output_tokens=getattr(u, "output_tokens", 0) or 0,
            cached_input_tokens=getattr(u, "cache_read_input_tokens", 0) or 0,
            # 1.25x-priced cache WRITES. Omitting these hid most of a cached
            # episode's input cost.
            cache_write_tokens=getattr(u, "cache_creation_input_tokens", 0) or 0,
            calls=1,
        )
        self._record(usage)
        return GenerationResult(
            text=text, usage=usage, reasoning_text=reasoning,
            # stop_reason == "max_tokens" is this provider's truncation signal.
            # thinking_blocks proves thinking ran even when `display` is
            # "omitted" and the block text therefore comes back empty.
            raw={"stop_reason": getattr(resp, "stop_reason", None),
                 "thinking_blocks": thinking_blocks,
                 "max_tokens_param": "max_tokens",
                 "max_tokens_sent": max_tokens,
                 # whether the SDK's non-streaming timeout guard forced the
                 # streaming path (provider contract, not a preference).
                 "streamed": streamed},
        )


class GeminiBackend(BaseModelBackend):
    def __init__(self, model_id: str, api_key_env: str = "GOOGLE_API_KEY",
                 **kwargs: Any) -> None:
        super().__init__(model_id=model_id, **kwargs)
        self.api_key_env = api_key_env
        self._client = None

    @property
    def provider(self) -> str:
        return "google"

    def _get_client(self):
        if self._client is None:
            try:
                from google import genai
            except ImportError as e:
                raise _MissingSDK("pip install google-genai") from e
            key = os.environ.get(self.api_key_env)
            if not key:
                raise RuntimeError(f"{self.api_key_env} is not set")
            self._client = genai.Client(api_key=key)
        return self._client

    def _thinking_config(self) -> Dict[str, Any]:
        """`include_thoughts` always; `thinking_budget` only if configured.

        Splitting these matters: `include_thoughts` changes only what comes BACK
        (it is what makes `reasoning_text` non-None for this provider), while
        `thinking_budget` changes how the model SAMPLES. A budget of 0 is
        rejected by this model ("This model only works in thinking mode"), so an
        unset config must send no budget key at all rather than a zero.
        """
        cfg: Dict[str, Any] = {"include_thoughts": True}
        budget = self.extra.get("thinking_budget")
        if budget is not None:
            cfg["thinking_budget"] = int(budget)
        return cfg

    def generate(self, system: str, messages: List[Dict[str, str]],
                 max_tokens: int = 1024) -> GenerationResult:
        client = self._get_client()
        contents = [
            {"role": ("model" if m["role"] == "assistant" else "user"),
             "parts": [{"text": m["content"]}]}
            for m in messages
        ]
        resp = client.models.generate_content(
            model=self.model_id,
            contents=contents,
            config={"system_instruction": system,
                    "temperature": self.temperature,
                    "max_output_tokens": max_tokens,
                    # Ask for THOUGHT SUMMARIES. Without this the provider
                    # returns thought TOKEN COUNTS but no thought text, so
                    # `reasoning_text` was necessarily None for this model while
                    # `usage.reasoning_tokens` proved reasoning had happened.
                    # `include_thoughts` only changes what is RETURNED.
                    # `thinking_budget` DOES affect sampling and is therefore
                    # DATA from the model config -- it is this provider's
                    # equivalent of official ImpossibleBench's
                    # `reasoning_tokens=4096`. Omitted when the config is silent,
                    # since a budget of 0 is rejected outright by this model.
                    "thinking_config": self._thinking_config()},
        )
        um = getattr(resp, "usage_metadata", None)
        usage = Usage(
            input_tokens=getattr(um, "prompt_token_count", 0) or 0,
            output_tokens=getattr(um, "candidates_token_count", 0) or 0,
            cached_input_tokens=getattr(um, "cached_content_token_count", 0) or 0,
            reasoning_tokens=getattr(um, "thoughts_token_count", 0) or 0,
            calls=1,
        )
        self._record(usage)
        cand = (getattr(resp, "candidates", None) or [None])[0]
        fr = getattr(cand, "finish_reason", None)
        fr = getattr(fr, "name", None) or (str(fr) if fr is not None else None)
        # How many parts came back, and whether any carried no recognised kind.
        # MEASURED: on our structured-action prompt this model returns TWO parts --
        # a complete, valid action in part[0] and an EMPTY part[1] -- and the API
        # then reports `finish_reason: MALFORMED_FUNCTION_CALL` for the whole
        # response. The action text is intact and parses; the flag is a false
        # alarm. Its cause is a harness/contract interaction: the prompt names
        # `run_tests`/`apply_patch` in prose without declaring them through this
        # provider's native `tools` parameter, so its function-calling path
        # half-triggers on a tool it was never given.
        parts = getattr(getattr(cand, "content", None), "parts", None) or []
        empty_parts = sum(
            1 for p in parts
            if not any(getattr(p, k, None)
                       for k in ("text", "function_call", "inline_data",
                                 "thought", "file_data")))
        # Split THOUGHT parts from ANSWER parts. This must be explicit: with
        # `include_thoughts` on, thought summaries arrive as ordinary text parts
        # flagged `thought=True`, and `resp.text` concatenates parts. Taking
        # `resp.text` would therefore prepend the model's thinking to the action
        # payload and break strict action parsing on every turn. The answer text
        # is built from NON-thought parts only; the thinking is carried
        # separately as the mediator it is.
        thought_chunks, answer_chunks = [], []
        for p in parts:
            t = getattr(p, "text", None)
            if not t:
                continue
            (thought_chunks if getattr(p, "thought", False)
             else answer_chunks).append(t)
        reasoning = "\n".join(thought_chunks).strip() or None
        text = "".join(answer_chunks)
        if not text and not thought_chunks:
            # No parts carried text (e.g. a pure function_call response). Fall
            # back to the SDK accessor rather than silently returning "".
            try:
                text = resp.text or ""
            except Exception:                                  # noqa: BLE001
                text = ""
        return GenerationResult(
            text=text, usage=usage, reasoning_text=reasoning,
            # NOTE the vocabulary: this provider's truncation signal is
            # `MAX_TOKENS`, NOT `length`. Reading only `finish_reason == "length"`
            # left Gemini's truncation completely undetected -- and treating any
            # non-STOP value as a failure would misread 5 of 6 healthy responses.
            raw={"finish_reason": fr,
                 "n_parts": len(parts), "n_empty_parts": empty_parts,
                 "max_tokens_param": "max_output_tokens",
                 "max_tokens_sent": max_tokens},
        )


class OpenAICompatibleBackend(OpenAIBackend):
    """For vLLM / SGLang-served open-weight models (spec section 7).

    Same wire protocol as OpenAI; only base_url and key env differ.

    Two auth modes:

    * default (``auth_mode`` unset) -- Bearer key from ``api_key_env`` (local
      servers accept any key, so it defaults to ``"EMPTY"``). UNCHANGED.
    * ``auth_mode == "modal_proxy"`` -- a Modal Shared Endpoint fronted by Modal
      *proxy auth*, which authenticates with ``Modal-Key`` / ``Modal-Secret``
      request headers (read from env at CALL time) rather than a Bearer key.
      The OpenAI SDK still requires *some* api_key string, so a dummy is passed;
      it never reaches the wire as auth.

    ``served_model_name`` (optional) is the value sent as the API ``model``
    param, letting ``model_id`` stay a friendly local alias (e.g.
    ``glm-5.3-flash``) while the endpoint is addressed by its served name
    (e.g. ``zai-org/GLM-5.3-Flash``).
    """

    #: env vars carrying Modal proxy-auth credentials (values are secrets and
    #: are never logged; only their presence is ever checked).
    MODAL_KEY_ENV = "MODAL_KEY"
    MODAL_SECRET_ENV = "MODAL_SECRET"

    def __init__(self, model_id: str, base_url: str,
                 api_key_env: str = "OPENAI_COMPATIBLE_API_KEY",
                 provider_name: str = "openai_compatible",
                 auth_mode: Optional[str] = None,
                 served_model_name: Optional[str] = None,
                 **kwargs: Any) -> None:
        super().__init__(model_id=model_id, api_key_env=api_key_env,
                         base_url=base_url, **kwargs)
        self._provider_name = provider_name
        self.auth_mode = auth_mode
        self.served_model_name = served_model_name

    @property
    def provider(self) -> str:
        return self._provider_name

    def _resolve_api_model(self) -> str:
        """Use served_model_name for the wire ``model`` param when present."""
        return self.served_model_name or self.model_id

    def _get_client(self):
        if self._client is None:
            try:
                from openai import OpenAI
            except ImportError as e:
                raise _MissingSDK("pip install openai") from e
            if self.auth_mode == "modal_proxy":
                # Modal proxy auth: creds travel as headers, not a Bearer key.
                key_id = os.environ.get(self.MODAL_KEY_ENV)
                secret = os.environ.get(self.MODAL_SECRET_ENV)
                missing = [name for name, val in (
                    (self.MODAL_KEY_ENV, key_id),
                    (self.MODAL_SECRET_ENV, secret)) if not val]
                if missing:
                    raise RuntimeError(
                        "modal_proxy auth requires environment variable(s) "
                        + " and ".join(missing)
                        + " to be set (they carry the Modal proxy credentials)")
                # dummy api_key: the SDK insists on one, but auth is via headers.
                self._client = OpenAI(
                    api_key="modal-proxy-auth",
                    base_url=self.base_url,
                    default_headers={"Modal-Key": key_id,
                                     "Modal-Secret": secret},
                )
            else:
                # local servers usually accept any key
                key = os.environ.get(self.api_key_env, "EMPTY")
                self._client = OpenAI(api_key=key, base_url=self.base_url)
        return self._client
