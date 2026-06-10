"""OpenAICompatAnswerer — Answerer impl for any OpenAI-compatible endpoint.

Why this exists (DECISION-ls.1):
- Local / self-hosted LLMs (Ollama, vLLM, LM Studio, llama.cpp server, ...)
  all expose the same OpenAI chat-completions API shape. One vendor keyed
  on a configurable ``base_url`` covers every one of them — no per-server
  vendor proliferation, and zero data leaves the machine when the endpoint
  is local (the confidential-document use case this was built for).
- We deliberately did NOT reuse the DeepSeek vendor with an overridden
  base_url: that vendor's semantics stay pinned to the official DeepSeek
  API; mixing "local endpoint" config into it would mislead readers.

Configuration (all env, all with safe local defaults — never hardcode a
non-local address per the sprint's scope rules):
  - JCONTRACT_LOCAL_LLM_BASE_URL  default ``http://localhost:11434/v1`` (Ollama)
  - JCONTRACT_LOCAL_LLM_MODEL     default ``qwen3:14b``
  - JCONTRACT_LOCAL_LLM_API_KEY   default ``ollama`` (Ollama ignores the key,
                                  but the openai SDK requires a non-empty one)

Architecture:
- Same skeleton as the other answerers: build_prompt (shared) -> one
  chat.completions call -> reasoning/think stripping -> validate_citations
  (shared) -> Answer. ONLY the inference backend differs from the Claude
  answerers — prompt assembly and citation post-processing are byte-for-byte
  the same shared pure functions, which is the fairness precondition for
  the local-vs-cloud answer A/B (DECISION-ls.10).
- Graceful degradation like the CLI answerers: an endpoint error (server
  down, model not pulled, timeout) returns the canonical low-confidence
  fallback instead of raising, so one hiccup cannot abort a 33-question
  eval run.

Reasoning-model output handling (DECISION-ls.11, live-verified 2026-06-11):
- Ollama 0.20.2's OpenAI-compat endpoint returns qwen3's chain-of-thought
  in a SEPARATE ``message.reasoning`` field; ``message.content`` is already
  clean. We read only ``content`` and never touch ``reasoning``.
- Other compat shims (and raw templates) can emit the chain-of-thought
  INLINE as ``<think>...</think>``. We strip those defensively: the probe
  showed think-text can itself contain a valid ``[file p.N]`` citation
  string, which would survive validate_citations and leak reasoning into
  the final answer if left in place.

Secret handling: the API key is read via config.get_local_llm_api_key()
only at first real call; never logged. For local Ollama it is a dummy.
"""

from __future__ import annotations

import re
from typing import ClassVar

import structlog
from openai import OpenAI, OpenAIError

from jcontract.answer.postprocess import compute_confidence, validate_citations
from jcontract.answer.prompt import build_prompt
from jcontract.config import (
    get_local_llm_api_key,
    get_local_llm_base_url,
    get_local_llm_model,
)
from jcontract.interfaces import Answer, Chunk

logger = structlog.get_logger(__name__)

# Generation parameters.
#
# temperature matches the Claude answerers (0.1) — part of the "only the
# backend differs" A/B fairness contract (DECISION-ls.10).
#
# max_tokens is HIGHER than the Claude answerers' 1024. What: 2048 budget.
# Why: on reasoning models served via OpenAI-compat shims the hidden
# chain-of-thought counts against the completion budget (live probe on
# Ollama 0.20.2 + qwen3:14b: completion_tokens=238 for a one-sentence
# answer). 1024 risks truncating the visible answer after a long think;
# 2048 gives headroom at zero cost (local inference). Context: recorded
# as part of [DECISION-ls.11].
_MAX_TOKENS = 2048
_TEMPERATURE = 0.1

# Default client-side timeout. Local first-call latency includes model
# load into VRAM (tens of seconds for a 14B on consumer GPUs) plus slow
# token generation on long contexts — far above cloud-API latencies.
DEFAULT_TIMEOUT_S = 300

# Inline chain-of-thought blocks emitted by some OpenAI-compat servers /
# chat templates for reasoning models (e.g. qwen3's <think> ... </think>).
# DOTALL: think blocks span many lines.
_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
# A truncated generation can leave an UNCLOSED <think> — everything after
# it is reasoning, never answer text, so it gets dropped too.
_THINK_OPEN_RE = re.compile(r"<think>.*\Z", re.DOTALL)


def _strip_think(text: str) -> str:
    """Remove inline ``<think>...</think>`` reasoning from model output.

    What: deletes every closed think block, then anything after a dangling
    unclosed ``<think>`` (truncated generation).
    Why: think-text may contain literal ``[file p.N]`` strings (observed in
    the 2026-06-11 live probe), so it would pass citation validation and
    contaminate the user-facing answer if not removed here, BEFORE
    validate_citations runs. Context: [DECISION-ls.11].
    """
    text = _THINK_BLOCK_RE.sub("", text)
    text = _THINK_OPEN_RE.sub("", text)
    return text.strip()


class OpenAICompatAnswerer:
    """Answerer backed by any OpenAI-compatible chat-completions endpoint.

    Primary target: a local Ollama server (zero cost, zero data egress).
    Also works against vLLM / LM Studio / any compat shim by pointing
    ``JCONTRACT_LOCAL_LLM_BASE_URL`` (or the ``base_url`` parameter) at it.

    Behavior matches the Claude answerers downstream of model output:
    same shared prompt template, same citation validation, same Answer
    dataclass — only the inference backend is swapped (DECISION-ls.10).
    """

    backend: ClassVar[str] = "local"

    def __init__(
        self,
        *,
        model: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        client: OpenAI | None = None,
        timeout_s: float = DEFAULT_TIMEOUT_S,
        domain_framing: str | None = None,
    ) -> None:
        # None → resolve from env at first use (lazy, so constructing the
        # answerer never reads the environment — mirrors ClaudeAnswerer and
        # keeps tests hermetic via the injected mock ``client``).
        self._model = model
        self._base_url = base_url
        self._api_key = api_key
        self._client = client
        self._timeout_s = timeout_s
        # Framing from the active DomainProfile (None → contract default),
        # same parameter contract as every other answerer vendor.
        self._domain_framing = domain_framing

    def _resolved_model(self) -> str:
        """Model id: explicit constructor arg wins, else env / default."""
        return self._model if self._model is not None else get_local_llm_model()

    def _ensure_client(self) -> OpenAI:
        """Lazy-create the OpenAI client pointed at the configured endpoint.

        Why lazy: same rationale as DeepSeekV4Parser._ensure_client — the
        constructor must stay side-effect free (tests inject a mock client,
        CLI may build the answerer before any call happens).
        """
        if self._client is None:
            self._client = OpenAI(
                base_url=self._base_url or get_local_llm_base_url(),
                # Ollama ignores the key but the SDK requires a non-empty
                # string; never logged either way.
                api_key=self._api_key or get_local_llm_api_key(),
                timeout=self._timeout_s,
            )
        return self._client

    def answer(self, question: str, context: list[Chunk]) -> Answer:
        """Build the shared prompt, call the endpoint, validate citations."""
        # Shared prompt assembly — identical input to what the Claude
        # answerers send, per the A/B fairness contract (DECISION-ls.10).
        system_prompt, user_message = build_prompt(
            question, context, domain_framing=self._domain_framing
        )
        model = self._resolved_model()

        # Metadata-only logging: never the question, prompt, or answer body
        # (chunk text may contain customer-supplied document content).
        logger.info(
            "openai_compat_answerer.call",
            model=model,
            n_chunks=len(context),
            user_message_chars=len(user_message),
        )

        # One plain chat call — system + user, no tools, no streaming.
        # Endpoint failures (server down / model missing / timeout) degrade
        # to the canonical fallback so a long eval run survives one hiccup,
        # matching the CLI answerers' contract.
        try:
            response = self._ensure_client().chat.completions.create(
                model=model,
                max_tokens=_MAX_TOKENS,
                temperature=_TEMPERATURE,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ],
            )
        except OpenAIError as exc:
            # error_type only — exception messages can embed the request URL
            # (and, on some failure modes, credential material).
            logger.warning(
                "openai_compat_answerer.api_error",
                model=model,
                error_type=type(exc).__name__,
            )
            return self._fallback_answer(context)

        # OpenAI response shape: choices[0].message.content (None in
        # tool-only flows, which we never trigger; guard anyway). The
        # ``reasoning`` field Ollama adds for thinking models is ignored
        # by construction — we only ever read ``content`` (DECISION-ls.11).
        content = response.choices[0].message.content if response.choices else None
        raw_text = _strip_think(content or "")

        # Token-usage logging (counts only). usage may be absent on some
        # compat shims; tolerate None.
        usage = getattr(response, "usage", None)
        logger.info(
            "openai_compat_answerer.complete",
            model=model,
            prompt_tokens=getattr(usage, "prompt_tokens", None),
            completion_tokens=getattr(usage, "completion_tokens", None),
            response_chars=len(raw_text),
        )

        # Shared citation enforcement — drops uncited sentences and
        # fabricated cites; behavior parity with every other answerer.
        cleaned_text, valid_citations, _n_dropped = validate_citations(raw_text, context)

        # Same placeholder confidence as the CLI answerers: "medium" for any
        # cited answer; callers holding real retrieval scores can override.
        confidence = compute_confidence([0.6, 0.6, 0.6, 0.6, 0.6])  # placeholder → "medium"

        return Answer(
            text=cleaned_text,
            citations=valid_citations,
            confidence=confidence,
            raw_context=context,
        )

    @staticmethod
    def _fallback_answer(context: list[Chunk]) -> Answer:
        """Canonical low-confidence fallback when the endpoint call fails."""
        return Answer(
            text="文档中未明确说明。",
            citations=[],
            confidence="low",
            raw_context=context,
        )
