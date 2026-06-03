"""Anthropic Claude implementation of the Answerer Protocol.

What
----
Default Answerer for j-contract Phase 1: wraps the Anthropic SDK, wires
up prompt assembly (answer/prompt.py) and citation enforcement
(answer/postprocess.py), and returns the Layer 0 ``Answer`` dataclass.

Why
---
Per docs/project_guideline.md §4 the answerer is a swappable interface;
this is the primary impl. DeepSeek fallback (Phase 4 S4.2) and Qwen
self-hosted alt will reuse the same prompt/postprocess pure functions.

Context — High-Risk Mode (first ANTHROPIC_API_KEY touch)
--------------------------------------------------------
Per dev-contract/12-mode-high-risk.md, this module honours three gates:

  Gate A (impact analysis): logged in the sub-sprint commit message.
  Gate B (dry-run / minimization):
    * API key is fetched ONLY at .answer() call time via
      jcontract.config.get_anthropic_api_key() — never read from
      os.environ directly here, never hardcoded.
    * Logging is minimized: model, input/output token usage, and the
      FIRST 200 CHARS of the answer text only. Full prompt body and
      full answer are never logged (chunk text may contain customer
      contract material).
    * Unit tests mock anthropic.Anthropic.messages.create so the test
      run never reaches the network.
  Gate C (audit trail): in the commit message.

Sub-sprint: p1-s1-ssC.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from anthropic import Anthropic

from jcontract.answer.postprocess import (
    compute_confidence,
    validate_citations,
)
from jcontract.answer.prompt import FALLBACK_NO_ANSWER, build_prompt
from jcontract.config import get_anthropic_api_key
from jcontract.interfaces.schema import Answer, Chunk

_log = logging.getLogger(__name__)

# Model selection.
#
# Per §ssC: "model claude-sonnet-4-5 (or the latest Sonnet available)".
# We pin a snapshot id rather than a moving alias so the prototype's
# behaviour is reproducible across runs and tests. Bumping the model is
# an explicit DECISION in a future sub-sprint, not a silent drift.
#
# DECISION (p1-s1-ssC): use claude-sonnet-4-5 (the alias). The exact
# snapshot id (e.g. claude-sonnet-4-5-20250929) can be substituted by
# passing ``model=`` to the constructor — useful for pinning in CI.
_DEFAULT_MODEL = "claude-sonnet-4-5"

# Generation parameters — fixed per §ssC.
_MAX_TOKENS = 1024
_TEMPERATURE = 0.1

# Log-truncation length for answer text. Keeps observability while
# avoiding any chance of dumping a multi-paragraph contract excerpt
# into logs that may be shipped off-host.
_ANSWER_LOG_PREFIX_LEN = 200


@dataclass
class ClaudeAnswerer:
    """Default Answerer impl backed by the Anthropic API.

    Constructor parameters:
        model:   Claude model id. Defaults to ``claude-sonnet-4-5``.
        client:  Optional pre-built ``anthropic.Anthropic`` client.
                 Injection point for tests so they can pass a mock
                 client without monkeypatching the constructor.
    """

    model: str = _DEFAULT_MODEL
    client: Anthropic | None = None
    # Phase 7 SS6: domain framing from the active DomainProfile. None →
    # the construction (contract) default in build_prompt (unchanged).
    domain_framing: str | None = None

    def _get_client(self) -> Anthropic:
        """Lazy client construction.

        Why lazy: keeps ``ClaudeAnswerer()`` cheap and side-effect free
        (no env access at import or instantiation), which matters because
        the eval harness may construct one before a test mocks anything.
        """
        if self.client is None:
            # The api_key call may raise RuntimeError if env is missing;
            # we let it propagate. The error message names only the key,
            # never the value.
            self.client = Anthropic(api_key=get_anthropic_api_key())
        return self.client

    def answer(self, question: str, context: list[Chunk]) -> Answer:
        """Produce a Chinese, citation-bound Answer.

        Pipeline:
          1. Assemble (system_prompt, user_message) — pure function.
          2. Call Anthropic ``messages.create``. Any SDK error propagates
             unchanged; we do NOT swallow exceptions.
          3. Extract the first text block from the response.
          4. Run citation guardrails (validate_citations).
          5. Compute confidence from chunk-derived signal (placeholder:
             returns "low" when no scores are available — the integrator
             can wrap this and inject retrieval scores at the fusion site).

        Args:
            question: The user's question, typically in Chinese.
            context:  Retrieved chunks (already ranked & truncated).

        Returns:
            ``Answer`` with text in Chinese, validated citations, a
            confidence label, and the chunks we fed the model (so eval
            and audit can reconstruct the context).

        Raises:
            anthropic.APIError (and subclasses): network / auth /
            rate-limit failures bubble up.
            RuntimeError: if ANTHROPIC_API_KEY is unset (from config).
        """
        system_prompt, user_message = build_prompt(
            question, context, domain_framing=self.domain_framing
        )

        client = self._get_client()

        # We deliberately do NOT log the question or the user_message.
        # Both may contain customer-supplied PDF text. Log only the
        # call metadata.
        _log.info(
            "claude_answerer.call",
            extra={"model": self.model, "n_chunks": len(context)},
        )

        # SDK call. Errors propagate intentionally per Protocol contract.
        response = client.messages.create(
            model=self.model,
            max_tokens=_MAX_TOKENS,
            temperature=_TEMPERATURE,
            system=system_prompt,
            messages=[{"role": "user", "content": user_message}],
        )

        raw_text = _extract_text(response)

        # Token-usage logging (no content). ``usage`` is part of every
        # Message response per the SDK type Message.usage: Usage.
        usage = getattr(response, "usage", None)
        if usage is not None:
            _log.info(
                "claude_answerer.usage",
                extra={
                    "input_tokens": getattr(usage, "input_tokens", None),
                    "output_tokens": getattr(usage, "output_tokens", None),
                },
            )

        cleaned_text, citations, n_dropped = validate_citations(raw_text, context)

        # Truncated answer log — under no circumstances dump the full body.
        _log.info(
            "claude_answerer.result",
            extra={
                "answer_prefix": cleaned_text[:_ANSWER_LOG_PREFIX_LEN],
                "n_sentences_dropped": n_dropped,
                "n_valid_citations": len(citations),
            },
        )

        # Confidence: with no retrieval scores threaded through here,
        # we default to "low" on the canonical fallback, otherwise we
        # leave the caller to override (an integrator wrapper that
        # holds the SearchResult.score list can call
        # ``compute_confidence`` and replace the field).
        #
        # Why not raise the abstraction up to the protocol? The
        # Answerer Protocol takes only ``list[Chunk]``, not
        # ``list[SearchResult]`` — preserving that simplicity matters
        # for swap-ability. Phase 2 may extend the contract.
        if cleaned_text.strip() == FALLBACK_NO_ANSWER:
            confidence = compute_confidence([])
        else:
            # Without scores we conservatively call it "medium": the
            # model produced a cited answer, but we lack the retrieval
            # signal to claim "high". Integrator can override.
            confidence = "medium"

        return Answer(
            text=cleaned_text,
            citations=citations,
            confidence=confidence,
            raw_context=context,
        )


def _extract_text(response: object) -> str:
    """Pull the concatenated text from an Anthropic Message response.

    The SDK returns ``response.content`` as a list of content blocks;
    for non-tool-use answers there is one ``TextBlock`` with ``.text``.
    We tolerate (a) blocks that don't have ``.text`` (e.g. tool_use)
    by skipping them, and (b) responses with no content at all by
    returning the fallback string so downstream guardrails handle it.

    Kept as a free function so tests can build minimal fakes:
        type("R", (), {"content": [type("B", (), {"text": "..."})]})
    """
    content = getattr(response, "content", None)
    if not content:
        return FALLBACK_NO_ANSWER

    parts: list[str] = []
    for block in content:
        text = getattr(block, "text", None)
        if isinstance(text, str):
            parts.append(text)

    if not parts:
        return FALLBACK_NO_ANSWER

    return "".join(parts).strip()
