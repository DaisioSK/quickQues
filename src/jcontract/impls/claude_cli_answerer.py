"""ClaudeCliAnswerer — Answerer impl using `claude -p` (Claude Code CLI).

Why this exists:
- Claude Code subscribers (Max / Pro plans) have a flat monthly quota
  that's much cheaper than per-token API for high-volume use.
- The `claude` CLI authenticates via OAuth (`claude login`), which lands
  the token in the OS keychain. Subprocess calls inherit that auth — our
  code never touches a key.

Architecture:
- subprocess.run with `claude -p` + `--output-format json` for structured
  output. The CLI returns one JSON object with the model response + usage.
- We reuse `answer/prompt.py::build_prompt` and `answer/postprocess.py`
  (vendor-agnostic) so behavior is consistent across API / CLI / Codex.
- Hardening flags disable agent behavior we don't want here:
    --tools ""                     no tool use
    --permission-mode bypassPermissions
    --no-session-persistence       don't save chat history
    --setting-sources ""           ignore project/local settings
    --disable-slash-commands       no skills
    --system-prompt <ours>         REPLACE Claude Code's default system
                                   prompt to keep responses constrained
                                   to our citation contract

Cost vs API path:
- API (ClaudeAnswerer): pay per token at the published rate.
- CLI (this): zero per-call charge if user has a Pro/Max subscription;
  uses subscription quota. Useful when running many evals or full
  ingest+answer cycles during development.

Secret handling: NONE in this code. The OAuth token is the user's, in their
keychain, accessed only by the `claude` binary. We don't read, log, or pass it.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from typing import ClassVar

import structlog

from jcontract.answer.postprocess import compute_confidence, validate_citations
from jcontract.answer.prompt import build_prompt
from jcontract.interfaces import Answer, Chunk

logger = structlog.get_logger(__name__)

DEFAULT_MODEL = "sonnet"  # CLI alias; resolves to the latest Sonnet.
DEFAULT_TIMEOUT_S = 180  # Generous: large contexts + slower-than-API


class ClaudeCliAnswerer:
    """Answerer implementation that invokes ``claude -p`` as a subprocess.

    Uses the caller's existing `claude login` OAuth session — no API key
    is read, stored, or transmitted by this class. Token usage counts
    against the user's Claude Code subscription quota.

    Behavior matches ClaudeAnswerer (API impl) byte-for-byte downstream
    of model output: same prompt template, same citation validation, same
    Answer dataclass.
    """

    backend: ClassVar[str] = "claude-cli"

    def __init__(
        self,
        *,
        model: str = DEFAULT_MODEL,
        claude_path: str | None = None,
        timeout_s: int = DEFAULT_TIMEOUT_S,
        domain_framing: str | None = None,
    ) -> None:
        # Resolve `claude` binary at construction time so we fail loud
        # before any answer() call if it's not installed.
        resolved = claude_path or shutil.which("claude")
        if not resolved:
            raise RuntimeError(
                "claude CLI not found in PATH. "
                "Install Claude Code (https://docs.claude.com/en/docs/claude-code) "
                "and run `claude login` to authenticate with your subscription."
            )
        self._claude_path = resolved
        self._model = model
        self._timeout_s = timeout_s
        # Phase 7 SS6: framing from the active DomainProfile (None → contract default).
        self._domain_framing = domain_framing

    def answer(self, question: str, context: list[Chunk]) -> Answer:
        """Build a prompt, call `claude -p`, validate citations, return Answer."""
        system_prompt, user_message = build_prompt(
            question, context, domain_framing=self._domain_framing
        )

        cmd = [
            self._claude_path,
            "-p",
            user_message,
            "--model",
            self._model,
            "--output-format",
            "json",
            # REPLACE (not append) Claude Code's default system prompt so the
            # CLI behaves like a plain LLM with our citation contract.
            "--system-prompt",
            system_prompt,
            # Hardening: turn off agent behavior we don't want.
            "--permission-mode",
            "bypassPermissions",
            "--no-session-persistence",
            "--setting-sources",
            "",
            "--tools",
            "",
            "--disable-slash-commands",
        ]

        logger.info(
            "claude_cli.invoke",
            model=self._model,
            n_chunks=len(context),
            user_message_chars=len(user_message),
        )

        try:
            # noqa rationale: cmd is fully program-constructed (no shell=True,
            # no string concatenation of untrusted input). Binary path is
            # resolved at __init__. User-controlled inputs (question + chunk
            # text) are passed as discrete argv items, not interpolated.
            result = subprocess.run(  # noqa: S603
                cmd,
                capture_output=True,
                text=True,
                timeout=self._timeout_s,
                check=False,  # we inspect returncode manually for better errors
            )
        except subprocess.TimeoutExpired:
            logger.warning("claude_cli.timeout", timeout_s=self._timeout_s)
            return self._fallback_answer(context)

        if result.returncode != 0:
            # Don't log stderr content — could contain debug noise. Surface
            # exit code + a hint; user should re-run with --debug if needed.
            logger.warning(
                "claude_cli.nonzero_exit",
                returncode=result.returncode,
                stderr_chars=len(result.stderr),
            )
            return self._fallback_answer(context)

        # claude -p --output-format json prints exactly one JSON object on stdout.
        try:
            data = json.loads(result.stdout)
        except json.JSONDecodeError:
            logger.warning(
                "claude_cli.bad_json",
                stdout_first_200=result.stdout[:200],
            )
            return self._fallback_answer(context)

        if data.get("is_error"):
            # Map structured CLI errors to graceful fallback; preserves the
            # contract that Answerer.answer never raises on quality issues.
            logger.warning(
                "claude_cli.api_error",
                subtype=data.get("subtype"),
                api_error_status=data.get("api_error_status"),
            )
            return self._fallback_answer(context)

        raw_text: str = data.get("result", "")
        # Token-usage logging — useful for tracking subscription quota burn,
        # never logs prompt or response bodies (only counts).
        usage = data.get("usage", {})
        logger.info(
            "claude_cli.complete",
            input_tokens=usage.get("input_tokens"),
            output_tokens=usage.get("output_tokens"),
            cache_creation_input_tokens=usage.get("cache_creation_input_tokens"),
            cache_read_input_tokens=usage.get("cache_read_input_tokens"),
            total_cost_usd=data.get("total_cost_usd"),  # 0 for subscription users
            stop_reason=data.get("stop_reason"),
            response_chars=len(raw_text),
        )

        # Reuse the SAME citation validation as the API answerer — drops
        # sentences with no citation, drops fabricated cites referencing
        # pages not in `context`. Behavior parity with ClaudeAnswerer.
        cleaned_text, valid_citations, _n_dropped = validate_citations(raw_text, context)

        # Without per-chunk retrieval scores in this layer, default to medium
        # for any cited answer; downstream caller (cli.py / eval runner) can
        # override using actual top-k scores when available.
        confidence = compute_confidence([0.6, 0.6, 0.6, 0.6, 0.6])  # placeholder → "medium"

        return Answer(
            text=cleaned_text,
            citations=valid_citations,
            confidence=confidence,
            raw_context=context,
        )

    @staticmethod
    def _fallback_answer(context: list[Chunk]) -> Answer:
        """Canonical low-confidence fallback when the CLI call fails."""
        return Answer(
            text="文档中未明确说明。",
            citations=[],
            confidence="low",
            raw_context=context,
        )
