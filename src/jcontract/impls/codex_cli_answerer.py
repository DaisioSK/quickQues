"""CodexCliAnswerer — Answerer impl using OpenAI's `codex` CLI.

Why this exists:
- OpenAI Codex CLI (https://github.com/openai/codex) lets ChatGPT Plus /
  Pro / Team subscribers run the model headlessly using OAuth from their
  ChatGPT account. Same value proposition as ClaudeCliAnswerer: flat
  monthly fee instead of per-token API.

Status (2026-05-28): SKELETON. This implementation is structured by
analogy with ClaudeCliAnswerer but **not validated end-to-end on this
session's environment** because `codex` is not installed locally. The
arg shape below follows OpenAI's documented CLI behavior; users
installing the CLI should verify the JSON output structure matches
what we parse (see _parse_output for the exact fields we look at).

Architecture: identical to ClaudeCliAnswerer (subprocess.run +
shared prompt + shared citation validation). The two classes are
intentionally near-duplicates — refactor to a common base only when a
third CLI variant lands (N=3 abstraction rule per project_guideline §5).

Once validated, the user runs:
    codex login                    # OAuth via ChatGPT account
    jcontract evaluate --answerer codex-cli
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

# OpenAI Codex CLI default model alias; user can override per-invocation.
DEFAULT_MODEL = "gpt-5"
DEFAULT_TIMEOUT_S = 180


class CodexCliAnswerer:
    """Answerer that invokes `codex` (OpenAI CLI) as a subprocess.

    Relies on the user having run `codex login` to bind their ChatGPT
    subscription. We never read, store, or transmit any auth token.
    """

    backend: ClassVar[str] = "codex-cli"

    def __init__(
        self,
        *,
        model: str = DEFAULT_MODEL,
        codex_path: str | None = None,
        timeout_s: int = DEFAULT_TIMEOUT_S,
        domain_framing: str | None = None,
    ) -> None:
        resolved = codex_path or shutil.which("codex")
        if not resolved:
            raise RuntimeError(
                "codex CLI not found in PATH. "
                "Install OpenAI Codex (https://github.com/openai/codex) and "
                "run `codex login` to authenticate with your ChatGPT subscription. "
                "Then re-run with --answerer codex-cli."
            )
        self._codex_path = resolved
        self._model = model
        self._timeout_s = timeout_s
        # Phase 7 SS6: framing from the active DomainProfile (None → contract default).
        self._domain_framing = domain_framing

    def answer(self, question: str, context: list[Chunk]) -> Answer:
        """Same shape as ClaudeCliAnswerer.answer; differs only in CLI args."""
        system_prompt, user_message = build_prompt(
            question, context, domain_framing=self._domain_framing
        )

        # Compose the prompt for codex: codex CLI uses --prompt / -p plus
        # optional --system-prompt depending on version. The arg shape below
        # follows the documented non-interactive ("exec") mode of recent
        # codex CLI versions; tweak if your codex version differs.
        cmd = [
            self._codex_path,
            "exec",  # non-interactive subcommand
            "--model",
            self._model,
            "--json",  # structured output
            "--no-color",
            "--sandbox",
            "read-only",  # codex shouldn't touch the filesystem
            "--full-auto",  # don't prompt for permission
            "--system-prompt",
            system_prompt,
            user_message,
        ]

        logger.info(
            "codex_cli.invoke",
            model=self._model,
            n_chunks=len(context),
            user_message_chars=len(user_message),
        )

        try:
            # noqa rationale: same as ClaudeCliAnswerer — argv is program-constructed,
            # no shell=True, no untrusted string concatenation.
            result = subprocess.run(  # noqa: S603
                cmd,
                capture_output=True,
                text=True,
                timeout=self._timeout_s,
                check=False,
            )
        except subprocess.TimeoutExpired:
            logger.warning("codex_cli.timeout", timeout_s=self._timeout_s)
            return self._fallback_answer(context)

        if result.returncode != 0:
            logger.warning(
                "codex_cli.nonzero_exit",
                returncode=result.returncode,
                stderr_chars=len(result.stderr),
            )
            return self._fallback_answer(context)

        raw_text = self._parse_output(result.stdout)
        if raw_text is None:
            logger.warning("codex_cli.unparseable", stdout_first_200=result.stdout[:200])
            return self._fallback_answer(context)

        cleaned_text, valid_citations, _ = validate_citations(raw_text, context)
        confidence = compute_confidence([0.6, 0.6, 0.6, 0.6, 0.6])  # → "medium"

        return Answer(
            text=cleaned_text,
            citations=valid_citations,
            confidence=confidence,
            raw_context=context,
        )

    @staticmethod
    def _parse_output(stdout: str) -> str | None:
        """Extract the model's response text from codex CLI's JSON output.

        codex CLI streams JSONL (newline-delimited JSON events) in --json
        mode. The final message-complete event carries the answer.

        If the schema changes upstream, update here and add a fixture to
        tests/test_codex_cli_answerer.py to lock it.
        """
        # Try the whole stdout as one JSON first (older codex versions).
        try:
            data = json.loads(stdout)
            if isinstance(data, dict) and "message" in data:
                return str(data["message"])
            if isinstance(data, dict) and "result" in data:
                return str(data["result"])
        except json.JSONDecodeError:
            pass

        # JSONL fallback — iterate events, take the last text payload.
        last_text: str | None = None
        for line in stdout.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            # Recognised shapes (be permissive while codex CLI stabilises):
            for key in ("message", "result", "content", "text"):
                if isinstance(event, dict) and key in event:
                    candidate = event[key]
                    if isinstance(candidate, str):
                        last_text = candidate

        return last_text

    @staticmethod
    def _fallback_answer(context: list[Chunk]) -> Answer:
        """Canonical low-confidence fallback when the CLI call fails."""
        return Answer(
            text="文档中未明确说明。",
            citations=[],
            confidence="low",
            raw_context=context,
        )
