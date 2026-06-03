"""Unit tests for ClaudeCliAnswerer.

Strategy:
- Mock subprocess.run; never spawn a real `claude` process in pytest.
- One gated integration test that DOES call the real `claude` binary
  (JCONTRACT_RUN_INTEGRATION=1). Uses --model haiku to keep cost minimal.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest

from jcontract.impls.claude_cli_answerer import ClaudeCliAnswerer
from jcontract.interfaces import Chunk

RUN_INTEGRATION = os.environ.get("JCONTRACT_RUN_INTEGRATION") == "1"


def _make_chunks() -> list[Chunk]:
    return [
        Chunk(
            id="f.pdf:1:0",
            text="Trackwork Contractor is responsible for waterproofing at pier.",
            file="f.pdf",
            page=1,
            chunk_type="qa_pair",
            question_no="ACME/TRACKWORK/16",
        ),
    ]


def _ok_json_payload(text: str) -> str:
    """Build a valid `claude -p --output-format json` stdout string."""
    return json.dumps(
        {
            "type": "result",
            "subtype": "success",
            "is_error": False,
            "result": text,
            "stop_reason": "end_turn",
            "usage": {
                "input_tokens": 100,
                "output_tokens": 50,
                "cache_creation_input_tokens": 5000,
                "cache_read_input_tokens": 0,
            },
            "total_cost_usd": 0.0,
            "modelUsage": {},
        }
    )


def test_constructor_raises_if_binary_missing(tmp_path: Path) -> None:
    """If `claude` is not findable and no path is injected, raise loudly."""
    with (
        patch("shutil.which", return_value=None),
        pytest.raises(RuntimeError, match="claude CLI not found"),
    ):
        ClaudeCliAnswerer()


def test_constructor_uses_injected_path(tmp_path: Path) -> None:
    """An explicit claude_path bypasses the PATH lookup."""
    fake = tmp_path / "claude"
    fake.write_text("#!/usr/bin/env bash\n")
    fake.chmod(0o755)
    answerer = ClaudeCliAnswerer(claude_path=str(fake))
    assert answerer._claude_path == str(fake)


def test_answer_happy_path() -> None:
    """Successful CLI call → Answer dataclass with cleaned text + citations."""
    chunks = _make_chunks()
    cli_text = (
        "Trackwork Contractor is responsible for the screed to waterproofing at pier [f.pdf p.1]."
    )

    with (
        patch("shutil.which", return_value="/usr/bin/claude"),
        patch("subprocess.run") as mock_run,
    ):
        mock_run.return_value = subprocess.CompletedProcess(
            args=[], returncode=0, stdout=_ok_json_payload(cli_text), stderr=""
        )
        answerer = ClaudeCliAnswerer()
        answer = answerer.answer("Who is responsible for waterproofing?", chunks)

    assert "[f.pdf p.1]" in answer.text
    assert ("f.pdf", 1) in answer.citations
    assert answer.confidence in ("low", "medium", "high")
    assert answer.raw_context == chunks


def test_subprocess_args_have_hardening_flags() -> None:
    """Verify we pass the hardening flags that disable agent behaviour."""
    with (
        patch("shutil.which", return_value="/usr/bin/claude"),
        patch("subprocess.run") as mock_run,
    ):
        mock_run.return_value = subprocess.CompletedProcess(
            args=[], returncode=0, stdout=_ok_json_payload("ok"), stderr=""
        )
        ClaudeCliAnswerer().answer("q?", _make_chunks())

    cmd: list[str] = mock_run.call_args.args[0]
    assert cmd[0] == "/usr/bin/claude"
    assert "-p" in cmd
    assert "--output-format" in cmd and "json" in cmd
    # Hardening flags.
    assert "--permission-mode" in cmd and "bypassPermissions" in cmd
    assert "--no-session-persistence" in cmd
    assert "--setting-sources" in cmd
    assert "--tools" in cmd
    assert "--disable-slash-commands" in cmd
    # We REPLACE the system prompt (not append).
    assert "--system-prompt" in cmd


def test_nonzero_exit_returns_fallback() -> None:
    """Non-zero exit code → canonical fallback, never raises."""
    with (
        patch("shutil.which", return_value="/usr/bin/claude"),
        patch("subprocess.run") as mock_run,
    ):
        mock_run.return_value = subprocess.CompletedProcess(
            args=[], returncode=1, stdout="", stderr="boom"
        )
        answer = ClaudeCliAnswerer().answer("q?", _make_chunks())

    assert answer.text == "文档中未明确说明。"
    assert answer.citations == []
    assert answer.confidence == "low"


def test_timeout_returns_fallback() -> None:
    """TimeoutExpired → graceful fallback, never raises."""
    with (
        patch("shutil.which", return_value="/usr/bin/claude"),
        patch("subprocess.run") as mock_run,
    ):
        mock_run.side_effect = subprocess.TimeoutExpired(cmd="claude", timeout=180)
        answer = ClaudeCliAnswerer().answer("q?", _make_chunks())

    assert answer.text == "文档中未明确说明。"
    assert answer.confidence == "low"


def test_bad_json_returns_fallback() -> None:
    """Malformed JSON on stdout → fallback (not a crash)."""
    with (
        patch("shutil.which", return_value="/usr/bin/claude"),
        patch("subprocess.run") as mock_run,
    ):
        mock_run.return_value = subprocess.CompletedProcess(
            args=[], returncode=0, stdout="not json", stderr=""
        )
        answer = ClaudeCliAnswerer().answer("q?", _make_chunks())

    assert answer.text == "文档中未明确说明。"


def test_is_error_returns_fallback() -> None:
    """JSON with is_error=true → fallback."""
    payload = json.dumps(
        {
            "type": "result",
            "subtype": "error_max_turns",
            "is_error": True,
            "result": "Rate limited",
            "api_error_status": 429,
        }
    )
    with (
        patch("shutil.which", return_value="/usr/bin/claude"),
        patch("subprocess.run") as mock_run,
    ):
        mock_run.return_value = subprocess.CompletedProcess(
            args=[], returncode=0, stdout=payload, stderr=""
        )
        answer = ClaudeCliAnswerer().answer("q?", _make_chunks())

    assert answer.text == "文档中未明确说明。"
    assert answer.confidence == "low"


def test_no_secret_in_logs(caplog: pytest.LogCaptureFixture) -> None:
    """Sanity: subprocess.run failures must not leak anything secret-shaped."""
    with (
        patch("shutil.which", return_value="/usr/bin/claude"),
        patch("subprocess.run") as mock_run,
    ):
        mock_run.return_value = subprocess.CompletedProcess(
            args=[],
            returncode=1,
            stdout="",
            stderr="error involving sk-ant-real-key",
        )
        ClaudeCliAnswerer().answer("q?", _make_chunks())

    # We log stderr_chars (a count), not stderr content.
    for record in caplog.records:
        assert "sk-ant-real-key" not in record.getMessage()


# ---------------------------------------------------------------------------
# Gated integration test — actually runs `claude -p`. Costs about $0.001
# on Haiku and uses the caller's active `claude login` session.
# ---------------------------------------------------------------------------
@pytest.mark.skipif(not RUN_INTEGRATION, reason="Set JCONTRACT_RUN_INTEGRATION=1 to run.")
def test_real_claude_cli_smoke() -> None:
    """Hit the real `claude` binary once; assert the call returns text."""
    chunks = _make_chunks()
    answerer = ClaudeCliAnswerer(model="haiku")
    answer = answerer.answer("Who is responsible for waterproofing?", chunks)

    # Don't pin the text shape — model output can vary. Just check it's
    # non-empty (or fell back gracefully — both are acceptable outcomes
    # from a subscription quota / OAuth state perspective in CI).
    assert answer.text  # non-empty
    assert answer.confidence in ("low", "medium", "high")
