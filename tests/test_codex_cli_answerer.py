"""Unit tests for CodexCliAnswerer.

Codex CLI is not installed on this dev environment, so all tests here
mock subprocess.run. The shape is parallel to test_claude_cli_answerer
so a real-codex smoke is easy to add when the binary lands.
"""

from __future__ import annotations

import json
import subprocess
from unittest.mock import patch

import pytest

from jcontract.impls.codex_cli_answerer import CodexCliAnswerer
from jcontract.interfaces import Chunk


def _make_chunks() -> list[Chunk]:
    return [
        Chunk(
            id="f.pdf:1:0",
            text="Trackwork Contractor is responsible for waterproofing at pier.",
            file="f.pdf",
            page=1,
            chunk_type="qa_pair",
        ),
    ]


def test_constructor_raises_if_binary_missing() -> None:
    with (
        patch("shutil.which", return_value=None),
        pytest.raises(RuntimeError, match="codex CLI not found"),
    ):
        CodexCliAnswerer()


def test_parses_single_json_object() -> None:
    """Older codex versions print one JSON object."""
    payload = json.dumps({"message": "Answer text [f.pdf p.1]."})
    with (
        patch("shutil.which", return_value="/usr/bin/codex"),
        patch("subprocess.run") as mock_run,
    ):
        mock_run.return_value = subprocess.CompletedProcess(
            args=[], returncode=0, stdout=payload, stderr=""
        )
        answer = CodexCliAnswerer().answer("q?", _make_chunks())

    assert "f.pdf p.1" in answer.text
    assert ("f.pdf", 1) in answer.citations


def test_parses_jsonl_event_stream() -> None:
    """Newer codex versions stream JSONL events; take the last text payload."""
    payload_lines = [
        json.dumps({"event": "thinking", "content": "..."}),
        json.dumps({"event": "partial", "text": "draft answer"}),
        json.dumps({"event": "complete", "message": "Final answer [f.pdf p.1]."}),
    ]
    with (
        patch("shutil.which", return_value="/usr/bin/codex"),
        patch("subprocess.run") as mock_run,
    ):
        mock_run.return_value = subprocess.CompletedProcess(
            args=[], returncode=0, stdout="\n".join(payload_lines), stderr=""
        )
        answer = CodexCliAnswerer().answer("q?", _make_chunks())

    assert "Final answer" in answer.text
    assert ("f.pdf", 1) in answer.citations


def test_nonzero_exit_returns_fallback() -> None:
    with (
        patch("shutil.which", return_value="/usr/bin/codex"),
        patch("subprocess.run") as mock_run,
    ):
        mock_run.return_value = subprocess.CompletedProcess(
            args=[], returncode=2, stdout="", stderr="auth required"
        )
        answer = CodexCliAnswerer().answer("q?", _make_chunks())

    assert answer.text == "文档中未明确说明。"
    assert answer.confidence == "low"


def test_unparseable_output_returns_fallback() -> None:
    with (
        patch("shutil.which", return_value="/usr/bin/codex"),
        patch("subprocess.run") as mock_run,
    ):
        mock_run.return_value = subprocess.CompletedProcess(
            args=[], returncode=0, stdout="garbage that isn't json", stderr=""
        )
        answer = CodexCliAnswerer().answer("q?", _make_chunks())

    assert answer.text == "文档中未明确说明。"


def test_timeout_returns_fallback() -> None:
    with (
        patch("shutil.which", return_value="/usr/bin/codex"),
        patch("subprocess.run") as mock_run,
    ):
        mock_run.side_effect = subprocess.TimeoutExpired(cmd="codex", timeout=180)
        answer = CodexCliAnswerer().answer("q?", _make_chunks())

    assert answer.text == "文档中未明确说明。"


def test_subprocess_args_include_sandbox_hardening() -> None:
    """codex exec must be sandboxed read-only so it can't touch the FS."""
    payload = json.dumps({"message": "ok"})
    with (
        patch("shutil.which", return_value="/usr/bin/codex"),
        patch("subprocess.run") as mock_run,
    ):
        mock_run.return_value = subprocess.CompletedProcess(
            args=[], returncode=0, stdout=payload, stderr=""
        )
        CodexCliAnswerer().answer("q?", _make_chunks())

    cmd: list[str] = mock_run.call_args.args[0]
    assert cmd[0] == "/usr/bin/codex"
    assert "exec" in cmd
    assert "--sandbox" in cmd and "read-only" in cmd
    assert "--full-auto" in cmd  # no interactive permission prompts
    assert "--system-prompt" in cmd
