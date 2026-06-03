"""Unit tests for the shared claude-CLI runner (no subprocess actually spawned)."""

from __future__ import annotations

import json
import subprocess
import types
from pathlib import Path

import pytest

from jcontract.impls._claude_cli_runner import run_claude_read_image, run_claude_text

_RUN_TARGET = "jcontract.impls._claude_cli_runner.subprocess.run"


def _fake_completed(stdout: str, returncode: int = 0) -> types.SimpleNamespace:
    return types.SimpleNamespace(returncode=returncode, stdout=stdout, stderr="")


def test_builds_expected_argv_and_returns_envelope(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    captured_cmd: list[str] = []
    captured_kwargs: dict[str, object] = {}

    def fake_run(cmd: list[str], **kwargs: object) -> types.SimpleNamespace:
        captured_cmd.extend(cmd)
        captured_kwargs.update(kwargs)
        return _fake_completed(json.dumps({"result": "hello", "usage": {"input_tokens": 5}}))

    monkeypatch.setattr(_RUN_TARGET, fake_run)

    data = run_claude_read_image(
        claude_path="/usr/bin/claude",
        render_dir=tmp_path,
        prompt="read the image",
        model="sonnet",
        timeout_s=120,
    )

    assert data["result"] == "hello"
    assert captured_cmd[0] == "/usr/bin/claude"
    assert "-p" in captured_cmd and "read the image" in captured_cmd
    assert "--model" in captured_cmd and "sonnet" in captured_cmd
    # Read tool whitelisted + render dir added so claude can load the image.
    assert "--allowedTools" in captured_cmd and "Read" in captured_cmd
    assert "--add-dir" in captured_cmd and str(tmp_path.resolve()) in captured_cmd
    # No shell, bounded by the timeout we passed, never raises on non-zero.
    assert captured_kwargs["timeout"] == 120
    assert captured_kwargs["check"] is False


def test_nonzero_exit_raises(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(_RUN_TARGET, lambda *a, **k: _fake_completed("", returncode=2))
    with pytest.raises(RuntimeError, match="exit 2"):
        run_claude_read_image(
            claude_path="/x", render_dir=tmp_path, prompt="p", model="m", timeout_s=1
        )


def test_bad_json_raises(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(_RUN_TARGET, lambda *a, **k: _fake_completed("not json"))
    with pytest.raises(RuntimeError, match="bad JSON"):
        run_claude_read_image(
            claude_path="/x", render_dir=tmp_path, prompt="p", model="m", timeout_s=1
        )


def test_is_error_envelope_raises(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        _RUN_TARGET,
        lambda *a, **k: _fake_completed(json.dumps({"is_error": True, "subtype": "rate_limit"})),
    )
    with pytest.raises(RuntimeError, match="api error"):
        run_claude_read_image(
            claude_path="/x", render_dir=tmp_path, prompt="p", model="m", timeout_s=1
        )


def test_timeout_propagates(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    def boom(*a: object, **k: object) -> types.SimpleNamespace:
        raise subprocess.TimeoutExpired(cmd="claude", timeout=1)

    monkeypatch.setattr(_RUN_TARGET, boom)
    with pytest.raises(subprocess.TimeoutExpired):
        run_claude_read_image(
            claude_path="/x", render_dir=tmp_path, prompt="p", model="m", timeout_s=1
        )


# ---- run_claude_text (text-only, no Read tool) ----


def test_text_builds_hardened_argv_without_read(monkeypatch: pytest.MonkeyPatch) -> None:
    captured_cmd: list[str] = []

    def fake_run(cmd: list[str], **kwargs: object) -> types.SimpleNamespace:
        captured_cmd.extend(cmd)
        return _fake_completed(json.dumps({"result": "graded"}))

    monkeypatch.setattr(_RUN_TARGET, fake_run)
    data = run_claude_text(
        claude_path="/usr/bin/claude", prompt="judge this", model="sonnet", timeout_s=60
    )

    assert data["result"] == "graded"
    assert captured_cmd[0] == "/usr/bin/claude"
    assert "-p" in captured_cmd and "judge this" in captured_cmd
    # Hardening: tools off (text-only), no Read tool / no add-dir.
    assert "--tools" in captured_cmd
    assert "--allowedTools" not in captured_cmd
    assert "--add-dir" not in captured_cmd
    # No system prompt unless asked.
    assert "--system-prompt" not in captured_cmd


def test_text_includes_system_prompt_when_given(monkeypatch: pytest.MonkeyPatch) -> None:
    captured_cmd: list[str] = []

    def fake_run(cmd: list[str], **kwargs: object) -> types.SimpleNamespace:
        captured_cmd.extend(cmd)
        return _fake_completed(json.dumps({"result": "x"}))

    monkeypatch.setattr(_RUN_TARGET, fake_run)
    run_claude_text(claude_path="/x", prompt="p", model="m", timeout_s=1, system_prompt="be strict")
    assert "--system-prompt" in captured_cmd and "be strict" in captured_cmd


def test_text_nonzero_exit_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(_RUN_TARGET, lambda *a, **k: _fake_completed("", returncode=3))
    with pytest.raises(RuntimeError, match="exit 3"):
        run_claude_text(claude_path="/x", prompt="p", model="m", timeout_s=1)


def test_text_is_error_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        _RUN_TARGET, lambda *a, **k: _fake_completed(json.dumps({"is_error": True}))
    )
    with pytest.raises(RuntimeError, match="api error"):
        run_claude_text(claude_path="/x", prompt="p", model="m", timeout_s=1)
