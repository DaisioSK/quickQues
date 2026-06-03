"""Unit tests for ClaudeCliVisionCaptioner (E11) — no subprocess spawned.

We inject a fake ``claude_path`` so __init__ passes, and monkeypatch the
shared ``run_claude_read_image`` runner so caption() exercises the
image-write + parse + cache path without shelling out.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from jcontract.impls import claude_cli_vision_captioner as mod
from jcontract.impls.claude_cli_vision_captioner import ClaudeCliVisionCaptioner
from jcontract.interfaces import DrawingCaption

FAKE_CLAUDE = "/bin/true"


def _captioner(tmp_path: Path) -> ClaudeCliVisionCaptioner:
    return ClaudeCliVisionCaptioner(
        cache_dir=tmp_path / "cache",
        render_dir=tmp_path / "render",
        claude_path=FAKE_CLAUDE,
    )


def test_happy_path_returns_caption(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    payload = {"caption_zh": "桥梁防水图", "entities": ["T/PRJ/CWD/WS/2101A"]}
    monkeypatch.setattr(
        mod, "run_claude_read_image", lambda **k: {"result": json.dumps(payload), "usage": {}}
    )
    cap = _captioner(tmp_path).caption(b"jpeg", "Drawing No. T/PRJ/CWD/WS/2101A")

    assert isinstance(cap, DrawingCaption)
    assert cap.caption_zh == "桥梁防水图"
    assert "T/PRJ/CWD/WS/2101A" in cap.entities
    # The transient render file is cleaned up.
    assert list((tmp_path / "render").glob("*.jpg")) == []


def test_cache_hit_skips_runner(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    calls = {"n": 0}

    def fake_runner(**k: object) -> dict[str, object]:
        calls["n"] += 1
        return {"result": json.dumps({"caption_zh": "首次", "entities": []}), "usage": {}}

    monkeypatch.setattr(mod, "run_claude_read_image", fake_runner)
    _captioner(tmp_path).caption(b"same-bytes", "")
    assert calls["n"] == 1

    result = _captioner(tmp_path).caption(b"same-bytes", "")
    assert result.caption_zh == "首次"
    assert calls["n"] == 1  # served from cache, runner not called again


def test_runner_failure_returns_empty_not_raising(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def boom(**k: object) -> dict[str, object]:
        raise RuntimeError("claude CLI exit 1")

    monkeypatch.setattr(mod, "run_claude_read_image", boom)
    cap = _captioner(tmp_path).caption(b"jpeg", "")
    assert cap.caption_zh == ""
    assert cap.entities == []


def test_non_json_result_returns_empty(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        mod, "run_claude_read_image", lambda **k: {"result": "prose not json", "usage": {}}
    )
    cap = _captioner(tmp_path).caption(b"jpeg", "")
    assert cap.caption_zh == ""


def test_missing_binary_raises_at_init(monkeypatch: pytest.MonkeyPatch) -> None:
    # With no claude_path and `claude` absent from PATH, fail loud at init.
    monkeypatch.setattr(
        "jcontract.impls.claude_cli_vision_captioner.shutil.which", lambda _name: None
    )
    with pytest.raises(RuntimeError, match="claude CLI not found"):
        ClaudeCliVisionCaptioner(claude_path=None)
