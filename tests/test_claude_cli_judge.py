"""Unit tests for ClaudeCliJudge (E12) — no subprocess spawned.

Inject a fake ``claude_path`` so __init__ passes, monkeypatch the shared
``run_claude_text`` runner so grading never shells out.
"""

from __future__ import annotations

import json
import math

import pytest

from jcontract.impls import claude_cli_judge as mod
from jcontract.impls.claude_cli_judge import ClaudeCliJudge, _parse_judge_json
from jcontract.interfaces.schema import Chunk

FAKE_CLAUDE = "/bin/true"


def _chunk() -> Chunk:
    return Chunk(
        id="f.pdf:1:0",
        text="Trackwork Contractor handles waterproofing.",
        file="f.pdf",
        page=1,
        chunk_type="paragraph",
    )


def _judge() -> ClaudeCliJudge:
    return ClaudeCliJudge(claude_path=FAKE_CLAUDE)


def _result(payload: dict[str, object]) -> dict[str, object]:
    return {"result": json.dumps(payload), "usage": {}}


def test_faithfulness_happy(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        mod, "run_claude_text", lambda **k: _result({"score": 0.9, "reasoning": "grounded"})
    )
    score = _judge().faithfulness("answer text", [_chunk()])
    assert score.score == 0.9
    assert score.reasoning == "grounded"


def test_answer_relevancy_happy(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        mod, "run_claude_text", lambda **k: _result({"score": 1.0, "reasoning": "on topic"})
    )
    score = _judge().answer_relevancy("question?", "answer")
    assert score.score == 1.0


def test_score_clamped_into_unit_range(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        mod, "run_claude_text", lambda **k: _result({"score": 95, "reasoning": "x"})
    )
    assert _judge().faithfulness("a", [_chunk()]).score == 1.0
    monkeypatch.setattr(
        mod, "run_claude_text", lambda **k: _result({"score": -3, "reasoning": "x"})
    )
    assert _judge().faithfulness("a", [_chunk()]).score == 0.0


def test_runner_failure_returns_nan_not_raising(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(**k: object) -> dict[str, object]:
        raise RuntimeError("claude CLI exit 1")

    monkeypatch.setattr(mod, "run_claude_text", boom)
    score = _judge().faithfulness("a", [_chunk()])
    assert math.isnan(score.score)
    assert "unavailable" in score.reasoning


def test_non_json_result_returns_nan(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        mod, "run_claude_text", lambda **k: {"result": "prose not json", "usage": {}}
    )
    assert math.isnan(_judge().answer_relevancy("q", "a").score)


def test_missing_binary_raises_at_init(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("jcontract.impls.claude_cli_judge.shutil.which", lambda _n: None)
    with pytest.raises(RuntimeError, match="claude CLI not found"):
        ClaudeCliJudge(claude_path=None)


# ---- _parse_judge_json (pure) ----


def test_parse_strips_fence() -> None:
    s = _parse_judge_json('```json\n{"score": 0.5, "reasoning": "r"}\n```')
    assert s.score == 0.5
    assert s.reasoning == "r"


def test_parse_wrong_shape_is_nan() -> None:
    assert math.isnan(_parse_judge_json('["not", "a", "dict"]').score)


def test_parse_missing_score_is_nan() -> None:
    assert math.isnan(_parse_judge_json('{"reasoning": "no score here"}').score)


def test_parse_non_numeric_score_is_nan() -> None:
    assert math.isnan(_parse_judge_json('{"score": "high"}').score)
