"""Unit tests for the shared caption helpers (Enhancement E11)."""

from __future__ import annotations

import json
from pathlib import Path

from jcontract.impls._caption_shared import (
    build_caption_prompt,
    parse_caption_payload,
    payload_to_caption,
    read_caption_cache,
    write_caption_cache,
)
from jcontract.interfaces import DrawingCaption


def test_build_prompt_injects_and_trims_nearby_text():
    prompt = build_caption_prompt("Drawing No. T/PRJ/CWD/WS/2101A")
    assert "T/PRJ/CWD/WS/2101A" in prompt
    assert "JSON object" in prompt  # the JSON-only directive survives


def test_build_prompt_trims_to_limit():
    prompt = build_caption_prompt("x" * 5000)
    # 1500-char cap on the grounding text keeps per-call cost bounded.
    # Check the boundary directly (the template itself contains a few x's).
    assert "x" * 1500 in prompt
    assert "x" * 1501 not in prompt


def test_parse_valid_json():
    payload = parse_caption_payload('{"caption_zh": "桥梁图", "entities": ["Clause 7.3"]}')
    assert payload == {"caption_zh": "桥梁图", "entities": ["Clause 7.3"]}


def test_parse_strips_markdown_fence():
    raw = '```json\n{"caption_zh": "ok", "entities": []}\n```'
    assert parse_caption_payload(raw) == {"caption_zh": "ok", "entities": []}


def test_parse_non_json_returns_empty():
    assert parse_caption_payload("just prose") == {"caption_zh": "", "entities": []}


def test_parse_wrong_shape_returns_empty():
    assert parse_caption_payload('["a", "b"]') == {"caption_zh": "", "entities": []}


def test_parse_coerces_non_list_entities():
    payload = parse_caption_payload('{"caption_zh": "ok", "entities": "not-a-list"}')
    assert payload == {"caption_zh": "ok", "entities": []}


def test_payload_to_caption():
    cap = payload_to_caption({"caption_zh": "图说", "entities": ["A", 7]})
    assert isinstance(cap, DrawingCaption)
    assert cap.caption_zh == "图说"
    assert cap.entities == ["A", "7"]  # entries coerced to str


def test_cache_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "x.caption.json"
    write_caption_cache(path, {"caption_zh": "缓存", "entities": ["E1"]})
    cap = read_caption_cache(path)
    assert cap is not None
    assert cap.caption_zh == "缓存"
    assert cap.entities == ["E1"]


def test_cache_miss_returns_none(tmp_path: Path) -> None:
    assert read_caption_cache(tmp_path / "nope.caption.json") is None


def test_cache_corrupt_returns_none(tmp_path: Path) -> None:
    path = tmp_path / "bad.caption.json"
    path.write_text("{not valid json", encoding="utf-8")
    assert read_caption_cache(path) is None


def test_cache_persists_empty_payload(tmp_path: Path) -> None:
    # Empty caption (ran-but-empty) must round-trip too — it's a valid state.
    path = tmp_path / "empty.caption.json"
    write_caption_cache(path, {"caption_zh": "", "entities": []})
    assert json.loads(path.read_text(encoding="utf-8")) == {"caption_zh": "", "entities": []}
    cap = read_caption_cache(path)
    assert cap is not None
    assert cap.caption_zh == ""
