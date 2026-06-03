"""Unit tests for ClaudeVisionCaptioner.

Strategy:
- All tests mock the anthropic.Anthropic client. No real API calls.
- Tests assert the public contract (DrawingCaption shape) plus the
  defensive parsing paths (non-JSON output, missing keys, markdown
  fences from a stubborn model) since these are non-trivial failure
  modes per the captioner's design notes.
"""

from __future__ import annotations

import json
import types
from unittest.mock import MagicMock

from jcontract.impls.claude_vision_captioner import (
    CACHE_SUFFIX,
    ClaudeVisionCaptioner,
)
from jcontract.interfaces import DrawingCaption


def _make_mock_response(text: str, in_tok: int = 1200, out_tok: int = 100) -> types.SimpleNamespace:
    """Fake Anthropic Messages response with one text block — same shape parser uses."""
    block = types.SimpleNamespace(type="text", text=text)
    usage = types.SimpleNamespace(input_tokens=in_tok, output_tokens=out_tok)
    return types.SimpleNamespace(content=[block], usage=usage)


def _make_mock_client(text: str) -> MagicMock:
    client = MagicMock()
    client.messages.create.return_value = _make_mock_response(text)
    return client


def test_returns_drawing_caption_on_valid_json(tmp_path):
    """Happy path: model emits valid JSON → caption_zh + entities populated."""
    payload = {
        "caption_zh": "桥梁防水构造图，含 3 层涂层结构，主要尺寸 50mm。",
        "entities": ["T/PRJ/CWD/WS/2101A", "Clause 7.3", "50mm"],
    }
    client = _make_mock_client(json.dumps(payload, ensure_ascii=False))
    captioner = ClaudeVisionCaptioner(cache_dir=tmp_path / "cache", client=client)

    result = captioner.caption(b"fake-jpeg-bytes", nearby_text="Drawing No. T/PRJ/CWD/WS/2101A")

    assert isinstance(result, DrawingCaption)
    assert result.caption_zh.startswith("桥梁")
    assert "T/PRJ/CWD/WS/2101A" in result.entities
    assert "50mm" in result.entities


def test_cache_hit_skips_api(tmp_path):
    """Second call with same image bytes + model must NOT call the API."""
    cache_dir = tmp_path / "cache"
    payload = {"caption_zh": "首次", "entities": []}
    client = _make_mock_client(json.dumps(payload))

    captioner_1 = ClaudeVisionCaptioner(cache_dir=cache_dir, client=client)
    captioner_1.caption(b"same-image-bytes", "")
    assert client.messages.create.call_count == 1

    # Confirm cache file landed under the expected suffix.
    cache_files = list(cache_dir.glob(f"*{CACHE_SUFFIX}"))
    assert len(cache_files) == 1, f"expected one cache file, got {cache_files}"

    # Second captioner instance shares the dir → pure cache hit.
    client_2 = _make_mock_client(json.dumps({"caption_zh": "SHOULD NOT REACH API", "entities": []}))
    captioner_2 = ClaudeVisionCaptioner(cache_dir=cache_dir, client=client_2)
    result = captioner_2.caption(b"same-image-bytes", "")
    assert result.caption_zh == "首次"
    assert client_2.messages.create.call_count == 0


def test_api_error_returns_empty_caption_not_raising(tmp_path):
    """Per Protocol contract: captioner.caption MUST NOT raise.

    A failure must yield an empty DrawingCaption so caller can record
    chunk.caption = "" (ran-but-empty) rather than crash the ingest.
    """
    client = MagicMock()
    client.messages.create.side_effect = RuntimeError("simulated API failure")
    captioner = ClaudeVisionCaptioner(cache_dir=tmp_path / "cache", client=client)

    result = captioner.caption(b"image-bytes", "")
    assert result.caption_zh == ""
    assert result.entities == []


def test_non_json_output_returns_empty(tmp_path):
    """When the model returns prose instead of JSON, fall back to empty."""
    client = _make_mock_client("This is not a valid JSON response, just prose.")
    captioner = ClaudeVisionCaptioner(cache_dir=tmp_path / "cache", client=client)

    result = captioner.caption(b"image-bytes", "")
    assert result.caption_zh == ""
    assert result.entities == []


def test_markdown_fence_around_json_is_stripped(tmp_path):
    """Some models wrap JSON in ```json fences despite prompt instructions.

    Captioner strips one layer of fences defensively before parsing.
    """
    payload = {"caption_zh": "解析成功", "entities": ["test"]}
    fenced = f"```json\n{json.dumps(payload, ensure_ascii=False)}\n```"
    client = _make_mock_client(fenced)
    captioner = ClaudeVisionCaptioner(cache_dir=tmp_path / "cache", client=client)

    result = captioner.caption(b"image-bytes", "")
    assert result.caption_zh == "解析成功"
    assert result.entities == ["test"]


def test_payload_shape_to_anthropic(tmp_path):
    """Verify image-before-text + JPEG media_type + prompt nearby_text injection."""
    client = _make_mock_client(json.dumps({"caption_zh": "ok", "entities": []}))
    captioner = ClaudeVisionCaptioner(cache_dir=tmp_path / "cache", client=client)
    captioner.caption(b"jpeg-bytes", nearby_text="Drawing No. T/PRJ/CWD/WS/2101A")

    call_kwargs = client.messages.create.call_args.kwargs
    assert call_kwargs["model"] == "claude-sonnet-4-5"

    messages = call_kwargs["messages"]
    assert messages[0]["role"] == "user"
    content = messages[0]["content"]
    # Image entry first.
    assert content[0]["type"] == "image"
    assert content[0]["source"]["media_type"] == "image/jpeg"
    # Text entry carries the prompt with nearby_text substituted in.
    assert content[1]["type"] == "text"
    assert "T/PRJ/CWD/WS/2101A" in content[1]["text"]
    # And the JSON-only directive is intact.
    assert "JSON object" in content[1]["text"]


def test_unexpected_json_shape_falls_back_to_empty(tmp_path):
    """Valid JSON but missing required key → fall back without crashing."""
    # Model returns a list instead of an object — unexpected shape.
    client = _make_mock_client(json.dumps(["unexpected", "shape"]))
    captioner = ClaudeVisionCaptioner(cache_dir=tmp_path / "cache", client=client)
    result = captioner.caption(b"image-bytes", "")
    assert result.caption_zh == ""
    assert result.entities == []


def test_entities_coerced_when_non_list(tmp_path):
    """If model emits entities as a string instead of list, coerce to empty list."""
    payload = {"caption_zh": "图说 OK", "entities": "should-be-list-but-isnt"}
    client = _make_mock_client(json.dumps(payload, ensure_ascii=False))
    captioner = ClaudeVisionCaptioner(cache_dir=tmp_path / "cache", client=client)
    result = captioner.caption(b"image-bytes", "")
    assert result.caption_zh == "图说 OK"  # caption survives
    assert result.entities == []  # bad type coerced safely
