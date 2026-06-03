"""Unit tests for DeepSeekVisionCaptioner (E11) — mocked OpenAI client, no API."""

from __future__ import annotations

import json
import types
from pathlib import Path
from unittest.mock import MagicMock

from jcontract.impls.deepseek_vision_captioner import DeepSeekVisionCaptioner
from jcontract.interfaces import DrawingCaption


def _mock_client(content: str) -> MagicMock:
    """OpenAI-shaped mock: choices[0].message.content + usage."""
    message = types.SimpleNamespace(content=content)
    choice = types.SimpleNamespace(message=message)
    usage = types.SimpleNamespace(prompt_tokens=100, completion_tokens=30)
    response = types.SimpleNamespace(choices=[choice], usage=usage)
    client = MagicMock()
    client.chat.completions.create.return_value = response
    return client


def test_happy_path(tmp_path: Path) -> None:
    payload = {"caption_zh": "桥梁图说", "entities": ["Clause 7.3"]}
    client = _mock_client(json.dumps(payload, ensure_ascii=False))
    cap = DeepSeekVisionCaptioner(cache_dir=tmp_path / "cache", client=client).caption(
        b"jpeg", "Drawing No. T/DEMO"
    )
    assert isinstance(cap, DrawingCaption)
    assert cap.caption_zh == "桥梁图说"
    assert cap.entities == ["Clause 7.3"]


def test_payload_shape_to_openai(tmp_path: Path) -> None:
    client = _mock_client(json.dumps({"caption_zh": "ok", "entities": []}))
    DeepSeekVisionCaptioner(cache_dir=tmp_path / "cache", client=client).caption(
        b"jpeg", "Drawing No. T/DEMO"
    )
    kwargs = client.chat.completions.create.call_args.kwargs
    content = kwargs["messages"][0]["content"]
    # image_url (data URI) before text, detail=high for OCR fidelity.
    assert content[0]["type"] == "image_url"
    assert content[0]["image_url"]["url"].startswith("data:image/jpeg;base64,")
    assert content[0]["image_url"]["detail"] == "high"
    assert content[1]["type"] == "text"
    assert "T/DEMO" in content[1]["text"]


def test_cache_hit_skips_api(tmp_path: Path) -> None:
    cache_dir = tmp_path / "cache"
    client = _mock_client(json.dumps({"caption_zh": "首次", "entities": []}))
    DeepSeekVisionCaptioner(cache_dir=cache_dir, client=client).caption(b"bytes", "")
    assert client.chat.completions.create.call_count == 1

    client_2 = _mock_client(json.dumps({"caption_zh": "SHOULD NOT REACH API", "entities": []}))
    result = DeepSeekVisionCaptioner(cache_dir=cache_dir, client=client_2).caption(b"bytes", "")
    assert result.caption_zh == "首次"
    assert client_2.chat.completions.create.call_count == 0


def test_api_error_returns_empty_not_raising(tmp_path: Path) -> None:
    client = MagicMock()
    client.chat.completions.create.side_effect = RuntimeError("simulated failure")
    cap = DeepSeekVisionCaptioner(cache_dir=tmp_path / "cache", client=client).caption(b"x", "")
    assert cap.caption_zh == ""
    assert cap.entities == []


def test_non_json_returns_empty(tmp_path: Path) -> None:
    client = _mock_client("just prose, not json")
    cap = DeepSeekVisionCaptioner(cache_dir=tmp_path / "cache", client=client).caption(b"x", "")
    assert cap.caption_zh == ""
