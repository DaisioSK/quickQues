"""Unit tests for OllamaVisionCaptioner (ssLB) — mocked OpenAI client, no server."""

from __future__ import annotations

import json
import types
from pathlib import Path
from unittest.mock import MagicMock

from jcontract.impls.ollama_vision_captioner import (
    OllamaVisionCaptioner,
    _to_openai_base_url,
)
from jcontract.interfaces import DrawingCaption


def _mock_client(content: str) -> MagicMock:
    """OpenAI-shaped mock: choices[0].message.content + usage."""
    message = types.SimpleNamespace(content=content)
    choice = types.SimpleNamespace(message=message)
    usage = types.SimpleNamespace(prompt_tokens=640, completion_tokens=132)
    response = types.SimpleNamespace(choices=[choice], usage=usage)
    client = MagicMock()
    client.chat.completions.create.return_value = response
    return client


def test_happy_path(tmp_path: Path) -> None:
    payload = {"caption_zh": "车站屋面图说", "entities": ["T/DEMO/0257"]}
    client = _mock_client(json.dumps(payload, ensure_ascii=False))
    cap = OllamaVisionCaptioner(cache_dir=tmp_path / "cache", client=client).caption(
        b"jpeg", "Drawing No. T/DEMO"
    )
    assert isinstance(cap, DrawingCaption)
    assert cap.caption_zh == "车站屋面图说"
    assert cap.entities == ["T/DEMO/0257"]


def test_payload_shape_to_openai(tmp_path: Path) -> None:
    """Vision content list: image_url data URI before text — the DECISION-ls.20 shape."""
    client = _mock_client(json.dumps({"caption_zh": "ok", "entities": []}))
    OllamaVisionCaptioner(cache_dir=tmp_path / "cache", client=client).caption(
        b"jpeg", "Drawing No. T/DEMO"
    )
    kwargs = client.chat.completions.create.call_args.kwargs
    content = kwargs["messages"][0]["content"]
    assert content[0]["type"] == "image_url"
    assert content[0]["image_url"]["url"].startswith("data:image/jpeg;base64,")
    assert content[1]["type"] == "text"
    assert "T/DEMO" in content[1]["text"]


def test_inline_think_block_is_stripped(tmp_path: Path) -> None:
    """A compat shim that inlines <think> must not break the JSON parse (ls.11)."""
    raw = '<think>looks like a roof plan...</think>{"caption_zh": "屋面图", "entities": []}'
    client = _mock_client(raw)
    cap = OllamaVisionCaptioner(cache_dir=tmp_path / "cache", client=client).caption(b"x", "")
    assert cap.caption_zh == "屋面图"


def test_cache_hit_skips_api(tmp_path: Path) -> None:
    cache_dir = tmp_path / "cache"
    client = _mock_client(json.dumps({"caption_zh": "首次", "entities": []}))
    OllamaVisionCaptioner(cache_dir=cache_dir, client=client).caption(b"bytes", "")
    assert client.chat.completions.create.call_count == 1

    client_2 = _mock_client(json.dumps({"caption_zh": "SHOULD NOT REACH API", "entities": []}))
    result = OllamaVisionCaptioner(cache_dir=cache_dir, client=client_2).caption(b"bytes", "")
    assert result.caption_zh == "首次"
    assert client_2.chat.completions.create.call_count == 0


def test_api_error_returns_empty_not_raising(tmp_path: Path) -> None:
    client = MagicMock()
    client.chat.completions.create.side_effect = RuntimeError("server down")
    cap = OllamaVisionCaptioner(cache_dir=tmp_path / "cache", client=client).caption(b"x", "")
    assert cap.caption_zh == ""
    assert cap.entities == []


def test_non_json_returns_empty(tmp_path: Path) -> None:
    client = _mock_client("just prose, not json")
    cap = OllamaVisionCaptioner(cache_dir=tmp_path / "cache", client=client).caption(b"x", "")
    assert cap.caption_zh == ""


def test_base_url_normalisation() -> None:
    """Server address → /v1 compat root, idempotently (no /v1/v1, no missing suffix)."""
    assert _to_openai_base_url("http://localhost:11434") == "http://localhost:11434/v1"
    assert _to_openai_base_url("http://localhost:11434/") == "http://localhost:11434/v1"
    assert _to_openai_base_url("http://localhost:11434/v1") == "http://localhost:11434/v1"


def test_model_env_default(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """No explicit model → JCONTRACT_OLLAMA_VL_MODEL, else the qwen3-vl:8b default."""
    client = _mock_client(json.dumps({"caption_zh": "ok", "entities": []}))
    monkeypatch.delenv("JCONTRACT_OLLAMA_VL_MODEL", raising=False)
    OllamaVisionCaptioner(cache_dir=tmp_path / "c1", client=client).caption(b"a", "")
    assert client.chat.completions.create.call_args.kwargs["model"] == "qwen3-vl:8b"

    monkeypatch.setenv("JCONTRACT_OLLAMA_VL_MODEL", "qwen3-vl:32b")
    OllamaVisionCaptioner(cache_dir=tmp_path / "c2", client=client).caption(b"b", "")
    assert client.chat.completions.create.call_args.kwargs["model"] == "qwen3-vl:32b"
