"""Unit tests for OpenAICompatAnswerer.

Strategy:
- All tests MOCK the openai client (same SimpleNamespace fixture style as
  test_deepseek_v4_parser.py). No real network / no real Ollama — the live
  end-to-end run is the sub-sprint's separate e2e gate (LESSON-2).
- Env handling is tested through the config accessors with monkeypatch,
  never by mutating real process env permanently.
"""

from __future__ import annotations

import types
from unittest.mock import MagicMock

import pytest
from openai import OpenAIError

from jcontract.config import (
    get_local_llm_api_key,
    get_local_llm_base_url,
    get_local_llm_model,
)
from jcontract.impls.openai_compat_answerer import (
    OpenAICompatAnswerer,
    _strip_think,
)
from jcontract.interfaces import Chunk


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


def _make_mock_response(text: str) -> types.SimpleNamespace:
    """Fake OpenAI chat.completions response — mirrors the SDK access pattern:
    response.choices[0].message.content / response.usage.*_tokens."""
    message = types.SimpleNamespace(content=text)
    choice = types.SimpleNamespace(message=message)
    usage = types.SimpleNamespace(prompt_tokens=100, completion_tokens=50)
    return types.SimpleNamespace(choices=[choice], usage=usage)


def _make_mock_client(text: str) -> MagicMock:
    client = MagicMock()
    client.chat.completions.create.return_value = _make_mock_response(text)
    return client


# ---------- env / defaults (config accessors) ----------


def test_env_defaults_point_at_local_ollama(monkeypatch: pytest.MonkeyPatch) -> None:
    """Unset env → safe local defaults (no data egress out of the box)."""
    monkeypatch.delenv("JCONTRACT_LOCAL_LLM_BASE_URL", raising=False)
    monkeypatch.delenv("JCONTRACT_LOCAL_LLM_MODEL", raising=False)
    monkeypatch.delenv("JCONTRACT_LOCAL_LLM_API_KEY", raising=False)

    assert get_local_llm_base_url() == "http://localhost:11434/v1"
    assert get_local_llm_model() == "qwen3:14b"
    # Ollama ignores the key; SDK just needs it non-empty.
    assert get_local_llm_api_key() == "ollama"


def test_env_overrides_are_honoured(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("JCONTRACT_LOCAL_LLM_BASE_URL", "http://localhost:8001/v1")
    monkeypatch.setenv("JCONTRACT_LOCAL_LLM_MODEL", "some-model:7b")
    monkeypatch.setenv("JCONTRACT_LOCAL_LLM_API_KEY", "secret-token")

    assert get_local_llm_base_url() == "http://localhost:8001/v1"
    assert get_local_llm_model() == "some-model:7b"
    assert get_local_llm_api_key() == "secret-token"


def test_model_env_flows_into_api_call(monkeypatch: pytest.MonkeyPatch) -> None:
    """The model id sent to the endpoint comes from env when not passed in."""
    monkeypatch.setenv("JCONTRACT_LOCAL_LLM_MODEL", "env-model:1b")
    client = _make_mock_client("答案 [f.pdf p.1]。")
    answerer = OpenAICompatAnswerer(client=client)

    answerer.answer("谁负责防水？", _make_chunks())

    kwargs = client.chat.completions.create.call_args.kwargs
    assert kwargs["model"] == "env-model:1b"


def test_constructor_model_beats_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("JCONTRACT_LOCAL_LLM_MODEL", "env-model:1b")
    client = _make_mock_client("答案 [f.pdf p.1]。")
    answerer = OpenAICompatAnswerer(client=client, model="ctor-model:2b")

    answerer.answer("谁负责防水？", _make_chunks())

    assert client.chat.completions.create.call_args.kwargs["model"] == "ctor-model:2b"


# ---------- think stripping (DECISION-ls.11) ----------


def test_strip_think_removes_closed_block() -> None:
    raw = "<think>推理过程，甚至含 [f.pdf p.1] 引用。</think>真正答案 [f.pdf p.1]。"
    assert _strip_think(raw) == "真正答案 [f.pdf p.1]。"


def test_strip_think_removes_multiple_blocks_and_keeps_answer() -> None:
    raw = "<think>a</think>第一句 [f.pdf p.1]。<think>b\nmultiline</think>"
    assert _strip_think(raw) == "第一句 [f.pdf p.1]。"


def test_strip_think_drops_unclosed_tail() -> None:
    """Truncated generation: everything after a dangling <think> is reasoning."""
    raw = "答案 [f.pdf p.1]。<think>截断的推理没有闭合"
    assert _strip_think(raw) == "答案 [f.pdf p.1]。"


def test_strip_think_noop_on_clean_text() -> None:
    assert _strip_think("干净答案 [f.pdf p.1]。") == "干净答案 [f.pdf p.1]。"


def test_answer_strips_inline_think_before_citation_validation() -> None:
    """A think block containing a VALID citation must not leak into the answer.

    This is the exact failure mode observed in the 2026-06-11 live probe:
    qwen3's reasoning text quoted the citation string, which would survive
    validate_citations if think were stripped after (or not at all).
    """
    chunks = _make_chunks()
    raw = (
        "<think>资料说承包商负责，引用是 [f.pdf p.1]。</think>"
        "轨道工程承包商负责桥墩防水 [f.pdf p.1]。"
    )
    client = _make_mock_client(raw)
    answerer = OpenAICompatAnswerer(client=client)

    ans = answerer.answer("谁负责桥墩防水？", chunks)

    assert "think" not in ans.text
    assert "推理" not in ans.text and "资料说" not in ans.text
    assert ans.text == "轨道工程承包商负责桥墩防水 [f.pdf p.1]。"
    assert ans.citations == [("f.pdf", 1)]


# ---------- answer pipeline (shared postprocess parity) ----------


def test_answer_happy_path_validates_citations() -> None:
    chunks = _make_chunks()
    # Newline-separated sentences — the shared splitter treats newlines as
    # soft sentence boundaries (models often emit one bullet per line).
    client = _make_mock_client(
        "轨道工程承包商负责防水 [f.pdf p.1]。\n这句没有引用会被丢弃。\n凭空引用 [ghost.pdf p.9]。"
    )
    answerer = OpenAICompatAnswerer(client=client)

    ans = answerer.answer("谁负责防水？", chunks)

    # Uncited + fabricated-cite sentences dropped by the SHARED postprocess.
    assert ans.text == "轨道工程承包商负责防水 [f.pdf p.1]。"
    assert ans.citations == [("f.pdf", 1)]
    assert ans.confidence == "medium"
    assert ans.raw_context == chunks


def test_answer_sends_shared_prompt_shape() -> None:
    """System + user roles carry the shared prompt parts (backend-swap only)."""
    client = _make_mock_client("答案 [f.pdf p.1]。")
    answerer = OpenAICompatAnswerer(client=client)

    answerer.answer("谁负责防水？", _make_chunks())

    messages = client.chat.completions.create.call_args.kwargs["messages"]
    assert [m["role"] for m in messages] == ["system", "user"]
    # Fingerprints of the shared template (answer/prompt.py).
    assert "MANDATORY CITATIONS" in messages[0]["content"]
    assert "<context_chunk" in messages[1]["content"]
    assert "<question>" in messages[1]["content"]


def test_endpoint_error_degrades_to_fallback() -> None:
    """Server down / model not pulled → canonical fallback, never an exception."""
    client = MagicMock()
    client.chat.completions.create.side_effect = OpenAIError("connection refused")
    answerer = OpenAICompatAnswerer(client=client)

    ans = answerer.answer("谁负责防水？", _make_chunks())

    assert ans.text == "文档中未明确说明。"
    assert ans.citations == []
    assert ans.confidence == "low"


def test_empty_content_returns_fallback() -> None:
    """None content (defensive) → fallback via shared validate_citations."""
    message = types.SimpleNamespace(content=None)
    choice = types.SimpleNamespace(message=message)
    response = types.SimpleNamespace(choices=[choice], usage=None)
    client = MagicMock()
    client.chat.completions.create.return_value = response
    answerer = OpenAICompatAnswerer(client=client)

    ans = answerer.answer("谁负责防水？", _make_chunks())

    assert ans.text == "文档中未明确说明"
    assert ans.citations == []
