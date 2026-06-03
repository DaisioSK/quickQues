"""Unit tests for DeepSeekV4Parser.

Strategy:
- All tests MOCK the openai.OpenAI client. No real API calls — the
  integration smoke is deferred to the user (DECISION-1.10 acceptance
  per dev-sprint.md 2026-05-29).
- The pypdfium2 render path IS exercised on the real synthetic fixture
  PDF, since that's pure-local code with no API dependency.
- We don't re-test the `_classify_page` heuristic here — that lives in
  test_claude_vision_parser.py and is shared via import. Reclassification
  + cache-key independence get one dedicated test so regressions in the
  vendor-prefixed cache path surface immediately.
"""

from __future__ import annotations

import os
import types
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from jcontract.impls.claude_vision_parser import (
    DRAWING_CAPTION_PROMPT,
    EMPTY_PAGE_SENTINEL,
    TEXT_OCR_PROMPT,
)
from jcontract.impls.deepseek_v4_parser import (
    DEEPSEEK_BASE_URL,
    DEFAULT_MODEL,
    DeepSeekV4Parser,
)

SYNTHETIC_PDF = Path("eval/fixtures/synthetic_contract_tqa.pdf")
RUN_INTEGRATION = os.environ.get("JCONTRACT_RUN_INTEGRATION") == "1"


def _make_mock_openai_response(
    text: str, prompt_tok: int = 1200, completion_tok: int = 200
) -> types.SimpleNamespace:
    """Build a fake OpenAI chat.completions response with one choice.

    Mirrors the openai SDK shape just enough for our parser's access pattern:
      response.choices[0].message.content
      response.usage.prompt_tokens
      response.usage.completion_tokens
    """
    message = types.SimpleNamespace(content=text)
    choice = types.SimpleNamespace(message=message)
    usage = types.SimpleNamespace(prompt_tokens=prompt_tok, completion_tokens=completion_tok)
    return types.SimpleNamespace(choices=[choice], usage=usage)


def _make_mock_client(text: str) -> MagicMock:
    """Return a mock OpenAI() client whose chat.completions.create returns `text`."""
    client = MagicMock()
    client.chat.completions.create.return_value = _make_mock_openai_response(text)
    return client


def test_default_model_is_flash():
    """User decision 2026-05-29: prototype default to flash for cost.

    Guarded as a test so an accidental constructor change surfaces in CI
    before it ships and quietly drives up the bill.
    """
    assert DEFAULT_MODEL == "deepseek-v4-flash"


def test_base_url_matches_official_endpoint():
    """Sanity: the documented DeepSeek OpenAI-compat base URL.

    A wrong URL here is silent until the first real API call — fast feedback
    via a fixture test.
    """
    assert DEEPSEEK_BASE_URL == "https://api.deepseek.com"


def test_renders_synthetic_pdf_without_api(tmp_path):
    """pypdfium2 render path works on the real synthetic fixture, no API calls."""
    client = _make_mock_client("rendered ok")
    parser = DeepSeekV4Parser(cache_dir=tmp_path / "cache", client=client, max_pages=1)

    pages = parser.parse(SYNTHETIC_PDF)

    assert len(pages) == 1
    assert pages[0].page_num == 1
    assert pages[0].text == "rendered ok"
    assert client.chat.completions.create.call_count == 1


def test_max_pages_bounds_processing(tmp_path):
    """max_pages=2 on a 4-page synthetic PDF should call the API exactly twice."""
    client = _make_mock_client("page text")
    parser = DeepSeekV4Parser(cache_dir=tmp_path / "cache", client=client, max_pages=2)

    pages = parser.parse(SYNTHETIC_PDF)

    assert len(pages) == 2
    assert [p.page_num for p in pages] == [1, 2]
    assert client.chat.completions.create.call_count == 2


def test_caches_results_under_vendor_prefix(tmp_path):
    """A second parse with same params should hit the cache: zero API calls.

    Also asserts the cache file uses the `deepseek-v4-` prefix so it can't
    collide with Anthropic vendor cache entries living in the same dir.
    """
    cache_dir = tmp_path / "cache"
    client = _make_mock_client("cached page text")
    parser_1 = DeepSeekV4Parser(cache_dir=cache_dir, client=client, max_pages=1)
    parser_1.parse(SYNTHETIC_PDF)
    assert client.chat.completions.create.call_count == 1

    # Cache file must start with our vendor prefix.
    cache_files = list(cache_dir.glob("deepseek-v4-*.txt"))
    assert len(cache_files) == 1, f"expected one prefixed cache file, got {cache_files}"

    # Second parser instance, same cache dir → pure cache hit.
    client_2 = _make_mock_client("SHOULD NOT BE RETURNED")
    parser_2 = DeepSeekV4Parser(cache_dir=cache_dir, client=client_2, max_pages=1)
    pages = parser_2.parse(SYNTHETIC_PDF)

    assert pages[0].text == "cached page text"
    assert client_2.chat.completions.create.call_count == 0  # cache hit


def test_skips_failing_page_gracefully(tmp_path):
    """API errors on one page must not blow up the whole batch.

    Contract: PDFParser must not raise on per-page extraction issues.
    """
    client = MagicMock()
    client.chat.completions.create.side_effect = RuntimeError("simulated API failure")
    parser = DeepSeekV4Parser(cache_dir=tmp_path / "cache", client=client, max_pages=2)

    pages = parser.parse(SYNTHETIC_PDF)

    assert len(pages) == 2
    assert all(p.text == "" for p in pages)


def test_empty_page_sentinel_is_normalised(tmp_path):
    """When the model returns the empty-page sentinel, store as empty string."""
    client = _make_mock_client(EMPTY_PAGE_SENTINEL)
    parser = DeepSeekV4Parser(cache_dir=tmp_path / "cache", client=client, max_pages=1)

    pages = parser.parse(SYNTHETIC_PDF)
    assert pages[0].text == ""


def test_request_payload_shape(tmp_path):
    """Verify model id, prompt, and OpenAI vision content list structure."""
    client = _make_mock_client("ok")
    parser = DeepSeekV4Parser(cache_dir=tmp_path / "cache", client=client, max_pages=1)
    parser.parse(SYNTHETIC_PDF)

    call_kwargs = client.chat.completions.create.call_args.kwargs
    # Default model honoured.
    assert call_kwargs["model"] == "deepseek-v4-flash"
    assert call_kwargs["max_tokens"] == 2048

    messages = call_kwargs["messages"]
    assert len(messages) == 1
    assert messages[0]["role"] == "user"

    content = messages[0]["content"]
    # OpenAI vision: image_url object BEFORE the text entry.
    assert content[0]["type"] == "image_url"
    image_url = content[0]["image_url"]
    assert image_url["url"].startswith("data:image/jpeg;base64,")
    # detail=high keeps small-font extraction usable on 150-DPI renders.
    assert image_url["detail"] == "high"
    # Text entry carries the construction-tender prompt or its drawing twin.
    assert content[1]["type"] == "text"
    assert "construction tender contract" in content[1]["text"]


def test_file_not_found_raises(tmp_path):
    """File-level errors (not extraction-quality) must raise loudly."""
    parser = DeepSeekV4Parser(cache_dir=tmp_path / "cache", client=MagicMock())
    with pytest.raises(FileNotFoundError):
        parser.parse(Path("does/not/exist.pdf"))


def test_no_deepseek_key_in_logs(tmp_path, caplog):
    """Sanity: API key patterns must not surface in logs even on failure paths.

    Mirrors test_claude_vision_parser's secret-audit test. We pin the secret
    pattern to "sk-" which matches DeepSeek's documented key shape; if the key
    leaks via repr(exc) the assert below will fire.
    """
    client = MagicMock()
    client.chat.completions.create.side_effect = RuntimeError("err with sk-secret-1234 in message")
    parser = DeepSeekV4Parser(cache_dir=tmp_path / "cache", client=client, max_pages=1)

    parser.parse(SYNTHETIC_PDF)
    # Parser logs error_type only, never the exception message body.
    for record in caplog.records:
        assert "sk-secret" not in record.getMessage()


def test_auto_classify_false_always_uses_text_prompt(tmp_path):
    """auto_classify=False forces TEXT_OCR_PROMPT for every page."""
    client = _make_mock_client("ok")
    parser = DeepSeekV4Parser(
        cache_dir=tmp_path / "cache",
        client=client,
        max_pages=1,
        auto_classify=False,
    )
    parser.parse(SYNTHETIC_PDF)

    call_kwargs = client.chat.completions.create.call_args.kwargs
    sent_text = call_kwargs["messages"][0]["content"][1]["text"]
    assert sent_text == TEXT_OCR_PROMPT
    assert sent_text != DRAWING_CAPTION_PROMPT


def test_drawing_page_uses_drawing_prompt(tmp_path, monkeypatch):
    """Classifier verdict 'drawing' must route DRAWING_CAPTION_PROMPT to the API."""
    client = _make_mock_client("rendered drawing extract")
    parser = DeepSeekV4Parser(cache_dir=tmp_path / "cache", client=client, max_pages=1)
    monkeypatch.setattr(parser, "_classify", lambda _jpeg: "drawing")
    parser.parse(SYNTHETIC_PDF)

    call_kwargs = client.chat.completions.create.call_args.kwargs
    sent_text = call_kwargs["messages"][0]["content"][1]["text"]
    assert sent_text == DRAWING_CAPTION_PROMPT


def test_drawing_and_text_have_separate_cache_keys(tmp_path, monkeypatch):
    """Cache key includes the prompt kind — reclassification re-OCRs.

    Same invariant as ClaudeVisionParser; duplicated as an integration test
    on the deepseek-v4 prefix so the cache layout doesn't silently regress
    if someone refactors the key builder.
    """
    cache_dir = tmp_path / "cache"

    client_a = _make_mock_client("TEXT extract")
    parser_a = DeepSeekV4Parser(cache_dir=cache_dir, client=client_a, max_pages=1)
    monkeypatch.setattr(parser_a, "_classify", lambda _jpeg: "text")
    pages_a = parser_a.parse(SYNTHETIC_PDF)
    assert pages_a[0].text == "TEXT extract"
    assert client_a.chat.completions.create.call_count == 1

    client_b = _make_mock_client("DRAWING extract")
    parser_b = DeepSeekV4Parser(cache_dir=cache_dir, client=client_b, max_pages=1)
    monkeypatch.setattr(parser_b, "_classify", lambda _jpeg: "drawing")
    pages_b = parser_b.parse(SYNTHETIC_PDF)

    # Different kind ⇒ different cache file ⇒ fresh API call.
    assert pages_b[0].text == "DRAWING extract"
    assert client_b.chat.completions.create.call_count == 1

    # Two distinct cache files in the dir, both with the deepseek-v4 prefix.
    cache_files = sorted(cache_dir.glob("deepseek-v4-*.txt"))
    assert len(cache_files) == 2


def test_model_override_is_honoured(tmp_path):
    """A caller upgrading to `deepseek-v4-pro` for quality must see it in payload."""
    client = _make_mock_client("ok")
    parser = DeepSeekV4Parser(
        cache_dir=tmp_path / "cache",
        client=client,
        max_pages=1,
        model="deepseek-v4-pro",
    )
    parser.parse(SYNTHETIC_PDF)

    call_kwargs = client.chat.completions.create.call_args.kwargs
    assert call_kwargs["model"] == "deepseek-v4-pro"


def test_handles_null_content_response(tmp_path):
    """Edge case: OpenAI shape allows `message.content` to be None.

    Some compat shims surface this when the model truncates. Parser must
    treat it as empty (logged + cached as "") rather than crashing on a
    None.strip() call.
    """
    null_response = types.SimpleNamespace(
        choices=[types.SimpleNamespace(message=types.SimpleNamespace(content=None))],
        usage=types.SimpleNamespace(prompt_tokens=100, completion_tokens=0),
    )
    client = MagicMock()
    client.chat.completions.create.return_value = null_response
    parser = DeepSeekV4Parser(cache_dir=tmp_path / "cache", client=client, max_pages=1)

    pages = parser.parse(SYNTHETIC_PDF)
    assert pages[0].text == ""


# ---------------------------------------------------------------------------
# Integration test — gated. Only runs with JCONTRACT_RUN_INTEGRATION=1 and
# DEEPSEEK_API_KEY set. Cost ~$0.001 per run (one synthetic page on flash).
# Deferred to user per DECISION-1.10 acceptance line.
# ---------------------------------------------------------------------------
@pytest.mark.skipif(
    not RUN_INTEGRATION or not os.environ.get("DEEPSEEK_API_KEY"),
    reason="Set JCONTRACT_RUN_INTEGRATION=1 and DEEPSEEK_API_KEY to run.",
)
def test_real_api_smoke_against_synthetic_pdf(tmp_path):
    """Hit the real DeepSeek API for one synthetic page — sanity check.

    Asserts the call succeeds and returns non-empty text. Does NOT assert
    specific content (model output varies); golden_cases handle accuracy.
    """
    parser = DeepSeekV4Parser(cache_dir=tmp_path / "cache", max_pages=1)
    pages = parser.parse(SYNTHETIC_PDF)

    assert len(pages) == 1
    assert len(pages[0].text) > 50
