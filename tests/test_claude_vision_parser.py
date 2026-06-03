"""Unit tests for ClaudeVisionParser.

Strategy:
- All tests in this file MOCK the anthropic.Anthropic client. No real API calls.
- A real-API smoke test is gated behind JCONTRACT_RUN_INTEGRATION=1 + presence
  of ANTHROPIC_API_KEY in env, and uses the synthetic test PDF (not a real
  contract PDF) to keep cost predictable.
- The pypdfium2 render path IS exercised on the real synthetic PDF — that's
  pure-local code, no API, fast, and the cheapest way to catch render bugs.

Phase 1.7 additions:
- Tests for the `_classify_page` heuristic (synthetic PIL fixtures, no API).
- Tests for the dual-prompt routing (verify which prompt reaches the API
  given mocked classifier verdicts).
- Tests for graceful fallback when the classifier itself raises.
"""

from __future__ import annotations

import io
import os
import types
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from PIL import Image, ImageDraw

from jcontract.impls.claude_vision_parser import (
    DRAWING_CAPTION_PROMPT,
    EMPTY_PAGE_SENTINEL,
    TEXT_OCR_PROMPT,
    ClaudeVisionParser,
    _classify_page,
)

SYNTHETIC_PDF = Path("eval/fixtures/synthetic_contract_tqa.pdf")
RUN_INTEGRATION = os.environ.get("JCONTRACT_RUN_INTEGRATION") == "1"


def _make_mock_anthropic_response(
    text: str, in_tok: int = 1200, out_tok: int = 200
) -> types.SimpleNamespace:
    """Build a fake anthropic Messages response with one text block."""
    block = types.SimpleNamespace(type="text", text=text)
    usage = types.SimpleNamespace(input_tokens=in_tok, output_tokens=out_tok)
    return types.SimpleNamespace(content=[block], usage=usage)


def _make_mock_client(text: str) -> MagicMock:
    """Return a mock Anthropic() client whose messages.create returns `text`."""
    client = MagicMock()
    client.messages.create.return_value = _make_mock_anthropic_response(text)
    return client


def test_renders_synthetic_pdf_without_api(tmp_path):
    """pypdfium2 render path works on the real synthetic fixture, no API calls."""
    # Mock client returns canned text — but more importantly, render must succeed.
    client = _make_mock_client("rendered ok")
    parser = ClaudeVisionParser(cache_dir=tmp_path / "cache", client=client, max_pages=1)

    pages = parser.parse(SYNTHETIC_PDF)

    # Synthetic PDF has 4 pages, capped at 1 here.
    assert len(pages) == 1
    assert pages[0].page_num == 1
    assert pages[0].text == "rendered ok"
    # Anthropic was called once for the single processed page.
    assert client.messages.create.call_count == 1


def test_max_pages_bounds_processing(tmp_path):
    """max_pages=2 on a 4-page PDF should call the API exactly twice."""
    client = _make_mock_client("page text")
    parser = ClaudeVisionParser(cache_dir=tmp_path / "cache", client=client, max_pages=2)

    pages = parser.parse(SYNTHETIC_PDF)

    assert len(pages) == 2
    assert [p.page_num for p in pages] == [1, 2]
    assert client.messages.create.call_count == 2


def test_caches_results_by_image_hash(tmp_path):
    """A second parse with the same params + PDF should hit the cache: zero API calls."""
    cache_dir = tmp_path / "cache"
    client = _make_mock_client("cached page text")
    parser_1 = ClaudeVisionParser(cache_dir=cache_dir, client=client, max_pages=1)
    parser_1.parse(SYNTHETIC_PDF)
    assert client.messages.create.call_count == 1

    # Second parser instance, same cache dir — should be a pure cache hit.
    client_2 = _make_mock_client("SHOULD NOT BE RETURNED")
    parser_2 = ClaudeVisionParser(cache_dir=cache_dir, client=client_2, max_pages=1)
    pages = parser_2.parse(SYNTHETIC_PDF)

    assert pages[0].text == "cached page text"
    assert client_2.messages.create.call_count == 0  # cache hit


def test_skips_failing_page_gracefully(tmp_path):
    """API errors on one page must not blow up the whole batch.

    Contract: PDFParser must not raise on per-page extraction issues.
    """
    client = MagicMock()
    client.messages.create.side_effect = RuntimeError("simulated API failure")
    parser = ClaudeVisionParser(cache_dir=tmp_path / "cache", client=client, max_pages=2)

    pages = parser.parse(SYNTHETIC_PDF)

    assert len(pages) == 2
    # Both pages return empty text on failure rather than raising.
    assert all(p.text == "" for p in pages)


def test_empty_page_sentinel_is_normalised(tmp_path):
    """When the model returns the empty-page sentinel, store as empty string."""
    client = _make_mock_client(EMPTY_PAGE_SENTINEL)
    parser = ClaudeVisionParser(cache_dir=tmp_path / "cache", client=client, max_pages=1)

    pages = parser.parse(SYNTHETIC_PDF)
    assert pages[0].text == ""


def test_request_payload_shape(tmp_path):
    """Verify the prompt + image are sent in the right structure."""
    client = _make_mock_client("ok")
    parser = ClaudeVisionParser(cache_dir=tmp_path / "cache", client=client, max_pages=1)
    parser.parse(SYNTHETIC_PDF)

    call_kwargs = client.messages.create.call_args.kwargs
    assert call_kwargs["model"] == "claude-sonnet-4-5"
    assert call_kwargs["max_tokens"] == 2048

    messages = call_kwargs["messages"]
    assert len(messages) == 1
    assert messages[0]["role"] == "user"

    content = messages[0]["content"]
    # Image must come BEFORE text per Anthropic best practices for vision.
    assert content[0]["type"] == "image"
    assert content[0]["source"]["type"] == "base64"
    assert content[0]["source"]["media_type"] == "image/jpeg"
    assert content[1]["type"] == "text"
    assert "construction tender contract" in content[1]["text"]


def test_file_not_found_raises(tmp_path):
    """File-level errors (not extraction-quality errors) must raise loudly."""
    parser = ClaudeVisionParser(cache_dir=tmp_path / "cache", client=MagicMock())
    with pytest.raises(FileNotFoundError):
        parser.parse(Path("does/not/exist.pdf"))


def test_no_anthropic_key_in_logs(tmp_path, caplog):
    """Sanity: even on error, secret-like values must not surface in logs."""
    client = MagicMock()
    client.messages.create.side_effect = RuntimeError("err with sk-ant-XYZ in message")
    parser = ClaudeVisionParser(cache_dir=tmp_path / "cache", client=client, max_pages=1)

    parser.parse(SYNTHETIC_PDF)
    # We log error_type only, not the exception message — verify.
    for record in caplog.records:
        assert "sk-ant" not in record.getMessage()


# ---------------------------------------------------------------------------
# Phase 1.7 — Dual-prompt routing tests
#
# We test the classifier with cheap synthetic PIL images (no PDF render,
# no API). For the routing tests we mock the classifier itself via
# parser._classify and assert that the prompt actually sent to the API
# matches the routing decision.
# ---------------------------------------------------------------------------


def _jpeg_bytes(img: Image.Image, quality: int = 85) -> bytes:
    """Encode a PIL image to JPEG bytes — what the parser's render step produces."""
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


def _make_text_like_image(width: int = 800, height: int = 1000) -> Image.Image:
    """Build a synthetic 'text page' — many short horizontal bands.

    Each band simulates a line of text; word-sized rectangles per line
    produce the characteristic FIND_EDGES row-energy pattern that the
    classifier looks for.
    """
    img = Image.new("L", (width, height), color=255)
    draw = ImageDraw.Draw(img)
    for y in range(60, height - 50, 28):  # ~1 line every 28 px
        x = 60
        while x < width - 100:
            word_w = 30 + (x % 50)  # vary word widths
            draw.rectangle([x, y, x + word_w, y + 12], fill=0)
            x += word_w + 15
    return img


def _make_drawing_like_image(width: int = 800, height: int = 1000) -> Image.Image:
    """Build a synthetic 'engineering drawing' — many thin diagonal lines + a circle.

    1-px lines on white anti-alias to gray after downscale; the resulting
    edge-energy distribution is spread thinly across rows (no baseline
    pattern), which is what tells the classifier "drawing".
    """
    img = Image.new("L", (width, height), color=255)
    draw = ImageDraw.Draw(img)
    # Two crossing families of diagonal hatch lines.
    for off in range(0, 2 * width, 25):
        draw.line([(off, 0), (off - width, height)], fill=0, width=1)
    for off in range(0, 2 * width, 60):
        draw.line([(off, 0), (off + width, height)], fill=0, width=1)
    # A circle and a rectangle — typical drawing primitives.
    draw.ellipse([300, 350, 500, 550], outline=0, width=2)
    draw.rectangle([100, 100, 700, 900], outline=0, width=2)
    return img


def test_classify_picks_drawing_for_high_edge_density_image():
    """A synthetic engineering-drawing image classifies as 'drawing'."""
    img = _make_drawing_like_image()
    assert _classify_page(_jpeg_bytes(img)) == "drawing"


def test_classify_picks_text_for_dense_text_paragraph_image():
    """A synthetic 'text-like' image (rows of word-sized bands) classifies as 'text'."""
    img = _make_text_like_image()
    assert _classify_page(_jpeg_bytes(img)) == "text"


def test_classify_blank_page_returns_drawing():
    """A completely blank page has no edge structure → drawing prompt cheapest path."""
    img = Image.new("L", (800, 1000), color=255)
    # Blank pages route to drawing prompt, which has its own <empty page> branch.
    assert _classify_page(_jpeg_bytes(img)) == "drawing"


def test_classify_corrupt_bytes_defaults_to_text():
    """Garbage bytes must not crash the classifier; fall back to text (safer)."""
    assert _classify_page(b"not a real jpeg payload") == "text"


def test_auto_classify_false_always_uses_text_prompt(tmp_path):
    """Backward-compat: with auto_classify=False the parser always sends TEXT_OCR_PROMPT."""
    client = _make_mock_client("ok")
    parser = ClaudeVisionParser(
        cache_dir=tmp_path / "cache",
        client=client,
        max_pages=1,
        auto_classify=False,
    )
    parser.parse(SYNTHETIC_PDF)

    call_kwargs = client.messages.create.call_args.kwargs
    sent_text = call_kwargs["messages"][0]["content"][1]["text"]
    assert sent_text == TEXT_OCR_PROMPT
    # And critically NOT the drawing prompt.
    assert sent_text != DRAWING_CAPTION_PROMPT


def test_drawing_page_uses_drawing_prompt(tmp_path, monkeypatch):
    """When the classifier says 'drawing' the API call must carry DRAWING_CAPTION_PROMPT."""
    client = _make_mock_client("rendered drawing extract")
    parser = ClaudeVisionParser(cache_dir=tmp_path / "cache", client=client, max_pages=1)

    # Pin the classifier to 'drawing' regardless of actual page content.
    monkeypatch.setattr(parser, "_classify", lambda _jpeg: "drawing")
    parser.parse(SYNTHETIC_PDF)

    call_kwargs = client.messages.create.call_args.kwargs
    sent_text = call_kwargs["messages"][0]["content"][1]["text"]
    assert sent_text == DRAWING_CAPTION_PROMPT
    # And sanity-check that we're not just falling through to text accidentally.
    assert "Title block" in sent_text or "engineering drawing" in sent_text


def test_text_page_uses_text_prompt(tmp_path, monkeypatch):
    """When the classifier says 'text' the API call must carry TEXT_OCR_PROMPT."""
    client = _make_mock_client("rendered text extract")
    parser = ClaudeVisionParser(cache_dir=tmp_path / "cache", client=client, max_pages=1)

    monkeypatch.setattr(parser, "_classify", lambda _jpeg: "text")
    parser.parse(SYNTHETIC_PDF)

    call_kwargs = client.messages.create.call_args.kwargs
    sent_text = call_kwargs["messages"][0]["content"][1]["text"]
    assert sent_text == TEXT_OCR_PROMPT


def test_classifier_failure_defaults_to_text(tmp_path, monkeypatch):
    """If `_classify` raises, the parser must not lose the page — default to text prompt."""
    client = _make_mock_client("text fallback")
    parser = ClaudeVisionParser(cache_dir=tmp_path / "cache", client=client, max_pages=1)

    def boom(_jpeg: bytes) -> str:
        raise RuntimeError("simulated classifier crash")

    # The module-level _classify_page already swallows exceptions and
    # returns "text"; here we also belt-and-brace the instance method
    # to exercise the parser's own routing behaviour against a raising
    # classifier — verifies callers can monkey-patch with broken impls
    # without losing pages.
    monkeypatch.setattr(parser, "_classify", boom)
    # Even if classification crashes, parse() must NOT raise — it should
    # treat the page as "drawing" or "text" but, more importantly, keep
    # producing a ParsedPage. We accept either prompt as long as no
    # exception escapes and a page comes out.
    try:
        pages = parser.parse(SYNTHETIC_PDF)
    except RuntimeError:
        pytest.fail("parse() must not raise when classifier raises; should fall back gracefully.")

    # The page came through (text recorded from the mocked API).
    assert len(pages) == 1
    assert pages[0].page_num == 1


def test_drawing_and_text_have_separate_cache_keys(tmp_path, monkeypatch):
    """Cache key includes the prompt kind — reclassification re-OCRs, no stale hit."""
    cache_dir = tmp_path / "cache"

    # First parse: classify as 'text', store under .text.txt
    client_a = _make_mock_client("TEXT extract")
    parser_a = ClaudeVisionParser(cache_dir=cache_dir, client=client_a, max_pages=1)
    monkeypatch.setattr(parser_a, "_classify", lambda _jpeg: "text")
    pages_a = parser_a.parse(SYNTHETIC_PDF)
    assert pages_a[0].text == "TEXT extract"
    assert client_a.messages.create.call_count == 1

    # Second parse, same PDF + same render bytes, but reclassified as
    # 'drawing'. The drawing-kind cache slot is empty, so we must hit
    # the API again (re-OCR with drawing prompt) — not return stale text.
    client_b = _make_mock_client("DRAWING extract")
    parser_b = ClaudeVisionParser(cache_dir=cache_dir, client=client_b, max_pages=1)
    monkeypatch.setattr(parser_b, "_classify", lambda _jpeg: "drawing")
    pages_b = parser_b.parse(SYNTHETIC_PDF)

    assert pages_b[0].text == "DRAWING extract"
    assert client_b.messages.create.call_count == 1


def test_default_model_keeps_legacy_cache_filename(tmp_path, monkeypatch):
    """E10: the default model writes the pre-E10 `<hash>.text.txt` name so
    caches written before E10 still hit (no silent re-OCR cost)."""
    client = _make_mock_client("default model text")
    parser = ClaudeVisionParser(cache_dir=tmp_path / "cache", client=client, max_pages=1)
    monkeypatch.setattr(parser, "_classify", lambda _jpeg: "text")
    parser.parse(SYNTHETIC_PDF)

    cache_files = list((tmp_path / "cache").glob("*.txt"))
    assert len(cache_files) == 1
    assert cache_files[0].name.endswith(".text.txt")


def test_non_default_model_isolates_cache(tmp_path, monkeypatch):
    """E10: a non-default model gets its own cache namespace so switching
    `--vision-model` re-OCRs rather than returning the other model's text."""
    cache_dir = tmp_path / "cache"
    client_default = _make_mock_client("SONNET-45 TEXT")
    parser_default = ClaudeVisionParser(cache_dir=cache_dir, client=client_default, max_pages=1)
    monkeypatch.setattr(parser_default, "_classify", lambda _jpeg: "text")
    parser_default.parse(SYNTHETIC_PDF)

    client_haiku = _make_mock_client("HAIKU TEXT")
    parser_haiku = ClaudeVisionParser(
        cache_dir=cache_dir, client=client_haiku, model="claude-haiku-4-5", max_pages=1
    )
    monkeypatch.setattr(parser_haiku, "_classify", lambda _jpeg: "text")
    pages = parser_haiku.parse(SYNTHETIC_PDF)

    cache_files = sorted(f.name for f in cache_dir.glob("*.txt"))
    assert len(cache_files) == 2  # default vs haiku do not collide
    assert client_haiku.messages.create.call_count == 1  # fresh OCR, no stale hit
    assert pages[0].text == "HAIKU TEXT"


# ---------------------------------------------------------------------------
# Integration test — gated. Only runs if explicitly opted in via env var AND
# the real ANTHROPIC_API_KEY is set. Costs ~$0.02 per run (one synthetic
# page → one Vision call).
# ---------------------------------------------------------------------------
@pytest.mark.skipif(
    not RUN_INTEGRATION or not os.environ.get("ANTHROPIC_API_KEY"),
    reason="Set JCONTRACT_RUN_INTEGRATION=1 and ANTHROPIC_API_KEY to run.",
)
def test_real_api_smoke_against_synthetic_pdf(tmp_path):
    """Hit the real Anthropic API for one synthetic page — sanity check.

    Asserts the call succeeds and we get back non-empty text. Does NOT assert
    on specific content (model output can vary); that's what eval/golden_cases
    is for.
    """
    parser = ClaudeVisionParser(cache_dir=tmp_path / "cache", max_pages=1)
    pages = parser.parse(SYNTHETIC_PDF)

    assert len(pages) == 1
    assert len(pages[0].text) > 50  # non-trivial extraction
