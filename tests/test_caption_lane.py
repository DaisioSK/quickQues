"""ssCL caption-lane tests: page_kind → drawing chunks → captions in the index.

What this file proves (the FORESHADOW-ls.4 fix, unit level):
  1. Zero default behaviour change — ``ParsedPage.page_kind`` defaults to
     "text" and an all-text document chunks byte-for-byte as before.
  2. The chunker emits exactly ONE ``chunk_type="drawing"`` chunk per
     drawing-classified page (DECISION-cq.10), including for pages whose
     OCR text is empty (pure-graphic pages are the lane's whole point).
  3. The IngestPipeline attaches captions to those chunks and the caption
     reaches the EMBEDDED text (``chunk_indexable_text`` fusion) — i.e.
     ``ingest.captioned`` actually fires and the caption is retrievable.
  4. Parsers surface page_kind: rapidocr (new call site) and the
     LLM vendors (existing classifier, now recorded on ParsedPage).

The real-model e2e (ollama captioner + real drawing page) runs in the
ssCL sub-sprint itself; here every external engine is faked.
"""

from __future__ import annotations

import types
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
from structlog.testing import capture_logs

from jcontract.impls.qa_chunker import QaAwareChunker
from jcontract.impls.rapidocr_parser import RapidOcrParser
from jcontract.ingest.pipeline import IngestPipeline
from jcontract.interfaces import DrawingCaption, ParsedPage
from jcontract.interfaces.schema import Chunk, chunk_indexable_text

SYNTHETIC_PDF = Path("eval/fixtures/synthetic_contract_tqa.pdf")

_TEXT_PAGE_BODY = (
    "Question No.: TQA-001\n"
    "What waterproofing applies at the pier?\n"
    "Answer: Per Clause 7.3 the Trackwork Contractor provides membrane waterproofing.\n"
)
_DRAWING_PAGE_BODY = "TITLE: PIER WATERPROOFING DETAIL Drawing No. T/PRJ/CWD/WS/2101A Rev A"


# --------------------------------------------------------------------------- #
# Schema default — the zero-behaviour-change guarantee
# --------------------------------------------------------------------------- #


def test_parsed_page_kind_defaults_to_text() -> None:
    """Every pre-ssCL construction site gets 'text' without changes."""
    page = ParsedPage(page_num=1, text="hello")
    assert page.page_kind == "text"


def test_all_text_document_chunks_identically_with_and_without_field() -> None:
    """Explicit page_kind='text' must be indistinguishable from the default."""
    chunker = QaAwareChunker()
    implicit = chunker.chunk([ParsedPage(page_num=1, text=_TEXT_PAGE_BODY)], "doc.pdf")
    explicit = chunker.chunk(
        [ParsedPage(page_num=1, text=_TEXT_PAGE_BODY, page_kind="text")], "doc.pdf"
    )
    assert implicit == explicit
    assert all(c.chunk_type != "drawing" for c in implicit)


# --------------------------------------------------------------------------- #
# Chunker drawing lane (DECISION-cq.10: one whole-page drawing chunk)
# --------------------------------------------------------------------------- #


def test_drawing_page_becomes_single_drawing_chunk() -> None:
    pages = [
        ParsedPage(page_num=1, text=_TEXT_PAGE_BODY),
        ParsedPage(page_num=2, text=_DRAWING_PAGE_BODY, page_kind="drawing"),
    ]
    chunks = QaAwareChunker().chunk(pages, "doc.pdf")

    drawing = [c for c in chunks if c.chunk_type == "drawing"]
    assert len(drawing) == 1
    assert drawing[0].page == 2
    assert drawing[0].text == _DRAWING_PAGE_BODY
    # Title-block cross-references are extracted like any other chunk.
    assert drawing[0].drawing_refs == ["T/PRJ/CWD/WS/2101A"]
    # The drawing page's text must NOT also appear in paragraph/qa chunks
    # (no double indexing).
    non_drawing = [c for c in chunks if c.chunk_type != "drawing"]
    assert all("PIER WATERPROOFING DETAIL" not in c.text for c in non_drawing)
    # And the text page still produces its qa_pair as before.
    assert any(c.chunk_type == "qa_pair" and c.question_no == "TQA-001" for c in non_drawing)


def test_text_page_chunks_unchanged_by_sibling_drawing_page() -> None:
    """Adding a trailing drawing page must not alter the text page's chunks."""
    chunker = QaAwareChunker()
    text_only = chunker.chunk([ParsedPage(page_num=1, text=_TEXT_PAGE_BODY)], "doc.pdf")
    mixed = chunker.chunk(
        [
            ParsedPage(page_num=1, text=_TEXT_PAGE_BODY),
            ParsedPage(page_num=2, text=_DRAWING_PAGE_BODY, page_kind="drawing"),
        ],
        "doc.pdf",
    )
    assert mixed[: len(text_only)] == text_only


def test_empty_text_drawing_page_still_emits_chunk() -> None:
    """A pure-graphic page (OCR found nothing) is exactly the page whose only
    retrievable surface will be the caption — it must not be dropped."""
    pages = [ParsedPage(page_num=1, text="", page_kind="drawing")]
    chunks = QaAwareChunker().chunk(pages, "doc.pdf")

    assert len(chunks) == 1
    assert chunks[0].chunk_type == "drawing"
    assert chunks[0].text == ""
    assert chunks[0].page == 1


def test_one_drawing_chunk_per_drawing_page_in_page_order() -> None:
    pages = [
        ParsedPage(page_num=1, text="d-one", page_kind="drawing"),
        ParsedPage(page_num=2, text=_TEXT_PAGE_BODY),
        ParsedPage(page_num=3, text="d-three", page_kind="drawing"),
    ]
    chunks = QaAwareChunker().chunk(pages, "doc.pdf")

    drawing = [c for c in chunks if c.chunk_type == "drawing"]
    assert [(c.page, c.text) for c in drawing] == [(1, "d-one"), (3, "d-three")]
    # Ids stay unique across the whole document.
    assert len({c.id for c in chunks}) == len(chunks)


# --------------------------------------------------------------------------- #
# Pipeline: caption attaches and reaches the embedded text
# --------------------------------------------------------------------------- #


class _FakeParser:
    def __init__(self, pages: list[ParsedPage]) -> None:
        self._pages = pages

    def parse(self, pdf_path: Path) -> list[ParsedPage]:
        return self._pages


class _FakeEmbedder:
    dim = 4

    def __init__(self) -> None:
        self.seen_texts: list[str] = []

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.seen_texts.extend(texts)
        return [[0.0] * self.dim for _ in texts]


class _FakeVectorStore:
    def __init__(self) -> None:
        self.chunks: list[Chunk] = []

    def add(self, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        self.chunks.extend(chunks)

    def count(self) -> int:
        return len(self.chunks)

    def search(self, vector: list[float], k: int = 5) -> list[Any]:
        return []


class _FakeKeywordIndex:
    def __init__(self) -> None:
        self.chunks: list[Chunk] = []

    def add(self, chunks: list[Chunk]) -> None:
        self.chunks.extend(chunks)

    def search(self, query: str, k: int = 5) -> list[Any]:
        return []


class _FakeCaptioner:
    def __init__(self) -> None:
        self.calls = 0

    def caption(self, image_bytes: bytes, ocr_text: str) -> DrawingCaption:
        self.calls += 1
        return DrawingCaption(caption_zh="桥墩防水构造图", entities=[])


def _run_pipeline_with_drawing_page(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[_FakeEmbedder, _FakeCaptioner, list[Chunk], list[Any]]:
    pages = [
        ParsedPage(page_num=1, text=_TEXT_PAGE_BODY),
        ParsedPage(page_num=2, text=_DRAWING_PAGE_BODY, page_kind="drawing"),
    ]
    embedder = _FakeEmbedder()
    captioner = _FakeCaptioner()
    store = _FakeVectorStore()
    # _attach_captions renders the drawing page; keep the test render-free.
    monkeypatch.setattr(
        "jcontract.impls.claude_vision_captioner.render_page_to_jpeg",
        lambda pdf_path, page_num: b"\xff\xd8fakejpeg",
    )
    pipeline = IngestPipeline(
        parser=_FakeParser(pages),
        chunker=QaAwareChunker(),
        embedder=embedder,
        vector_store=store,
        keyword_index=_FakeKeywordIndex(),
        chunks_snapshot_path=tmp_path / "chunks_snapshot.jsonl",
        captioner=captioner,
    )
    with capture_logs() as logs:
        n = pipeline.ingest(SYNTHETIC_PDF)
    assert n == len(store.chunks)
    return embedder, captioner, store.chunks, logs


def test_pipeline_captions_drawing_chunk_and_embeds_caption(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    embedder, captioner, chunks, logs = _run_pipeline_with_drawing_page(tmp_path, monkeypatch)

    drawing = [c for c in chunks if c.chunk_type == "drawing"]
    assert len(drawing) == 1
    # One captioner call per drawing page (whole-page chunk, DECISION-cq.10).
    assert captioner.calls == 1
    assert drawing[0].caption == "桥墩防水构造图"
    # The caption must reach the retrievable text surface: the embedder saw
    # the fused text (chunk_indexable_text), not the bare OCR text.
    fused = chunk_indexable_text(drawing[0])
    assert "Caption: 桥墩防水构造图" in fused
    assert fused in embedder.seen_texts
    # FORESHADOW-ls.4 anchor: ingest.captioned actually fires now.
    assert any(entry["event"] == "ingest.captioned" for entry in logs)


def test_pipeline_without_captioner_leaves_drawing_caption_none(tmp_path: Path) -> None:
    """--no-caption ingest: drawing chunks exist but caption stays None."""
    pages = [ParsedPage(page_num=1, text=_DRAWING_PAGE_BODY, page_kind="drawing")]
    embedder = _FakeEmbedder()
    pipeline = IngestPipeline(
        parser=_FakeParser(pages),
        chunker=QaAwareChunker(),
        embedder=embedder,
        vector_store=_FakeVectorStore(),
        keyword_index=_FakeKeywordIndex(),
        chunks_snapshot_path=tmp_path / "chunks_snapshot.jsonl",
        captioner=None,
    )
    with capture_logs() as logs:
        pipeline.ingest(SYNTHETIC_PDF)
    assert not any(entry["event"] == "ingest.captioned" for entry in logs)
    # Embedder saw the bare OCR text (no Caption: separator injected).
    assert embedder.seen_texts == [_DRAWING_PAGE_BODY]


# --------------------------------------------------------------------------- #
# Parsers surface page_kind
# --------------------------------------------------------------------------- #


def _rapidocr_engine(txt: str) -> MagicMock:
    box = [[10.0, 10.0], [100.0, 10.0], [100.0, 40.0], [10.0, 40.0]]
    engine = MagicMock()
    engine.return_value = types.SimpleNamespace(boxes=[box], txts=(txt,), scores=(0.99,))
    return engine


def test_rapidocr_synthetic_text_page_is_text_kind(tmp_path: Path) -> None:
    """The real heuristic on the real text fixture must say 'text'."""
    parser = RapidOcrParser(
        cache_dir=tmp_path / "cache", engine=_rapidocr_engine("hello"), max_pages=1
    )
    pages = parser.parse(SYNTHETIC_PDF)
    assert pages[0].page_kind == "text"


def test_rapidocr_drawing_verdict_lands_on_parsed_page(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    parser = RapidOcrParser(
        cache_dir=tmp_path / "cache", engine=_rapidocr_engine("title block"), max_pages=1
    )
    monkeypatch.setattr(parser, "_classify", lambda _jpeg: "drawing")
    pages = parser.parse(SYNTHETIC_PDF)
    assert pages[0].page_kind == "drawing"
    # The verdict must not leak into OCR output or cache layout.
    assert pages[0].text == "title block"
    assert len(list((tmp_path / "cache").glob("rapidocr-*.text.txt"))) == 1


def test_rapidocr_auto_classify_off_forces_text(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    parser = RapidOcrParser(
        cache_dir=tmp_path / "cache",
        engine=_rapidocr_engine("x"),
        max_pages=1,
        auto_classify=False,
    )
    monkeypatch.setattr(parser, "_classify", lambda _jpeg: "drawing")
    pages = parser.parse(SYNTHETIC_PDF)
    assert pages[0].page_kind == "text"


def test_rapidocr_classifier_crash_falls_back_to_text(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    parser = RapidOcrParser(cache_dir=tmp_path / "cache", engine=_rapidocr_engine("x"), max_pages=1)

    def boom(_jpeg: bytes) -> str:
        raise RuntimeError("simulated classifier crash")

    monkeypatch.setattr(parser, "_classify", boom)
    pages = parser.parse(SYNTHETIC_PDF)
    assert pages[0].page_kind == "text"
    assert pages[0].text == "x"


def test_claude_vision_parser_records_page_kind(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The LLM vendor's existing classifier verdict now reaches ParsedPage."""
    from jcontract.impls.claude_vision_parser import ClaudeVisionParser

    block = types.SimpleNamespace(type="text", text="drawing extract")
    usage = types.SimpleNamespace(input_tokens=10, output_tokens=5)
    client = MagicMock()
    client.messages.create.return_value = types.SimpleNamespace(content=[block], usage=usage)

    parser = ClaudeVisionParser(cache_dir=tmp_path / "cache", client=client, max_pages=1)
    monkeypatch.setattr(parser, "_classify", lambda _jpeg: "drawing")
    pages = parser.parse(SYNTHETIC_PDF)

    assert pages[0].page_kind == "drawing"
    # Prompt routing still follows the same verdict (single classification).
    sent_text = client.messages.create.call_args.kwargs["messages"][0]["content"][1]["text"]
    assert "engineering drawing" in sent_text


def test_deepseek_parser_records_page_kind(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from jcontract.impls.deepseek_v4_parser import DeepSeekV4Parser

    message = types.SimpleNamespace(content="extract")
    choice = types.SimpleNamespace(message=message)
    client = MagicMock()
    client.chat.completions.create.return_value = types.SimpleNamespace(
        choices=[choice], usage=None
    )

    parser = DeepSeekV4Parser(cache_dir=tmp_path / "cache", client=client, max_pages=1)
    monkeypatch.setattr(parser, "_classify", lambda _jpeg: "drawing")
    pages = parser.parse(SYNTHETIC_PDF)

    assert pages[0].page_kind == "drawing"
