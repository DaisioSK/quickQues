"""Unit + smoke tests for ``PyPdfParser``.

Strategy:
  * Unit tests build a tiny in-memory PDF with known content using
    pypdf's writer + a hand-built text content stream. This keeps tests
    hermetic — no on-disk fixture, no network — while still exercising
    the real ``pypdf.PdfReader`` code path our impl uses.
  * The ``test_smoke_real_pdf`` test is marked ``@pytest.mark.slow``
    and runs against ``input-docs/Contract DEMO(1of9) TQA.pdf`` if
    present. The TQA samples are scan-only image PDFs (no text layer),
    so we only assert *parser-shape* invariants (page count, 1-indexed
    page_num) and gracefully document the empty-text reality as a known
    Phase-2-OCR follow-up.
"""

from __future__ import annotations

import io
from pathlib import Path

import pytest
from pypdf import PdfWriter
from pypdf.generic import (
    DictionaryObject,
    NameObject,
    NumberObject,
    StreamObject,
)

from jcontract.impls.pypdf_parser import PyPdfParser
from jcontract.interfaces.schema import ParsedPage

# ---------------------------------------------------------------------------
# Helper: build a tiny PDF with controllable per-page text. We construct
# the content stream by hand because pypdf has no high-level write-text
# API. The output is a real PDF that pypdf can read back.
# ---------------------------------------------------------------------------


def _build_text_pdf(per_page_text: list[str]) -> bytes:
    """Return PDF bytes whose pages contain the given UTF-friendly lines.

    Only ASCII / latin-1-safe characters are supported (we use Helvetica
    Type1). That is enough to exercise the parser; chunker tests don't
    go through PDF at all, they feed ParsedPage directly.
    """
    writer = PdfWriter()
    for text in per_page_text:
        page = writer.add_blank_page(width=612, height=792)

        # Build a PDF text-show operator sequence: start text, set font,
        # move to (50, 750), show each line then move down 16 points.
        ops: list[str] = ["BT", "/F1 12 Tf", "50 750 Td"]
        for i, line in enumerate(text.split("\n")):
            if i > 0:
                ops.append("0 -16 Td")
            safe = line.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
            ops.append(f"({safe}) Tj")
        ops.append("ET")
        body = "\n".join(ops).encode("latin-1")

        stream = StreamObject()
        # _data + /Length are how pypdf's StreamObject expects raw stream
        # content; we bypass the public encode path because we control
        # the bytes directly.
        stream._data = body
        stream[NameObject("/Length")] = NumberObject(len(body))
        content_ref = writer._add_object(stream)

        font = DictionaryObject(
            {
                NameObject("/Type"): NameObject("/Font"),
                NameObject("/Subtype"): NameObject("/Type1"),
                NameObject("/BaseFont"): NameObject("/Helvetica"),
            }
        )
        font_ref = writer._add_object(font)

        page[NameObject("/Contents")] = content_ref
        page[NameObject("/Resources")] = DictionaryObject(
            {
                NameObject("/Font"): DictionaryObject({NameObject("/F1"): font_ref}),
            }
        )

    buf = io.BytesIO()
    writer.write(buf)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------


def test_parse_returns_one_parsedpage_per_source_page(tmp_path: Path) -> None:
    pdf_bytes = _build_text_pdf(["page one body", "page two body", "third"])
    pdf_path = tmp_path / "tiny.pdf"
    pdf_path.write_bytes(pdf_bytes)

    pages = PyPdfParser().parse(pdf_path)

    assert len(pages) == 3
    assert all(isinstance(p, ParsedPage) for p in pages)


def test_page_num_is_one_indexed(tmp_path: Path) -> None:
    pdf_path = tmp_path / "tiny.pdf"
    pdf_path.write_bytes(_build_text_pdf(["a", "b"]))

    pages = PyPdfParser().parse(pdf_path)

    assert [p.page_num for p in pages] == [1, 2]


def test_extracted_text_contains_source_words(tmp_path: Path) -> None:
    pdf_path = tmp_path / "tiny.pdf"
    pdf_path.write_bytes(
        _build_text_pdf(
            [
                "Question No.: TQA-001\nDrawing No. T/PRJ/CWD/WS/2101A",
                "Section 7\nClause 7.3 waterproofing scope.",
            ]
        )
    )

    pages = PyPdfParser().parse(pdf_path)

    assert "Question No" in pages[0].text
    assert "T/PRJ/CWD/WS/2101A" in pages[0].text
    assert "Section 7" in pages[1].text


def test_tables_field_is_empty_list(tmp_path: Path) -> None:
    # pypdf does not isolate tables; the field must be present but empty
    # so downstream code can treat .tables uniformly.
    pdf_path = tmp_path / "tiny.pdf"
    pdf_path.write_bytes(_build_text_pdf(["body"]))

    pages = PyPdfParser().parse(pdf_path)

    assert pages[0].tables == []


def test_missing_file_raises_loudly(tmp_path: Path) -> None:
    # File-level errors MUST surface — they signal a misconfigured
    # ingest pipeline, not a single-page extraction blip.
    missing = tmp_path / "does-not-exist.pdf"
    with pytest.raises((FileNotFoundError, OSError)):
        PyPdfParser().parse(missing)


# ---------------------------------------------------------------------------
# Smoke test — runs against the real contract PDF when available.
#
# The two DEMO sample PDFs are image-only scans, so pypdf returns
# empty strings. We still verify parser shape (pypdf opens the file,
# walks pages, hands back ParsedPage objects with the right page_num
# sequence). Text-quality assertions await the OCR impl (Phase 2).
# ---------------------------------------------------------------------------

_REAL_PDF = Path("input-docs/Contract DEMO(1of9) TQA.pdf")


@pytest.mark.slow
def test_smoke_real_pdf_shape() -> None:
    if not _REAL_PDF.exists():
        pytest.skip(f"Real PDF not available at {_REAL_PDF}")

    pages = PyPdfParser().parse(_REAL_PDF)

    # Parser shape invariants (vendor-independent):
    assert len(pages) > 5, "expected DEMO TQA to span many pages"
    assert pages[0].page_num == 1
    assert pages[-1].page_num == len(pages)
    # Pages are in strict ascending order with no gaps:
    assert [p.page_num for p in pages] == list(range(1, len(pages) + 1))

    # Reality check: TQA samples are scanned image PDFs. Text extraction
    # via pypdf will return empty strings. We do NOT assert any text
    # because that would be faking results. The integrator will swap in
    # an OCR-backed parser in Phase 2 before re-running the smoke.
    total_text = sum(len(p.text) for p in pages)
    if total_text == 0:
        pytest.xfail(
            "TQA PDF is image-only — pypdf cannot extract text. "
            "FORESHADOW: Phase 2 OCR will replace pypdf as default for "
            "scan-heavy contracts."
        )
