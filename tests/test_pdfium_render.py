"""Concurrency-determinism tests for the shared pdfium render helpers.

Regression for the 2026-06-10 canary finding: batch-ingest Phase A renders
pages from multiple threads; unserialized pdfium use drifted at the pixel
level (and segfaulted under concurrent open/close), splitting the
sha256-addressed OCR cache and re-OCR'ing pages at index time.
[DECISION-ab3.46 dev-sprint v3 §13]
"""

from __future__ import annotations

import hashlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pypdfium2 as pdfium

from jcontract.impls._pdfium_render import render_page_jpeg, render_pdf_page_jpeg

SYNTHETIC_PDF = Path("eval/fixtures/synthetic_contract_tqa.pdf")

DPI = 150
JPEG_QUALITY = 85


def _hash_page_concurrent_entry(pdf_path: Path, page_num: int) -> str:
    """sha256 via render_pdf_page_jpeg — the thread-safe batch entry point."""
    jpeg = render_pdf_page_jpeg(pdf_path, page_num, dpi=DPI, jpeg_quality=JPEG_QUALITY)
    return hashlib.sha256(jpeg).hexdigest()


def _page_count(pdf_path: Path) -> int:
    pdf = pdfium.PdfDocument(str(pdf_path))
    try:
        return len(pdf)
    finally:
        pdf.close()


def test_render_deterministic_sequential() -> None:
    # Two sequential renders of the same page must be byte-identical —
    # the foundation of the content-addressed OCR cache.
    assert _hash_page_concurrent_entry(SYNTHETIC_PDF, 1) == _hash_page_concurrent_entry(
        SYNTHETIC_PDF, 1
    )


def test_render_deterministic_under_thread_concurrency() -> None:
    # 8 threads x 4 pages x 2 repeats through render_pdf_page_jpeg (the
    # safe concurrent entry: open->render->close all inside the global
    # pdfium lock) must (a) not crash — a render-only lock segfaulted
    # here because concurrent open/close also touches pdfium global
    # state — and (b) reproduce the sequential hashes exactly. Without
    # serialisation this drifted at the pixel level, the bug that
    # double-OCR'd the 2026-06-10 sonnet canary.
    n_pages = min(4, _page_count(SYNTHETIC_PDF))

    expected = {
        page_num: _hash_page_concurrent_entry(SYNTHETIC_PDF, page_num)
        for page_num in range(1, n_pages + 1)
    }

    jobs = [page_num for page_num in range(1, n_pages + 1) for _ in range(2)]
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(lambda n: (n, _hash_page_concurrent_entry(SYNTHETIC_PDF, n)), jobs))

    for page_num, digest in results:
        assert digest == expected[page_num], f"page {page_num} drifted"


def test_sequential_and_concurrent_entry_points_produce_identical_bytes() -> None:
    # THE cache-key interop assertion: parse() (sequential, renders via
    # render_page_jpeg on an open document) and batch-ingest (concurrent,
    # renders via render_pdf_page_jpeg) must produce byte-identical JPEGs
    # for the same page/dpi/quality — otherwise Phase B re-OCRs every
    # page Phase A just paid for (the canary's double-quota bug).
    n_pages = min(4, _page_count(SYNTHETIC_PDF))

    pdf = pdfium.PdfDocument(str(SYNTHETIC_PDF))
    try:
        for page_index in range(n_pages):
            via_page = render_page_jpeg(pdf[page_index], dpi=DPI, jpeg_quality=JPEG_QUALITY)
            via_path = render_pdf_page_jpeg(
                SYNTHETIC_PDF, page_index + 1, dpi=DPI, jpeg_quality=JPEG_QUALITY
            )
            assert via_page == via_path, f"page {page_index + 1}: entry points diverged"
    finally:
        pdf.close()
