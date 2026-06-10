"""Serialized pdfium access — shared by every vision parser/captioner.

What:
    A process-global lock (``_PDFIUM_LOCK``) that serialises **all** pdfium
    operations, plus two render entry points built on it:

    * ``render_page_jpeg(page, ...)`` — rasterise an already-open
      ``pdfium.PdfPage``. For single-threaded sequential callers
      (``parser.parse`` loops) that own the document lifecycle.
    * ``render_pdf_page_jpeg(pdf_path, page_num, ...)`` — open the
      document, load the page, rasterise, and close — all inside the
      lock. The ONLY safe entry point for concurrent callers
      (``batch-ingest`` Phase A workers).

Why:
    pdfium is NOT thread-safe — and not just for rendering: opening,
    loading pages from, and closing documents all touch pdfium's global
    state. ``batch-ingest`` Phase A runs OCR workers via
    ``asyncio.to_thread``, and concurrent rasterisation produces
    pixel-level drift between runs. The JPEG bytes are the
    content-addressed OCR cache key (sha256), so drifting pixels split
    the cache namespace and every page silently re-OCRs at index time —
    double quota spend. Observed on the 2026-06-10 sonnet canary: 3/4
    pages re-OCR'd in batch Phase B. A render-only lock was tried first
    and segfaulted under an 8-thread open/close test (other threads were
    inside pdfium document setup while a render held the lock).
    Rendering costs ~100ms/page vs 15-25s per OCR network call, so
    serialising every pdfium call leaves OCR concurrency throughput
    intact.

Context:
    [DECISION-ab3.46 dev-sprint v3 §13] — one global lock chosen over a
    per-document lock (pdfium's mutable state is process-wide, not
    per-document) and over ``--max-concurrent 1`` (would serialise the
    network calls too, ~4x slower wall-clock on a 4100-page corpus).
    JPEG encoding stays outside the lock: PIL encodes identical pixels
    deterministically and is thread-safe for independent Image objects,
    so both entry points produce byte-identical JPEGs for the same page
    — the property the cross-entry-point cache hit depends on.
"""

from __future__ import annotations

import io
import threading
from pathlib import Path

import pypdfium2 as pdfium
from PIL import Image

# One lock for ALL pdfium calls (open/load/render/close) — see module
# docstring for why render-only scope segfaults.
_PDFIUM_LOCK = threading.Lock()


def _encode_jpeg(pil_image: Image.Image, jpeg_quality: int) -> bytes:
    """JPEG-encode outside the lock — deterministic for identical pixels."""
    buf = io.BytesIO()
    pil_image.save(buf, format="JPEG", quality=jpeg_quality)
    return buf.getvalue()


def render_page_jpeg(page: pdfium.PdfPage, *, dpi: int, jpeg_quality: int) -> bytes:
    """Rasterise an already-open page to JPEG bytes (sequential callers).

    The caller owns the document lifecycle and must not share it across
    threads. Concurrent callers must use ``render_pdf_page_jpeg`` instead
    — opening/closing a document outside the lock races with this render.
    """
    with _PDFIUM_LOCK:
        pil_image = page.render(scale=dpi / 72.0).to_pil()
    return _encode_jpeg(pil_image, jpeg_quality)


def render_pdf_page_jpeg(pdf_path: Path, page_num: int, *, dpi: int, jpeg_quality: int) -> bytes:
    """Open ``pdf_path``, render 1-indexed ``page_num`` to JPEG, close — thread-safe.

    The full pdfium lifecycle (open → load page → render → close) happens
    inside ``_PDFIUM_LOCK`` so any number of threads can call this
    concurrently. Only the JPEG encode runs outside the lock. Produces
    bytes identical to ``render_page_jpeg`` for the same page/dpi/quality
    (asserted by tests/test_pdfium_render.py) — both paths feed the same
    sha256 cache key. [DECISION-ab3.46 dev-sprint v3 §13]
    """
    with _PDFIUM_LOCK:
        pdf = pdfium.PdfDocument(str(pdf_path))
        try:
            pil_image = pdf[page_num - 1].render(scale=dpi / 72.0).to_pil()
        finally:
            pdf.close()
    return _encode_jpeg(pil_image, jpeg_quality)
