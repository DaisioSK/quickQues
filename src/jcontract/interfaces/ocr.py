"""OCREngine Protocol + OCRBlock — Layer 0 (Phase 2).

Finalized by sub-sprint p2-ss-prep. Was a placeholder Protocol shape in
Phase 1; now a concrete dataclass for the per-block output plus a
refined Protocol for the engine.

Why a separate OCREngine vs PDFParser:
- PDFParser returns ParsedPage (one row of text per page) and is the
  ingest-side abstraction.
- OCREngine returns OCRBlock (a region with bbox + confidence) and is a
  lower-level abstraction. A PDFParser impl MAY wrap an OCREngine (e.g.
  impls/paddleocr_parser.py in sub-sprint p2-ssPaddleOCR will assemble
  OCRBlocks into a ParsedPage), but an OCREngine consumer (e.g. a
  future bbox-aware highlighter) can use it directly without paying for
  ParsedPage assembly.

Default impl: impls/paddleocr_engine.py (sub-sprint p2-ssPaddleOCR;
RapidOCR-onnxruntime backend, no API key required — DECISION-2.paddle.1
in docs/dev-sprint.md).
Replacement candidates per docs/project_guideline.md §4:
  - Tesseract (FORESHADOW; weak on Chinese)
  - 阿里 OCR / Cloud OCR (FORESHADOW; needs API key, defeats offline)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


@dataclass(frozen=True)
class OCRBlock:
    """A single OCR'd region with location and confidence.

    Coordinates: ``bbox = (x0, y0, x1, y1)`` in PDF user-space units
    (1 unit = 1/72 inch — same as pypdfium2's native coordinate system).
    Page-relative; (0, 0) is page top-left to match the upstream
    rendering convention. Impls that produce pixel-space bboxes MUST
    convert before returning.

    Confidence: float in [0.0, 1.0]. Engine-specific semantics — for
    RapidOCR it's the recognizer's per-block softmax confidence; for
    other engines it may be a fused detection+recognition score.
    Callers MAY filter on confidence (the default impl drops < 0.5) but
    MUST NOT compare across engines — see SearchResult.score docstring
    for the same caveat at retrieval layer.
    """

    text: str
    page_num: int  # 1-indexed, matches ParsedPage / Chunk convention
    bbox: tuple[float, float, float, float]
    confidence: float


class OCREngine(Protocol):
    """Run OCR over a set of pages from a PDF and return per-region blocks.

    Args:
        pdf_path: Path to the PDF on disk.
        page_nums: 1-indexed list of page numbers to OCR. Empty list is
            valid and MUST return an empty list (cheap no-op).

    Returns:
        Flat list of OCRBlocks across all requested pages, in
        reading-order within each page and ordered by page_num across
        pages. Pages where OCR found nothing return zero blocks for that
        page (NOT a "<empty>" sentinel block — that's a presentation
        concern, not the engine's job).

        Impls MUST NOT raise on per-page extraction issues — log a
        warning and produce zero blocks for that page. The PDFParser
        contract reasons the same way (impls/claude_vision_parser.py).
    """

    def ocr_pages(self, pdf_path: Path, page_nums: list[int]) -> list[OCRBlock]: ...
