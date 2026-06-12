"""Table structure assembly — rapid-table SLANet-plus → cell list → markdown.

What:
    The ssTB mechanism kernel: ``structure_table`` feeds one rendered page
    (JPEG bytes) plus the page's existing RapidOCR results into the
    rapid-table SLANet-plus structure model and returns a flat
    ``list[TableCell]`` — normalized geometry + logical row/col indices +
    text. Two deterministic pure renderers turn that list into the
    user-facing views: ``render_markdown`` (embedding/retrieval view) and
    ``render_elements`` (JSONL geometry view for citation/highlight).

Why:
    Element list is the ground truth, markdown is derived [DECISION-tt.21
    dev-sprint v6 §13]: retrieval wants a linearized text table, while
    page-highlighting wants geometry — deriving md from the cell list keeps
    the two views consistent by construction. Engine choice (rapid-table
    3.0.2, SLANETPLUS on CPU onnxruntime) per [DECISION-tt.20]: the only
    candidate with non-zero output on all 5 frozen sample table pages.

Context:
    Mechanism only — nothing here is wired into the chunker or default
    ingest (chunk_type="table" activation is a contract-level change,
    FORESHADOW-tt.1). Consumed by the ``table-preview`` CLI command.
    Known engine behaviour, recorded not fixed: SLANet-plus tends to absorb
    page prose (letterhead, headings) into the first row's cells.
"""

from __future__ import annotations

import io
import json
import threading
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from html.parser import HTMLParser
from typing import Any

import structlog
from PIL import Image

logger = structlog.get_logger(__name__)

# Default page-level rotation metadata for the elements view. Rotation
# detection is an upstream, separate mechanism (DECISION-tt.23, delivered
# by ssRT's _page_orient probe) — callers that rotated the frame before
# structuring pass the applied rotation to render_elements; everything
# else keeps the as-rendered frame and this 0 default.
ROTATION = 0

# Normalized coordinates are rounded to 4 decimal places — at 150 DPI / A4
# (~1240px wide) that is sub-pixel precision, and fixed-width output keeps
# the elements view byte-deterministic. [DECISION-tt.22]
_COORD_DECIMALS = 4


@dataclass(frozen=True)
class TableCell:
    """One table cell: logical grid position + normalized geometry + text.

    ``row``/``col`` are the logical start indices, ``row_end``/``col_end``
    the inclusive end indices (end > start means the cell spans). Geometry
    is ``x, y, w, h`` in [0, 1] relative to the rendered page frame,
    clamped — SLANet-plus logical bboxes can overflow the image (w=1.117
    measured on a frozen sample page), so clamping is mandatory, not
    defensive decoration. [DECISION-tt.22]
    """

    row: int
    row_end: int
    col: int
    col_end: int
    x: float
    y: float
    w: float
    h: float
    text: str


class _TdTextExtractor(HTMLParser):
    """Collect the text of every <td>/<th> in document order (stdlib only).

    Why hand-rolled: the SLANet-plus matcher emits a flat, machine-generated
    ``<table><tr><td ...>`` document whose td order is index-aligned with
    ``cell_bboxes``/``logic_points`` (verified live on the frozen sample
    pages: 61/61/61 and 55/55/55). A full HTML library (bs4) for this is a
    dependency the 8-question check would reject — stdlib HTMLParser
    handles the well-formed subset deterministically.
    """

    def __init__(self) -> None:
        super().__init__()
        self.tds: list[str] = []
        self._buf: list[str] = []
        self._in_td = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in ("td", "th"):
            self._in_td = True
            self._buf = []

    def handle_endtag(self, tag: str) -> None:
        if tag in ("td", "th") and self._in_td:
            self.tds.append("".join(self._buf))
            self._in_td = False

    def handle_data(self, data: str) -> None:
        if self._in_td:
            self._buf.append(data)


# Process-wide engine singleton: first construction loads the slanet-plus
# ONNX session (and may trigger the one-time ~7.5MB model download), so it
# must happen once per process, not once per page. The call lock mirrors
# the rapidocr_parser belt-and-braces stance — nothing calls this
# concurrently today, but the kernel is thread-safe by construction.
_ENGINE: Any = None
_ENGINE_LOCK = threading.Lock()


def _ensure_engine() -> Any:
    """Lazily build the process-wide rapid-table engine (SLANETPLUS, CPU).

    What: constructs with ``use_ocr=False`` (so NO internal RapidOCR engine
    is built), then flips ``cfg.use_ocr = True`` so the text-matching path
    in ``RapidTable.__call__`` runs against the caller-supplied
    ``ocr_results``.

    Why the flip: with plain ``use_ocr=True`` the constructor builds a full
    internal RapidOCR (det/rec/cls ONNX sessions, PP-OCRv4 defaults) that a
    caller-supplied ``ocr_results`` then never invokes — measured 1.60s
    init + duplicate OCR RAM vs 0.06s without, for byte-identical output
    HTML (live-verified on frozen sample page TQA p.3). The flag is only
    read inside ``__call__``; the version is pinned ==3.0.2 and the e2e
    anchor would catch a behaviour change on upgrade. [DECISION-tt.31]
    """
    global _ENGINE
    with _ENGINE_LOCK:
        if _ENGINE is None:
            from rapid_table import ModelType, RapidTable, RapidTableInput

            engine = RapidTable(RapidTableInput(model_type=ModelType.SLANETPLUS, use_ocr=False))
            engine.cfg.use_ocr = True  # see docstring — [DECISION-tt.31]
            _ENGINE = engine
    return _ENGINE


def page_ocr_results(jpeg_bytes: bytes) -> tuple[Any, Any, Any] | None:
    """Run RapidOCR on one rendered page, returning the raw (boxes, txts, scores).

    Why here and not via RapidOcrParser: the parser's public surface
    returns assembled plain text only — its .txt cache cannot store the
    box geometry that table structuring needs, so the preview lane runs a
    fresh OCR pass (~1s/page, acceptable for a single-page command).
    Engine params are the SAME PP-OCRv5 det/rec pair as
    ``RapidOcrParser._ensure_engine`` so the recognized text matches what
    ingest would have seen for the page. Returns None when the page has no
    recognizable text (blank page → boxes/txts are None).
    """
    from rapidocr import OCRVersion, RapidOCR

    ocr = RapidOCR(
        params={
            "Det.ocr_version": OCRVersion.PPOCRV5,
            "Rec.ocr_version": OCRVersion.PPOCRV5,
        }
    )
    result = ocr(jpeg_bytes)
    if result.boxes is None or result.txts is None or len(result.txts) == 0:
        return None
    return (result.boxes, result.txts, result.scores)


def structure_table(
    jpeg_bytes: bytes,
    ocr_results: tuple[Any, Any, Any] | None,
    *,
    engine: Any = None,
) -> list[TableCell]:
    """Structure one page's table from JPEG bytes + existing OCR results.

    ``ocr_results`` is the raw RapidOCR triple ``(boxes, txts, scores)`` —
    passed straight through to rapid-table, which then SKIPS its internal
    OCR entirely (live-verified: ``get_ocr_results`` returns early), so a
    page already OCR'd by RapidOcrParser is never OCR'd twice.
    [DECISION-tt.30]

    Failure semantics: empty list, never raise — a page where structuring
    fails (no OCR text, engine error, malformed result) must not abort the
    caller's loop, mirroring the per-page stance of the PDF parsers.
    ``engine`` is a test seam; production uses the process-wide singleton.
    """
    # No OCR evidence → no table. format_ocr_results would crash on None
    # boxes, so this guard is correctness, not just an optimisation.
    if ocr_results is None:
        return []
    boxes, txts, _scores = ocr_results
    if boxes is None or txts is None or len(txts) == 0:
        return []

    try:
        eng = engine if engine is not None else _ensure_engine()
        with _ENGINE_LOCK:
            result = eng(jpeg_bytes, ocr_results=[ocr_results])

        if not result.pred_htmls or len(result.cell_bboxes) == 0:
            return []

        # Normalization frame = the rendered page itself. PIL reads the
        # size from the JPEG header without decoding pixels — cheap, and
        # independent of the engine's internal image representation.
        with Image.open(io.BytesIO(jpeg_bytes)) as img:
            page_w, page_h = img.size

        return _assemble_cells(
            result.pred_htmls[0],
            result.cell_bboxes[0],
            result.logic_points[0],
            page_w=page_w,
            page_h=page_h,
        )
    except Exception as exc:  # noqa: BLE001 — per-page failure must not abort the caller
        logger.warning("table_assemble.structure_error", error_type=type(exc).__name__)
        return []


def _assemble_cells(
    pred_html: str,
    cell_bboxes: Any,
    logic_points: Any,
    *,
    page_w: int,
    page_h: int,
) -> list[TableCell]:
    """Zip the engine's three parallel outputs into the TableCell list.

    HTML td order, ``cell_bboxes`` rows and ``logic_points`` rows are
    index-aligned (verified live, see _TdTextExtractor docstring); the
    defensive ``min(...)`` keeps a hypothetical off-by-N drift from raising
    instead of truncating — same stance as the ssTB-R PoC.
    """
    extractor = _TdTextExtractor()
    extractor.feed(pred_html)
    tds = extractor.tds

    n = min(len(tds), len(cell_bboxes), len(logic_points))
    if not (len(tds) == len(cell_bboxes) == len(logic_points)):
        logger.warning(
            "table_assemble.length_mismatch",
            tds=len(tds),
            bboxes=len(cell_bboxes),
            logic=len(logic_points),
        )

    cells: list[TableCell] = []
    for i in range(n):
        row_start, row_end, col_start, col_end = (int(v) for v in logic_points[i])
        x, y, w, h = _normalize_bbox(cell_bboxes[i], page_w=page_w, page_h=page_h)
        # Whitespace inside a cell collapses to single spaces — the matcher
        # concatenates OCR fragments and the raw spacing carries no signal.
        cells.append(
            TableCell(
                row=row_start,
                row_end=row_end,
                col=col_start,
                col_end=col_end,
                x=x,
                y=y,
                w=w,
                h=h,
                text=" ".join(tds[i].split()),
            )
        )
    return cells


def _normalize_bbox(
    bbox8: Sequence[float], *, page_w: int, page_h: int
) -> tuple[float, float, float, float]:
    """4-point pixel polygon → clamped, normalized (x, y, w, h) in [0, 1].

    The polygon reduces to its axis-aligned hull, corners clamp into the
    page frame BEFORE normalizing: SLANet-plus logical bboxes overflow the
    image (max x 1411.6px on a 1241px-wide frozen page, w=1.117 when left
    unclamped) — an out-of-range coordinate would break any downstream
    highlight overlay. [DECISION-tt.22]
    """
    xs = [float(v) for v in bbox8[0::2]]
    ys = [float(v) for v in bbox8[1::2]]
    x1 = min(max(min(xs), 0.0), float(page_w))
    x2 = min(max(max(xs), 0.0), float(page_w))
    y1 = min(max(min(ys), 0.0), float(page_h))
    y2 = min(max(max(ys), 0.0), float(page_h))
    return (
        round(x1 / page_w, _COORD_DECIMALS),
        round(y1 / page_h, _COORD_DECIMALS),
        round((x2 - x1) / page_w, _COORD_DECIMALS),
        round((y2 - y1) / page_h, _COORD_DECIMALS),
    )


def render_markdown(cells: Sequence[TableCell]) -> str:
    """Cell list → markdown table (pure, deterministic). Empty list → "".

    The grid comes from the LOGICAL indices, not from the HTML row tags —
    logic_points is the structure ground truth and md is a derived view
    [DECISION-tt.21]. A spanning cell renders its text at its start
    (row, col) anchor; the spanned-over positions stay blank (markdown has
    no colspan — perfect span reproduction is out of scope, FORESHADOW
    §13). Two cells claiming the same anchor concatenate in input order,
    so the renderer is total: any cell list produces a valid table.
    """
    if not cells:
        return ""

    n_rows = max(c.row_end for c in cells) + 1
    n_cols = max(c.col_end for c in cells) + 1
    grid: list[list[str]] = [["" for _ in range(n_cols)] for _ in range(n_rows)]
    for cell in cells:
        text = _escape_md_cell(cell.text)
        existing = grid[cell.row][cell.col]
        grid[cell.row][cell.col] = f"{existing} {text}".strip() if existing else text

    # First grid row is the header — these scans' tables carry their column
    # captions in row 0, and markdown requires a header row anyway.
    lines = ["| " + " | ".join(grid[0]) + " |", "|" + "---|" * n_cols]
    lines += ["| " + " | ".join(row) + " |" for row in grid[1:]]
    return "\n".join(lines)


def _escape_md_cell(text: str) -> str:
    """Escape the one character that breaks a markdown table row."""
    return text.replace("|", "\\|")


def render_elements(cells: Sequence[TableCell], *, rotation: int = ROTATION) -> str:
    """Cell list → JSONL elements view (pure, deterministic).

    Line 1 is the page-level metadata record (rotation per DECISION-tt.23:
    the orientation correction the caller applied to the frame BEFORE
    structuring — cell geometry is normalized to that corrected frame —
    0 when the frame was used as rendered; plus the cell count); every
    following line is one TableCell in dataclass field order. JSONL so
    downstream consumers (citation highlighting, FORESHADOW-tt.1 chunker
    wiring) can stream it without a parser dependency.
    """
    meta = {"rotation": rotation, "cells": len(cells)}
    lines = [json.dumps(meta, ensure_ascii=False)]
    lines += [json.dumps(asdict(cell), ensure_ascii=False) for cell in cells]
    return "\n".join(lines)
