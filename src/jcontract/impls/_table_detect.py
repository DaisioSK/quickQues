"""Table-region detection — localize tables on a page before SLANet structuring.

What:
    ``detect_table_regions`` runs a document-layout model (rapid-layout,
    PP-LAYOUT-TABLE ONNX on CPU) over one rendered page (JPEG bytes) and
    returns the normalized bounding boxes of the regions classified as
    tables. ``filter_ocr_to_regions`` is a pure helper that keeps only the
    OCR boxes whose centroid falls inside any detected region — the subset
    that then feeds ``structure_table``.

Why:
    SLANet-plus is the structure engine (DECISION-tt.20, reused — DECISION-pm.4),
    but feeding it the WHOLE page makes it absorb page prose (letterhead,
    headings) into the first row's cells (recorded behaviour in
    _table_assemble). The missing piece is upstream localization: on the
    frozen mixed sample (TQA p.3) the 10 letterhead/heading/footer OCR boxes
    sit OUTSIDE the detected table band (y 0.24-0.72) while the 53 table
    boxes sit inside — so filtering to the region removes exactly the prose
    that corrupted the first row. [DECISION-pm.30]

    Detector choice — rapid-layout 1.2.1, PP-LAYOUT-TABLE model: same RapidAI
    ONNX-CPU family as the rapidocr/rapid-table we already ship (Apache-2.0,
    6.7MB pure-python wheel + one-time ~7.5MB onnx download, every transitive
    dep already present except a numpy 1->2 bump). Live-verified table-region
    recall 1.0 on both frozen samples (mixed TQA p.3, rotated dense #4 p.273).
    Rejected: PP-DocLayoutV3 (heavier, multi-class — its extra classes only
    confirmed the boundary, not needed for table-only localization),
    DocLayout-YOLO (also bundled in rapid-layout, comparable recall but the
    single-class PP-LAYOUT-TABLE is the cleanest fit). [DECISION-pm.30]

Context:
    Mechanism only, opt-in — nothing here changes default ``table-preview``
    behaviour (DECISION-pm.31). Not wired into the chunker or ingest
    (FORESHADOW-pm.2/tt.1). The process-wide engine singleton mirrors
    _table_assemble's _ensure_engine stance: one ONNX session per process.
"""

from __future__ import annotations

import io
import threading
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import structlog
from PIL import Image

logger = structlog.get_logger(__name__)

# rapid-layout class name for a table region. The PP-LAYOUT-TABLE model is
# table-specialized (single "table" class), but the multi-class doc-layout
# models in the same package also emit "table" — matching by class name
# keeps detect_table_regions correct if the model is ever swapped.
_TABLE_CLASS = "table"

# Minimum detection confidence to accept a region. PP-LAYOUT-TABLE scored
# 0.87 / 0.84 on the two frozen samples; 0.5 keeps a comfortable margin
# below that while rejecting spurious low-confidence boxes. [DECISION-pm.31]
_DEFAULT_CONF_THRESH = 0.5


@dataclass(frozen=True)
class TableRegion:
    """One detected table region: normalized (x, y, w, h) in [0, 1] + score.

    Geometry is relative to the rendered page frame and clamped into it,
    mirroring TableCell's normalized convention so the two views compose.
    """

    x: float
    y: float
    w: float
    h: float
    score: float


# Process-wide engine singleton: first construction loads the layout ONNX
# session (and may trigger the one-time model download), so it must happen
# once per process. Lock mirrors _table_assemble / rapidocr_parser.
_ENGINE: Any = None
_ENGINE_LOCK = threading.Lock()


def _ensure_engine() -> Any:
    """Lazily build the process-wide rapid-layout engine (PP-LAYOUT-TABLE, CPU).

    Constructed with the default onnxruntime engine — the same CPU
    onnxruntime the rest of the stack uses. [DECISION-pm.30]
    """
    global _ENGINE
    with _ENGINE_LOCK:
        if _ENGINE is None:
            from rapid_layout import ModelType, RapidLayout, RapidLayoutInput

            _ENGINE = RapidLayout(RapidLayoutInput(model_type=ModelType.PP_LAYOUT_TABLE))
    return _ENGINE


def detect_table_regions(
    jpeg_bytes: bytes,
    *,
    engine: Any = None,
    conf_thresh: float = _DEFAULT_CONF_THRESH,
) -> list[TableRegion]:
    """Detect table regions on one rendered page, returned newest-confident first.

    Returns normalized table bounding boxes (sorted by descending score). A
    page with no detected table yields an empty list — never raises, so a
    detection failure degrades to "no region" (the caller then has the
    choice to fall back to whole-page structuring) rather than aborting.
    ``engine`` is a test seam; production uses the process-wide singleton.
    """
    try:
        with Image.open(io.BytesIO(jpeg_bytes)) as img:
            page_w, page_h = img.size
            # rapid-layout takes an HWC BGR ndarray (cv2 convention); decode
            # via PIL (already a dep) and flip RGB->BGR, no cv2 import needed.
            rgb = np.asarray(img.convert("RGB"))
        bgr = rgb[:, :, ::-1]

        eng = engine if engine is not None else _ensure_engine()
        with _ENGINE_LOCK:
            result = eng(bgr)
    except Exception as exc:  # noqa: BLE001 — detection failure must not abort the caller
        logger.warning("table_detect.detect_error", error_type=type(exc).__name__)
        return []

    boxes = getattr(result, "boxes", None)
    class_names = getattr(result, "class_names", None)
    scores = getattr(result, "scores", None)
    if boxes is None or class_names is None or scores is None or len(boxes) == 0:
        return []

    regions: list[TableRegion] = []
    for box, cls, score in zip(boxes, class_names, scores, strict=False):
        if str(cls) != _TABLE_CLASS or float(score) < conf_thresh:
            continue
        regions.append(_normalize_region(box, score=float(score), page_w=page_w, page_h=page_h))
    regions.sort(key=lambda r: r.score, reverse=True)
    return regions


def _normalize_region(
    box: Sequence[float], *, score: float, page_w: int, page_h: int
) -> TableRegion:
    """rapid-layout (x1, y1, x2, y2) pixel box -> clamped normalized TableRegion."""
    x1, y1, x2, y2 = (float(v) for v in box)
    x1 = min(max(x1, 0.0), float(page_w))
    x2 = min(max(x2, 0.0), float(page_w))
    y1 = min(max(y1, 0.0), float(page_h))
    y2 = min(max(y2, 0.0), float(page_h))
    return TableRegion(
        x=round(x1 / page_w, 4),
        y=round(y1 / page_h, 4),
        w=round((x2 - x1) / page_w, 4),
        h=round((y2 - y1) / page_h, 4),
        score=round(score, 4),
    )


def filter_ocr_to_regions(
    ocr_results: tuple[Any, Any, Any] | None,
    regions: Sequence[TableRegion],
    *,
    page_w: int,
    page_h: int,
) -> tuple[Any, Any, Any] | None:
    """Keep only the OCR boxes whose centroid falls inside any table region.

    Pure given its inputs. ``ocr_results`` is the raw RapidOCR triple
    ``(boxes, txts, scores)`` (boxes = N x 4 x 2 pixel polygons). Returns a
    triple of the same shape holding the kept subset, or None when nothing
    survives (so structure_table — which treats None / empty as "no table" —
    short-circuits cleanly). With no regions, returns None: an empty region
    set means "detector found no table", not "keep everything".

    Membership is centroid-in-any-region: a box belongs to the table if its
    center sits inside a detected band. This is what removed the 10
    letterhead/heading/footer boxes on the frozen sample while keeping all
    53 table boxes. [DECISION-pm.30]
    """
    if ocr_results is None or not regions:
        return None
    boxes, txts, scores = ocr_results
    if boxes is None or txts is None or len(txts) == 0:
        return None

    # Denormalize region bounds to pixels once.
    bounds = [
        (r.x * page_w, r.y * page_h, (r.x + r.w) * page_w, (r.y + r.h) * page_h) for r in regions
    ]

    keep_idx: list[int] = []
    for i, poly in enumerate(boxes):
        xs = [float(p[0]) for p in poly]
        ys = [float(p[1]) for p in poly]
        cx = (min(xs) + max(xs)) / 2
        cy = (min(ys) + max(ys)) / 2
        if any(rx1 <= cx <= rx2 and ry1 <= cy <= ry2 for rx1, ry1, rx2, ry2 in bounds):
            keep_idx.append(i)

    if not keep_idx:
        return None

    kept_boxes = np.asarray([boxes[i] for i in keep_idx])
    kept_txts = [txts[i] for i in keep_idx]
    kept_scores = np.asarray([scores[i] for i in keep_idx])
    return (kept_boxes, kept_txts, kept_scores)
