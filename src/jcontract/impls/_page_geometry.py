"""Page-geometry signals + region-aware reading order from OCR boxes (ssGE).

What:
    Two related mechanisms over one rendered page's OCR box list:
    1. ``page_geometry`` — per-page layout signals (``n_columns``,
       ``max_band_gap``, ``box_coverage``, ``order_divergence``) that make
       "the text is there but assembled in the wrong order" *detectable*
       instead of invisible. They ride in the rapidocr metrics sidecar and
       surface through the ``ocr-quality`` flag rules.
    2. ``assemble_regions`` — the opt-in region-aware assembly: split the
       page into horizontal strips (empty y bands), split each strip into
       columns (empty x channels), read columns left-to-right with the same
       line banding the default assembler uses. Fixes side-by-side
       interleave ("并排穿插") that a pure y-band sweep merges into one line.

Why:
    The v4 L5 fidelity comparison showed low-ratio pages whose characters
    are all present but ordered wrong (multi-column/table layouts,
    FORESHADOW-ls.3). Wrong order is a *geometry* property, so the signals
    are computed from box geometry where the engine ran — exactly like the
    score-based ssQA signals. Formulas in [DECISION-pl.20], algorithm
    parameters calibrated live on the two frozen multi-column specimens
    [DECISION-pl.23] (dev-sprint v7 §13).

Context:
    Pure geometry: no engine, no I/O — callers pass the RapidOCR 4-point
    quads plus the rendered frame's pixel size. The default assembly path
    (``rapidocr_parser._assemble_reading_order``) is intentionally NOT
    rewired through this module: it is frozen by the zero-behaviour-change
    mandate, so this module re-derives the same banding rule on indices
    (documented drift risk accepted over touching the frozen path). Boxes
    must come from the *upright* frame — the parser computes geometry after
    the ssRT rotation step, otherwise column/band signals are noise.
    Nested multi-column layouts recurse no further than strip→column
    (FORESHADOW-pl.3).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

# --- Region-split parameters [DECISION-pl.23] -------------------------------
# Calibrated live (2026-06-12) on the frozen multi-column specimens
# (T/DEMO TQA p.2 + consolidated-spec p.50, see dev-sprint v7 §13):
#
# STRIP_GAP_FRAC   minimum empty horizontal band (fraction of page height)
#                  that separates two strips. 0.01 ≈ 17px at 150dpi/A4 —
#                  splits table blocks from prose (24-67px gaps measured)
#                  while paragraph internals (≤16px) stay merged.
STRIP_GAP_FRAC = 0.01
# COLUMN_GAP_FRAC  minimum empty vertical channel (fraction of page width)
#                  that separates two columns within a strip. 0.01 ≈ 12px.
COLUMN_GAP_FRAC = 0.01
# BOX_SHRINK_FRAC  horizontal inset applied per box side (fraction of page
#                  width) before the x projection. OCR detection boxes are
#                  loose: the measured channel between two real table
#                  columns was 5px — invisible to a 12px threshold until
#                  each side retreats ~6px. 0.005 ≈ 6px.
BOX_SHRINK_FRAC = 0.005
# MIN_COLUMN_BOXES a strip only splits when EVERY resulting column keeps at
#                  least this many boxes. Guards against "column-reading" a
#                  pair of header boxes that merely sit side by side
#                  (measured: the guard recovered the header order AND
#                  improved the frozen-ratio on the p.50 specimen).
MIN_COLUMN_BOXES = 2

Rect = tuple[float, float, float, float]  # (x0, y0, x1, y1), pixel space


def rects_from_boxes(boxes: Sequence[Sequence[Sequence[float]]]) -> list[Rect]:
    """Reduce RapidOCR 4-point quads ``[[x,y]*4]`` to axis-aligned rects."""
    rects: list[Rect] = []
    for box in boxes:
        xs = [float(pt[0]) for pt in box]
        ys = [float(pt[1]) for pt in box]
        rects.append((min(xs), min(ys), max(xs), max(ys)))
    return rects


def _merge_intervals(intervals: list[tuple[float, float]], min_gap: float) -> list[list[float]]:
    """Union sorted 1-D intervals, merging neighbours closer than ``min_gap``.

    The surviving gaps are exactly the "empty bands" the strip/column
    splitters cut on: anything narrower than ``min_gap`` is treated as
    covered (jitter, not layout).
    """
    merged: list[list[float]] = []
    for start, end in sorted(intervals):
        if merged and start - merged[-1][1] < min_gap:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return merged


def _split_on_axis(
    rects: Sequence[Rect],
    indices: Sequence[int],
    *,
    axis: int,
    min_gap: float,
    shrink: float = 0.0,
) -> list[list[int]]:
    """Partition ``indices`` into groups separated by empty bands on ``axis``.

    axis 0 = x (columns, left-to-right), axis 1 = y (strips, top-to-bottom).
    ``shrink`` insets each interval end by that many pixels before the
    projection (x only — see BOX_SHRINK_FRAC); a box narrower than the
    double inset degrades to its centre point instead of inverting.
    Every box is assigned to the merged span containing its centre; a
    centre that lands in a gap (possible only through shrink asymmetry)
    falls into the last span rather than being dropped — boxes are never
    lost.
    """
    intervals: list[tuple[float, float]] = []
    for i in indices:
        lo, hi = (rects[i][0], rects[i][2]) if axis == 0 else (rects[i][1], rects[i][3])
        lo, hi = lo + shrink, hi - shrink
        if hi < lo:
            # Box narrower than the double inset: degrade to its centre
            # point on this axis instead of producing an inverted interval.
            centre = (
                (rects[i][0] + rects[i][2]) / 2 if axis == 0 else (rects[i][1] + rects[i][3]) / 2
            )
            lo = hi = centre
        intervals.append((lo, hi))

    spans = _merge_intervals(intervals, min_gap)
    groups: list[list[int]] = [[] for _ in spans]
    for i in indices:
        centre = (rects[i][0] + rects[i][2]) / 2 if axis == 0 else (rects[i][1] + rects[i][3]) / 2
        target = len(spans) - 1
        for span_idx, (start, end) in enumerate(spans):
            if start <= centre <= end:
                target = span_idx
                break
        groups[target].append(i)
    return [g for g in groups if g]


def band_lines(rects: Sequence[Rect], indices: Sequence[int]) -> list[list[int]]:
    """Band ``indices`` into visual lines — index twin of the default rule.

    Same banding as ``rapidocr_parser._assemble_reading_order`` (a box joins
    the current line when its top edge is within half its own height of the
    line anchor; lines sort left-to-right): that path is frozen by the
    zero-default-change mandate and works on (box, txt) pairs, while the
    geometry signals and the regions mode need index orderings — so the rule
    is re-stated here on indices. Tie-break inside a line is (x_left, index)
    where the default uses (x_left, text); they diverge only when two boxes
    share an identical float x_left — not observed on real pages.
    """
    ordered = sorted(indices, key=lambda i: (rects[i][1], rects[i][0], i))
    lines: list[list[int]] = []
    anchor_y: float | None = None
    for i in ordered:
        y_top, height = rects[i][1], max(rects[i][3] - rects[i][1], 1.0)
        if anchor_y is None or (y_top - anchor_y) > height * 0.5:
            lines.append([])
            anchor_y = y_top
        lines[-1].append(i)
    return [sorted(line, key=lambda i: (rects[i][0], i)) for line in lines]


def split_columns(
    rects: Sequence[Rect],
    indices: Sequence[int],
    page_width: float,
    *,
    min_boxes: int = 1,
) -> list[list[int]]:
    """Split ``indices`` into columns on empty x channels, left-to-right.

    ``min_boxes`` > 1 activates the degenerate-column guard: when any
    resulting column would hold fewer boxes, the split is rejected and the
    original group returns whole (regions assembly passes
    MIN_COLUMN_BOXES; the ``n_columns`` signal counts raw geometric
    columns with the default 1). [DECISION-pl.23]
    """
    columns = _split_on_axis(
        rects,
        indices,
        axis=0,
        min_gap=COLUMN_GAP_FRAC * page_width,
        shrink=BOX_SHRINK_FRAC * page_width,
    )
    if len(columns) > 1 and any(len(col) < min_boxes for col in columns):
        return [list(indices)]
    return columns


def split_strips(
    rects: Sequence[Rect], indices: Sequence[int], page_height: float
) -> list[list[int]]:
    """Split ``indices`` into horizontal strips on empty y bands, top-to-bottom."""
    return _split_on_axis(rects, indices, axis=1, min_gap=STRIP_GAP_FRAC * page_height)


def regions_lines(rects: Sequence[Rect], page_width: float, page_height: float) -> list[list[int]]:
    """Region-aware line ordering: strips → columns → banded lines.

    One recursion level by design (strip, then column, then plain banding
    inside the column) — complex nested layouts stay out of scope
    (FORESHADOW-pl.3). A page with no empty bands degrades to exactly the
    default banding, so single-column pages read identically either way.
    """
    all_indices = list(range(len(rects)))
    lines: list[list[int]] = []
    for strip in split_strips(rects, all_indices, page_height):
        for column in split_columns(rects, strip, page_width, min_boxes=MIN_COLUMN_BOXES):
            lines.extend(band_lines(rects, column))
    return lines


def assemble_regions(
    boxes: Sequence[Sequence[Sequence[float]]],
    txts: Sequence[str],
    page_width: float,
    page_height: float,
) -> str:
    """Region-aware twin of ``_assemble_reading_order`` (opt-in lane only).

    Same output shape — blocks joined by a space within a line, lines by a
    newline — but ordered by ``regions_lines``, so side-by-side blocks come
    out column-contiguous instead of interleaved.
    """
    rects = rects_from_boxes(boxes)
    return "\n".join(
        " ".join(str(txts[i]) for i in line)
        for line in regions_lines(rects, page_width, page_height)
    )


def _normalized_kendall_distance(order_a: Sequence[int], order_b: Sequence[int]) -> float:
    """Share of element pairs the two orderings disagree on (0 = identical).

    Normalized Kendall tau distance: discordant pairs / C(n, 2). O(n²) on
    box counts (≤ a few hundred per page) is negligible next to the ~1s
    OCR pass that produced the boxes. [DECISION-pl.20]
    """
    n = len(order_a)
    if n < 2:
        return 0.0
    position_in_b = {value: pos for pos, value in enumerate(order_b)}
    discordant = 0
    for i in range(n):
        for j in range(i + 1, n):
            if position_in_b[order_a[i]] > position_in_b[order_a[j]]:
                discordant += 1
    return discordant / (n * (n - 1) / 2)


def page_geometry(
    boxes: Sequence[Sequence[Sequence[float]]],
    page_width: float,
    page_height: float,
) -> dict[str, Any]:
    """Per-page layout signals from the OCR boxes (sidecar schema, ssGE).

    Signals [DECISION-pl.20]:
      n_columns         geometric column count over the whole page — empty
                        x-channel clustering, no min-box guard (the signal
                        reports raw geometry; the guard is an assembly
                        policy). 0 when the page has no boxes.
      max_band_gap      widest in-line horizontal gap between neighbouring
                        boxes, as a fraction of page width — the direct
                        measurement of "2cm of blank collapses into one
                        space". null when no line holds two boxes.
      box_coverage      total box area / page area (overlaps not deduped —
                        a cheap monotonic text-density proxy; ssVR reuses
                        this). 0.0 when the page has no boxes.
      order_divergence  normalized Kendall distance between the default
                        y-band reading order and the region-aware order —
                        how much the two assemblies *disagree*, i.e. how
                        layout-sensitive this page is. 0.0 when fewer than
                        two boxes.

    Null semantics follow the ssQA sidecar (DECISION-cq.20): a signal with
    no evidence is null, never a fake 0. ``geometry_version`` stamps the
    formula generation so future revisions can tell records apart
    [DECISION-pl.21].
    """
    rects = rects_from_boxes(boxes)
    all_indices = list(range(len(rects)))

    n_columns = len(split_columns(rects, all_indices, page_width)) if rects else 0

    # Widest within-line gap, measured on the DEFAULT banding — the lane
    # whose flattening behaviour the signal is meant to expose.
    default_lines = band_lines(rects, all_indices)
    gaps = [
        max(0.0, rects[right][0] - rects[left][2])
        for line in default_lines
        for left, right in zip(line, line[1:], strict=False)
    ]
    max_band_gap = (max(gaps) / page_width) if gaps else None

    page_area = page_width * page_height
    box_coverage = (
        sum((x1 - x0) * (y1 - y0) for x0, y0, x1, y1 in rects) / page_area if page_area else 0.0
    )

    default_order = [i for line in default_lines for i in line]
    regions_order = [i for line in regions_lines(rects, page_width, page_height) for i in line]
    order_divergence = _normalized_kendall_distance(default_order, regions_order)

    return {
        "geometry_version": 1,
        "n_columns": n_columns,
        "max_band_gap": max_band_gap,
        "box_coverage": box_coverage,
        "order_divergence": order_divergence,
    }
