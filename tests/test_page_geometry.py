"""Unit tests for the ssGE page-geometry signals + region-aware assembly.

Strategy (mirrors test_rapidocr_parser.py):
- Pure geometry, no engine, no I/O: boxes are hand-built 4-point quads on a
  synthetic 1000x1400 "page". Layout cases mirror the real failure modes
  the signals must catch — two columns, side-by-side interleave, a large
  in-line gap, plain single column.
- The default-assembly twin (band_lines) is asserted against
  _assemble_reading_order's documented behaviour so the two implementations
  cannot silently drift apart.
"""

from __future__ import annotations

from jcontract.impls._page_geometry import (
    assemble_regions,
    band_lines,
    page_geometry,
    rects_from_boxes,
    regions_lines,
    split_columns,
    split_strips,
)
from jcontract.impls.rapidocr_parser import _assemble_reading_order

PAGE_W, PAGE_H = 1000.0, 1400.0


def _box(x0: float, y0: float, x1: float, y1: float) -> list[list[float]]:
    """4-point quad (clockwise from top-left) — the shape rapidocr returns."""
    return [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]


def _two_column_page() -> tuple[list[list[list[float]]], tuple[str, ...]]:
    """Two side-by-side columns whose lines interleave under y-banding.

    Left column x[100,450], right column x[520,900] — a 70px channel
    (7% of width) separates them; lines of the two columns sit on the
    same visual rows, with row spacing below the strip threshold (1% of
    height = 14px) so the body stays one strip.
    """
    boxes = [
        _box(100, 100, 450, 130),  # L1
        _box(520, 105, 900, 135),  # R1
        _box(100, 140, 450, 170),  # L2
        _box(520, 145, 900, 175),  # R2
    ]
    return boxes, ("L1", "R1", "L2", "R2")


# ---------------------------------------------------------------------------
# Signals
# ---------------------------------------------------------------------------


def test_two_columns_detected():
    boxes, _ = _two_column_page()
    geo = page_geometry(boxes, PAGE_W, PAGE_H)
    assert geo["n_columns"] == 2
    assert geo["geometry_version"] == 1


def test_single_column_page_counts_one():
    boxes = [_box(100, 100, 900, 130), _box(100, 150, 900, 180)]
    geo = page_geometry(boxes, PAGE_W, PAGE_H)
    assert geo["n_columns"] == 1
    # Full-width lines never interleave: both orders agree.
    assert geo["order_divergence"] == 0.0


def test_max_band_gap_measures_in_line_blank():
    # One visual row, two blocks separated by a 300px blank (30% of width).
    boxes = [_box(100, 100, 300, 130), _box(600, 102, 900, 132)]
    geo = page_geometry(boxes, PAGE_W, PAGE_H)
    assert abs(geo["max_band_gap"] - 0.3) < 1e-9


def test_max_band_gap_null_without_two_block_lines():
    # Every line holds a single box — no in-line gap evidence.
    boxes = [_box(100, 100, 900, 130), _box(100, 200, 900, 230)]
    geo = page_geometry(boxes, PAGE_W, PAGE_H)
    assert geo["max_band_gap"] is None


def test_box_coverage_sums_areas():
    # Two 200x100 boxes on a 1000x1400 page → 40000 / 1400000.
    boxes = [_box(0, 0, 200, 100), _box(300, 300, 500, 400)]
    geo = page_geometry(boxes, PAGE_W, PAGE_H)
    assert abs(geo["box_coverage"] - 40000 / 1400000) < 1e-9


def test_order_divergence_positive_on_interleaved_columns():
    boxes, _ = _two_column_page()
    geo = page_geometry(boxes, PAGE_W, PAGE_H)
    # Default order: L1 R1 L2 R2; regions order: L1 L2 R1 R2 — exactly the
    # (R1, L2) pair flips out of 6 pairs.
    assert abs(geo["order_divergence"] - 1 / 6) < 1e-9


def test_empty_page_signals_are_null_or_zero():
    geo = page_geometry([], PAGE_W, PAGE_H)
    assert geo["n_columns"] == 0
    assert geo["max_band_gap"] is None
    assert geo["box_coverage"] == 0.0
    assert geo["order_divergence"] == 0.0


# ---------------------------------------------------------------------------
# Split primitives
# ---------------------------------------------------------------------------


def test_split_columns_orders_left_to_right():
    boxes, _ = _two_column_page()
    rects = rects_from_boxes(boxes)
    cols = split_columns(rects, range(len(rects)), PAGE_W)
    assert cols == [[0, 2], [1, 3]]


def test_split_columns_min_boxes_guard_rejects_degenerate_split():
    # A lone header pair side by side: column-reading it would invert the
    # natural row order, so the guard keeps the group whole.
    rects = rects_from_boxes([_box(100, 100, 300, 130), _box(600, 102, 900, 132)])
    assert split_columns(rects, range(2), PAGE_W, min_boxes=2) == [[0, 1]]
    # Without the guard the raw geometric split is visible (n_columns uses this).
    assert split_columns(rects, range(2), PAGE_W) == [[0], [1]]


def test_full_width_box_suppresses_column_split():
    boxes, _ = _two_column_page()
    boxes.append(_box(100, 50, 900, 80))  # banner spanning the channel
    rects = rects_from_boxes(boxes)
    assert len(split_columns(rects, range(len(rects)), PAGE_W)) == 1


def test_split_strips_cuts_on_empty_y_band():
    # 1% of height = 14px: a 60px blank splits, a 5px jitter does not.
    rects = rects_from_boxes(
        [_box(100, 100, 900, 130), _box(100, 135, 900, 165), _box(100, 225, 900, 255)]
    )
    assert split_strips(rects, range(3), PAGE_H) == [[0, 1], [2]]


# ---------------------------------------------------------------------------
# Region-aware assembly
# ---------------------------------------------------------------------------


def test_regions_assembly_reads_columns_contiguously():
    boxes, txts = _two_column_page()
    assert assemble_regions(boxes, txts, PAGE_W, PAGE_H) == "L1\nL2\nR1\nR2"
    # The default sweep interleaves the same boxes — the bug regions fixes.
    assert _assemble_reading_order(boxes, txts) == "L1 R1\nL2 R2"


def test_regions_assembly_strips_before_columns():
    # Full-width title above a two-column body: the title strip must stay
    # on top and must not destroy the body's column split.
    boxes, txts = _two_column_page()
    boxes.insert(0, _box(100, 20, 900, 50))
    all_txts = ("TITLE", *txts)
    assert assemble_regions(boxes, all_txts, PAGE_W, PAGE_H) == "TITLE\nL1\nL2\nR1\nR2"


def test_regions_assembly_matches_default_on_single_column():
    # No empty bands to cut on → regions degrades to exactly the default.
    boxes = [_box(100, 100, 900, 130), _box(100, 150, 400, 180), _box(450, 152, 900, 182)]
    txts = ("first line", "second", "line")
    assert assemble_regions(boxes, txts, PAGE_W, PAGE_H) == _assemble_reading_order(boxes, txts)


def test_regions_assembly_empty_inputs():
    assert assemble_regions([], (), PAGE_W, PAGE_H) == ""


def test_band_lines_mirrors_default_banding_rule():
    # Same three-box case the frozen default tests use: y jitter within half
    # a box height bands together, larger steps split.
    boxes = [_box(400, 102, 500, 132), _box(10, 105, 100, 135), _box(200, 100, 300, 130)]
    rects = rects_from_boxes(boxes)
    lines = band_lines(rects, range(3))
    assert lines == [[1, 2, 0]]  # left, mid, right — one visual line
    assert _assemble_reading_order(boxes, ("right", "left", "mid")) == "left mid right"


def test_regions_lines_flatten_covers_every_box_once():
    boxes, _ = _two_column_page()
    rects = rects_from_boxes(boxes)
    flat = [i for line in regions_lines(rects, PAGE_W, PAGE_H) for i in line]
    assert sorted(flat) == [0, 1, 2, 3]
