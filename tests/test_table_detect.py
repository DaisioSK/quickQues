"""Unit tests for table-region detection + the `table-preview --detect` path (ssTable).

Strategy (mirrors test_table_assemble.py):
- NO real layout model: ``detect_table_regions`` takes an injected fake
  engine whose result mimics the rapid-layout output shape
  (.boxes / .class_names / .scores).
- ``filter_ocr_to_regions`` is pure — exercised directly on raw OCR triples.
- NO real rendering/OCR/detection in the CLI test: the pdfium render entry
  point, page_ocr_results, structure_table, and the two detection helpers
  are monkeypatched at their import locations (the command imports them
  lazily at call time).
- Covered: table-class + confidence filtering, normalization/clamping,
  score sort, detection-failure -> empty (never raise), centroid membership
  (keeps in-region boxes, drops out-of-region prose), empty-region -> None,
  pure-table page (all boxes kept), and the CLI --detect opt-in (default
  off = whole-page, on = region-filtered) + summary line.
"""

from __future__ import annotations

import io
import types
from collections.abc import Callable
from typing import Any

import numpy as np
import pytest
from PIL import Image
from typer.testing import CliRunner

from jcontract.cli import app
from jcontract.impls._table_detect import (
    TableRegion,
    detect_table_regions,
    filter_ocr_to_regions,
)

runner = CliRunner()

PAGE_W, PAGE_H = 100, 200


def _jpeg(width: int = PAGE_W, height: int = PAGE_H) -> bytes:
    """A real decodable JPEG so detect_table_regions can read the frame size."""
    buf = io.BytesIO()
    Image.new("RGB", (width, height), color=(255, 255, 255)).save(buf, format="JPEG")
    return buf.getvalue()


def _layout_result(
    boxes: list[list[float]],
    class_names: list[str],
    scores: list[float],
) -> types.SimpleNamespace:
    """Mimic the rapid-layout output for one image."""
    return types.SimpleNamespace(
        boxes=np.asarray(boxes, dtype=float) if boxes else np.asarray([]),
        class_names=class_names,
        scores=scores,
    )


def _engine(result: types.SimpleNamespace) -> Callable[[Any], types.SimpleNamespace]:
    return lambda _img: result


# ---------------------------------------------------------------------------
# detect_table_regions
# ---------------------------------------------------------------------------


def test_detect_keeps_only_table_class():
    # one "table", one "text" — only the table survives.
    result = _layout_result(
        boxes=[[10, 40, 90, 180], [10, 5, 90, 30]],
        class_names=["table", "text"],
        scores=[0.9, 0.8],
    )
    regions = detect_table_regions(_jpeg(), engine=_engine(result))
    assert len(regions) == 1
    r = regions[0]
    # normalized + clamped against the 100x200 frame.
    assert (r.x, r.y, r.w, r.h) == (0.1, 0.2, 0.8, 0.7)
    assert r.score == 0.9


def test_detect_drops_low_confidence():
    result = _layout_result([[10, 40, 90, 180]], ["table"], [0.3])
    assert detect_table_regions(_jpeg(), engine=_engine(result)) == []
    # raising the bar lets it through when we lower the threshold.
    kept = detect_table_regions(_jpeg(), engine=_engine(result), conf_thresh=0.2)
    assert len(kept) == 1


def test_detect_sorts_by_score_desc():
    result = _layout_result(
        boxes=[[0, 0, 50, 100], [0, 100, 50, 200]],
        class_names=["table", "table"],
        scores=[0.6, 0.95],
    )
    regions = detect_table_regions(_jpeg(), engine=_engine(result))
    assert [r.score for r in regions] == [0.95, 0.6]


def test_detect_clamps_overflowing_box():
    # box overflows the frame on both axes — clamps into [0,1].
    result = _layout_result([[-20, -10, 130, 260]], ["table"], [0.9])
    (r,) = detect_table_regions(_jpeg(), engine=_engine(result))
    assert (r.x, r.y, r.w, r.h) == (0.0, 0.0, 1.0, 1.0)


def test_detect_no_table_returns_empty():
    result = _layout_result([[10, 5, 90, 30]], ["text"], [0.9])
    assert detect_table_regions(_jpeg(), engine=_engine(result)) == []


def test_detect_empty_result_returns_empty():
    result = _layout_result([], [], [])
    assert detect_table_regions(_jpeg(), engine=_engine(result)) == []


def test_detect_engine_failure_returns_empty_never_raises():
    def broken(_img: Any) -> Any:
        raise RuntimeError("onnx exploded")

    assert detect_table_regions(_jpeg(), engine=broken) == []


# ---------------------------------------------------------------------------
# filter_ocr_to_regions  (pure)
# ---------------------------------------------------------------------------


def _poly(x1: float, y1: float, x2: float, y2: float) -> list[list[float]]:
    return [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]


def test_filter_keeps_in_region_drops_out():
    # region covers y 40..180 (the "table band"). prose box at top is dropped.
    region = TableRegion(x=0.1, y=0.2, w=0.8, h=0.7, score=0.9)  # px y 40..180
    boxes = [
        _poly(20, 10, 80, 25),  # prose, centroid y~17 -> OUT
        _poly(20, 50, 80, 70),  # table row, centroid y~60 -> IN
        _poly(20, 150, 80, 170),  # table row, centroid y~160 -> IN
    ]
    txts = ("letterhead", "row-a", "row-b")
    scores = (0.9, 0.9, 0.9)
    out = filter_ocr_to_regions((np.asarray(boxes), txts, scores), [region], page_w=100, page_h=200)
    assert out is not None
    kept_boxes, kept_txts, kept_scores = out
    assert list(kept_txts) == ["row-a", "row-b"]
    assert len(kept_boxes) == 2
    assert len(kept_scores) == 2


def test_filter_pure_table_keeps_all():
    region = TableRegion(x=0.0, y=0.0, w=1.0, h=1.0, score=0.9)
    boxes = [_poly(10, 10, 90, 30), _poly(10, 100, 90, 120)]
    triple = (np.asarray(boxes), ("a", "b"), (0.9, 0.9))
    out = filter_ocr_to_regions(triple, [region], page_w=100, page_h=200)
    assert out is not None
    assert list(out[1]) == ["a", "b"]


def test_filter_no_regions_returns_none():
    triple = (np.asarray([_poly(10, 10, 90, 30)]), ("a",), (0.9,))
    assert filter_ocr_to_regions(triple, [], page_w=100, page_h=200) is None


def test_filter_none_ocr_returns_none():
    region = TableRegion(x=0.0, y=0.0, w=1.0, h=1.0, score=0.9)
    assert filter_ocr_to_regions(None, [region], page_w=100, page_h=200) is None


def test_filter_nothing_in_region_returns_none():
    # box centroid above the region band -> nothing survives -> None.
    region = TableRegion(x=0.1, y=0.5, w=0.8, h=0.4, score=0.9)  # px y 100..180
    boxes = [_poly(20, 10, 80, 25)]
    assert (
        filter_ocr_to_regions((np.asarray(boxes), ("x",), (0.9,)), [region], page_w=100, page_h=200)
        is None
    )


# ---------------------------------------------------------------------------
# CLI: table-preview --detect (opt-in)
# ---------------------------------------------------------------------------

# OCR triple = prose box at top + table box in the band.
_OCR = (
    np.asarray([_poly(20, 10, 80, 25), _poly(20, 100, 80, 160)]),
    ("PROSE-letterhead", "TABLE-cell"),
    (0.9, 0.9),
)
_REGION = TableRegion(x=0.1, y=0.4, w=0.8, h=0.5, score=0.9)  # px y 80..180


def _patch_cli(
    monkeypatch: pytest.MonkeyPatch,
    *,
    regions: list[TableRegion],
    capture: dict[str, Any],
) -> None:
    monkeypatch.setattr(
        "jcontract.impls._pdfium_render.render_pdf_page_jpeg",
        lambda pdf_path, page, *, dpi, jpeg_quality: _jpeg(),
    )
    monkeypatch.setattr("jcontract.impls._table_assemble.page_ocr_results", lambda jpeg: _OCR)

    # structure_table records the OCR triple it was handed, returns one cell
    # echoing how many boxes it saw, so the test can prove what was fed in.
    def fake_structure(jpeg: bytes, ocr_results: Any) -> list[Any]:
        from jcontract.impls._table_assemble import TableCell

        n = 0 if ocr_results is None else len(ocr_results[1])
        capture["n_boxes_fed"] = n
        capture["txts_fed"] = list(ocr_results[1]) if ocr_results is not None else []
        return [TableCell(0, 0, 0, 0, 0.0, 0.0, 1.0, 1.0, f"fed={n}")]

    monkeypatch.setattr("jcontract.impls._table_assemble.structure_table", fake_structure)
    monkeypatch.setattr(
        "jcontract.impls._table_detect.detect_table_regions",
        lambda jpeg: regions,
    )


def test_cli_default_off_feeds_whole_page(tmp_path, monkeypatch):
    capture: dict[str, Any] = {}
    _patch_cli(monkeypatch, regions=[_REGION], capture=capture)
    pdf = tmp_path / "doc.pdf"
    pdf.write_bytes(b"%PDF-fake")
    result = runner.invoke(app, ["table-preview", str(pdf), "--page", "1"])
    assert result.exit_code == 0, result.output
    # no --detect: both boxes fed (legacy whole-page), no detect note.
    assert capture["n_boxes_fed"] == 2
    assert "PROSE-letterhead" in capture["txts_fed"]
    assert "[detect:" not in result.output


def test_cli_detect_filters_to_region(tmp_path, monkeypatch):
    capture: dict[str, Any] = {}
    _patch_cli(monkeypatch, regions=[_REGION], capture=capture)
    pdf = tmp_path / "doc.pdf"
    pdf.write_bytes(b"%PDF-fake")
    result = runner.invoke(app, ["table-preview", str(pdf), "--page", "1", "--detect"])
    assert result.exit_code == 0, result.output
    # --detect: only the in-region table box fed; prose dropped.
    assert capture["n_boxes_fed"] == 1
    assert capture["txts_fed"] == ["TABLE-cell"]
    assert "[detect: 1 table region(s)]" in result.output


def test_cli_detect_no_region_falls_back_to_whole_page(tmp_path, monkeypatch):
    capture: dict[str, Any] = {}
    _patch_cli(monkeypatch, regions=[], capture=capture)
    pdf = tmp_path / "doc.pdf"
    pdf.write_bytes(b"%PDF-fake")
    result = runner.invoke(app, ["table-preview", str(pdf), "--page", "1", "--detect"])
    assert result.exit_code == 0, result.output
    # detector found nothing -> structure the whole page rather than nothing.
    assert capture["n_boxes_fed"] == 2
    assert "[detect: 0 table region(s)]" in result.output
