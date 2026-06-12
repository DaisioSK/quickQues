"""Unit tests for the ssVR needs-vision classifier v2 (classify_page_v2).

Strategy:
- The decision table is exercised with SYNTHETIC pixel fixtures (PIL-drawn
  JPEGs with controlled ink ratios) + mocked box signals, one test per
  branch boundary — the real-page calibration lives in the frozen 14-page
  judgment table (project repo, dev-sprint v7 §13), not here.
- Parser-level tests mock the rapidocr engine (same fixture pattern as
  test_rapidocr_parser) and assert the v2 verdict flows from the metrics
  sidecar into ParsedPage.page_kind, the v1 default stays byte-identical,
  and the JCONTRACT_PAGE_CLASSIFY env resolution behaves.
"""

from __future__ import annotations

import io
import json
import types
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from PIL import Image, ImageDraw

from jcontract.impls._page_classify import _classify_page, classify_page_v2
from jcontract.impls.rapidocr_parser import RapidOcrParser, _resolve_classify_version

SYNTHETIC_PDF = Path("eval/fixtures/synthetic_contract_tqa.pdf")


def _jpeg(image: Image.Image) -> bytes:
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=85)
    return buf.getvalue()


def _page_with_ink(ink_fraction: float, size: tuple[int, int] = (800, 1000)) -> bytes:
    """White page with a black band covering ``ink_fraction`` of its area."""
    img = Image.new("L", size, 255)
    band_height = int(size[1] * ink_fraction)
    if band_height:
        ImageDraw.Draw(img).rectangle((0, 0, size[0], band_height), fill=0)
    return _jpeg(img)


# ---------------------------------------------------------------------------
# classify_page_v2 decision table (mocked box signals, synthetic pixels)
# ---------------------------------------------------------------------------


def test_missing_box_signals_fall_back_to_v1():
    """No sidecar evidence (pre-ssGE cache / boxless vendor) → v1 verdict."""
    blank = _page_with_ink(0.0)
    assert classify_page_v2(blank, boxes=None, box_coverage=None) == _classify_page(blank)
    assert classify_page_v2(blank, boxes=10, box_coverage=None) == _classify_page(blank)
    assert classify_page_v2(blank, boxes=None, box_coverage=0.3) == _classify_page(blank)


def test_zero_boxes_is_drawing():
    """No text at all: ink is purely graphical; blank matches v1's sentinel."""
    assert classify_page_v2(_page_with_ink(0.0), boxes=0, box_coverage=0.0) == "drawing"
    assert classify_page_v2(_page_with_ink(0.3), boxes=0, box_coverage=0.0) == "drawing"


def test_filled_page_is_drawing_regardless_of_boxes():
    """Photo/halftone rule inherited from v1: ink > 0.5 → drawing."""
    filled = _jpeg(Image.new("L", (800, 1000), 0))
    assert classify_page_v2(filled, boxes=40, box_coverage=0.4) == "drawing"


def test_sparse_title_page_is_text():
    """空旷页: almost no coverage AND almost no ink → the words are the page."""
    # ~1% ink ≈ the calibration title pages (dark .003-.008, cover .0074-.021).
    title = _page_with_ink(0.008)
    assert classify_page_v2(title, boxes=3, box_coverage=0.02) == "text"


def test_sparse_drawing_with_real_ink_is_not_carved_out():
    """A sparse spec drawing (low coverage, ink over the bar) must stay drawing.

    Mirrors calibration p.559: cover .030 < 0.10 but dark .042 > 0.02, so the
    sparse rule passes it through to the fragmentation test (51 boxes, mean
    box area .00059 < .001 → drawing).
    """
    drawing = _page_with_ink(0.05)
    assert classify_page_v2(drawing, boxes=51, box_coverage=0.030) == "drawing"


def test_fragmented_boxes_are_drawing():
    """Many small label boxes (mean area < 0.1% of page) → drawing."""
    inked = _page_with_ink(0.05)
    # 100 boxes covering 5% → mean .0005 < .001.
    assert classify_page_v2(inked, boxes=100, box_coverage=0.05) == "drawing"


def test_full_line_boxes_are_text():
    """Dense text page: large line boxes (mean area ≥ 0.1%) → text."""
    inked = _page_with_ink(0.08)
    # 50 boxes covering 35% → mean .007 ≥ .001.
    assert classify_page_v2(inked, boxes=50, box_coverage=0.35) == "text"


def test_fragmentation_boundary_is_exclusive():
    """Exactly at V2_FRAGMENT_BOX_FRAC the page is NOT fragmented → text."""
    inked = _page_with_ink(0.08)
    # 100 boxes covering 10% → mean exactly .001 → not < .001 → text.
    assert classify_page_v2(inked, boxes=100, box_coverage=0.10) == "text"


def test_pixel_decode_error_falls_back_to_v1_then_text():
    """Corrupt frame with valid box signals → v1 fallback → v1's own text default."""
    assert classify_page_v2(b"not a jpeg", boxes=10, box_coverage=0.5) == "text"


# ---------------------------------------------------------------------------
# Version resolution (explicit arg > env > default) [DECISION-pl.31]
# ---------------------------------------------------------------------------


def test_resolve_default_is_v1(monkeypatch):
    monkeypatch.delenv("JCONTRACT_PAGE_CLASSIFY", raising=False)
    assert _resolve_classify_version(None) == "v1"


def test_resolve_env_opts_in(monkeypatch):
    monkeypatch.setenv("JCONTRACT_PAGE_CLASSIFY", "v2")
    assert _resolve_classify_version(None) == "v2"


def test_resolve_explicit_arg_beats_env(monkeypatch):
    monkeypatch.setenv("JCONTRACT_PAGE_CLASSIFY", "v2")
    assert _resolve_classify_version("v1") == "v1"


def test_resolve_unknown_version_raises(monkeypatch):
    monkeypatch.delenv("JCONTRACT_PAGE_CLASSIFY", raising=False)
    with pytest.raises(ValueError, match="page classify version"):
        RapidOcrParser(classify_version="v3")
    monkeypatch.setenv("JCONTRACT_PAGE_CLASSIFY", "bogus")
    with pytest.raises(ValueError, match="page classify version"):
        RapidOcrParser()


# ---------------------------------------------------------------------------
# Parser wiring: sidecar → classify_page_v2 → ParsedPage.page_kind
# ---------------------------------------------------------------------------


def _box(x0: float, y0: float, x1: float, y1: float) -> list[list[float]]:
    return [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]


def _make_engine(boxes: list[list[list[float]]], txts: tuple[str, ...]) -> MagicMock:
    engine = MagicMock()
    engine.return_value = types.SimpleNamespace(
        boxes=boxes, txts=txts, scores=tuple(0.99 for _ in txts)
    )
    return engine


def test_v2_routes_fragmented_page_to_drawing(tmp_path):
    """200 tiny label boxes → fresh sidecar carries the geometry → drawing."""
    # Synthetic page renders ~1240x1754 px at 150 DPI (~2.2M px). 200 boxes of
    # 40x12 px → coverage ≈ .0044, mean box area ≈ .000022 < .001 → drawing.
    boxes = [
        _box(20 + (i % 20) * 60, 40 + (i // 20) * 80, 60 + (i % 20) * 60, 52 + (i // 20) * 80)
        for i in range(200)
    ]
    engine = _make_engine(boxes, tuple(f"d{i}" for i in range(200)))
    parser = RapidOcrParser(
        cache_dir=tmp_path / "cache", engine=engine, max_pages=1, classify_version="v2"
    )

    pages = parser.parse(SYNTHETIC_PDF)

    assert pages[0].page_kind == "drawing"


def test_v2_routes_full_line_boxes_to_text(tmp_path):
    """40 full-width line boxes → large mean box area → text."""
    boxes = [_box(60, 60 + i * 40, 1180, 90 + i * 40) for i in range(40)]
    engine = _make_engine(boxes, tuple(f"line {i}" for i in range(40)))
    parser = RapidOcrParser(
        cache_dir=tmp_path / "cache", engine=engine, max_pages=1, classify_version="v2"
    )

    pages = parser.parse(SYNTHETIC_PDF)

    assert pages[0].page_kind == "text"


def test_v2_pre_geometry_sidecar_falls_back_to_v1(tmp_path):
    """Cache hit on a pre-ssGE sidecar (no box_coverage key) → v1 verdict.

    Backfilling inside ingest is forbidden (DECISION-cq.21) so v2 must defer
    instead of forcing an engine run. [DECISION-pl.33]
    """
    cache_dir = tmp_path / "cache"
    engine_1 = _make_engine([_box(60, 60, 1180, 90)], ("seed text",))
    parser_1 = RapidOcrParser(cache_dir=cache_dir, engine=engine_1, max_pages=1)
    (page_1,) = parser_1.parse(SYNTHETIC_PDF)
    v1_kind = page_1.page_kind

    # Strip the geometry keys from the sidecar to simulate a pre-ssGE record.
    (sidecar,) = cache_dir.glob("*.metrics.json")
    record = json.loads(sidecar.read_text(encoding="utf-8"))
    for key in (
        "geometry_version",
        "n_columns",
        "max_band_gap",
        "box_coverage",
        "order_divergence",
    ):
        record.pop(key, None)
    sidecar.write_text(json.dumps(record), encoding="utf-8")

    engine_2 = _make_engine([], ())
    parser_2 = RapidOcrParser(
        cache_dir=cache_dir, engine=engine_2, max_pages=1, classify_version="v2"
    )
    (page_2,) = parser_2.parse(SYNTHETIC_PDF)

    engine_2.assert_not_called()  # pure cache hit — no ingest-path backfill
    assert page_2.page_kind == v1_kind


def test_default_version_is_v1_and_ignores_sidecar(tmp_path, monkeypatch):
    """Env unset → v1; verdict identical to the shared pixel heuristic."""
    monkeypatch.delenv("JCONTRACT_PAGE_CLASSIFY", raising=False)
    engine = _make_engine([_box(60, 60, 1180, 90)], ("hello",))
    parser = RapidOcrParser(cache_dir=tmp_path / "cache", engine=engine, max_pages=1)

    assert parser._classify_version == "v1"
    (page,) = parser.parse(SYNTHETIC_PDF)
    # The synthetic fixture is a text-dense page; v1 must keep saying text.
    assert page.page_kind == "text"


def test_auto_classify_off_forces_text_even_on_v2(tmp_path):
    boxes = [_box(20 + i * 6, 40, 50 + i * 6, 52) for i in range(100)]
    engine = _make_engine(boxes, tuple(f"d{i}" for i in range(100)))
    parser = RapidOcrParser(
        cache_dir=tmp_path / "cache",
        engine=engine,
        max_pages=1,
        classify_version="v2",
        auto_classify=False,
    )

    (page,) = parser.parse(SYNTHETIC_PDF)

    assert page.page_kind == "text"
