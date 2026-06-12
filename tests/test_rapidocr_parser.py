"""Unit tests for RapidOcrParser (E3, sub-sprint ssLC).

Strategy:
- All tests MOCK the rapidocr engine (a callable returning an object with
  .boxes / .txts / .scores — mirrored from the real RapidOCROutput shape,
  verified live 2026-06-11). No model download, no onnxruntime inference.
- The pypdfium2 render path IS exercised on the real synthetic fixture PDF
  (pure-local code), same as the other vision-parser test modules.
- Reading-order assembly gets direct unit tests on _assemble_reading_order
  — that's this vendor's only novel logic; cache/error handling mirrors the
  established vendor pattern and is re-asserted here so the rapidocr-
  prefixed cache layout can't silently regress.
"""

from __future__ import annotations

import json
import types
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from jcontract.impls.rapidocr_parser import RapidOcrParser, _assemble_reading_order

SYNTHETIC_PDF = Path("eval/fixtures/synthetic_contract_tqa.pdf")


def _box(x0: float, y0: float, x1: float, y1: float) -> list[list[float]]:
    """4-point quad (clockwise from top-left) — the shape rapidocr returns."""
    return [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]


def _make_result(
    boxes: list[list[list[float]]] | None, txts: tuple[str, ...] | None
) -> types.SimpleNamespace:
    """Fake RapidOCROutput: only the attrs the parser reads."""
    scores = None if txts is None else tuple(0.99 for _ in txts)
    return types.SimpleNamespace(boxes=boxes, txts=txts, scores=scores)


def _make_engine(boxes: list[list[list[float]]] | None, txts: tuple[str, ...] | None) -> MagicMock:
    engine = MagicMock()
    engine.return_value = _make_result(boxes, txts)
    return engine


# ---------------------------------------------------------------------------
# Reading-order assembly
# ---------------------------------------------------------------------------


def test_reading_order_sorts_top_to_bottom():
    boxes = [_box(10, 300, 200, 330), _box(10, 100, 200, 130), _box(10, 200, 200, 230)]
    txts = ("third", "first", "second")
    assert _assemble_reading_order(boxes, txts) == "first\nsecond\nthird"


def test_same_visual_line_sorts_left_to_right():
    """Boxes on one row with few-pixel y jitter must join one line, x-sorted."""
    boxes = [
        _box(400, 102, 500, 132),  # right block, slightly higher
        _box(10, 105, 100, 135),  # left block
        _box(200, 100, 300, 130),  # middle block
    ]
    txts = ("right", "left", "mid")
    assert _assemble_reading_order(boxes, txts) == "left mid right"


def test_line_banding_splits_on_clear_vertical_gap():
    """Two rows separated by more than half a box height stay separate lines."""
    boxes = [_box(10, 100, 100, 130), _box(200, 100, 300, 130), _box(10, 160, 100, 190)]
    txts = ("a", "b", "c")
    assert _assemble_reading_order(boxes, txts) == "a b\nc"


def test_empty_inputs_give_empty_text():
    assert _assemble_reading_order([], ()) == ""


# ---------------------------------------------------------------------------
# Parser behaviour on the synthetic fixture (engine mocked)
# ---------------------------------------------------------------------------


def test_renders_synthetic_pdf_and_assembles_text(tmp_path):
    engine = _make_engine([_box(10, 10, 100, 40)], ("hello clause",))
    parser = RapidOcrParser(cache_dir=tmp_path / "cache", engine=engine, max_pages=1)

    pages = parser.parse(SYNTHETIC_PDF)

    assert len(pages) == 1
    assert pages[0].page_num == 1
    assert pages[0].text == "hello clause"
    assert engine.call_count == 1
    # Engine receives raw JPEG bytes (rapidocr decodes internally).
    (jpeg_arg,) = engine.call_args.args
    assert isinstance(jpeg_arg, bytes)
    assert jpeg_arg[:2] == b"\xff\xd8"  # JPEG SOI marker


def test_max_pages_bounds_processing(tmp_path):
    engine = _make_engine([_box(10, 10, 100, 40)], ("page text",))
    parser = RapidOcrParser(cache_dir=tmp_path / "cache", engine=engine, max_pages=2)

    pages = parser.parse(SYNTHETIC_PDF)

    assert [p.page_num for p in pages] == [1, 2]
    assert engine.call_count == 2


def test_caches_results_under_rapidocr_prefix(tmp_path):
    """Second parse = pure cache hit, zero engine calls; filename is namespaced.

    The `rapidocr-<hash>.text.txt` layout must never collide with the Claude
    (`<hash>.text*.txt`) or DeepSeek (`deepseek-v4-<hash>*`) entries sharing
    data/ocr_cache/.
    """
    cache_dir = tmp_path / "cache"
    engine_1 = _make_engine([_box(10, 10, 100, 40)], ("cached text",))
    RapidOcrParser(cache_dir=cache_dir, engine=engine_1, max_pages=1).parse(SYNTHETIC_PDF)
    assert engine_1.call_count == 1

    cache_files = list(cache_dir.glob("rapidocr-*.text.txt"))
    assert len(cache_files) == 1, f"expected one prefixed cache file, got {cache_files}"

    engine_2 = _make_engine([_box(10, 10, 100, 40)], ("SHOULD NOT BE RETURNED",))
    pages = RapidOcrParser(cache_dir=cache_dir, engine=engine_2, max_pages=1).parse(SYNTHETIC_PDF)

    assert pages[0].text == "cached text"
    assert engine_2.call_count == 0  # cache hit


def test_non_default_model_type_gets_own_cache_namespace(tmp_path):
    """model_type='server' must append .ppocrv5-server — no cross-model reuse."""
    cache_dir = tmp_path / "cache"
    engine = _make_engine([_box(10, 10, 100, 40)], ("server text",))
    parser = RapidOcrParser(cache_dir=cache_dir, engine=engine, max_pages=1, model_type="server")
    parser.parse(SYNTHETIC_PDF)

    cache_files = list(cache_dir.glob("rapidocr-*.text.ppocrv5-server.txt"))
    assert len(cache_files) == 1, f"expected server-suffixed cache file, got {cache_files}"


def test_blank_page_normalised_to_empty_and_cached(tmp_path):
    """RapidOCR returns txts=None on blank pages — store '' and cache it."""
    cache_dir = tmp_path / "cache"
    engine = _make_engine(None, None)
    pages = RapidOcrParser(cache_dir=cache_dir, engine=engine, max_pages=1).parse(SYNTHETIC_PDF)

    assert pages[0].text == ""
    # Blank verdict is cached — a re-ingest must not re-OCR the page.
    cache_files = list(cache_dir.glob("rapidocr-*.txt"))
    assert len(cache_files) == 1
    assert cache_files[0].read_text(encoding="utf-8") == ""


def test_engine_error_returns_empty_and_does_not_cache(tmp_path):
    """Per-page engine failures must not abort the batch NOR poison the cache."""
    cache_dir = tmp_path / "cache"
    engine = MagicMock(side_effect=RuntimeError("simulated engine failure"))
    pages = RapidOcrParser(cache_dir=cache_dir, engine=engine, max_pages=2).parse(SYNTHETIC_PDF)

    assert len(pages) == 2
    assert all(p.text == "" for p in pages)
    # Transient failure → no cache file → next ingest retries.
    assert list(cache_dir.glob("rapidocr-*.txt")) == []


def test_file_not_found_raises(tmp_path):
    """File-level errors (not extraction-quality) must raise loudly."""
    parser = RapidOcrParser(cache_dir=tmp_path / "cache", engine=MagicMock())
    with pytest.raises(FileNotFoundError):
        parser.parse(Path("does/not/exist.pdf"))


# ---------------------------------------------------------------------------
# ssRT auto-rotate: opt-in probe, rotation sidecar cache, rotation field
# ---------------------------------------------------------------------------


def _scored_result(
    boxes: list[list[list[float]]], txts: tuple[str, ...], scores: tuple[float, ...]
) -> types.SimpleNamespace:
    """Fake RapidOCROutput with EXPLICIT scores (the probe gates on min)."""
    return types.SimpleNamespace(boxes=boxes, txts=txts, scores=scores)


def _page1_frames() -> dict[int, bytes]:
    """The exact frames the parser will probe for fixture page 1.

    render_pdf_page_jpeg produces byte-identical output to the parser's
    internal render_page_jpeg (asserted by tests/test_pdfium_render.py),
    so keying the fake engine by these bytes pins the whole probe path.
    """
    from jcontract.impls._page_orient import ROTATIONS, rotate_jpeg
    from jcontract.impls._pdfium_render import render_pdf_page_jpeg
    from jcontract.impls.rapidocr_parser import DEFAULT_DPI, DEFAULT_JPEG_QUALITY

    base = render_pdf_page_jpeg(
        SYNTHETIC_PDF, 1, dpi=DEFAULT_DPI, jpeg_quality=DEFAULT_JPEG_QUALITY
    )
    return {rot: rotate_jpeg(base, rot, jpeg_quality=DEFAULT_JPEG_QUALITY) for rot in ROTATIONS}


def _frame_keyed_engine(table: dict[bytes, types.SimpleNamespace]) -> MagicMock:
    return MagicMock(side_effect=lambda jpeg_bytes: table[jpeg_bytes])


def test_default_parse_never_probes_and_rotation_is_zero(tmp_path):
    """auto_rotate off (default): zero new files, zero extra engine calls."""
    cache_dir = tmp_path / "cache"
    engine = _make_engine([_box(10, 10, 100, 40)], ("plain text",))
    pages = RapidOcrParser(cache_dir=cache_dir, engine=engine, max_pages=1).parse(SYNTHETIC_PDF)

    assert pages[0].rotation == 0
    assert engine.call_count == 1
    assert list(cache_dir.glob("*.rotation*.json")) == []


def test_auto_rotate_good_page_probes_once_and_caches_decision(tmp_path):
    """Gate-passing page: one engine run total (the probe IS the page OCR —
    text + metrics land in the normal cache, _ocr_jpeg then cache-hits)."""
    cache_dir = tmp_path / "cache"
    frames = _page1_frames()
    engine = _frame_keyed_engine(
        {frames[0]: _scored_result([_box(10, 10, 100, 40)], ("good text",), (0.99,))}
    )
    parser = RapidOcrParser(cache_dir=cache_dir, engine=engine, max_pages=1, auto_rotate=True)

    pages = parser.parse(SYNTHETIC_PDF)

    assert pages[0].rotation == 0
    assert pages[0].text == "good text"
    assert engine.call_count == 1  # gate passed → no 4x probing
    sidecars = list(cache_dir.glob("rapidocr-*.rotation.json"))
    assert len(sidecars) == 1
    decision = json.loads(sidecars[0].read_text(encoding="utf-8"))
    assert decision["rotation"] == 0
    assert decision["gated"] is False


def test_auto_rotate_low_quality_page_picks_winning_rotation(tmp_path):
    """Gated page: four probes, the high-mass direction wins, the page's
    text comes from the WINNING frame, and ParsedPage.rotation records it."""
    cache_dir = tmp_path / "cache"
    frames = _page1_frames()
    engine = _frame_keyed_engine(
        {
            # rotation 0: low min_score (0.60 < 0.756 gate) + little text.
            frames[0]: _scored_result(
                [_box(10, 10, 100, 40), _box(10, 60, 100, 90)],
                ("frag", "ment"),
                (0.95, 0.60),
            ),
            # rotation 90: long confident text — the clear winner.
            frames[90]: _scored_result(
                [_box(10, 10, 400, 40), _box(10, 60, 400, 90)],
                ("this is the recovered readable line", "and a second line"),
                (0.98, 0.97),
            ),
            frames[180]: _scored_result([_box(10, 10, 50, 40)], ("junk",), (0.50,)),
            frames[270]: _scored_result([_box(10, 10, 50, 40)], ("junk",), (0.50,)),
        }
    )
    parser = RapidOcrParser(cache_dir=cache_dir, engine=engine, max_pages=1, auto_rotate=True)

    pages = parser.parse(SYNTHETIC_PDF)

    assert pages[0].rotation == 90
    assert pages[0].text == "this is the recovered readable line\nand a second line"
    # 4 probe runs, NO 5th run: the winning frame's probe already cached it.
    assert engine.call_count == 4
    sidecars = list(cache_dir.glob("rapidocr-*.rotation.json"))
    assert len(sidecars) == 1
    decision = json.loads(sidecars[0].read_text(encoding="utf-8"))
    assert decision["rotation"] == 90
    assert decision["gated"] is True
    assert set(decision["probes"]) == {"0", "90", "180", "270"}


def test_auto_rotate_second_parse_reuses_cached_decision(tmp_path):
    """Re-ingest must not re-probe: rotation sidecar + winner's text cache
    make the second parse engine-free with the identical result."""
    cache_dir = tmp_path / "cache"
    frames = _page1_frames()
    table = {
        frames[0]: _scored_result([_box(10, 10, 100, 40)], ("frag",), (0.60,)),
        frames[90]: _scored_result(
            [_box(10, 10, 400, 40)], ("recovered readable line of text",), (0.98,)
        ),
        frames[180]: _scored_result([_box(10, 10, 50, 40)], ("j",), (0.50,)),
        frames[270]: _scored_result([_box(10, 10, 50, 40)], ("j",), (0.50,)),
    }
    first = RapidOcrParser(
        cache_dir=cache_dir, engine=_frame_keyed_engine(table), max_pages=1, auto_rotate=True
    ).parse(SYNTHETIC_PDF)

    engine_2 = MagicMock(side_effect=AssertionError("second parse must not OCR"))
    second = RapidOcrParser(
        cache_dir=cache_dir, engine=engine_2, max_pages=1, auto_rotate=True
    ).parse(SYNTHETIC_PDF)

    assert engine_2.call_count == 0
    assert second[0].rotation == first[0].rotation == 90
    assert second[0].text == first[0].text == "recovered readable line of text"


def test_auto_rotate_engine_error_degrades_to_zero_and_does_not_cache(tmp_path):
    """Transient engine failure: rotation 0, NO sidecar (next run retries),
    page text falls back to the normal error stance ('' un-cached)."""
    cache_dir = tmp_path / "cache"
    engine = MagicMock(side_effect=RuntimeError("simulated engine failure"))
    parser = RapidOcrParser(cache_dir=cache_dir, engine=engine, max_pages=1, auto_rotate=True)

    pages = parser.parse(SYNTHETIC_PDF)

    assert pages[0].rotation == 0
    assert pages[0].text == ""
    assert list(cache_dir.glob("*.rotation*.json")) == []
    assert list(cache_dir.glob("rapidocr-*.txt")) == []
