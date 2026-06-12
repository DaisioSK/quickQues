"""Unit tests for the table-structure helper + `table-preview` command (ssTB).

Strategy (mirrors test_ocr_gallery.py):
- NO real structure engine: `structure_table` takes an injected fake engine
  whose result mimics the rapid-table output shape (pred_htmls /
  cell_bboxes / logic_points, index-aligned).
- NO real rendering/OCR in the CLI tests: the pdfium render entry point and
  the helper functions are monkeypatched at module level (the command
  imports them lazily at call time).
- Covered surfaces: cell assembly + td/bbox/logic alignment, bbox clamping
  (DECISION-tt.22), markdown rendering from logical indices including spans
  and pipe escaping (DECISION-tt.21), elements JSONL shape (rotation meta,
  DECISION-tt.23), empty-page/engine-failure fallbacks (return empty, never
  raise), and the CLI md/elements/--out/usage-error paths.
"""

from __future__ import annotations

import io
import json
import types

import pytest
from PIL import Image
from typer.testing import CliRunner

from jcontract.cli import app
from jcontract.impls._table_assemble import (
    TableCell,
    _assemble_cells,
    _normalize_bbox,
    render_elements,
    render_markdown,
    structure_table,
)

runner = CliRunner()


def _fake_jpeg(width: int = 100, height: int = 200) -> bytes:
    """A real decodable JPEG so PIL can read the normalization frame size."""
    buf = io.BytesIO()
    Image.new("RGB", (width, height), color=(255, 255, 255)).save(buf, format="JPEG")
    return buf.getvalue()


def _bbox8(x1: float, y1: float, x2: float, y2: float) -> list[float]:
    """Axis-aligned 4-point polygon in rapid-table's flat 8-float layout."""
    return [x1, y1, x2, y1, x2, y2, x1, y2]


def _engine_result(
    html: str, bboxes: list[list[float]], logic: list[list[int]]
) -> types.SimpleNamespace:
    """Mimic RapidTableOutput for one image."""
    return types.SimpleNamespace(
        pred_htmls=[html],
        cell_bboxes=[bboxes],
        logic_points=[logic],
    )


OCR_TRIPLE = ([[[0, 0], [10, 0], [10, 10], [0, 10]]], ("text",), (0.9,))


# ---------------------------------------------------------------------------
# _normalize_bbox — clamping is mandatory (DECISION-tt.22)
# ---------------------------------------------------------------------------


def test_normalize_bbox_basic():
    assert _normalize_bbox(_bbox8(10, 20, 60, 120), page_w=100, page_h=200) == (
        0.1,
        0.1,
        0.5,
        0.5,
    )


def test_normalize_bbox_clamps_overflow():
    # SLANet-plus logical bboxes overflow the page (w=1.117 measured live);
    # clamped output must stay inside [0, 1].
    x, y, w, h = _normalize_bbox(_bbox8(-5, -10, 140, 250), page_w=100, page_h=200)
    assert (x, y) == (0.0, 0.0)
    assert (w, h) == (1.0, 1.0)


def test_normalize_bbox_rounds_to_4_decimals():
    x, _, w, _ = _normalize_bbox(_bbox8(1, 0, 2, 1), page_w=300, page_h=300)
    assert x == round(1 / 300, 4)
    assert w == round(1 / 300, 4)


# ---------------------------------------------------------------------------
# _assemble_cells — td/bbox/logic zip
# ---------------------------------------------------------------------------


def test_assemble_cells_aligns_three_outputs():
    html = "<html><body><table><tr><td>A</td><td>B  b</td></tr></table></body></html>"
    cells = _assemble_cells(
        html,
        [_bbox8(0, 0, 50, 100), _bbox8(50, 0, 100, 100)],
        [[0, 0, 0, 0], [0, 0, 1, 1]],
        page_w=100,
        page_h=200,
    )
    assert [c.text for c in cells] == ["A", "B b"]  # whitespace collapsed
    assert (cells[0].row, cells[0].col) == (0, 0)
    assert (cells[1].row, cells[1].col) == (0, 1)
    assert cells[0].w == 0.5


def test_assemble_cells_truncates_on_length_mismatch():
    # Hypothetical off-by-N drift between tds and bboxes must truncate,
    # not raise (same stance as the ssTB-R PoC).
    html = "<table><tr><td>A</td><td>B</td><td>C</td></tr></table>"
    cells = _assemble_cells(
        html,
        [_bbox8(0, 0, 50, 100)],
        [[0, 0, 0, 0], [0, 0, 1, 1]],
        page_w=100,
        page_h=200,
    )
    assert len(cells) == 1


# ---------------------------------------------------------------------------
# render_markdown — derived view from logical indices (DECISION-tt.21)
# ---------------------------------------------------------------------------


def _cell(
    row: int, col: int, text: str, row_end: int | None = None, col_end: int | None = None
) -> TableCell:
    return TableCell(
        row=row,
        row_end=row if row_end is None else row_end,
        col=col,
        col_end=col if col_end is None else col_end,
        x=0.0,
        y=0.0,
        w=0.1,
        h=0.1,
        text=text,
    )


def test_render_markdown_grid():
    md = render_markdown([_cell(0, 0, "H1"), _cell(0, 1, "H2"), _cell(1, 0, "a"), _cell(1, 1, "b")])
    assert md.splitlines() == [
        "| H1 | H2 |",
        "|---|---|",
        "| a | b |",
    ]


def test_render_markdown_span_anchors_text_at_start():
    # A colspan cell renders at its (row, col) anchor; spanned positions
    # stay blank (markdown has no colspan).
    md = render_markdown([_cell(0, 0, "wide", col_end=2), _cell(1, 1, "x")])
    assert md.splitlines() == [
        "| wide |  |  |",
        "|---|---|---|",
        "|  | x |  |",
    ]


def test_render_markdown_escapes_pipes():
    md = render_markdown([_cell(0, 0, "a|b")])
    assert "a\\|b" in md


def test_render_markdown_empty_is_empty_string():
    assert render_markdown([]) == ""


# ---------------------------------------------------------------------------
# render_elements — JSONL geometry view (rotation meta per DECISION-tt.23)
# ---------------------------------------------------------------------------


def test_render_elements_meta_line_and_cells():
    lines = render_elements([_cell(0, 0, "中文 | cell")]).splitlines()
    assert json.loads(lines[0]) == {"rotation": 0, "cells": 1}
    record = json.loads(lines[1])
    assert record["text"] == "中文 | cell"
    assert {"row", "row_end", "col", "col_end", "x", "y", "w", "h", "text"} <= set(record)


def test_render_elements_empty_has_meta_only():
    lines = render_elements([]).splitlines()
    assert json.loads(lines[0]) == {"rotation": 0, "cells": 0}
    assert len(lines) == 1


def test_render_elements_records_applied_rotation():
    """ssRT: a caller that structured an auto-rotated frame records the
    correction in the meta line (geometry is relative to that frame)."""
    lines = render_elements([_cell(0, 0, "x")], rotation=90).splitlines()
    assert json.loads(lines[0]) == {"rotation": 90, "cells": 1}


# ---------------------------------------------------------------------------
# structure_table — engine orchestration + failure semantics
# ---------------------------------------------------------------------------


def test_structure_table_passes_ocr_results_through():
    seen: dict[str, object] = {}

    def fake_engine(img, ocr_results=None):
        seen["ocr_results"] = ocr_results
        return _engine_result(
            "<table><tr><td>v</td></tr></table>",
            [_bbox8(0, 0, 50, 100)],
            [[0, 0, 0, 0]],
        )

    cells = structure_table(_fake_jpeg(), OCR_TRIPLE, engine=fake_engine)
    # The raw rapidocr triple goes through verbatim, wrapped in the
    # one-image batch list rapid-table expects. [DECISION-tt.30]
    assert seen["ocr_results"] == [OCR_TRIPLE]
    assert len(cells) == 1
    assert cells[0].text == "v"


def test_structure_table_none_ocr_results_returns_empty():
    def exploding_engine(img, ocr_results=None):  # pragma: no cover - must not run
        raise AssertionError("engine must not be called without OCR evidence")

    assert structure_table(_fake_jpeg(), None, engine=exploding_engine) == []


def test_structure_table_empty_boxes_returns_empty():
    def exploding_engine(img, ocr_results=None):  # pragma: no cover - must not run
        raise AssertionError("engine must not be called without OCR evidence")

    assert structure_table(_fake_jpeg(), (None, None, None), engine=exploding_engine) == []
    assert structure_table(_fake_jpeg(), ([], (), ()), engine=exploding_engine) == []


def test_structure_table_engine_error_returns_empty():
    def broken_engine(img, ocr_results=None):
        raise RuntimeError("onnx exploded")

    assert structure_table(_fake_jpeg(), OCR_TRIPLE, engine=broken_engine) == []


def test_structure_table_no_html_returns_empty():
    def empty_engine(img, ocr_results=None):
        return types.SimpleNamespace(pred_htmls=[], cell_bboxes=[], logic_points=[])

    assert structure_table(_fake_jpeg(), OCR_TRIPLE, engine=empty_engine) == []


# ---------------------------------------------------------------------------
# CLI table-preview
# ---------------------------------------------------------------------------

CELLS = [_cell(0, 0, "H"), _cell(1, 0, "v")]


def _patch_pipeline(
    monkeypatch: pytest.MonkeyPatch,
    cells: list[TableCell],
    *,
    ocr: tuple[object, object, object] | None = OCR_TRIPLE,
) -> None:
    monkeypatch.setattr(
        "jcontract.impls._pdfium_render.render_pdf_page_jpeg",
        lambda pdf_path, page, *, dpi, jpeg_quality: b"fake-jpeg",
    )
    monkeypatch.setattr("jcontract.impls._table_assemble.page_ocr_results", lambda jpeg: ocr)
    monkeypatch.setattr(
        "jcontract.impls._table_assemble.structure_table",
        lambda jpeg, ocr_results: cells,
    )


def test_cli_md_to_stdout(tmp_path, monkeypatch):
    _patch_pipeline(monkeypatch, CELLS)
    pdf = tmp_path / "doc.pdf"
    pdf.write_bytes(b"%PDF-fake")
    result = runner.invoke(app, ["table-preview", str(pdf), "--page", "1"])
    assert result.exit_code == 0, result.output
    assert "| H |" in result.output
    assert "| v |" in result.output


def test_cli_elements_format(tmp_path, monkeypatch):
    _patch_pipeline(monkeypatch, CELLS)
    pdf = tmp_path / "doc.pdf"
    pdf.write_bytes(b"%PDF-fake")
    result = runner.invoke(app, ["table-preview", str(pdf), "--page", "1", "--format", "elements"])
    assert result.exit_code == 0, result.output
    first = json.loads(result.output.splitlines()[0])
    assert first == {"rotation": 0, "cells": 2}


def test_cli_out_writes_file(tmp_path, monkeypatch):
    _patch_pipeline(monkeypatch, CELLS)
    pdf = tmp_path / "doc.pdf"
    pdf.write_bytes(b"%PDF-fake")
    out = tmp_path / "table.md"
    result = runner.invoke(app, ["table-preview", str(pdf), "--page", "1", "--out", str(out)])
    assert result.exit_code == 0, result.output
    assert "| v |" in out.read_text(encoding="utf-8")


def test_cli_empty_page_reports_no_table(tmp_path, monkeypatch):
    _patch_pipeline(monkeypatch, [], ocr=None)
    pdf = tmp_path / "doc.pdf"
    pdf.write_bytes(b"%PDF-fake")
    result = runner.invoke(app, ["table-preview", str(pdf), "--page", "1"])
    assert result.exit_code == 0, result.output
    assert "no table structure detected" in result.output


def test_cli_rejects_unknown_format(tmp_path, monkeypatch):
    _patch_pipeline(monkeypatch, CELLS)
    pdf = tmp_path / "doc.pdf"
    pdf.write_bytes(b"%PDF-fake")
    result = runner.invoke(app, ["table-preview", str(pdf), "--page", "1", "--format", "html"])
    assert result.exit_code != 0
    assert "--format must be 'md' or 'elements'" in result.output


def test_cli_out_of_range_page_is_usage_error(tmp_path, monkeypatch):
    def raise_index_error(pdf_path, page, *, dpi, jpeg_quality):
        raise IndexError("page index out of range")

    monkeypatch.setattr("jcontract.impls._pdfium_render.render_pdf_page_jpeg", raise_index_error)
    pdf = tmp_path / "doc.pdf"
    pdf.write_bytes(b"%PDF-fake")
    result = runner.invoke(app, ["table-preview", str(pdf), "--page", "99"])
    assert result.exit_code != 0
    assert "out of range" in result.output
