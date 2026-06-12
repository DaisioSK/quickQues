"""Unit tests for the `ocr-gallery` human-triage export command (ssTG).

Strategy (mirrors test_ocr_quality.py):
- NO real rendering: both pdfium render entry points are monkeypatched to
  return deterministic fake JPEG bytes (the synthetic fixture PDF is only
  ever opened for its page count in the no---quality path).
- NO real OCR: the rapidocr engine is injected via `_ensure_engine`
  monkeypatching, exactly like the ocr-quality CLI tests.
- Covered surfaces: quality-JSONL parsing, threshold filtering, worst-first
  sorting (DECISION-tt.10), --top truncation, index.md row/header format,
  full-text .txt vs 80-char preview (DECISION-tt.12), stored-flag
  re-evaluation (DECISION-tt.13), and usage errors from the shared rule
  parser.
"""

from __future__ import annotations

import json
import types
from collections.abc import Callable
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from typer.testing import CliRunner

from jcontract.cli import app
from jcontract.impls.rapidocr_parser import RapidOcrParser

SYNTHETIC_PDF = (
    Path(__file__).parent.parent / "eval/fixtures/synthetic_contract_tqa.pdf"
).resolve()

runner = CliRunner()


def _box(x0: float, y0: float, x1: float, y1: float) -> list[list[float]]:
    return [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]


def _quality_record(
    page_num: int, min_score: float | None, *, flagged: bool = False, **extra: object
) -> dict[str, object]:
    """One archived ocr-quality JSONL record (report projection, not sidecar)."""
    record: dict[str, object] = {
        "page_num": page_num,
        "boxes": 10,
        "mean_score": 0.9,
        "min_score": min_score,
        "low_score_ratio": 0.0,
        "non_alnum_ratio": 0.05,
        "garbled_ratio": 0.0,
        "flagged": flagged,
        "flag_reasons": [],
    }
    record.update(extra)
    return record


def _write_quality_jsonl(path: Path, records: list[dict[str, object]]) -> None:
    path.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in records) + "\n",
        encoding="utf-8",
    )


def _patch_render_and_engine(
    monkeypatch: pytest.MonkeyPatch, *, page_text: Callable[[int], str]
) -> MagicMock:
    """Fake both render entry points + the OCR engine.

    `page_text(page_num)` decides what the fake engine "reads" on each page;
    the fake JPEG bytes embed the page number so every page gets a distinct
    OCR cache key (mirroring real content-addressing).
    """
    fake_jpegs: dict[bytes, int] = {}

    def fake_render_pdf_page_jpeg(
        pdf_path: Path, page_num: int, *, dpi: int, jpeg_quality: int
    ) -> bytes:
        assert (dpi, jpeg_quality) == (150, 85)  # cache-key-standard geometry
        jpeg = f"FAKEJPEG-{page_num}".encode()
        fake_jpegs[jpeg] = page_num
        return jpeg

    def fake_render_page_jpeg(page: object, *, dpi: int, jpeg_quality: int) -> bytes:
        # The no---quality scan path renders via the page-object entry point;
        # page objects don't carry their index, so derive it from call order.
        page_num = len(fake_jpegs) + 1
        jpeg = f"FAKEJPEG-{page_num}".encode()
        fake_jpegs[jpeg] = page_num
        return jpeg

    def fake_engine(jpeg_bytes: bytes) -> types.SimpleNamespace:
        text = page_text(fake_jpegs[jpeg_bytes])
        return types.SimpleNamespace(boxes=[_box(10, 10, 100, 40)], txts=(text,), scores=(0.5,))

    monkeypatch.setattr(
        "jcontract.impls._pdfium_render.render_pdf_page_jpeg", fake_render_pdf_page_jpeg
    )
    monkeypatch.setattr("jcontract.impls.rapidocr_parser.render_page_jpeg", fake_render_page_jpeg)
    engine = MagicMock(side_effect=fake_engine)
    monkeypatch.setattr(RapidOcrParser, "_ensure_engine", lambda self: engine)
    return engine


# ---------------------------------------------------------------------------
# Filtering + sorting + file layout (archived --quality path)
# ---------------------------------------------------------------------------


def test_gallery_filters_sorts_and_writes_files(tmp_path, monkeypatch):
    """min_score:0.756 selects exactly the sub-threshold pages, worst first."""
    monkeypatch.chdir(tmp_path)
    _patch_render_and_engine(monkeypatch, page_text=lambda n: f"ocr text page {n}")
    quality = tmp_path / "report.jsonl"
    _write_quality_jsonl(
        quality,
        [
            _quality_record(1, 0.9),
            _quality_record(2, 0.6),
            _quality_record(3, 0.4),
            _quality_record(4, 0.8),
            _quality_record(5, 0.5),
        ],
    )
    out = tmp_path / "gallery"

    result = runner.invoke(
        app,
        [
            "ocr-gallery",
            str(SYNTHETIC_PDF),
            "--quality",
            str(quality),
            "--flag-below",
            "min_score:0.756",
            "--out",
            str(out),
        ],
    )
    assert result.exit_code == 0, result.output

    # Flagged pages (2, 3, 5) get jpg+txt; clean pages (1, 4) get nothing.
    assert sorted(p.name for p in out.glob("p*")) == [
        "p0002.jpg",
        "p0002.txt",
        "p0003.jpg",
        "p0003.txt",
        "p0005.jpg",
        "p0005.txt",
    ]
    assert (out / "p0003.jpg").read_bytes() == b"FAKEJPEG-3"
    assert (out / "p0003.txt").read_text(encoding="utf-8") == "ocr text page 3"

    # Worst-first ordering: 0.4 < 0.5 < 0.6 (ascending triggered signal,
    # DECISION-tt.10). Data rows = table lines whose first cell is a number.
    index = (out / "index.md").read_text(encoding="utf-8")
    rows = [
        line
        for line in index.splitlines()
        if line.startswith("| ") and line.split("|")[1].strip().isdigit()
    ]
    assert [r.split("|")[1].strip() for r in rows] == ["3", "5", "2"]

    # Header is self-describing: pdf / rules / counts.
    assert SYNTHETIC_PDF.name in index
    assert "flag rules: min_score<0.756" in index
    assert "pages scanned: 5" in index
    assert "flagged: 3" in index
    assert "exported: 3" in index
    assert "flagged: 3/5 page(s); exported: 3" in result.output


def test_top_truncates_to_worst_n(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _patch_render_and_engine(monkeypatch, page_text=lambda n: f"page {n}")
    quality = tmp_path / "report.jsonl"
    _write_quality_jsonl(
        quality,
        [_quality_record(1, 0.6), _quality_record(2, 0.4), _quality_record(3, 0.5)],
    )
    out = tmp_path / "gallery"

    result = runner.invoke(
        app,
        [
            "ocr-gallery",
            str(SYNTHETIC_PDF),
            "--quality",
            str(quality),
            "--flag-below",
            "min_score:0.756",
            "--top",
            "1",
            "--out",
            str(out),
        ],
    )
    assert result.exit_code == 0, result.output

    # Only the single worst page (p2, min_score 0.4) is exported …
    assert sorted(p.name for p in out.glob("p*")) == ["p0002.jpg", "p0002.txt"]
    index = (out / "index.md").read_text(encoding="utf-8")
    # … but the header still reports the full flagged population.
    assert "flagged: 3" in index
    assert "exported: 1 (--top 1)" in index


def test_index_row_format_and_preview_truncation(tmp_path, monkeypatch):
    """Row = page | trigger | relative image link | sanitized 80-char preview."""
    monkeypatch.chdir(tmp_path)
    long_text = "line|one\nline two " + "x" * 100  # newline + pipe + >80 chars
    _patch_render_and_engine(monkeypatch, page_text=lambda n: long_text)
    quality = tmp_path / "report.jsonl"
    _write_quality_jsonl(quality, [_quality_record(7, 0.5)])
    out = tmp_path / "gallery"

    result = runner.invoke(
        app,
        [
            "ocr-gallery",
            str(SYNTHETIC_PDF),
            "--quality",
            str(quality),
            "--flag-below",
            "min_score:0.756",
            "--out",
            str(out),
        ],
    )
    assert result.exit_code == 0, result.output

    index = (out / "index.md").read_text(encoding="utf-8")
    (row,) = [line for line in index.splitlines() if line.startswith("| 7 ")]
    # Fixed cells verbatim: page | trigger (_flag_reasons format) | image link.
    prefix = "| 7 | min_score=0.5<0.756 | [p0007.jpg](p0007.jpg) | "
    assert row.startswith(prefix)
    # Preview cell: single-line, 80 chars max, pipes escaped so the raw OCR
    # pipe cannot add a 5th table cell.
    preview_cell = row[len(prefix) : -2]  # strip trailing " |"
    assert "\\|" in preview_cell
    assert preview_cell.replace("\\|", "|") == " ".join(long_text.split())[:80]
    assert "\n" not in preview_cell

    # Full untruncated text (newlines intact) lives in the .txt
    # (DECISION-tt.12).
    assert (out / "p0007.txt").read_text(encoding="utf-8") == long_text


def test_archived_flag_fields_are_ignored(tmp_path, monkeypatch):
    """Rules re-evaluate from signals; stored flagged=false must not hide a
    bad page (archived reports may predate any threshold, DECISION-tt.13)."""
    monkeypatch.chdir(tmp_path)
    _patch_render_and_engine(monkeypatch, page_text=lambda n: f"page {n}")
    quality = tmp_path / "report.jsonl"
    _write_quality_jsonl(
        quality,
        [
            _quality_record(1, 0.4, flagged=False),  # bad page, archived as unflagged
            _quality_record(2, 0.9, flagged=True, flag_reasons=["stale"]),  # vice versa
        ],
    )
    out = tmp_path / "gallery"

    result = runner.invoke(
        app,
        [
            "ocr-gallery",
            str(SYNTHETIC_PDF),
            "--quality",
            str(quality),
            "--flag-below",
            "min_score:0.756",
            "--out",
            str(out),
        ],
    )
    assert result.exit_code == 0, result.output
    assert sorted(p.name for p in out.glob("p*")) == ["p0001.jpg", "p0001.txt"]


def test_flag_above_selects_higher_is_worse_signals(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _patch_render_and_engine(monkeypatch, page_text=lambda n: f"page {n}")
    quality = tmp_path / "report.jsonl"
    _write_quality_jsonl(
        quality,
        [
            _quality_record(1, 0.9, garbled_ratio=0.5),
            _quality_record(2, 0.9, garbled_ratio=0.0),
        ],
    )
    out = tmp_path / "gallery"

    result = runner.invoke(
        app,
        [
            "ocr-gallery",
            str(SYNTHETIC_PDF),
            "--quality",
            str(quality),
            "--flag-above",
            "garbled_ratio:0.3",
            "--out",
            str(out),
        ],
    )
    assert result.exit_code == 0, result.output
    assert sorted(p.name for p in out.glob("p*.jpg")) == ["p0001.jpg"]
    assert "garbled_ratio=0.5>0.3" in (out / "index.md").read_text(encoding="utf-8")


def test_null_signal_never_triggers(tmp_path, monkeypatch):
    """engine_error / zero-box records carry null signals — no evidence, no
    flag (same semantics as ocr-quality, DECISION-cq.20)."""
    monkeypatch.chdir(tmp_path)
    _patch_render_and_engine(monkeypatch, page_text=lambda n: f"page {n}")
    quality = tmp_path / "report.jsonl"
    _write_quality_jsonl(
        quality,
        [_quality_record(1, None, engine_error="RuntimeError"), _quality_record(2, 0.5)],
    )
    out = tmp_path / "gallery"

    result = runner.invoke(
        app,
        [
            "ocr-gallery",
            str(SYNTHETIC_PDF),
            "--quality",
            str(quality),
            "--flag-below",
            "min_score:0.756",
            "--out",
            str(out),
        ],
    )
    assert result.exit_code == 0, result.output
    assert sorted(p.name for p in out.glob("p*.jpg")) == ["p0002.jpg"]


# ---------------------------------------------------------------------------
# No --quality: fresh scan path
# ---------------------------------------------------------------------------


def test_without_quality_report_runs_fresh_scan(tmp_path, monkeypatch):
    """No --quality → the ocr-quality scan runs (mocked engine, score 0.5)
    and the gallery exports from its records. Fixture PDF has 4 pages."""
    monkeypatch.chdir(tmp_path)
    engine = _patch_render_and_engine(monkeypatch, page_text=lambda n: f"scanned page {n}")
    out = tmp_path / "gallery"

    result = runner.invoke(
        app,
        [
            "ocr-gallery",
            str(SYNTHETIC_PDF),
            "--flag-below",
            "min_score:0.756",
            "--top",
            "2",
            "--out",
            str(out),
        ],
    )
    assert result.exit_code == 0, result.output

    assert engine.call_count > 0  # the scan actually OCR'd (no sidecars yet)
    index = (out / "index.md").read_text(encoding="utf-8")
    assert "pages scanned: 4" in index
    # All 4 pages score 0.5 → equal margin → page-number tie-break; --top 2.
    assert sorted(p.name for p in out.glob("p*.jpg")) == ["p0001.jpg", "p0002.jpg"]
    # Export re-uses the .txt cache the scan just backfilled — same fake
    # bytes → same sha256 → cache hit, text matches the scan's OCR.
    assert (out / "p0001.txt").read_text(encoding="utf-8") == "scanned page 1"


# ---------------------------------------------------------------------------
# Usage errors (shared rule parser)
# ---------------------------------------------------------------------------


def test_no_rules_is_usage_error(tmp_path):
    result = runner.invoke(app, ["ocr-gallery", str(SYNTHETIC_PDF), "--out", str(tmp_path / "g")])
    assert result.exit_code != 0
    assert "--flag-below" in result.output
    assert not (tmp_path / "g").exists()


def test_unknown_signal_is_usage_error(tmp_path):
    result = runner.invoke(
        app,
        [
            "ocr-gallery",
            str(SYNTHETIC_PDF),
            "--flag-below",
            "bogus:0.5",
            "--out",
            str(tmp_path / "g"),
        ],
    )
    assert result.exit_code != 0
    assert "bogus" in result.output


def test_non_numeric_threshold_is_usage_error(tmp_path):
    result = runner.invoke(
        app,
        [
            "ocr-gallery",
            str(SYNTHETIC_PDF),
            "--flag-below",
            "min_score:abc",
            "--out",
            str(tmp_path / "g"),
        ],
    )
    assert result.exit_code != 0


# ---------------------------------------------------------------------------
# ssHD --dpi: hi-res exported image, OCR text stays on the standard frame
# ---------------------------------------------------------------------------


def test_dpi_option_exports_hires_image_but_standard_text(tmp_path, monkeypatch):
    """--dpi 300 changes ONLY the exported jpg; the .txt must come from the
    standard 150dpi cache-key frame (the engine never sees hi-dpi bytes).
    [DECISION-pl.41]"""
    monkeypatch.chdir(tmp_path)

    def fake_render_pdf_page_jpeg(
        pdf_path: Path, page_num: int, *, dpi: int, jpeg_quality: int
    ) -> bytes:
        assert jpeg_quality == 85
        return f"FAKEJPEG-{page_num}-{dpi}".encode()

    def fake_engine(jpeg_bytes: bytes) -> types.SimpleNamespace:
        # The OCR path must only ever receive the 150dpi standard frame.
        assert jpeg_bytes.endswith(b"-150"), jpeg_bytes
        return types.SimpleNamespace(
            boxes=[_box(10, 10, 100, 40)], txts=("standard frame text",), scores=(0.5,)
        )

    monkeypatch.setattr(
        "jcontract.impls._pdfium_render.render_pdf_page_jpeg", fake_render_pdf_page_jpeg
    )
    engine = MagicMock(side_effect=fake_engine)
    monkeypatch.setattr(RapidOcrParser, "_ensure_engine", lambda self: engine)

    quality = tmp_path / "report.jsonl"
    _write_quality_jsonl(quality, [_quality_record(1, 0.9), _quality_record(2, 0.4)])
    out = tmp_path / "gallery"

    result = runner.invoke(
        app,
        [
            "ocr-gallery",
            str(SYNTHETIC_PDF),
            "--quality",
            str(quality),
            "--flag-below",
            "min_score:0.756",
            "--out",
            str(out),
            "--dpi",
            "300",
        ],
    )
    assert result.exit_code == 0, result.output

    # Image = the hi-dpi render; text = OCR of the standard frame.
    assert (out / "p0002.jpg").read_bytes() == b"FAKEJPEG-2-300"
    assert (out / "p0002.txt").read_text(encoding="utf-8") == "standard frame text"
    assert engine.call_count == 1


def test_default_dpi_renders_each_page_once(tmp_path, monkeypatch):
    """Without --dpi the export is byte-identical to pre-ssHD behaviour and
    the page is rendered exactly once (no redundant second render)."""
    monkeypatch.chdir(tmp_path)
    render_calls: list[tuple[int, int]] = []

    def fake_render_pdf_page_jpeg(
        pdf_path: Path, page_num: int, *, dpi: int, jpeg_quality: int
    ) -> bytes:
        render_calls.append((page_num, dpi))
        return f"FAKEJPEG-{page_num}-{dpi}".encode()

    def fake_engine(jpeg_bytes: bytes) -> types.SimpleNamespace:
        return types.SimpleNamespace(boxes=[_box(10, 10, 100, 40)], txts=("text",), scores=(0.5,))

    monkeypatch.setattr(
        "jcontract.impls._pdfium_render.render_pdf_page_jpeg", fake_render_pdf_page_jpeg
    )
    engine = MagicMock(side_effect=fake_engine)
    monkeypatch.setattr(RapidOcrParser, "_ensure_engine", lambda self: engine)

    quality = tmp_path / "report.jsonl"
    _write_quality_jsonl(quality, [_quality_record(1, 0.4)])
    out = tmp_path / "gallery"

    result = runner.invoke(
        app,
        [
            "ocr-gallery",
            str(SYNTHETIC_PDF),
            "--quality",
            str(quality),
            "--flag-below",
            "min_score:0.756",
            "--out",
            str(out),
        ],
    )
    assert result.exit_code == 0, result.output
    assert (out / "p0001.jpg").read_bytes() == b"FAKEJPEG-1-150"
    assert render_calls == [(1, 150)]
