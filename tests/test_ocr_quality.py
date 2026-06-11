"""Unit tests for the rapidocr quality-metrics sidecar + `ocr-quality` CLI (ssQA).

Strategy (mirrors test_rapidocr_parser.py):
- The rapidocr engine is ALWAYS mocked (callable returning .boxes/.txts/
  .scores) — no model download, no onnxruntime inference.
- The pypdfium2 render path IS exercised on the real synthetic fixture PDF.
- Covered surfaces: metric computation (_page_metrics), sidecar write timing
  (engine run = write; ingest cache hit = NO backfill — DECISION-cq.21),
  quality_metrics sidecar-first read + force-run backfill (DECISION-cq.22),
  and the CLI's caller-supplied flag thresholds (DECISION-cq.20).
"""

from __future__ import annotations

import json
import types
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from typer.testing import CliRunner

from jcontract.cli import app
from jcontract.impls.rapidocr_parser import RapidOcrParser, _page_metrics

SYNTHETIC_PDF = (
    Path(__file__).parent.parent / "eval/fixtures/synthetic_contract_tqa.pdf"
).resolve()

runner = CliRunner()


def _box(x0: float, y0: float, x1: float, y1: float) -> list[list[float]]:
    return [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]


def _make_engine(
    boxes: list[list[list[float]]] | None,
    txts: tuple[str, ...] | None,
    scores: tuple[float, ...] | None = None,
) -> MagicMock:
    if scores is None and txts is not None:
        scores = tuple(0.99 for _ in txts)
    engine = MagicMock()
    engine.return_value = types.SimpleNamespace(boxes=boxes, txts=txts, scores=scores)
    return engine


# ---------------------------------------------------------------------------
# _page_metrics — pure metric computation
# ---------------------------------------------------------------------------


def test_page_metrics_score_aggregates():
    m = _page_metrics(3, (0.9, 0.5, 0.7, 0.95), "Clause 4.2 applies")
    assert m["page_num"] == 3
    assert m["boxes"] == 4
    assert m["scores"] == [0.9, 0.5, 0.7, 0.95]
    assert m["mean_score"] == pytest.approx((0.9 + 0.5 + 0.7 + 0.95) / 4)
    assert m["min_score"] == pytest.approx(0.5)
    # Only 0.5 is < 0.7 (the frozen threshold is strict-less-than).
    assert m["low_score_ratio"] == pytest.approx(1 / 4)


def test_page_metrics_char_ratios_cjk_counts_as_alnum():
    # 8 non-whitespace chars: 6 alnum (Clause + 4 + 条 — CJK isalnum) + "." + "§".
    m = _page_metrics(1, (0.9,), "Clause4. 条 §")
    assert m["alnum_ratio"] == pytest.approx(8 / 10)
    # "§" (U+00A7) is outside the expected charset → garbled; "." and CJK are not.
    assert m["garbled_ratio"] == pytest.approx(1 / 10)


def test_page_metrics_zero_boxes_and_empty_text_yield_nulls():
    """No evidence → null, not fake 0.0 (DECISION-cq.21 null semantics)."""
    m = _page_metrics(1, (), "")
    assert m["boxes"] == 0
    assert m["scores"] == []
    assert m["mean_score"] is None
    assert m["min_score"] is None
    assert m["low_score_ratio"] is None
    assert m["alnum_ratio"] is None
    assert m["garbled_ratio"] is None


# ---------------------------------------------------------------------------
# Sidecar write timing (ingest path)
# ---------------------------------------------------------------------------


def test_engine_run_writes_metrics_sidecar_next_to_text(tmp_path):
    cache_dir = tmp_path / "cache"
    engine = _make_engine([_box(10, 10, 100, 40)], ("hello clause",), scores=(0.8,))
    RapidOcrParser(cache_dir=cache_dir, engine=engine, max_pages=1).parse(SYNTHETIC_PDF)

    sidecars = list(cache_dir.glob("rapidocr-*.metrics.json"))
    assert len(sidecars) == 1, f"expected one metrics sidecar, got {sidecars}"
    metrics = json.loads(sidecars[0].read_text(encoding="utf-8"))
    assert metrics["page_num"] == 1
    assert metrics["boxes"] == 1
    assert metrics["scores"] == [0.8]
    assert metrics["mean_score"] == pytest.approx(0.8)
    # Sidecar shares the content-hash stem with the .txt (same namespace).
    (txt,) = cache_dir.glob("rapidocr-*.text.txt")
    assert sidecars[0].name == txt.name.replace(".text.txt", ".metrics.json")


def test_non_default_model_sidecar_gets_model_suffix(tmp_path):
    cache_dir = tmp_path / "cache"
    engine = _make_engine([_box(10, 10, 100, 40)], ("server text",))
    RapidOcrParser(cache_dir=cache_dir, engine=engine, max_pages=1, model_type="server").parse(
        SYNTHETIC_PDF
    )

    assert len(list(cache_dir.glob("rapidocr-*.metrics.ppocrv5-server.json"))) == 1


def test_ingest_cache_hit_does_not_backfill_sidecar(tmp_path):
    """.txt present + sidecar absent → ingest must NOT run the engine nor
    create the sidecar (zero ingest-performance regression, DECISION-cq.21)."""
    cache_dir = tmp_path / "cache"
    engine_1 = _make_engine([_box(10, 10, 100, 40)], ("cached text",))
    RapidOcrParser(cache_dir=cache_dir, engine=engine_1, max_pages=1).parse(SYNTHETIC_PDF)
    for sidecar in cache_dir.glob("rapidocr-*.metrics.json"):
        sidecar.unlink()  # simulate a pre-ssQA cache: text only

    engine_2 = _make_engine([_box(10, 10, 100, 40)], ("SHOULD NOT RUN",))
    pages = RapidOcrParser(cache_dir=cache_dir, engine=engine_2, max_pages=1).parse(SYNTHETIC_PDF)

    assert pages[0].text == "cached text"
    assert engine_2.call_count == 0
    assert list(cache_dir.glob("rapidocr-*.metrics.json")) == []


def test_engine_error_writes_no_sidecar(tmp_path):
    cache_dir = tmp_path / "cache"
    engine = MagicMock(side_effect=RuntimeError("boom"))
    RapidOcrParser(cache_dir=cache_dir, engine=engine, max_pages=1).parse(SYNTHETIC_PDF)
    assert list(cache_dir.glob("rapidocr-*.metrics.json")) == []


# ---------------------------------------------------------------------------
# quality_metrics — sidecar-first read / force-run backfill
# ---------------------------------------------------------------------------


def test_quality_metrics_reads_existing_sidecar_without_engine(tmp_path):
    cache_dir = tmp_path / "cache"
    engine_1 = _make_engine([_box(10, 10, 100, 40)], ("first pass",), scores=(0.65,))
    RapidOcrParser(cache_dir=cache_dir, engine=engine_1, max_pages=1).parse(SYNTHETIC_PDF)

    engine_2 = _make_engine([_box(10, 10, 100, 40)], ("SHOULD NOT RUN",))
    records = RapidOcrParser(cache_dir=cache_dir, engine=engine_2, max_pages=1).quality_metrics(
        SYNTHETIC_PDF
    )

    assert engine_2.call_count == 0
    assert len(records) == 1
    assert records[0]["scores"] == [0.65]
    assert records[0]["low_score_ratio"] == pytest.approx(1.0)


def test_quality_metrics_force_runs_engine_on_txt_only_cache(tmp_path):
    """.txt cached but no sidecar → quality scan re-runs the engine and
    backfills the sidecar; the existing .txt is NOT rewritten (cq.22)."""
    cache_dir = tmp_path / "cache"
    engine_1 = _make_engine([_box(10, 10, 100, 40)], ("ingest text",))
    RapidOcrParser(cache_dir=cache_dir, engine=engine_1, max_pages=1).parse(SYNTHETIC_PDF)
    for sidecar in cache_dir.glob("rapidocr-*.metrics.json"):
        sidecar.unlink()
    (txt,) = cache_dir.glob("rapidocr-*.text.txt")
    txt_mtime = txt.stat().st_mtime_ns

    engine_2 = _make_engine([_box(10, 10, 100, 40)], ("requality text",), scores=(0.4,))
    records = RapidOcrParser(cache_dir=cache_dir, engine=engine_2, max_pages=1).quality_metrics(
        SYNTHETIC_PDF
    )

    assert engine_2.call_count == 1  # forced despite .txt cache hit
    assert records[0]["scores"] == [0.4]
    (sidecar,) = cache_dir.glob("rapidocr-*.metrics.json")  # backfilled
    assert json.loads(sidecar.read_text(encoding="utf-8"))["scores"] == [0.4]
    assert txt.stat().st_mtime_ns == txt_mtime  # ingest cache bytes untouched
    assert txt.read_text(encoding="utf-8") == "ingest text"


def test_quality_metrics_cold_cache_backfills_both_files(tmp_path):
    cache_dir = tmp_path / "cache"
    engine = _make_engine([_box(10, 10, 100, 40)], ("cold page",), scores=(0.9,))
    records = RapidOcrParser(cache_dir=cache_dir, engine=engine, max_pages=1).quality_metrics(
        SYNTHETIC_PDF
    )

    assert records[0]["boxes"] == 1
    (txt,) = cache_dir.glob("rapidocr-*.text.txt")
    assert txt.read_text(encoding="utf-8") == "cold page"
    assert len(list(cache_dir.glob("rapidocr-*.metrics.json"))) == 1


def test_quality_metrics_engine_error_yields_degenerate_record(tmp_path):
    cache_dir = tmp_path / "cache"
    engine = MagicMock(side_effect=RuntimeError("boom"))
    records = RapidOcrParser(cache_dir=cache_dir, engine=engine, max_pages=2).quality_metrics(
        SYNTHETIC_PDF
    )

    assert len(records) == 2  # scan never aborts
    assert all(r["engine_error"] == "RuntimeError" for r in records)
    assert all(r["boxes"] is None for r in records)
    assert list(cache_dir.glob("rapidocr-*")) == []  # transient → retry next scan


# ---------------------------------------------------------------------------
# ocr-quality CLI — caller-supplied thresholds (DECISION-cq.20)
# ---------------------------------------------------------------------------


@pytest.fixture
def cli_engine(tmp_path, monkeypatch):
    """Run the CLI from tmp_path (data/ocr_cache lands there) with a fake
    two-result engine: page 1 healthy, page 2 low-score + garbled."""
    monkeypatch.chdir(tmp_path)
    results = [
        types.SimpleNamespace(
            boxes=[_box(10, 10, 100, 40)], txts=("Healthy clause text",), scores=(0.95,)
        ),
        types.SimpleNamespace(
            boxes=[_box(10, 10, 100, 40), _box(10, 60, 100, 90)],
            txts=("ª¤ºø", "noisy"),
            scores=(0.42, 0.91),
        ),
    ]
    engine = MagicMock(side_effect=results)
    monkeypatch.setattr(RapidOcrParser, "_ensure_engine", lambda self: engine)
    return engine


def test_cli_flag_below_marks_low_mean_score_page(cli_engine, tmp_path):
    out = tmp_path / "report.jsonl"
    result = runner.invoke(
        app,
        [
            "ocr-quality",
            str(SYNTHETIC_PDF),
            "--max-pages",
            "2",
            "--out",
            str(out),
            "--flag-below",
            "mean_score:0.8",
        ],
    )
    assert result.exit_code == 0, result.output

    records = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert [r["page_num"] for r in records] == [1, 2]
    assert records[0]["flagged"] is False
    assert records[1]["flagged"] is True
    assert records[1]["flag_reasons"] == ["mean_score=0.665<0.8"]
    # The five pre-registered signals + garbled heuristic are all present.
    expected_signals = (
        "mean_score",
        "min_score",
        "low_score_ratio",
        "boxes",
        "non_alnum_ratio",
        "garbled_ratio",
    )
    for signal in expected_signals:
        assert signal in records[0]
    # Raw score lists stay in the sidecar, not the report.
    assert "scores" not in records[0]
    assert "flagged: 1/2 page(s) -> [2]" in result.output


def test_cli_flag_above_catches_higher_is_worse_signals(cli_engine, tmp_path):
    result = runner.invoke(
        app,
        [
            "ocr-quality",
            str(SYNTHETIC_PDF),
            "--max-pages",
            "2",
            "--flag-above",
            "garbled_ratio:0.3",
        ],
    )
    assert result.exit_code == 0, result.output
    records = [json.loads(line) for line in result.output.splitlines() if line.startswith("{")]
    assert records[0]["flagged"] is False
    assert records[1]["flagged"] is True  # "ª¤ºø" → garbled_ratio 4/9 > 0.3


def test_cli_no_rules_flags_nothing(cli_engine, tmp_path):
    result = runner.invoke(app, ["ocr-quality", str(SYNTHETIC_PDF), "--max-pages", "2"])
    assert result.exit_code == 0, result.output
    records = [json.loads(line) for line in result.output.splitlines() if line.startswith("{")]
    assert all(r["flagged"] is False for r in records)
    assert "flag rules: none supplied" in result.output


def test_cli_unknown_signal_is_usage_error(cli_engine):
    result = runner.invoke(
        app,
        ["ocr-quality", str(SYNTHETIC_PDF), "--max-pages", "1", "--flag-below", "bogus:0.5"],
    )
    assert result.exit_code != 0
    assert "bogus" in result.output


def test_cli_non_numeric_threshold_is_usage_error(cli_engine):
    result = runner.invoke(
        app,
        ["ocr-quality", str(SYNTHETIC_PDF), "--max-pages", "1", "--flag-below", "mean_score:abc"],
    )
    assert result.exit_code != 0
