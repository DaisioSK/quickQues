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


# ---------------------------------------------------------------------------
# ssGE geometry signals: sidecar fields, pre-ssGE compat, CLI flag rules
# ---------------------------------------------------------------------------


def test_engine_run_sidecar_carries_geometry_signals(tmp_path):
    """An engine run persists the ssGE geometry block alongside the ssQA
    score metrics (same write-once timing). [DECISION-pl.21]"""
    cache_dir = tmp_path / "cache"
    # Two side-by-side columns, two boxes each: n_columns=2, in-line gap.
    boxes = [
        _box(100, 100, 450, 130),
        _box(620, 105, 1000, 135),
        _box(100, 140, 450, 170),
        _box(620, 145, 1000, 175),
    ]
    engine = _make_engine(boxes, ("L1", "R1", "L2", "R2"))
    RapidOcrParser(cache_dir=cache_dir, engine=engine, max_pages=1).parse(SYNTHETIC_PDF)

    (sidecar,) = cache_dir.glob("rapidocr-*.metrics.json")
    metrics = json.loads(sidecar.read_text(encoding="utf-8"))
    assert metrics["geometry_version"] == 1
    assert metrics["n_columns"] == 2
    assert metrics["max_band_gap"] > 0.1
    assert 0.0 < metrics["box_coverage"] < 1.0
    assert metrics["order_divergence"] > 0.0


def test_quality_metrics_reads_pre_ssge_sidecar_without_geometry(tmp_path):
    """A pre-ssGE sidecar (no geometry keys) must read back as-is: no
    engine run, no crash, geometry simply absent. W6 replays archived 45c
    sidecars/JSONL — backward read compatibility is a hard requirement."""
    cache_dir = tmp_path / "cache"
    engine_1 = _make_engine([_box(10, 10, 100, 40)], ("old text",), scores=(0.7,))
    RapidOcrParser(cache_dir=cache_dir, engine=engine_1, max_pages=1).parse(SYNTHETIC_PDF)
    (sidecar,) = cache_dir.glob("rapidocr-*.metrics.json")
    old = json.loads(sidecar.read_text(encoding="utf-8"))
    geometry_keys = (
        "geometry_version",
        "n_columns",
        "max_band_gap",
        "box_coverage",
        "order_divergence",
    )
    for key in geometry_keys:
        old.pop(key)
    sidecar.write_text(json.dumps(old, ensure_ascii=False), encoding="utf-8")

    engine_2 = _make_engine([_box(10, 10, 100, 40)], ("SHOULD NOT RUN",))
    records = RapidOcrParser(cache_dir=cache_dir, engine=engine_2, max_pages=1).quality_metrics(
        SYNTHETIC_PDF
    )

    assert engine_2.call_count == 0  # sidecar-first read still holds
    assert "n_columns" not in records[0]


def test_cli_report_nulls_geometry_on_pre_ssge_records(tmp_path, monkeypatch):
    """ocr-quality over a pre-ssGE sidecar: geometry columns are null in the
    JSONL and a geometry flag rule never triggers on them (null semantics,
    DECISION-cq.20/pl.21)."""
    monkeypatch.chdir(tmp_path)
    cache_dir = Path("data/ocr_cache")
    engine = _make_engine([_box(10, 10, 100, 40)], ("legacy page",), scores=(0.9,))
    monkeypatch.setattr(RapidOcrParser, "_ensure_engine", lambda self: engine)
    RapidOcrParser(cache_dir=cache_dir, max_pages=1).parse(SYNTHETIC_PDF)
    (sidecar,) = cache_dir.glob("rapidocr-*.metrics.json")
    old = json.loads(sidecar.read_text(encoding="utf-8"))
    geometry_keys = (
        "geometry_version",
        "n_columns",
        "max_band_gap",
        "box_coverage",
        "order_divergence",
    )
    for key in geometry_keys:
        old.pop(key)
    sidecar.write_text(json.dumps(old, ensure_ascii=False), encoding="utf-8")

    result = runner.invoke(
        app,
        [
            "ocr-quality",
            str(SYNTHETIC_PDF),
            "--max-pages",
            "1",
            "--flag-above",
            "order_divergence:0.0",
        ],
    )
    assert result.exit_code == 0, result.output
    records = [json.loads(line) for line in result.output.splitlines() if line.startswith("{")]
    assert records[0]["n_columns"] is None
    assert records[0]["order_divergence"] is None
    assert records[0]["flagged"] is False  # null never triggers a rule


def test_cli_geometry_flag_rule_triggers_on_fresh_scan(tmp_path, monkeypatch):
    """New scans expose the geometry signals to --flag rules plug-and-play:
    a two-column page trips --flag-above n_columns:1."""
    monkeypatch.chdir(tmp_path)
    boxes = [
        _box(100, 100, 450, 130),
        _box(620, 105, 1000, 135),
        _box(100, 140, 450, 170),
        _box(620, 145, 1000, 175),
    ]
    engine = _make_engine(boxes, ("L1", "R1", "L2", "R2"))
    monkeypatch.setattr(RapidOcrParser, "_ensure_engine", lambda self: engine)

    result = runner.invoke(
        app,
        [
            "ocr-quality",
            str(SYNTHETIC_PDF),
            "--max-pages",
            "1",
            "--flag-above",
            "n_columns:1",
        ],
    )
    assert result.exit_code == 0, result.output
    records = [json.loads(line) for line in result.output.splitlines() if line.startswith("{")]
    assert records[0]["n_columns"] == 2
    assert records[0]["flagged"] is True
    assert records[0]["flag_reasons"] == ["n_columns=2>1"]


def test_cli_assembly_regions_scans_into_own_namespace(tmp_path, monkeypatch):
    """ocr-quality --assembly regions writes .regions-suffixed artifacts and
    leaves the default namespace empty. [DECISION-pl.22]"""
    monkeypatch.chdir(tmp_path)
    engine = _make_engine([_box(10, 10, 100, 40)], ("solo line",), scores=(0.9,))
    monkeypatch.setattr(RapidOcrParser, "_ensure_engine", lambda self: engine)

    result = runner.invoke(
        app,
        ["ocr-quality", str(SYNTHETIC_PDF), "--max-pages", "1", "--assembly", "regions"],
    )
    assert result.exit_code == 0, result.output
    cache_dir = Path("data/ocr_cache")
    assert len(list(cache_dir.glob("rapidocr-*.metrics.regions.json"))) == 1
    assert list(cache_dir.glob("rapidocr-*.metrics.json")) == []


def test_cli_unknown_assembly_is_usage_error(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(
        app, ["ocr-quality", str(SYNTHETIC_PDF), "--max-pages", "1", "--assembly", "diagonal"]
    )
    assert result.exit_code != 0
    assert "assembly" in result.output
