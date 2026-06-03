"""Unit tests for eval report comparison (Enhancement E12 / ssEvalCompare)."""

from __future__ import annotations

from typing import Any

from jcontract.eval.compare import compare_reports


def test_diffs_metrics_mean() -> None:
    a = {"metrics_mean": {"recall_at_5": 0.5, "recall_at_10": 0.8}}
    b = {"metrics_mean": {"recall_at_5": 0.7, "recall_at_10": 0.8}}
    diff = compare_reports(a, b)["metrics_mean"]

    assert diff["recall_at_5"]["a"] == 0.5
    assert diff["recall_at_5"]["b"] == 0.7
    assert abs(diff["recall_at_5"]["delta"] - 0.2) < 1e-9
    assert diff["recall_at_10"]["delta"] == 0.0  # unchanged → zero delta


def test_metric_missing_on_one_side_has_none_delta() -> None:
    a = {"metrics_mean": {"recall_at_5": 0.5}}
    b = {"metrics_mean": {"recall_at_5": 0.5, "faithfulness": 0.9}}
    diff = compare_reports(a, b)["metrics_mean"]

    # faithfulness only exists on B → a=None, delta=None (don't fake a 0 baseline).
    assert diff["faithfulness"]["a"] is None
    assert diff["faithfulness"]["b"] == 0.9
    assert diff["faithfulness"]["delta"] is None


def test_diffs_by_category() -> None:
    a = {"metrics_by_category": {"drawing": {"recall_at_5": 0.2, "n_cases": 6.0}}}
    b = {"metrics_by_category": {"drawing": {"recall_at_5": 0.6, "n_cases": 6.0}}}
    drawing = compare_reports(a, b)["metrics_by_category"]["drawing"]

    assert abs(drawing["recall_at_5"]["delta"] - 0.4) < 1e-9  # caption ROI signal
    assert drawing["n_cases"]["delta"] == 0.0


def test_category_only_in_one_report() -> None:
    a: dict[str, Any] = {"metrics_by_category": {}}
    b = {"metrics_by_category": {"drawing": {"recall_at_5": 0.6}}}
    by_cat = compare_reports(a, b)["metrics_by_category"]

    assert "drawing" in by_cat
    assert by_cat["drawing"]["recall_at_5"]["a"] is None
    assert by_cat["drawing"]["recall_at_5"]["delta"] is None


def test_missing_blocks_default_empty() -> None:
    # Older reports may lack metrics_by_category entirely — no crash.
    diff = compare_reports({"metrics_mean": {}}, {"metrics_mean": {}})
    assert diff["metrics_mean"] == {}
    assert diff["metrics_by_category"] == {}
