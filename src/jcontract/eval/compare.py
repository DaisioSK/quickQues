"""Compare two eval reports (Enhancement E12 / ssEvalCompare).

What:
  ``compare_reports(a, b)`` diffs the ``metrics_mean`` and
  ``metrics_by_category`` blocks of two reports produced by
  ``eval/runner.run_eval`` (A = baseline, B = candidate).
Why:
  Every meaningful eval question is an A/B: does ``--caption`` raise the
  drawing-category recall (E2 ROI)? does ``--vision-model sonnet`` beat
  haiku? does the reranker help? A single run's numbers can't answer
  that; the diff can. Pure functions here, CLI printing in cli.py.
Context:
  A report dict has at least ``metrics_mean: dict[str, float]`` and
  ``metrics_by_category: dict[str, dict[str, float]]`` (older reports may
  lack the latter — treated as empty). We never mutate the inputs.
"""

from __future__ import annotations

from typing import Any


def _diff_metric_block(
    a: dict[str, float], b: dict[str, float]
) -> dict[str, dict[str, float | None]]:
    """Diff two flat {metric: value} dicts → {metric: {a, b, delta}}.

    A metric present in only one side gets ``None`` on the missing side and
    a ``None`` delta (can't subtract a missing measurement — don't fake 0).
    """
    out: dict[str, dict[str, float | None]] = {}
    for metric in sorted(set(a) | set(b)):
        av = a.get(metric)
        bv = b.get(metric)
        delta = bv - av if av is not None and bv is not None else None
        out[metric] = {"a": av, "b": bv, "delta": delta}
    return out


def compare_reports(report_a: dict[str, Any], report_b: dict[str, Any]) -> dict[str, Any]:
    """Diff two eval reports. Returns a structured, JSON-friendly diff.

    Shape::

        {
          "metrics_mean": {metric: {"a", "b", "delta"}, ...},
          "metrics_by_category": {category: {metric: {"a","b","delta"}}, ...},
        }

    ``delta`` is ``b - a`` (positive = candidate improved over baseline) or
    ``None`` when a metric is absent on either side.
    """
    mean = _diff_metric_block(
        report_a.get("metrics_mean", {}) or {},
        report_b.get("metrics_mean", {}) or {},
    )

    cat_a: dict[str, dict[str, float]] = report_a.get("metrics_by_category", {}) or {}
    cat_b: dict[str, dict[str, float]] = report_b.get("metrics_by_category", {}) or {}
    by_category: dict[str, dict[str, dict[str, float | None]]] = {}
    for category in sorted(set(cat_a) | set(cat_b)):
        by_category[category] = _diff_metric_block(
            cat_a.get(category, {}) or {},
            cat_b.get(category, {}) or {},
        )

    return {"metrics_mean": mean, "metrics_by_category": by_category}
