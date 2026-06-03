"""Pure metric functions for the RAG eval pipeline.

What:
  Recall@k, citation accuracy, keyword hit rate, plus an aggregate helper.
Why:
  Phase 1 prototype needs a small, deterministic, dependency-free
  metric layer. We deliberately avoid LLM-as-judge / RAGAS (deferred to
  Phase 2) — these four numbers are enough to detect regressions in
  retrieval quality and citation faithfulness.
Context:
  - Caller is eval/runner.py, which feeds per-case dicts in.
  - expected_sources entries are dicts {"file": str, "page_min": int,
    "page_max": int} (see EvalCase schema in interfaces/schema.py and
    eval/golden_cases.jsonl). We use page RANGES, not exact pages, so
    metrics tolerate chunker boundary drift.
  - All scalar metrics return values in [0.0, 1.0].
"""

from __future__ import annotations

from typing import Any

from jcontract.interfaces.schema import SearchResult


def _page_in_expected(
    file: str,
    page: int,
    expected_sources: list[dict[str, Any]],
) -> bool:
    """Check whether (file, page) falls in ANY expected_sources entry.

    What: filename equality + page range containment.
    Why:  expected_sources is a list because a single question may legally
          be answered from multiple distinct passages in the document set.
    """
    for src in expected_sources:
        if src.get("file") != file:
            continue
        page_min = int(src.get("page_min", 0))
        page_max = int(src.get("page_max", 0))
        if page_min <= page <= page_max:
            return True
    return False


def recall_at_k(
    retrieved: list[SearchResult],
    expected_sources: list[dict[str, Any]],
    k: int,
) -> float:
    """Binary recall@k: 1.0 if ANY top-k chunk lands in expected source range.

    What:
      Slice retrieved[:k]; for each SearchResult check whether its chunk's
      (file, page) is inside any expected_sources entry. Return 1.0 on first
      hit, else 0.0.
    Why binary, not fractional:
      Phase-1 golden cases assert "at least one relevant passage is
      retrieved", not "all passages". Fractional recall would require
      per-case ground-truth chunk-count, which we don't have. This matches
      common Recall@k usage in RAG eval (RAGAS context_recall variant).
    Edge cases:
      - k=0 → always 0.0 (no retrieval window).
      - retrieved shorter than k → only check what we have.
      - empty expected_sources → 0.0 (cannot match anything).
    """
    if k <= 0:
        return 0.0
    if not expected_sources:
        return 0.0
    for result in retrieved[:k]:
        if _page_in_expected(result.chunk.file, result.chunk.page, expected_sources):
            return 1.0
    return 0.0


def citation_accuracy(
    answer_citations: list[tuple[str, int]],
    expected_sources: list[dict[str, Any]],
) -> float:
    """Fraction of answer citations that land in expected source ranges.

    What:
      For each (file, page) tuple in answer_citations, check membership in
      expected_sources. Return hits / total.
    Why:
      Catches the "the model cited something, but the wrong page" failure
      mode. Distinct from recall@k which measures retrieval, this measures
      answer fidelity to expected provenance.
    Edge cases:
      - empty answer_citations → 0.0. We treat "no citations" as a failure
        of the citation contract (every factual sentence must cite — see
        Answerer protocol docstring). The eval surfaces it as a 0; downstream
        ssC postprocess.py is responsible for dropping uncited sentences,
        and an answer with zero citations means the answerer found nothing.
    """
    if not answer_citations:
        return 0.0
    hits = sum(
        1 for file, page in answer_citations if _page_in_expected(file, page, expected_sources)
    )
    return hits / len(answer_citations)


def keyword_hit_rate(
    answer_text: str,
    expected_keywords: list[str],
) -> float:
    """Fraction of expected keywords that appear in the answer text.

    What:
      Case-insensitive substring match for each keyword in answer_text.
      Chinese keywords match as-is (Unicode strings; .lower() is a no-op for
      CJK characters, so case folding is safe to apply uniformly).
    Why:
      Cheap, deterministic surrogate for semantic correctness. A correct
      answer about "桥梁防水" should mention at least "防水" and
      "Trackwork Contractor" or similar. False positives possible (keyword
      appears in unrelated context) but acceptable for prototype.
    Edge cases:
      - empty expected_keywords → 0.0 (defensive: nothing to measure).
      - empty answer_text → 0.0 (no possible hits).
    """
    if not expected_keywords:
        return 0.0
    text_lower = answer_text.lower()
    hits = sum(1 for kw in expected_keywords if kw.lower() in text_lower)
    return hits / len(expected_keywords)


def aggregate(results: list[dict[str, Any]]) -> dict[str, float]:
    """Compute mean of each numeric metric across per-case dicts.

    What:
      Scan `results` for keys whose values are numeric (int or float, but
      not bool — bool is a subclass of int in Python and would silently
      coerce). Return {key: mean across all cases that report it}.
    Why:
      Runner builds a list of per-case metric dicts; we expose a single
      one-shot "mean across the run" rollup. Keys we don't expect (e.g.
      "id", "category") are skipped naturally because they're strings.
    Edge cases:
      - empty results → {} (no cases, no means; caller decides how to log).
      - a metric missing on some cases → averaged over cases that DO have
        it (we don't fabricate zeros for missing measurements; answer_fn
        was likely None for those cases).
    """
    if not results:
        return {}

    sums: dict[str, float] = {}
    counts: dict[str, int] = {}
    for case in results:
        for key, value in case.items():
            # Exclude bool (subclass of int) so a stray flag doesn't average.
            if isinstance(value, bool):
                continue
            if isinstance(value, int | float):
                sums[key] = sums.get(key, 0.0) + float(value)
                counts[key] = counts.get(key, 0) + 1

    return {key: sums[key] / counts[key] for key in sums}


def aggregate_by_category(results: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    """Group per-case dicts by their ``category`` and aggregate each group.

    What:
      Returns ``{category: {metric: mean, ..., "n_cases": count}}``. Cases
      with no ``category`` key fall under ``"_uncategorized"``.
    Why (Enhancement E2 / p2-ssEval):
      The overall mean hides per-category movement. To measure the ROI of
      drawing captioning (E11) we need the recall@k of the ``drawing``
      cases specifically — run the same eval with ``--caption`` on vs off
      and compare ``metrics_by_category["drawing"]["recall_at_5"]``. The
      n_cases per group tells you how much weight to give each number.
    """
    groups: dict[str, list[dict[str, Any]]] = {}
    for case in results:
        category = case.get("category", "_uncategorized")
        key = category if isinstance(category, str) else "_uncategorized"
        groups.setdefault(key, []).append(case)

    out: dict[str, dict[str, float]] = {}
    for category, group in groups.items():
        metrics = aggregate(group)
        metrics["n_cases"] = float(len(group))
        out[category] = metrics
    return out
