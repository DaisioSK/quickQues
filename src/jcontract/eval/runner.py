"""Eval runner: orchestrates golden cases through injected search/answer fns.

What:
  - load_golden_cases(): read JSONL → list[EvalCase].
  - run_eval(): for each case, call injected search_fn (and optionally
    answer_fn), compute per-case metrics, then dump a timestamped JSON
    report to data/eval-results/.
Why:
  - Decouple "what we measure" (metrics.py) from "what we test against"
    (real retriever / answerer in retrieve/ and impls/). The runner accepts
    callables so tests can inject stubs and the integrator can inject the
    real hybrid retriever + claude_answerer.
  - JSON output is meant to be diff-friendly across runs; we keep it
    sorted-key + pretty-printed for human review.
Context:
  - search_fn signature: (question: str) -> list[SearchResult].
  - answer_fn signature: (question: str, results: list[SearchResult]) -> Answer.
  - The runner does NOT swallow exceptions from injected fns — if the
    retriever crashes on one case, the whole run aborts. Phase-1 is
    deterministic enough that a partial run is more confusing than helpful.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from jcontract.eval.metrics import (
    aggregate,
    aggregate_by_category,
    citation_accuracy,
    keyword_hit_rate,
    recall_at_k,
)
from jcontract.interfaces.judge import Judge
from jcontract.interfaces.schema import Answer, EvalCase, SearchResult

SearchFn = Callable[[str], list[SearchResult]]
AnswerFn = Callable[[str, list[SearchResult]], Answer]


def load_golden_cases(path: Path) -> list[EvalCase]:
    """Parse a JSONL file into a list of EvalCase dataclasses.

    What:
      One JSON object per non-blank line; lines starting with '#' are
      treated as comments and skipped (defensive — we don't currently emit
      comments, but golden_cases.jsonl is human-edited).
    Why:
      JSONL chosen over a single JSON array because (a) trivial to grep
      and diff per case, (b) easy to append a new case without touching
      list brackets, (c) streamable if the corpus grows. See DECISION
      report.
    """
    cases: list[EvalCase] = []
    with path.open(encoding="utf-8") as fh:
        for raw_line in fh:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            cases.append(EvalCase.from_dict(json.loads(line)))
    return cases


def _evaluate_case(
    case: EvalCase,
    search_fn: SearchFn,
    answer_fn: AnswerFn | None,
    judge: Judge | None = None,
) -> dict[str, Any]:
    """Run one case through retrieval (+ optional answer + optional judge).

    Returns a flat dict suitable for aggregate() — all metric keys are
    floats; identifying keys (id, category, question) are strings.
    """
    # Pull a top-10 window once; recall@5 and recall@10 both work off it.
    # Why request 10 and slice down: avoids two separate retrieval calls
    # when the underlying impl supports a single k=10 query cheaply.
    retrieved = search_fn(case.question)

    per_case: dict[str, Any] = {
        "id": case.id,
        "category": case.category,
        "question": case.question,
        "recall_at_5": recall_at_k(retrieved, case.expected_sources, k=5),
        "recall_at_10": recall_at_k(retrieved, case.expected_sources, k=10),
        "n_retrieved": len(retrieved),
    }

    if answer_fn is not None:
        ans = answer_fn(case.question, retrieved)
        per_case["citation_accuracy"] = citation_accuracy(ans.citations, case.expected_sources)
        per_case["keyword_hit_rate"] = keyword_hit_rate(ans.text, case.expected_keywords)
        per_case["n_citations"] = len(ans.citations)
        per_case["confidence"] = ans.confidence
        per_case["answer_text"] = ans.text

        # LLM-as-judge answer-quality metrics (E12, reference-free). We
        # grade against ans.raw_context — the exact chunks the answerer
        # saw — for faithfulness, and the question for relevancy. A NaN
        # score means the judge failed; we DON'T record it (so it's
        # excluded from the aggregate, not counted as 0 — and never
        # written as invalid `NaN` JSON).
        if judge is not None:
            faith = judge.faithfulness(ans.text, ans.raw_context)
            if not math.isnan(faith.score):
                per_case["faithfulness"] = faith.score
            relevancy = judge.answer_relevancy(case.question, ans.text)
            if not math.isnan(relevancy.score):
                per_case["answer_relevancy"] = relevancy.score

    return per_case


def run_eval(
    cases: list[EvalCase],
    search_fn: SearchFn | None = None,
    answer_fn: AnswerFn | None = None,
    output_dir: Path = Path("data/eval-results"),
    judge: Judge | None = None,
) -> dict[str, Any]:
    """Run the full eval suite and write a timestamped JSON report.

    What:
      Iterate cases → per-case metrics → aggregate mean → dump JSON.
    Why search_fn is required, answer_fn is optional:
      Retrieval is the first thing a Phase-1 prototype needs to validate;
      answering is gated on retrieval working. Allowing answer_fn=None
      lets us evaluate raw retrieval before the answerer (or the API key)
      is wired up.
    Why fail-fast on missing search_fn:
      A None search_fn means the integrator forgot to wire the real
      retriever. Silently producing an empty report would mask the bug.
    Returns:
      The same dict that gets written to disk, so callers (tests + the
      integrator script) can assert on it without re-reading the file.
    """
    if search_fn is None:
        raise ValueError(
            "run_eval requires a search_fn — pass the real retriever or a test stub. "
            "See eval/runner.py docstring for the expected signature."
        )

    per_case = [_evaluate_case(case, search_fn, answer_fn, judge) for case in cases]
    metrics_mean = aggregate(per_case)
    metrics_by_category = aggregate_by_category(per_case)

    timestamp = datetime.now(UTC).strftime("%Y-%m-%dT%H-%M-%SZ")
    report: dict[str, Any] = {
        "timestamp": timestamp,
        "n_cases": len(cases),
        "metrics_mean": metrics_mean,
        # Per-category rollup so drawing-case recall is separable from the
        # overall mean — the lever for measuring caption ROI (E2 / E11).
        "metrics_by_category": metrics_by_category,
        "per_case": per_case,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{timestamp}.json"
    # default=_json_default handles dataclasses that may sneak in via
    # answer_fn return values (e.g. raw_context Chunks).
    with output_path.open("w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2, default=_json_default, sort_keys=True)

    return report


def _json_default(obj: Any) -> Any:
    """Best-effort JSON serializer for dataclasses sneaking into the report.

    Why:
      Stub answer_fn implementations in tests may return Answer with
      raw_context Chunks; json.dump would otherwise crash. dataclasses.asdict
      handles both Answer and Chunk recursively.
    """
    try:
        return asdict(obj)
    except TypeError:
        # Fall back to str() for anything truly opaque — better a string
        # than a runtime crash that hides the real failure.
        return str(obj)
