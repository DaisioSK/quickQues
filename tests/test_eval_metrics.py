"""Unit tests for eval/metrics.py and eval/runner.py.

Covers:
  - recall_at_k: hit / miss / page boundaries / wrong file
  - citation_accuracy: full / partial / zero / empty
  - keyword_hit_rate: full / partial / case-insensitive / Chinese unicode
  - aggregate: mean across cases, ignores non-numeric, ignores bool
  - runner: stubbed search/answer fns, JSON file written to tmp_path,
    structural assertions on the report
  - golden_cases.jsonl: parses cleanly and has >= 6 cases covering the
    required categories
"""

from __future__ import annotations

import json
from pathlib import Path

from jcontract.eval.metrics import (
    aggregate,
    aggregate_by_category,
    citation_accuracy,
    keyword_hit_rate,
    recall_at_k,
)
from jcontract.eval.runner import load_golden_cases, run_eval
from jcontract.interfaces.judge import JudgeScore
from jcontract.interfaces.schema import Answer, Chunk, EvalCase, SearchResult

# ---------- helpers ----------


def _make_chunk(file: str, page: int, idx: int = 0, text: str = "x") -> Chunk:
    return Chunk(
        id=f"{file}:{page}:{idx}",
        text=text,
        file=file,
        page=page,
        chunk_type="paragraph",
    )


def _make_result(file: str, page: int, score: float = 0.9) -> SearchResult:
    return SearchResult(chunk=_make_chunk(file, page), score=score)


# ---------- recall_at_k ----------


def test_recall_at_k_hit_first_position() -> None:
    retrieved = [_make_result("DEMO.pdf", 10), _make_result("DEMO.pdf", 999)]
    expected = [{"file": "DEMO.pdf", "page_min": 1, "page_max": 50}]
    assert recall_at_k(retrieved, expected, k=5) == 1.0


def test_recall_at_k_miss() -> None:
    retrieved = [_make_result("DEMO.pdf", 200)]
    expected = [{"file": "DEMO.pdf", "page_min": 1, "page_max": 50}]
    assert recall_at_k(retrieved, expected, k=5) == 0.0


def test_recall_at_k_page_boundary_min() -> None:
    # page == page_min must be a hit (inclusive bound).
    retrieved = [_make_result("DEMO.pdf", 1)]
    expected = [{"file": "DEMO.pdf", "page_min": 1, "page_max": 50}]
    assert recall_at_k(retrieved, expected, k=5) == 1.0


def test_recall_at_k_page_boundary_max() -> None:
    # page == page_max must be a hit (inclusive bound).
    retrieved = [_make_result("DEMO.pdf", 50)]
    expected = [{"file": "DEMO.pdf", "page_min": 1, "page_max": 50}]
    assert recall_at_k(retrieved, expected, k=5) == 1.0


def test_recall_at_k_outside_window() -> None:
    # Hit exists in retrieved but past the k window — should miss.
    retrieved = [
        _make_result("DEMO.pdf", 999),
        _make_result("DEMO.pdf", 998),
        _make_result("DEMO.pdf", 10),
    ]
    expected = [{"file": "DEMO.pdf", "page_min": 1, "page_max": 50}]
    assert recall_at_k(retrieved, expected, k=2) == 0.0
    # Widening k to 3 includes the real hit.
    assert recall_at_k(retrieved, expected, k=3) == 1.0


def test_recall_at_k_wrong_file_same_page() -> None:
    retrieved = [_make_result("OTHER.pdf", 10)]
    expected = [{"file": "DEMO.pdf", "page_min": 1, "page_max": 50}]
    assert recall_at_k(retrieved, expected, k=5) == 0.0


def test_recall_at_k_empty_expected() -> None:
    retrieved = [_make_result("DEMO.pdf", 10)]
    assert recall_at_k(retrieved, [], k=5) == 0.0


def test_recall_at_k_k_zero() -> None:
    retrieved = [_make_result("DEMO.pdf", 10)]
    expected = [{"file": "DEMO.pdf", "page_min": 1, "page_max": 50}]
    assert recall_at_k(retrieved, expected, k=0) == 0.0


# ---------- citation_accuracy ----------


def test_citation_accuracy_full() -> None:
    citations = [("DEMO.pdf", 12), ("DEMO.pdf", 30)]
    expected = [{"file": "DEMO.pdf", "page_min": 1, "page_max": 50}]
    assert citation_accuracy(citations, expected) == 1.0


def test_citation_accuracy_partial() -> None:
    citations = [("DEMO.pdf", 12), ("DEMO.pdf", 999)]
    expected = [{"file": "DEMO.pdf", "page_min": 1, "page_max": 50}]
    assert citation_accuracy(citations, expected) == 0.5


def test_citation_accuracy_zero() -> None:
    citations = [("DEMO.pdf", 999)]
    expected = [{"file": "DEMO.pdf", "page_min": 1, "page_max": 50}]
    assert citation_accuracy(citations, expected) == 0.0


def test_citation_accuracy_empty_citations() -> None:
    expected = [{"file": "DEMO.pdf", "page_min": 1, "page_max": 50}]
    assert citation_accuracy([], expected) == 0.0


def test_citation_accuracy_wrong_file() -> None:
    citations = [("OTHER.pdf", 12)]
    expected = [{"file": "DEMO.pdf", "page_min": 1, "page_max": 50}]
    assert citation_accuracy(citations, expected) == 0.0


# ---------- keyword_hit_rate ----------


def test_keyword_hit_rate_full() -> None:
    text = "桥梁防水由 Trackwork Contractor 负责"
    keywords = ["防水", "Trackwork Contractor"]
    assert keyword_hit_rate(text, keywords) == 1.0


def test_keyword_hit_rate_partial() -> None:
    text = "桥梁防水的讨论"
    keywords = ["防水", "Trackwork Contractor", "waterproofing"]
    # Only 防水 hits → 1/3.
    assert abs(keyword_hit_rate(text, keywords) - 1 / 3) < 1e-9


def test_keyword_hit_rate_case_insensitive() -> None:
    text = "WATERPROOFING is the responsibility of the Trackwork Contractor"
    keywords = ["waterproofing", "trackwork contractor"]
    assert keyword_hit_rate(text, keywords) == 1.0


def test_keyword_hit_rate_unicode_chinese() -> None:
    text = "依据 Clause 7.3，桥梁防水由承包商负责。"
    keywords = ["防水", "承包商", "Clause 7.3"]
    assert keyword_hit_rate(text, keywords) == 1.0


def test_keyword_hit_rate_empty_keywords() -> None:
    assert keyword_hit_rate("any text", []) == 0.0


def test_keyword_hit_rate_empty_text() -> None:
    # Empty text means no possible hits.
    assert keyword_hit_rate("", ["防水"]) == 0.0


# ---------- aggregate ----------


def test_aggregate_returns_mean() -> None:
    results = [
        {"id": "q1", "recall_at_5": 1.0, "recall_at_10": 1.0},
        {"id": "q2", "recall_at_5": 0.0, "recall_at_10": 1.0},
        {"id": "q3", "recall_at_5": 0.5, "recall_at_10": 0.0},
    ]
    out = aggregate(results)
    assert abs(out["recall_at_5"] - 0.5) < 1e-9
    assert abs(out["recall_at_10"] - 2 / 3) < 1e-9
    # 'id' is non-numeric → not aggregated.
    assert "id" not in out


def test_aggregate_empty() -> None:
    assert aggregate([]) == {}


def test_aggregate_ignores_bool() -> None:
    # bool is subclass of int; we must NOT silently average it.
    results = [
        {"recall_at_5": 1.0, "flag": True},
        {"recall_at_5": 0.0, "flag": False},
    ]
    out = aggregate(results)
    assert "recall_at_5" in out
    assert "flag" not in out


def test_aggregate_missing_metric_skipped() -> None:
    # answer_fn was None for q2 → no citation_accuracy. Average only over q1.
    results = [
        {"id": "q1", "recall_at_5": 1.0, "citation_accuracy": 0.5},
        {"id": "q2", "recall_at_5": 0.0},
    ]
    out = aggregate(results)
    assert abs(out["recall_at_5"] - 0.5) < 1e-9
    assert abs(out["citation_accuracy"] - 0.5) < 1e-9


def test_aggregate_by_category_groups_and_counts() -> None:
    results = [
        {"id": "d1", "category": "drawing", "recall_at_5": 1.0},
        {"id": "d2", "category": "drawing", "recall_at_5": 0.0},
        {"id": "p1", "category": "procedure", "recall_at_5": 1.0},
    ]
    out = aggregate_by_category(results)
    assert set(out) == {"drawing", "procedure"}
    assert abs(out["drawing"]["recall_at_5"] - 0.5) < 1e-9
    assert out["drawing"]["n_cases"] == 2.0
    assert abs(out["procedure"]["recall_at_5"] - 1.0) < 1e-9
    assert out["procedure"]["n_cases"] == 1.0


def test_aggregate_by_category_handles_missing_category() -> None:
    out = aggregate_by_category([{"id": "x", "recall_at_5": 1.0}])
    assert "_uncategorized" in out
    assert out["_uncategorized"]["n_cases"] == 1.0


def test_aggregate_by_category_empty() -> None:
    assert aggregate_by_category([]) == {}


# ---------- runner ----------


def test_run_eval_requires_search_fn() -> None:
    case = EvalCase(
        id="q001",
        question="?",
        expected_sources=[{"file": "x.pdf", "page_min": 1, "page_max": 10}],
        expected_keywords=["k"],
        category="responsibility",
    )
    try:
        run_eval([case], search_fn=None)
    except ValueError as e:
        assert "search_fn" in str(e)
    else:
        raise AssertionError("expected ValueError when search_fn is None")


def test_run_eval_writes_json_and_returns_structure(tmp_path: Path) -> None:
    case = EvalCase(
        id="q001",
        question="桥梁防水的责任方是谁？",
        expected_sources=[{"file": "DEMO.pdf", "page_min": 1, "page_max": 50}],
        expected_keywords=["防水", "Trackwork Contractor"],
        category="responsibility",
    )

    # Stub search_fn returns a list with one in-range hit.
    def stub_search(_q: str) -> list[SearchResult]:
        return [_make_result("DEMO.pdf", 12), _make_result("DEMO.pdf", 999)]

    # Stub answer_fn returns an Answer with one correct cite + one keyword hit.
    def stub_answer(_q: str, results: list[SearchResult]) -> Answer:
        return Answer(
            text="桥梁防水由 Trackwork Contractor 负责。",
            citations=[("DEMO.pdf", 12)],
            confidence="high",
            raw_context=[r.chunk for r in results],
        )

    report = run_eval(
        [case],
        search_fn=stub_search,
        answer_fn=stub_answer,
        output_dir=tmp_path,
    )

    # In-memory structure assertions.
    assert report["n_cases"] == 1
    assert "timestamp" in report
    assert "metrics_mean" in report
    # E2: per-category rollup present and groups this case under its category.
    assert "metrics_by_category" in report
    assert report["metrics_by_category"]["responsibility"]["n_cases"] == 1.0
    assert "per_case" in report
    assert len(report["per_case"]) == 1
    pc = report["per_case"][0]
    assert pc["id"] == "q001"
    assert pc["recall_at_5"] == 1.0
    assert pc["recall_at_10"] == 1.0
    assert pc["citation_accuracy"] == 1.0
    assert pc["keyword_hit_rate"] == 1.0

    # On-disk assertions.
    written = list(tmp_path.glob("*.json"))
    assert len(written) == 1
    with written[0].open(encoding="utf-8") as fh:
        from_disk = json.load(fh)
    assert from_disk["n_cases"] == 1
    assert from_disk["per_case"][0]["id"] == "q001"
    # Unicode answer text should round-trip without escaping.
    assert "桥梁防水" in from_disk["per_case"][0]["answer_text"]


def test_run_eval_without_answer_fn(tmp_path: Path) -> None:
    # When answer_fn is None, only retrieval metrics should be present.
    case = EvalCase(
        id="q002",
        question="?",
        expected_sources=[{"file": "DEMO.pdf", "page_min": 1, "page_max": 50}],
        expected_keywords=["k"],
        category="responsibility",
    )

    def stub_search(_q: str) -> list[SearchResult]:
        return [_make_result("DEMO.pdf", 10)]

    report = run_eval([case], search_fn=stub_search, output_dir=tmp_path)
    pc = report["per_case"][0]
    assert pc["recall_at_5"] == 1.0
    assert "citation_accuracy" not in pc
    assert "keyword_hit_rate" not in pc


# ---------- golden cases file ----------


def test_golden_cases_parse_and_cover_categories() -> None:
    # Path is relative to the package, not cwd, so the test is robust.
    path = Path(__file__).parent.parent / "src" / "jcontract" / "eval" / "golden_cases.jsonl"
    cases = load_golden_cases(path)
    assert len(cases) >= 6
    categories = {c.category for c in cases}
    required = {
        "responsibility",
        "reference",
        "definition",
        "procedure",
        "quantity",
        "revision",
        # Enhancement E2 (p2-ssEval): drawing-only cases for caption ROI.
        "drawing",
    }
    assert required.issubset(categories), f"missing categories: {required - categories}"
    # E2 deliverable: at least 5 drawing cases so the drawing-category
    # recall@k rollup is meaningful when comparing --caption on vs off.
    drawing_cases = [c for c in cases if c.category == "drawing"]
    assert len(drawing_cases) >= 5
    # Each case has non-empty question + keywords + at least one source.
    for c in cases:
        assert c.question.strip()
        assert c.expected_keywords
        assert c.expected_sources
        for src in c.expected_sources:
            assert "file" in src
            assert int(src["page_min"]) >= 1
            assert int(src["page_max"]) >= int(src["page_min"])


# ---------- E12: optional expected_answer + judge wiring ----------


def test_eval_case_expected_answer_optional() -> None:
    # Lines without the key parse to None (backward compatible).
    without = EvalCase.from_dict(
        {
            "id": "a",
            "question": "q",
            "expected_sources": [{"file": "f.pdf", "page_min": 1, "page_max": 1}],
            "expected_keywords": ["k"],
            "category": "definition",
        }
    )
    assert without.expected_answer is None
    # A golden line that includes it round-trips as a string.
    with_ans = EvalCase.from_dict(
        {
            "id": "b",
            "question": "q",
            "expected_sources": [{"file": "f.pdf", "page_min": 1, "page_max": 1}],
            "expected_keywords": ["k"],
            "category": "definition",
            "expected_answer": "理想答案",
        }
    )
    assert with_ans.expected_answer == "理想答案"


class _StubJudge:
    """Judge stub returning fixed scores; one NaN to test exclusion."""

    def __init__(self, faith: float, relevancy: float) -> None:
        self._faith = faith
        self._relevancy = relevancy

    def faithfulness(self, answer: str, context: list[Chunk]) -> JudgeScore:
        return JudgeScore(score=self._faith, reasoning="stub")

    def answer_relevancy(self, question: str, answer: str) -> JudgeScore:
        return JudgeScore(score=self._relevancy, reasoning="stub")


def test_run_eval_records_judge_metrics(tmp_path: Path) -> None:
    case = EvalCase(
        id="q1",
        question="谁负责防水?",
        expected_sources=[{"file": "DEMO.pdf", "page_min": 1, "page_max": 50}],
        expected_keywords=["防水"],
        category="responsibility",
    )

    def stub_search(_q: str) -> list[SearchResult]:
        return [_make_result("DEMO.pdf", 12)]

    def stub_answer(_q: str, results: list[SearchResult]) -> Answer:
        return Answer(
            text="防水由 Trackwork Contractor 负责。",
            citations=[("DEMO.pdf", 12)],
            confidence="high",
            raw_context=[r.chunk for r in results],
        )

    report = run_eval(
        [case],
        search_fn=stub_search,
        answer_fn=stub_answer,
        output_dir=tmp_path,
        judge=_StubJudge(faith=0.8, relevancy=0.9),
    )
    pc = report["per_case"][0]
    assert pc["faithfulness"] == 0.8
    assert pc["answer_relevancy"] == 0.9
    assert "faithfulness" in report["metrics_mean"]


def test_run_eval_excludes_nan_judge_score(tmp_path: Path) -> None:
    case = EvalCase(
        id="q1",
        question="q",
        expected_sources=[{"file": "DEMO.pdf", "page_min": 1, "page_max": 50}],
        expected_keywords=["k"],
        category="definition",
    )

    def stub_search(_q: str) -> list[SearchResult]:
        return [_make_result("DEMO.pdf", 12)]

    def stub_answer(_q: str, results: list[SearchResult]) -> Answer:
        return Answer(text="ans", citations=[], confidence="low", raw_context=[])

    # Faithfulness NaN (judge failed) → excluded; relevancy present.
    report = run_eval(
        [case],
        search_fn=stub_search,
        answer_fn=stub_answer,
        output_dir=tmp_path,
        judge=_StubJudge(faith=float("nan"), relevancy=0.7),
    )
    pc = report["per_case"][0]
    assert "faithfulness" not in pc  # NaN dropped, not written as invalid JSON
    assert pc["answer_relevancy"] == 0.7
