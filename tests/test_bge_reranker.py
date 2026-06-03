"""Tests for BgeReranker — Phase 1.8 P1 sub-sprint ssC.

What:
    Covers BgeReranker against the Reranker Protocol contract
    (interfaces/reranker.py). The behavioural contract is:
      1. Same number of candidates in/out
      2. Output sorted by new score (desc)
      3. Empty input → empty output, no model load
      4. Single input → single output (with new score)
      5. A relevant-but-low-ranked candidate moves up
      6. Cross-lingual (CN question vs EN chunk) works — this is the
         core differentiator vs a bi-encoder, validating the model
         choice in DECISION.

Why no mock for the cross-encoder itself:
    Mocking the model would only verify that we call ``predict()`` with
    a list of pairs — that's plumbing, not the contract that matters.
    The contract that matters is "relevant chunks rank above irrelevant
    ones", which requires the real model. We pay the ~568MB one-time
    model download in CI; subsequent runs are cached.

    The empty / single-item tests intentionally do NOT trigger the model
    load (verified by inspecting ``_model`` is None after empty input).

Performance note:
    Model loads once per pytest session via a session-scoped fixture.
    First run on a fresh machine: ~30-60s download + ~3-5s load. Cached
    runs: ~3-5s load only.
"""

from __future__ import annotations

import pytest

from jcontract.impls.bge_reranker import DEFAULT_MODEL, BgeReranker
from jcontract.interfaces.schema import Chunk, SearchResult

# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _make_result(chunk_id: str, text: str, score: float = 0.5) -> SearchResult:
    """Build a SearchResult with a minimal Chunk. Score arg is the *input*
    score — the reranker will overwrite it on its output.
    """
    chunk = Chunk(
        id=chunk_id,
        text=text,
        file="dummy.pdf",
        page=1,
        chunk_type="paragraph",
    )
    return SearchResult(chunk=chunk, score=score)


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def reranker() -> BgeReranker:
    """Single reranker per module — model download is the slow part.

    Module-scoped (not session) so other test modules don't inherit the
    state of this one's model. The model lazy-loads on first
    ``rerank()`` call regardless of fixture scope.
    """
    return BgeReranker()


# --------------------------------------------------------------------------- #
# Construction / lazy-load contract
# --------------------------------------------------------------------------- #


def test_construction_does_not_load_model() -> None:
    """Building a BgeReranker must NOT touch the network or load weights.

    Why: callers can wire this into a config-driven pipeline where the
    reranker stage is conditional; paying ~30s model download just to
    discover the user disabled it would be unacceptable.
    """
    rr = BgeReranker()
    assert rr.model_name == DEFAULT_MODEL
    # _model is the lazy slot; should be None until rerank() runs.
    assert rr._model is None


def test_rerank_handles_empty_input() -> None:
    """Empty candidate list → empty output, no model load (fresh instance)."""
    rr = BgeReranker()
    out = rr.rerank("any question", [])
    assert out == []
    # Critical: empty input must not have triggered a 30s model download.
    assert rr._model is None


# --------------------------------------------------------------------------- #
# Behavioural contract (uses the real model — slow, module-scoped fixture)
# --------------------------------------------------------------------------- #


def test_rerank_returns_same_count(reranker: BgeReranker) -> None:
    """Length-preserving: 10 in → 10 out. No silent drops."""
    candidates = [
        _make_result(f"c{i}", f"some chunk text number {i} about contracts") for i in range(10)
    ]
    out = reranker.rerank("contract clause", candidates)
    assert len(out) == 10
    # Same set of chunk ids — no fabrication, no duplication.
    assert {r.chunk.id for r in out} == {c.chunk.id for c in candidates}


def test_rerank_handles_single_item(reranker: BgeReranker) -> None:
    """Single-item input → single-item output with a new score.

    We don't assert ordering (only one slot); we assert (a) length is 1,
    (b) the chunk is the same, (c) the score is a float (the real model
    produced it — not the input 0.5 sentinel).
    """
    candidates = [_make_result("only", "Trackwork Contractor handles waterproofing", 0.5)]
    out = reranker.rerank("who handles waterproofing", candidates)
    assert len(out) == 1
    assert out[0].chunk.id == "only"
    # Score should be a finite float, and not the input sentinel.
    assert isinstance(out[0].score, float)
    # Cross-encoder logits for a clear positive match should be well above
    # the 0.5 input score; the assert below is a soft sanity check that
    # the score was actually rewritten, not a model-quality assertion.
    assert out[0].score != 0.5


def test_rerank_reorders_when_better_match_lower(reranker: BgeReranker) -> None:
    """A candidate that's obviously the right answer but at input rank 5
    must move toward rank 1 after reranking.

    Setup: 5 distractor chunks (about unrelated topics) ordered first,
    then the obvious match at index 5. With a working cross-encoder the
    match must end up at index 0 of the rerank output.
    """
    candidates = [
        _make_result("c0", "The cafeteria menu lists rice and noodles for lunch.", 0.9),
        _make_result("c1", "Tomorrow will be sunny with a chance of rain.", 0.8),
        _make_result("c2", "Football match tickets sold out within an hour.", 0.7),
        _make_result("c3", "The library closes at 9pm on weekdays.", 0.6),
        _make_result("c4", "Stock prices climbed for the third week running.", 0.5),
        _make_result(
            "c5",
            "The Trackwork Contractor is responsible for waterproofing the bridge piers.",
            0.4,
        ),
    ]
    out = reranker.rerank("Who is responsible for bridge waterproofing?", candidates)
    assert out[0].chunk.id == "c5", (
        f"Expected the waterproofing chunk at rank 1; got order {[r.chunk.id for r in out]}"
    )


def test_rerank_output_sorted_descending(reranker: BgeReranker) -> None:
    """Output must be in descending score order — the Protocol contract.

    A retriever consuming reranked results truncates to top-k; if the
    sort were ascending, top-k would silently be worst-k.
    """
    candidates = [
        _make_result("a", "Bridge waterproofing per Clause 7.3 is the trackwork scope."),
        _make_result("b", "The weather report mentions rain on Tuesday."),
        _make_result("c", "Trackwork Contractor waterproofs the bridge deck."),
        _make_result("d", "A recipe for chocolate cake calls for two eggs."),
    ]
    out = reranker.rerank("bridge waterproofing responsibility", candidates)
    scores = [r.score for r in out]
    assert scores == sorted(scores, reverse=True), f"Reranker output not sorted desc: {scores}"


def test_rerank_chinese_query_against_english_chunk(reranker: BgeReranker) -> None:
    """Cross-lingual: Chinese question must rank the English answer chunk
    above an irrelevant Chinese chunk.

    This is THE differentiator vs a bi-encoder. bge-reranker-v2-m3 is
    multilingual; the test fails if we accidentally swap to an English-
    only cross-encoder.
    """
    candidates = [
        _make_result(
            "irrelevant_cn",
            "本周天气晴朗，气温在 25 到 30 度之间，适合户外活动。",
        ),
        _make_result(
            "irrelevant_en",
            "The quarterly earnings report exceeded analyst expectations.",
        ),
        _make_result(
            "relevant_en",
            "The Trackwork Contractor shall be responsible for waterproofing "
            "at the bridge pier as set out in Clause 7.3.",
        ),
        _make_result(
            "irrelevant_mixed",
            "请参考附件 A 中的 schedule 安排相关 site visit。",
        ),
    ]
    out = reranker.rerank("桥梁防水谁负责", candidates)
    assert out[0].chunk.id == "relevant_en", (
        f"Cross-lingual rerank failed. Expected 'relevant_en' at rank 1; "
        f"got order {[r.chunk.id for r in out]} with scores "
        f"{[round(r.score, 3) for r in out]}"
    )


# --------------------------------------------------------------------------- #
# Optional slow-marked batch-size sanity (50 candidates)
# --------------------------------------------------------------------------- #


@pytest.mark.slow
def test_rerank_handles_larger_batch(reranker: BgeReranker) -> None:
    """Validate batch_size=16 path against 50 candidates — multi-batch run.

    Marked ``slow`` so default CI keeps moving but we still catch
    regressions in the batching code on the nightly / local runs.
    """
    # 49 distractors + 1 match at index 27 (mid-list to exercise batches).
    candidates = []
    for i in range(50):
        if i == 27:
            candidates.append(
                _make_result(
                    "match",
                    "Trackwork Contractor is responsible for waterproofing the "
                    "bridge piers under Clause 7.3 of the DEMO contract.",
                )
            )
        else:
            candidates.append(_make_result(f"d{i}", f"Filler chunk number {i} about other topics."))

    out = reranker.rerank("bridge waterproofing responsibility under DEMO", candidates)
    assert len(out) == 50
    assert out[0].chunk.id == "match"
