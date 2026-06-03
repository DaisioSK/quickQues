"""BM25-backed KeywordIndex implementation.

What:
    In-memory ``rank_bm25.BM25Okapi`` index with jieba-based tokenization.
    Provides sparse / keyword retrieval as the second leg of the Phase 1
    hybrid retriever (the first leg being QdrantStore dense vectors).

Why jieba for ALL text (DECISION):
    The j-contract corpus is bilingual: clause text is English, but many
    queries (and some annotations) are Chinese. We need a tokenizer that
    splits Chinese without spaces. The two viable paths were:

      (a) language-detect then dispatch: heavy, fragile on mixed-language
          sentences ("桥梁 waterproofing 责任方"), one more dep (langdetect).
      (b) jieba on everything: jieba's ``cut`` treats whitespace-separated
          ASCII tokens as their own tokens (verified: "waterproofing at
          pier" → ["waterproofing", " ", "at", " ", "pier"]; we filter
          whitespace and punctuation post-tokenize). Cost: a single call
          per doc, ~10µs for short text. Benefit: one code path, robust on
          code-mixed input.

    We chose (b). The "8-question dep gate" (dev-contract/24-domain-deps-
    env.md): jieba is (1) already in pyproject, (2) MIT licensed, (3) pure
    Python ~3MB, (4) actively maintained, (5) the de-facto standard for
    Chinese segmentation, (6) no native deps, (7) deterministic, (8) used
    by Qdrant's own Chinese sample notebooks. Pass.

Why in-memory (DECISION):
    Phase 1 corpus is one PDF (~ few hundred chunks at most). Persisting
    BM25 state to disk adds I/O complexity for zero prototype benefit.
    Phase 2 will swap in a persistent backend (Whoosh / Tantivy / bge-m3
    sparse) — the KeywordIndex Protocol absorbs that.

Why raw BM25 scores (no normalization):
    BM25 scores are unbounded and scale-dependent on corpus stats. The
    retrieve/hybrid.py layer (integrator) uses Reciprocal Rank Fusion
    which only cares about rank ordering, not score magnitude. Returning
    raw scores keeps debugging honest (low BM25 means actually weak
    keyword overlap, not a normalization artifact).
"""

from __future__ import annotations

import string
from typing import ClassVar

# Why type: ignore on both: jieba and rank_bm25 ship no py.typed marker.
# The KeywordIndex Protocol boundary here is fully typed, so untyped
# third-parties only leak at these imports; suppress narrowly per
# project guideline §"no blanket ignores".
import jieba  # type: ignore[import-untyped]
from rank_bm25 import BM25Okapi  # type: ignore[import-untyped]

from jcontract.interfaces.schema import Chunk, SearchResult, chunk_indexable_text

# Punctuation set spans ASCII + common CJK marks. Anything in here is
# dropped from tokens before BM25 sees them — it skews term frequency
# otherwise (a clause heavy on commas would dominate).
_PUNCT = set(string.punctuation) | {
    "，",
    "。",
    "、",
    "；",
    "：",
    "！",
    "？",
    "「",
    "」",
    "『",
    "』",
    "（",
    "）",
    "【",
    "】",
    "《",
    "》",
    "“",
    "”",
    "‘",
    "’",
}


def _tokenize(text: str) -> list[str]:
    """Tokenize bilingual text with jieba, lower-casing for ASCII tokens.

    Returns words and CJK characters/phrases as separate tokens; strips
    whitespace and punctuation. Lower-casing is unicode-safe (no-op for
    CJK, normalizes English so "Waterproofing" matches "waterproofing").
    """
    tokens: list[str] = []
    # cut_all=False (default) gives "accurate mode" — best for retrieval.
    for raw in jieba.cut(text, cut_all=False):
        tok = raw.strip().lower()
        if not tok:
            continue
        if all(ch in _PUNCT for ch in tok):
            continue
        tokens.append(tok)
    return tokens


class Bm25Index:
    """In-memory BM25 keyword index. Implements ``KeywordIndex`` Protocol.

    Notes on idempotency:
      ``add()`` re-adding a Chunk with the same id REPLACES the prior
      entry. This matches the Protocol contract ("must support re-adding
      the same Chunk id idempotently"). Re-add triggers a single index
      rebuild — fine at prototype scale; Phase 2 may switch to an
      incremental backend if corpus grows past ~10k chunks.
    """

    backend: ClassVar[str] = "rank_bm25+jieba"

    def __init__(self) -> None:
        # Two parallel lists keep the implementation transparent:
        #   _chunks[i] is the Chunk; _tokens[i] is its tokenization.
        # BM25Okapi consumes _tokens and produces scores in index order,
        # which we then map back to chunks.
        self._chunks: list[Chunk] = []
        self._tokens: list[list[str]] = []
        self._bm25: BM25Okapi | None = None
        # id → list-position lookup for fast idempotent re-adds.
        self._id_index: dict[str, int] = {}

    def add(self, chunks: list[Chunk]) -> None:
        if not chunks:
            return
        for chunk in chunks:
            # Phase 2: chunk_indexable_text folds Chinese caption into the
            # indexable string when present. Text-only chunks pass through
            # unchanged so existing tokenizations stay stable.
            toks = _tokenize(chunk_indexable_text(chunk))
            if chunk.id in self._id_index:
                pos = self._id_index[chunk.id]
                self._chunks[pos] = chunk
                self._tokens[pos] = toks
            else:
                self._id_index[chunk.id] = len(self._chunks)
                self._chunks.append(chunk)
                self._tokens.append(toks)
        # BM25Okapi requires at least one token per doc; the empty-text
        # edge case (parser emits a blank chunk) would otherwise crash on
        # division by zero inside rank_bm25. Replace empties with a single
        # sentinel token that won't match real queries.
        safe_tokens = [t if t else ["__empty__"] for t in self._tokens]
        self._bm25 = BM25Okapi(safe_tokens)

    def search(self, query: str, k: int) -> list[SearchResult]:
        if self._bm25 is None or not self._chunks:
            return []
        query_tokens = _tokenize(query)
        if not query_tokens:
            return []
        scores = self._bm25.get_scores(query_tokens)
        # argsort descending; take top-k. Iterate Python-side to avoid
        # pulling numpy into the interface boundary (return type is
        # list[SearchResult] of plain floats).
        ranked = sorted(
            range(len(scores)),
            key=lambda i: float(scores[i]),
            reverse=True,
        )[:k]
        return [SearchResult(chunk=self._chunks[i], score=float(scores[i])) for i in ranked]
