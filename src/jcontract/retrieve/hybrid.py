"""Hybrid retrieval — Layer 2.

Runs the VectorStore and KeywordIndex queries in parallel and fuses
the two ranked lists with Reciprocal Rank Fusion (RRF).

Why RRF (not weighted score sum): vector cosine and BM25 scores are
on different scales; normalizing them requires per-corpus calibration.
RRF is rank-based, calibration-free, and a strong default for hybrid
retrieval (Cormack et al. 2009).
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

from jcontract.interfaces import (
    Chunk,
    Embedder,
    KeywordIndex,
    Reranker,
    SearchResult,
    VectorStore,
)

# Standard RRF k constant. Lower → top-ranked items dominate; higher → smoother.
# 60 is the value Cormack et al. recommend.
RRF_K = 60


def rrf_fuse(rankings: list[list[SearchResult]], k_constant: int = RRF_K) -> list[SearchResult]:
    """Fuse N ranked lists into one using Reciprocal Rank Fusion.

    For each unique chunk, sum 1/(k_constant + rank) across all rankings
    in which it appears. Higher fused score = better.

    Chunks present in multiple rankings naturally bubble up; chunks only
    in one list still rank by their position in that list.
    """
    fused_scores: dict[str, float] = {}
    chunks_by_id: dict[str, Chunk] = {}

    for ranking in rankings:
        for rank, result in enumerate(ranking, start=1):
            cid = result.chunk.id
            fused_scores[cid] = fused_scores.get(cid, 0.0) + 1.0 / (k_constant + rank)
            chunks_by_id[cid] = result.chunk

    ordered = sorted(fused_scores.items(), key=lambda kv: kv[1], reverse=True)
    return [SearchResult(chunk=chunks_by_id[cid], score=score) for cid, score in ordered]


class HybridRetriever:
    """Vector + keyword retrieval with RRF fusion + optional reranker.

    Pipeline:
      1. Vector + BM25 retrieve top-N (per_backend_k each) in parallel
      2. RRF fuse into one ranked list
      3. (Optional) Cross-encoder rerank top-M of the fused list
      4. Return top-k

    The reranker is OPTIONAL because:
      - At small corpus sizes (<1k chunks) RRF alone usually suffices.
      - At large corpus sizes (4k+ chunks) reranking improves Recall@5
        materially per RAG-eval literature (see reference/).
      - Reranker model load (~2GB on disk, ~30-50ms per pair on CPU)
        is a real cost users should opt into explicitly when needed.
    """

    def __init__(
        self,
        embedder: Embedder,
        vector_store: VectorStore,
        keyword_index: KeywordIndex,
        per_backend_k: int = 20,
        reranker: Reranker | None = None,
        rerank_top_n: int = 30,
    ) -> None:
        self.embedder = embedder
        self.vector_store = vector_store
        self.keyword_index = keyword_index
        # Each backend retrieves this many candidates; RRF then re-ranks
        # the union and the caller truncates to top-k.
        self.per_backend_k = per_backend_k
        # Reranker is plumbed as a Protocol so any future impl (cohere,
        # jina, custom) drops in without touching the retrieval path.
        self.reranker = reranker
        # How many candidates to feed the (expensive) reranker. Larger N
        # → better recall but ~linear slow-down. 30 is a common default.
        self.rerank_top_n = rerank_top_n

    def search(self, query: str, k: int = 5) -> list[SearchResult]:
        """Run vector and keyword retrieval in parallel, fuse, optionally rerank, return top-k."""
        # Embed the query once (cheap; we only need 1 vector).
        query_vector = self.embedder.embed([query])[0]

        # Run the two backends concurrently. ThreadPoolExecutor is fine because
        # both calls are I/O-bound (Qdrant network + BM25 in-process scan are
        # tiny CPU). max_workers=2 keeps it predictable.
        with ThreadPoolExecutor(max_workers=2) as pool:
            vec_future = pool.submit(self.vector_store.search, query_vector, self.per_backend_k)
            kw_future = pool.submit(self.keyword_index.search, query, self.per_backend_k)
            vec_results = vec_future.result()
            kw_results = kw_future.result()

        fused = rrf_fuse([vec_results, kw_results])

        if self.reranker is None:
            return fused[:k]

        # Rerank only the top-N to keep latency bounded.
        candidates_for_rerank = fused[: self.rerank_top_n]
        reranked = self.reranker.rerank(query, candidates_for_rerank)
        return reranked[:k]
