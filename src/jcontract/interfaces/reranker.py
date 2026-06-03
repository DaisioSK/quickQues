"""Reranker Protocol — Layer 0 (deferred per plan Phase 3 S3.2).

Placeholder. Production target: bge-reranker-v2-m3 (multilingual cross
encoder). Phase 1 prototype skips reranking; if hybrid Recall@k turns
out poor on the golden eval, this gets re-prioritised.

Kept in the registry so future impls land against a stable contract.
"""

from __future__ import annotations

from typing import Protocol

from .schema import SearchResult


class Reranker(Protocol):
    """Re-score (question, chunk) pairs with a stronger cross-encoder.

    Implementations take the fused candidate list (top-N from hybrid
    retrieval) and return them reordered with new scores. They MAY drop
    candidates below a threshold; that policy lives in the impl, not in
    the caller.
    """

    def rerank(self, question: str, candidates: list[SearchResult]) -> list[SearchResult]: ...
