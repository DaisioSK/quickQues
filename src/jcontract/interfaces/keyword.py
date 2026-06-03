"""KeywordIndex Protocol — Layer 0.

Default impl: impls/bm25_index.py (Phase 1 S1.1 ssB) — in-memory rank_bm25
with jieba tokenization for Chinese queries.

Replacement candidates per docs/project_guideline.md §4:
  - bge-m3 sparse embeddings (deferred to Phase 2 to keep prototype slim)
  - Whoosh / Tantivy (disk-persistent BM25)

Important: SearchResult.score from a KeywordIndex is on a different scale
than a VectorStore's. The retrieve/hybrid.py layer uses RRF (reciprocal
rank fusion), not linear score combination.
"""

from __future__ import annotations

from typing import Protocol

from .schema import Chunk, SearchResult


class KeywordIndex(Protocol):
    """BM25-style sparse keyword retrieval.

    Contract:
      - ``add()`` appends chunks into the index. Implementations may choose
        in-memory only (Phase 1) or persistent (later) — but must support
        re-adding the same Chunk id idempotently.
      - ``search()`` returns top-k by raw BM25 score, descending.
    """

    def add(self, chunks: list[Chunk]) -> None: ...

    def search(self, query: str, k: int) -> list[SearchResult]: ...
