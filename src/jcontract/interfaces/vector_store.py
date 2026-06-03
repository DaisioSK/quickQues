"""VectorStore Protocol — Layer 0.

Default impl: impls/qdrant_store.py (Phase 1 S1.1 ssB).
Replacement candidates per docs/project_guideline.md §4:
  - pgvector (when Postgres is the canonical DB)
  - Chroma (lighter, dev-only)

The store is responsible for persistence; the embedder is not.
"""

from __future__ import annotations

from typing import Protocol

from .schema import Chunk, SearchResult


class VectorStore(Protocol):
    """Dense-vector retrieval backend.

    Contract:
      - ``add()`` upserts: re-adding a Chunk with the same id replaces it.
      - ``search()`` returns SearchResult ordered by descending similarity.
      - ``count()`` returns the total point count for the active collection.
      - Implementations own collection creation lazily — the first ``add``
        creates the collection sized from the first batch's vector length.
    """

    def add(self, chunks: list[Chunk], vectors: list[list[float]]) -> None: ...

    def search(self, query_vector: list[float], k: int) -> list[SearchResult]: ...

    def count(self) -> int: ...
