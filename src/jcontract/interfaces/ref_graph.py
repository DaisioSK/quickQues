"""RefGraph Protocol — Layer 0.

Cross-reference graph: a project-specific structure that maps Drawing No.
/ Clause / Question No. / Section / Revision tokens extracted from chunks
into an entity-mention index. This is the differentiator for
construction-contract retrieval (heavily cross-referenced documents) —
vector + BM25 retrieval can miss exact-token references that an
embedder didn't catch; the RefGraph guarantees exact-match recall by
entity, complementing fuzzy retrieval.

Phase 1.8 lands the first concrete impl (SqliteRefGraph) for
cross-document entity mention queries:

    "Drawing T/PRJ/CWD/WS/2101A — which docs / pages cite it?"
    "Question No. ACME/TRACKWORK/16 — where else does it appear?"
    "Clause 7.3 — which Q&A pairs reference it?"
"""

from __future__ import annotations

from typing import Protocol

from .schema import Chunk


class RefGraph(Protocol):
    """Index and query entity-mention edges across chunks.

    Concrete impls (e.g. ``impls/sqlite_ref_graph.py``) are responsible
    for persistence; this Protocol only fixes the verbs.

    DECISION-1.8-ssE: Two operational methods (``stats``, ``close``) were
    added to the Protocol vs the original Phase-2-placeholder draft. They
    let the eval pipeline assert non-empty indexing and let the CLI close
    DB connections deterministically. Keeping them on the Protocol means
    every impl (including a hypothetical future in-memory or Neo4j one)
    must answer to the same operational shape.
    """

    def index(self, chunks: list[Chunk]) -> None:
        """Insert/update entity mentions for these chunks. MUST be idempotent."""

    def mentions_of(self, entity_type: str, entity_value: str) -> list[Chunk]:
        """Return all chunks where this entity is mentioned.

        Impls MAY return minimal Chunk objects (text may be empty) — the
        caller can use (file, page) to fetch the full chunk from the
        vector store if needed.
        """

    def entities_in(self, chunk_id: str) -> list[tuple[str, str]]:
        """Return list of ``(entity_type, entity_value)`` tuples for a chunk."""

    def stats(self) -> dict[str, int]:
        """Return counts: at least ``chunks``, ``entities``, ``mentions``."""

    def close(self) -> None:
        """Release backend resources (DB connection, files, etc.)."""
