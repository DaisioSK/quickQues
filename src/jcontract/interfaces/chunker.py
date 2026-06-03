"""Chunker Protocol — Layer 0.

Default impl: impls/qa_chunker.py (Phase 1 S1.1 ssA) — structure-aware
chunker that recognises Q&A pairs, tables, drawings, and falls back to
paragraph splits.

A Chunker accepts ParsedPage output (from a PDFParser) and produces
Chunk objects with full metadata + extracted Drawing/Clause references.
"""

from __future__ import annotations

from typing import Protocol

from .schema import Chunk, ParsedPage


class Chunker(Protocol):
    """Turn a parsed PDF into retrievable Chunk units.

    Contract:
      - ``file`` is propagated into every returned Chunk.file (display name,
        not absolute path — keep relative or basename so citations are
        portable across machines).
      - Chunks MUST preserve page numbers (Chunk.page) for citation grounding.
      - Chunk.id MUST be unique within a single document and stable across
        re-ingestion of the same input (so re-runs upsert into vector store
        deterministically).
      - For Q&A pairs, ``question_no`` should be populated when extractable.
    """

    def chunk(self, pages: list[ParsedPage], file: str) -> list[Chunk]: ...
