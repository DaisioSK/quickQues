"""Embedder Protocol — Layer 0.

Default impl: impls/fastembed_embedder.py (Phase 1 S1.1 ssB) — uses
fastembed (ONNX) with a multilingual e5 model (Chinese + English).

Replacement candidates per docs/project_guideline.md §4:
  - bge-m3 (BAAI) — Phase 2 target, supports dense + sparse + multi-vector
  - OpenAI text-embedding-3-small (API, no local infra)
"""

from __future__ import annotations

from typing import Protocol


class Embedder(Protocol):
    """Map text into dense vectors.

    Contract:
      - ``embed()`` must accept a batch and return vectors in the same order.
      - ``dim`` reports the vector dimensionality (used by VectorStore to
        size its collection at creation time).
      - The same text MUST produce the same vector across calls (no
        randomness; needed for caching + reproducible eval).
    """

    @property
    def dim(self) -> int: ...

    def embed(self, texts: list[str]) -> list[list[float]]: ...
