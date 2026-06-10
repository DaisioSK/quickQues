"""Qdrant-backed VectorStore implementation.

What:
    Thin wrapper around qdrant-client that satisfies the ``VectorStore``
    Protocol. Lazily creates the collection on first ``add()`` so callers
    don't have to know the embedding dim ahead of time.

Why Qdrant (vs Chroma, pgvector):
    - Phase 1 prototype runs a single docker-compose service (already
      wired in repo root). No DB schema migrations needed for prototype.
    - Native cosine + HNSW + payload filtering — enough for hybrid +
      future metadata filters (file/page/section).
    - bge-m3 sparse vectors land in Phase 2 → Qdrant's named-vector API
      will absorb that without breaking the collection.

Why deterministic UUID5 ids (DECISION):
    Qdrant point ids must be unsigned int OR UUID string. Our domain
    ids are like ``"Contract DEMO(1of9) TQA.pdf:12:3"`` (file:page:idx)
    which isn't UUID-shaped. Hashing via uuid5 with a fixed namespace
    gives us:
        - stable: same Chunk.id → same UUID across runs (re-adds are
          true upserts, not duplicates).
        - reversible: original Chunk.id round-trips via the payload, so
          downstream code never sees the synthetic UUID.

Why Cosine distance (DECISION):
    The default mpnet model normalizes outputs to unit length, so
    Cosine ≡ Dot in practice. We pick Cosine for clarity and so a future
    non-normalized model doesn't silently produce wrong scores.

Context:
    Phase 1 S1.1 ssB. Score returned in SearchResult is the raw cosine
    similarity from Qdrant (range roughly [-1, 1], typically [0, 1] for
    e5/mpnet families). The retrieve/hybrid.py layer (integrator) fuses
    this with BM25 scores via Reciprocal Rank Fusion — DO NOT normalize
    here.
"""

from __future__ import annotations

import uuid
from dataclasses import asdict
from typing import Any, ClassVar

from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse
from qdrant_client.models import Distance, PointStruct, VectorParams

from jcontract.config import load_app_config
from jcontract.interfaces.schema import Chunk, ChunkType, SearchResult

# Fixed namespace for deriving point UUIDs from Chunk.id. The exact value
# is arbitrary but MUST be stable: changing it invalidates every existing
# collection. uuid4() output captured once for this repo.
_ID_NAMESPACE = uuid.UUID("8a2f5d8e-7e1c-4d6a-9b3f-1a2b3c4d5e6f")

# Max points per upsert request. 256 × 1024-dim float32 vectors + payload
# ≈ 1-2 MB per request — comfortably inside the REST timeout even on slow
# (WSL2) disk; whole-volume single-request upserts timed out.
# [DECISION-ab3.61 dev-sprint v3 §13]
_UPSERT_BATCH_SIZE = 256


def _point_uuid(chunk_id: str) -> str:
    """Map a domain Chunk.id to a deterministic UUID string."""
    return str(uuid.uuid5(_ID_NAMESPACE, chunk_id))


def _chunk_to_payload(chunk: Chunk) -> dict[str, Any]:
    """Serialize a Chunk into a JSON-safe dict for Qdrant payload."""
    # asdict handles the dataclass fully; tuple bbox becomes a list,
    # which is fine — _payload_to_chunk converts back.
    return asdict(chunk)


def _payload_to_chunk(payload: dict[str, Any]) -> Chunk:
    """Reconstruct a Chunk from a Qdrant payload dict.

    Why we re-cast bbox to tuple: Chunk.bbox is typed as
    ``tuple[float,float,float,float] | None`` and JSON-roundtrip turns
    tuples into lists. Restoring the tuple keeps downstream code's
    static typing honest.
    """
    bbox_raw = payload.get("bbox")
    bbox: tuple[float, float, float, float] | None
    if bbox_raw is None:
        bbox = None
    else:
        bbox = (
            float(bbox_raw[0]),
            float(bbox_raw[1]),
            float(bbox_raw[2]),
            float(bbox_raw[3]),
        )
    # chunk_type is a Literal; cast through str to satisfy mypy.
    chunk_type: ChunkType = payload["chunk_type"]
    return Chunk(
        id=str(payload["id"]),
        text=str(payload["text"]),
        file=str(payload["file"]),
        page=int(payload["page"]),
        chunk_type=chunk_type,
        section_path=payload.get("section_path"),
        revision=payload.get("revision"),
        drawing_refs=list(payload.get("drawing_refs") or []),
        clause_refs=list(payload.get("clause_refs") or []),
        question_no=payload.get("question_no"),
        bbox=bbox,
    )


class QdrantStore:
    """Qdrant-backed dense-vector store. Implements the ``VectorStore`` Protocol.

    Constructor params:
      collection_name: logical name; one collection per (project, embedder).
      url: override for QDRANT_URL env (test injection).
      api_key: optional, for managed Qdrant Cloud; local docker leaves this None.
      distance: Cosine by default; see module docstring for rationale.
    """

    backend: ClassVar[str] = "qdrant"

    def __init__(
        self,
        collection_name: str = "contract",
        *,
        url: str | None = None,
        api_key: str | None = None,
        distance: Distance = Distance.COSINE,
    ) -> None:
        cfg = load_app_config()
        self._collection = collection_name
        self._distance = distance
        # Why route through config.py: keeps the env-var contract in one
        # place (project_guideline.md §6.1). Direct kwarg overrides win
        # for tests that need to point at an ephemeral instance.
        # What: explicit REST timeout (default is just 5s).
        # Why:  large ``wait=True`` upserts (hundreds of points × 1024-dim
        #       vectors) exceed 5s under WSL2 disk I/O — observed
        #       ResponseHandlingException("timed out") on a 625-page volume
        #       (2026-06-10 full-corpus ingest, [DECISION-ab3.61]).
        self._client = QdrantClient(
            url=url or cfg.qdrant_url,
            api_key=api_key,
            timeout=120,
        )

    @property
    def collection_name(self) -> str:
        return self._collection

    def _ensure_collection(self, vector_size: int) -> None:
        """Create the collection on first write; idempotent on subsequent calls."""
        if self._client.collection_exists(self._collection):
            return
        self._client.create_collection(
            collection_name=self._collection,
            vectors_config=VectorParams(size=vector_size, distance=self._distance),
        )

    def add(self, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        if len(chunks) != len(vectors):
            raise ValueError(f"chunks ({len(chunks)}) and vectors ({len(vectors)}) length mismatch")
        if not chunks:
            return
        self._ensure_collection(vector_size=len(vectors[0]))
        points = [
            PointStruct(
                id=_point_uuid(chunk.id),
                vector=vec,
                payload=_chunk_to_payload(chunk),
            )
            for chunk, vec in zip(chunks, vectors, strict=True)
        ]
        # wait=True: prototype prioritizes correctness over throughput.
        # Eval needs to ``add → search`` in the same test without a race.
        # Why batched: a whole-volume upsert (thousands of points, tens of
        # MB) in one request times out even with a generous client timeout;
        # 256-point batches keep each request small and make progress
        # incremental. [DECISION-ab3.61 dev-sprint v3 §13]
        for start in range(0, len(points), _UPSERT_BATCH_SIZE):
            self._client.upsert(
                collection_name=self._collection,
                points=points[start : start + _UPSERT_BATCH_SIZE],
                wait=True,
            )

    def search(self, query_vector: list[float], k: int) -> list[SearchResult]:
        try:
            hits = self._client.search(
                collection_name=self._collection,
                query_vector=query_vector,
                limit=k,
                with_payload=True,
            )
        except UnexpectedResponse as e:
            # Collection might not exist yet — return empty rather than crash
            # the retrieval pipeline. Caller can detect via empty list +
            # store.count() == 0.
            if "doesn't exist" in str(e) or "Not found" in str(e):
                return []
            raise
        results: list[SearchResult] = []
        for h in hits:
            if h.payload is None:
                # Defensive: a point without payload shouldn't exist in our
                # write path, but skip it rather than crash search.
                continue
            results.append(SearchResult(chunk=_payload_to_chunk(h.payload), score=float(h.score)))
        return results

    def count(self) -> int:
        if not self._client.collection_exists(self._collection):
            return 0
        return int(self._client.count(collection_name=self._collection, exact=True).count)

    # ---- test helpers (not part of the Protocol) ----

    def _drop(self) -> None:
        """Delete the collection. Used only by integration tests for cleanup."""
        if self._client.collection_exists(self._collection):
            self._client.delete_collection(self._collection)
