"""Ingest pipeline — Layer 1.

Wires PDFParser -> Chunker -> Embedder -> (VectorStore, KeywordIndex).
Also persists the chunk list to a JSONL snapshot so the in-memory
KeywordIndex can be rehydrated across CLI invocations (Qdrant is
already persistent; BM25 lives in memory).
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import structlog

from jcontract.interfaces import (
    Chunk,
    Chunker,
    Embedder,
    KeywordIndex,
    PDFParser,
    RefGraph,
    VectorStore,
    VisionCaptioner,
)
from jcontract.interfaces.schema import chunk_indexable_text

logger = structlog.get_logger(__name__)


class IngestPipeline:
    """Single-document ingest orchestration with chunk snapshot persistence."""

    def __init__(
        self,
        parser: PDFParser,
        chunker: Chunker,
        embedder: Embedder,
        vector_store: VectorStore,
        keyword_index: KeywordIndex,
        chunks_snapshot_path: Path,
        ref_graph: RefGraph | None = None,
        captioner: VisionCaptioner | None = None,
    ) -> None:
        self.parser = parser
        self.chunker = chunker
        self.embedder = embedder
        self.vector_store = vector_store
        self.keyword_index = keyword_index
        # JSONL file where ingested Chunk dicts are appended. Used by the
        # CLI on subsequent invocations to rehydrate the in-memory BM25
        # index without re-running the (slow) ingest pipeline.
        self.chunks_snapshot_path = chunks_snapshot_path
        # Optional cross-document entity mention index (RefGraph). When
        # provided, every ingested chunk's drawing_refs / clause_refs /
        # question_no / section / revision are indexed for exact-match
        # lookup via `jcontract refs <type> <value>`.
        self.ref_graph = ref_graph
        # Optional Phase 2 VisionCaptioner. When provided, drawing-type
        # chunks get a Chinese caption attached before indexing so caption
        # text participates in both vector and BM25 retrieval (see
        # interfaces.schema.chunk_indexable_text + DECISION-2.cap.3).
        # When None, drawing chunks keep caption=None — current behaviour.
        self.captioner = captioner

    def ingest(self, pdf_path: Path) -> int:
        """Run the full pipeline for one PDF; return number of chunks indexed."""
        logger.info("ingest.start", pdf=str(pdf_path))

        pages = self.parser.parse(pdf_path)
        logger.info("ingest.parsed", pages=len(pages), pdf=pdf_path.name)

        chunks = self.chunker.chunk(pages, pdf_path.name)
        logger.info("ingest.chunked", chunks=len(chunks), pdf=pdf_path.name)

        if not chunks:
            logger.warning("ingest.empty", pdf=pdf_path.name)
            return 0

        # Phase 2 (sub-sprint p2-ssCaption) — when a captioner is wired,
        # attach a Chinese caption to each drawing chunk BEFORE embedding
        # so caption text gets fused into the chunk's indexed
        # representation (see chunk_indexable_text). Captioner failures
        # never raise per Protocol contract; they yield empty captions
        # which we record as "" (distinguishable from None = never ran).
        if self.captioner is not None:
            self._attach_captions(chunks, pdf_path)

        # Embed all chunks in one batch — Embedder impls handle batching
        # internally. chunk_indexable_text folds caption into the text
        # when a caption is present (text-only chunks pass through
        # unchanged so existing embeddings stay stable).
        vectors = self.embedder.embed([chunk_indexable_text(c) for c in chunks])
        logger.info("ingest.embedded", n=len(vectors), dim=self.embedder.dim)

        # Index into both backends. Bm25Index also reads chunk.caption
        # via chunk_indexable_text internally, so caption participates
        # in keyword search too.
        self.vector_store.add(chunks, vectors)
        self.keyword_index.add(chunks)
        logger.info("ingest.indexed", vector_count=self.vector_store.count())

        # Cross-document entity index for "which docs cite Drawing X" type queries.
        if self.ref_graph is not None:
            self.ref_graph.index(chunks)
            stats = self.ref_graph.stats()
            logger.info("ingest.ref_graph_updated", **stats)

        # Persist chunks for cross-invocation BM25 rehydration.
        self._append_snapshot(chunks)
        logger.info("ingest.snapshot_written", path=str(self.chunks_snapshot_path))

        return len(chunks)

    def _attach_captions(self, chunks: list[Chunk], pdf_path: Path) -> None:
        """Mutate drawing-type chunks to add a Chinese caption.

        Implementation choices:
          - Local import of render_page_to_jpeg keeps pypdfium2/PIL out
            of the top-level pipeline imports for users who don't enable
            the captioner.
          - We render each drawing page on demand (not bulk pre-render)
            so we don't pay for pages we end up not needing. Memoize
            within this single ingest call to avoid double-rendering
            when two drawing chunks share a page (rare today — chunker
            usually emits one drawing chunk per page — but cheap insurance).
          - Captioner errors are silent per Protocol contract; we still
            set chunk.caption = "" so the snapshot/state distinguishes
            "ran but empty" from "never ran" (None).
        """
        from jcontract.impls.claude_vision_captioner import render_page_to_jpeg

        # Per-page render cache scoped to this ingest call only.
        page_image_cache: dict[int, bytes] = {}
        drawing_count = 0
        for chunk in chunks:
            if chunk.chunk_type != "drawing":
                continue
            drawing_count += 1
            if chunk.page not in page_image_cache:
                page_image_cache[chunk.page] = render_page_to_jpeg(pdf_path, chunk.page)
            image_bytes = page_image_cache[chunk.page]
            # Captioner returns DrawingCaption; we only need caption_zh
            # for the chunk field. entities is left as a FORESHADOW —
            # could feed into chunk.drawing_refs (N=2 upgrade once we
            # see whether the model's entity extraction is reliable).
            cap = self.captioner.caption(image_bytes, chunk.text)  # type: ignore[union-attr]
            chunk.caption = cap.caption_zh  # may be "" when API failed
        if drawing_count:
            logger.info(
                "ingest.captioned",
                pdf=pdf_path.name,
                drawing_chunks=drawing_count,
                unique_pages=len(page_image_cache),
            )

    def _append_snapshot(self, chunks: list[Chunk]) -> None:
        self.chunks_snapshot_path.parent.mkdir(parents=True, exist_ok=True)
        with self.chunks_snapshot_path.open("a", encoding="utf-8") as fh:
            for c in chunks:
                fh.write(json.dumps(asdict(c), ensure_ascii=False) + "\n")


def load_chunks_snapshot(path: Path) -> list[Chunk]:
    """Rehydrate the chunk list from the JSONL snapshot. Empty if file missing."""
    if not path.exists():
        return []
    out: list[Chunk] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            # bbox arrives as list[float] from JSON; coerce back to tuple.
            if d.get("bbox") is not None:
                d["bbox"] = tuple(d["bbox"])
            out.append(Chunk(**d))
    return out
