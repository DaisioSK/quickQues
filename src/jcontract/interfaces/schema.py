"""Layer 0 dataclasses shared across the system.

All ingest, retrieval, answer, and eval code agrees on these types. Vendor
implementations consume Chunk in / out; business code never reaches into
vendor-specific objects.

Per docs/project_guideline.md §3.1 (layering) and §4 (interface registry).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, cast

# Concrete vocab for chunk_type. A new value requires a project_guideline
# update + sub-sprint (interface change is contract layer).
ChunkType = Literal["qa_pair", "table", "paragraph", "drawing"]

Confidence = Literal["high", "medium", "low"]

# Page-level classification produced by vision parsers (ssCL, closes
# FORESHADOW-ls.4). "drawing" routes the page into the caption lane:
# the chunker emits a chunk_type="drawing" chunk for it, which the
# IngestPipeline's optional VisionCaptioner then captions. Parsers that
# cannot classify (pypdf — no rendered image) leave the default "text".
PageKind = Literal["text", "drawing"]


@dataclass(frozen=True)
class ParsedPage:
    """One page from a PDF, post text extraction but pre chunking.

    The parser is responsible for assembling readable text in document
    order; the chunker decides how to split it. ``tables`` is a list of
    pre-extracted tables in markdown form (empty when parser couldn't
    isolate tables — they will appear inline in ``text``).

    ``page_kind`` (ssCL) carries the vision parsers' text-vs-drawing
    classification to the chunker so drawing pages can produce
    chunk_type="drawing" chunks (the --caption lane). The "text" default
    is the zero-behaviour-change guarantee for every pre-existing
    construction site (pypdf parser, tests, snapshots): a ParsedPage
    built without the field chunks exactly as before.
    """

    page_num: int  # 1-indexed, matches what the user sees in a PDF reader
    text: str
    tables: list[str] = field(default_factory=list)
    page_kind: PageKind = "text"


@dataclass
class Chunk:
    """A retrievable unit of content with full provenance metadata.

    Every retrieval result and every answer citation traces back to one
    of these. Page numbers MUST be 1-indexed and match the source PDF.

    Mutable (frozen=False) so impls/qa_chunker.py can compose chunks
    incrementally during a single ``chunk()`` call before yielding, AND
    so the Phase 2 IngestPipeline can attach a captioner-produced
    ``caption`` field to drawing chunks before they hit the indexers.
    Treat as immutable once IngestPipeline.ingest returns.
    """

    id: str  # stable id, recommended format: "<file_stem>:<page>:<idx>"
    text: str
    file: str  # original PDF filename (e.g. "Contract DEMO(1of9) TQA.pdf")
    page: int  # 1-indexed
    chunk_type: ChunkType
    section_path: str | None = None  # e.g. "Section 7 > Clause 7.3"
    revision: str | None = None  # e.g. "Rev A" / "Revision 0"
    drawing_refs: list[str] = field(default_factory=list)
    clause_refs: list[str] = field(default_factory=list)
    question_no: str | None = None  # e.g. "ACME/TRACKWORK/16"
    # Phase 2 (sub-sprint p2-ss-prep) — VisionCaptioner-produced Chinese
    # caption for drawing chunks. Three meaningful states:
    #   None  → captioner never ran (text chunks, or drawing chunks
    #           ingested before Phase 2 went live)
    #   ""    → captioner ran but produced nothing (model error,
    #           classified as drawing but actually blank, etc.)
    #   str   → captioner produced a Chinese description; embedder +
    #           BM25 will concat this with `text` before indexing so
    #           caption participates in retrieval.
    # See DECISION-2.prep.3 in docs/dev-sprint.md for why None vs "".
    caption: str | None = None
    # bbox is a Phase 5 (UI highlight) field; parsers may leave it None.
    bbox: tuple[float, float, float, float] | None = None


@dataclass(frozen=True)
class SearchResult:
    """A single hit from any retrieval backend (vector / keyword / graph).

    ``score`` semantics differ per backend (cosine for vector, BM25 for
    keyword) — DO NOT compare across backends. The fusion layer in
    retrieve/hybrid.py uses Reciprocal Rank Fusion (RRF) instead of
    score addition.
    """

    chunk: Chunk
    score: float


@dataclass(frozen=True)
class Answer:
    """Output of the Answerer layer.

    ``citations`` is the list of (file, page) tuples extracted from the
    answer text. ``raw_context`` is what was fed to the LLM, kept for
    audit + eval downstream.
    """

    text: str
    citations: list[tuple[str, int]]
    confidence: Confidence
    raw_context: list[Chunk]


@dataclass(frozen=True)
class EvalCase:
    """A golden test case for the evaluation pipeline.

    ``expected_sources`` entries use page ranges (page_min/page_max) so
    eval is tolerant to chunker boundary shifts. ``category`` is a free
    label used to group metrics by question type.

    Schema also lives in eval/golden_cases.jsonl (one JSON object per
    line); ``from_dict()`` parses both.
    """

    id: str
    question: str
    expected_sources: list[dict[str, str | int]]  # [{"file":..,"page_min":..,"page_max":..}]
    expected_keywords: list[str]
    category: str
    # Enhancement E12: the gold/reference answer, when a golden set provides
    # one. Optional + last so existing cases (and golden_cases.jsonl lines
    # without the key) keep parsing unchanged. None = no reference answer →
    # only reference-free metrics (recall, faithfulness, relevancy) apply;
    # a non-None value unlocks the future correctness metric (DECISION-e12.3).
    expected_answer: str | None = None

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> EvalCase:
        # JSON Lines payloads come in as raw dicts; we trust the schema
        # of golden_cases.jsonl (validated by the eval runner's tests).
        raw_answer = d.get("expected_answer")
        return cls(
            id=str(d["id"]),
            question=str(d["question"]),
            expected_sources=cast(list[dict[str, str | int]], d["expected_sources"]),
            expected_keywords=cast(list[str], d["expected_keywords"]),
            category=str(d["category"]),
            expected_answer=None if raw_answer is None else str(raw_answer),
        )


def chunk_indexable_text(chunk: Chunk) -> str:
    """Return the text used by retrieval indexers (embedder + BM25).

    Phase 2 (sub-sprint p2-ssCaption) — when a VisionCaptioner has filled
    ``chunk.caption`` with a non-empty Chinese description, we concatenate
    it with the chunk's own text so caption tokens participate in both
    dense retrieval (one fused vector covers chunk + caption) and sparse
    retrieval (BM25 sees caption terms as part of the same doc).

    Concat shape: ``<text>\\n\\nCaption: <caption>``. The literal
    "Caption:" prefix is in English on purpose — it stays out of jieba's
    Chinese tokenization path and produces a clean separator the
    embedder treats as a section boundary.

    DECISION-2.cap.3 (docs/dev-sprint.md): single fused index entry per
    chunk rather than two parallel vectors (text-vec + caption-vec). Two
    vectors would double Qdrant storage + retrieve complexity for a
    prototype-stage feature whose value is still being measured. N=2
    upgrade path: if caption-only queries dominate, split into two
    vectors with max(score) fusion.

    Three input states (matches Chunk.caption docstring):
      ``None`` (captioner never ran) → text only
      ``""``  (captioner ran-empty)  → text only — empty caption adds
                                       nothing but the separator would,
                                       so we skip it
      ``str`` (real caption)         → text + caption
    """
    if chunk.caption:  # truthy: non-None AND non-empty
        return f"{chunk.text}\n\nCaption: {chunk.caption}"
    return chunk.text
