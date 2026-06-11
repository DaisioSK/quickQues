"""Layer 0: Protocol abstractions for all swappable components.

Per docs/project_guideline.md §4. Re-exports the 10 core protocols + the
shared schema dataclasses so business code can do a single import.

Active (Phase 1 + Phase 1.5..1.10 prototypes):
  - PDFParser, Chunker, Embedder, VectorStore, KeywordIndex, Answerer,
    Reranker (Phase 1.8), RefGraph (Phase 1.8)

Phase 2 (finalized contracts, impls upcoming):
  - VisionCaptioner + DrawingCaption (sub-sprint p2-ss-prep)
  - OCREngine + OCRBlock (sub-sprint p2-ss-prep)
"""

from __future__ import annotations

from .answerer import Answerer
from .chunker import Chunker
from .domain_profile import DomainProfile, RefRule, StructureSpec
from .embedding import Embedder
from .judge import Judge, JudgeScore
from .keyword import KeywordIndex
from .ocr import OCRBlock, OCREngine
from .parser import PDFParser
from .redactor import RedactionResult, Redactor
from .ref_graph import RefGraph
from .reranker import Reranker
from .schema import (
    Answer,
    Chunk,
    ChunkType,
    Confidence,
    EvalCase,
    PageKind,
    ParsedPage,
    SearchResult,
)
from .vector_store import VectorStore
from .vision import DrawingCaption, VisionCaptioner

__all__ = [
    # Schema
    "Answer",
    "Chunk",
    "ChunkType",
    "Confidence",
    "DomainProfile",
    "DrawingCaption",
    "EvalCase",
    "OCRBlock",
    "PageKind",
    "ParsedPage",
    "RefRule",
    "SearchResult",
    "StructureSpec",
    # Active protocols
    "Answerer",
    "Chunker",
    "Embedder",
    "Judge",
    "JudgeScore",
    "KeywordIndex",
    "OCREngine",
    "PDFParser",
    "RedactionResult",
    "Redactor",
    "RefGraph",
    "Reranker",
    "VectorStore",
    "VisionCaptioner",
]
