"""Pydantic request/response models for the HTTP API.

Why these are distinct from Layer 0 dataclasses (`interfaces.schema`):
- Layer 0 types serve internal contracts (chunker → embedder → retriever
  → answerer). They evolve with internal refactors.
- Pydantic API schemas serve the HTTP contract. They evolve with client
  needs (web UI today, possibly third-party clients later).
- Separating them lets us add internal fields without breaking the API
  surface, and vice-versa. DECISION-5.api.2 in dev-sprint.md.

Why we keep the schemas minimal for MVP:
- Sub-sprint p5-s1-ssApi MVP-cut explicitly drops trace_summary,
  agent_path, and streaming-related fields. Anything beyond
  `answer + citations + confidence` is YAGNI for the first cut.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    """User question payload.

    `max_length=1000` is a defense-in-depth limit per
    docs/project_guideline.md §6.2 (untrusted-input boundary). The
    answerer's prompt template would also clip very long inputs, but
    rejecting at the API edge gives a clean 422 instead of a runtime
    surprise.
    """

    question: str = Field(min_length=1, max_length=1000)


class CitationOut(BaseModel):
    """A (file, page) pair extracted from the answer or from retrieval.

    Used by the frontend to render clickable citation chips that link
    to the PDF viewer route. The `page` is 1-indexed and matches what
    the user sees in a PDF reader (the Chunk schema enforces this
    invariant upstream).
    """

    file: str
    page: int


# `confidence` widens Layer 0's Confidence (`high | medium | low`) with
# a `"none"` case meaning "no answerer was configured" (e.g.
# ANTHROPIC_API_KEY missing). The frontend can render a banner like
# "retrieval-only mode" in that case.
ApiConfidence = Literal["high", "medium", "low", "none"]


class AskResponse(BaseModel):
    """Answer + supporting citations.

    Graceful-degradation contract:
    - When no answerer is configured → `answer` carries a fallback
      message and `confidence="none"`; `citations` still populated from
      retrieval so the UI is useful.
    - When the index is empty → `answer="(no documents indexed yet — \
run jcontract ingest first)"`, `citations=[]`, `confidence="none"`.
    - Otherwise normal Claude/DeepSeek-shaped output.
    """

    answer: str
    citations: list[CitationOut] = Field(default_factory=list)
    confidence: ApiConfidence


class SearchResultOut(BaseModel):
    """Single retrieval hit for the debug `/search` endpoint.

    Why a separate shape from AskResponse: `/search` shows the raw
    retrieval surface (one entry per matched chunk) without going
    through the answerer. Lets the frontend or a debugging human inspect
    "what would the LLM have seen" without paying for the LLM call.
    """

    file: str
    page: int
    chunk_type: str
    score: float
    preview: str  # first ~200 chars of chunk.text, for human-readable triage


class HealthResponse(BaseModel):
    """`/healthz` payload."""

    status: Literal["ok"]
    qdrant_count: int
