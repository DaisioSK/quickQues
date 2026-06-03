"""VisionCaptioner Protocol + DrawingCaption — Layer 0 (Phase 2).

Finalized by sub-sprint p2-ss-prep. Was a placeholder Protocol shape in
Phase 1; now a concrete dataclass for the output type plus a refined
Protocol for the captioner itself.

Why this exists:
- Drawing-type chunks (engineering schematics on a single PDF page)
  need a richer textual representation than raw OCR. The vision parser
  already extracts visible labels via DRAWING_CAPTION_PROMPT, but for
  cross-lingual retrieval we want a concise Chinese caption ("这是一张
  DEMO 桥梁防水构造图...") plus an entity list (Drawing No., clauses,
  key terms) so caption text participates in both vector and BM25
  retrieval alongside the original chunk text.
- The Captioner runs AFTER the chunker has identified drawing chunks
  and AFTER the parser has rendered the page. We pass image_bytes
  rather than a Path so the captioner is decoupled from filesystem
  state — see DECISION-2.prep.1 in docs/dev-sprint.md.

Default impl: impls/claude_vision_captioner.py (sub-sprint p2-ssCaption).
Replacement candidates per docs/project_guideline.md §4:
  - Qwen2-VL (self-hosted alt — FORESHADOW, N=2)
  - GPT-4o (Phase 7+ if cost trade-off changes)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol


@dataclass(frozen=True)
class DrawingCaption:
    """Output of a VisionCaptioner: Chinese caption + entity list.

    DECISION-2.prep.2 (docs/dev-sprint.md): single-language `caption_zh`
    plus a flat `entities` list — no English caption, no separate
    `drawing_refs` vs `other_entities` split. Rationale: the user-facing
    target is Chinese; the chunker's regex layer already pulls Drawing
    No. and Clause refs into Chunk.drawing_refs / clause_refs from the
    OCR'd text, so separating entity buckets here would duplicate that
    work. If a future phase needs an English caption (cross-language
    indexing, multi-tenant) it lands as a N=2 upgrade.

    Empty `caption_zh` plus empty `entities` is the canonical "captioner
    ran but produced nothing useful" state — distinct from
    `Chunk.caption is None` ("captioner never ran"). See DECISION-2.prep.3.
    """

    caption_zh: str
    entities: list[str] = field(default_factory=list)


class VisionCaptioner(Protocol):
    """Caption an image (drawing page extracted from a PDF) for retrieval.

    Args:
        image_bytes: Rendered page image bytes. Caller is responsible for
            the encoding (typically JPEG @ 150 DPI from pypdfium2 — same
            shape ClaudeVisionParser already produces). Captioner does
            NOT re-render; it consumes the bytes directly.
        nearby_text: Surrounding chunk text from the same page (already
            OCR'd by the parser). Lets the captioner ground its output
            against terms in scope ("dimension 50mm" makes sense only if
            you know what's being measured).

    Returns:
        A DrawingCaption. Impls MUST NOT raise on a single-page failure
        (model error, malformed JSON output, image too small) — return
        an empty DrawingCaption(caption_zh="", entities=[]) instead so
        ingest does not lose the page. Caller writes the empty caption
        to chunk.caption = "" (distinguishable from None = never ran).
    """

    def caption(self, image_bytes: bytes, nearby_text: str) -> DrawingCaption: ...
