"""Answerer Protocol — Layer 0.

Default impl: impls/claude_answerer.py (Phase 1 S1.1 ssC, High-Risk Mode).
Replacement candidates per docs/project_guideline.md §4:
  - DeepSeek-V3 (Phase 4 S4.2 — cost fallback)
  - Qwen2.5-72B (self-hosted alt)

The Answerer is responsible for end-to-end production of a grounded
Chinese answer with verifiable citations, given a question and a list
of retrieved Chunks. Citation enforcement (drop fabricated cites, drop
sentences with no cite) is the impl's job; integrator code does not
post-process answers.
"""

from __future__ import annotations

from typing import Protocol

from .schema import Answer, Chunk


class Answerer(Protocol):
    """Generate a citation-bound answer from retrieved context.

    Contract:
      - The answer text MUST be in Chinese (user-facing language).
      - Every factual sentence MUST end with at least one [filename p.X]
        citation; sentences without citations are dropped by the impl.
      - Citation (file, page) tuples MUST refer to chunks in ``context``;
        impls must validate and drop fabricated citations before returning.
      - If the impl cannot answer from context, it returns the canonical
        fallback "文档中未明确说明" with confidence=low and empty citations.
      - Implementations must NOT log secret key material or full prompt
        bodies that may contain customer-supplied PDF content.
    """

    def answer(self, question: str, context: list[Chunk]) -> Answer: ...
