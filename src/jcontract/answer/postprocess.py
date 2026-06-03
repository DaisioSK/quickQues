"""Citation parsing, validation, and confidence scoring (Phase 1 S1.1 ssC).

What
----
Pure functions that run AFTER the LLM returns text:
  * ``parse_citations``   — regex-extract ``[file p.N]`` tuples.
  * ``validate_citations`` — drop sentences with NO citation OR with
    fabricated citations (file/page not in the context we fed the LLM).
  * ``compute_confidence`` — bucket retrieval-score signal into the
    Confidence Literal defined in interfaces/schema.py.

Why
---
Per the Answerer Protocol contract, the impl (not the integrator) is
responsible for enforcing citation hygiene. Keeping the enforcement in
pure functions means:
  (a) we can unit-test fabrication-rejection without invoking the SDK;
  (b) future Answerer impls (DeepSeek, Qwen) reuse the same logic by
      importing these helpers.

Context
-------
Citation grammar (what we instruct the model to emit, per
``answer/prompt.py``):

    [<filename> p.<page_number>]

  - Filename may contain spaces, parentheses, dots (e.g. ``Contract DEMO(1of9) TQA.pdf``).
  - Page is a non-negative integer.
  - The bracket is always the LAST token of a factual sentence.

Sub-sprint: p1-s1-ssC.  Mode: High-Risk.
"""

from __future__ import annotations

import re

from jcontract.interfaces.schema import Chunk, Confidence

# Citation regex.
#
# Anatomy:
#   \[          literal "["
#   ([^\[\]]+?) filename: any chars except brackets, non-greedy
#               (lets us match nested punctuation like "TQA.pdf" but
#               not accidentally swallow a later "[" if model emits two)
#   \s+p\.      literal " p."  (allows one-or-more whitespace, matches the
#               prompt's example exactly while being lenient about extra spaces)
#   (\d+)       page number — captured as group 2
#   \]          literal "]"
#
# We do NOT anchor to sentence-end; ``parse_citations`` is purely a
# scanner. ``validate_citations`` handles sentence-level enforcement.
_CITATION_RE = re.compile(r"\[([^\[\]]+?)\s+p\.(\d+)\]")

# Sentence splitter.
#
# We split on Chinese sentence terminators (。！？) and Western (.!?),
# keeping the terminator with the preceding sentence so citation-at-end
# detection works. Newlines are also treated as soft splits because the
# model sometimes emits one bullet per line without a terminator.
#
# Why not nltk / spacy: those are 200MB+ deps for a 4-line regex job;
# this regex handles the bilingual contract-Q&A style well enough and
# is empirically validated by tests.
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[。！？.!?])\s+|\n+")

# Confidence thresholds — match the spec in §ssC Execution Plan step 2.
# Top-5 mean retrieval score is the input signal; the integrator passes
# in whatever similarity scores its retrieval stack produces (cosine for
# vector, normalized RRF for fusion).
_CONFIDENCE_HIGH_MIN = 0.7
_CONFIDENCE_MEDIUM_MIN = 0.5
_TOP_K_FOR_CONFIDENCE = 5


def parse_citations(text: str) -> list[tuple[str, int]]:
    """Extract all ``[filename p.N]`` citations from ``text``.

    Returns each match as ``(filename, page)``. Preserves order and
    duplicates — callers that want a deduped set can use ``set(...)``.

    Examples:
        >>> parse_citations("桥梁防水由 Trackwork Contractor 负责 [TQA p.12]。")
        [('TQA', 12)]
        >>> parse_citations("a [F.pdf p.1]. b [F.pdf p.2]. c [F.pdf p.1].")
        [('F.pdf', 1), ('F.pdf', 2), ('F.pdf', 1)]
    """
    return [(m.group(1).strip(), int(m.group(2))) for m in _CITATION_RE.finditer(text)]


def _split_sentences(text: str) -> list[str]:
    """Split text into sentences, preserving terminator and trimming whitespace."""
    parts = _SENTENCE_SPLIT_RE.split(text.strip())
    return [p.strip() for p in parts if p.strip()]


def validate_citations(
    text: str,
    context_chunks: list[Chunk],
) -> tuple[str, list[tuple[str, int]], int]:
    """Drop sentences that violate citation rules.

    A sentence is dropped if EITHER:
      - it contains NO ``[filename p.N]`` citation, OR
      - every citation in it refers to a ``(file, page)`` pair NOT
        present in ``context_chunks``.

    A sentence with a mix of valid + fabricated citations is KEPT —
    the valid one(s) anchor it, but the fabricated tuples are removed
    from the returned citation list. (Dropping the whole sentence in
    that mixed case would discard real information; tests cover this.)

    Special case — exact fallback string ``文档中未明确说明``: kept as-is
    with no citations (matches Answerer Protocol contract).

    Args:
        text:           Raw LLM output.
        context_chunks: The chunks fed to the LLM. Citation tuples in
                        the answer must reference a (file, page) pair
                        in this list.

    Returns:
        ``(cleaned_text, valid_citations, n_dropped_sentences)``.
        ``valid_citations`` preserves order and may contain duplicates,
        same as ``parse_citations``.
    """
    from jcontract.answer.prompt import FALLBACK_NO_ANSWER

    # Canonical no-answer passthrough. We compare on stripped text so a
    # trailing newline doesn't break the check.
    if text.strip() == FALLBACK_NO_ANSWER:
        return FALLBACK_NO_ANSWER, [], 0

    valid_pairs = {(c.file, c.page) for c in context_chunks}

    kept_sentences: list[str] = []
    kept_citations: list[tuple[str, int]] = []
    n_dropped = 0

    for sentence in _split_sentences(text):
        cites_in_sentence = parse_citations(sentence)
        if not cites_in_sentence:
            # Rule: factual sentences MUST cite. Drop.
            n_dropped += 1
            continue

        valid_in_sentence = [c for c in cites_in_sentence if c in valid_pairs]
        if not valid_in_sentence:
            # All citations fabricated → drop the whole sentence.
            n_dropped += 1
            continue

        # Sentence has at least one real citation: keep the sentence,
        # but only record the citations that actually point to context.
        kept_sentences.append(sentence)
        kept_citations.extend(valid_in_sentence)

    # Empty result: nothing survived enforcement. Return the canonical
    # fallback rather than empty string so the contract holds.
    if not kept_sentences:
        return FALLBACK_NO_ANSWER, [], n_dropped

    cleaned = " ".join(kept_sentences)
    return cleaned, kept_citations, n_dropped


def compute_confidence(top_scores: list[float]) -> Confidence:
    """Bucket retrieval scores into a Confidence label.

    Rule (matches §ssC Execution Plan step 2):
      mean(top-5 scores) > 0.7  → "high"
      mean(top-5 scores) > 0.5  → "medium"
      else                       → "low"

    If fewer than 5 scores are provided, the mean is computed over what
    is available. Empty list → "low" (no signal = no confidence).
    """
    if not top_scores:
        return "low"

    sample = top_scores[:_TOP_K_FOR_CONFIDENCE]
    mean = sum(sample) / len(sample)

    if mean > _CONFIDENCE_HIGH_MIN:
        return "high"
    if mean > _CONFIDENCE_MEDIUM_MIN:
        return "medium"
    return "low"
