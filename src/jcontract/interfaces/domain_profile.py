"""DomainProfile + StructureSpec — Layer 0 (Phase 7).

Why this exists:
- j-contract is a DOMAIN-AGNOSTIC document knowledge base (see the 定位声明
  at the top of docs/project_guideline.md). The only things that vary by
  domain are: (1) the OCR / caption / answer-framing PROMPTS, (2) the
  CHUNKER's structural regexes (Q&A boundaries, cross-reference rules,
  section headers), and (3) display copy (suggested questions). Everything
  else — parse → chunk → retrieve → answer → eval — is domain-neutral.
- A DomainProfile bundles exactly those domain-variable pieces so the core
  pipeline depends on this Protocol-level object, never on construction
  vocabulary. Construction (project DEMO) is the `contract` profile; `document`
  is the neutral default that works on arbitrary PDFs (financial reports,
  specs, manuals). Adding a domain = adding a `profiles/<name>.yaml`.

Storage / selection:
- Bodies live in `profiles/<name>.yaml`; `impls/domain_profile_registry.py`
  `load_profile(name)` parses + validates them into these dataclasses.
- A profile is bound per-collection (a `data/<collection>/profile.txt`
  sidecar, wired in later sub-sprints); the default is `contract`.

These dataclasses are frozen value objects (no behaviour). The chunker's
regex *flags* are NOT stored here — the chunker applies its canonical
flags (Q&A + headers: IGNORECASE|MULTILINE; ref rules: IGNORECASE),
matching the original construction implementation byte-for-byte.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class RefRule:
    """One cross-reference extraction rule for the chunker.

    ``pattern`` is a regex with a single capturing group; every match's
    group(1) is collected (sorted, de-duped) into the Chunk field named
    ``target_field`` (e.g. "drawing_refs", "clause_refs"). Applied with
    IGNORECASE by the chunker.
    """

    pattern: str
    target_field: str


@dataclass(frozen=True)
class StructureSpec:
    """The domain-variable structural rules consumed by the chunker.

    A fully-neutral spec (``qa_block_pattern=None``, empty ``ref_rules``,
    None header patterns) makes the chunker fall back to pure paragraph
    splitting — the right behaviour for an arbitrary document with no
    known structure. The `contract` profile reproduces the original
    construction regexes verbatim.
    """

    # Regex whose group(1) is the question id; marks a Q&A block start.
    # None → the document has no Q&A structure (paragraph chunks only).
    qa_block_pattern: str | None = None
    # Cross-reference extractors (e.g. Drawing No., Clause refs).
    ref_rules: tuple[RefRule, ...] = ()
    # Header patterns (group(1) = header text) used to build section_path.
    section_header_pattern: str | None = None
    clause_header_pattern: str | None = None


@dataclass(frozen=True)
class DomainProfile:
    """Everything that varies by document domain.

    The core pipeline (ingest / chunk / answer prompts) reads its
    domain-specific strings + structure from here, so no module hardcodes
    a domain. See module docstring + docs/project_guideline.md §4.
    """

    name: str
    # First framing sentence of the answer system prompt (the rest of the
    # rules are domain-neutral and live in answer/prompt.py).
    answer_framing: str
    # Vision OCR prompt for text-heavy pages.
    ocr_text_prompt: str
    # Vision prompt for drawing/figure pages.
    ocr_drawing_prompt: str
    # VisionCaptioner prompt (JSON-only caption of a figure/drawing).
    caption_prompt: str
    # Chunker structural rules.
    structure: StructureSpec
    # Empty-state example questions shown in the UI.
    suggested_questions: tuple[str, ...] = field(default_factory=tuple)
