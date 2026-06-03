"""Q&A-aware structure chunker (default Chunker impl).

What:
    Takes ``list[ParsedPage]`` and emits ``list[Chunk]`` carrying
    page-grounded provenance, structural metadata (section_path,
    question_no), and extracted cross-references (drawing_refs,
    clause_refs).

Why structure-aware (not naive sliding window):
    Construction contract TQA (Technical Query & Answer) documents are
    organised as Q&A pairs whose answers can reference Drawings and
    Clauses elsewhere in the contract. A naive overlap-window chunker
    would split a Q from its A, destroying retrieval intent. We honour
    the document's natural structure and only fall back to paragraph
    splits when no Q&A boundary is present.

Algorithm (per sub-sprint p1-s1-ssA spec):
    1. Concatenate all page text, but keep a map of char-offset -> page
       so each emitted chunk gets the correct ``page`` (first page where
       the chunk text begins).
    2. Walk the text. When a "Question No." header is found, the slice
       from that header to the next header (or EOF) becomes a single
       ``qa_pair`` chunk with ``question_no`` populated.
    3. Between Q&A blocks (or when there are none), split on blank-line
       paragraph boundaries; merge tiny paragraphs and split oversized
       ones at sentence-ish boundaries — target 400-800 chars.
    4. For every chunk, extract Drawing-No and Clause references via
       regex; also detect ``Section X`` / ``Clause X.Y`` headers
       appearing immediately before / inside a chunk and record into
       ``section_path``.

Key DECISIONs (all logged in spec's "关键决策" gate):
    * Max Q&A chunk size: 2000 chars. Longer Q&A bodies are split into
      multiple chunks; ``question_no`` propagates to every split so
      retrieval-by-question-id still works.
    * Paragraph target window: 400-800 chars. Below 200 we merge with
      the next paragraph; above 1000 we split at the nearest
      sentence-ending punctuation.
    * Page assignment: a chunk inherits the page where its FIRST
      character originates (not the last). Mid-page-boundary chunks
      keep one page — citation accuracy guarded by retrieval recall, not
      by chunker exactness, per RAG-eval norms.

Context:
    Phase 1 S1.1 ssA. Consumed by ingest/pipeline.py.
"""

from __future__ import annotations

import re

from jcontract.interfaces import StructureSpec
from jcontract.interfaces.schema import Chunk, ChunkType, ParsedPage

# ---------------------------------------------------------------------------
# Regex catalogue. Compiled once at module load — these run on every page.
# ---------------------------------------------------------------------------

# A "Question No." header marks the start of a Q&A block. Captures the
# question identifier (e.g. "ACME/TRACKWORK/16", "TQA-001", "12").
# - Case-insensitive, anchored to line start (MULTILINE).
# - Allows the spelling "Question No.", "Question No:", "Question No",
#   followed by an optional separator (":" or ".") and the id.
_QUESTION_NO_RE = re.compile(
    r"^\s*Question\s*No[.:]?\s*[:.]?\s*([\w/\-]+)",
    re.IGNORECASE | re.MULTILINE,
)

# Drawing No. references appear as e.g. "Drawing No. T/PRJ/CWD/WS/2101A"
# or "Dwg. T/PRJ/CWD/WS/2101A". The trailing element must be digits +
# optional revision letter (per spec).
_DRAWING_REF_RE = re.compile(
    r"(?:Drawing\s*No\.?\s*|Dwg\.?\s*)([\w/\-]+/\d+[A-Z]?)",
    re.IGNORECASE,
)

# Clause references like "Clause 7.3.1" or "Cl. 4.2". Captures the dotted
# number; whitespace between keyword and number is required to avoid
# matching things like "Clause7" (unlikely in real contracts).
_CLAUSE_REF_RE = re.compile(
    r"(?:Clause|Cl\.?)\s+(\d+(?:\.\d+)*)",
    re.IGNORECASE,
)

# Section/Clause header lines used to build section_path. We treat a line
# whose entire content is "Section X" or "Clause X.Y[.Z...]" (with optional
# title trailing) as a header. Section paths are *additive*: once we see a
# Section, it stays current until the next Section header.
_SECTION_HDR_RE = re.compile(
    r"^\s*(Section\s+\d+[A-Z]?)\b",
    re.IGNORECASE | re.MULTILINE,
)
_CLAUSE_HDR_RE = re.compile(
    r"^\s*(Clause\s+\d+(?:\.\d+)*)\b",
    re.IGNORECASE | re.MULTILINE,
)

# Sentence terminators we use when force-splitting an oversized paragraph.
# Includes Chinese full stop and exclamation/question marks for bilingual
# content (the project handles English + Chinese contracts).
_SENTENCE_END_RE = re.compile(r"[.!?。！？](?=\s|$)")

# Tunables (DECISION values per the module docstring).
_QA_MAX_CHARS = 2000
_PARA_TARGET_MIN = 400
_PARA_TARGET_MAX = 800
_PARA_MERGE_THRESHOLD = 200  # below this, merge with next paragraph
_PARA_FORCE_SPLIT = 1000  # above this, force-split at sentence boundary


class QaAwareChunker:
    """Default structure-aware chunker.

    ``chunk()`` is pure given its inputs; one instance is safe to share
    across documents / threads.

    Phase 7 SS3: the domain-specific regexes (Q&A boundary, cross-reference
    rules, section/clause headers) come from a ``StructureSpec`` (part of a
    DomainProfile). ``structure=None`` reproduces the original construction
    (contract) behaviour byte-for-byte by using the module-level compiled
    regexes. A neutral spec (no qa pattern, no ref rules, no header
    patterns — e.g. the ``document`` profile) makes the chunker fall back
    to pure paragraph splitting with empty tender fields.
    """

    def __init__(self, structure: StructureSpec | None = None) -> None:
        if structure is None:
            # Construction default — module-level compiled regexes verbatim.
            self._qa_re: re.Pattern[str] | None = _QUESTION_NO_RE
            self._section_re: re.Pattern[str] | None = _SECTION_HDR_RE
            self._clause_re: re.Pattern[str] | None = _CLAUSE_HDR_RE
            self._ref_rules: list[tuple[re.Pattern[str], str]] = [
                (_DRAWING_REF_RE, "drawing_refs"),
                (_CLAUSE_REF_RE, "clause_refs"),
            ]
        else:
            # Canonical flags match the original impl: Q&A + headers are
            # line-anchored (MULTILINE) + case-insensitive; ref rules are
            # case-insensitive. None pattern → that feature is disabled.
            qa = structure.qa_block_pattern
            sec = structure.section_header_pattern
            cl = structure.clause_header_pattern
            self._qa_re = re.compile(qa, re.IGNORECASE | re.MULTILINE) if qa else None
            self._section_re = re.compile(sec, re.IGNORECASE | re.MULTILINE) if sec else None
            self._clause_re = re.compile(cl, re.IGNORECASE | re.MULTILINE) if cl else None
            self._ref_rules = [
                (re.compile(r.pattern, re.IGNORECASE), r.target_field) for r in structure.ref_rules
            ]

    def chunk(self, pages: list[ParsedPage], file: str) -> list[Chunk]:
        """Slice ``pages`` into chunks; see module docstring for algorithm."""
        if not pages:
            return []

        # ------------------------------------------------------------------
        # Build a flat text + a parallel page-index for every character.
        # We insert a single "\n" between pages so paragraph detection
        # works across page breaks but page-attribution stays exact.
        # ------------------------------------------------------------------
        text_parts: list[str] = []
        # char_to_page[i] = source page (1-indexed) of character i in the
        # concatenated string. Built lazily via running offsets.
        page_starts: list[tuple[int, int]] = []  # (offset, page_num)
        running = 0
        for page in pages:
            page_starts.append((running, page.page_num))
            text_parts.append(page.text)
            running += len(page.text)
            # Page break separator. We keep it as a single \n so it
            # contributes to "blank line" detection only when the page
            # already ends with one.
            text_parts.append("\n")
            running += 1

        full_text = "".join(text_parts)

        def page_of(offset: int) -> int:
            """Lookup source page for a character offset in ``full_text``.

            Linear scan is fine: pages list is <200 for contract PDFs.
            """
            current = page_starts[0][1] if page_starts else 1
            for start, num in page_starts:
                if start <= offset:
                    current = num
                else:
                    break
            return current

        # ------------------------------------------------------------------
        # Pass 1: locate every "Question No." header. These partition
        # the document into [pre-Q region][Q1 region][Q2 region]...
        # ------------------------------------------------------------------
        q_matches = list(self._qa_re.finditer(full_text)) if self._qa_re else []

        chunks: list[Chunk] = []
        chunk_idx = 0

        def emit(
            text: str,
            start_offset: int,
            chunk_type: ChunkType,
            question_no: str | None,
            section_path: str | None,
        ) -> None:
            """Build a Chunk, populate refs from regex scan, and append."""
            nonlocal chunk_idx
            clean = text.strip()
            if not clean:
                return
            page = page_of(start_offset)
            # Cross-reference extraction is profile-driven (SS3): each rule
            # collects into its target Chunk field. Construction default has
            # drawing_refs + clause_refs rules; a neutral profile has none.
            refs: dict[str, list[str]] = {}
            for rule_re, target_field in self._ref_rules:
                vals = sorted(set(rule_re.findall(clean)))
                if vals:
                    refs[target_field] = vals
            chunk_id = f"{_file_stem(file)}:{page}:{chunk_idx}"
            chunks.append(
                Chunk(
                    id=chunk_id,
                    text=clean,
                    file=file,
                    page=page,
                    chunk_type=chunk_type,
                    section_path=section_path,
                    revision=None,  # revision detection deferred (Phase 2)
                    drawing_refs=refs.get("drawing_refs", []),
                    clause_refs=refs.get("clause_refs", []),
                    question_no=question_no,
                )
            )
            chunk_idx += 1

        # Track current section_path as we walk; populated by header scans
        # within each region.
        def current_section_path(region_text: str) -> str | None:
            """Return the latest "Section N > Clause N.M" string we see
            inside ``region_text``, or None.

            We scan headers in document order; the last Section / Clause
            wins. This matches how a human reader infers "I'm in Section
            7" from the most recent header.
            """
            section: str | None = None
            clause: str | None = None
            if self._section_re:
                for m in self._section_re.finditer(region_text):
                    section = m.group(1).strip()
            if self._clause_re:
                # Avoid mistaking a Clause REFERENCE ("see Clause 7.3")
                # for a header by requiring the match to be at the
                # start of a line (already enforced by MULTILINE).
                for m in self._clause_re.finditer(region_text):
                    clause = m.group(1).strip()
            if section and clause:
                return f"{section} > {clause}"
            return section or clause

        # ------------------------------------------------------------------
        # Pass 2: emit pre-Q region (if any) as paragraph chunks.
        # ------------------------------------------------------------------
        first_q_offset = q_matches[0].start() if q_matches else len(full_text)
        pre_q = full_text[:first_q_offset]
        if pre_q.strip():
            for para_text, para_offset in _paragraphs(pre_q, base_offset=0):
                section_path = current_section_path(full_text[: para_offset + len(para_text)])
                emit(
                    para_text,
                    para_offset,
                    chunk_type="paragraph",
                    question_no=None,
                    section_path=section_path,
                )

        # ------------------------------------------------------------------
        # Pass 3: each Q region becomes one (possibly split) qa_pair chunk.
        # ------------------------------------------------------------------
        for i, m in enumerate(q_matches):
            q_start = m.start()
            q_end = q_matches[i + 1].start() if i + 1 < len(q_matches) else len(full_text)
            qa_text = full_text[q_start:q_end]
            question_no = m.group(1).strip()
            section_path = current_section_path(full_text[:q_end])

            # If the Q&A fits in one chunk, emit as a single qa_pair.
            # Otherwise split at sentence boundaries while keeping the
            # question_no on every fragment.
            if len(qa_text) <= _QA_MAX_CHARS:
                emit(
                    qa_text,
                    q_start,
                    chunk_type="qa_pair",
                    question_no=question_no,
                    section_path=section_path,
                )
            else:
                for frag_text, frag_offset in _split_oversized(qa_text, base_offset=q_start):
                    emit(
                        frag_text,
                        frag_offset,
                        chunk_type="qa_pair",
                        question_no=question_no,
                        section_path=section_path,
                    )

        return chunks


# ---------------------------------------------------------------------------
# Helpers (module-private; not part of the Chunker protocol).
# ---------------------------------------------------------------------------


def _file_stem(file: str) -> str:
    """Return a chunk-id-friendly stem of ``file``.

    We strip the extension and collapse spaces/slashes so ids stay safe
    to use in URLs / file names downstream. We deliberately keep this
    cheap (no Path() call) because chunk creation is hot.
    """
    name = file.rsplit("/", 1)[-1]  # basename
    if "." in name:
        name = name.rsplit(".", 1)[0]
    return name.replace(" ", "_")


def _paragraphs(text: str, base_offset: int) -> list[tuple[str, int]]:
    """Split ``text`` into paragraphs of 400-800 chars (target).

    Returns ``[(paragraph_text, absolute_offset), ...]``. The offset is
    where the paragraph starts in the *caller's* coordinate system
    (``base_offset`` + index within text), so the chunker can map back
    to a source page.

    Algorithm:
      * Initial split on blank lines (one or more consecutive newlines
        with only whitespace).
      * Merge tiny paragraphs (<200 chars) into the next non-tiny one.
      * Force-split oversize paragraphs (>1000 chars) at the last
        sentence terminator before the soft max (800).
    """
    # Find all paragraph boundaries (blank lines). re.split keeps content
    # but loses offsets; use finditer on the separator and slice by hand.
    sep_re = re.compile(r"\n\s*\n")
    parts: list[tuple[str, int]] = []
    cursor = 0
    for m in sep_re.finditer(text):
        chunk = text[cursor : m.start()]
        if chunk.strip():
            parts.append((chunk, base_offset + cursor))
        cursor = m.end()
    tail = text[cursor:]
    if tail.strip():
        parts.append((tail, base_offset + cursor))

    # Merge tiny paragraphs forward.
    merged: list[tuple[str, int]] = []
    pending_text = ""
    pending_offset: int | None = None
    for part_text, part_offset in parts:
        if pending_text:
            # Stitch with a newline to preserve "between paragraphs" feel.
            pending_text = f"{pending_text}\n\n{part_text}"
        else:
            pending_text = part_text
            pending_offset = part_offset
        if len(pending_text) >= _PARA_MERGE_THRESHOLD and pending_offset is not None:
            merged.append((pending_text, pending_offset))
            pending_text = ""
            pending_offset = None
    if pending_text and pending_offset is not None:
        # Trailing tiny paragraph: keep it standalone rather than drop.
        merged.append((pending_text, pending_offset))

    # Force-split oversize paragraphs.
    final: list[tuple[str, int]] = []
    for part_text, part_offset in merged:
        if len(part_text) <= _PARA_FORCE_SPLIT:
            final.append((part_text, part_offset))
            continue
        final.extend(_split_oversized(part_text, base_offset=part_offset))
    return final


def _split_oversized(text: str, base_offset: int) -> list[tuple[str, int]]:
    """Force a long string into <= ``_PARA_TARGET_MAX``-char fragments.

    We prefer to break at sentence terminators (Chinese + English).
    If no terminator is found within the target window, we hard-cut at
    the soft max so we never emit a single 10k-char chunk that would
    blow up the embedder's context.
    """
    fragments: list[tuple[str, int]] = []
    cursor = 0
    n = len(text)
    while cursor < n:
        remaining = n - cursor
        if remaining <= _PARA_TARGET_MAX:
            fragments.append((text[cursor:], base_offset + cursor))
            break
        # Look for the last sentence terminator within [target_min, target_max]
        # of the current cursor. If found, cut there; else hard-cut at max.
        window_end = cursor + _PARA_TARGET_MAX
        window = text[cursor + _PARA_TARGET_MIN : window_end]
        last_term = None
        for m in _SENTENCE_END_RE.finditer(window):
            last_term = m  # keep iterating to find the rightmost
        # Cut at the rightmost terminator within the target window if any,
        # else hard-cut at the soft max. ``last_term.end()`` includes the
        # terminator itself so the fragment ends cleanly on punctuation.
        cut = cursor + _PARA_TARGET_MIN + last_term.end() if last_term is not None else window_end
        fragments.append((text[cursor:cut], base_offset + cursor))
        cursor = cut
    return fragments
