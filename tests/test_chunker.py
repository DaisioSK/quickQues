"""Unit tests for ``QaAwareChunker``.

We feed ``ParsedPage`` objects directly (no PDF round-trip needed),
which keeps each test focused on one chunker behaviour. The parser
already has its own coverage in test_parser.py.

Coverage:
  * Q&A pair detection + question_no extraction
  * Drawing No. extraction
  * Clause No. extraction + section_path assembly
  * Empty-page robustness
  * Page-number propagation (chunk.page matches source ParsedPage.page_num)
  * Oversize Q&A splits keep question_no on every fragment
"""

from __future__ import annotations

from jcontract.impls.qa_chunker import QaAwareChunker
from jcontract.interfaces.schema import Chunk, ParsedPage


def _chunk_by_type(chunks: list[Chunk], chunk_type: str) -> list[Chunk]:
    return [c for c in chunks if c.chunk_type == chunk_type]


# ---------------------------------------------------------------------------
# Q&A detection
# ---------------------------------------------------------------------------


def test_question_no_creates_qa_pair_chunk() -> None:
    pages = [
        ParsedPage(
            page_num=12,
            text=(
                "Question No.: ACME/TRACKWORK/16\n"
                "Please confirm waterproofing scope.\n\n"
                "Answer: Waterproofing is by the Trackwork Contractor.\n"
            ),
        )
    ]

    chunks = QaAwareChunker().chunk(pages, "Contract DEMO(1of9) TQA.pdf")

    qa = _chunk_by_type(chunks, "qa_pair")
    assert len(qa) == 1
    assert qa[0].question_no == "ACME/TRACKWORK/16"
    assert "Answer" in qa[0].text
    assert qa[0].page == 12
    # id should encode file stem + page + index
    assert qa[0].id.startswith("Contract_DEMO(1of9)_TQA:12:")


def test_multiple_questions_become_separate_chunks() -> None:
    pages = [
        ParsedPage(
            page_num=1,
            text=(
                "Question No.: Q1\nFirst question body.\n"
                "Answer: First answer.\n\n"
                "Question No.: Q2\nSecond question body.\n"
                "Answer: Second answer.\n"
            ),
        )
    ]

    chunks = QaAwareChunker().chunk(pages, "doc.pdf")
    qa = _chunk_by_type(chunks, "qa_pair")

    assert len(qa) == 2
    assert {c.question_no for c in qa} == {"Q1", "Q2"}


def test_question_no_variants_parsed() -> None:
    # Spec regex must tolerate "Question No.:", "Question No:",
    # "Question No.", differing separators / whitespace.
    pages = [
        ParsedPage(page_num=1, text="Question No.: TQA-001\nbody one\nAnswer: a\n"),
        ParsedPage(page_num=2, text="Question No: TQA-002\nbody two\nAnswer: b\n"),
        ParsedPage(page_num=3, text="Question No 003\nbody three\nAnswer: c\n"),
    ]

    chunks = QaAwareChunker().chunk(pages, "doc.pdf")
    qa = _chunk_by_type(chunks, "qa_pair")

    assert sorted(c.question_no or "" for c in qa) == ["003", "TQA-001", "TQA-002"]


# ---------------------------------------------------------------------------
# Reference extraction
# ---------------------------------------------------------------------------


def test_drawing_no_extracted_into_drawing_refs() -> None:
    pages = [
        ParsedPage(
            page_num=5,
            text=(
                "Question No.: Q1\nReference drawing is Drawing No. "
                "T/PRJ/CWD/WS/2101A for the pier waterproofing detail.\n"
                "Answer: confirmed.\n"
            ),
        )
    ]

    chunks = QaAwareChunker().chunk(pages, "doc.pdf")

    assert any("T/PRJ/CWD/WS/2101A" in c.drawing_refs for c in chunks)


def test_drawing_no_short_form_dwg_extracted() -> None:
    pages = [
        ParsedPage(
            page_num=1,
            text="Refer to Dwg. T/PRJ/CWD/WS/3050B for details. End.",
        )
    ]

    chunks = QaAwareChunker().chunk(pages, "doc.pdf")

    flat: list[str] = []
    for c in chunks:
        flat.extend(c.drawing_refs)
    assert "T/PRJ/CWD/WS/3050B" in flat


def test_clause_refs_extracted() -> None:
    pages = [
        ParsedPage(
            page_num=1,
            text=(
                "The scope is defined in Clause 7.3.1 and supplemented "
                "by Cl. 4.2. End of paragraph one.\n\n"
                "Paragraph two adds context to Clause 7.3.1 again."
            ),
        )
    ]

    chunks = QaAwareChunker().chunk(pages, "doc.pdf")
    flat: list[str] = []
    for c in chunks:
        flat.extend(c.clause_refs)

    assert "7.3.1" in flat
    assert "4.2" in flat


def test_section_header_populates_section_path() -> None:
    pages = [
        ParsedPage(
            page_num=2,
            text=(
                "Section 7\n"
                "Clause 7.3\n"
                "This paragraph belongs under Section 7 > Clause 7.3.\n"
                "It is long enough to be retained as its own chunk after "
                "the tiny-paragraph merge step picks it up cleanly."
            ),
        )
    ]

    chunks = QaAwareChunker().chunk(pages, "doc.pdf")

    paths = [c.section_path for c in chunks if c.section_path]
    assert paths, "expected at least one chunk with section_path set"
    assert any("Section 7" in p for p in paths)


# ---------------------------------------------------------------------------
# Robustness
# ---------------------------------------------------------------------------


def test_empty_pages_do_not_crash() -> None:
    pages = [
        ParsedPage(page_num=1, text=""),
        ParsedPage(page_num=2, text="   \n\n   "),  # whitespace only
    ]

    chunks = QaAwareChunker().chunk(pages, "doc.pdf")
    assert chunks == []


def test_empty_input_returns_empty() -> None:
    chunks = QaAwareChunker().chunk([], "doc.pdf")
    assert chunks == []


def test_pre_question_text_emitted_as_paragraph_chunks() -> None:
    # Content before the first "Question No." should still be retrievable,
    # not silently dropped. Build a paragraph long enough to clear the
    # tiny-merge threshold (>200 chars).
    long_pre = (
        "This is the preamble describing the construction contract scope. "
        "It explains responsibilities and references applicable standards "
        "such as BS 8102 and Clause 7.3. There are no questions yet, but "
        "the content must still be retrievable for queries about scope.\n"
    )
    pages = [
        ParsedPage(
            page_num=1,
            text=(long_pre + "\nQuestion No.: Q1\nBody.\nAnswer: A.\n"),
        )
    ]

    chunks = QaAwareChunker().chunk(pages, "doc.pdf")

    paragraphs = _chunk_by_type(chunks, "paragraph")
    assert paragraphs, "preamble paragraphs should not be dropped"
    assert any("preamble" in c.text for c in paragraphs)


def test_oversized_qa_pair_splits_with_propagated_question_no() -> None:
    # Build an answer body longer than the 2000-char Q&A cap so the
    # chunker is forced to split. question_no must propagate to every
    # fragment so retrieval by question id still works.
    long_answer = "Sentence about waterproofing. " * 200  # ~6000 chars
    pages = [
        ParsedPage(
            page_num=1,
            text=f"Question No.: BIG-001\nAnswer: {long_answer}",
        )
    ]

    chunks = QaAwareChunker().chunk(pages, "doc.pdf")
    qa = _chunk_by_type(chunks, "qa_pair")

    assert len(qa) > 1, "long Q&A must be split into multiple chunks"
    assert all(c.question_no == "BIG-001" for c in qa)
    # No fragment should exceed the soft max (allow a small slack
    # because we break at sentence boundaries past the target).
    assert all(len(c.text) <= 1200 for c in qa)


def test_chunk_id_is_stable_and_unique() -> None:
    # Stable: same input → same ids (so re-ingest is idempotent).
    # Unique within a doc: ids must not collide.
    pages = [
        ParsedPage(
            page_num=1,
            text="Question No.: Q1\nA.\n\nQuestion No.: Q2\nB.\n",
        )
    ]

    chunker = QaAwareChunker()
    a = chunker.chunk(pages, "doc.pdf")
    b = chunker.chunk(pages, "doc.pdf")

    assert [c.id for c in a] == [c.id for c in b]  # stable
    assert len({c.id for c in a}) == len(a)  # unique
