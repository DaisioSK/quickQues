"""Phase 7 SS3: chunker is now StructureSpec-driven.

Verifies (a) the construction default is unchanged, (b) load_profile("contract")
reproduces the default exactly, and (c) the neutral `document` profile
degrades to pure paragraph chunks with empty tender fields.
"""

from __future__ import annotations

from jcontract.impls.domain_profile_registry import load_profile
from jcontract.impls.qa_chunker import QaAwareChunker
from jcontract.interfaces.schema import Chunk, ParsedPage

# A snippet with construction structure: a Section header, a Question No.
# block, and Drawing No. + Clause references inside the answer.
_PAGE = ParsedPage(
    page_num=1,
    text=(
        "Section 7 General Requirements\n\n"
        "Question No. ACME/TRACKWORK/16\n"
        "Answer: Refer to Drawing No. T/PRJ/CWD/WS/2101A and Clause 7.3 for the detail.\n"
    ),
)


def _chunks(chunker: QaAwareChunker) -> list[Chunk]:
    return chunker.chunk([_PAGE], file="sample.pdf")


def test_default_chunker_extracts_construction_structure() -> None:
    chunks = _chunks(QaAwareChunker())
    qa = [c for c in chunks if c.chunk_type == "qa_pair"]
    assert qa, "default chunker should detect the Question No. block"
    c = qa[0]
    assert c.question_no == "ACME/TRACKWORK/16"
    assert "T/PRJ/CWD/WS/2101A" in c.drawing_refs
    assert "7.3" in c.clause_refs
    assert c.section_path is not None and "Section 7" in c.section_path


def test_contract_profile_structure_matches_default_behaviour() -> None:
    # Building the chunker from the contract profile must yield identical chunks
    # to the zero-arg default (the byte-for-byte guarantee at chunk level).
    default = _chunks(QaAwareChunker())
    from_profile = _chunks(QaAwareChunker(load_profile("contract").structure))
    assert [c.__dict__ for c in default] == [c.__dict__ for c in from_profile]


def test_document_profile_degrades_to_paragraphs() -> None:
    chunks = _chunks(QaAwareChunker(load_profile("document").structure))
    # No Q&A detection → no qa_pair chunks; everything is paragraph.
    assert chunks, "should still emit paragraph chunks"
    assert all(c.chunk_type == "paragraph" for c in chunks)
    # Tender metadata fields stay empty under the neutral profile.
    for c in chunks:
        assert c.question_no is None
        assert c.drawing_refs == []
        assert c.clause_refs == []
        assert c.section_path is None
