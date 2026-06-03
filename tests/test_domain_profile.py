"""Tests for the DomainProfile registry (Phase 7 SS1).

The critical guarantee: the `contract` profile reproduces today's construction
prompts + chunker regexes BYTE-FOR-BYTE, so wiring profiles in later
sub-sprints cannot regress the live DEMO index.
"""

from __future__ import annotations

import pytest

from jcontract.impls import qa_chunker as qc
from jcontract.impls._caption_shared import CAPTION_PROMPT
from jcontract.impls.claude_vision_parser import DRAWING_CAPTION_PROMPT, TEXT_OCR_PROMPT
from jcontract.impls.domain_profile_registry import available_profiles, load_profile
from jcontract.interfaces import DomainProfile


def test_contract_prompts_match_source_constants_byte_for_byte() -> None:
    p = load_profile("contract")
    assert isinstance(p, DomainProfile)
    assert p.ocr_text_prompt == TEXT_OCR_PROMPT
    assert p.ocr_drawing_prompt == DRAWING_CAPTION_PROMPT
    assert p.caption_prompt == CAPTION_PROMPT


def test_contract_structure_matches_chunker_regexes() -> None:
    sp = load_profile("contract").structure
    assert sp.qa_block_pattern == qc._QUESTION_NO_RE.pattern
    assert sp.section_header_pattern == qc._SECTION_HDR_RE.pattern
    assert sp.clause_header_pattern == qc._CLAUSE_HDR_RE.pattern
    by_field = {r.target_field: r.pattern for r in sp.ref_rules}
    assert by_field == {
        "drawing_refs": qc._DRAWING_REF_RE.pattern,
        "clause_refs": qc._CLAUSE_REF_RE.pattern,
    }


def test_contract_answer_framing_is_template_prefix() -> None:
    # The framing must be exactly the part of the system prompt before rule 1,
    # so SS2's split reassembles the identical prompt.
    from jcontract.answer.prompt import _SYSTEM_PROMPT_TEMPLATE

    assert load_profile("contract").answer_framing == _SYSTEM_PROMPT_TEMPLATE.split("\n\n1.")[0]
    assert "construction" in load_profile("contract").answer_framing


def test_document_profile_is_neutral() -> None:
    p = load_profile("document")
    # Neutral structure → chunker paragraph fallback.
    assert p.structure.qa_block_pattern is None
    assert p.structure.ref_rules == ()
    assert p.structure.section_header_pattern is None
    # No construction vocabulary in the neutral prompts.
    assert "construction" not in p.ocr_text_prompt.lower()
    assert "drawing no" not in p.ocr_text_prompt.lower()
    # caption prompt stays str.format-compatible (placeholder + literal braces).
    assert "{nearby_text}" in p.caption_prompt
    assert '{{"caption_zh"' in p.caption_prompt
    assert "JSON object" in p.caption_prompt


def test_caption_prompt_formats_without_error() -> None:
    # Both profiles' caption prompts must survive str.format (build_caption_prompt).
    for name in ("contract", "document"):
        rendered = load_profile(name).caption_prompt.format(nearby_text="grounding text")
        assert "grounding text" in rendered


def test_available_profiles_lists_both() -> None:
    names = available_profiles()
    assert "contract" in names
    assert "document" in names


def test_unknown_profile_raises() -> None:
    with pytest.raises(ValueError, match="Unknown domain profile"):
        load_profile("does-not-exist")
