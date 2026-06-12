"""Unit tests for DictRegexRedactor + JsonlMappingStore + `redact-preview` CLI (ssDI).

SYNTHETIC DICTIONARY ONLY — no real corpus entity may ever appear in this
repo (dev-sprint v5 ssDI data red line / DECISION-cq.5).

Covered surfaces:
- byte-exact roundtrip (utf-8) over literals + regex patterns
- corpus-stable placeholders across calls AND across instances (store reload)
- span-conflict semantics: contained + partially-intersecting candidates
  never break the roundtrip (REMOVE_INTERSECTIONS-equivalent, DECISION-ls.41)
- mapping store persistence: JSONL on disk, counter continuation on reload
- security red line: repr/str/exception messages never carry entity names
- CLI redact-preview redact + --restore directions
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from jcontract.cli import app
from jcontract.impls.dict_regex_redactor import (
    _PLACEHOLDER_RE,
    DictRegexRedactor,
    JsonlMappingStore,
)

runner = CliRunner()

SYNTHETIC_DICT = {
    "entities": {
        "ORG": ["Acme Corp Pte. Ltd.", "Acme Corp", "Globex"],
        "PERSON": ["Jane Doe", "John Q Public"],
    },
    "patterns": {
        "DOC_NO": [r"X/DOC/[A-Z]+/\d+[A-Z]?"],
        "MONEY": [r"S?\$\s?[\d,]+(?:\.\d{2})?"],
    },
}

SAMPLE = (
    "Between Acme Corp Pte. Ltd. and Globex, signed by Jane Doe.\n"
    "Ref X/DOC/CWD/2101A, contract sum S$ 226,612,000.00 (see Acme Corp).\n"
    "John Q Public witnessed; 中文上下文也保持原样。\n"
)


def _make_redactor(
    tmp_path: Path,
    dictionary: Mapping[str, object] | None = None,
    store_name: str = "map.jsonl",
    tier: str = "standard",
) -> DictRegexRedactor:
    dict_path = tmp_path / "dict.yaml"
    dict_path.write_text(
        yaml.safe_dump(dictionary or SYNTHETIC_DICT, allow_unicode=True), encoding="utf-8"
    )
    return DictRegexRedactor(dictionary_path=dict_path, store_path=tmp_path / store_name, tier=tier)


# ---------------------------------------------------------------------------
# roundtrip + placeholder behaviour
# ---------------------------------------------------------------------------


def test_roundtrip_byte_exact(tmp_path):
    redactor = _make_redactor(tmp_path)
    result = redactor.redact(SAMPLE)
    assert result.redacted_text != SAMPLE
    assert redactor.restore(result.redacted_text).encode("utf-8") == SAMPLE.encode("utf-8")


def test_redacted_text_contains_no_entities_but_placeholders(tmp_path):
    redactor = _make_redactor(tmp_path)
    redacted = redactor.redact(SAMPLE).redacted_text
    for entity in ["Acme Corp", "Globex", "Jane Doe", "John Q Public", "X/DOC/CWD/2101A"]:
        assert entity not in redacted
    assert "<ORG_0>" in redacted
    assert "<PERSON_0>" in redacted
    assert "<DOC_NO_0>" in redacted
    assert "<MONEY_0>" in redacted
    # non-entity text untouched
    assert "中文上下文也保持原样" in redacted


def test_same_entity_same_placeholder_across_calls_and_mapping_delta(tmp_path):
    redactor = _make_redactor(tmp_path)
    first = redactor.redact("Globex hired Jane Doe.")
    assert first.mapping_delta == 2
    assert first.spans_replaced == 2
    second = redactor.redact("Jane Doe left Globex later.")
    assert second.mapping_delta == 0  # both entities already mapped
    assert "<ORG_" in first.redacted_text and "<ORG_" in second.redacted_text
    # exact same placeholder for the same entity
    org_token = [t for t in first.redacted_text.split() if t.startswith("<ORG_")][0]
    assert org_token.rstrip(".") in second.redacted_text


def test_placeholder_stable_across_instances_via_store_reload(tmp_path):
    r1 = _make_redactor(tmp_path)
    redacted1 = r1.redact("Globex memo").redacted_text
    # fresh instance, same store path -> same numbering, and a NEW entity
    # continues the counter instead of reusing an index
    r2 = _make_redactor(tmp_path)
    redacted2 = r2.redact("Globex and Acme Corp memo").redacted_text
    assert redacted1.split()[0] in redacted2  # Globex placeholder identical
    assert redacted2.count("<ORG_0>") + redacted2.count("<ORG_1>") == 2
    # restoring text produced by r1 works from r2 (persisted mapping)
    assert r2.restore(redacted1) == "Globex memo"


def test_store_file_is_jsonl_and_counter_continues(tmp_path):
    r1 = _make_redactor(tmp_path)
    r1.redact("Globex")
    store_path = tmp_path / "map.jsonl"
    lines = store_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    store = JsonlMappingStore(store_path)
    placeholder, created = store.placeholder_for("ORG", "Acme Corp")
    assert (placeholder, created) == ("<ORG_1>", True)
    assert store.placeholder_for("ORG", "Acme Corp") == ("<ORG_1>", False)
    assert len(store) == 2


# ---------------------------------------------------------------------------
# span-conflict semantics (REMOVE_INTERSECTIONS-equivalent)
# ---------------------------------------------------------------------------


def test_contained_span_longest_wins_roundtrip(tmp_path):
    redactor = _make_redactor(tmp_path)
    text = "Acme Corp Pte. Ltd. is not just Acme Corp."
    result = redactor.redact(text)
    # the full form wins where contained; the short form still matches standalone
    assert "<ORG_0>" in result.redacted_text and "<ORG_1>" in result.redacted_text
    assert "Acme" not in result.redacted_text
    assert redactor.restore(result.redacted_text).encode() == text.encode()


def test_partially_intersecting_spans_keep_one_and_stay_reversible(tmp_path):
    dictionary = {
        "entities": {"ORG": ["Alpha Beta"], "PRODUCT": ["Beta Gamma"]},
        "patterns": {},
    }
    redactor = _make_redactor(tmp_path, dictionary=dictionary)
    text = "see Alpha Beta Gamma here"  # candidates intersect on "Beta"
    result = redactor.redact(text)
    assert result.spans_replaced == 1  # the intersecting candidate is dropped
    assert redactor.restore(result.redacted_text).encode() == text.encode()


def test_regex_and_literal_overlap_is_reversible(tmp_path):
    dictionary = {
        "entities": {"ORG": ["DOC Holdings"]},
        "patterns": {"DOC_NO": [r"X/DOC/[A-Z]+/\d+"]},
    }
    redactor = _make_redactor(tmp_path, dictionary=dictionary)
    text = "X/DOC/AB/12 issued by DOC Holdings"
    result = redactor.redact(text)
    assert result.spans_replaced == 2
    assert redactor.restore(result.redacted_text).encode() == text.encode()


# ---------------------------------------------------------------------------
# security red line: no entity content in repr / errors
# ---------------------------------------------------------------------------


def test_reprs_never_contain_entity_names(tmp_path):
    redactor = _make_redactor(tmp_path)
    result = redactor.redact(SAMPLE)
    store = JsonlMappingStore(tmp_path / "map.jsonl")
    for blob in (repr(redactor), str(redactor), repr(store), str(store)):
        for entity in ["Acme", "Globex", "Jane", "Doe", "Public"]:
            assert entity not in blob, f"entity leaked into {blob!r}"
    # RedactionResult only carries redacted text + counts
    assert "Globex" not in repr(result)


def test_redact_guard_message_carries_placeholder_only(tmp_path):
    redactor = _make_redactor(tmp_path)
    redacted = redactor.redact("Globex memo").redacted_text
    with pytest.raises(ValueError) as excinfo:
        redactor.redact(redacted)  # already contains a known placeholder
    assert "<ORG_0>" in str(excinfo.value)
    assert "Globex" not in str(excinfo.value)


def test_invalid_regex_error_does_not_quote_pattern(tmp_path):
    dictionary = {"entities": {}, "patterns": {"BAD": ["(unclosed"]}}
    with pytest.raises(ValueError) as excinfo:
        _make_redactor(tmp_path, dictionary=dictionary)
    assert "patterns.BAD[0]" in str(excinfo.value)
    assert "(unclosed" not in str(excinfo.value)
    assert excinfo.value.__cause__ is None  # no chained re.error carrying the pattern


# ---------------------------------------------------------------------------
# restore edge semantics
# ---------------------------------------------------------------------------


def test_restore_leaves_unknown_placeholder_untouched(tmp_path):
    redactor = _make_redactor(tmp_path)
    assert redactor.restore("see <ORG_99> and <NEVER_ISSUED_3>") == (
        "see <ORG_99> and <NEVER_ISSUED_3>"
    )


def test_restore_does_not_interpret_backslashes_in_originals(tmp_path):
    dictionary = {"entities": {"ORG": [r"Acme\1 Corp"]}, "patterns": {}}
    redactor = _make_redactor(tmp_path, dictionary=dictionary)
    text = r"by Acme\1 Corp today"
    result = redactor.redact(text)
    assert result.spans_replaced == 1
    assert redactor.restore(result.redacted_text) == text


# ---------------------------------------------------------------------------
# dictionary validation
# ---------------------------------------------------------------------------


def test_lowercase_type_key_rejected(tmp_path):
    dictionary = {"entities": {"org": ["Acme Corp"]}, "patterns": {}}
    with pytest.raises(ValueError, match="entity type key"):
        _make_redactor(tmp_path, dictionary=dictionary)


def test_empty_dictionary_rejected(tmp_path):
    with pytest.raises(ValueError, match="defines no entities or patterns"):
        _make_redactor(tmp_path, dictionary={"entities": {}, "patterns": {}})


# ---------------------------------------------------------------------------
# strict tier (ssRX): proper-noun heuristic + digit-string recognizer
# ---------------------------------------------------------------------------

# DEMO-only entity-dense sample: person (Mr prefix, hyphenated), company with
# abbreviation dots, money, phone segmentation, drawing/ref numbers, an
# ALL-CAPS heading, a date, and CJK text that must survive untouched.
STRICT_SAMPLE = (
    "Mr Tan Ah-Kow of Demo Builders Pte. Ltd. (UEN 201912345K) called +65 6123 4567.\n"
    "Contract sum S$ 1,234,567.89 under T/DEMO/CWD/2101A, drawing DWG-1023 rev 2.\n"
    "SECTION 7 GENERAL: the works near 中文地名 start on 2026-06-12.\n"
)


def _strip_placeholders(text: str) -> str:
    return _PLACEHOLDER_RE.sub("", text)


def test_strict_no_leak_regression(tmp_path):
    """Leak regression [DECISION-tt.2]: after strict redaction, NO capital
    letter and NO >=2-digit string may survive outside placeholders."""
    redactor = _make_redactor(tmp_path, tier="strict")
    redacted = redactor.redact(STRICT_SAMPLE).redacted_text
    residue = _strip_placeholders(redacted)
    assert re.search(r"[A-Z]", residue) is None, f"capital leaked: {residue!r}"
    # no two digits survive, even separated by one grouping char (tt.41 floor)
    assert re.search(r"\d[,.\- ]?\d", residue) is None, f"digits leaked: {residue!r}"
    # the lowercase skeleton and CJK context survive (cloud-ordering signal)
    assert "called" in residue and "the works near" in residue
    assert "中文地名" in residue
    # a lone digit is below the >=2 floor and stays [DECISION-tt.41]
    assert "rev 2." in residue


def test_strict_roundtrip_byte_exact(tmp_path):
    redactor = _make_redactor(tmp_path, tier="strict")
    result = redactor.redact(STRICT_SAMPLE)
    restored = redactor.restore(result.redacted_text)
    assert restored.encode("utf-8") == STRICT_SAMPLE.encode("utf-8")


def test_strict_same_word_same_token_idempotent(tmp_path):
    redactor = _make_redactor(tmp_path, tier="strict")
    first = redactor.redact(STRICT_SAMPLE)
    second = redactor.redact(STRICT_SAMPLE)
    # identical text -> identical placeholders, zero new mapping entries
    assert second.redacted_text == first.redacted_text
    assert second.mapping_delta == 0
    # the same entity in a NEW sentence reuses its corpus-stable token
    third = redactor.redact("Demo Builders Pte. Ltd. replied.").redacted_text
    pn_tokens_first = set(re.findall(r"<PN_\d+>", first.redacted_text))
    assert set(re.findall(r"<PN_\d+>", third)) <= pn_tokens_first


def test_strict_default_tier_behaviour_unchanged(tmp_path):
    """Regression guard: the default tier must NOT pick up the heuristics."""
    standard = _make_redactor(tmp_path, tier="standard", store_name="std.jsonl")
    redacted = standard.redact(STRICT_SAMPLE).redacted_text
    # only the MONEY pattern of SYNTHETIC_DICT matches this sample; proper
    # nouns and digit strings outside the dictionary survive verbatim
    assert "Mr Tan Ah-Kow" in redacted
    assert "Demo Builders Pte. Ltd." in redacted
    assert "+65 6123 4567" in redacted
    assert "<MONEY_0>" in redacted and "1,234,567.89" not in redacted
    assert "<PN_" not in redacted and "<NUM_" not in redacted
    # explicit default: omitting tier == "standard"
    dict_path = tmp_path / "dict.yaml"
    implicit = DictRegexRedactor(dictionary_path=dict_path, store_path=tmp_path / "implicit.jsonl")
    assert implicit.redact(STRICT_SAMPLE).redacted_text == redacted


def test_strict_dictionary_entities_still_win_overlaps(tmp_path):
    redactor = _make_redactor(tmp_path, tier="strict")
    text = "Acme Corp Pte. Ltd. signed; Jane Doe agreed."
    result = redactor.redact(text)
    # dictionary spans survive with their semantic types (longest-span +
    # deterministic tie-break), heuristics only mop up the rest
    assert "<ORG_" in result.redacted_text
    assert "<PERSON_" in result.redacted_text
    assert redactor.restore(result.redacted_text).encode() == text.encode()


def test_unknown_tier_rejected(tmp_path):
    with pytest.raises(ValueError, match="unknown tier"):
        _make_redactor(tmp_path, tier="paranoid")


def test_strict_repr_carries_tier_but_no_content(tmp_path):
    redactor = _make_redactor(tmp_path, tier="strict")
    redactor.redact(STRICT_SAMPLE)
    blob = repr(redactor)
    assert "strict" in blob
    for fragment in ["Tan", "Demo Builders", "6123", "201912345"]:
        assert fragment not in blob


def test_cli_tier_strict_and_restore(tmp_path):
    dict_path = tmp_path / "dict.yaml"
    dict_path.write_text(yaml.safe_dump(SYNTHETIC_DICT, allow_unicode=True), encoding="utf-8")
    store_path = tmp_path / "maps" / "strict.map.jsonl"
    text_path = tmp_path / "page.txt"
    text_path.write_text(STRICT_SAMPLE, encoding="utf-8")
    common = ["--dictionary", str(dict_path), "--map-store", str(store_path)]

    redacted_path = tmp_path / "page.redacted.txt"
    res = runner.invoke(
        app,
        ["redact-preview", str(text_path), *common, "--tier", "strict"]
        + ["--out", str(redacted_path)],
    )
    assert res.exit_code == 0, res.output
    residue = _strip_placeholders(redacted_path.read_text(encoding="utf-8"))
    assert re.search(r"[A-Z]", residue) is None
    assert re.search(r"\d[,.\- ]?\d", residue) is None

    restored_path = tmp_path / "page.roundtrip.txt"
    res = runner.invoke(
        app,
        ["redact-preview", str(redacted_path), *common, "--restore", "--out", str(restored_path)],
    )
    assert res.exit_code == 0, res.output
    assert restored_path.read_bytes() == text_path.read_bytes()


def test_cli_unknown_tier_fails_fast(tmp_path):
    dict_path = tmp_path / "dict.yaml"
    dict_path.write_text(yaml.safe_dump(SYNTHETIC_DICT, allow_unicode=True), encoding="utf-8")
    text_path = tmp_path / "page.txt"
    text_path.write_text("hello", encoding="utf-8")
    res = runner.invoke(
        app,
        [
            "redact-preview",
            str(text_path),
            "--dictionary",
            str(dict_path),
            "--map-store",
            str(tmp_path / "m.jsonl"),
            "--tier",
            "paranoid",
        ],
    )
    assert res.exit_code != 0
    assert "unknown tier" in res.output


# ---------------------------------------------------------------------------
# CLI: redact-preview
# ---------------------------------------------------------------------------


def _write_cli_fixtures(tmp_path: Path) -> tuple[Path, Path, Path]:
    dict_path = tmp_path / "dict.yaml"
    dict_path.write_text(yaml.safe_dump(SYNTHETIC_DICT, allow_unicode=True), encoding="utf-8")
    store_path = tmp_path / "maps" / "corpus.map.jsonl"
    text_path = tmp_path / "page.txt"
    text_path.write_text(SAMPLE, encoding="utf-8")
    return dict_path, store_path, text_path


def test_cli_redact_then_restore_roundtrip(tmp_path):
    dict_path, store_path, text_path = _write_cli_fixtures(tmp_path)
    redacted_path = tmp_path / "page.redacted.txt"
    common = ["--dictionary", str(dict_path), "--map-store", str(store_path)]

    res = runner.invoke(
        app,
        ["redact-preview", str(text_path), *common, "--out", str(redacted_path)],
    )
    assert res.exit_code == 0, res.output
    redacted = redacted_path.read_text(encoding="utf-8")
    assert redacted != SAMPLE and "<ORG_0>" in redacted
    assert store_path.exists()  # mapping store persisted

    restored_path = tmp_path / "page.roundtrip.txt"
    res = runner.invoke(
        app,
        ["redact-preview", str(redacted_path), *common, "--restore", "--out", str(restored_path)],
    )
    assert res.exit_code == 0, res.output
    assert restored_path.read_bytes() == text_path.read_bytes()


def test_cli_stdout_is_verbatim_text(tmp_path):
    dict_path, store_path, text_path = _write_cli_fixtures(tmp_path)
    res = runner.invoke(
        app,
        [
            "redact-preview",
            str(text_path),
            "--dictionary",
            str(dict_path),
            "--map-store",
            str(store_path),
        ],
    )
    assert res.exit_code == 0, res.output
    assert "<PERSON_0>" in res.stdout
    assert "Jane Doe" not in res.stdout


def test_cli_requires_dictionary_and_store(tmp_path):
    _, _, text_path = _write_cli_fixtures(tmp_path)
    res = runner.invoke(
        app,
        ["redact-preview", str(text_path)],
        env={"JCONTRACT_REDACTION_DICT": None, "JCONTRACT_REDACTION_MAP": None},
    )
    assert res.exit_code != 0
    assert "JCONTRACT_REDACTION_DICT" in res.output
