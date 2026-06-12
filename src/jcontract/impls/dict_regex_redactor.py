"""DictRegexRedactor — reversible pseudonymization via dictionary + regex whitelist.

Formalizes the v4 Presidio PoC as a zero-new-dependency impl
[DECISION-cq.30]: deterministic literal-dictionary + regex-pattern matching
with longest-span-wins conflict resolution (semantics equivalent to
presidio's ``REMOVE_INTERSECTIONS`` — the precondition for byte-exact
restore, DECISION-ls.41) and corpus-stable ``<TYPE_N>`` placeholders backed
by an append-only JSONL mapping store (InstanceCounter pattern,
DECISION-ls.42). NER is deliberately absent: on this kind of corpus a
generic NER contributed ~0 recall and high noise (UNCERTAIN-ls.2 closure);
recall comes from dictionary coverage.

Mechanism only — NOT wired into ingest/answer (DECISION-cq.4). The only
consumer is the ``redact-preview`` CLI demo command.

Tiers [DECISION-tt.2]: the replacement set is selectable at construction.

- ``standard`` (default): dictionary literals + regex whitelist only —
  byte-for-byte the pre-tier behaviour, guarded by regression tests.
- ``strict`` (pre-cloud-dispatch): adds two rule-based recognizers on top —
  a proper-noun heuristic (capitalized-word sequences -> ``<PN_N>``) and a
  digit-string recognizer (>=2 digits incl. thousands/decimal/phone
  grouping -> ``<NUM_N>``). Strict semantics: cloud-dispatch safety beats
  readability — over-masking is acceptable, a leak is the incident, so the
  heuristics carry NO false-positive suppression (sentence-initial words
  and ALL-CAPS headings are masked too) [DECISION-tt.40, DECISION-tt.41].
  The lowercase function-word/verb skeleton survives, which is what the
  cloud page-ordering task needs (DECISION-tt.2). No NER (cq.30/ls.2).

Security discipline (dev-contract/21, the mapping store is the restore key):
- Mapping/dictionary CONTENT (entity names) never enters logs, exception
  messages, or ``__repr__`` of any object in this module — reprs and errors
  carry counts, type names, and placeholders only.
- Dictionary and mapping store are caller-supplied data files that live
  OUTSIDE this repository (DECISION-cq.5); tests use synthetic dictionaries.

Dictionary file format (YAML, ``yaml.safe_load``):

.. code-block:: yaml

    entities:           # literal mentions, matched case-sensitively
      ORG:
        - "Acme Corp Pte. Ltd."
      PERSON:
        - "Jane Doe"
    patterns:           # regex whitelist, one or more patterns per type
      MONEY:
        - 'S?\\$\\s?[\\d,]+(?:\\.\\d{2})?'

Entity-type keys must match ``[A-Z][A-Z0-9_]*`` so that placeholders
(``<ORG_0>``, ``<MONEY_12>``) are unambiguously parseable on restore.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import yaml

from jcontract.interfaces.redactor import RedactionResult

# Placeholder token shape: <TYPE_N> where TYPE is [A-Z][A-Z0-9_]* and N is
# the per-type instance counter. Greedy [A-Z0-9_]* + backtracking splits
# <DRAWING_NO_3> correctly into ("DRAWING_NO", "3").
_PLACEHOLDER_RE = re.compile(r"<([A-Z][A-Z0-9_]*)_(\d+)>")

_TYPE_KEY_RE = re.compile(r"[A-Z][A-Z0-9_]*\Z")

# Valid tier names; "standard" must stay the default forever (zero behaviour
# change for existing callers is a regression-tested contract).
_TIERS = ("standard", "strict")

# --- strict-tier recognizers [DECISION-tt.2] -------------------------------
#
# Proper-noun heuristic: one or more capitalized words (digits/apostrophes
# allowed after the initial capital, so "X107"/"O'Brien" are single words),
# joined by horizontal whitespace, "&", or "-"; an abbreviation dot joins
# only when another capitalized word follows ("Pte. Ltd." is one span, a
# sentence-final dot is left in place so the mapping key for a name is the
# same with or without trailing punctuation).
#
# What/Why: no word-boundary anchor on purpose — "\b" fails after CJK
# characters ("中文Apple") and a left-context guard would skip mid-word
# capitals; the scanner masking ANY capital run is the safe direction
# (multi-mask > leak) [DECISION-tt.40]. Sequences never join across
# newlines: a heading and the next line's sentence-initial word must not
# fuse into one giant mapping key.
_PN_WORD = r"[A-Z][A-Za-z0-9'’]*"
_PROPER_NOUN_RE = re.compile(rf"{_PN_WORD}(?:(?:\.?[ \t]+|[ \t]*&[ \t]*|-){_PN_WORD})*")

# Digit-string recognizer: digit groups joined by "," "." "-" or a single
# space (thousands separators, decimals, dates, phone segmentation), kept
# only when the match carries >=2 digit characters in total. A lone digit
# ("Section 7") is below the user-decided >=2 floor and stays
# [DECISION-tt.41]. No boundary anchors for the same reason as above
# ("x107" must still surrender its "107").
_DIGIT_RUN_RE = re.compile(r"\d+(?:[,.\- ]\d+)*")


class JsonlMappingStore:
    """Append-only JSONL persistence of (entity_type, entity_text) -> placeholder.

    One JSON object per line: ``{"entity_type", "entity_text", "placeholder"}``.
    Append-only keeps the store crash-safe and audit-friendly; numbering is
    derived from the persisted placeholders on load, so the same entity gets
    the same ``<TYPE_N>`` across sessions and new entities never reuse an
    index [DECISION-cq.32].

    The store content is the restore key: this class never logs it and its
    ``repr`` exposes only the path and entry count.
    """

    def __init__(self, path: Path) -> None:
        self._path = path
        self._forward: dict[tuple[str, str], str] = {}
        self._reverse: dict[str, str] = {}
        self._counters: dict[str, int] = {}
        if path.exists():
            self._load()

    def _load(self) -> None:
        with self._path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                entry = json.loads(line)
                etype = entry["entity_type"]
                etext = entry["entity_text"]
                placeholder = entry["placeholder"]
                self._forward[(etype, etext)] = placeholder
                self._reverse[placeholder] = etext
                match = _PLACEHOLDER_RE.fullmatch(placeholder)
                if match is None:
                    raise ValueError(
                        f"mapping store {str(self._path)!r} contains a malformed placeholder"
                    )
                index = int(match.group(2))
                self._counters[etype] = max(self._counters.get(etype, 0), index + 1)

    def placeholder_for(self, entity_type: str, entity_text: str) -> tuple[str, bool]:
        """Return ``(placeholder, created)`` for an entity, persisting new entries."""
        key = (entity_type, entity_text)
        existing = self._forward.get(key)
        if existing is not None:
            return existing, False
        index = self._counters.get(entity_type, 0)
        placeholder = f"<{entity_type}_{index}>"
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._path.open("a", encoding="utf-8") as f:
            f.write(
                json.dumps(
                    {
                        "entity_type": entity_type,
                        "entity_text": entity_text,
                        "placeholder": placeholder,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
        self._forward[key] = placeholder
        self._reverse[placeholder] = entity_text
        self._counters[entity_type] = index + 1
        return placeholder, True

    def original_for(self, placeholder: str) -> str | None:
        """Reverse lookup; None for placeholders this store never issued."""
        return self._reverse.get(placeholder)

    def knows_placeholder(self, placeholder: str) -> bool:
        return placeholder in self._reverse

    def __len__(self) -> int:
        return len(self._forward)

    def __repr__(self) -> str:  # content red line: path + count only
        return f"JsonlMappingStore(path={str(self._path)!r}, entries={len(self._forward)})"


class DictRegexRedactor:
    """Dictionary + regex-whitelist reversible pseudonymizer (implements ``Redactor``).

    Matching is fully deterministic — no NER, no scores. Span conflicts are
    resolved longest-span-wins with a fixed tie-break (start asc, type asc),
    then non-overlapping spans are substituted; this preserves the
    "replaced interval text == mapping key" invariant that byte-exact
    restore requires (REMOVE_INTERSECTIONS-equivalent, DECISION-ls.41).

    ``tier="strict"`` additionally runs the proper-noun and digit-string
    recognizers (module docstring); ``tier="standard"`` (default) is the
    unchanged dictionary+regex behaviour [DECISION-tt.2].
    """

    def __init__(self, dictionary_path: Path, store_path: Path, tier: str = "standard") -> None:
        # Tier is a constructor parameter, not an env var: which replacement
        # set to apply is a per-call-site decision (e.g. "this text is about
        # to leave the machine"), not deployment configuration
        # [DECISION-tt.42].
        if tier not in _TIERS:
            raise ValueError(f"unknown tier {tier!r}; expected one of {_TIERS}")
        self._tier = tier
        self._dictionary_path = dictionary_path
        self._store = JsonlMappingStore(store_path)
        self._literals: list[tuple[str, str]] = []  # (entity_type, literal)
        self._patterns: list[tuple[str, re.Pattern[str]]] = []
        self._load_dictionary(dictionary_path)

    # ------------------------------------------------------------------
    # dictionary loading
    # ------------------------------------------------------------------
    def _load_dictionary(self, path: Path) -> None:
        # Error messages name types/indices, never entry values: the
        # dictionary holds real entity names (21-security red line).
        with path.open(encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        if not isinstance(raw, dict):
            raise ValueError(f"dictionary {str(path)!r} must be a YAML mapping")
        entities = raw.get("entities") or {}
        patterns = raw.get("patterns") or {}
        if not isinstance(entities, dict) or not isinstance(patterns, dict):
            raise ValueError("'entities' and 'patterns' must be mappings of TYPE -> list")
        for section in (entities, patterns):
            for etype, values in section.items():
                if not isinstance(etype, str) or not _TYPE_KEY_RE.fullmatch(etype):
                    raise ValueError(
                        f"entity type key {etype!r} must match [A-Z][A-Z0-9_]* "
                        "(placeholder parseability)"
                    )
                if not isinstance(values, list):
                    raise ValueError(f"dictionary section for type {etype!r} must be a list")
        for etype, values in entities.items():
            for i, literal in enumerate(values):
                if not isinstance(literal, str) or not literal:
                    raise ValueError(f"entities.{etype}[{i}] must be a non-empty string")
                self._literals.append((etype, literal))
        for etype, values in patterns.items():
            for i, pattern in enumerate(values):
                if not isinstance(pattern, str) or not pattern:
                    raise ValueError(f"patterns.{etype}[{i}] must be a non-empty string")
                try:
                    compiled = re.compile(pattern)
                except re.error as exc:
                    # `from None` + .msg only: the chained exception / args
                    # would carry the pattern text (identifying content).
                    raise ValueError(
                        f"patterns.{etype}[{i}] is not a valid regex: {exc.msg}"
                    ) from None
                self._patterns.append((etype, compiled))
        if not self._literals and not self._patterns:
            raise ValueError(f"dictionary {str(path)!r} defines no entities or patterns")

    # ------------------------------------------------------------------
    # span selection
    # ------------------------------------------------------------------
    def _candidate_spans(self, text: str) -> list[tuple[int, int, str]]:
        spans: list[tuple[int, int, str]] = []
        for etype, literal in self._literals:
            for match in re.finditer(re.escape(literal), text):
                spans.append((match.start(), match.end(), etype))
        for etype, compiled in self._patterns:
            for match in compiled.finditer(text):
                if match.start() < match.end():  # ignore zero-width matches
                    spans.append((match.start(), match.end(), etype))
        if self._tier == "strict":
            # Heuristic candidates feed the SAME selection/mapping machinery
            # as dictionary hits: longest-span-wins keeps dictionary entries
            # (usually longer, typed) on top where they overlap, and the
            # shared store gives same-word-same-token across calls/sessions.
            for match in _PROPER_NOUN_RE.finditer(text):
                spans.append((match.start(), match.end(), "PN"))
            for match in _DIGIT_RUN_RE.finditer(text):
                # Enforce the >=2-digits floor over the WHOLE match so that
                # separator-joined groups of single digits ("1-2") are still
                # masked — multi-mask over leak [DECISION-tt.41].
                if sum(ch.isdigit() for ch in match.group(0)) >= 2:
                    spans.append((match.start(), match.end(), "NUM"))
        return spans

    @staticmethod
    def _select_spans(candidates: list[tuple[int, int, str]]) -> list[tuple[int, int, str]]:
        """Resolve overlaps: longest span wins, ties by (start, type); greedy keep.

        The survivors are pairwise non-overlapping, so every replaced
        interval equals its mapping key exactly — partial intersections
        (which break byte-exact restore under presidio's default merge
        strategy, Learn ls.41) are dropped, like REMOVE_INTERSECTIONS.
        """
        ordered = sorted(candidates, key=lambda s: (-(s[1] - s[0]), s[0], s[2]))
        kept: list[tuple[int, int, str]] = []
        for start, end, etype in ordered:
            if all(end <= k_start or start >= k_end for k_start, k_end, _ in kept):
                kept.append((start, end, etype))
        kept.sort(key=lambda s: s[0])
        return kept

    # ------------------------------------------------------------------
    # Redactor protocol
    # ------------------------------------------------------------------
    def redact(self, text: str) -> RedactionResult:
        """Pseudonymize ``text``; placeholders are corpus-stable via the store."""
        # Reversibility guard: if the input already contains a placeholder
        # this store issued, restore(redact(text)) could not reproduce the
        # input byte-exactly. Fail fast; the message carries the placeholder
        # token only (placeholders are not sensitive).
        for match in _PLACEHOLDER_RE.finditer(text):
            if self._store.knows_placeholder(match.group(0)):
                raise ValueError(
                    f"input already contains mapped placeholder {match.group(0)}; "
                    "redact would not be reversible"
                )
        spans = self._select_spans(self._candidate_spans(text))
        parts: list[str] = []
        cursor = 0
        mapping_delta = 0
        for start, end, etype in spans:
            placeholder, created = self._store.placeholder_for(etype, text[start:end])
            mapping_delta += int(created)
            parts.append(text[cursor:start])
            parts.append(placeholder)
            cursor = end
        parts.append(text[cursor:])
        return RedactionResult(
            redacted_text="".join(parts),
            spans_replaced=len(spans),
            mapping_delta=mapping_delta,
        )

    def restore(self, text: str) -> str:
        """Replace known placeholders with their original entity text.

        Unknown ``<TYPE_N>`` tokens (never issued by this store) pass
        through unchanged: restore is lenient on arbitrary text, while
        ``restore(redact(t).redacted_text) == t`` holds byte-exactly for
        this redactor's own output [DECISION-cq.32].
        """

        def _sub(match: re.Match[str]) -> str:
            original = self._store.original_for(match.group(0))
            return original if original is not None else match.group(0)

        return _PLACEHOLDER_RE.sub(_sub, text)

    def __repr__(self) -> str:  # content red line: counts + tier name only
        return (
            f"DictRegexRedactor(tier={self._tier!r}, literals={len(self._literals)}, "
            f"patterns={len(self._patterns)}, store={self._store!r})"
        )
