"""Redactor Protocol — reversible pseudonymization (Layer 0, ssDI).

Mechanism-only capability component (DECISION-cq.4): NOT wired into the
ingest/answer pipelines. Implementations replace sensitive entity mentions
with stable ``<TYPE_N>`` placeholders and can restore the original text
byte-exactly from a persistent mapping store.

Security contract (dev-contract/21, High-Risk mapping data):
``RedactionResult`` deliberately carries **counts, never mapping content**
— the entity->placeholder mapping is the restore key and must never surface
in results, logs, exception messages, or ``repr`` [DECISION-cq.31].
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class RedactionResult:
    """Outcome of one :meth:`Redactor.redact` call.

    ``redacted_text`` is the pseudonymized text (safe to print/log);
    ``spans_replaced`` counts the text spans substituted in this call;
    ``mapping_delta`` counts the NEW entity->placeholder entries this call
    persisted to the mapping store (0 = every entity was already known,
    i.e. placeholders are corpus-stable across calls/sessions).
    """

    redacted_text: str
    spans_replaced: int
    mapping_delta: int


class Redactor(Protocol):
    """Reversible pseudonymizer over plain text.

    ``redact`` must be deterministic for a given (dictionary, store) state:
    the same entity always maps to the same placeholder, across pages and
    across sessions. ``restore(redact(text).redacted_text)`` must equal
    ``text`` byte-exactly (utf-8) — the REMOVE_INTERSECTIONS-equivalent
    span-conflict semantics this requires are part of the contract
    [DECISION-ls.41, DECISION-cq.32].
    """

    def redact(self, text: str) -> RedactionResult: ...

    def restore(self, text: str) -> str: ...
