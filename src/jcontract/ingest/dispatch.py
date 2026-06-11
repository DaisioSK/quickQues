"""ProviderDispatcher — deterministic hash lottery + provenance log (ssMP).

Mechanism-only capability component (DECISION-cq.4): NOT wired into the
ingest pipeline, and this module never talks to any provider. The pool is
a plain list of provider *names* (opaque strings — e.g. ``["claude",
"openai"]``); no vendor SDK is imported, no client is constructed, no
network call happens anywhere in this file. Stdlib-only on purpose — a
zero-network guarantee that tests/test_dispatch.py asserts at import time.

Assignment rule [DECISION-cq.40]:
    ``pool[int(content_hash, 16) % len(pool)]`` — the sha256 hex digest is
    parsed as ONE big-endian integer (``int(hex, 16)``: leftmost hex digit
    most significant), then reduced modulo the pool size. Deterministic
    over true randomness because idempotence is the whole point
    (FORESHADOW-cq.2 design input): re-running a plan re-produces the same
    page→provider mapping, so provider-side caches stay warm and an
    interrupted batch can resume without pages silently switching vendors.
    Pool ORDER participates in the rule — reordering or resizing the pool
    is a config change that reassigns pages and must be recorded
    explicitly (the provenance log does exactly that, DECISION-cq.42).

Provenance log [DECISION-cq.42]:
    Append-only JSONL audit trail, one record per (content_hash,
    task_kind, provider) triple. Idempotent by dedup: re-appending an
    existing triple is a no-op, so re-runs leave the file byte-identical;
    a pool change that maps the same page to a DIFFERENT provider appends
    a new line — the audit trail shows the reassignment instead of hiding
    it. ``redaction_applied`` is a reserved contract field for the ssDI
    redactor's output (always ``None`` until the wiring sprint).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

# Hex digits in a sha256 digest — assign() validates length so a truncated
# hash (different modulus residue!) fails fast instead of mis-assigning.
_SHA256_HEX_LEN = 64

#: JSON object keys of one provenance record, in serialization order.
PROVENANCE_FIELDS = (
    "content_hash",
    "provider",
    "assigned_at",
    "task_kind",
    "redaction_applied",
    "notes",
)


class ProviderDispatcher:
    """Deterministic content-hash → provider-name lottery over a fixed pool.

    ``pool`` is an ordered list of unique provider names. ``assign`` is a
    pure function of ``(content_hash, pool)``: same input, same pool →
    same output, forever (locked by known-vector unit tests). Nothing
    here knows what a provider *is* — instantiation/dispatch of real
    vendors belongs to a future wiring sprint (FORESHADOW-cq.2).
    """

    def __init__(self, pool: list[str]) -> None:
        if not pool:
            raise ValueError("provider pool must not be empty")
        cleaned = [name.strip() for name in pool]
        if any(not name for name in cleaned):
            raise ValueError("provider pool entries must be non-empty names")
        if len(set(cleaned)) != len(cleaned):
            # Duplicates would silently skew the lottery weights — reject
            # rather than guess whether weighting was intended.
            raise ValueError(f"provider pool contains duplicate names: {cleaned}")
        self._pool = cleaned

    @property
    def pool(self) -> list[str]:
        """The (ordered) pool — a copy, so callers can't mutate the rule."""
        return list(self._pool)

    def assign(self, content_hash: str) -> str:
        """Map a sha256 hex digest to a provider name. [DECISION-cq.40]

        Big-endian convention: ``int(content_hash, 16)`` reads the digest
        left-to-right, most-significant hex digit first.
        """
        digest = content_hash.strip().lower()
        if len(digest) != _SHA256_HEX_LEN:
            raise ValueError(
                f"content_hash must be a {_SHA256_HEX_LEN}-char sha256 hex digest, "
                f"got {len(digest)} chars"
            )
        try:
            value = int(digest, 16)
        except ValueError as exc:
            raise ValueError("content_hash must be hexadecimal") from exc
        return self._pool[value % len(self._pool)]


@dataclass(frozen=True)
class ProvenanceRecord:
    """One audit-trail entry: which provider a content hash was assigned to.

    ``redaction_applied`` is the reserved ssDI contract field — ``True``/
    ``False`` once the redactor is wired in front of real dispatch; always
    ``None`` while dispatch stays dry-run-only (DECISION-cq.4).
    """

    content_hash: str
    provider: str
    assigned_at: str
    task_kind: str
    redaction_applied: bool | None
    notes: str

    def to_json(self) -> str:
        """Serialize with a stable key order (PROVENANCE_FIELDS)."""
        data = asdict(self)
        return json.dumps({k: data[k] for k in PROVENANCE_FIELDS}, ensure_ascii=False)


class ProvenanceLog:
    """Append-only JSONL provenance recorder with idempotent re-append.

    Dedup key = ``(content_hash, task_kind, provider)`` [DECISION-cq.42]:
    re-running the same plan against the same pool appends nothing (the
    file — timestamps included — stays byte-identical); a pool change
    that reassigns a page appends a NEW line, so config changes are
    explicitly recorded rather than overwritten. Timestamps live ONLY
    here, never in the plan output — that is what keeps `dispatch-plan`
    double-runs byte-identical while the audit trail still says when.
    """

    def __init__(self, path: Path) -> None:
        self._path = path
        self._seen: set[tuple[str, str, str]] = set()
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                rec = json.loads(line)
                self._seen.add((rec["content_hash"], rec["task_kind"], rec["provider"]))

    @property
    def path(self) -> Path:
        return self._path

    def append(
        self,
        *,
        content_hash: str,
        provider: str,
        task_kind: str,
        redaction_applied: bool | None = None,
        notes: str = "",
        assigned_at: str | None = None,
    ) -> bool:
        """Record one assignment; return True if a line was written.

        ``assigned_at`` defaults to the system clock (UTC, ISO-8601,
        seconds precision) — caller may inject a timestamp for
        reproducible tests. Returns False (and writes nothing) when the
        ``(content_hash, task_kind, provider)`` triple is already logged.
        """
        key = (content_hash, task_kind, provider)
        if key in self._seen:
            return False
        record = ProvenanceRecord(
            content_hash=content_hash,
            provider=provider,
            assigned_at=assigned_at
            or datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z"),
            task_kind=task_kind,
            redaction_applied=redaction_applied,
            notes=notes,
        )
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._path.open("a", encoding="utf-8") as fh:
            fh.write(record.to_json() + "\n")
        self._seen.add(key)
        return True
