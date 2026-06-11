"""ProviderDispatcher + ProvenanceLog + dispatch-plan tests (ssMP).

Locks the deterministic assignment rule (DECISION-cq.40: sha256 hex parsed
as a big-endian integer, modulo pool size), the provenance idempotency
semantics (DECISION-cq.42: dedup on (content_hash, task_kind, provider)),
and the zero-network guarantee (DECISION-cq.43: the dispatch-plan command
path never imports a vendor SDK — checked in a fresh subprocess).
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from jcontract.ingest.dispatch import (
    PROVENANCE_FIELDS,
    ProvenanceLog,
    ProviderDispatcher,
)

# Known vectors — sha256 of b"", b"page-1", b"page-2", b"page-3". The
# expected providers are HARD-CODED (not re-derived via int(h, 16) %, which
# would be tautological): if the rule's byte-order or modulus convention
# ever drifts, these literals fail loudly. [DECISION-cq.40]
HASH_EMPTY = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
HASH_PAGE1 = "0eb236e50de35c59c03b63629624351af778cc33fbc55a92254e3c29e58e6255"
HASH_PAGE2 = "0f6724ab77e487b74587299fac8c3336030f4157c86202d5a3f5ca64a6059442"
HASH_PAGE3 = "fe5d32f06cf188ad797ee1a75504e24b41df491e7951d4bcef64b81f78cbdefc"


class TestAssignmentRule:
    def test_known_vectors_pool_of_two(self) -> None:
        d = ProviderDispatcher(["alpha", "beta"])
        # int(hex, 16) % 2 == 1 → index 1; % 2 == 0 → index 0.
        assert d.assign(HASH_EMPTY) == "beta"
        assert d.assign(HASH_PAGE1) == "beta"
        assert d.assign(HASH_PAGE2) == "alpha"
        assert d.assign(HASH_PAGE3) == "alpha"

    def test_known_vectors_pool_of_three(self) -> None:
        d = ProviderDispatcher(["alpha", "beta", "gamma"])
        assert d.assign(HASH_EMPTY) == "beta"
        assert d.assign(HASH_PAGE1) == "beta"
        assert d.assign(HASH_PAGE2) == "beta"
        assert d.assign(HASH_PAGE3) == "alpha"

    def test_deterministic_across_instances(self) -> None:
        # Same input + same pool → same output, including on a fresh
        # instance (no hidden per-instance state).
        first = ProviderDispatcher(["claude", "openai"]).assign(HASH_PAGE1)
        second = ProviderDispatcher(["claude", "openai"]).assign(HASH_PAGE1)
        assert first == second

    def test_pool_order_changes_assignment(self) -> None:
        # Pool ORDER is part of the rule: reordering is a config change
        # (DECISION-cq.40). HASH_EMPTY maps to index 1 in a 2-pool.
        assert ProviderDispatcher(["a", "b"]).assign(HASH_EMPTY) == "b"
        assert ProviderDispatcher(["b", "a"]).assign(HASH_EMPTY) == "a"

    def test_pool_resize_reassigns(self) -> None:
        # Growing the pool changes the modulus — assignments may move
        # (documented re-dispatch semantics, not stable hashing).
        assert ProviderDispatcher(["a", "b"]).assign(HASH_PAGE2) == "a"
        assert ProviderDispatcher(["a", "b", "c"]).assign(HASH_PAGE2) == "b"

    def test_case_and_whitespace_insensitive_hash(self) -> None:
        d = ProviderDispatcher(["a", "b"])
        assert d.assign(HASH_EMPTY.upper()) == d.assign(HASH_EMPTY)
        assert d.assign(f" {HASH_EMPTY}\n") == d.assign(HASH_EMPTY)

    def test_rejects_bad_input(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            ProviderDispatcher([])
        with pytest.raises(ValueError, match="non-empty"):
            ProviderDispatcher(["a", "  "])
        with pytest.raises(ValueError, match="duplicate"):
            ProviderDispatcher(["a", "a"])
        d = ProviderDispatcher(["a", "b"])
        with pytest.raises(ValueError, match="64-char"):
            d.assign("deadbeef")  # truncated digest = different residue
        with pytest.raises(ValueError, match="hexadecimal"):
            d.assign("z" * 64)

    def test_pool_property_is_a_copy(self) -> None:
        d = ProviderDispatcher(["a", "b"])
        d.pool.append("mallory")
        assert d.pool == ["a", "b"]


class TestProvenanceLog:
    def test_schema_and_field_order(self, tmp_path: Path) -> None:
        log = ProvenanceLog(tmp_path / "prov.jsonl")
        assert log.append(
            content_hash=HASH_PAGE1,
            provider="claude",
            task_kind="page-ocr",
            assigned_at="2026-06-11T00:00:00Z",
            notes="unit",
        )
        line = (tmp_path / "prov.jsonl").read_text(encoding="utf-8").splitlines()[0]
        record = json.loads(line)
        assert tuple(record) == PROVENANCE_FIELDS
        assert record == {
            "content_hash": HASH_PAGE1,
            "provider": "claude",
            "assigned_at": "2026-06-11T00:00:00Z",
            "task_kind": "page-ocr",
            "redaction_applied": None,  # reserved ssDI contract field
            "notes": "unit",
        }

    def test_reappend_same_triple_is_noop(self, tmp_path: Path) -> None:
        path = tmp_path / "prov.jsonl"
        log = ProvenanceLog(path)
        assert log.append(content_hash=HASH_PAGE1, provider="claude", task_kind="page-ocr")
        before = path.read_bytes()
        # Same (hash, task_kind, provider) → no write, file byte-identical.
        assert not log.append(content_hash=HASH_PAGE1, provider="claude", task_kind="page-ocr")
        assert path.read_bytes() == before

    def test_dedup_survives_reopen(self, tmp_path: Path) -> None:
        # Idempotency must hold across sessions (断点续跑): a fresh
        # ProvenanceLog reloads seen triples from disk.
        path = tmp_path / "prov.jsonl"
        ProvenanceLog(path).append(content_hash=HASH_PAGE1, provider="claude", task_kind="page-ocr")
        before = path.read_bytes()
        assert not ProvenanceLog(path).append(
            content_hash=HASH_PAGE1, provider="claude", task_kind="page-ocr"
        )
        assert path.read_bytes() == before

    def test_pool_change_appends_new_line(self, tmp_path: Path) -> None:
        # A pool change that reassigns the same page to a different
        # provider is RECORDED (new line), not hidden. [DECISION-cq.42]
        path = tmp_path / "prov.jsonl"
        log = ProvenanceLog(path)
        assert log.append(content_hash=HASH_PAGE1, provider="claude", task_kind="page-ocr")
        assert log.append(content_hash=HASH_PAGE1, provider="openai", task_kind="page-ocr")
        assert len(path.read_text(encoding="utf-8").splitlines()) == 2

    def test_default_timestamp_is_utc_iso(self, tmp_path: Path) -> None:
        log = ProvenanceLog(tmp_path / "prov.jsonl")
        log.append(content_hash=HASH_PAGE2, provider="a", task_kind="t")
        record = json.loads((tmp_path / "prov.jsonl").read_text(encoding="utf-8"))
        assert record["assigned_at"].endswith("Z")
        assert "T" in record["assigned_at"]


SYNTHETIC_PDF = Path("eval/fixtures/synthetic_contract_tqa.pdf")

# Runs dispatch-plan end-to-end in a FRESH interpreter, then asserts no
# vendor/LLM SDK was imported anywhere on the command path — the
# assertion-level zero-network guarantee. [DECISION-cq.43]
_ZERO_NETWORK_PROBE = """
import sys
from typer.testing import CliRunner
from jcontract.cli import app

runner = CliRunner()
result = runner.invoke(
    app,
    ["dispatch-plan", "{pdf}", "--pool", "alpha,beta", "--max-pages", "2"],
)
assert result.exit_code == 0, result.output
banned = {{"anthropic", "openai", "ollama", "llama_parse", "llama_cloud"}}
loaded = banned & {{m.split(".")[0] for m in sys.modules}}
assert not loaded, f"vendor SDKs imported on the dispatch-plan path: {{loaded}}"
print("ZERO_NETWORK_OK")
"""


@pytest.mark.slow
def test_dispatch_plan_zero_network_and_deterministic(tmp_path: Path) -> None:
    assert SYNTHETIC_PDF.exists(), "synthetic fixture missing"
    # noqa rationale: argv is fully program-constructed (sys.executable +
    # a literal -c script; no shell=True, no user input) — same stance as
    # impls/_claude_cli_runner.py.
    proc = subprocess.run(  # noqa: S603
        [sys.executable, "-c", _ZERO_NETWORK_PROBE.format(pdf=SYNTHETIC_PDF)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr or proc.stdout
    assert "ZERO_NETWORK_OK" in proc.stdout


@pytest.mark.slow
def test_dispatch_plan_cli_double_run_byte_identical(tmp_path: Path) -> None:
    from typer.testing import CliRunner

    from jcontract.cli import app

    runner = CliRunner()
    outputs: list[str] = []
    plans: list[bytes] = []
    prov = tmp_path / "prov.jsonl"
    for run in range(2):
        out = tmp_path / f"plan-{run}.jsonl"
        result = runner.invoke(
            app,
            [
                "dispatch-plan",
                str(SYNTHETIC_PDF),
                "--pool",
                "claude,openai",
                "--max-pages",
                "3",
                "--out",
                str(out),
                "--provenance",
                str(prov),
            ],
        )
        assert result.exit_code == 0, result.output
        outputs.append(result.stdout)
        plans.append(out.read_bytes())

    assert outputs[0] == outputs[1]  # terminal table byte-identical
    assert plans[0] == plans[1]  # plan JSONL byte-identical

    # Provenance: 3 pages logged once; the re-run appended nothing.
    records = [
        json.loads(line) for line in prov.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    assert len(records) == 3
    assert all(tuple(r) == PROVENANCE_FIELDS for r in records)
    assert all(r["redaction_applied"] is None for r in records)
