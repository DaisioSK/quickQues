"""Tests for per-collection data layout + migrate-layout (Phase 7 SS5)."""

from __future__ import annotations

from pathlib import Path

from jcontract.paths import (
    apply_layout_migration,
    legacy_files_present,
    paths_for,
    plan_layout_migration,
    read_profile_name,
    write_profile_sidecar,
)


def test_paths_for_is_per_collection(tmp_path: Path) -> None:
    cp = paths_for("finance", data_dir=tmp_path)
    assert cp.root == tmp_path / "finance"
    assert cp.chunks_snapshot == tmp_path / "finance" / "chunks_snapshot.jsonl"
    assert cp.ref_graph == tmp_path / "finance" / "ref_graph.db"
    assert cp.eval_results == tmp_path / "finance" / "eval-results"
    assert cp.profile_sidecar == tmp_path / "finance" / "profile.txt"
    # Two collections never share a path.
    assert paths_for("contract", data_dir=tmp_path).root != cp.root


def test_no_legacy_files_is_noop(tmp_path: Path) -> None:
    assert legacy_files_present(tmp_path) is False
    assert plan_layout_migration("contract", data_dir=tmp_path) == []
    assert apply_layout_migration("contract", data_dir=tmp_path) == []


def test_migration_moves_legacy_into_collection(tmp_path: Path) -> None:
    # Seed pre-Phase-7 flat layout.
    (tmp_path / "chunks_snapshot.jsonl").write_text("{}", encoding="utf-8")
    (tmp_path / "ref_graph.db").write_text("db", encoding="utf-8")
    (tmp_path / "eval-results").mkdir()
    (tmp_path / "eval-results" / "run.json").write_text("{}", encoding="utf-8")

    assert legacy_files_present(tmp_path) is True
    plan = plan_layout_migration("contract", data_dir=tmp_path)
    assert {src.name for src, _ in plan} == {
        "chunks_snapshot.jsonl",
        "ref_graph.db",
        "eval-results",
    }

    done = apply_layout_migration("contract", data_dir=tmp_path)
    assert len(done) == 3
    # Files now live under data/contract/ and the legacy locations are gone.
    assert (tmp_path / "contract" / "chunks_snapshot.jsonl").exists()
    assert (tmp_path / "contract" / "eval-results" / "run.json").exists()
    assert not (tmp_path / "chunks_snapshot.jsonl").exists()
    # Idempotent: a second run finds nothing to move.
    assert apply_layout_migration("contract", data_dir=tmp_path) == []


def test_profile_sidecar_round_trip(tmp_path: Path) -> None:
    # Default when no sidecar.
    assert read_profile_name("finance", data_dir=tmp_path) == "contract"
    write_profile_sidecar("finance", "document", data_dir=tmp_path)
    assert read_profile_name("finance", data_dir=tmp_path) == "document"
