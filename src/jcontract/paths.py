"""Per-collection data layout (Phase 7 SS5).

Each knowledge base (Qdrant collection) gets its own ``data/<collection>/``
subtree so multiple corpora coexist without colliding (the BM25 snapshot,
RefGraph SQLite, ingest checkpoint, eval results, and the profile sidecar
are all per-collection). The OCR / caption caches stay GLOBAL because they
are SHA-256 content-addressed — identical page bytes yield identical text
regardless of which collection requested them.

Before Phase 7 these lived as flat ``data/<file>`` globals. ``migrate-layout``
(see plan_layout_migration / apply_layout_migration) moves the legacy files
into ``data/contract/`` so the existing index isn't orphaned.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

DATA_DIR = Path("data")

# Shared, content-addressed caches (cross-collection safe — do NOT move).
OCR_CACHE_DIR = DATA_DIR / "ocr_cache"
CAPTION_CACHE_DIR = DATA_DIR / "caption_cache"

# Legacy flat filenames that migrate-layout relocates into data/<collection>/.
_LEGACY_NAMES = (
    "chunks_snapshot.jsonl",
    "ref_graph.db",
    "ingest_checkpoint.jsonl",
    "eval-results",
)


@dataclass(frozen=True)
class CollectionPaths:
    """Resolved per-collection artifact paths."""

    root: Path
    chunks_snapshot: Path
    ref_graph: Path
    ingest_checkpoint: Path
    eval_results: Path
    profile_sidecar: Path


def paths_for(collection: str, data_dir: Path = DATA_DIR) -> CollectionPaths:
    """Return the ``data/<collection>/`` artifact paths for a knowledge base."""
    root = data_dir / collection
    return CollectionPaths(
        root=root,
        chunks_snapshot=root / "chunks_snapshot.jsonl",
        ref_graph=root / "ref_graph.db",
        ingest_checkpoint=root / "ingest_checkpoint.jsonl",
        eval_results=root / "eval-results",
        profile_sidecar=root / "profile.txt",
    )


def legacy_files_present(data_dir: Path = DATA_DIR) -> bool:
    """True if any pre-Phase-7 flat ``data/<file>`` artifact still exists."""
    return any((data_dir / name).exists() for name in _LEGACY_NAMES)


def plan_layout_migration(
    collection: str = "contract", data_dir: Path = DATA_DIR
) -> list[tuple[Path, Path]]:
    """Return the (src, dst) moves to relocate legacy flat files into
    ``data/<collection>/``. Only includes files that actually exist."""
    dest = data_dir / collection
    moves: list[tuple[Path, Path]] = []
    for name in _LEGACY_NAMES:
        src = data_dir / name
        if src.exists():
            moves.append((src, dest / name))
    return moves


def apply_layout_migration(
    collection: str = "contract", data_dir: Path = DATA_DIR
) -> list[tuple[Path, Path]]:
    """Execute plan_layout_migration; create the dest dir and move each file.

    Skips a move whose destination already exists (idempotent / safe re-run).
    Returns the moves actually performed.
    """
    moves = plan_layout_migration(collection, data_dir)
    done: list[tuple[Path, Path]] = []
    if not moves:
        return done
    dest_root = data_dir / collection
    dest_root.mkdir(parents=True, exist_ok=True)
    for src, dst in moves:
        if dst.exists():
            continue  # don't clobber an already-migrated artifact
        shutil.move(str(src), str(dst))
        done.append((src, dst))
    return done


def write_profile_sidecar(collection: str, profile_name: str, data_dir: Path = DATA_DIR) -> None:
    """Persist which DomainProfile a collection was ingested under."""
    p = paths_for(collection, data_dir)
    p.root.mkdir(parents=True, exist_ok=True)
    p.profile_sidecar.write_text(profile_name + "\n", encoding="utf-8")


def read_profile_name(collection: str, data_dir: Path = DATA_DIR, default: str = "contract") -> str:
    """Read a collection's bound DomainProfile name; ``default`` if no sidecar."""
    sidecar = paths_for(collection, data_dir).profile_sidecar
    if sidecar.exists():
        name = sidecar.read_text(encoding="utf-8").strip()
        if name:
            return name
    return default
