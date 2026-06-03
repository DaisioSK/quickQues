"""Tests for ``impls/sqlite_ref_graph.SqliteRefGraph``.

Covers the contract from sub-sprint p1.8-ssE: idempotency,
multi-entity-type extraction, cross-document mentions, stats counts,
empty-metadata handling, persistence across close/reopen.

We use ``tmp_path`` for on-disk DBs (the only way to exercise the
persistence test); the rest could run on ``:memory:`` but staying
consistent with on-disk paths keeps the tests honest about pragmas
and parent-dir creation.
"""

from __future__ import annotations

from pathlib import Path

from jcontract.impls.sqlite_ref_graph import SqliteRefGraph
from jcontract.interfaces.schema import Chunk


def _chunk(
    chunk_id: str,
    file: str = "Contract DEMO(1of9) TQA.pdf",
    page: int = 1,
    *,
    drawing_refs: list[str] | None = None,
    clause_refs: list[str] | None = None,
    question_no: str | None = None,
    section_path: str | None = None,
    revision: str | None = None,
    chunk_type: str = "qa_pair",
) -> Chunk:
    """Build a Chunk for tests; only the metadata fields the RefGraph
    consumes need to be set, so this helper hides the boilerplate."""
    return Chunk(
        id=chunk_id,
        text="(elided)",
        file=file,
        page=page,
        chunk_type=chunk_type,  # type: ignore[arg-type]
        drawing_refs=drawing_refs or [],
        clause_refs=clause_refs or [],
        question_no=question_no,
        section_path=section_path,
        revision=revision,
    )


def test_index_then_mentions_of_returns_correct_chunks(tmp_path: Path) -> None:
    """``mentions_of`` returns every chunk that lists the entity."""
    g = SqliteRefGraph(db_path=tmp_path / "ref.db")
    chunks = [
        _chunk("doc:1:0", page=1, drawing_refs=["T/PRJ/CWD/WS/2101A"]),
        _chunk("doc:2:0", page=2, drawing_refs=["T/PRJ/CWD/WS/2101A", "T/PRJ/X/Y/3000B"]),
        _chunk("doc:3:0", page=3, drawing_refs=["T/PRJ/X/Y/3000B"]),
    ]
    g.index(chunks)

    hits_2101A = g.mentions_of("drawing", "T/PRJ/CWD/WS/2101A")
    assert {h.id for h in hits_2101A} == {"doc:1:0", "doc:2:0"}

    hits_3000B = g.mentions_of("drawing", "T/PRJ/X/Y/3000B")
    assert {h.id for h in hits_3000B} == {"doc:2:0", "doc:3:0"}

    # Provenance fields round-trip through the projection table.
    h = hits_2101A[0]
    assert h.file == "Contract DEMO(1of9) TQA.pdf"
    assert h.page in (1, 2)
    assert h.chunk_type == "qa_pair"
    g.close()


def test_index_is_idempotent(tmp_path: Path) -> None:
    """Re-indexing the same chunks does not duplicate rows."""
    g = SqliteRefGraph(db_path=tmp_path / "ref.db")
    chunks = [
        _chunk(
            "doc:1:0",
            drawing_refs=["T/PRJ/CWD/WS/2101A"],
            clause_refs=["7.3"],
            question_no="ACME/TRACKWORK/16",
        ),
    ]
    g.index(chunks)
    stats_after_first = g.stats()

    # Run it twice more — totals MUST stay identical.
    g.index(chunks)
    g.index(chunks)
    stats_after_third = g.stats()

    assert stats_after_first == stats_after_third
    # And the lookup still returns exactly one chunk (not three).
    assert len(g.mentions_of("drawing", "T/PRJ/CWD/WS/2101A")) == 1
    g.close()


def test_mentions_of_no_results_returns_empty_list(tmp_path: Path) -> None:
    """Looking up an unknown entity returns ``[]`` rather than raising."""
    g = SqliteRefGraph(db_path=tmp_path / "ref.db")
    g.index([_chunk("doc:1:0", drawing_refs=["T/PRJ/CWD/WS/2101A"])])

    assert g.mentions_of("drawing", "DOES/NOT/EXIST") == []
    assert g.mentions_of("clause", "999.999") == []
    # Unknown type also returns empty — we don't validate the type
    # against an allowlist (callers use module-level constants).
    assert g.mentions_of("not_a_real_type", "anything") == []
    g.close()


def test_entities_in_returns_all_types(tmp_path: Path) -> None:
    """A chunk carrying drawing + clause + question_no surfaces all 3."""
    g = SqliteRefGraph(db_path=tmp_path / "ref.db")
    chunk = _chunk(
        "doc:1:0",
        drawing_refs=["T/PRJ/CWD/WS/2101A"],
        clause_refs=["7.3"],
        question_no="ACME/TRACKWORK/16",
        section_path="Section 7 > Clause 7.3",
        revision="Rev A",
    )
    g.index([chunk])

    entities = g.entities_in("doc:1:0")
    # Expect 5 entries: 1 drawing, 1 clause, 1 question_no, 1 section, 1 revision.
    types = {t for t, _ in entities}
    assert types == {"drawing", "clause", "question_no", "section", "revision"}
    assert ("drawing", "T/PRJ/CWD/WS/2101A") in entities
    assert ("clause", "7.3") in entities
    assert ("question_no", "ACME/TRACKWORK/16") in entities
    # Section path keeps only the top-level "Section 7" half.
    assert ("section", "Section 7") in entities
    assert ("revision", "Rev A") in entities
    g.close()


def test_mentions_across_multiple_files(tmp_path: Path) -> None:
    """The same drawing referenced from two files surfaces both chunks."""
    g = SqliteRefGraph(db_path=tmp_path / "ref.db")
    g.index(
        [
            _chunk(
                "a:1:0",
                file="Contract DEMO(1of9) TQA.pdf",
                page=1,
                drawing_refs=["T/PRJ/CWD/WS/2101A"],
            ),
            _chunk(
                "b:5:0",
                file="Contract DEMO(2of9) Consol.pdf",
                page=5,
                drawing_refs=["T/PRJ/CWD/WS/2101A"],
            ),
        ]
    )

    hits = g.mentions_of("drawing", "T/PRJ/CWD/WS/2101A")
    files = {h.file for h in hits}
    assert files == {
        "Contract DEMO(1of9) TQA.pdf",
        "Contract DEMO(2of9) Consol.pdf",
    }
    g.close()


def test_stats_returns_correct_counts(tmp_path: Path) -> None:
    """``stats()`` reports total chunks/entities/mentions + per-type counts."""
    g = SqliteRefGraph(db_path=tmp_path / "ref.db")
    g.index(
        [
            _chunk(
                "a:1:0",
                drawing_refs=["DWG_A", "DWG_B"],
                clause_refs=["7.3"],
                question_no="Q1",
                section_path="Section 7",
            ),
            _chunk(
                "a:2:0",
                drawing_refs=["DWG_A"],  # duplicate drawing entity
                clause_refs=["7.4"],
                revision="Rev B",
            ),
        ]
    )

    s = g.stats()
    assert s["chunks"] == 2
    # Distinct entities: 2 drawings + 2 clauses + 1 question_no + 1 section + 1 revision = 7.
    assert s["entities"] == 7
    # Mentions: chunk a:1:0 has 5 (2 drawings + 1 clause + 1 question_no + 1 section);
    #           chunk a:2:0 has 3 (1 drawing + 1 clause + 1 revision).
    # Total = 8.
    assert s["mentions"] == 8
    assert s["drawings"] == 2
    assert s["clauses"] == 2
    assert s["question_nos"] == 1
    assert s["sections"] == 1
    assert s["revisions"] == 1
    g.close()


def test_indexing_empty_metadata_skips_silently(tmp_path: Path) -> None:
    """A chunk with no refs registers in ``chunks`` but adds 0 entities."""
    g = SqliteRefGraph(db_path=tmp_path / "ref.db")
    g.index([_chunk("doc:1:0")])  # all metadata fields default / empty

    s = g.stats()
    assert s["chunks"] == 1
    assert s["entities"] == 0
    assert s["mentions"] == 0
    # And ``entities_in`` returns [] rather than raising KeyError.
    assert g.entities_in("doc:1:0") == []
    g.close()


def test_close_then_reopen_persists_data(tmp_path: Path) -> None:
    """Data survives close + reopen against the same DB file."""
    db_path = tmp_path / "ref.db"

    g1 = SqliteRefGraph(db_path=db_path)
    g1.index(
        [
            _chunk(
                "doc:1:0",
                drawing_refs=["T/PRJ/CWD/WS/2101A"],
                question_no="ACME/TRACKWORK/16",
            )
        ]
    )
    stats_first = g1.stats()
    g1.close()

    # Re-open. Schema is IF NOT EXISTS so no clash; data must persist.
    g2 = SqliteRefGraph(db_path=db_path)
    assert g2.stats() == stats_first
    hits = g2.mentions_of("question_no", "ACME/TRACKWORK/16")
    assert len(hits) == 1
    assert hits[0].id == "doc:1:0"
    g2.close()


def test_close_is_safe_to_call_twice(tmp_path: Path) -> None:
    """Double-close doesn't raise — matters for context-manager + atexit."""
    g = SqliteRefGraph(db_path=tmp_path / "ref.db")
    g.close()
    g.close()  # no exception
