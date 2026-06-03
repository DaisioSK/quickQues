"""SQLite-backed RefGraph implementation (sub-sprint p1.8-ssE).

What:
    Cross-document entity mention index. Given a stream of ``Chunk``
    objects whose metadata (``drawing_refs``, ``clause_refs``,
    ``question_no``, ``section_path``, ``revision``) has already been
    populated by the chunker, this class persists the bipartite graph
    ``(chunk) --mentions--> (entity)`` to a tiny SQLite file so it can
    answer:

      * ``mentions_of("drawing", "T/PRJ/CWD/WS/2101A")`` →
        every chunk in every PDF that names that drawing
      * ``entities_in("ContractDEMO(1of9)TQA:42:7")`` →
        every entity referenced by that single chunk
      * ``stats()`` → corpus-level counts for ingest reporting + eval

Why SQLite (DECISION):
    The prototype's per-corpus mention table is ~10^4 rows (a few PDFs ×
    a few hundred chunks × a handful of refs each). SQLite is:
      1. Already in CPython stdlib — no new dep
      2. ACID, single-file, easy to ship in ``data/`` alongside Qdrant
      3. Indexable on ``(type, value)`` for sub-ms lookups at our scale
      4. Trivially replaceable later (Phase 3+) with Neo4j / DuckDB if
         the workload grows graph-traversal-heavy
    The 8-question dep gate (dev-contract/24-domain-deps-env): stdlib —
    auto-pass.

Why we don't re-extract entities from ``chunk.text``:
    ``impls/qa_chunker.py`` already runs the regex catalogue and stores
    results in ``chunk.drawing_refs`` / ``chunk.clause_refs`` /
    ``chunk.question_no``. Duplicating the regexes here would be a
    second source of truth bug-magnet. We consume the structured fields
    and index those. If the chunker improves its extraction, this layer
    benefits automatically.

Why denormalised projection table ``chunks``:
    ``mentions_of`` needs to return enough provenance for a UI to render
    "file foo.pdf, page 42" without joining back to Qdrant on every
    query. A 4-column denormalised projection (id, file, page,
    chunk_type) is ~30 bytes/chunk — cheap. Full chunk text stays in
    the vector store; we don't duplicate it (size + the
    single-source-of-truth principle).

Idempotency:
    ``index()`` re-applied to the same chunks does not duplicate rows.
    Both ``entities`` and ``mentions`` use UNIQUE constraints with
    INSERT OR IGNORE; the ``chunks`` projection uses INSERT OR REPLACE
    so a chunk that's been re-parsed (e.g. new section_path) updates
    in place.

Context:
    Phase 1.8 ssE. Consumed by ingest/pipeline.py (wiring is the
    integrator's job — see TODO at module bottom).
"""

from __future__ import annotations

import contextlib
import sqlite3
from pathlib import Path

from jcontract.interfaces.schema import Chunk

# Entity-type tags. Kept here (not in schema.py) because they're an
# implementation detail of THIS impl; a Neo4j impl might use different
# labels. Callers query via these constants for type safety.
ENTITY_DRAWING = "drawing"
ENTITY_CLAUSE = "clause"
ENTITY_QUESTION_NO = "question_no"
ENTITY_SECTION = "section"
ENTITY_REVISION = "revision"


# ---------------------------------------------------------------------------
# Schema DDL. Kept as a module-level string so the schema is reviewable in
# one glance and so test fixtures can spin up an in-memory DB with the
# exact same shape.
# ---------------------------------------------------------------------------
_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS entities (
    id    INTEGER PRIMARY KEY AUTOINCREMENT,
    type  TEXT NOT NULL,
    value TEXT NOT NULL,
    UNIQUE(type, value)
);

CREATE TABLE IF NOT EXISTS chunks (
    id         TEXT PRIMARY KEY,
    file       TEXT NOT NULL,
    page       INTEGER NOT NULL,
    chunk_type TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS mentions (
    chunk_id  TEXT NOT NULL,
    entity_id INTEGER NOT NULL,
    UNIQUE(chunk_id, entity_id),
    FOREIGN KEY (chunk_id)  REFERENCES chunks(id),
    FOREIGN KEY (entity_id) REFERENCES entities(id)
);

CREATE INDEX IF NOT EXISTS idx_entities_type_value ON entities(type, value);
CREATE INDEX IF NOT EXISTS idx_mentions_entity     ON mentions(entity_id);
CREATE INDEX IF NOT EXISTS idx_mentions_chunk      ON mentions(chunk_id);
"""


class SqliteRefGraph:
    """SQLite-backed cross-document entity mention index.

    Entity types tracked (from Chunk metadata, no re-parsing):

      * ``drawing``     ← every value in ``chunk.drawing_refs``
      * ``clause``      ← every value in ``chunk.clause_refs``
      * ``question_no`` ← ``chunk.question_no`` (singular per chunk)
      * ``section``     ← top-level ``Section N`` parsed from
                          ``chunk.section_path`` (the ``"Section 7"`` in
                          ``"Section 7 > Clause 7.3"``). Clauses are
                          already covered by ``clause_refs``, so we only
                          take the Section half here to avoid duplication.
      * ``revision``    ← ``chunk.revision`` (e.g. ``"Rev A"``)

    Implements ``RefGraph`` Protocol.
    """

    def __init__(self, db_path: Path = Path("data/ref_graph.db")) -> None:
        # Parent dir auto-create so callers don't have to mkdir before
        # constructing. Matches QdrantStore's pattern (impls/qdrant_store).
        # Special-case ``:memory:`` so tests can spin up a transient DB.
        self.db_path = db_path
        if str(db_path) != ":memory:":
            db_path.parent.mkdir(parents=True, exist_ok=True)
        # check_same_thread=False: the ingest pipeline currently runs
        # single-threaded but the eval runner may dispatch in a worker
        # pool. SQLite serializes writes internally; we keep our
        # connection lock-free at the Python level.
        self._conn = sqlite3.connect(str(db_path), check_same_thread=False)
        # Foreign keys are off by default in SQLite; enable for safety
        # even though our INSERT order avoids dangling refs.
        self._conn.execute("PRAGMA foreign_keys = ON;")
        self._conn.executescript(_SCHEMA_SQL)
        self._conn.commit()

    # -- write path ---------------------------------------------------------

    def index(self, chunks: list[Chunk]) -> None:
        """Insert/update entity mentions for these chunks. Idempotent.

        For each chunk we:
          1. Upsert the chunk projection (id, file, page, chunk_type).
          2. For every entity reference in metadata, upsert the entity
             then upsert the (chunk_id, entity_id) edge.

        Empty-metadata chunks (no drawing/clause/question_no/revision/
        section) are still recorded in ``chunks`` so ``entities_in()``
        returns ``[]`` (vs raising) for known chunks. This matches how
        ``mentions_of`` for an unknown entity returns ``[]``.
        """
        if not chunks:
            return
        cur = self._conn.cursor()
        try:
            for chunk in chunks:
                # 1. Project the chunk. INSERT OR REPLACE so a re-indexed
                # chunk with updated metadata (e.g. new section_path) is
                # not stale.
                cur.execute(
                    "INSERT OR REPLACE INTO chunks (id, file, page, chunk_type) "
                    "VALUES (?, ?, ?, ?)",
                    (chunk.id, chunk.file, chunk.page, chunk.chunk_type),
                )
                # 2. Gather all (type, value) entities from metadata.
                for ent_type, ent_value in _entities_from_chunk(chunk):
                    # Entity upsert: try insert; on UNIQUE clash, look up.
                    # Two-step instead of ON CONFLICT to keep the lookup
                    # path (entity_id) explicit and compatible with old
                    # SQLite versions that lack RETURNING (we don't ship
                    # ours but stay portable).
                    cur.execute(
                        "INSERT OR IGNORE INTO entities (type, value) VALUES (?, ?)",
                        (ent_type, ent_value),
                    )
                    row = cur.execute(
                        "SELECT id FROM entities WHERE type = ? AND value = ?",
                        (ent_type, ent_value),
                    ).fetchone()
                    entity_id = int(row[0])
                    # Mention edge: idempotent via UNIQUE(chunk_id, entity_id).
                    cur.execute(
                        "INSERT OR IGNORE INTO mentions (chunk_id, entity_id) VALUES (?, ?)",
                        (chunk.id, entity_id),
                    )
            self._conn.commit()
        except Exception:
            # If anything goes wrong mid-batch, leave the DB in the
            # pre-batch state — callers can retry safely.
            self._conn.rollback()
            raise

    # -- read path ----------------------------------------------------------

    def mentions_of(self, entity_type: str, entity_value: str) -> list[Chunk]:
        """All chunks that mention this entity.

        Returns minimal ``Chunk`` instances reconstructed from the
        denormalised projection: ``id``, ``file``, ``page``, ``chunk_type``
        are real; ``text`` is empty, refs/metadata are defaults. Callers
        needing the full chunk should look it up by ``id`` in the vector
        store.
        """
        cur = self._conn.cursor()
        rows = cur.execute(
            "SELECT c.id, c.file, c.page, c.chunk_type "
            "FROM chunks c "
            "JOIN mentions m ON m.chunk_id = c.id "
            "JOIN entities e ON e.id = m.entity_id "
            "WHERE e.type = ? AND e.value = ? "
            "ORDER BY c.file, c.page, c.id",
            (entity_type, entity_value),
        ).fetchall()
        return [
            Chunk(
                id=str(row[0]),
                text="",  # not stored here; caller can rehydrate from vector store
                file=str(row[1]),
                page=int(row[2]),
                chunk_type=row[3],
            )
            for row in rows
        ]

    def entities_in(self, chunk_id: str) -> list[tuple[str, str]]:
        """All ``(type, value)`` entities referenced by this chunk."""
        cur = self._conn.cursor()
        rows = cur.execute(
            "SELECT e.type, e.value "
            "FROM entities e "
            "JOIN mentions m ON m.entity_id = e.id "
            "WHERE m.chunk_id = ? "
            "ORDER BY e.type, e.value",
            (chunk_id,),
        ).fetchall()
        return [(str(row[0]), str(row[1])) for row in rows]

    def stats(self) -> dict[str, int]:
        """Corpus counts. Useful for ingest reporting + eval assertions.

        Always returns ``chunks``, ``entities``, ``mentions`` plus a
        per-type breakdown (``drawings``, ``clauses``, ``question_nos``,
        ``sections``, ``revisions``). Types with zero rows are still
        included (value 0) so the key set is stable for tests.
        """
        cur = self._conn.cursor()
        stats: dict[str, int] = {
            "chunks": int(cur.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]),
            "entities": int(cur.execute("SELECT COUNT(*) FROM entities").fetchone()[0]),
            "mentions": int(cur.execute("SELECT COUNT(*) FROM mentions").fetchone()[0]),
        }
        # Per-type entity count (NOT mention count) — answers "how many
        # distinct drawings do we know about?", which is what users tend
        # to ask in dashboards.
        type_to_key = {
            ENTITY_DRAWING: "drawings",
            ENTITY_CLAUSE: "clauses",
            ENTITY_QUESTION_NO: "question_nos",
            ENTITY_SECTION: "sections",
            ENTITY_REVISION: "revisions",
        }
        for ent_type, key in type_to_key.items():
            row = cur.execute(
                "SELECT COUNT(*) FROM entities WHERE type = ?", (ent_type,)
            ).fetchone()
            stats[key] = int(row[0])
        return stats

    def close(self) -> None:
        """Release the DB connection. Safe to call multiple times."""
        # SQLite connections raise ProgrammingError on double-close;
        # suppress so context-manager + atexit hooks don't surface it.
        with contextlib.suppress(sqlite3.ProgrammingError):
            self._conn.close()


# ---------------------------------------------------------------------------
# Helpers (module-private).
# ---------------------------------------------------------------------------


def _entities_from_chunk(chunk: Chunk) -> list[tuple[str, str]]:
    """Flatten a Chunk's metadata into ``[(type, value), ...]``.

    Drops empty / whitespace-only values silently — a chunk whose
    metadata fields are all unset/empty contributes zero entities (it
    still gets a row in ``chunks`` so we know it exists). De-duplicates
    within the chunk: re-listing the same drawing twice in
    ``drawing_refs`` should not double-count.
    """
    out: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()

    def add(ent_type: str, value: str | None) -> None:
        if value is None:
            return
        v = value.strip()
        if not v:
            return
        key = (ent_type, v)
        if key in seen:
            return
        seen.add(key)
        out.append(key)

    for d in chunk.drawing_refs:
        add(ENTITY_DRAWING, d)
    for c in chunk.clause_refs:
        add(ENTITY_CLAUSE, c)
    add(ENTITY_QUESTION_NO, chunk.question_no)
    add(ENTITY_REVISION, chunk.revision)

    # Section: only emit the top-level "Section N" portion of
    # ``section_path`` (the ``"Section 7"`` in ``"Section 7 > Clause 7.3"``).
    # The Clause portion, if present, is already represented in
    # ``clause_refs`` for any chunk whose body references it; the header
    # alone doesn't merit a separate "clause" mention.
    section = _top_level_section(chunk.section_path)
    add(ENTITY_SECTION, section)

    return out


def _top_level_section(section_path: str | None) -> str | None:
    """Return e.g. ``"Section 7"`` from ``"Section 7 > Clause 7.3"``.

    Returns ``None`` if the path is empty or doesn't start with a
    Section header (so a path that's just ``"Clause 7.3"`` yields no
    section entity — the clause is captured by ``clause_refs`` instead).
    """
    if not section_path:
        return None
    # The chunker uses " > " as the level separator.
    first = section_path.split(" > ", 1)[0].strip()
    if not first:
        return None
    # Only emit when the leading segment is actually a "Section X"
    # header. This keeps the entity vocabulary clean.
    if first.lower().startswith("section "):
        return first
    return None


# ---------------------------------------------------------------------------
# TODO for integrator (Phase 1.8 wire-up):
#   * ingest/pipeline.py: after vector_store.add(chunks) and
#     keyword_index.add(chunks), call ref_graph.index(chunks).
#   * cli.py: add ``jcontract refs <type> <value>`` subcommand that
#     constructs SqliteRefGraph(data/ref_graph.db) and prints
#     mentions_of(...) as a table.
#   * config.py: optional path override
#     ``JCONTRACT_REF_GRAPH_PATH=data/ref_graph.db``.
# ---------------------------------------------------------------------------
