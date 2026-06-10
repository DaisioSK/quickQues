"""Unit + integration tests for the indexing layer (Phase 1 S1.1 ssB).

What:
    Covers FastEmbedEmbedder, QdrantStore, and Bm25Index against their
    Protocol contracts (interfaces/embedding.py, vector_store.py,
    keyword.py).

Why split unit vs integration:
    - Embedder: real model exercise IS the unit test — there's no useful
      mock (we need to assert dim and determinism, both of which require
      the ONNX runtime). Costly first run (~1GB download) but cached.
    - Qdrant: requires a running ``docker-compose up -d qdrant``. Tests
      use ``skipif(_qdrant_reachable())`` so they self-skip cleanly when
      Qdrant is down. We deliberately do NOT use a ``pytest.mark.integration``
      tag because the repo runs pytest with ``--strict-markers`` and the
      marker registry is in pyproject.toml which is out of scope for this
      sub-sprint. The skipif probe is the cleaner gate anyway.
    - BM25: pure in-memory, no skipif needed.
"""

from __future__ import annotations

import socket
from contextlib import closing

import pytest

from jcontract.impls.bm25_index import Bm25Index
from jcontract.impls.fastembed_embedder import (
    _MODEL_DIMS,
    DEFAULT_MODEL,
    FastEmbedEmbedder,
)
from jcontract.impls.qdrant_store import QdrantStore, _point_uuid
from jcontract.interfaces.schema import Chunk


def _qdrant_reachable(host: str = "localhost", port: int = 6333) -> bool:
    """TCP probe so the test can self-skip cleanly when Qdrant is down."""
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as sock:
        sock.settimeout(0.5)
        try:
            sock.connect((host, port))
            return True
        except OSError:
            return False


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def embedder() -> FastEmbedEmbedder:
    """Single embedder per module — model download is the slow part."""
    return FastEmbedEmbedder()


@pytest.fixture
def sample_chunks() -> list[Chunk]:
    """A small bilingual corpus exercising the j-contract domain vocab."""
    return [
        Chunk(
            id="doc.pdf:1:0",
            text="The Trackwork Contractor is responsible for waterproofing at the pier.",
            file="doc.pdf",
            page=1,
            chunk_type="paragraph",
            section_path="Section 7 > Clause 7.3",
            clause_refs=["7.3"],
        ),
        Chunk(
            id="doc.pdf:2:0",
            text="桥梁防水责任方为轨道工程承建商，相关图纸编号 T/PRJ/CWD/WS/2101A。",
            file="doc.pdf",
            page=2,
            chunk_type="paragraph",
            drawing_refs=["T/PRJ/CWD/WS/2101A"],
        ),
        Chunk(
            id="doc.pdf:3:0",
            text="Concrete grade C40/20 applies to all structural elements per Clause 5.1.",
            file="doc.pdf",
            page=3,
            chunk_type="paragraph",
            clause_refs=["5.1"],
        ),
    ]


# --------------------------------------------------------------------------- #
# FastEmbedEmbedder
# --------------------------------------------------------------------------- #


def test_fastembed_dim_matches(embedder: FastEmbedEmbedder) -> None:
    """Embedded vector length equals declared ``dim``."""
    vecs = embedder.embed(["hello world"])
    assert len(vecs) == 1
    assert len(vecs[0]) == embedder.dim
    # And dim matches the static table — guards against silent model swap.
    assert embedder.dim == _MODEL_DIMS[DEFAULT_MODEL]


def test_fastembed_cache_dir_persistent_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """Cache resolves to ~/.cache/fastembed, NOT the reboot-wiped tempdir.

    # Why: fastembed's own default is <tempdir>/fastembed_cache — every
    # reboot then re-downloads ~1GB of weights (P1Fixes 2026-06-10).
    """
    import tempfile

    from jcontract.impls.fastembed_embedder import _resolve_cache_dir

    monkeypatch.delenv("FASTEMBED_CACHE_PATH", raising=False)
    resolved = _resolve_cache_dir()
    assert "fastembed" in resolved
    assert not resolved.startswith(tempfile.gettempdir())
    assert ".cache" in resolved


def test_fastembed_cache_dir_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    """FASTEMBED_CACHE_PATH wins over the built-in default when set."""
    from jcontract.impls.fastembed_embedder import _resolve_cache_dir

    monkeypatch.setenv("FASTEMBED_CACHE_PATH", "/custom/model-cache")
    assert _resolve_cache_dir() == "/custom/model-cache"


def test_fastembed_deterministic(embedder: FastEmbedEmbedder) -> None:
    """Same input → byte-identical vector across two calls.

    Why: caching, reproducible eval, and stable test snapshots all rely
    on the Embedder Protocol's no-randomness contract.
    """
    v1 = embedder.embed(["桥梁防水责任方"])[0]
    v2 = embedder.embed(["桥梁防水责任方"])[0]
    assert v1 == v2


def test_fastembed_batch_preserves_order(embedder: FastEmbedEmbedder) -> None:
    """Batch embedding must yield vectors in input order (Protocol contract)."""
    texts = ["alpha", "beta", "gamma"]
    batch = embedder.embed(texts)
    # Cross-check: each text embedded alone should equal its batch slot.
    for i, t in enumerate(texts):
        single = embedder.embed([t])[0]
        assert batch[i] == single


def test_fastembed_empty_input_returns_empty(embedder: FastEmbedEmbedder) -> None:
    """Empty batch short-circuits without model load."""
    assert embedder.embed([]) == []


def test_fastembed_rejects_unknown_model() -> None:
    """Unknown model name must fail at construction, not on first embed."""
    with pytest.raises(ValueError, match="Unknown fastembed model"):
        FastEmbedEmbedder(model_name="not-a-real-model")


# --------------------------------------------------------------------------- #
# Bm25Index
# --------------------------------------------------------------------------- #


def test_bm25_roundtrip_english_keyword(sample_chunks: list[Chunk]) -> None:
    """English query ``waterproofing`` returns its source chunk in top-1."""
    idx = Bm25Index()
    idx.add(sample_chunks)
    results = idx.search("waterproofing", k=3)
    assert results, "BM25 returned no results for a present keyword"
    assert results[0].chunk.id == "doc.pdf:1:0"
    assert results[0].score > 0


def test_bm25_roundtrip_chinese_query(sample_chunks: list[Chunk]) -> None:
    """Chinese query ``防水`` resolves via jieba to the Chinese chunk."""
    idx = Bm25Index()
    idx.add(sample_chunks)
    results = idx.search("防水", k=3)
    assert results, "BM25 returned no results for Chinese keyword"
    # The Chinese chunk should top this — only it contains 防水.
    assert results[0].chunk.id == "doc.pdf:2:0"


def test_bm25_idempotent_readd(sample_chunks: list[Chunk]) -> None:
    """Re-adding same Chunk.id replaces, doesn't duplicate."""
    idx = Bm25Index()
    idx.add(sample_chunks)
    idx.add(sample_chunks)  # second add — should be a no-op for ranking
    results = idx.search("waterproofing", k=10)
    # Distinct chunk ids only.
    ids = [r.chunk.id for r in results]
    assert len(ids) == len(set(ids))
    assert len(ids) <= len(sample_chunks)


def test_bm25_empty_index_returns_empty() -> None:
    """Search before any add → empty list, no crash."""
    assert Bm25Index().search("anything", k=5) == []


def test_bm25_handles_empty_text() -> None:
    """A chunk with empty text shouldn't crash BM25 (division-by-zero guard).

    Why we don't assert chunk 'b' wins here:
        rank_bm25 IDF can collapse to ~0 in tiny 2-doc corpora when a
        term appears in 50% of docs — the test would be measuring corpus
        statistics, not our sentinel logic. The real guarantee we care
        about is: ``add`` + ``search`` complete without raising.
    """
    idx = Bm25Index()
    idx.add(
        [
            Chunk(id="a", text="", file="f.pdf", page=1, chunk_type="paragraph"),
            Chunk(
                id="b",
                text="waterproofing of the pier deck",
                file="f.pdf",
                page=2,
                chunk_type="paragraph",
            ),
            Chunk(
                id="c",
                text="concrete grade C40/20 specifications",
                file="f.pdf",
                page=3,
                chunk_type="paragraph",
            ),
        ]
    )
    results = idx.search("waterproofing", k=3)
    # With 3 docs and only chunk 'b' containing the query term, BM25 IDF
    # is well-defined and 'b' must top the ranking.
    assert results[0].chunk.id == "b"


# --------------------------------------------------------------------------- #
# BM25 + Phase 2 caption (sub-sprint p2-ssCaption)
# --------------------------------------------------------------------------- #


def test_bm25_includes_chunk_caption_in_index() -> None:
    """Phase 2: a drawing chunk's Chinese caption must contribute to BM25 hits.

    Setup: one drawing chunk whose ``text`` is just the OCR'd Drawing No.
    (English-only) and one paragraph chunk with the same Drawing No. but
    no Chinese content. A Chinese query like "防水" should rank the
    drawing chunk first if and only if the caption text is folded into
    BM25 via chunk_indexable_text.
    """
    idx = Bm25Index()
    idx.add(
        [
            Chunk(
                id="drawing-with-cap",
                text="Drawing No. T/PRJ/CWD/WS/2101A",
                file="f.pdf",
                page=1,
                chunk_type="drawing",
                caption="桥梁防水构造图，含三层涂层结构和压顶混凝土板。",
            ),
            Chunk(
                id="paragraph-without-cap",
                text="Drawing No. T/PRJ/CWD/WS/2101A is referenced.",
                file="f.pdf",
                page=2,
                chunk_type="paragraph",
            ),
        ]
    )
    results = idx.search("防水", k=3)
    # Only the captioned chunk has Chinese tokens matching "防水".
    assert len(results) >= 1
    assert results[0].chunk.id == "drawing-with-cap"


def test_bm25_caption_none_keeps_text_only_indexing() -> None:
    """Chunks with caption=None tokenize text only (regression guard).

    A chunk with caption=None must produce the exact same token set it
    would without the caption field — i.e. Phase 2 must NOT silently
    degrade existing English retrieval for non-captioned chunks.
    """
    idx = Bm25Index()
    idx.add(
        [
            Chunk(
                id="c1",
                text="waterproofing details on the pier deck",
                file="f.pdf",
                page=1,
                chunk_type="paragraph",
                caption=None,
            ),
            Chunk(
                id="c2",
                text="concrete grade C40/20 specifications",
                file="f.pdf",
                page=2,
                chunk_type="paragraph",
                caption=None,
            ),
        ]
    )
    results = idx.search("waterproofing", k=2)
    assert results[0].chunk.id == "c1"


def test_bm25_caption_empty_string_treated_as_no_caption() -> None:
    """caption="" means "captioner ran but produced nothing".

    Per DECISION-2.cap.3, an empty caption must NOT add the "Caption:"
    separator to the indexable text — otherwise BM25 would pick up the
    literal word "Caption" and skew tokenization.
    """
    from jcontract.interfaces.schema import chunk_indexable_text

    chunk = Chunk(
        id="x",
        text="waterproofing details",
        file="f.pdf",
        page=1,
        chunk_type="drawing",
        caption="",
    )
    # Empty caption falls through to text-only, same as None.
    assert chunk_indexable_text(chunk) == "waterproofing details"


# --------------------------------------------------------------------------- #
# QdrantStore
# --------------------------------------------------------------------------- #


pytestmark_qdrant = pytest.mark.skipif(
    not _qdrant_reachable(),
    reason="Qdrant not reachable on localhost:6333 — run `docker-compose up -d qdrant`",
)


@pytest.fixture
def store():
    """Fresh collection per test; drop on teardown.

    Untyped fixture: pyproject's mypy override allows untyped defs in
    tests/, which lets us avoid the noisy ``Iterator[QdrantStore]``
    generator-return annotation.
    """
    s = QdrantStore(collection_name="jcontract_test_ssb")
    s._drop()  # clean slate
    yield s
    s._drop()


@pytestmark_qdrant
def test_qdrant_roundtrip(
    store: QdrantStore,
    embedder: FastEmbedEmbedder,
    sample_chunks: list[Chunk],
) -> None:
    """add 3 chunks → search → 3 results returned with payload intact."""
    vectors = embedder.embed([c.text for c in sample_chunks])
    store.add(sample_chunks, vectors)

    assert store.count() == 3

    # Use a chunk's own embedding as the query — top-1 must be itself.
    results = store.search(query_vector=vectors[0], k=3)
    assert len(results) == 3
    assert results[0].chunk.id == sample_chunks[0].id
    # Cosine self-similarity should be ~1.0 (allowing fp drift).
    assert results[0].score > 0.99


@pytestmark_qdrant
def test_qdrant_chunk_payload_roundtrip(
    store: QdrantStore,
    embedder: FastEmbedEmbedder,
    sample_chunks: list[Chunk],
) -> None:
    """Full Chunk dataclass round-trips through Qdrant payload."""
    vectors = embedder.embed([c.text for c in sample_chunks])
    store.add(sample_chunks, vectors)

    results = store.search(query_vector=vectors[1], k=1)
    out = results[0].chunk
    src = sample_chunks[1]
    assert out.id == src.id
    assert out.text == src.text
    assert out.file == src.file
    assert out.page == src.page
    assert out.chunk_type == src.chunk_type
    assert out.drawing_refs == src.drawing_refs
    assert out.clause_refs == src.clause_refs
    assert out.section_path == src.section_path


@pytestmark_qdrant
def test_qdrant_idempotent_upsert(
    store: QdrantStore,
    embedder: FastEmbedEmbedder,
    sample_chunks: list[Chunk],
) -> None:
    """Re-adding same Chunk.id doesn't grow the point count."""
    vectors = embedder.embed([c.text for c in sample_chunks])
    store.add(sample_chunks, vectors)
    store.add(sample_chunks, vectors)  # same ids → upsert
    assert store.count() == 3


@pytestmark_qdrant
def test_qdrant_search_empty_collection(store: QdrantStore) -> None:
    """Search before any add → empty list, no crash."""
    assert store.search(query_vector=[0.1] * 768, k=5) == []
    assert store.count() == 0


def test_qdrant_point_uuid_deterministic() -> None:
    """uuid5 mapping is stable across calls — pure-Python sanity test."""
    assert _point_uuid("doc.pdf:1:0") == _point_uuid("doc.pdf:1:0")
    assert _point_uuid("a") != _point_uuid("b")


# --------------------------------------------------------------------------- #
# _build_stack BM25 degradation warning (P1Fixes 2026-06-10)
# --------------------------------------------------------------------------- #


def test_build_stack_warns_on_missing_snapshot(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Snapshot empty + Qdrant non-empty → loud bm25_snapshot_missing warning.

    # Why: hybrid silently degrading to vector-only burned a multi-day eval
    # run (2026-06-08); the warning is the regression guard for that.
    """
    from jcontract import cli as cli_mod

    monkeypatch.setattr(cli_mod, "load_chunks_snapshot", lambda _p: [])
    monkeypatch.setattr(QdrantStore, "count", lambda self: 7)
    cli_mod._build_stack("ghost-collection")
    out = capsys.readouterr().out
    assert "bm25_snapshot_missing" in out
    assert "ghost-collection" in out


def test_build_stack_quiet_on_fresh_collection(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Snapshot empty + Qdrant empty (fresh collection) → no warning noise."""
    from jcontract import cli as cli_mod

    monkeypatch.setattr(cli_mod, "load_chunks_snapshot", lambda _p: [])
    monkeypatch.setattr(QdrantStore, "count", lambda self: 0)
    cli_mod._build_stack("fresh-collection")
    assert "bm25_snapshot_missing" not in capsys.readouterr().out
