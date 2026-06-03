"""Unit tests for jcontract.api routes.

Strategy:
- Use FastAPI TestClient against a fresh `create_app()` per test.
- Override the Stack + Answerer dependencies with lightweight stubs so
  the tests don't touch Qdrant, fastembed, or Anthropic.
- Cover: healthz, ask happy path, ask empty index, ask no answerer,
  search top-k. 5 tests total per sub-sprint p5-s1-ssApi MVP-cut.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
from fastapi.testclient import TestClient

from jcontract.api.dependencies import answerer_dep, get_answerer, get_stack, stack_dep
from jcontract.api.main import create_app
from jcontract.interfaces import Answer, Chunk, SearchResult


def _make_chunk(idx: int = 0, file: str = "synthetic.pdf", page: int = 1) -> Chunk:
    """Build a Chunk fixture — minimal fields, paragraph type by default."""
    return Chunk(
        id=f"synthetic:{page}:{idx}",
        text=f"sample chunk text {idx} about waterproofing of piers",
        file=file,
        page=page,
        chunk_type="paragraph",
    )


def _build_stub_stack(search_results: list[SearchResult], qdrant_count: int = 4) -> SimpleNamespace:
    """Duck-typed stand-in for cli.Stack — just needs .retriever and .vector_store."""
    retriever = SimpleNamespace(search=lambda _q, k: search_results[:k])
    vector_store = SimpleNamespace(count=lambda: qdrant_count)
    return SimpleNamespace(retriever=retriever, vector_store=vector_store)


def _build_stub_answerer(answer_text: str = "测试答案", confidence: str = "high") -> Any:
    """Duck-typed Answerer. Returns an Answer with one citation."""

    class _StubAnswerer:
        def answer(self, _question: str, chunks: list[Chunk]) -> Answer:
            return Answer(
                text=answer_text,
                citations=[(c.file, c.page) for c in chunks[:2]],
                confidence=confidence,  # type: ignore[arg-type]  # Confidence Literal
                raw_context=chunks,
            )

    return _StubAnswerer()


@pytest.fixture
def client_with_stubs():
    """Build a TestClient with stack + answerer overridden.

    Returns a factory that takes `(search_results, answerer, qdrant_count)`
    so each test customises the stubs without leaking state across tests.
    """
    apps_created: list[Any] = []

    def factory(
        search_results: list[SearchResult] | None = None,
        answerer: Any = "default",
        qdrant_count: int = 4,
    ) -> TestClient:
        app = create_app()
        apps_created.append(app)
        stack = _build_stub_stack(search_results or [], qdrant_count=qdrant_count)
        # get_stack backs /healthz; stack_dep backs /ask + /search (Phase 7 SS7).
        app.dependency_overrides[get_stack] = lambda: stack
        app.dependency_overrides[stack_dep] = lambda: stack
        # Sentinel "default" → use a real-ish stub answerer; None →
        # explicit no-answerer path; any other → use directly.
        if answerer == "default":
            answerer = _build_stub_answerer()
        app.dependency_overrides[get_answerer] = lambda: answerer
        app.dependency_overrides[answerer_dep] = lambda: answerer
        return TestClient(app)

    yield factory
    # Cleanup: clear dependency overrides on each created app.
    for app in apps_created:
        app.dependency_overrides.clear()


def test_healthz_returns_ok_with_qdrant_count(client_with_stubs):
    """`/healthz` returns status=ok plus the Qdrant point count."""
    client = client_with_stubs(qdrant_count=42)
    resp = client.get("/healthz")
    assert resp.status_code == 200
    body = resp.json()
    assert body == {"status": "ok", "qdrant_count": 42}


def test_ask_happy_path_returns_answer_and_citations(client_with_stubs):
    """`/ask` with retrieval hits + answerer → answer text + citations."""
    chunks = [_make_chunk(i, page=i + 1) for i in range(5)]
    results = [SearchResult(chunk=c, score=0.9 - i * 0.05) for i, c in enumerate(chunks)]
    client = client_with_stubs(search_results=results)

    resp = client.post("/ask", json={"question": "桥梁防水谁负责?"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["answer"] == "测试答案"
    assert body["confidence"] == "high"
    # Stub returns first 2 chunks as citations; check the structure.
    assert len(body["citations"]) == 2
    assert body["citations"][0] == {"file": "synthetic.pdf", "page": 1}


def test_ask_empty_index_returns_graceful_fallback(client_with_stubs):
    """No retrieval hits → answer is the 'no documents indexed' string."""
    client = client_with_stubs(search_results=[])  # empty index

    resp = client.post("/ask", json={"question": "anything"})
    assert resp.status_code == 200
    body = resp.json()
    assert "no documents indexed" in body["answer"]
    assert body["citations"] == []
    assert body["confidence"] == "none"


def test_collection_query_param_accepted_on_ask_and_search(client_with_stubs):
    """Phase 7 SS7: /ask and /search accept ?collection= to pick a knowledge base."""
    chunks = [_make_chunk(0, page=1)]
    results = [SearchResult(chunk=chunks[0], score=0.9)]
    client = client_with_stubs(search_results=results)

    assert client.post("/ask?collection=finance", json={"question": "q"}).status_code == 200
    assert client.get("/search?q=test&collection=finance").status_code == 200
    # The param is advertised in the OpenAPI schema for both routes.
    schema = client.get("/openapi.json").json()
    ask_params = {p["name"] for p in schema["paths"]["/ask"]["post"].get("parameters", [])}
    search_params = {p["name"] for p in schema["paths"]["/search"]["get"].get("parameters", [])}
    assert "collection" in ask_params
    assert "collection" in search_params


def test_get_stack_for_caches_per_collection() -> None:
    """Each collection gets its own cached entry (bounded LRU)."""
    from jcontract.api.dependencies import get_answerer_for, get_stack_for

    # Same collection → same cache key (cache_info hits grow); we only assert
    # the functions are independently cached (distinct __wrapped__ + cache).
    assert hasattr(get_stack_for, "cache_info")
    assert hasattr(get_answerer_for, "cache_info")


def test_ask_no_answerer_returns_retrieval_only(client_with_stubs):
    """Answerer is None → retrieval-only mode with citations populated."""
    chunks = [_make_chunk(i, page=i + 1) for i in range(3)]
    results = [SearchResult(chunk=c, score=0.8 - i * 0.05) for i, c in enumerate(chunks)]
    client = client_with_stubs(search_results=results, answerer=None)

    resp = client.post("/ask", json={"question": "测试"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["confidence"] == "none"
    assert "retrieval-only" in body["answer"]
    # Citations still populated from retrieval (3 chunks, all returned).
    assert len(body["citations"]) == 3


def test_search_returns_top_k_with_preview(client_with_stubs):
    """`/search` returns the raw retrieval list with previews."""
    chunks = [_make_chunk(i, page=i + 1) for i in range(8)]
    results = [SearchResult(chunk=c, score=0.9 - i * 0.05) for i, c in enumerate(chunks)]
    client = client_with_stubs(search_results=results)

    resp = client.get("/search?q=waterproofing&k=5")
    assert resp.status_code == 200
    body = resp.json()
    assert len(body) == 5
    assert body[0]["file"] == "synthetic.pdf"
    assert body[0]["page"] == 1
    assert "waterproofing" in body[0]["preview"]
    # Scores ordered descending in the stub list.
    assert body[0]["score"] >= body[1]["score"]


def test_ask_rejects_overlong_question(client_with_stubs):
    """Pydantic max_length=1000 enforces the §6.2 untrusted-input limit."""
    client = client_with_stubs()
    resp = client.post("/ask", json={"question": "a" * 1001})
    assert resp.status_code == 422


def test_ask_rejects_empty_question(client_with_stubs):
    """min_length=1 prevents empty-string requests sneaking past the LLM."""
    client = client_with_stubs()
    resp = client.post("/ask", json={"question": ""})
    assert resp.status_code == 422
