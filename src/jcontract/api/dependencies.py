"""Shared dependency providers for FastAPI routes.

Pattern: lru_cache singletons rather than a lifespan-managed app.state.
- Why lru_cache: simpler call sites — routes just `Depends(get_stack)`
  without needing the Request object. Works in TestClient (each test
  can clear the cache between runs for isolation).
- Why not lifespan: app.state pattern is cleaner for resources that
  need explicit teardown (db pool close on shutdown). Our Stack
  components have no explicit close requirement — Qdrant client and
  fastembed model are GC-collected fine.

Why we reuse the CLI's `_build_stack` directly:
- That function already encapsulates the right wiring (Embedder,
  VectorStore, KeywordIndex, BM25 rehydration from JSONL snapshot).
- DRY — if we duplicated the wiring here, future changes (new impl
  swap, snapshot path move) would need to land in two places.
- The cli module top-level is import-safe (no I/O at import time —
  just decorator wiring), so the import doesn't pull in a runtime
  side-effect.
"""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Annotated

import structlog
from fastapi import Query

from jcontract.cli import _build_stack, _maybe_build_answerer
from jcontract.impls.domain_profile_registry import load_profile
from jcontract.interfaces import Answerer
from jcontract.paths import read_profile_name

logger = structlog.get_logger(__name__)


# Default collection when a request omits ?collection=. Matches the CLI.
DEFAULT_COLLECTION = os.environ.get("JCONTRACT_COLLECTION", "contract")

# Phase 7 SS7: the API can serve MANY collections in one process. We cache
# per-collection stacks/answerers in a small bounded LRU so a `?collection=`
# fan-out can't grow memory unbounded; evicted entries just rebuild on next
# use. maxsize=8 covers a handful of coexisting knowledge bases.
_MAX_COLLECTIONS = 8


@lru_cache(maxsize=_MAX_COLLECTIONS)
def get_stack_for(collection: str):  # type: ignore[no-untyped-def]  # Stack is private in cli.py
    """Build (and memoize) the retriever + index stack for one collection.

    Process restart (or LRU eviction) is the reload mechanism — same
    trade-off the CLI makes. RRF-only (no reranker) for MVP latency.
    """
    logger.info("api.stack_build", collection=collection)
    return _build_stack(collection, use_reranker=False)


@lru_cache(maxsize=_MAX_COLLECTIONS)
def get_answerer_for(collection: str) -> Answerer | None:
    """Build an Answerer for a collection, framed by its DomainProfile.

    Graceful degradation: when no key/binary is available `/ask` still
    returns retrieval-only results (confidence="none"). The collection's
    profile (its `profile.txt` sidecar) supplies the answer framing so a
    finance corpus isn't answered with construction framing.

    Backend via JCONTRACT_ANSWERER_BACKEND env (claude-api default).
    """
    backend = os.environ.get("JCONTRACT_ANSWERER_BACKEND", "claude-api")
    framing = load_profile(read_profile_name(collection)).answer_framing
    try:
        answerer = _maybe_build_answerer(backend, domain_framing=framing)
    except Exception as exc:  # noqa: BLE001
        # CLI binary missing for claude-cli/codex-cli → degrade not fail.
        logger.warning("api.answerer_unavailable", error_type=type(exc).__name__)
        return None
    if answerer is None:
        logger.info("api.answerer_disabled", backend=backend, collection=collection)
    else:
        logger.info("api.answerer_ready", backend=backend, collection=collection)
    return answerer


# Back-compat shims for the default collection (used by healthz / any caller
# that doesn't pass a collection).
def get_stack():  # type: ignore[no-untyped-def]
    """Default-collection stack (delegates to get_stack_for)."""
    return get_stack_for(DEFAULT_COLLECTION)


def get_answerer() -> Answerer | None:
    """Default-collection answerer (delegates to get_answerer_for)."""
    return get_answerer_for(DEFAULT_COLLECTION)


# FastAPI dependency providers that read ?collection= and resolve the
# per-collection stack/answerer. Routes Depend() on these (so tests can
# still override them via app.dependency_overrides). Same query-param name
# across both → FastAPI surfaces a single ?collection= on each route.
def stack_dep(  # type: ignore[no-untyped-def]
    collection: Annotated[str, Query(description="Knowledge base to query.")] = DEFAULT_COLLECTION,
):
    """Request-scoped: the retriever stack for the requested collection."""
    return get_stack_for(collection)


def answerer_dep(
    collection: Annotated[str, Query(description="Knowledge base to query.")] = DEFAULT_COLLECTION,
) -> Answerer | None:
    """Request-scoped: the profile-framed answerer for the requested collection."""
    return get_answerer_for(collection)
