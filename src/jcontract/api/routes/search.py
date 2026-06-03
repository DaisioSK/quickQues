"""`/search` — debug endpoint exposing the raw retriever output.

Returns top-k chunks (file / page / chunk_type / score / preview)
WITHOUT going through the answerer. Lets developers and curious users
inspect "what would the LLM have seen" without paying for the LLM
call. Equivalent to `jcontract search <q>` in the CLI.

Why we expose this in MVP at all (it's debug):
- During Phase 5 frontend bring-up we'll be poking the API directly to
  validate retrieval quality before adding the LLM layer of variance.
- Cheap to add (no schema dance, no auth bypass since the whole API
  is currently unauthenticated MVP).
- Removing it later is one delete.
"""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, Depends, Query

from jcontract.api.dependencies import stack_dep
from jcontract.api.schemas import SearchResultOut

router = APIRouter()


# Preview length matches `jcontract search` CLI output for consistency
# (cli.py:215 prints chunk.text[:160]; we widen a touch to 200 since
# JSON consumers can afford the extra bytes).
_PREVIEW_CHARS = 200


@router.get("/search", response_model=list[SearchResultOut])
def search(
    q: Annotated[str, Query(min_length=1, max_length=1000)],
    stack: Annotated[Any, Depends(stack_dep)],
    k: Annotated[int, Query(ge=1, le=50)] = 10,
) -> list[SearchResultOut]:
    """Hybrid retrieve (vector + BM25 RRF fused) and return top-k chunks.

    `q` max_length mirrors AskRequest — same defense-in-depth limit
    on untrusted input. `k` capped at 50 prevents accidental "k=10000"
    from spilling the entire index into the response. `?collection=`
    (consumed by stack_dep) selects the knowledge base (default contract).
    """
    results = stack.retriever.search(q, k=k)
    return [
        SearchResultOut(
            file=r.chunk.file,
            page=r.chunk.page,
            chunk_type=r.chunk.chunk_type,
            score=float(r.score),
            preview=r.chunk.text.strip().replace("\n", " ")[:_PREVIEW_CHARS],
        )
        for r in results
    ]
