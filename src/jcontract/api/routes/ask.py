"""`/ask` — the user-facing chat endpoint.

Pipeline (MVP-cut, no SSE / no Agent):
  1. Hybrid retrieve top-10 chunks.
  2. If no chunks → graceful fallback ("no documents indexed yet").
  3. If no answerer (no API key, no CLI) → return retrieval-derived
     citations + a "retrieval-only mode" answer.
  4. Otherwise call answerer.answer(question, top-5 chunks) and return
     its `text` + `citations` + `confidence`.

Why retrieve top-10 but only feed top-5 to the answerer:
- Top-5 fits comfortably in Claude's context with prompt budget for
  the system message + question.
- Top-10 gives the API caller more candidates for UI affordances
  (e.g. "see more sources") without paying for re-retrieval. Today we
  drop the bottom-5 before answering; if the frontend later wants
  them, we can extend the response — purely additive change.
"""

from __future__ import annotations

from typing import Annotated, Any

import structlog
from fastapi import APIRouter, Depends

from jcontract.api.dependencies import answerer_dep, stack_dep
from jcontract.api.schemas import AskRequest, AskResponse, CitationOut
from jcontract.interfaces import Answerer

logger = structlog.get_logger(__name__)

router = APIRouter()


# Top-K constants. RETRIEVE_K is what we pull from the hybrid retriever;
# ANSWER_K is what we feed the LLM. Phase 1.7 + 1.8 used these same
# values empirically — keep them aligned with the CLI eval setup so the
# API doesn't quietly answer "differently" from `jcontract evaluate`.
RETRIEVE_K = 10
ANSWER_K = 5


_EMPTY_INDEX_FALLBACK = "(no documents indexed yet — run `jcontract ingest <pdf>` first)"
_NO_ANSWERER_FALLBACK_PREFIX = (
    "(retrieval-only mode — no answerer configured. Citations below are top retrieval hits.)"
)


@router.post("/ask", response_model=AskResponse)
def ask(
    payload: AskRequest,
    stack: Annotated[Any, Depends(stack_dep)],
    answerer: Annotated[Answerer | None, Depends(answerer_dep)],
) -> AskResponse:
    """Answer a Chinese question against one collection's ingested corpus.

    Phase 7 SS7: `?collection=` (consumed by the deps) selects the knowledge
    base (default contract); the stack + answerer are resolved per-collection, and
    the answerer is framed by that collection's DomainProfile.
    """
    results = stack.retriever.search(payload.question, k=RETRIEVE_K)

    # Retrieval trace — one structured log line per /ask so the live
    # `make dev` terminal shows exactly what the hybrid retriever
    # surfaced (file:page@score) before the LLM ever sees it. This is the
    # primary debug hook for "why did it answer that?": if the right
    # chunk isn't in this list, it's a retrieval problem, not an
    # answerer problem. Truncated to the top RETRIEVE_K we actually use.
    logger.info(
        "api.ask_retrieved",
        q_len=len(payload.question),
        hits=[f"{r.chunk.file}:p{r.chunk.page}@{r.score:.4f}" for r in results],
    )

    # Empty index — graceful, NOT a 500. The frontend should still render
    # the answer string in the message bubble.
    if not results:
        logger.info("api.ask_empty_index", q_len=len(payload.question))
        return AskResponse(
            answer=_EMPTY_INDEX_FALLBACK,
            citations=[],
            confidence="none",
        )

    # No answerer configured — return retrieval-only response. The
    # citations field carries the top-5 file/page pairs so the UI can
    # still link out to the PDF viewer.
    if answerer is None:
        logger.info("api.ask_no_answerer", q_len=len(payload.question), hits=len(results))
        retrieval_citations = [
            CitationOut(file=r.chunk.file, page=r.chunk.page) for r in results[:ANSWER_K]
        ]
        # Why we synthesize an answer string rather than 503ing:
        # the frontend's happy path renders `answer` in the message
        # bubble. A 503 would force a separate UX branch for a
        # legitimate (just under-configured) state.
        return AskResponse(
            answer=_NO_ANSWERER_FALLBACK_PREFIX,
            citations=retrieval_citations,
            confidence="none",
        )

    # Happy path: answerer present, retrieval non-empty.
    answer = answerer.answer(payload.question, [r.chunk for r in results[:ANSWER_K]])
    logger.info(
        "api.ask_complete",
        q_len=len(payload.question),
        confidence=answer.confidence,
        n_citations=len(answer.citations),
    )

    return AskResponse(
        answer=answer.text,
        citations=[CitationOut(file=f, page=p) for f, p in answer.citations],
        # answer.confidence is Layer 0's Confidence (`high|medium|low`).
        # ApiConfidence widens with "none"; the cast is type-safe.
        confidence=answer.confidence,
    )
