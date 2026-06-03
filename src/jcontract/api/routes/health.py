"""`/healthz` — liveness + minimal stack-state probe.

Returns 200 + a small JSON payload the frontend / monitoring can poll
to confirm the backend is up AND has a usable Qdrant connection. We
include `qdrant_count` because "process running but Qdrant disconnected"
is a real failure mode worth catching cheaply.
"""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, Depends

from jcontract.api.dependencies import get_stack
from jcontract.api.schemas import HealthResponse

router = APIRouter()


@router.get("/healthz", response_model=HealthResponse)
def healthz(stack: Annotated[Any, Depends(get_stack)]) -> HealthResponse:
    """Return 200 + Qdrant point count.

    Why `Any` typing on the stack param: the cli's Stack dataclass is
    private. Importing it here would tie Layer 4 to a CLI internal —
    we just need `.vector_store.count()` from the duck-typed object.
    """
    return HealthResponse(status="ok", qdrant_count=stack.vector_store.count())
