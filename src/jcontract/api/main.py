"""FastAPI app instance — Layer 4 entry point.

Boot:
    uv run uvicorn jcontract.api.main:app --port 8000 --reload

Wire:
- CORS allowed origin = http://localhost:3000 only (Next.js dev server).
  Production CORS is an Enhancement (E6) along with docker-compose.
- Routes mounted: /healthz, /ask, /search (MVP scope, see
  DECISION-orch-5 in docs/dev-sprint.md for what's deferred).
- Dependencies (Stack, Answerer) are lru_cache singletons in
  jcontract.api.dependencies — see that module for rationale.
"""

from __future__ import annotations

import structlog
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from jcontract.api.routes import ask, files, health, search

logger = structlog.get_logger(__name__)


def create_app() -> FastAPI:
    """Build the FastAPI app.

    Why a factory function: lets tests get a fresh app without dragging
    in the singleton dependency caches from a previous test run. The
    module-level `app` (below) is still exported for uvicorn's
    `import_string:app` reference.
    """
    app = FastAPI(
        title="j-contract",
        version="0.1.0",
        description=(
            "Construction contract knowledge retrieval AI. "
            "Ask Chinese questions about English contract PDFs."
        ),
    )

    # CORS — dev-friendly: accept any localhost / 127.0.0.1 / private-IP
    # origin so WSL2 / Docker / LAN-test setups don't trip over a strict
    # allowlist. Safe because `allow_credentials=False` — no cookies or
    # auth tokens flow, so a malicious origin can only read public API
    # responses (no privilege escalation surface). Tighten when auth lands.
    #
    # LESSON-mvp.cors.1 (2026-05-30): the original `allow_origins=
    # ["http://localhost:3000"]` broke WSL2 setups where Windows browsers
    # access the frontend via 127.0.0.1 or the WSL eth0 IP. Fetch errors
    # surfaced as opaque "Failed to fetch" in the browser console with
    # no clear CORS message because the browser blocks the response
    # before exposing details to JS.
    app.add_middleware(
        CORSMiddleware,
        allow_origin_regex=r"^https?://(localhost|127\.0\.0\.1|0\.0\.0\.0|\[::1\]|172\.\d+\.\d+\.\d+|192\.168\.\d+\.\d+|10\.\d+\.\d+\.\d+)(:\d+)?$",
        allow_credentials=False,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["Content-Type", "Accept"],
    )

    app.include_router(health.router, tags=["health"])
    app.include_router(ask.router, tags=["chat"])
    app.include_router(search.router, tags=["debug"])
    app.include_router(files.router, tags=["files"])

    # `app.routes` is `list[BaseRoute]`; only the concrete Route /
    # Mount subclasses carry `.path`. Use getattr to keep the log
    # tolerant of future route types (WebSocketRoute etc.) without a
    # mypy attr-defined error.
    logger.info(
        "api.app_created",
        routes=[getattr(r, "path", None) for r in app.routes if getattr(r, "path", None)],
    )
    return app


# Module-level app instance for `uvicorn jcontract.api.main:app` to
# import. Tests build their own via `create_app()` to keep state fresh.
app = create_app()
