"""HTTP API surface — Layer 4.

Per docs/project_guideline.md §3.1 Layer 4 (API). Wraps the same
retriever + answerer stack the CLI uses, exposed as FastAPI routes for
the Web UI (Phase 5) and any future programmatic consumer.

Started by sub-sprint p5-s1-ssApi (MVP-cut): /healthz + /ask + /search.
Deferred to Enhancement: /ingest upload (E4), feedback (E8), SSE
streaming (FORESHADOW-mvp.api.1), Agent path (E7).
"""

from __future__ import annotations
