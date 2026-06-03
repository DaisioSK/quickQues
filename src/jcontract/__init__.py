"""j-contract: construction contract knowledge retrieval AI.

Layer 0 of the architecture per docs/project_guideline.md §3.1:
  - interfaces/  — Protocol abstractions (filled in Phase 1 S1.1)
  - impls/       — vendor implementations (Phase 1 S1.2+)
  - ingest/      — pipeline orchestration (Phase 2)
  - retrieve/    — hybrid search (Phase 3)
  - answer/      — LLM synthesis + citation guards (Phase 4)
  - agent/       — multi-step orchestration (Phase 4 S4.5)
  - api/         — FastAPI surface (Phase 5)
  - cli/         — command-line entrypoints
  - config.py    — centralized secret + config loading
"""

__version__ = "0.1.0"
