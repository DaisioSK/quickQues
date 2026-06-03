"""Vendor-specific implementations of interfaces.

Each subdirectory holds one vendor (e.g. impls/llamaparse, impls/claude,
impls/qdrant). Business code in ingest/retrieve/answer/etc. MUST import only
from jcontract.interfaces, never from jcontract.impls directly — the
implementation choice is wired by config.yaml at startup time.

Populated starting Phase 1 S1.2.
"""
