"""Pytest configuration shared across the test suite.

Why this file exists:
    ``pyproject.toml`` sets ``--strict-markers``, which forbids any
    ``@pytest.mark.<name>`` whose marker isn't registered. The Phase 1
    S1.1 ssA spec requires a ``@pytest.mark.slow`` smoke test for the
    real-PDF integration. Registering the marker here (rather than in
    pyproject.toml) keeps the change scoped to test infrastructure and
    avoids touching the build config across worktrees.

    Other sub-sprints (ssB integration tests, ssC API tests) will likely
    want additional markers; add them via ``config.addinivalue_line``
    below as they land.
"""

from __future__ import annotations

import pytest


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "slow: integration / smoke tests that read real artefacts "
        "(PDFs, network) — kept in the default run so we catch "
        "regressions, but flagged so CI can selectively skip them.",
    )
