"""Shared OCR-cache-key helper for the vision PDF parsers.

Both ``ClaudeVisionParser`` (API) and ``ClaudeCliVisionParser``
(subscription) cache OCR results on disk keyed by SHA-256 of the
rendered JPEG (+ prompt kind). Enhancement E10 (``p1.5-ssVisionModelSelect``)
makes the OCR model selectable, which means the model now affects the
output — so it must participate in the cache key, or switching
``--vision-model haiku → sonnet`` would silently return the stale
lower-fidelity text.

DECISION-e10.cache.1 (docs/dev-sprint.md): the model slug is appended to
the cache filename ONLY when the model differs from that parser's
historical default. Reason: the maintainer already ran a full DEMO
ingest with each parser's default model; those on-disk caches use the
un-suffixed filename. Suppressing the suffix for the default keeps every
existing cache valid (honours the 2026-05-30 "不重跑现有索引" constraint),
while any non-default model gets an isolated cache namespace and a fresh
OCR pass — which is exactly the point of E10.

N=2: two parsers need identical suffix logic, so it lives here rather
than being duplicated per impl (project_guideline.md §5 rule 1).
"""

from __future__ import annotations

import re

# Filename-safe slug: collapse any run of characters that aren't
# alphanumeric / dash / dot into a single dash. Model ids like
# "claude-sonnet-4-5" pass through unchanged; "sonnet" stays "sonnet".
_SLUG_RE = re.compile(r"[^A-Za-z0-9.-]+")


def model_cache_suffix(model: str, default_model: str) -> str:
    """Return the cache-filename component for ``model``.

    Empty string when ``model`` equals ``default_model`` (legacy
    un-suffixed filename, preserves existing caches); otherwise
    ``".<slug>"`` so a non-default model lands in its own namespace.

    >>> model_cache_suffix("haiku", "haiku")
    ''
    >>> model_cache_suffix("sonnet", "haiku")
    '.sonnet'
    >>> model_cache_suffix("claude-sonnet-4-5", "haiku")
    '.claude-sonnet-4-5'
    """
    if model == default_model:
        return ""
    slug = _SLUG_RE.sub("-", model).strip("-")
    return f".{slug}"
