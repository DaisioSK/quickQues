"""FastEmbed-backed Embedder implementation.

What:
    Wraps fastembed's ONNX-runtime ``TextEmbedding`` class to produce dense
    multilingual vectors for the j-contract corpus (Chinese contract text +
    English clauses + drawing numbers).

Why ONNX over PyTorch:
    qdrant-client[fastembed] ships ONNX runtime only (no torch). This keeps
    the prototype container ~2GB lighter and runs on CPU at acceptable
    latency for the DEMO corpus size (single-doc Phase 1).

Why this specific model (DECISION):
    The original spec called for ``intfloat/multilingual-e5-base`` (768 dim,
    ~280MB). That model is NOT in fastembed's supported list as of
    fastembed 0.4 / qdrant-client 1.12. The closest substitutes are:

    - ``sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2``
      (384 dim, ~220MB) — fast, light, multilingual.
    - ``intfloat/multilingual-e5-large`` (1024 dim, ~2.24GB) — heavy,
      slow first download, but the e5 family is best-in-class for retrieval.
    - ``sentence-transformers/paraphrase-multilingual-mpnet-base-v2``
      (768 dim, ~1GB) — the 768-dim multilingual middle ground.

    For Phase 1 prototype we default to ``paraphrase-multilingual-mpnet-
    base-v2``: it preserves the 768-dim contract from the spec, keeps the
    multilingual coverage, and avoids the 2.24GB e5-large download on a
    fresh dev box. The Embedder Protocol allows swap at runtime, so a
    future Phase 2 sub-sprint can upgrade to bge-m3 (also 1024-dim) with
    only a config flip.

Context:
    Phase 1 S1.1 ssB. Consumed by ingest/pipeline.py (integrator) and by
    retrieve/hybrid.py for query embedding. The same instance is reused
    for indexing and query — fastembed is thread-safe per its model card.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import ClassVar

# Why type: ignore: fastembed ships no py.typed marker (as of 0.4). The
# Embedder Protocol boundary here is fully typed (list[list[float]]) so the
# untyped third-party only leaks at this single import; suppress narrowly.
from fastembed import TextEmbedding  # type: ignore[import-untyped]

# Why a curated whitelist with hard-coded dims:
#   fastembed exposes ``list_supported_models()`` at runtime, but loading
#   the model to discover its dim takes 30+ seconds (ONNX download +
#   tokenizer init). Hard-coding lets ``dim`` return instantly before any
#   model load — VectorStore needs ``dim`` to size its collection lazily.
_MODEL_DIMS: dict[str, int] = {
    "sentence-transformers/paraphrase-multilingual-mpnet-base-v2": 768,
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2": 384,
    "intfloat/multilingual-e5-large": 1024,
}

DEFAULT_MODEL = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"


def _resolve_cache_dir() -> str:
    """Model cache location: ``$FASTEMBED_CACHE_PATH`` or ``~/.cache/fastembed``.

    # What: always pass an explicit, home-anchored cache_dir to fastembed.
    # Why:  the pinned fastembed defaults to ``<tempdir>/fastembed_cache``
    #       (/tmp on Linux) when cache_dir is omitted — wiped on reboot, so
    #       every reboot re-downloads ~1GB+ of ONNX weights. ~/.cache is the
    #       XDG-standard persistent cache: survives reboots, outside any git
    #       repo, shared across venvs/projects.
    # Context: 2026-06-10 P1Fixes — a reboot cost a 25-min re-download mid-eval.
    """
    return os.environ.get("FASTEMBED_CACHE_PATH") or str(Path.home() / ".cache" / "fastembed")


class FastEmbedEmbedder:
    """ONNX multilingual embedder. Implements the ``Embedder`` Protocol.

    First call to ``embed()`` triggers an automatic model download (~1GB
    for the default model) into ``_resolve_cache_dir()`` (defaults to
    ``~/.cache/fastembed/``). Subsequent calls are cache hits and only pay
    tokenization + ONNX inference cost.
    """

    # Class-level marker so callers can detect impl identity without isinstance
    # imports cycling back through Protocol-only modules.
    backend: ClassVar[str] = "fastembed"

    def __init__(self, model_name: str = DEFAULT_MODEL) -> None:
        if model_name not in _MODEL_DIMS:
            # Why: fail fast with the closed set so an unknown name doesn't
            # silently lead to a wrong-dim collection later.
            raise ValueError(
                f"Unknown fastembed model {model_name!r}. Supported: {sorted(_MODEL_DIMS)}"
            )
        self._model_name = model_name
        self._dim = _MODEL_DIMS[model_name]
        # Lazy: don't pay download cost at construction; build on first embed.
        self._model: TextEmbedding | None = None

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def model_name(self) -> str:
        return self._model_name

    def _ensure_model(self) -> TextEmbedding:
        if self._model is None:
            self._model = TextEmbedding(model_name=self._model_name, cache_dir=_resolve_cache_dir())
        return self._model

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts. Order preserved.

        Empty list short-circuits — avoids triggering a model download for
        callers that probe ``dim`` only.
        """
        if not texts:
            return []
        model = self._ensure_model()
        # fastembed's ``embed`` returns a generator of numpy arrays.
        # We materialize as Python lists to satisfy the Protocol contract
        # (list[list[float]]) and to keep the impl deps-free downstream.
        return [vec.tolist() for vec in model.embed(texts)]
