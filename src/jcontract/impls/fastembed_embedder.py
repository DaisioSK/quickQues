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
    fresh dev box. The Embedder Protocol allows swap at runtime: e5-large
    is built in, and BAAI/bge-m3 (1024-dim) is registered below as a
    fastembed custom model [DECISION-ab3.1 dev-sprint v3 §13] — select via
    JCONTRACT_EMBED_MODEL.

Context:
    Phase 1 S1.1 ssB. Consumed by ingest/pipeline.py (integrator) and by
    retrieve/hybrid.py for query embedding. The same instance is reused
    for indexing and query — fastembed is thread-safe per its model card.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import ClassVar

# Note: fastembed 0.3.x shipped no py.typed marker and needed a narrow
# `type: ignore[import-untyped]` here; 0.8.0 is typed, so imports are clean.
from fastembed import TextEmbedding
from fastembed.common.model_description import ModelSource, PoolingType

# Why a curated whitelist with hard-coded dims:
#   fastembed exposes ``list_supported_models()`` at runtime, but loading
#   the model to discover its dim takes 30+ seconds (ONNX download +
#   tokenizer init). Hard-coding lets ``dim`` return instantly before any
#   model load — VectorStore needs ``dim`` to size its collection lazily.
_MODEL_DIMS: dict[str, int] = {
    "sentence-transformers/paraphrase-multilingual-mpnet-base-v2": 768,
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2": 384,
    "intfloat/multilingual-e5-large": 1024,
    "BAAI/bge-m3": 1024,
}

DEFAULT_MODEL = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"


def _register_bge_m3() -> None:
    """Register BAAI/bge-m3 (dense) as a fastembed custom model. Idempotent.

    What:
        fastembed 0.8.0 has no built-in bge-m3 dense model; this registers
        the official BAAI ONNX export via ``TextEmbedding.add_custom_model``
        so ``FastEmbedEmbedder(model_name="BAAI/bge-m3")`` just works.

    Why these exact parameters (verified against the HF repo on 2026-06-10):
        - sources/model_file: the *official* ``BAAI/bge-m3`` repo ships an
          ``onnx/`` export (graph ``onnx/model.onnx`` + external weights
          ``onnx/model.onnx_data``, ~2.27GB). Chosen over community exports
          (aapot/bge-m3-onnx etc.): first-party = no typosquat risk, MIT
          license, 28M+ downloads. [DECISION-ab3.20 dev-sprint v3 §13]
        - pooling=CLS + normalization=True: matches the model's own
          ``1_Pooling/config.json`` (``pooling_mode_cls_token: true``); the
          ONNX graph's first output is ``token_embeddings`` (B, S, 1024), so
          fastembed's CLS pooling + normalize reproduces the official dense
          embedding.
        - dim=1024: ``config.json`` hidden_size (XLMRobertaModel).
        - additional_files: only ``onnx/model.onnx_data`` — the sole external
          file the graph references; keeps the snapshot download from pulling
          the 2.27GB ``pytorch_model.bin``.

    Why idempotent guard instead of try/except:
        ``add_custom_model`` raises ValueError on duplicate registration;
        re-imports (pytest collection, importlib.reload) must not blow up,
        and swallowing ValueError blindly could mask real config errors.

    Context:
        EmbedAB3 ssA2 — bge-m3 is the Phase-2 candidate dense model for the
        embedder A/B eval. [DECISION-ab3.1 dev-sprint v3 §13]
    """
    registered = {m["model"] for m in TextEmbedding.list_supported_models()}
    if "BAAI/bge-m3" in registered:
        return
    TextEmbedding.add_custom_model(
        model="BAAI/bge-m3",
        pooling=PoolingType.CLS,
        normalization=True,
        sources=ModelSource(hf="BAAI/bge-m3"),
        dim=1024,
        model_file="onnx/model.onnx",
        description="Multilingual dense embedding (XLM-RoBERTa), official BAAI ONNX export.",
        license="mit",
        size_in_gb=2.27,
        additional_files=["onnx/model.onnx_data"],
    )


# Module-level: registration only mutates fastembed's in-process model
# registry (no download, no I/O), so import stays cheap and every consumer
# of this module sees bge-m3 as available.
_register_bge_m3()


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


def _resolve_batch_size() -> int:
    """Embedding batch size: ``$JCONTRACT_EMBED_BATCH`` or fastembed's default 256.

    # What: expose fastembed's ``embed(batch_size=...)`` as an env knob.
    # Why:  onnxruntime activation memory scales with batch_size x the longest
    #       sequence in the batch. With long-sequence models (bge-m3, 8192
    #       tokens) and full-page chunks, batch 256 peaks ~15GB RSS and gets
    #       OOM-killed on 16GB hosts (observed twice on a 1049-page volume,
    #       2026-06-11, DECISION-cq.7). A smaller batch trades throughput for
    #       a bounded arena. Default 256 == fastembed's own default, so
    #       behavior is unchanged unless the env is set.
    # Context: stopgap for FORESHADOW-ls.5; the structural fix (staged
    #       subprocesses / streaming upsert) is charted under E13.
    """
    raw = os.environ.get("JCONTRACT_EMBED_BATCH", "")
    if not raw:
        return 256
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"JCONTRACT_EMBED_BATCH must be an integer, got {raw!r}") from exc
    if value < 1:
        raise ValueError(f"JCONTRACT_EMBED_BATCH must be >= 1, got {value}")
    return value


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
        # batch_size is env-tunable to bound onnxruntime arena memory on
        # low-RAM hosts (see _resolve_batch_size Why).
        return [vec.tolist() for vec in model.embed(texts, batch_size=_resolve_batch_size())]
