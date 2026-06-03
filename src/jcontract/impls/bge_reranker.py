"""BGE multilingual cross-encoder reranker (Phase 1.8 P1 retrieval quality).

What:
    Implements the ``Reranker`` Protocol with BAAI/bge-reranker-v2-m3, a
    multilingual cross-encoder that re-scores ``(question, chunk.text)``
    pairs end-to-end inside one transformer forward pass. Unlike a
    bi-encoder (FastEmbedEmbedder), the cross-encoder reads both inputs
    together and is materially stronger at fine-grained relevance — at
    the cost of needing one forward pass per candidate.

Why we apply rerank only to fused top-N:
    Cross-encoding the whole corpus is O(N) per query (vs O(1) for a
    vector backend). The hybrid retriever stays the candidate generator;
    this reranker is the cleanup step on at most ~20-50 fused hits. That
    keeps p50 query latency bounded while letting us pay full attention
    to the candidate set the user actually sees.

Why sentence-transformers (DECISION — added new runtime dep):
    fastembed 0.3.6 (pinned transitively by qdrant-client[fastembed]==1.12.1)
    ships only bi-encoder text/image/late-interaction; it has no
    cross-encoder support. Upgrading fastembed past 0.4 would require
    unpinning qdrant-client which is risky for a single feature. The
    8-question dep check (dev-contract/24-domain-deps-env.md) on
    sentence-transformers==5.5.1 passes:

      1. stdlib enough — no, tensor inference required
      2. existing deps enough — no (fastembed 0.3.6 has no reranker)
      3. active — yes (UKPLab + HF, weekly releases, 15k+ stars)
      4. license — Apache 2.0, compatible
      5. size — heavy (~2GB w/ torch + nvidia wheels) but we'll
         need torch by Phase 2 (bge-m3 etc.) — paying it now is fine
      6. maintainers trustworthy — UKPLab / HuggingFace, established
      7. CVE — none known as of 2026-05
      8. alternatives — community ONNX exports (typosquatting risk +
         unverified provenance), FlagEmbedding (superset of ST),
         own ONNX runner (out of scope). ST is the BGE authors'
         own README-recommended path.

    Version pinned exact (==5.5.1) per pyproject.toml convention.

Why lazy model load:
    The model is ~568MB on disk and triggers a download + ~2-5s init on
    first use. Callers that build the reranker but never call ``rerank()``
    (e.g. test collection traversal, CLI command that didn't hit the
    rerank path) must not pay that cost.

Context:
    Phase 1.8 P1 sub-sprint ssC. Will be wired into ``retrieve/hybrid.py``
    after RRF fusion by the integrator agent (NOT in this sub-sprint —
    we only land the impl + tests). Protocol contract:
    ``src/jcontract/interfaces/reranker.py``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar, cast

from jcontract.interfaces import SearchResult

if TYPE_CHECKING:
    # Keep the heavy import out of module-load time. sentence-transformers
    # transitively imports torch (~2GB on disk, ~3-5s import time). Pulling
    # it on every ``from jcontract.impls.bge_reranker import ...`` would
    # punish every CLI invocation and every test that just collects the
    # module. We only need the type at typecheck time.
    from sentence_transformers import CrossEncoder

# BAAI/bge-reranker-v2-m3 — multilingual (CN + EN + ~100 languages),
# 568MB on disk, ~50ms/pair on CPU. The "v2-m3" variant is the current
# (2024-2026) BGE reranker default; multilingual matches our CN+EN corpus.
#
# Smaller alternatives (swap via constructor):
#   - "BAAI/bge-reranker-base"  — ~300MB, ~30ms/pair, EN+CN biased
#   - "BAAI/bge-reranker-large" — ~560MB, EN-focused
#   - "BAAI/bge-reranker-v2-gemma" — better but 2.5GB and English-leaning
DEFAULT_MODEL = "BAAI/bge-reranker-v2-m3"

# Empirical: 16 fits comfortably in 8GB RAM with the m3 model; bumping to
# 32 gains ~10% throughput on CPU but risks OOM on small machines. The
# constructor takes batch_size so test envs can drop it.
DEFAULT_BATCH_SIZE = 16


class BgeReranker:
    """Multilingual cross-encoder reranker. Implements ``Reranker`` Protocol.

    Re-scores ``(question, chunk.text)`` pairs and returns the candidates
    reordered (descending) by the new score. Input length is preserved —
    we never drop candidates here; threshold-based filtering belongs to
    the caller so policy stays in one place.
    """

    # Class-level marker so callers can detect impl identity without
    # importing the impl module just to ``isinstance`` check.
    backend: ClassVar[str] = "bge-cross-encoder"

    def __init__(
        self,
        *,
        model: str = DEFAULT_MODEL,
        batch_size: int = DEFAULT_BATCH_SIZE,
    ) -> None:
        # No validation against a closed model list (unlike fastembed_embedder):
        # cross-encoders are interchangeable via HF Hub names, and an unknown
        # name will fail clearly at first ``rerank()`` call when
        # CrossEncoder() raises. Closing the list would only re-encode the
        # HF model registry here, which adds maintenance burden for no gain.
        self._model_name = model
        self._batch_size = batch_size
        # Lazy: model load happens on first rerank(), not __init__.
        self._model: CrossEncoder | None = None

    @property
    def model_name(self) -> str:
        return self._model_name

    def _ensure_model(self) -> CrossEncoder:
        if self._model is None:
            # Local import so the heavy torch + transformers stack is only
            # paid when reranking actually runs. ~3-5s first-call overhead.
            from sentence_transformers import CrossEncoder

            # device=None lets sentence-transformers auto-pick CUDA if
            # available else CPU. We don't force CPU because future GPU
            # deployment (Phase 4+) should work without code change.
            self._model = CrossEncoder(self._model_name)
        return self._model

    def rerank(
        self,
        question: str,
        candidates: list[SearchResult],
    ) -> list[SearchResult]:
        """Re-score candidates against ``question`` with a cross-encoder.

        Returns candidates in descending score order. Length unchanged.
        Empty input short-circuits without loading the model (mirror of
        FastEmbedEmbedder.embed's empty-batch behaviour).
        """
        # Short-circuit BEFORE _ensure_model so callers can probe rerank()
        # on an empty candidate list without paying the model download.
        if not candidates:
            return []
        # Single-item input: a cross-encoder pass still produces a score,
        # but the ordering is degenerate. Score-replacement is the only
        # observable effect; we still run inference so the caller can use
        # the absolute score for confidence/threshold decisions downstream.
        # (Not optimized away — explicit consistency > micro-optimization.)

        model = self._ensure_model()
        pairs = [(question, c.chunk.text) for c in candidates]

        # ``predict`` returns a numpy array of shape (len(pairs),) when
        # convert_to_numpy=True (the default). Each score is the raw
        # classifier logit — higher = more relevant. We don't apply softmax
        # because we only need ordering + a comparable scalar; softmax over
        # 1-class output is identity-ish anyway.
        #
        # Why ``cast``: sentence-transformers 5.5 types ``predict``'s input
        # as a giant union covering text/image/audio/video pair inputs.
        # ``list`` is invariant in Python's type system, so our concrete
        # ``list[tuple[str, str]]`` doesn't satisfy the broader union even
        # though the runtime accepts it. Narrowing to ``Any`` at the call
        # site is the minimal-blast-radius way to honour mypy strict mode
        # without losing the type of ``pairs`` elsewhere.
        scores = model.predict(
            cast(Any, pairs),
            batch_size=self._batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
        )

        # Pair each original SearchResult with its new score, sort desc.
        # We build a fresh SearchResult (frozen dataclass) so callers
        # comparing before/after don't accidentally share mutable state.
        rescored = [
            SearchResult(chunk=cand.chunk, score=float(new_score))
            for cand, new_score in zip(candidates, scores, strict=True)
        ]
        rescored.sort(key=lambda r: r.score, reverse=True)
        return rescored
