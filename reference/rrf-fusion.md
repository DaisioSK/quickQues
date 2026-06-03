# Reciprocal Rank Fusion (RRF) — Cheatsheet

**Date stamped**: 2026-05-28.

## What RRF is

A rank-based fusion algorithm for combining N ranked result lists into one. For each unique item, sum `1 / (k_constant + rank_in_list)` across all lists it appears in. Higher fused score = better.

```
def rrf_fuse(rankings, k_constant=60):
    scores = {}
    for ranking in rankings:
        for rank, item in enumerate(ranking, start=1):
            scores[item.id] = scores.get(item.id, 0.0) + 1.0 / (k_constant + rank)
    return sorted(scores.items(), key=lambda x: -x[1])
```

## Why we use it in j-contract

j-contract's hybrid retriever runs two backends in parallel:
- **VectorStore** → cosine similarity scores (0-1, higher better)
- **KeywordIndex** → BM25 scores (positive, unbounded, scale depends on corpus)

**The two scores are incomparable.** Linear weighted fusion (`score = α·cos + β·BM25`) requires per-corpus calibration of α/β — brittle, breaks when corpus changes.

RRF sidesteps this entirely: it uses **rank**, not score. An item at rank 1 in either list gets `1/(k+1)` regardless of whether the score was 0.95 cosine or 47.3 BM25.

## The `k_constant` parameter

- **k = 60** is the canonical value from Cormack et al. (SIGIR 2009)
- Lower k (e.g. 10) → top-ranked items dominate heavily; mid-rank items rarely surface
- Higher k (e.g. 200) → flatter weighting; deeper-rank items can compete
- For j-contract: stick with 60 unless evaluation shows top-1 being correct but top-2/3 being noise (try k=30); or top-1 being wrong but the right answer is at rank 4-5 in both lists (try k=120).

## When NOT to use RRF

- When you have **labelled training data** to learn fusion weights → learn-to-rank (LambdaMART etc.) beats RRF
- When you have **identical score scales** (e.g. two vector backends with the same embedder) → arithmetic mean works fine
- When you want to combine **>5 backends** → RRF stays robust but may want CombMNZ (Fox & Shaw 1994) — still rank-based but emphasises items found by multiple backends

## Common mistakes

1. **Mixing relevance score and rank.** Don't compute RRF on `1/(k+score)`; it must be `1/(k+rank)`. Score is irrelevant to RRF.
2. **De-duplication by reference instead of stable id.** When the same chunk appears in two backends with slightly different object instances, dedupe by `chunk.id`, not by Python object identity.
3. **Truncating per-backend results too aggressively.** If you only take top-3 from each backend, items at rank 4+ never get a chance. j-contract uses `per_backend_k=20` then truncates after fusion.

## j-contract impl reference

- Implementation: [`src/jcontract/retrieve/hybrid.py`](../src/jcontract/retrieve/hybrid.py) — `rrf_fuse()` and `HybridRetriever.search()`
- Decision rationale: `DECISION-1.1.11` in [`docs/dev_log.md`](../docs/dev_log.md)
- Default constants: `RRF_K = 60`, `per_backend_k = 20`, return `top-k = 5`

## Sources

- **Primary paper**: Cormack, Clarke & Buettcher. *Reciprocal Rank Fusion outperforms Condorcet and individual rank learning methods.* SIGIR 2009. ([PDF mirror](https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf))
- Implementation references: Elasticsearch RRF, Weaviate hybrid, Qdrant's own RRF utility (qdrant-client ≥ 1.10 has a built-in `models.Fusion.RRF`)
- General overview: [RRF in retrieval — Pinecone blog](https://www.pinecone.io/learn/hybrid-search-intro/) (cached concept; URL may change)
