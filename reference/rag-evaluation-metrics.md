# RAG Evaluation Metrics — Cheatsheet

**Date stamped**: 2026-05-28. Re-check if reading after 2026-11.

## Core metric families

RAG eval splits into two stages — **retrieval** and **generation** — plus one **end-to-end** layer. Reference the one matching what you're measuring; don't conflate.

### 1. Retrieval metrics

Measure whether the right chunks come back, before any LLM synthesis.

| Metric | What it measures | Notes |
|---|---|---|
| **Recall@k** | Of all relevant docs that should be retrieved, what fraction appears in top-k | Binary form (per-query 0/1: any relevant in top-k) is common when per-query relevant counts are unknown. Fractional form (relevant_in_topk / total_relevant) needs labelled relevance counts. |
| **Precision@k** | Of the top-k retrieved, what fraction is actually relevant | Useful when you care about noise in the top-k context fed to the LLM. |
| **MRR (Mean Reciprocal Rank)** | 1 / rank_of_first_relevant, averaged across queries | Best when the user only reads top-1 / top-3. |
| **NDCG@k** | Rank-aware, supports graded relevance (1/2/3-star) | Heavier to label; usually overkill for prototypes. |
| **Context Precision / Recall** | RAGAS framework names; same family as above | Use when integrating RAGAS. |

### 2. Generation metrics

Measure the LLM-synthesized answer against the retrieved context and a ground truth.

| Metric | What it measures | How to check |
|---|---|---|
| **Faithfulness / Groundedness** | Every claim in the answer is supported by retrieved context (no hallucination) | LLM-as-judge or rule-based: split answer into claims, ask "does context support this?" |
| **Answer Relevance** | The answer addresses the question (not off-topic) | LLM-as-judge or embedding cos(question, answer) above threshold |
| **Citation Accuracy** | Every `[source p.X]` in the answer points to a real chunk that was actually retrieved | Programmatic check (no judge needed) — j-contract uses this in `eval/metrics.py::citation_accuracy()` |
| **Hallucination Rate** | Fraction of answer claims NOT in retrieved context | Inverse of faithfulness |
| **Keyword Hit Rate** | Whether expected key terms appear in the answer (cheap proxy for correctness) | Programmatic, no LLM needed — j-contract uses this when no ground-truth answer text exists |

### 3. End-to-end

| Metric | What it measures |
|---|---|
| **Factuality (vs. gold answer)** | Compare answer to a hand-authored gold answer (BLEU, ROUGE, or LLM-as-judge) |
| **Task success rate** | Whether the user could act on the answer (binary, human judge) |

## 2026 target thresholds (industry-common, not strict)

These come from the 2026 RAG eval blog posts cited below. Adjust for your domain — construction contracts are higher-stakes than support-bot QA, so targets shift up.

| Use case | Target |
|---|---|
| Narrow-domain knowledge base | **Precision@5 ≥ 0.7** |
| Broad corpus search | **Recall@20 ≥ 0.8** |
| Production-quality answers | **Faithfulness ≥ 0.85**, **Citation accuracy ≥ 0.95** |
| j-contract Phase 1 target (per `docs/project_guideline.md` §6) | Recall@5 > 85%, answer correctness > 80%, **citation accuracy > 95%** (non-tech users distrust wrong citations more than wrong answers) |

## Pitfalls

1. **Don't average scores across incomparable backends.** Vector cosine and BM25 scores are on different scales — see [`rrf-fusion.md`](rrf-fusion.md).
2. **Don't use LLM-as-judge with the same model that generates the answer.** Self-judging inflates scores. Use a different model family or use programmatic checks.
3. **Don't measure faithfulness without retrieval recall.** If retrieval missed the answer, the LLM may hallucinate — that's a retrieval bug, not a faithfulness bug.
4. **Avoid hand-picked golden cases that are too easy.** Cover failure categories explicitly: out-of-distribution questions, ambiguous questions, multi-hop questions, "not in document" questions.

## j-contract design choices anchored here

| Choice | Rationale |
|---|---|
| Use **Recall@5 and Recall@10** + **Citation Accuracy** + **Keyword Hit Rate** as core metrics (`eval/metrics.py`) | Programmatic only — no LLM-as-judge to keep eval cheap and reproducible. LLM judge can be added later for faithfulness. |
| **Binary Recall@k** (per-query 0/1) | Per `DECISION-1.1.10` (dev_log Phase 1) — lacks per-question relevant-chunk counts |
| **Page-range golden cases** (page_min/page_max) | Tolerates chunker-boundary shifts; cited in `DECISION-1.1.9` |
| **6 golden cases across 6 categories** | Coverage > volume for a prototype |
| Skip RAGAS for Phase 1 | RAGAS needs an additional LLM API; defer until faithfulness becomes a measured bottleneck |

## Frameworks (when scaling up)

- **RAGAS** — Python, supports faithfulness + answer relevance + context precision out of the box. Heavy: needs OpenAI/Anthropic API for the judge.
- **ARES** — Academic framework, similar to RAGAS.
- **LangSmith** — LangChain's eval platform; nice UI, locked to LangChain ecosystem.
- **Confident AI / deepeval** — Open-source pytest-flavored; can run programmatic + LLM-as-judge in CI.

For j-contract: stay with hand-rolled `eval/metrics.py` for as long as it fits. Add RAGAS/deepeval only when we need faithfulness measurement at scale.

## Sources

Cached on 2026-05-28 (search query: "RAG evaluation metrics recall@k MRR faithfulness citation accuracy 2026 best practices"):

- [RAG Evaluation: A Complete Guide for 2025 — Maxim](https://www.getmaxim.ai/articles/rag-evaluation-a-complete-guide-for-2025/)
- [RAG Evaluation: 2026 Metrics and Benchmarks for Enterprise AI Systems — Label Your Data](https://labelyourdata.com/articles/llm-fine-tuning/rag-evaluation)
- [RAG Evaluation Metrics, Frameworks & Testing (2026) — PreMAI](https://blog.premai.io/rag-evaluation-metrics-frameworks-testing-2026/)
- [RAG Evaluation Metrics: Answer Relevancy, Faithfulness, And More — Confident AI](https://www.confident-ai.com/blog/rag-evaluation-metrics-answer-relevancy-faithfulness-and-more)
- [Evaluation Metrics for RAG Systems — GeeksforGeeks](https://www.geeksforgeeks.org/nlp/evaluation-metrics-for-retrieval-augmented-generation-rag-systems/)
