# reference/

Curated external knowledge cached locally so future agents (and humans) can read project context **without needing internet access**. Each file is a short, focused cheatsheet — not a full paper dump — so an agent can skim the relevant one and proceed.

## Contents

| File | When to read |
|---|---|
| [`rag-evaluation-metrics.md`](rag-evaluation-metrics.md) | Designing or modifying `src/jcontract/eval/` — what to measure, target thresholds, common pitfalls |
| [`rrf-fusion.md`](rrf-fusion.md) | Touching `src/jcontract/retrieve/hybrid.py` — why RRF, k constant, alternatives |
| [`claude-vision-ocr.md`](claude-vision-ocr.md) | Working on `impls/claude_vision_parser.py` or any Vision-API code — request shape, cost, image limits |
| [`construction-contract-domain.md`](construction-contract-domain.md) | Anyone unfamiliar with building-construction tender docs — glossary (TSA, TQA, Rev, Drawing No., Clause) and the DEMO structure observed in this project |
| [`llm-subscription-vs-api.md`](llm-subscription-vs-api.md) | Working on Answerer impls (claude-api / claude-cli / codex-cli) — auth modes, cost models, CLI invocation patterns, when to use which |
| [`deepseek-v4-vision.md`](deepseek-v4-vision.md) | Working on `impls/deepseek_v4_parser.py` (Phase 1.10) or batch-ingest cost tuning — DeepSeek V4 OpenAI-compatible vision API shape, 4-vendor cost comparison, model selection guide |

## Rules

- **Each entry is "what you need + where it came from"** — both the summary and the source URL/citation, so an agent can re-verify if stale.
- **Date-stamped at the top** of each file. If it's older than 6 months and the agent is making a high-stakes decision, re-fetch.
- **No marketing copy** — keep the technical signal high.
- **Do not extend without need**: add new files only when a future agent would genuinely benefit. Files here are load-bearing, not aspirational.

## Why this directory exists

Per [`dev-contract/01-project-seasee.md`](../dev-contract/01-project-seasee.md) §2.5: future agents (including the same model in the next session) have **no memory of the current session's web searches**. Without reference/, decisions like "we chose RRF k=60 because Cormack 2009 recommends it" would require the agent to re-search and possibly land on different sources.

This is a `multi-agent / multi-session context loss` mitigation. Same spirit as `dev_log.md` (decisions), but for **external knowledge** rather than internal decisions.
