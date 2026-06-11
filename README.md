# quickQues

**A domain-agnostic document knowledge-base AI — production-style Retrieval-Augmented
Generation (RAG).** Ask questions in Chinese over a corpus of PDFs (contracts, financial
reports, specs, manuals) and get grounded Chinese answers with **clickable, page-level
citations**.

The domain-specific bits (OCR / caption / answer prompts, chunking structure, example
questions) live in a swappable **DomainProfile** (`profiles/<name>.yaml`); the core
pipeline (parse → chunk → retrieve → answer → eval) is domain-neutral. Adding a new
domain = a new profile + its own isolated knowledge base (collection) — **no core code
change**. The bundled example domain is a construction contract corpus.

> **Note on scope.** This is a portfolio/reference implementation extracted from a real
> single-tenant deployment. The shipped data (`profiles/contract.yaml`, the synthetic
> fixture, the golden eval set) is synthetic — no real client data is included.

---

## Highlights

- **Hybrid retrieval** — dense embeddings (`bge-m3` via fastembed) + sparse keyword
  search (BM25 / jieba), fused with **Reciprocal Rank Fusion (RRF)** and re-ranked by a
  **`bge-reranker-v2-m3` cross-encoder**. Cross-lingual: a Chinese question retrieves
  over English/Chinese source text. Vector store: **Qdrant**.
- **Forced citation grounding** — every factual sentence must emit an exact
  `[file p.page]` citation; a post-processor drops uncited sentences; low-confidence
  answers **degrade gracefully to retrieval-only mode** instead of hallucinating.
  Untrusted document text is tag-isolated in the prompt to resist injection.
- **Multimodal ingest** — swappable vision-OCR parsers for scanned PDFs and engineering
  drawings (Anthropic Claude Vision API · zero-key Claude Code CLI · DeepSeek V4 Vision),
  plus drawing "captioners" that make diagrams retrievable. Content-addressed,
  model-aware OCR/caption cache → idempotent, cost-bounded re-ingest.
- **Dependency-inverted, layered architecture** — Interfaces → Ingest → Retrieve+Answer
  → API → Web UI. **12 core capabilities each defined as a Python Protocol/ABC** with
  vendor implementations injected; business logic imports interfaces only, never vendor
  SDKs. Pluggable LLM backends (Claude API / Claude CLI / Codex CLI).
- **RAG evaluation harness** — retrieval recall@k with per-category rollup (incl.
  drawing-only cases), citation accuracy, an A/B eval-comparison tool, and a pluggable
  **LLM-as-a-judge** scoring faithfulness / answer-relevancy.
- **Full web app** — FastAPI backend (`/ask`, `/search`, `/healthz`, `/files`) with
  multi-collection support (`?collection=`), and a Next.js (App Router) + Tailwind chat
  UI whose citation chips deep-link into an in-browser PDF viewer at the cited page.
- **Engineering rigor** — strict-typed Python (`mypy`), `ruff` lint+format, and a full
  `pytest` suite, all behind a one-command quality gate.

---

## Architecture

```
┌─────────────────────────────────────┐
│ Layer 5: Web UI (Next.js + PDF view) │
├─────────────────────────────────────┤
│ Layer 4: API (FastAPI)               │
├─────────────────────────────────────┤
│ Layer 2: Retrieve + Answer           │
├─────────────────────────────────────┤
│ Layer 1: Ingest Pipeline             │
├─────────────────────────────────────┤
│ Layer 0: Interfaces + Schemas        │
│   (Protocol / dataclass)             │
└─────────────────────────────────────┘
        ↑ concrete impls injected from impls/<vendor>/
```

The 12 injectable interfaces: `PDFParser`, `OCREngine`, `VisionCaptioner`, `Chunker`,
`Embedder`, `VectorStore`, `KeywordIndex`, `Reranker`, `Answerer`, `RefGraph`, `Judge`,
`DomainProfile`. Swapping a vendor = swapping an impl; adding a domain = adding a profile.

---

## Quick start

Requires Python 3.12 + [uv](https://docs.astral.sh/uv/) + Docker.

```bash
uv sync                       # install into a uv-managed venv
cp .env.example .env          # fill ANTHROPIC_API_KEY etc. (optional — see below)
bash scripts/check.sh         # quality gate: ruff + mypy + pytest
```

`ANTHROPIC_API_KEY` is optional: without it the API falls back to **retrieval-only mode**
(citations, no LLM-written answer), or you can point the answerer at a Claude Code / Codex
CLI subscription (zero API key) via `JCONTRACT_ANSWERER_BACKEND`.

### Answerer backends

| Backend (`--answerer` / `JCONTRACT_ANSWERER_BACKEND`) | Runs on | Env vars (default) |
|---|---|---|
| `claude-api` (default) | Anthropic API, per-token | `ANTHROPIC_API_KEY` (required) |
| `claude-cli` | `claude` CLI, Claude Code subscription quota | none (needs `claude login`) |
| `codex-cli` | `codex` CLI, ChatGPT subscription quota | none (needs `codex login`) |
| `local` | any OpenAI-compatible endpoint — Ollama / vLLM / LM Studio; zero cost and zero data egress when local | `JCONTRACT_LOCAL_LLM_BASE_URL` (`http://localhost:11434/v1`), `JCONTRACT_LOCAL_LLM_MODEL` (`qwen3:14b`), `JCONTRACT_LOCAL_LLM_API_KEY` (`ollama` — placeholder, Ollama ignores it) |

### Caption backends (`ingest --caption`)

| Backend (`--caption-backend`) | Runs on | Env vars (default) |
|---|---|---|
| `claude-cli` (default) | `claude` CLI, Claude Code subscription quota | none (needs `claude login`) |
| `claude-api` | Anthropic API Vision, per-token | `ANTHROPIC_API_KEY` (required) |
| `deepseek` | DeepSeek V4 Vision, per-token | `DEEPSEEK_API_KEY` (required) |
| `ollama` | local VLM via Ollama — zero cost, page images never leave the machine | `JCONTRACT_OLLAMA_BASE_URL` (`http://localhost:11434`), `JCONTRACT_OLLAMA_VL_MODEL` (`qwen3-vl:8b`) |

### Run the app

```bash
# 1. Ingest at least one PDF (scanned PDFs need a vision parser)
uv run jcontract ingest path/to/doc.pdf --parser claude-cli-vision

# 2. Backend (bind 0.0.0.0 for non-local browsers)
uv run uvicorn jcontract.api.main:app --host 0.0.0.0 --port 8000 --reload

# 3. Frontend
cd web && npm install && npm run dev      # http://localhost:3000
```

### Add a new domain

```bash
# 1. Drop a profiles/<name>.yaml (prompts + chunking structure + example questions)
# 2. Ingest into an isolated collection
uv run jcontract ingest report.pdf --collection finance --domain finance
# 3. Query it — collections coexist in one process
curl 'http://localhost:8000/ask?collection=finance' -d '{"question":"..."}'
```

---

## Tech stack

Python 3.12 · FastAPI · Next.js (App Router) · Tailwind · Qdrant · `bge-m3` /
`bge-reranker-v2-m3` (fastembed) · rank-bm25 + jieba · Anthropic Claude · DeepSeek ·
SQLite (reference graph) · uv · ruff · mypy · pytest · Docker.

## License

[MIT](LICENSE)
