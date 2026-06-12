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
  over English/Chinese source text. Vector store: **Qdrant**. Embedding batch size is
  env-tunable (`JCONTRACT_EMBED_BATCH`, default 256) to bound memory on low-RAM hosts.
- **Forced citation grounding** — every factual sentence must emit an exact
  `[file p.page]` citation; a post-processor drops uncited sentences; low-confidence
  answers **degrade gracefully to retrieval-only mode** instead of hallucinating.
  Untrusted document text is tag-isolated in the prompt to resist injection.
- **Multimodal ingest** — swappable vision-OCR parsers for scanned PDFs and engineering
  drawings (Anthropic Claude Vision API · zero-key Claude Code CLI · DeepSeek V4 Vision),
  plus drawing "captioners" that make diagrams retrievable. Content-addressed,
  model-aware OCR/caption cache → idempotent, cost-bounded re-ingest.
- **Dependency-inverted, layered architecture** — Interfaces → Ingest → Retrieve+Answer
  → API → Web UI. **13 core capabilities each defined as a Python Protocol/ABC** with
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

The 13 injectable interfaces: `PDFParser`, `OCREngine`, `VisionCaptioner`, `Chunker`,
`Embedder`, `VectorStore`, `KeywordIndex`, `Reranker`, `Answerer`, `RefGraph`, `Judge`,
`DomainProfile`, `Redactor`. Swapping a vendor = swapping an impl; adding a domain =
adding a profile.

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

### Parser backends (`ingest --parser`)

| Backend (`--parser`) | Runs on | Env vars (default) |
|---|---|---|
| `pypdf` (default) | pure-Python text extraction — free, but blind on scanned/image PDFs | none |
| `claude-vision` | Anthropic API Vision, per-token | `ANTHROPIC_API_KEY` (required) |
| `claude-cli-vision` | `claude` CLI, Claude Code subscription quota | none (needs `claude login`) |
| `deepseek-v4` | DeepSeek V4 Vision, per-token (cheapest API option) | `DEEPSEEK_API_KEY` (required) |
| `rapidocr` | local CPU OCR (PP-OCRv5 via ONNX Runtime) — zero cost, fully offline after a one-time ~20MB model download; lower fidelity than LLM vision | none |

#### Auto-rotate (`--auto-rotate`, rapidocr only, opt-in)

Scanned corpora often contain sideways/upside-down pages that OCR into ordered
fragments. With `--auto-rotate` (off by default — zero behaviour change), pages whose
initial OCR quality is low (per-box `min_score < 0.756`, the same "low quality" signal
`ocr-quality` flags on) are re-OCR'd in all four 90° rotations and the direction with
the highest OCR mass (chars × mean confidence) wins, provided it beats the as-rendered
frame by ≥1.10x. The decision (plus four-direction evidence) is cached in a
`*.rotation.json` sidecar keyed by the original frame's hash, so re-ingest never
re-probes; the upright frame's OCR lands in its own content-addressed cache entry.
`table-preview --auto-rotate` reuses the same probe and sidecar before structuring.

```bash
uv run jcontract ingest scan.pdf --parser rapidocr --auto-rotate
uv run jcontract table-preview scan.pdf --page 9 --auto-rotate --format elements
```

#### Region-aware assembly (`--assembly regions`, rapidocr only, opt-in)

The default reading-order assembly sweeps boxes top-to-bottom in y bands, which
interleaves side-by-side layouts (two columns, label/value tables) into mixed lines.
`--assembly regions` (default: `default` — zero behaviour change) splits the page into
horizontal strips on empty y bands, splits each strip into columns on empty x channels,
and reads columns left-to-right, so column text comes out contiguous. A non-default
mode caches into its own `.regions`-suffixed namespace (`rapidocr-<sha>.text.regions.txt`)
and never touches existing cache entries. Available on `ingest`, `ocr-quality`, and
`ocr-gallery`.

Layout problems are also *detectable* before opting in: every fresh OCR pass stores four
page-geometry signals in the metrics sidecar — `n_columns` (empty-channel column count),
`max_band_gap` (widest in-line blank, fraction of page width), `box_coverage` (text-box
area / page area), `order_divergence` (how much the default and region orders disagree) —
and `ocr-quality`/`ocr-gallery` accept them in `--flag-below`/`--flag-above` rules.
Records written before this feature simply report `null` for these signals.

```bash
uv run jcontract ocr-quality scan.pdf --flag-above n_columns:1 --flag-above max_band_gap:0.25
uv run jcontract ingest scan.pdf --parser rapidocr --assembly regions
```

#### Needs-vision classifier v2 (`JCONTRACT_PAGE_CLASSIFY=v2`, rapidocr only, opt-in)

The text-vs-drawing verdict (`page_kind`) gates the drawing/caption lane. The default
v1 classifier is pixel-only and has two known failure classes: dense spec drawings /
maps / charts pass as "text" (their graphics never become retrievable), and near-empty
divider/title pages trigger as "drawing" (wasted captions — measured 64.5% empty). With
`JCONTRACT_PAGE_CLASSIFY=v2` (default `v1` — zero behaviour change) the rapidocr parser
re-frames the question as "is the text alone enough?": pages whose OCR text arrives as
many small fragment boxes (mean box area < 0.1% of the page) are drawings; pages with
almost no box coverage *and* almost no ink are empty-ish dividers and stay text. Box
statistics are reused from the metrics sidecar (`boxes`, `box_coverage`); pages whose
cached sidecar predates the geometry signals keep the v1 verdict (existing sidecars are
never rewritten — re-OCR into a fresh cache for full v2 coverage). The verdict is
biased toward "drawing" on ambiguity: captions are additive (OCR text still indexes),
while a missed drawing is permanently unretrievable. LLM vision parsers have no box
data and always use v1.

```bash
JCONTRACT_PAGE_CLASSIFY=v2 uv run jcontract ingest scan.pdf --parser rapidocr --caption
```

### Redaction preview (`redact-preview`)

Reversible pseudonymization for confidential corpora — a standalone mechanism component
(not wired into ingest): a caller-supplied dictionary (entity literals) + regex whitelist
replace sensitive mentions with corpus-stable `<TYPE_N>` placeholders; a persistent
mapping store guarantees the same entity gets the same placeholder across files and
sessions, and `--restore` reverses byte-exactly. Zero new dependencies, no NER.

`--tier strict` (default: `standard`, the behaviour above) additionally masks every
capitalized-word sequence (`<PN_N>`) and every >=2-digit string (`<NUM_N>`) — a
deliberately over-masking, rule-based setting for text that is about to leave the
machine; the lowercase word skeleton survives and the result restores byte-exactly
through the same mapping store.

```bash
# dictionary + mapping store live in YOUR data directory, never in this repo;
# the mapping store is the restore key — gitignore it.
export JCONTRACT_REDACTION_DICT=path/to/dictionary.yaml
export JCONTRACT_REDACTION_MAP=path/to/maps/corpus.map.jsonl
uv run jcontract redact-preview page.txt --out page.redacted.txt
uv run jcontract redact-preview page.redacted.txt --restore --out page.roundtrip.txt
diff page.txt page.roundtrip.txt   # empty = byte-exact
```

### Dispatch plan (`dispatch-plan`)

Deterministic page→provider routing *plan* for multi-vendor corpora — a standalone
mechanism component (not wired into ingest, zero network): each page's rendered-JPEG
sha256 picks a provider name from your pool via `hash % pool_size`, so the same PDF and
pool always produce a byte-identical plan (idempotent, cache-friendly, resumable).
Assignments are appended to a JSONL provenance log (audit trail; re-runs append nothing).
Pool entries are opaque names — no vendor SDK is imported, no client constructed.

```bash
# pool is required (flag or JCONTRACT_DISPATCH_POOL env) — order matters.
uv run jcontract dispatch-plan doc.pdf --pool claude,openai \
  --out plan.jsonl --provenance provenance.jsonl
```

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
