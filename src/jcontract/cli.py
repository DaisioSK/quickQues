"""Typer-based CLI: jcontract ingest / search / evaluate.

Wires impls + ingest pipeline + hybrid retriever + (optional) Claude
answerer + eval runner. Operates against a single Qdrant collection
(default ``contract``) plus a JSONL chunk snapshot at
``data/chunks_snapshot.jsonl`` (used to rehydrate the in-memory BM25
index across CLI invocations).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated

import structlog
import typer

from jcontract.eval.compare import compare_reports
from jcontract.eval.runner import run_eval
from jcontract.impls.bm25_index import Bm25Index
from jcontract.impls.domain_profile_registry import load_profile
from jcontract.impls.fastembed_embedder import DEFAULT_MODEL as EMBED_DEFAULT_MODEL
from jcontract.impls.fastembed_embedder import FastEmbedEmbedder
from jcontract.impls.pypdf_parser import PyPdfParser
from jcontract.impls.qa_chunker import QaAwareChunker
from jcontract.impls.qdrant_store import QdrantStore
from jcontract.impls.sqlite_ref_graph import SqliteRefGraph
from jcontract.ingest.pipeline import IngestPipeline, load_chunks_snapshot
from jcontract.interfaces import (
    Answer,
    Answerer,
    Chunk,
    DomainProfile,
    EvalCase,
    Judge,
    PDFParser,
    SearchResult,
    VisionCaptioner,
)
from jcontract.paths import (
    apply_layout_migration,
    legacy_files_present,
    paths_for,
    plan_layout_migration,
    read_profile_name,
    write_profile_sidecar,
)
from jcontract.retrieve.hybrid import HybridRetriever

logger = structlog.get_logger(__name__)

# Phase 7 SS5: data artifacts are per-collection (data/<collection>/...);
# resolve them via paths_for(collection) inside each command. Only the
# golden eval set is collection-independent (it lives in the source tree).
GOLDEN_CASES_PATH = Path("src/jcontract/eval/golden_cases.jsonl")

app = typer.Typer(
    help="j-contract — construction contract knowledge retrieval AI",
    no_args_is_help=True,
)


@dataclass
class Stack:
    """Bundle of all wired components — built once per CLI invocation."""

    embedder: FastEmbedEmbedder
    vector_store: QdrantStore
    keyword_index: Bm25Index
    retriever: HybridRetriever


def _build_stack(collection: str, *, use_reranker: bool = False) -> Stack:
    """Construct all impls and wire them. Loads BM25 from snapshot if present.

    `use_reranker=True` plugs `BgeReranker` (cross-encoder) into the
    retriever. Off by default because model load is heavy (~2GB on disk,
    multi-second first-call latency). Recommended for evals at 4k+ chunk
    scale where vector+BM25 alone gets noisy.
    """
    # What: embedding model is env-selectable (JCONTRACT_EMBED_MODEL); the
    #       whitelist in fastembed_embedder still fail-fasts unknown names.
    # Why:  A/B-ing dense models (mpnet vs e5-large) needs side-by-side
    #       collections; a flag through every command signature is heavier
    #       than one env read at the single construction site (ssEmbedAB).
    #       Dim mismatches against an existing collection surface loudly as
    #       Qdrant errors, so a wrong env can't silently cross-query.
    embed_model = os.environ.get("JCONTRACT_EMBED_MODEL", EMBED_DEFAULT_MODEL)
    embedder = FastEmbedEmbedder(model_name=embed_model)
    if embed_model != EMBED_DEFAULT_MODEL:
        logger.info("cli.embed_model_override", model=embed_model, dim=embedder.dim)
    # QdrantStore infers vector size from the first `add()` batch — no need
    # to pass embedder.dim explicitly (see impls/qdrant_store.py docstring).
    vector_store = QdrantStore(collection_name=collection)
    keyword_index = Bm25Index()

    # Rehydrate BM25 from prior ingest if a snapshot exists. Qdrant survives
    # process restarts; the in-memory BM25 does not, hence the snapshot.
    # Snapshot is per-collection (Phase 7 SS5).
    if legacy_files_present():
        logger.warning(
            "cli.legacy_data_layout",
            hint="found pre-Phase-7 flat data/ files; run `jcontract migrate-layout` "
            "to move them into data/<collection>/ (otherwise this collection looks empty)",
        )
    snapshot_path = paths_for(collection).chunks_snapshot
    cached = load_chunks_snapshot(snapshot_path)
    if cached:
        keyword_index.add(cached)
        logger.info("cli.bm25_rehydrated", chunks=len(cached))
    else:
        # What: shout when BM25 has nothing to rehydrate but Qdrant holds
        #       vectors — hybrid retrieval silently degrades to vector-only.
        # Why:  exactly this bit us on 2026-06-08 (recall report ran for days
        #       on vector-only without anyone noticing). A fresh/empty
        #       collection is fine and stays quiet; Qdrant being unreachable
        #       must not block stack construction, hence the broad except.
        try:
            indexed = vector_store.count()
        except Exception:  # noqa: BLE001 — health is checked elsewhere
            indexed = 0
        if indexed > 0:
            logger.warning(
                "cli.bm25_snapshot_missing",
                collection=collection,
                expected_path=str(snapshot_path),
                qdrant_points=indexed,
                hint="hybrid retrieval degraded to vector-only; restore the "
                "snapshot or re-ingest so BM25 can rehydrate",
            )

    reranker = None
    if use_reranker:
        # Local import: keeps sentence-transformers + torch out of the
        # cold-start path when the user runs without --rerank.
        from jcontract.impls.bge_reranker import BgeReranker

        reranker = BgeReranker()
        logger.info("cli.reranker_enabled", model=reranker.model_name)

    retriever = HybridRetriever(embedder, vector_store, keyword_index, reranker=reranker)
    return Stack(embedder, vector_store, keyword_index, retriever)


def _maybe_build_answerer(
    backend: str = "claude-api", domain_framing: str | None = None
) -> Answerer | None:
    """Construct an Answerer for the chosen backend; None if requirements unmet.

    The eval pipeline gracefully degrades when no answerer is available
    (only retrieval metrics get computed). ``domain_framing`` (from the
    collection's DomainProfile) flows into the answer prompt; None → contract.

    Backends:
      - claude-api  : Anthropic API direct (needs ANTHROPIC_API_KEY)
      - claude-cli  : `claude` CLI via subprocess (needs `claude login`,
                      uses Claude Code subscription quota — no per-call $$)
      - codex-cli   : `codex` CLI via subprocess (needs `codex login`,
                      uses ChatGPT Plus/Pro subscription quota)
      - local       : any OpenAI-compatible endpoint, default local Ollama
                      (JCONTRACT_LOCAL_LLM_BASE_URL / _MODEL / _API_KEY;
                      zero cost, zero data egress)
    """
    if backend == "claude-api":
        if not os.environ.get("ANTHROPIC_API_KEY"):
            return None
        # Local import — keeps anthropic SDK out of CLI startup when not used.
        from jcontract.impls.claude_answerer import ClaudeAnswerer

        return ClaudeAnswerer(domain_framing=domain_framing)
    if backend == "claude-cli":
        # No env check — the impl raises a clear error at construction time
        # if `claude` binary isn't in PATH.
        from jcontract.impls.claude_cli_answerer import ClaudeCliAnswerer

        return ClaudeCliAnswerer(domain_framing=domain_framing)
    if backend == "codex-cli":
        from jcontract.impls.codex_cli_answerer import CodexCliAnswerer

        return CodexCliAnswerer(domain_framing=domain_framing)
    if backend == "local":
        # No env check — every JCONTRACT_LOCAL_LLM_* var has a safe local
        # default (Ollama on localhost). Endpoint problems surface at call
        # time as the graceful fallback answer, not at construction.
        from jcontract.impls.openai_compat_answerer import OpenAICompatAnswerer

        return OpenAICompatAnswerer(domain_framing=domain_framing)
    raise typer.BadParameter(
        f"Unknown answerer backend '{backend}'. "
        "Choose from: claude-api, claude-cli, codex-cli, local."
    )


def _build_judge(backend: str) -> Judge:
    """Construct a Judge (LLM-as-judge for answer quality) — Enhancement E12.

    Backends:
      - claude-cli (default): `claude` CLI subprocess, subscription, NO key.
    Lazy import keeps the impl out of cold-start when --judge is off.
    """
    if backend == "claude-cli":
        from jcontract.impls.claude_cli_judge import ClaudeCliJudge

        return ClaudeCliJudge()
    raise typer.BadParameter(f"Unknown judge backend '{backend}'. Choose from: claude-cli.")


def _build_captioner(backend: str, profile: DomainProfile | None = None) -> VisionCaptioner:
    """Construct a VisionCaptioner for the chosen backend (Enhancement E11).

    Backends:
      - claude-cli (default): ``claude`` CLI subprocess, subscription quota,
                              NO API key — matches the no-key project default.
      - claude-api          : Anthropic SDK Vision (needs ANTHROPIC_API_KEY).
      - deepseek            : DeepSeek V4 Vision via OpenAI-compat API
                              (needs DEEPSEEK_API_KEY).

    Lazy imports keep the heavy SDKs out of cold-start when --caption is off.
    ``profile`` (Phase 7) supplies the caption prompt; None → contract default.
    """
    if backend == "claude-cli":
        from jcontract.impls.claude_cli_vision_captioner import ClaudeCliVisionCaptioner

        return ClaudeCliVisionCaptioner(profile=profile)
    if backend == "claude-api":
        from jcontract.impls.claude_vision_captioner import ClaudeVisionCaptioner

        return ClaudeVisionCaptioner(profile=profile)
    if backend == "deepseek":
        from jcontract.impls.deepseek_vision_captioner import DeepSeekVisionCaptioner

        return DeepSeekVisionCaptioner(profile=profile)
    if backend == "ollama":
        from jcontract.impls.ollama_vision_captioner import OllamaVisionCaptioner

        return OllamaVisionCaptioner(profile=profile)
    raise typer.BadParameter(
        f"Unknown caption backend '{backend}'. "
        f"Choose from: claude-cli, claude-api, deepseek, ollama."
    )


def _build_parser(
    name: str,
    max_pages: int | None,
    vision_model: str | None = None,
    profile: DomainProfile | None = None,
) -> PDFParser:
    """Select the PDFParser impl by name. Lazy imports keep heavy deps
    (anthropic / openai SDKs, pypdfium2 / sentence-transformers) out of
    cold-start when the user picks a different backend.

    Backends:
      - pypdf            : text-only, free, no API needed; useless on scanned PDFs
      - claude-vision    : Anthropic SDK Vision (per-token, needs ANTHROPIC_API_KEY)
      - claude-cli-vision: claude CLI subprocess + Read tool (subscription quota,
                           needs `claude login`, $0 marginal for Max/Pro users)
      - deepseek-v4      : DeepSeek V4 Vision via OpenAI-compatible API
                           (per-token, needs DEEPSEEK_API_KEY; ~3-5x cheaper
                           per page than claude-vision on flash variant)
      - rapidocr         : local CPU OCR (PP-OCRv5 via ONNX Runtime) — $0,
                           no API key, fully offline; lower fidelity than
                           the LLM vision vendors (E3, ssLC)

    E10: ``vision_model`` overrides the OCR model for the two Claude
    vision parsers (e.g. "sonnet" for higher fidelity than the default
    "haiku" on claude-cli-vision). None keeps each parser's own default.
    Ignored for pypdf (no model) and deepseek-v4 (different model vocab —
    pick the variant via the deepseek parser default; FORESHADOW E10.1).
    """
    if name == "pypdf":
        return PyPdfParser()
    if name == "claude-vision":
        from jcontract.impls.claude_vision_parser import ClaudeVisionParser

        # Pass model= only when overridden so the impl's own DEFAULT_MODEL
        # (and the legacy un-suffixed cache namespace) stays in effect.
        if vision_model:
            return ClaudeVisionParser(max_pages=max_pages, model=vision_model, profile=profile)
        return ClaudeVisionParser(max_pages=max_pages, profile=profile)
    if name == "claude-cli-vision":
        from jcontract.impls.claude_cli_vision_parser import ClaudeCliVisionParser

        if vision_model:
            return ClaudeCliVisionParser(max_pages=max_pages, model=vision_model, profile=profile)
        return ClaudeCliVisionParser(max_pages=max_pages, profile=profile)
    if name == "deepseek-v4":
        from jcontract.impls.deepseek_v4_parser import DeepSeekV4Parser

        return DeepSeekV4Parser(max_pages=max_pages, profile=profile)
    if name == "rapidocr":
        from jcontract.impls.rapidocr_parser import RapidOcrParser

        # profile / vision_model intentionally not forwarded: RapidOCR takes
        # no prompt — its output depends only on pixels + ONNX weights, so a
        # profile cannot change it and must not fork its cache namespace.
        return RapidOcrParser(max_pages=max_pages)
    raise typer.BadParameter(
        f"Unknown parser '{name}'. "
        f"Choose from: pypdf, claude-vision, claude-cli-vision, deepseek-v4, rapidocr."
    )


@app.command()
def ingest(
    pdf_path: Annotated[Path, typer.Argument(exists=True, readable=True)],
    collection: Annotated[str, typer.Option(help="Qdrant collection name")] = "contract",
    domain: Annotated[
        str,
        typer.Option(
            help=(
                "DomainProfile name (profiles/<name>.yaml): contract (construction, default) "
                "| document (neutral, any PDF). Drives OCR/caption/answer prompts + "
                "chunking structure. Persisted as the collection's profile."
            ),
        ),
    ] = "contract",
    parser: Annotated[
        str,
        typer.Option(
            help=(
                "PDF parser: pypdf | claude-vision (API key) | "
                "claude-cli-vision (subscription) | deepseek-v4 (API key, cheapest) | "
                "rapidocr (local CPU, offline, free)"
            ),
        ),
    ] = "pypdf",
    max_pages: Annotated[
        int | None,
        typer.Option(
            help=(
                "Limit OCR to first N pages (cost control for any vision parser). None = all pages."
            ),
        ),
    ] = None,
    vision_model: Annotated[
        str | None,
        typer.Option(
            help=(
                "OCR model for claude-vision / claude-cli-vision parsers "
                "(e.g. 'sonnet' for higher text fidelity than the default 'haiku'). "
                "A non-default model re-OCRs into its own cache namespace. "
                "Ignored for pypdf / deepseek-v4 / rapidocr."
            ),
        ),
    ] = None,
    caption: Annotated[
        bool,
        typer.Option(
            "--caption/--no-caption",
            help=(
                "Caption drawing-type chunks (adds Chinese descriptions to the "
                "retrieval index). Backend chosen by --caption-backend."
            ),
        ),
    ] = False,
    caption_backend: Annotated[
        str,
        typer.Option(
            help=(
                "Caption backend: claude-cli (subscription, no key — default) | "
                "claude-api (ANTHROPIC_API_KEY) | deepseek (DEEPSEEK_API_KEY) | "
                "ollama (local VLM, no key)."
            ),
        ),
    ] = "claude-cli",
) -> None:
    """Parse, chunk, embed, and index one PDF into Qdrant + BM25 + RefGraph."""
    # Phase 7: the DomainProfile drives prompts + chunking structure, and is
    # persisted as this collection's profile (read back by search/eval/ask).
    profile = load_profile(domain)
    cp = paths_for(collection)
    cp.root.mkdir(parents=True, exist_ok=True)
    write_profile_sidecar(collection, domain)
    stack = _build_stack(collection)
    ref_graph = SqliteRefGraph(db_path=cp.ref_graph)
    # Captioner is opt-in: it spends quota/$$ per drawing page and pulls in
    # heavy lazy paths. _build_captioner picks the vendor (E11); the default
    # claude-cli needs no API key, matching the no-key project default.
    captioner = None
    if caption:
        captioner = _build_captioner(caption_backend, profile=profile)
        typer.echo(
            f"Captioner enabled (backend={caption_backend}); "
            "drawing chunks will get Chinese captions."
        )
    try:
        pipeline = IngestPipeline(
            parser=_build_parser(parser, max_pages, vision_model, profile=profile),
            chunker=QaAwareChunker(profile.structure),
            embedder=stack.embedder,
            vector_store=stack.vector_store,
            keyword_index=stack.keyword_index,
            chunks_snapshot_path=cp.chunks_snapshot,
            ref_graph=ref_graph,
            captioner=captioner,
        )
        n = pipeline.ingest(pdf_path)
        typer.echo(f"Indexed {n} chunks from {pdf_path.name} (parser={parser}, domain={domain})")
        typer.echo(
            f"Qdrant collection '{collection}' now holds {stack.vector_store.count()} points."
        )
        typer.echo(f"RefGraph stats: {ref_graph.stats()}")
    finally:
        ref_graph.close()


@app.command()
def search(
    query: Annotated[str, typer.Argument()],
    k: Annotated[int, typer.Option(help="Number of results to return")] = 5,
    collection: Annotated[str, typer.Option()] = "contract",
    rerank: Annotated[
        bool,
        typer.Option(
            "--rerank/--no-rerank",
            help="Apply BGE cross-encoder reranker after RRF fusion (heavier; better at scale).",
        ),
    ] = False,
) -> None:
    """Hybrid (vector + BM25 ± reranker) search; prints top-k chunks with file + page."""
    stack = _build_stack(collection, use_reranker=rerank)
    results = stack.retriever.search(query, k=k)

    if not results:
        typer.echo("No results. Did you `jcontract ingest <pdf>` first?")
        raise typer.Exit(code=1)

    for i, r in enumerate(results, start=1):
        c = r.chunk
        preview = c.text.strip().replace("\n", " ")[:160]
        typer.echo(f"\n[{i}] score={r.score:.4f} file={c.file} p.{c.page} type={c.chunk_type}")
        if c.question_no:
            typer.echo(f"     Q.No: {c.question_no}")
        if c.drawing_refs:
            typer.echo(f"     drawings: {c.drawing_refs}")
        if c.clause_refs:
            typer.echo(f"     clauses:  {c.clause_refs}")
        typer.echo(f"     preview: {preview}")


@app.command("evaluate")
def evaluate(
    collection: Annotated[str, typer.Option()] = "contract",
    golden_path: Annotated[Path, typer.Option()] = GOLDEN_CASES_PATH,
    enable_answer: Annotated[
        bool,
        typer.Option(help="If set + answerer backend available, also run the answerer."),
    ] = True,
    answerer: Annotated[
        str,
        typer.Option(
            help=(
                "Answerer backend: claude-api (per-token via API key), "
                "claude-cli (Claude Code subscription quota), "
                "codex-cli (ChatGPT subscription quota), or "
                "local (OpenAI-compatible endpoint, default local Ollama)."
            ),
        ),
    ] = "claude-api",
    rerank: Annotated[
        bool,
        typer.Option("--rerank/--no-rerank", help="Apply BGE cross-encoder reranker."),
    ] = False,
    judge: Annotated[
        str | None,
        typer.Option(
            help=(
                "LLM-as-judge backend for answer-quality metrics (faithfulness + "
                "answer_relevancy): claude-cli (subscription, no key). None = off. "
                "Needs an answerer (--enable-answer)."
            ),
        ),
    ] = None,
) -> None:
    """Run the eval pipeline against golden cases; write timestamped JSON to data/eval-results/."""
    stack = _build_stack(collection, use_reranker=rerank)
    # The answerer's framing comes from this collection's bound DomainProfile.
    eval_profile = load_profile(read_profile_name(collection))

    # Load golden cases.
    cases: list[EvalCase] = []
    with golden_path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            cases.append(EvalCase.from_dict(json.loads(line)))
    typer.echo(f"Loaded {len(cases)} golden cases from {golden_path}")

    # Inject the real search + (maybe) answer functions.
    def search_fn(q: str) -> list[SearchResult]:
        return stack.retriever.search(q, k=10)

    answer_fn = None
    if enable_answer:
        try:
            answerer_impl = _maybe_build_answerer(
                answerer, domain_framing=eval_profile.answer_framing
            )
        except RuntimeError as exc:
            # Missing CLI binary → graceful degrade with a clear message.
            answerer_impl = None
            typer.echo(f"Answerer '{answerer}' unavailable: {exc}")

        if answerer_impl is not None:

            def answer_fn(q: str, results: list[SearchResult]) -> Answer:
                # answerer.answer takes (question, list[Chunk])
                return answerer_impl.answer(q, [r.chunk for r in results[:5]])

            typer.echo(
                f"Answerer wired ({answerer}). Will measure citation_accuracy + keyword_hit_rate."
            )
        elif answerer == "claude-api":
            typer.echo("ANTHROPIC_API_KEY not set; running retrieval-only eval.")

    # Optional LLM-as-judge (E12). Only meaningful when an answerer is wired
    # (the judge grades the answer); build it defensively so a missing CLI
    # binary degrades gracefully instead of aborting the run.
    judge_impl: Judge | None = None
    if judge is not None:
        if answer_fn is None:
            typer.echo("--judge given but no answerer available; skipping judge metrics.")
        else:
            try:
                judge_impl = _build_judge(judge)
                typer.echo(f"Judge wired ({judge}). Will measure faithfulness + answer_relevancy.")
            except RuntimeError as exc:
                typer.echo(f"Judge '{judge}' unavailable: {exc}")

    eval_results_dir = paths_for(collection).eval_results
    summary = run_eval(
        cases=cases,
        search_fn=search_fn,
        answer_fn=answer_fn,
        output_dir=eval_results_dir,
        judge=judge_impl,
    )

    typer.echo("\n=== Eval Summary ===")
    typer.echo(f"  n_cases: {summary['n_cases']}")
    typer.echo("  metrics_mean:")
    for k, v in summary["metrics_mean"].items():
        typer.echo(f"    {k}: {v:.4f}" if isinstance(v, float) else f"    {k}: {v}")

    # Per-category recall — lets you compare e.g. the `drawing` cases with
    # --caption on vs off (Enhancement E2). Recall keys only, to stay compact.
    by_cat = summary.get("metrics_by_category", {})
    if by_cat:
        typer.echo("  recall_at_5 by category:")
        for cat in sorted(by_cat):
            m = by_cat[cat]
            r5 = m.get("recall_at_5")
            n = int(m.get("n_cases", 0))
            r5_str = f"{r5:.4f}" if isinstance(r5, float) else "n/a"
            typer.echo(f"    {cat} (n={n}): {r5_str}")

    typer.echo(f"\nResults written to {eval_results_dir}/")


@app.command("eval-compare")
def eval_compare(
    report_a: Annotated[Path, typer.Argument(exists=True, readable=True, help="Baseline report")],
    report_b: Annotated[Path, typer.Argument(exists=True, readable=True, help="Candidate report")],
) -> None:
    """Diff two eval JSON reports (A=baseline, B=candidate) — Enhancement E12.

    Use to answer A/B questions: did --caption raise drawing recall (E2 ROI)?
    did --vision-model sonnet beat haiku? delta = B - A (positive = better).

    Example:
      jcontract eval-compare data/eval-results/<no-caption>.json \\
                             data/eval-results/<caption>.json
    """
    with report_a.open(encoding="utf-8") as fh:
        a = json.load(fh)
    with report_b.open(encoding="utf-8") as fh:
        b = json.load(fh)

    diff = compare_reports(a, b)

    def _fmt(cell: float | None) -> str:
        return f"{cell:.4f}" if isinstance(cell, float) else "  n/a "

    def _delta(d: float | None) -> str:
        if d is None:
            return ""
        arrow = " ↑" if d > 0 else (" ↓" if d < 0 else "  ")
        return f"{d:+.4f}{arrow}"

    def _print_block(block: dict[str, dict[str, float | None]]) -> None:
        for metric in block:
            row = block[metric]
            typer.echo(
                f"    {metric:<20} A={_fmt(row['a'])}  B={_fmt(row['b'])}  Δ={_delta(row['delta'])}"
            )

    typer.echo(f"=== eval-compare ===\n  A (baseline):  {report_a}\n  B (candidate): {report_b}")
    typer.echo("\n  metrics_mean:")
    _print_block(diff["metrics_mean"])

    by_cat = diff["metrics_by_category"]
    if by_cat:
        typer.echo("\n  metrics_by_category:")
        for cat in by_cat:
            typer.echo(f"  [{cat}]")
            _print_block(by_cat[cat])


@app.command("batch-ingest")
def batch_ingest(
    pdf_paths: Annotated[list[Path], typer.Argument(exists=True, readable=True)],
    collection: Annotated[str, typer.Option()] = "contract",
    domain: Annotated[
        str,
        typer.Option(
            help="DomainProfile name (contract | document | ...); persisted for collection."
        ),
    ] = "contract",
    parser: Annotated[
        str,
        typer.Option(
            help=(
                "OCR backend: claude-vision (API key) | claude-cli-vision (subscription) "
                "| deepseek-v4 (API key, cheapest)"
            ),
        ),
    ] = "claude-vision",
    vision_model: Annotated[
        str | None,
        typer.Option(
            help=(
                "OCR model for claude-vision / claude-cli-vision (e.g. 'sonnet'). "
                "Non-default re-OCRs into its own cache namespace. Ignored for deepseek-v4."
            ),
        ),
    ] = None,
    max_concurrent: Annotated[
        int,
        typer.Option(help="Concurrent OCR calls (rate-limit dependent)."),
    ] = 4,
    max_budget_usd: Annotated[
        float | None,
        typer.Option(help="Abort when cumulative metered cost crosses this. None = unlimited."),
    ] = None,
    resume: Annotated[
        bool,
        typer.Option("--resume/--no-resume", help="Skip pages recorded as done in checkpoint."),
    ] = True,
    estimated_cost_per_page: Annotated[
        float,
        typer.Option(help="Metered $/page estimate for budget guard. Sonnet≈0.015, Haiku≈0.003."),
    ] = 0.015,
) -> None:
    """Concurrent + resumable + budget-guarded multi-PDF OCR ingest (Phase 1.7).

    Pipeline:
      Phase A (slow, asyncio.Semaphore-bounded): OCR every page across all
        PDFs concurrently. Checkpoint after each page. Cached pages cost
        nothing — re-running is safe.
      Phase B (fast, sequential): chunk + embed + index each PDF using
        the cached OCR text. RefGraph is updated as part of Phase B.
    """
    import asyncio  # local — keeps cold-start lean

    import pypdfium2 as pdfium

    # One parser shared across the batch — its OCR cache is content-addressed
    # so it's safe across multiple PDFs AND across parser backends (each vendor
    # uses its own cache filename prefix to avoid cross-vendor pollution).
    from jcontract.impls._pdfium_render import render_pdf_page_jpeg
    from jcontract.impls.claude_cli_vision_parser import ClaudeCliVisionParser
    from jcontract.impls.claude_vision_parser import ClaudeVisionParser
    from jcontract.impls.deepseek_v4_parser import DeepSeekV4Parser
    from jcontract.ingest.batch import BatchIngest, BudgetExceededError

    # Phase 7: profile drives prompts + chunking + is persisted for the collection.
    profile = load_profile(domain)
    write_profile_sidecar(collection, domain)

    parser_impl: ClaudeVisionParser | ClaudeCliVisionParser | DeepSeekV4Parser
    # E10: forward --vision-model to the Claude vision parsers only.
    if parser == "claude-vision":
        parser_impl = (
            ClaudeVisionParser(model=vision_model, profile=profile)
            if vision_model
            else ClaudeVisionParser(profile=profile)
        )
    elif parser == "claude-cli-vision":
        parser_impl = (
            ClaudeCliVisionParser(model=vision_model, profile=profile)
            if vision_model
            else ClaudeCliVisionParser(profile=profile)
        )
    elif parser == "deepseek-v4":
        parser_impl = DeepSeekV4Parser(profile=profile)
    else:
        raise typer.BadParameter(
            f"batch-ingest only supports OCR backends. "
            f"Got '{parser}', expected claude-vision, claude-cli-vision, or deepseek-v4."
        )

    # Enumerate page counts cheaply (open + len, no rendering).
    page_counts: dict[Path, int] = {}
    for pdf_path in pdf_paths:
        pdf = pdfium.PdfDocument(str(pdf_path))
        try:
            page_counts[pdf_path] = len(pdf)
        finally:
            pdf.close()

    total_pages = sum(page_counts.values())
    typer.echo(
        f"Batch ingest: {len(pdf_paths)} PDFs, {total_pages} pages total, "
        f"concurrent={max_concurrent}, budget={max_budget_usd or 'unlimited'} USD"
    )

    # Per-page worker for the batch orchestrator. asyncio.to_thread keeps
    # the sync render + API call off the event loop without blocking other
    # workers. Render goes through `render_pdf_page_jpeg` — the only
    # thread-safe pdfium entry point (open→render→close inside the global
    # pdfium lock) — then `_ocr_jpeg` (cache check + vendor call, no
    # pdfium) runs concurrently. [DECISION-ab3.46]
    async def ocr_one_page(pdf_path: Path, page_num: int) -> tuple[str, float]:
        def sync_ocr() -> tuple[str, float]:
            jpeg = render_pdf_page_jpeg(
                pdf_path,
                page_num,
                dpi=parser_impl._dpi,
                jpeg_quality=parser_impl._jpeg_quality,
            )
            text = parser_impl._ocr_jpeg(jpeg, page_num, pdf_path.name)
            # Conservative cost estimate — overcounts cached pages but
            # that's safer for a budget guard. Actual subscription users
            # see this as informational; API-key users get real billing.
            cost = estimated_cost_per_page if text else 0.0
            return text, cost

        return await asyncio.to_thread(sync_ocr)

    # Per-collection artifacts (Phase 7 SS5).
    cp = paths_for(collection)
    cp.root.mkdir(parents=True, exist_ok=True)
    # If --no-resume, write to a fresh checkpoint path so we don't clobber
    # the existing one (in case the user wants to come back to it).
    checkpoint_path = (
        cp.ingest_checkpoint
        if resume
        else cp.ingest_checkpoint.with_name(cp.ingest_checkpoint.stem + ".fresh.jsonl")
    )

    batch = BatchIngest(
        pdf_paths=list(pdf_paths),
        page_counts=page_counts,
        checkpoint_path=checkpoint_path,
        max_concurrent=max_concurrent,
        max_budget_usd=max_budget_usd,
    )

    try:
        summary = asyncio.run(batch.run(ocr_one_page_fn=ocr_one_page))
    except BudgetExceededError as exc:
        typer.echo(f"\nBudget exceeded: {exc}", err=True)
        typer.echo("Checkpoint preserved — re-run to resume.", err=True)
        raise typer.Exit(code=2) from exc

    typer.echo(
        f"\nPhase A done: {summary.pages_done} OCR'd, "
        f"{summary.pages_skipped} skipped (cache/checkpoint), "
        f"{summary.pages_error} errors, "
        f"${summary.total_cost_usd:.2f} estimated, "
        f"{summary.elapsed_s:.1f}s"
    )

    # Phase B: chunk + embed + index each PDF. The parser re-renders each
    # page (~50ms) and hits cache for the OCR text → no API calls.
    stack = _build_stack(collection)
    ref_graph = SqliteRefGraph(db_path=cp.ref_graph)
    try:
        for pdf_path in pdf_paths:
            pipeline = IngestPipeline(
                parser=parser_impl,
                chunker=QaAwareChunker(profile.structure),
                embedder=stack.embedder,
                vector_store=stack.vector_store,
                keyword_index=stack.keyword_index,
                chunks_snapshot_path=cp.chunks_snapshot,
                ref_graph=ref_graph,
            )
            n = pipeline.ingest(pdf_path)
            typer.echo(f"  {pdf_path.name}: {n} chunks indexed")
        typer.echo(
            f"\nFinal: Qdrant collection '{collection}' = "
            f"{stack.vector_store.count()} points; RefGraph = {ref_graph.stats()}"
        )
    finally:
        ref_graph.close()


@app.command()
def refs(
    entity_type: Annotated[
        str,
        typer.Argument(
            help="Entity type: drawing | clause | question_no | section | revision",
        ),
    ],
    entity_value: Annotated[str, typer.Argument(help="Exact value to look up.")],
    collection: Annotated[str, typer.Option()] = "contract",
) -> None:
    """Query the RefGraph for all chunks that mention this entity.

    Examples:
      jcontract refs drawing T/PRJ/CWD/WS/2101A
      jcontract refs question_no ACME/TRACKWORK/16
      jcontract refs clause 7.3
    """
    ref_graph_path = paths_for(collection).ref_graph
    if not ref_graph_path.exists():
        typer.echo(f"No RefGraph DB at {ref_graph_path}. Run `jcontract ingest` first.")
        raise typer.Exit(code=1)

    ref_graph = SqliteRefGraph(db_path=ref_graph_path)
    try:
        chunks = ref_graph.mentions_of(entity_type, entity_value)
        if not chunks:
            typer.echo(f"No mentions found for {entity_type}={entity_value!r}.")
            typer.echo(f"RefGraph stats: {ref_graph.stats()}")
            return

        typer.echo(f"Found {len(chunks)} mention(s) of {entity_type}={entity_value!r}:\n")
        for c in chunks:
            typer.echo(f"  {c.file}  p.{c.page}  ({c.chunk_type})  id={c.id}")
    finally:
        ref_graph.close()


@app.command("migrate-layout")
def migrate_layout(
    collection: Annotated[
        str, typer.Option(help="Collection to receive the legacy flat files.")
    ] = "contract",
    apply: Annotated[
        bool,
        typer.Option("--apply/--dry-run", help="Actually move files (default: dry-run preview)."),
    ] = False,
) -> None:
    """Move pre-Phase-7 flat data/ files into data/<collection>/ (Phase 7 SS5).

    Relocates chunks_snapshot.jsonl / ref_graph.db / ingest_checkpoint.jsonl /
    eval-results into the per-collection subtree so the existing index isn't
    orphaned by the new layout. Dry-run by default; pass --apply. Idempotent.
    """
    moves = plan_layout_migration(collection)
    if not moves:
        typer.echo("No legacy data/ files found — nothing to migrate.")
        return
    typer.echo(f"{'APPLYING' if apply else 'DRY-RUN'} — move legacy files into data/{collection}/:")
    for src, dst in moves:
        typer.echo(f"  {src}  ->  {dst}")
    if not apply:
        typer.echo("\nRe-run with --apply to perform the move.")
        return
    done = apply_layout_migration(collection)
    typer.echo(f"\nMoved {len(done)} item(s) into data/{collection}/.")


@app.command()
def show_chunks(
    n: Annotated[int, typer.Option(help="Show first N chunks from the snapshot")] = 10,
    collection: Annotated[str, typer.Option()] = "contract",
) -> None:
    """Debug: dump the first N chunks from the snapshot file."""
    chunks: list[Chunk] = load_chunks_snapshot(paths_for(collection).chunks_snapshot)
    typer.echo(f"Total snapshot chunks: {len(chunks)}")
    for c in chunks[:n]:
        typer.echo(f"  {c.id}  type={c.chunk_type}  p.{c.page}  qno={c.question_no}")
        typer.echo(f"    text[:80]: {c.text[:80]!r}")
    if not chunks:
        typer.echo("(empty — run `jcontract ingest <pdf>` first)")


# ssQA: signals the `ocr-quality` report exposes for --flag-below/--flag-above.
# The first five are the PRE-REGISTERED candidate list (dev-sprint v5
# §预注册评测协议 3, frozen — do not add/remove); garbled_ratio is the
# additional garbled-text heuristic the same protocol carries alongside them.
_QUALITY_SIGNALS = (
    "mean_score",
    "min_score",
    "low_score_ratio",
    "boxes",
    "non_alnum_ratio",
    "garbled_ratio",
)


def _parse_flag_rules(specs: list[str], option_name: str) -> list[tuple[str, float]]:
    """Parse repeated ``<signal>:<value>`` flag specs into (signal, threshold).

    Unknown signal names and non-numeric thresholds fail fast as usage
    errors — a typo'd signal silently flagging nothing would corrupt the
    L5 calibration downstream.
    """
    rules: list[tuple[str, float]] = []
    for spec in specs:
        signal, sep, raw_value = spec.partition(":")
        if not sep or signal not in _QUALITY_SIGNALS:
            raise typer.BadParameter(
                f"{option_name} expects <signal>:<value> with signal one of "
                f"{', '.join(_QUALITY_SIGNALS)}; got {spec!r}"
            )
        try:
            rules.append((signal, float(raw_value)))
        except ValueError as exc:
            raise typer.BadParameter(
                f"{option_name} threshold must be numeric; got {spec!r}"
            ) from exc
    return rules


def _quality_report_record(metrics: dict[str, object]) -> dict[str, object]:
    """Project a metrics sidecar dict onto the per-page JSONL report record.

    Emits the five pre-registered signals + garbled_ratio. ``non_alnum_ratio``
    (the registered "非字母数字占比" signal) derives from the sidecar's stored
    ``alnum_ratio`` as 1 - alnum_ratio; the raw per-box score list stays in
    the sidecar (lossless record) and out of the report (readable record).
    """

    def _round4(value: object) -> float | None:
        return round(value, 4) if isinstance(value, int | float) else None

    alnum_ratio = metrics.get("alnum_ratio")
    record: dict[str, object] = {
        "page_num": metrics["page_num"],
        "boxes": metrics["boxes"],
        "mean_score": _round4(metrics.get("mean_score")),
        "min_score": _round4(metrics.get("min_score")),
        "low_score_ratio": _round4(metrics.get("low_score_ratio")),
        "non_alnum_ratio": (
            round(1.0 - alnum_ratio, 4) if isinstance(alnum_ratio, int | float) else None
        ),
        "garbled_ratio": _round4(metrics.get("garbled_ratio")),
    }
    if "engine_error" in metrics:
        record["engine_error"] = metrics["engine_error"]
    return record


def _flag_reasons(
    record: dict[str, object],
    below_rules: list[tuple[str, float]],
    above_rules: list[tuple[str, float]],
) -> list[str]:
    """Evaluate caller-supplied threshold rules against one report record.

    A null signal (undefined — e.g. mean_score on a zero-box page) never
    triggers a rule: there is no evidence to compare. Zero-box pages remain
    catchable via the always-defined ``boxes`` signal. [DECISION-cq.20]
    """
    reasons: list[str] = []
    for signal, threshold in below_rules:
        value = record.get(signal)
        if isinstance(value, int | float) and value < threshold:
            reasons.append(f"{signal}={value:g}<{threshold:g}")
    for signal, threshold in above_rules:
        value = record.get(signal)
        if isinstance(value, int | float) and value > threshold:
            reasons.append(f"{signal}={value:g}>{threshold:g}")
    return reasons


@app.command("ocr-quality")
def ocr_quality(
    pdf_path: Annotated[Path, typer.Argument(exists=True, readable=True)],
    max_pages: Annotated[
        int | None,
        typer.Option(help="Scan only the first N pages. None = all pages."),
    ] = None,
    out: Annotated[
        Path | None,
        typer.Option(
            help=(
                "Write the per-page JSONL report here. Without --out the JSONL lines go to stdout."
            ),
        ),
    ] = None,
    flag_below: Annotated[
        list[str] | None,
        typer.Option(
            "--flag-below",
            help=(
                "Flag pages where <signal> is BELOW <value>, e.g. "
                "--flag-below mean_score:0.85 (repeatable; rules OR together). "
                f"Signals: {', '.join(_QUALITY_SIGNALS)}."
            ),
        ),
    ] = None,
    flag_above: Annotated[
        list[str] | None,
        typer.Option(
            "--flag-above",
            help=(
                "Flag pages where <signal> is ABOVE <value>, e.g. "
                "--flag-above garbled_ratio:0.2 — for the higher-is-worse "
                "signals that --flag-below cannot express (repeatable)."
            ),
        ),
    ] = None,
) -> None:
    """Per-page OCR quality report for the rapidocr lane (ssQA).

    Emits one JSONL record per page with the five pre-registered quality
    signals (mean_score, min_score, low_score_ratio, boxes, non_alnum_ratio)
    plus the garbled-text heuristic (garbled_ratio), and a terminal summary.

    Locate + mark ONLY: this command ships NO built-in thresholds — pass
    them via --flag-below/--flag-above once the L5 calibration has derived
    them (mechanism/policy separation, DECISION-cq.20) — and performs no
    routing/re-OCR of flagged pages (FORESHADOW-cq.1 pending).

    Reads the metrics sidecar when present; otherwise force-runs the local
    OCR engine (~1s/page CPU, even on .txt cache hits) and backfills the
    sidecar so the next scan is free. Fully offline, zero quota.
    """
    # Lazy import — keeps opencv/onnxruntime out of cold-start for other
    # commands (same stance as _build_parser).
    from jcontract.impls.rapidocr_parser import RapidOcrParser

    below_rules = _parse_flag_rules(flag_below or [], "--flag-below")
    above_rules = _parse_flag_rules(flag_above or [], "--flag-above")

    parser = RapidOcrParser(max_pages=max_pages)
    pages = parser.quality_metrics(pdf_path)

    records: list[dict[str, object]] = []
    for metrics in pages:
        record = _quality_report_record(metrics)
        reasons = _flag_reasons(record, below_rules, above_rules)
        record["flagged"] = bool(reasons)
        record["flag_reasons"] = reasons
        records.append(record)

    jsonl_lines = [json.dumps(r, ensure_ascii=False) for r in records]
    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("\n".join(jsonl_lines) + "\n", encoding="utf-8")
    else:
        for line in jsonl_lines:
            typer.echo(line)

    # Terminal summary table: per-signal aggregates over the pages where the
    # signal is defined (nulls excluded), then the flagged-page roll-up.
    typer.echo(f"\n=== ocr-quality: {pdf_path.name} ({len(records)} pages) ===")
    typer.echo(f"  {'signal':<16} {'mean':>8} {'min':>8} {'max':>8} {'n_def':>6}")
    for signal in _QUALITY_SIGNALS:
        values = [v for r in records if isinstance(v := r.get(signal), int | float)]
        if values:
            mean_v = sum(values) / len(values)
            typer.echo(
                f"  {signal:<16} {mean_v:>8.4f} {min(values):>8.4f} "
                f"{max(values):>8.4f} {len(values):>6}"
            )
        else:
            typer.echo(f"  {signal:<16} {'n/a':>8} {'n/a':>8} {'n/a':>8} {0:>6}")

    errored = [r["page_num"] for r in records if "engine_error" in r]
    if errored:
        typer.echo(f"  engine errors: {len(errored)} page(s) -> {errored}")

    if below_rules or above_rules:
        rule_text = ", ".join(
            [f"{s}<{t:g}" for s, t in below_rules] + [f"{s}>{t:g}" for s, t in above_rules]
        )
        flagged_pages = [r["page_num"] for r in records if r["flagged"]]
        typer.echo(f"  flag rules: {rule_text}")
        typer.echo(f"  flagged: {len(flagged_pages)}/{len(records)} page(s) -> {flagged_pages}")
    else:
        typer.echo(
            "  flag rules: none supplied — no pages flagged (pass --flag-below/--flag-above)"
        )

    if out is not None:
        typer.echo(f"  JSONL report: {out}")


@app.command("redact-preview")
def redact_preview(
    text_file: Annotated[Path, typer.Argument(exists=True, readable=True)],
    restore: Annotated[
        bool,
        typer.Option(
            "--restore",
            help="Reverse direction: replace known <TYPE_N> placeholders with the originals.",
        ),
    ] = False,
    dictionary: Annotated[
        Path | None,
        typer.Option(
            envvar="JCONTRACT_REDACTION_DICT",
            help=(
                "Redaction dictionary YAML (entities + regex patterns). Lives in your "
                "project data dir, NOT in this repo."
            ),
        ),
    ] = None,
    map_store: Annotated[
        Path | None,
        typer.Option(
            envvar="JCONTRACT_REDACTION_MAP",
            help=(
                "Persistent entity->placeholder mapping store (JSONL, created on first "
                "use). This file is the RESTORE KEY — keep it out of git and logs."
            ),
        ),
    ] = None,
    out: Annotated[
        Path | None,
        typer.Option(help="Write the result here instead of stdout."),
    ] = None,
) -> None:
    """Reversible-pseudonymization demo over one text file (ssDI).

    Mechanism only — NOT wired into ingest/answer (DECISION-cq.4): this is a
    local, offline preview of the Redactor component. Placeholders are
    corpus-stable: the same entity gets the same <TYPE_N> across files and
    sessions because numbering persists in the mapping store.

    The result text goes to stdout verbatim (no trailing newline added), so
    `redact-preview f | redact-preview /dev/stdin --restore | diff - f`
    style byte-exact checks work; the one-line summary goes to stderr.
    Summary and errors never include entity names (21-security).
    """
    from jcontract.impls.dict_regex_redactor import DictRegexRedactor

    if dictionary is None or map_store is None:
        raise typer.BadParameter(
            "a dictionary and a mapping store are required: pass --dictionary/--map-store "
            "or set JCONTRACT_REDACTION_DICT / JCONTRACT_REDACTION_MAP"
        )
    if not dictionary.exists():
        raise typer.BadParameter(f"dictionary not found: {dictionary}")

    redactor = DictRegexRedactor(dictionary_path=dictionary, store_path=map_store)
    text = text_file.read_text(encoding="utf-8")
    if restore:
        result_text = redactor.restore(text)
        summary = f"restored placeholders in {len(text)} chars"
    else:
        result = redactor.redact(text)
        result_text = result.redacted_text
        summary = (
            f"replaced {result.spans_replaced} span(s), "
            f"{result.mapping_delta} new mapping entrie(s) persisted"
        )
    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(result_text, encoding="utf-8")
    else:
        typer.echo(result_text, nl=False)
    typer.echo(f"redact-preview: {summary} (store: {map_store})", err=True)


@app.command("dispatch-plan")
def dispatch_plan(
    pdf_path: Annotated[Path, typer.Argument(exists=True, readable=True)],
    pool: Annotated[
        str | None,
        typer.Option(
            envvar="JCONTRACT_DISPATCH_POOL",
            help=(
                "Comma-separated provider names, e.g. 'claude,openai'. No default on "
                "purpose — the pool is YOUR routing config, not this tool's opinion."
            ),
        ),
    ] = None,
    max_pages: Annotated[
        int | None,
        typer.Option(help="Plan only the first N pages. None = all pages."),
    ] = None,
    out: Annotated[
        Path | None,
        typer.Option(
            help="Write the per-page plan JSONL here. Without --out the JSONL lines go to stdout."
        ),
    ] = None,
    provenance: Annotated[
        Path | None,
        typer.Option(
            help=(
                "Append assignments to this provenance JSONL audit log (idempotent: "
                "re-running the same plan against the same pool appends nothing)."
            ),
        ),
    ] = None,
    task_kind: Annotated[
        str,
        typer.Option(help="Task label recorded per assignment (plan + provenance)."),
    ] = "page-ocr",
) -> None:
    """Dry-run page→provider dispatch plan — deterministic, zero network (ssMP).

    Renders each page (same 150dpi/q85 geometry as the vision parsers, so
    the sha256 content hash lands in the SAME namespace as the OCR cache
    keys), assigns it a provider name via the deterministic hash lottery
    (sha256 % pool size, DECISION-cq.40), and emits the per-page table.
    Same PDF + same pool → byte-identical output, every run.

    Mechanism only (DECISION-cq.4): pool entries are opaque NAMES — no
    vendor module is imported, no client constructed, no network touched
    (asserted by tests/test_dispatch.py). Not wired into ingest; routing
    of flagged pages stays pending (FORESHADOW-cq.1/cq.2).

    Deterministic plan goes to stdout/--out; run-status (provenance append
    counts) goes to stderr so double-run diffs stay empty.
    """
    import hashlib

    import pypdfium2 as pdfium

    from jcontract.impls._pdfium_render import render_page_jpeg

    # Same render geometry as the vision parsers (rapidocr_parser module
    # import is light — the OCR engine itself loads lazily elsewhere).
    from jcontract.impls.rapidocr_parser import DEFAULT_DPI, DEFAULT_JPEG_QUALITY
    from jcontract.ingest.dispatch import ProvenanceLog, ProviderDispatcher

    if pool is None:
        raise typer.BadParameter(
            "a provider pool is required: pass --pool a,b or set JCONTRACT_DISPATCH_POOL"
        )
    try:
        dispatcher = ProviderDispatcher([name.strip() for name in pool.split(",")])
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc

    # Sequential render loop owning the document lifecycle — the
    # render_page_jpeg contract (DECISION-ab3.46), same as parser.parse.
    plan: list[dict[str, object]] = []
    pdf = pdfium.PdfDocument(str(pdf_path))
    try:
        n_pages = len(pdf) if max_pages is None else min(max_pages, len(pdf))
        for page_idx in range(n_pages):
            jpeg_bytes = render_page_jpeg(
                pdf[page_idx], dpi=DEFAULT_DPI, jpeg_quality=DEFAULT_JPEG_QUALITY
            )
            content_hash = hashlib.sha256(jpeg_bytes).hexdigest()
            plan.append(
                {
                    "page_num": page_idx + 1,
                    "content_hash": content_hash,
                    "provider": dispatcher.assign(content_hash),
                    "task_kind": task_kind,
                }
            )
    finally:
        pdf.close()

    jsonl_lines = [json.dumps(r, ensure_ascii=False) for r in plan]
    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("\n".join(jsonl_lines) + "\n", encoding="utf-8")
    else:
        for line in jsonl_lines:
            typer.echo(line)

    typer.echo(f"\n=== dispatch-plan (dry-run): {pdf_path.name} ({len(plan)} pages) ===")
    typer.echo(f"  pool: {dispatcher.pool}  task_kind: {task_kind}")
    typer.echo(f"  {'page':>5}  {'content_hash':<16}  provider")
    for record in plan:
        hash_prefix = str(record["content_hash"])[:12] + "…"
        typer.echo(f"  {record['page_num']:>5}  {hash_prefix:<16}  {record['provider']}")
    for name in dispatcher.pool:
        n_assigned = sum(1 for r in plan if r["provider"] == name)
        typer.echo(f"  {name}: {n_assigned}/{len(plan)} page(s)")
    if out is not None:
        # stderr: the path is run-status, not plan content — stdout must
        # stay byte-identical across runs even when --out names differ.
        typer.echo(f"dispatch-plan: plan JSONL written to {out}", err=True)

    if provenance is not None:
        log = ProvenanceLog(provenance)
        appended = sum(
            log.append(
                content_hash=str(r["content_hash"]),
                provider=str(r["provider"]),
                task_kind=task_kind,
                redaction_applied=None,  # reserved ssDI field — dry-run never redacts
                notes=f"dispatch-plan dry-run; pool={','.join(dispatcher.pool)}",
            )
            for r in plan
        )
        # stderr on purpose: append count differs between first/second run,
        # while stdout must stay byte-identical for the determinism check.
        typer.echo(
            f"dispatch-plan: provenance {appended} new / "
            f"{len(plan) - appended} already logged (log: {provenance})",
            err=True,
        )


def main() -> None:
    """Entry point for the `jcontract` console script."""
    # Configure structlog to emit human-friendly logs by default.
    structlog.configure(
        processors=[
            structlog.processors.add_log_level,
            structlog.processors.StackInfoRenderer(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.dev.ConsoleRenderer(),
        ]
    )
    app()


if __name__ == "__main__":
    main()
