"""Batch OCR ingest orchestrator — concurrent, resumable, budget-guarded.

Why this exists:
    Phase 1.5's ClaudeVisionParser processes pages serially with a per-page
    cache. For Phase 1.7's ~4100-page DEMO 9-PDF ingest, serial OCR would
    take ~5.7h and cost ~$50-100. We need three orthogonal capabilities
    layered on top of the existing parser:

      1. Concurrency  — N pages in-flight via asyncio.Semaphore so a single
         slow page doesn't stall the batch. ~8x speedup at N=8 (network +
         API latency dominate; the CPU work per page is tiny).
      2. Checkpointing — record per-(file, page) completion to a JSONL file
         so a killed process resumes where it left off. Distinct from the
         parser's image-hash cache: that cache is content-addressed and
         saves $ on the API; this checkpoint records what THIS BATCH has
         processed (chunker + embedder + index work happens above the cache
         layer and is what we want to skip on resume).
      3. Budget guard — track cumulative $ reported by the underlying
         parse callable. If a max_budget_usd cap is set, raise
         BudgetExceededError cleanly BEFORE starting a new page so we
         never abort mid-page (in-flight pages always finish).

Design choice: this orchestrator does NOT instantiate ClaudeVisionParser
directly. It takes a callable `ocr_one_page_fn(pdf_path, page_num) ->
(text, cost_usd)`. That keeps unit tests pure (mock callable, no real OCR)
and lets the main agent wire whatever per-page function fits — including
non-OCR pipelines or different vision backends.

Per-page errors are non-fatal: we log + record status="error" + continue
(same contract as ClaudeVisionParser's per-page graceful degradation).
This matches the contract in interfaces/parser.py:
"Never raise on extraction-quality issues for a single page".
"""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import structlog

logger = structlog.get_logger(__name__)


# Awaitable signature for the per-page worker the orchestrator drives.
# Returning (text, cost_usd) lets the orchestrator track $ without knowing
# the vendor; cost_usd may be 0.0 for cache hits (the underlying parser is
# free to report 0 when it hit its own cache).
OcrOnePageFn = Callable[[Path, int], Awaitable[tuple[str, float]]]


class BudgetExceededError(RuntimeError):
    """Raised by BatchIngest.run() when cumulative cost crosses max_budget_usd.

    Carries (spent_usd, limit_usd) for callers that want to surface the
    exact spend. Raised AFTER all in-flight pages complete and the
    checkpoint is fully flushed — partial progress is safe to resume.
    """

    def __init__(self, spent_usd: float, limit_usd: float) -> None:
        self.spent_usd = spent_usd
        self.limit_usd = limit_usd
        super().__init__(f"Budget exceeded: spent ${spent_usd:.4f} >= limit ${limit_usd:.4f}")


@dataclass(frozen=True)
class BatchProgress:
    """One row appended to the checkpoint JSONL per processed page."""

    file: str  # basename only — Path.name, keeps the checkpoint portable
    page: int  # 1-indexed, matches ParsedPage.page_num
    status: Literal["done", "skipped_cached", "error"]
    error_msg: str | None = None
    cost_usd: float = 0.0


@dataclass(frozen=True)
class BatchSummary:
    """Aggregate stats returned by BatchIngest.run() on completion."""

    files_processed: list[str]
    pages_done: int
    pages_skipped: int
    pages_error: int
    total_cost_usd: float
    elapsed_s: float


@dataclass
class _RunState:
    """Mutable state shared by all coroutines within a single run().

    Why a dataclass not class attributes: keeps BatchIngest stateless
    across runs (each run() gets its own state). Also makes the contract
    of what is shared between coroutines crystal-clear.
    """

    spent_usd: float = 0.0
    pages_done: int = 0
    pages_skipped: int = 0
    pages_error: int = 0
    pages_attempted: int = 0  # done + error; excludes skipped (for ETA)
    budget_tripped: bool = False
    # Asyncio lock guarding the checkpoint file + counter mutations. We
    # don't need fine-grained locks because per-page critical section is
    # tiny (one JSON line write) and pages are dominated by network I/O.
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)


class BatchIngest:
    """Concurrent + resumable + budget-guarded multi-PDF OCR orchestrator.

    Usage (sketch):

        parser = ClaudeVisionParser(...)

        async def ocr_one_page(pdf_path: Path, page_num: int) -> tuple[str, float]:
            # Render + call vision + return (text, cost_usd).
            # The main-agent integrator writes this glue; we just consume.
            ...

        batch = BatchIngest(
            pdf_paths=[Path("input-docs/Contract DEMO(1of9) TQA.pdf"), ...],
            max_concurrent=8,
            max_budget_usd=20.0,
        )
        summary = asyncio.run(batch.run(ocr_one_page_fn=ocr_one_page))

    Reentry: re-invoking run() with the same checkpoint_path skips pages
    that were recorded with status="done". Pages with status="error" are
    retried (errors are usually transient — rate limits, timeouts).
    """

    def __init__(
        self,
        *,
        pdf_paths: list[Path],
        checkpoint_path: Path = Path("data/ingest_checkpoint.jsonl"),
        max_concurrent: int = 4,
        max_budget_usd: float | None = None,
        progress_every: int = 25,
        page_counts: dict[Path, int] | None = None,
    ) -> None:
        """Configure the batch.

        page_counts: optional mapping pdf_path -> total page count. If
        omitted, the orchestrator does not know how many pages each PDF
        has and must rely on the caller via ``run(page_counts=...)`` or
        by passing pre-built (path, page) tuples. We keep this lightweight
        — discovering page counts requires opening each PDF, which is a
        concern of the integrator (it already opened them).
        """
        if max_concurrent < 1:
            raise ValueError("max_concurrent must be >= 1")
        if max_budget_usd is not None and max_budget_usd < 0:
            raise ValueError("max_budget_usd must be non-negative")
        if progress_every < 1:
            raise ValueError("progress_every must be >= 1")

        self.pdf_paths = list(pdf_paths)
        self.checkpoint_path = checkpoint_path
        self.max_concurrent = max_concurrent
        self.max_budget_usd = max_budget_usd
        self.progress_every = progress_every
        self.page_counts = page_counts or {}

    # -------------------- Checkpoint I/O --------------------

    def load_checkpoint(self) -> set[tuple[str, int]]:
        """Return set of (file_basename, page_num) entries with status='done'.

        We only treat ``done`` as "skip on resume". ``error`` pages get
        retried — they're usually transient (rate limit, timeout) and the
        whole point of resume is to make progress, including on what
        failed last time. ``skipped_cached`` rows can be treated the same
        as done (already complete from a previous run).
        """
        if not self.checkpoint_path.exists():
            return set()

        done: set[tuple[str, int]] = set()
        with self.checkpoint_path.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    # Tolerate a partially-written trailing line if the process
                    # was killed mid-flush. Skip it; we'll just redo that page.
                    logger.warning("batch.checkpoint_corrupt_line", line=line[:80])
                    continue
                if rec.get("status") in {"done", "skipped_cached"}:
                    done.add((str(rec["file"]), int(rec["page"])))
        return done

    def _append_checkpoint(self, progress: BatchProgress) -> None:
        """Append one progress row as JSON. Caller holds the asyncio lock.

        We open-append-close on every row rather than holding a file
        handle: this keeps every row durable (fsync on close), and the
        per-row overhead is negligible compared to a single API call.
        If a future profile shows this is a bottleneck, switch to a
        long-lived file handle + flush() per write under the same lock.
        """
        self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        # Build the dict explicitly so we control field order/types
        # (dataclasses.asdict would also work but is heavier).
        rec = {
            "file": progress.file,
            "page": progress.page,
            "status": progress.status,
            "error_msg": progress.error_msg,
            "cost_usd": progress.cost_usd,
        }
        line = json.dumps(rec, ensure_ascii=False)
        with self.checkpoint_path.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")
            fh.flush()

    # -------------------- Main run loop --------------------

    async def run(
        self,
        *,
        ocr_one_page_fn: OcrOnePageFn,
        page_counts: dict[Path, int] | None = None,
    ) -> BatchSummary:
        """Drive the batch to completion (or budget exceeded).

        ``page_counts`` overrides the constructor mapping if provided.
        Returns a BatchSummary on normal completion. Raises
        BudgetExceededError if max_budget_usd was set and crossed — the
        checkpoint will be fully flushed before the exception propagates.
        """
        counts = page_counts if page_counts is not None else self.page_counts
        if not counts:
            raise ValueError(
                "BatchIngest.run requires page_counts (either via constructor or run kwarg)"
            )

        # Build the (path, page) work list, filtering against checkpoint.
        already_done = self.load_checkpoint()
        work: list[tuple[Path, int]] = []
        pre_skipped = 0
        for pdf in self.pdf_paths:
            n_pages = counts.get(pdf)
            if n_pages is None:
                raise ValueError(f"No page count provided for {pdf}")
            for page_num in range(1, n_pages + 1):
                if (pdf.name, page_num) in already_done:
                    pre_skipped += 1
                    continue
                work.append((pdf, page_num))

        total_to_process = len(work)
        logger.info(
            "batch.start",
            files=len(self.pdf_paths),
            pages_to_process=total_to_process,
            pages_already_done=pre_skipped,
            max_concurrent=self.max_concurrent,
            max_budget_usd=self.max_budget_usd,
        )

        state = _RunState(pages_skipped=pre_skipped)
        semaphore = asyncio.Semaphore(self.max_concurrent)
        start_time = time.monotonic()

        async def worker(pdf: Path, page: int) -> None:
            # Acquire concurrency slot.
            async with semaphore:
                # Budget gate — re-checked here (not just before scheduling)
                # so that a budget crossed by EARLIER concurrent pages
                # blocks us from starting. We must NOT abort a page once
                # the API call is in-flight (that wastes the spend).
                if state.budget_tripped:
                    return

                if self.max_budget_usd is not None and state.spent_usd >= self.max_budget_usd:
                    # Mark tripped and let any waiting workers bail.
                    state.budget_tripped = True
                    return

                # Run the per-page callable. Errors here are per-page only
                # (parser contract: don't abort the batch on one bad page).
                progress: BatchProgress
                try:
                    text, cost = await ocr_one_page_fn(pdf, page)
                    # text is intentionally unused here — the orchestrator
                    # only tracks completion + cost. Downstream chunking/
                    # indexing is the integrator's responsibility (it sees
                    # the full text via the same callable's side effects
                    # or a separate pipeline call). We log a length though
                    # so progress logs show *something* happening.
                    progress = BatchProgress(
                        file=pdf.name,
                        page=page,
                        status="done",
                        cost_usd=cost,
                    )
                except Exception as exc:  # noqa: BLE001
                    # Broad except is correct here: parser is a black box,
                    # any failure mode (timeout, rate limit, malformed PDF
                    # page) must be recorded then surfaced via summary,
                    # not allowed to kill the whole batch.
                    logger.warning(
                        "batch.page_error",
                        file=pdf.name,
                        page=page,
                        error_type=type(exc).__name__,
                        error_msg=str(exc),
                    )
                    progress = BatchProgress(
                        file=pdf.name,
                        page=page,
                        status="error",
                        error_msg=f"{type(exc).__name__}: {exc}",
                        cost_usd=0.0,
                    )

                # Update shared state + flush checkpoint atomically.
                async with state.lock:
                    state.pages_attempted += 1
                    if progress.status == "done":
                        state.pages_done += 1
                        state.spent_usd += progress.cost_usd
                    else:
                        state.pages_error += 1

                    self._append_checkpoint(progress)

                    # Progress log — under the lock so the snapshot is
                    # consistent. progress_every counts attempted pages
                    # (done + error), not raw schedule index.
                    if state.pages_attempted % self.progress_every == 0:
                        self._log_progress(state, total_to_process, start_time)

        # Schedule all tasks. The semaphore inside worker() bounds
        # concurrency to max_concurrent at any instant. We use gather()
        # rather than as_completed so any exception in the orchestrator
        # itself (not per-page) propagates after all in-flight finish.
        tasks = [asyncio.create_task(worker(pdf, page)) for pdf, page in work]
        if tasks:
            await asyncio.gather(*tasks)

        elapsed = time.monotonic() - start_time

        # Final progress log if last batch didn't hit the modulus.
        if state.pages_attempted > 0 and state.pages_attempted % self.progress_every != 0:
            self._log_progress(state, total_to_process, start_time)

        summary = BatchSummary(
            files_processed=[p.name for p in self.pdf_paths],
            pages_done=state.pages_done,
            pages_skipped=state.pages_skipped,
            pages_error=state.pages_error,
            total_cost_usd=round(state.spent_usd, 6),
            elapsed_s=round(elapsed, 3),
        )

        # Raise AFTER all in-flight pages have settled and checkpoint is
        # consistent — caller can safely re-run for resume.
        if state.budget_tripped and self.max_budget_usd is not None:
            logger.warning(
                "batch.budget_exceeded",
                spent_usd=state.spent_usd,
                limit_usd=self.max_budget_usd,
                pages_done=state.pages_done,
            )
            raise BudgetExceededError(state.spent_usd, self.max_budget_usd)

        logger.info(
            "batch.complete",
            pages_done=state.pages_done,
            pages_skipped=state.pages_skipped,
            pages_error=state.pages_error,
            total_cost_usd=state.spent_usd,
            elapsed_s=elapsed,
        )
        return summary

    def _log_progress(self, state: _RunState, total_to_process: int, start_time: float) -> None:
        """Emit a structured progress log with ETA via linear extrapolation."""
        elapsed = time.monotonic() - start_time
        # ETA assumes remaining pages take same avg time as completed so
        # far. Good enough for human-readable progress; concurrency
        # smooths variance.
        if state.pages_attempted > 0 and total_to_process > 0:
            avg_per_page = elapsed / state.pages_attempted
            remaining = max(0, total_to_process - state.pages_attempted)
            eta_s = avg_per_page * remaining
        else:
            eta_s = 0.0

        logger.info(
            "batch.progress",
            pages_done=state.pages_done,
            pages_error=state.pages_error,
            pages_attempted=state.pages_attempted,
            pages_total=total_to_process,
            elapsed_s=round(elapsed, 1),
            eta_s=round(eta_s, 1),
            cumulative_cost_usd=round(state.spent_usd, 4),
        )
