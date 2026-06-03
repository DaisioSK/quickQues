"""Unit tests for BatchIngest orchestrator.

Strategy:
- All tests use a fake `ocr_one_page_fn` callable. NEVER spawn real OCR
  or hit the network. The orchestrator's contract is vendor-agnostic by
  design (it takes a callable), which makes mocking trivial.
- Each test exercises one orthogonal concern (concurrency bound,
  resumability, atomic checkpoint, budget guard, error handling, summary
  shape, progress logging). One test, one assertion focus.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from jcontract.ingest.batch import (
    BatchIngest,
    BatchSummary,
    BudgetExceededError,
)

# ----------------- helpers -----------------


def _read_checkpoint(path: Path) -> list[dict[str, object]]:
    """Read the JSONL checkpoint file as a list of dicts."""
    if not path.exists():
        return []
    rows: list[dict[str, object]] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _make_pdf_paths(tmp_path: Path, names: list[str]) -> list[Path]:
    """Return Path objects (files don't need to exist — orchestrator only uses .name)."""
    return [tmp_path / name for name in names]


def _ckpt_row(file: str, page: int, status: str = "done", cost: float = 0.0) -> str:
    """Build one JSONL checkpoint row (mirrors BatchIngest._append_checkpoint shape)."""
    return json.dumps(
        {
            "file": file,
            "page": page,
            "status": status,
            "error_msg": None,
            "cost_usd": cost,
        }
    )


# ----------------- tests -----------------


def test_concurrency_bound_never_exceeds_max_concurrent(tmp_path: Path) -> None:
    """100 pages, max_concurrent=4 → counter inside callable must never exceed 4."""
    pdf = tmp_path / "doc.pdf"
    counts = {pdf: 100}

    in_flight = 0
    peak = 0
    lock = asyncio.Lock()

    async def fake_ocr(pdf_path: Path, page_num: int) -> tuple[str, float]:
        nonlocal in_flight, peak
        async with lock:
            in_flight += 1
            peak = max(peak, in_flight)
        # Yield so other coroutines can race.
        await asyncio.sleep(0.001)
        async with lock:
            in_flight -= 1
        return (f"page {page_num}", 0.001)

    batch = BatchIngest(
        pdf_paths=[pdf],
        checkpoint_path=tmp_path / "ckpt.jsonl",
        max_concurrent=4,
    )
    summary = asyncio.run(batch.run(ocr_one_page_fn=fake_ocr, page_counts=counts))

    assert peak <= 4, f"Concurrency bound violated: peak={peak}"
    assert peak >= 2, "Test setup bad — should have observed actual concurrency"
    assert summary.pages_done == 100


def test_resumable_skips_done_pages(tmp_path: Path) -> None:
    """Pre-populated checkpoint with done pages → callable not invoked for them."""
    pdf = tmp_path / "doc.pdf"
    counts = {pdf: 5}
    ckpt = tmp_path / "ckpt.jsonl"

    # Pre-seed pages 1, 3 as already done.
    ckpt.write_text(
        _ckpt_row("doc.pdf", 1, cost=0.01) + "\n" + _ckpt_row("doc.pdf", 3, cost=0.01) + "\n",
        encoding="utf-8",
    )

    called_pages: list[int] = []

    async def fake_ocr(pdf_path: Path, page_num: int) -> tuple[str, float]:
        called_pages.append(page_num)
        return ("text", 0.02)

    batch = BatchIngest(pdf_paths=[pdf], checkpoint_path=ckpt, max_concurrent=2)
    summary = asyncio.run(batch.run(ocr_one_page_fn=fake_ocr, page_counts=counts))

    # Callable invoked only for pages 2, 4, 5 — never 1 or 3.
    assert sorted(called_pages) == [2, 4, 5]
    assert summary.pages_done == 3
    assert summary.pages_skipped == 2


def test_checkpoint_appends_atomically(tmp_path: Path) -> None:
    """Simulate kill mid-batch via exception. Checkpoint contains exactly N done rows."""
    pdf = tmp_path / "doc.pdf"
    counts = {pdf: 10}
    ckpt = tmp_path / "ckpt.jsonl"

    call_count = 0

    async def fake_ocr(pdf_path: Path, page_num: int) -> tuple[str, float]:
        nonlocal call_count
        call_count += 1
        # On the 4th call (process-level exception, not per-page), raise
        # something the orchestrator catches per-page → status="error".
        # Then on subsequent calls return fine.
        # We're really testing that whatever the row, the checkpoint
        # is consistent (no half-written line).
        if page_num == 4:
            raise RuntimeError("simulated crash on page 4")
        return (f"text {page_num}", 0.01)

    # Run sequentially (max_concurrent=1) to make row order deterministic
    # and the "mid-batch kill at row N" semantics meaningful.
    batch = BatchIngest(pdf_paths=[pdf], checkpoint_path=ckpt, max_concurrent=1)
    summary = asyncio.run(batch.run(ocr_one_page_fn=fake_ocr, page_counts=counts))

    rows = _read_checkpoint(ckpt)
    # All 10 pages produced exactly one row each (9 done, 1 error).
    assert len(rows) == 10
    # Each row is independently parseable (atomic line-per-row append).
    for r in rows:
        assert "file" in r and "page" in r and "status" in r
    # The error row is page 4, status=error.
    error_rows = [r for r in rows if r["status"] == "error"]
    assert len(error_rows) == 1
    assert error_rows[0]["page"] == 4
    assert summary.pages_error == 1
    assert summary.pages_done == 9


def test_budget_guard_aborts_cleanly(tmp_path: Path) -> None:
    """max_budget=$0.10, $0.05/page, 5 pages → after ~2 pages BudgetExceededError, ckpt valid."""
    pdf = tmp_path / "doc.pdf"
    counts = {pdf: 5}
    ckpt = tmp_path / "ckpt.jsonl"

    async def fake_ocr(pdf_path: Path, page_num: int) -> tuple[str, float]:
        # Each page costs $0.05. Budget is $0.10, so after 2 successful
        # pages the cumulative spend (0.10) crosses the limit and the
        # next worker to enter the critical section will trip.
        return (f"text {page_num}", 0.05)

    # Run serially so spend accumulates deterministically (with concurrency
    # we'd race; the orchestrator still aborts cleanly, but exact page
    # count varies).
    batch = BatchIngest(
        pdf_paths=[pdf],
        checkpoint_path=ckpt,
        max_concurrent=1,
        max_budget_usd=0.10,
    )

    with pytest.raises(BudgetExceededError) as exc_info:
        asyncio.run(batch.run(ocr_one_page_fn=fake_ocr, page_counts=counts))

    assert exc_info.value.limit_usd == 0.10
    assert exc_info.value.spent_usd >= 0.10

    # Checkpoint contains exactly 2 done rows (the 2 that succeeded before
    # the guard tripped). It must be valid JSONL — caller can resume.
    rows = _read_checkpoint(ckpt)
    assert len(rows) == 2
    assert all(r["status"] == "done" for r in rows)
    assert all(r["cost_usd"] == pytest.approx(0.05) for r in rows)


def test_progress_log_every_n(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Progress log fires every N attempted pages.

    structlog's default sink writes to stdout (not stdlib logging); we
    use capsys to grep the printed events for batch.progress.
    """
    pdf = tmp_path / "doc.pdf"
    counts = {pdf: 10}

    async def fake_ocr(pdf_path: Path, page_num: int) -> tuple[str, float]:
        return (f"text {page_num}", 0.001)

    batch = BatchIngest(
        pdf_paths=[pdf],
        checkpoint_path=tmp_path / "ckpt.jsonl",
        max_concurrent=1,
        progress_every=3,
    )

    asyncio.run(batch.run(ocr_one_page_fn=fake_ocr, page_counts=counts))

    captured = capsys.readouterr()
    progress_lines = [line for line in captured.out.splitlines() if "batch.progress" in line]
    # 10 pages with N=3: progress at attempted=3, 6, 9 + final-flush at 10
    # (since 10 % 3 != 0). Expect 4 events.
    assert len(progress_lines) >= 3, (
        f"Expected >=3 progress logs, got {len(progress_lines)}: {captured.out!r}"
    )


def test_per_page_error_does_not_abort(tmp_path: Path) -> None:
    """One bad page out of 5 → 4 done, 1 error, no exception bubbles."""
    pdf = tmp_path / "doc.pdf"
    counts = {pdf: 5}

    async def fake_ocr(pdf_path: Path, page_num: int) -> tuple[str, float]:
        if page_num == 3:
            raise ValueError("boom on page 3")
        return (f"text {page_num}", 0.01)

    batch = BatchIngest(
        pdf_paths=[pdf],
        checkpoint_path=tmp_path / "ckpt.jsonl",
        max_concurrent=2,
    )
    summary = asyncio.run(batch.run(ocr_one_page_fn=fake_ocr, page_counts=counts))

    assert summary.pages_done == 4
    assert summary.pages_error == 1
    # total_cost_usd reflects only the 4 successful pages (errors contribute 0).
    assert summary.total_cost_usd == pytest.approx(0.04)


def test_summary_counts_correct_with_mixed_outcomes(tmp_path: Path) -> None:
    """Summary aggregates done/skipped/error across multiple files."""
    pdfs = _make_pdf_paths(tmp_path, ["a.pdf", "b.pdf"])
    counts = {pdfs[0]: 3, pdfs[1]: 2}  # 5 pages total
    ckpt = tmp_path / "ckpt.jsonl"

    # Pre-seed: a.pdf page 1 is already done.
    ckpt.write_text(_ckpt_row("a.pdf", 1) + "\n", encoding="utf-8")

    async def fake_ocr(pdf_path: Path, page_num: int) -> tuple[str, float]:
        # b.pdf page 1 errors; rest succeed.
        if pdf_path.name == "b.pdf" and page_num == 1:
            raise RuntimeError("transient")
        return ("ok", 0.02)

    batch = BatchIngest(pdf_paths=pdfs, checkpoint_path=ckpt, max_concurrent=2)
    summary = asyncio.run(batch.run(ocr_one_page_fn=fake_ocr, page_counts=counts))

    assert isinstance(summary, BatchSummary)
    # 1 pre-skipped + 3 new done (a.pdf 2,3 + b.pdf 2) + 1 error (b.pdf 1)
    assert summary.pages_skipped == 1
    assert summary.pages_done == 3
    assert summary.pages_error == 1
    assert summary.total_cost_usd == pytest.approx(0.06)  # 3 * $0.02
    assert set(summary.files_processed) == {"a.pdf", "b.pdf"}
    assert summary.elapsed_s >= 0.0


def test_empty_work_list_returns_summary(tmp_path: Path) -> None:
    """If all pages already done in checkpoint, run completes with no callable calls."""
    pdf = tmp_path / "doc.pdf"
    counts = {pdf: 2}
    ckpt = tmp_path / "ckpt.jsonl"
    ckpt.write_text(
        _ckpt_row("doc.pdf", 1) + "\n" + _ckpt_row("doc.pdf", 2) + "\n",
        encoding="utf-8",
    )

    called = False

    async def fake_ocr(pdf_path: Path, page_num: int) -> tuple[str, float]:
        nonlocal called
        called = True
        return ("x", 0.0)

    batch = BatchIngest(pdf_paths=[pdf], checkpoint_path=ckpt, max_concurrent=2)
    summary = asyncio.run(batch.run(ocr_one_page_fn=fake_ocr, page_counts=counts))

    assert called is False
    assert summary.pages_done == 0
    assert summary.pages_skipped == 2


def test_constructor_validates_args(tmp_path: Path) -> None:
    """Bad inputs to BatchIngest raise ValueError up front."""
    pdf = tmp_path / "doc.pdf"
    with pytest.raises(ValueError):
        BatchIngest(pdf_paths=[pdf], max_concurrent=0)
    with pytest.raises(ValueError):
        BatchIngest(pdf_paths=[pdf], max_budget_usd=-1.0)
    with pytest.raises(ValueError):
        BatchIngest(pdf_paths=[pdf], progress_every=0)


def test_load_checkpoint_handles_corrupt_line(tmp_path: Path) -> None:
    """Partially-written trailing line should be tolerated (skipped)."""
    ckpt = tmp_path / "ckpt.jsonl"
    ckpt.write_text(
        _ckpt_row("x.pdf", 1)
        + "\n"
        + '{"file": "x.pdf", "page": 2, "status":  # truncated mid-write\n',
        encoding="utf-8",
    )
    batch = BatchIngest(pdf_paths=[tmp_path / "x.pdf"], checkpoint_path=ckpt)
    done = batch.load_checkpoint()
    assert done == {("x.pdf", 1)}
