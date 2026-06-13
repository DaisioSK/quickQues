"""policy-trace — per-page valve decision mock against a PagefixPolicy (ssMock).

What:
    Given a :class:`~jcontract.impls._pagefix_policy.PagefixPolicy` and a
    page's signals, ``trace_page`` reproduces the four ssPageFix valves'
    routing decision *without* touching the index, the caption lane, or any
    write path — it just reports, per page, the final route + the reason +
    the signals that drove it. ``classify_v2_mock`` is the load-bearing
    piece: a byte-faithful re-statement of
    :func:`jcontract.impls._page_classify.classify_page_v2`'s decision tree,
    reading its thresholds from the policy instead of the module constants.

Why a mock (DECISION-pm.20):
    Seeing what a different policy does to the corpus used to mean a full
    re-ingest (render -> OCR -> classify -> caption -> embed, ~4h). The
    routing decision, though, is a pure function of cheap, already-cached
    per-page signals (OCR box count + ssGE ``box_coverage`` + an optional
    ink ratio). So we read those signals — from the q45d quality-scan JSONL
    for the whole corpus (zero OCR, seconds) or from a fresh light render
    for a single PDF — and run ONLY the decision tree. "Change a threshold,
    see what flips" drops from hours to a CLI call.

Faithfulness contract (DECISION-pm.21):
    ``classify_v2_mock`` must agree with ``classify_page_v2`` on every input.
    It is NOT a paraphrase: the branch order, the comparisons, and the
    fall-throughs mirror the framework function line-for-line, only sourcing
    thresholds from the policy. A pytest pins this by feeding the SAME
    synthetic pixels + box signals to both functions across a matrix that
    exercises every branch boundary (tests/test_policy_trace.py), and the
    PoC verified it live on the frozen 14-page calibration set (verdicts
    drawing={24,38,50,559,23}, rest text — dev-sprint v8 §13 DECISION-pm.1).

Box-only fast path (DECISION-pm.22, closes UNCERTAIN-pm.2):
    The corpus JSONL carries ``boxes`` + ``box_coverage`` but no ink ratio.
    Without ``dark_ratio`` two branches are unevaluable: rule 3 (filled ->
    drawing) and rule 4 (sparse -> text, which needs BOTH coverage AND ink
    near-zero). We pass ``dark_ratio=None`` and make the mock SKIP both when
    ink evidence is absent — never fire them on a guess. Skipping rule 4 is
    the load-bearing choice: low-coverage real drawings (dimension-label /
    spec pages, cov<.10 but genuine ink) must NOT be carved out to text, and
    sparse title pages reach rule 6's text verdict anyway via the fragment
    test (their few large boxes clear the frag bar). This is exactly the PoC's
    box-rule set (boxes==0 OR fragmented). On the W6 corpus it reproduces the
    framework drawing set within the ±10 tolerance (233 vs the 225 live
    snapshot) and moves none of the frozen calibration verdicts. A single PDF
    run renders each page and computes the exact ``dark_ratio``, so rules 3-4
    DO fire there and per-PDF traces are exact.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from jcontract.impls._page_classify import _classify_page
from jcontract.impls._pagefix_policy import PagefixPolicy
from jcontract.interfaces.schema import PageKind


@dataclass(frozen=True)
class PageTrace:
    """One page's mocked routing decision under a policy.

    ``route`` is the page-fix outcome the mock cares about: ``"text"`` (the
    text lane carries the page) or ``"drawing"`` (needs the vision/caption
    lane). ``signals`` echoes the inputs that drove it so a flip is
    explainable from the JSONL alone; ``reason`` is the human string naming
    the firing branch.
    """

    page: int
    route: PageKind
    classify_verdict: PageKind
    reason: str
    signals: dict[str, Any]


def classify_v2_mock(
    boxes: int | None,
    box_coverage: float | None,
    dark_ratio: float | None,
    policy: PagefixPolicy,
) -> tuple[PageKind | None, str]:
    """Mirror of ``classify_page_v2``'s decision tree, thresholds from policy.

    Returns ``(verdict, reason)``. A ``None`` verdict means "no box evidence
    -> defer to the v1 pixel heuristic" (rule 1) — the caller supplies the
    v1 fallback, since the mock may not have pixels (JSONL fast path).

    Branch order is line-for-line with
    :func:`jcontract.impls._page_classify.classify_page_v2` [DECISION-pm.21]:

      1. ``boxes`` / ``box_coverage`` unavailable -> defer to v1.
      2. ``boxes == 0`` -> drawing.
      3. ``dark_ratio`` known AND > ``v2_filled_dark_ratio`` -> drawing.
         (``dark_ratio is None`` => ink evidence absent => rule cannot fire,
         the box-only fast-path convention, DECISION-pm.22.)
      4. ``dark_ratio`` known AND ``box_coverage < v2_sparse_cover`` AND
         ``dark_ratio < v2_sparse_dark`` -> text. This rule needs the ink
         signal (both halves must be near-zero); with ``dark_ratio is None``
         it is SKIPPED, not fired — firing it on coverage alone would route
         low-coverage drawings (dimension-label pages: cov<.10 but real ink)
         to text, the exact mis-route the box-only path must avoid. Sparse
         title pages are reached by rule 6 anyway (their few large boxes pass
         the fragment test), so skipping rule 4 keeps the corpus verdicts
         (DECISION-pm.22, matches PoC's box-rule set).
      5. ``box_coverage / boxes < v2_fragment_box_frac`` -> drawing.
      6. otherwise -> text.
    """
    if boxes is None or box_coverage is None:
        return None, "no-box-signals -> fallback-v1"
    if boxes == 0:
        return "drawing", "boxes==0 (no text -> graphical)"
    if dark_ratio is not None and dark_ratio > policy.v2_filled_dark_ratio:
        return "drawing", f"dark {dark_ratio:.3f} > {policy.v2_filled_dark_ratio} (filled)"
    if (
        dark_ratio is not None
        and box_coverage < policy.v2_sparse_cover
        and dark_ratio < policy.v2_sparse_dark
    ):
        return (
            "text",
            f"sparse cov {box_coverage:.3f} < {policy.v2_sparse_cover} (title/divider)",
        )
    if box_coverage / boxes < policy.v2_fragment_box_frac:
        return (
            "drawing",
            f"frag {box_coverage / boxes:.5f} < {policy.v2_fragment_box_frac} (spec/map/chart)",
        )
    return "text", "default -> text"


def trace_page(
    page: int,
    boxes: int | None,
    box_coverage: float | None,
    dark_ratio: float | None,
    policy: PagefixPolicy,
    *,
    jpeg_bytes: bytes | None = None,
) -> PageTrace:
    """Mock the classify valve's routing decision for one page under ``policy``.

    ``policy-trace`` exists to study the ssVR classify thresholds, so the
    route always reflects the v2 decision tree at the policy's thresholds
    (the thing under calibration) — independent of the ``needs_vision_v2``
    TOGGLE, whose job is to gate whether the real *pipeline* runs v2
    (DECISION-pm.3 / FORESHADOW-pm.1), not whether this tool computes the v2
    verdict. The toggle's state is still recorded in ``signals`` so a trace
    shows whether the policy would have the pipeline act on this verdict.

    ``jpeg_bytes`` (single-PDF path) lets rule 1's v1 fallback run the real
    pixel heuristic; on the JSONL fast path it is ``None`` and a deferred
    verdict is reported as ``"text"`` (v1's safe default) with the reason
    naming the gap.
    """
    verdict_opt, reason = classify_v2_mock(boxes, box_coverage, dark_ratio, policy)
    if verdict_opt is None:
        # Rule 1: no box evidence -> v1 fallback (pixels if we have them).
        verdict: PageKind = _classify_page(jpeg_bytes) if jpeg_bytes is not None else "text"
        reason = f"{reason} (v1={verdict})"
    else:
        verdict = verdict_opt
    return PageTrace(
        page=page,
        route=verdict,
        classify_verdict=verdict,
        reason=reason,
        signals={**_signals(boxes, box_coverage, dark_ratio), "v2_toggle": policy.needs_vision_v2},
    )


def _signals(
    boxes: int | None, box_coverage: float | None, dark_ratio: float | None
) -> dict[str, Any]:
    sig: dict[str, Any] = {"boxes": boxes, "box_coverage": box_coverage}
    if boxes and box_coverage is not None:
        sig["mean_box_area"] = round(box_coverage / boxes, 6)
    if dark_ratio is not None:
        sig["dark_ratio"] = round(dark_ratio, 4)
    return sig


def load_signal_jsonl(*paths: Path) -> list[dict[str, Any]]:
    """Read the q45d quality-scan JSONL records (boxes / box_coverage / page).

    These are the cached per-page geometry signals from the v7 W6 quality
    full-scan — the zero-OCR corpus source for the fast path.
    """
    records: list[dict[str, Any]] = []
    for path in paths:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
    return records


def trace_signals(records: list[dict[str, Any]], policy: PagefixPolicy) -> Iterator[PageTrace]:
    """Trace every record's routing decision under ``policy`` (zero OCR)."""
    for rec in records:
        yield trace_page(
            page=int(rec["page_num"]),
            boxes=rec.get("boxes"),
            box_coverage=rec.get("box_coverage"),
            dark_ratio=rec.get("dark_ratio"),  # absent in W6 JSONL -> None
            policy=policy,
        )


@dataclass(frozen=True)
class Flip:
    """One page whose route changed between two policies."""

    page: int
    route_a: PageKind
    route_b: PageKind
    reason_a: str
    reason_b: str


def diff_traces(traces_a: list[PageTrace], traces_b: list[PageTrace]) -> list[Flip]:
    """Pages whose route differs between policy A and policy B.

    Diff is POSITIONAL, not page-number-keyed: the two trace lists come from
    tracing the SAME ordered records under each policy, so record ``i`` is
    the same physical page on both sides. Keying by ``page`` would be wrong
    when a corpus concatenates several PDFs/JSONL whose page numbers restart
    at 1 (the q45d Part1/Part2 scans both number from 1 — a page-keyed diff
    would cross-match Part1 p.23 with Part2 p.23). The ``page`` carried on
    each Flip is the source label only.
    """
    if len(traces_a) != len(traces_b):
        raise ValueError(
            f"diff requires aligned traces (got {len(traces_a)} vs {len(traces_b)}); "
            "both policies must trace the same source."
        )
    flips: list[Flip] = []
    for ta, tb in zip(traces_a, traces_b, strict=True):
        if ta.route != tb.route:
            flips.append(
                Flip(
                    page=ta.page,
                    route_a=ta.route,
                    route_b=tb.route,
                    reason_a=ta.reason,
                    reason_b=tb.reason,
                )
            )
    return flips
