"""Unit tests for policy-trace (ssMock): mock faithfulness + diff + fast path.

Strategy:
- FAITHFULNESS (the load-bearing contract, DECISION-pm.21): the mock classify
  tree must equal the framework ``classify_page_v2`` on EVERY input. Proven
  CI-safely by feeding the SAME synthetic pixels + box signals to both
  functions across a matrix that exercises every branch boundary (no real
  corpus, no de-identified filenames). The PoC additionally verified this
  live on the frozen 14-page calibration set; that verdict table is pinned
  here as DATA (box signals only) so a drifted decision tree fails loudly
  without shipping the real PDFs.
- BOX-ONLY FAST PATH (DECISION-pm.22): with no ink ratio, rule 3 cannot fire
  and rule 4's sparse->text carve-out fires on coverage alone — asserted to
  reproduce the calibration drawing set exactly.
- DIFF: a tightened fragmentation threshold flips a borderline page from
  drawing to text; the diff reports only the changed page with before/after.
"""

from __future__ import annotations

import io

import pytest
from PIL import Image, ImageDraw

from jcontract.impls._page_classify import classify_page_v2
from jcontract.impls._pagefix_policy import PagefixPolicy
from jcontract.impls._policy_trace import (
    Flip,
    PageTrace,
    classify_v2_mock,
    diff_traces,
    trace_page,
    trace_signals,
)

DEFAULT = PagefixPolicy()


def _jpeg(image: Image.Image) -> bytes:
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=85)
    return buf.getvalue()


def _page_with_ink(ink_fraction: float, size: tuple[int, int] = (800, 1000)) -> bytes:
    """White page with a black band covering ``ink_fraction`` of its area."""
    img = Image.new("L", size, 255)
    band_height = int(size[1] * ink_fraction)
    if band_height:
        ImageDraw.Draw(img).rectangle((0, 0, size[0], band_height), fill=0)
    return _jpeg(img)


def _mock_then_v1(
    jpeg: bytes,
    boxes: int | None,
    box_coverage: float | None,
    dark_ratio: float | None,
    policy: PagefixPolicy,
) -> str:
    """The mock verdict with the same v1 fallback the trace applies (rule 1)."""
    verdict, _ = classify_v2_mock(boxes, box_coverage, dark_ratio, policy)
    if verdict is None:
        from jcontract.impls._page_classify import _classify_page

        return _classify_page(jpeg)
    return verdict


# ---------------------------------------------------------------------------
# Faithfulness: mock decision tree == framework classify_page_v2, every branch
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("ink_fraction", [0.0, 0.01, 0.05, 0.3, 0.55, 0.8])
@pytest.mark.parametrize(
    ("boxes", "box_coverage"),
    [
        (None, None),  # rule 1: no box evidence -> v1
        (None, 0.3),  # rule 1: partial evidence -> v1
        (10, None),  # rule 1
        (0, 0.0),  # rule 2: boxes==0 -> drawing
        (4, 0.02),  # rule 4: sparse -> text
        (3, 0.019),  # rule 4 boundary (calibration title page)
        (60, 0.028),  # rule 5: fragmented -> drawing
        (146, 0.2026),  # rule 6: dense text (calibration p.774)
        (209, 0.1301),  # rule 5: fragmented but high coverage (calibration p.50)
    ],
)
def test_mock_matches_framework_classify_v2(ink_fraction, boxes, box_coverage):
    """Same pixels + boxes -> the mock (with rule-1 v1 fallback) == framework."""
    jpeg = _page_with_ink(ink_fraction)
    from jcontract.impls._page_classify import _dark_ratio

    dark = _dark_ratio(jpeg)
    framework = classify_page_v2(jpeg, boxes=boxes, box_coverage=box_coverage)
    mock = _mock_then_v1(jpeg, boxes, box_coverage, dark, DEFAULT)
    assert mock == framework, (
        f"DIVERGENCE ink={ink_fraction} boxes={boxes} cov={box_coverage}: "
        f"mock={mock} framework={framework}"
    )


def test_mock_matches_framework_filled_rule():
    """Rule 3: a heavily-inked page -> drawing in both (needs exact dark_ratio)."""
    from jcontract.impls._page_classify import _dark_ratio

    jpeg = _page_with_ink(0.8)  # > 0.5 filled
    dark = _dark_ratio(jpeg)
    framework = classify_page_v2(jpeg, boxes=146, box_coverage=0.2026)
    mock = _mock_then_v1(jpeg, 146, 0.2026, dark, DEFAULT)
    assert framework == "drawing"
    assert mock == "drawing"


# ---------------------------------------------------------------------------
# Frozen 14-page calibration verdicts (DATA, box signals only — no real PDFs)
# ---------------------------------------------------------------------------
# dev-sprint v8 §13 DECISION-pm.1: the ssVR calibration set's box signals
# (boxes, box_coverage) from the W6 q45d quality scan, with the KNOWN live
# classify_page_v2 verdict. drawing = {24,38,50,559,23}, rest text. Pinned so
# the box-only fast path provably reproduces the framework verdicts.
CALIBRATION = [
    # (label, boxes, box_coverage, expected_verdict)
    ("draw p.24", 65, 0.0379, "drawing"),
    ("draw p.38", 68, 0.0365, "drawing"),
    ("draw p.50", 209, 0.1301, "drawing"),
    ("draw p.559", 25, 0.0158, "drawing"),
    ("draw p.23", 60, 0.0281, "drawing"),
    ("title p.32", 2, 0.0074, "text"),
    ("title p.3", 4, 0.0210, "text"),
    ("title p.16", 3, 0.0190, "text"),
    ("table p.774", 146, 0.2026, "text"),
]


def test_box_only_fast_path_reproduces_calibration():
    """Box-only mock (dark_ratio=None) == the frozen live v2 verdicts. 0 drift."""
    mismatches = []
    for label, boxes, cov, expected in CALIBRATION:
        verdict, reason = classify_v2_mock(boxes, cov, None, DEFAULT)
        if verdict != expected:
            mismatches.append(f"{label}: got {verdict} ({reason}), want {expected}")
    assert not mismatches, "calibration drift:\n" + "\n".join(mismatches)


def test_box_only_drawing_set_matches_known():
    """The drawing subset under the default policy == {24,38,50,559,23} labels."""
    drawings = {
        label.split()[1]
        for label, boxes, cov, _ in CALIBRATION
        if classify_v2_mock(boxes, cov, None, DEFAULT)[0] == "drawing"
    }
    assert drawings == {"p.24", "p.38", "p.50", "p.559", "p.23"}


# ---------------------------------------------------------------------------
# trace_signals (JSONL fast path) + diff
# ---------------------------------------------------------------------------
def test_trace_signals_routes_from_box_records():
    records = [
        {"page_num": 24, "boxes": 65, "box_coverage": 0.0379},  # drawing
        {"page_num": 774, "boxes": 146, "box_coverage": 0.2026},  # text
        {"page_num": 99, "boxes": 0, "box_coverage": 0.0},  # boxes==0 drawing
    ]
    traces = {t.page: t for t in trace_signals(records, DEFAULT)}
    assert traces[24].route == "drawing"
    assert traces[774].route == "text"
    assert traces[99].route == "drawing"
    assert traces[24].signals["mean_box_area"] == pytest.approx(0.0379 / 65, rel=1e-3)


def test_tighter_fragment_threshold_flips_drawing_to_text():
    """A borderline fragmented page flips drawing->text when frag frac tightens."""
    # p.50: mean box area 0.1301/209 = 0.000622 -> drawing at 0.001, text at 0.0005.
    rec = [{"page_num": 50, "boxes": 209, "box_coverage": 0.1301}]
    tight = PagefixPolicy(v2_fragment_box_frac=0.0005)
    a = list(trace_signals(rec, DEFAULT))
    b = list(trace_signals(rec, tight))
    assert a[0].route == "drawing"
    assert b[0].route == "text"
    flips = diff_traces(a, b)
    assert flips == [Flip(50, "drawing", "text", a[0].reason, b[0].reason)]


def test_diff_is_positional_not_page_keyed():
    """Concatenated corpora with restarting page numbers must not cross-match.

    Two JSONL parts both number from p.1; a page-keyed diff would compare
    part1 p.1 against part2 p.1 (different physical pages). Positional diff
    aligns record i to record i, so a flip is reported per physical page.
    """
    rec = [
        {"page_num": 1, "boxes": 209, "box_coverage": 0.1301},  # part1 p.1: mean .000622
        {"page_num": 1, "boxes": 76, "box_coverage": 0.249},  # part2 p.1: text
    ]
    default = list(trace_signals(rec, DEFAULT))
    tight = list(trace_signals(rec, PagefixPolicy(v2_fragment_box_frac=0.0005)))
    # part1 p.1 (mean .000622, between .0005 and .001) flips drawing->text under
    # tight; part2 p.1 (mean .0033) stays text. A page-keyed diff (both p.1)
    # would collapse the two — positional diff keeps them distinct.
    flips = diff_traces(default, tight)
    assert [f.route_a + "->" + f.route_b for f in flips] == ["drawing->text"]


def test_diff_misaligned_lengths_raise():
    a = list(trace_signals([{"page_num": 1, "boxes": 10, "box_coverage": 0.3}], DEFAULT))
    with pytest.raises(ValueError, match="aligned"):
        diff_traces(a, [])


def test_diff_empty_when_policies_agree():
    rec = [{"page_num": 1, "boxes": 10, "box_coverage": 0.3}]
    a = list(trace_signals(rec, DEFAULT))
    b = list(trace_signals(rec, DEFAULT))
    assert diff_traces(a, b) == []


def test_trace_page_pdf_path_uses_v1_fallback_on_no_boxes():
    """jpeg given but box signals None -> rule 1 v1 fallback runs on pixels."""
    jpeg = _page_with_ink(0.05)  # a normal-ink page -> v1 'text'
    trace = trace_page(7, None, None, None, DEFAULT, jpeg_bytes=jpeg)
    assert isinstance(trace, PageTrace)
    assert trace.classify_verdict in ("text", "drawing")
    assert "fallback-v1" in trace.reason
