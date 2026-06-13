"""Shared text-vs-drawing page classifier (extracted from claude_vision_parser).

What:
    ``_classify_page(jpeg_bytes) -> PageKind`` — the cheap, API-free
    heuristic (dark-pixel ratio + edge-energy row distribution) that
    decides whether a rendered page is text-heavy or drawing-heavy.

    ``classify_page_v2(jpeg_bytes, boxes=, box_coverage=) -> PageKind`` —
    the ssVR "needs vision" judge: same question re-framed as "is the text
    alone enough to carry this page's information?", answered from OCR box
    statistics (box count + the ssGE ``box_coverage`` signal, reused not
    recomputed) joined with the v1 pixel ink signal. Opt-in; only the
    rapidocr lane has box data to feed it [DECISION-pl.32].

Why a shared module (ssCL):
    The heuristic was born in claude_vision_parser (Phase 1.7) and was
    already imported by deepseek_v4_parser (N=2 / §5.3: calibration
    thresholds live in ONE place). ssCL adds a third consumer — the
    rapidocr parser needs the same verdict to set ``ParsedPage.page_kind``
    and activate the drawing/caption lane — so the classifier moves to
    this vendor-neutral home instead of making the local-CPU vendor
    import the Anthropic vendor's module. The function body and all
    threshold constants are moved verbatim (pure relocation, zero
    behaviour change); claude_vision_parser re-exports them so existing
    imports keep working. ssVR adds v2 beside v1 in the same home for the
    same reason: one module owns every page-classification threshold.
"""

from __future__ import annotations

import io

import structlog
from PIL import Image, ImageFilter, ImageStat

from jcontract.interfaces.schema import PageKind

logger = structlog.get_logger(__name__)


# Classifier tunables. Calibrated against the real synthetic_contract_tqa.pdf
# (text-based, 10pt Helvetica) + synthetic line-drawing fixtures (see
# tests/test_claude_vision_parser.py).
#
# When in doubt we default to "text" because the text prompt is the
# safer fallback: a drawing run through the text prompt still extracts
# whatever labels are present (the prompt explicitly asks for Drawing
# No. and Clause refs); a text page run through the drawing prompt may
# omit body paragraphs ("Do NOT add commentary").
#
# Empirical signals on calibration fixtures (downsampled to 512 px,
# JPEG q=85 round-trip):
#                       dark180  strong_row   verdict
#   synthetic TQA p1    0.032    0.221        text  (10pt Helvetica @ 150 DPI)
#   synthetic TQA p4    0.025    0.184        text
#   text-dense bands    0.293    0.190        text  (block-shaped synthetic words)
#   drawing-dense       0.075    0.004        drawing  (many thin diagonals)
#   drawing-sparse      0.015    0.016        drawing
#   blank page          0.000    0.004        drawing
#
# The **strong-row ratio** (rows whose FIND_EDGES energy is > 1.5x mean)
# is the dominant discriminator — text pages produce many evenly-spaced
# strong rows from character baselines; drawings concentrate edge
# energy into a few title-block rows. Dark-pixel ratio acts as a sanity
# floor (truly-blank pages slip past edge tests).
_TEXT_MIN_STRONG_ROW_RATIO = 0.08
# Dark threshold 180 catches anti-aliased thin text (10pt @ 150 DPI ->
# downscaled -> mostly mid-gray pixels, never < 128). A page with
# < _BLANK_DARK_RATIO ink is treated as blank → drawing prompt.
_DARK_THRESHOLD = 180
_BLANK_DARK_RATIO = 0.005
# Heavily-filled pages (photos, halftones) > this ratio → drawing.
_FILLED_DARK_RATIO = 0.5


def _classify_page(jpeg_bytes: bytes) -> PageKind:
    """Cheap, API-free classifier: is this page text-heavy or drawing-heavy?

    Heuristic (no ML, no API calls — runs on the rendered JPEG only):

      1. Decode to grayscale via PIL, downscale to 512 px long edge.
      2. Dark-pixel ratio at threshold 180 — catches anti-aliased text
         (small font @ 150 DPI never renders pixels darker than 128
         after a JPEG round-trip + bilinear downscale).
      3. Strong-row ratio via FIND_EDGES: count rows whose edge energy
         is > 1.5x mean. Text pages produce many strong rows (one per
         character baseline + descenders); drawings concentrate edge
         energy into a few title-block rows and otherwise scatter
         thinly across the sheet.

    Decision (in order):
      * Dark ratio < `_BLANK_DARK_RATIO` → "drawing" (blank or trivially
        sparse — drawing prompt's "return <empty page>" branch handles
        it cheaply).
      * Dark ratio > `_FILLED_DARK_RATIO` → "drawing" (photo / halftone
        / heavily-shaded; not a real text page).
      * Strong-row ratio < `_TEXT_MIN_STRONG_ROW_RATIO` → "drawing"
        (edge energy too concentrated / scattered for a text page).
      * Otherwise → "text".

    Failure mode: any unexpected error (corrupt JPEG, unsupported PIL
    mode, OOM on absurdly-large input) returns "text" — the safer
    default. The fallback is logged so misclassifications can be
    diagnosed by re-running with verbose logging.
    """
    try:
        with Image.open(io.BytesIO(jpeg_bytes)) as raw:
            gray = raw.convert("L")
        # Aggressive downscale: classification needs structure, not detail.
        # Capping the long edge at 512 px makes the rest of the heuristic
        # O(<300k) pixels regardless of input DPI.
        gray.thumbnail((512, 512), Image.Resampling.BILINEAR)
        width, height = gray.size
        pixels = list(gray.getdata())
        n_pixels = width * height
        if n_pixels == 0:
            return "text"

        # Step 1: dark-pixel ratio.
        dark_count = sum(1 for p in pixels if p < _DARK_THRESHOLD)
        dark_ratio = dark_count / n_pixels

        if dark_ratio < _BLANK_DARK_RATIO:
            # Blank / near-blank → drawing (the drawing prompt's empty-page
            # branch returns the sentinel cheaply).
            return "drawing"
        if dark_ratio > _FILLED_DARK_RATIO:
            # Mostly filled → photo / halftone → drawing.
            return "drawing"

        # Step 2: edge-energy distribution.
        edges = gray.filter(ImageFilter.FIND_EDGES)
        edge_pixels = list(edges.getdata())
        row_sums = [sum(edge_pixels[r * width : (r + 1) * width]) for r in range(height)]
        total_edge_energy = sum(row_sums) or 1  # avoid div-by-zero
        mean_row_energy = total_edge_energy / height
        strong_rows = sum(1 for s in row_sums if s > 1.5 * mean_row_energy)
        strong_row_ratio = strong_rows / height

        if strong_row_ratio < _TEXT_MIN_STRONG_ROW_RATIO:
            return "drawing"

        return "text"

    except Exception:  # noqa: BLE001
        # Why broad except: classification is a non-critical preflight —
        # any failure (corrupt bytes, unsupported mode, OOM on huge img)
        # must not drop the page. Default to "text" which is the safer
        # prompt — see docstring.
        logger.warning("vision_parser.classify_error_fallback_text")
        return "text"


# ---------------------------------------------------------------------------
# ssVR: classify_page_v2 — "needs vision" judge from OCR-box + pixel signals
# ---------------------------------------------------------------------------
#
# v2 tunables [DECISION-pl.30]. Calibrated live (2026-06-12) on the frozen
# 14-page set (dev-sprint v7 §13, ssVR): 6 drawing pages (rotated schedule
# diagrams, a linkway spec drawing, a rotated tender-drawings list) vs 8 text
# pages (5 plain/table text pages + 3 near-empty divider/title pages).
# Signals measured on the UPRIGHT frame (after the ssRT rotation step):
#
#                         dark    boxes   box_coverage   coverage/boxes
#   drawings (5 pp)      .035-.111  51-276   .030-.199    .00048-.00072
#   dense text (6 pp)    .056-.120  42-146   .203-.543    .00139-.0111
#   title pages (3 pp)   .003-.008   2-4     .0074-.021   .0037-.0063
#
# The dominant discriminator is MEAN BOX AREA (box_coverage / boxes): a
# drawing's text arrives as many small fragments (dimension labels, title
# blocks), a text page's as full-width line boxes — the two populations are
# separated 1.39x on each side of 0.001 (geometric midpoint of .00072 vs
# .00139), where raw box_coverage alone leaves only a 1.018x gap
# (.199 drawing vs .2026 text). Title pages ("空旷页") are carved out FIRST
# by the sparse rule — almost no box coverage AND almost no ink means the
# few words ARE the whole page, so captioning buys nothing (this kills the
# v5 over-trigger: 64.5% empty captions on appendix title pages).
#
# Bias [DECISION-pl.30]: when in doubt, send to vision. The caption lane is
# ADDITIVE (a drawing page's OCR text still enters the index), so a false
# "drawing" costs GPU seconds while a false "text" makes the image's meaning
# permanently unretrievable. Hence the sparse rule requires BOTH signals to
# be near-zero, and every other ambiguous branch falls through toward the
# fragmentation test rather than an early "text".
#
# V2_SPARSE_COVER / V2_SPARSE_DARK: a page is "empty-ish" (→ text) only when
# box coverage AND ink are both tiny. Margins on the calibration set: title
# pages max cover .021 (4.8x under .10) and max dark .008 (2.5x under .02);
# the sparsest real drawing (p.559 linkway spec) clears the dark bar at .042
# (2.1x over) so it falls through to the fragmentation test.
V2_SPARSE_COVER = 0.10
V2_SPARSE_DARK = 0.02
# V2_FRAGMENT_BOX_FRAC: mean box area (box_coverage / boxes) below this →
# the page's text is fragmented labels → drawing. 0.001 is the geometric
# midpoint of the calibration populations (drawings ≤ .00072, text pages
# ≥ .00139 — 1.39x margin each side).
V2_FRAGMENT_BOX_FRAC = 0.001
# Heavily-inked pages (photos / halftones) route to drawing regardless of
# box stats — inherited unchanged from v1 (_FILLED_DARK_RATIO).
V2_FILLED_DARK_RATIO = _FILLED_DARK_RATIO


def _dark_ratio(jpeg_bytes: bytes) -> float:
    """v1's ink signal in isolation (same decode → grayscale → 512px → t180).

    Extracted for v2 instead of calling ``_classify_page`` because v2 needs
    the raw ratio, not v1's verdict; ``_classify_page`` itself stays frozen
    verbatim (zero-default-change mandate — its early-return structure means
    sharing this helper would alter its blank/filled fast paths).
    """
    with Image.open(io.BytesIO(jpeg_bytes)) as raw:
        gray = raw.convert("L")
    gray.thumbnail((512, 512), Image.Resampling.BILINEAR)
    width, height = gray.size
    n_pixels = width * height
    if n_pixels == 0:
        return 0.0
    dark_count = sum(1 for p in gray.getdata() if p < _DARK_THRESHOLD)
    return float(dark_count) / float(n_pixels)


def classify_page_v2(
    jpeg_bytes: bytes,
    *,
    boxes: int | None,
    box_coverage: float | None,
    sparse_cover: float = V2_SPARSE_COVER,
    sparse_dark: float = V2_SPARSE_DARK,
    fragment_box_frac: float = V2_FRAGMENT_BOX_FRAC,
    filled_dark_ratio: float = V2_FILLED_DARK_RATIO,
) -> PageKind:
    """ssVR v2 verdict: is this page's information carried by its text alone?

    Inputs: the rendered UPRIGHT frame (callers must resolve ssRT rotation
    first — sideways pixels make the ink signal noise) plus the OCR box
    statistics from the rapidocr metrics sidecar: ``boxes`` (ssQA) and
    ``box_coverage`` (ssGE — reused, never recomputed here).

    The four thresholds are keyword args defaulting to the calibrated module
    constants, so existing callers are byte-unchanged; a PagefixPolicy-driven
    caller (ssCfg) injects them from config instead. [DECISION-pm.11]

    Decision (in order) [DECISION-pl.30]:
      1. Box signals unavailable (pre-ssGE sidecar, no sidecar) → defer to
         the v1 pixel heuristic — v2 never guesses without its evidence.
      2. ``boxes == 0`` → drawing: no text at all, so whatever ink exists
         is purely graphical (and a truly blank page matches v1's cheap
         blank→drawing sentinel path).
      3. Ink > ``filled_dark_ratio`` → drawing (photo/halftone; v1 rule
         carried over).
      4. Sparse page (coverage < ``sparse_cover`` AND ink < ``sparse_dark``)
         → text: divider/title pages — the few words are the whole page,
         captioning them yields empty captions.
      5. Fragmented text (``box_coverage / boxes`` < ``fragment_box_frac``)
         → drawing: many small label boxes = spec drawings / maps / charts
         (the v1 under-trigger class).
      6. Otherwise → text.

    Failure mode: any unexpected error (corrupt JPEG, PIL OOM) defers to
    ``_classify_page`` — which itself degrades to "text" — so a page is
    never lost to classification (same belt-and-braces stance as v1).
    """
    try:
        if boxes is None or box_coverage is None:
            # No box evidence (pre-ssGE/ssQA caches, vendor without boxes):
            # fall back to the v1 verdict rather than judging blind.
            logger.info("page_classify.v2_no_box_signals_fallback_v1")
            return _classify_page(jpeg_bytes)

        if boxes == 0:
            # No text anywhere: ink (if any) is purely graphical; a blank
            # page rides the drawing prompt's cheap empty-page branch (v1
            # parity).
            return "drawing"

        dark_ratio = _dark_ratio(jpeg_bytes)
        if dark_ratio > filled_dark_ratio:
            return "drawing"
        if box_coverage < sparse_cover and dark_ratio < sparse_dark:
            return "text"
        if box_coverage / boxes < fragment_box_frac:
            return "drawing"
        return "text"

    except Exception:  # noqa: BLE001
        # Why broad except: classification is a non-critical preflight — any
        # failure must not drop the page. v1 is the defined degraded mode
        # (and v1 itself degrades to "text").
        logger.warning("page_classify.v2_error_fallback_v1")
        return _classify_page(jpeg_bytes)


# `_ImageStat` is re-exported so future calibrators (e.g. mean luminance
# bands per quadrant) can compose richer signals without re-importing.
_ImageStat = ImageStat
