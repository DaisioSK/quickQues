"""Shared text-vs-drawing page classifier (extracted from claude_vision_parser).

What:
    ``_classify_page(jpeg_bytes) -> PageKind`` — the cheap, API-free
    heuristic (dark-pixel ratio + edge-energy row distribution) that
    decides whether a rendered page is text-heavy or drawing-heavy.

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
    imports keep working.
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


# `_ImageStat` is re-exported so future calibrators (e.g. mean luminance
# bands per quadrant) can compose richer signals without re-importing.
_ImageStat = ImageStat
