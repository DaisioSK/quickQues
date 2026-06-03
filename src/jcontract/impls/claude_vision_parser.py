"""ClaudeVisionParser — PDFParser impl using Claude Vision OCR.

Why this exists:
- Real input-docs/Contract DEMO*.pdf are 100% image-only scans (verified
  by ssA Phase 1; PDF content streams only contain `/Im16 Do` image-show
  operators, no text). Text-only parsers (pypdf, pdfplumber) extract 0
  chars. Claude Vision is the path that works.

Architecture:
- pypdfium2 renders each page to a PIL Image (no system deps; ships pdfium
  binary in wheels).
- Each rendered page is sent to Anthropic's Vision API with one of two
  task-specific prompts:
    * TEXT_OCR_PROMPT — preserves Q&A markers, Drawing No. refs, Section/
      Clause structure for the chunker's regex layer.
    * DRAWING_CAPTION_PROMPT — extracts title block, labels, dimensions,
      handwritten annotations for engineering drawings (where pure OCR
      would return useless line/arrow noise).
  A cheap, API-free classifier inspects the rendered JPEG and routes
  to the right prompt. See `_classify_page` for the heuristic.
- Results are cached by SHA-256 of the rendered JPEG bytes — re-running
  ingest does not pay for the same page twice. The chosen prompt name
  is included in the cache key so a drawing-vs-text reclassification
  on a re-run does not silently return stale text.

Cost: ~$0.012-$0.027 per A4 page on Sonnet 4.5; see
reference/claude-vision-ocr.md for the token math.

Secret handling: API key read via config.get_anthropic_api_key() — the
same path ssC's ClaudeAnswerer uses. No key in code, no logging.
"""

from __future__ import annotations

import base64
import hashlib
import io
import logging
from pathlib import Path
from typing import ClassVar, Literal

import pypdfium2 as pdfium
import structlog
from anthropic import Anthropic
from PIL import Image, ImageFilter, ImageStat

from jcontract.config import get_anthropic_api_key
from jcontract.impls._ocr_cache_key import model_cache_suffix
from jcontract.interfaces import DomainProfile, ParsedPage

logger = structlog.get_logger(__name__)

# Suppress noisy pypdfium2 INFO logs during rendering.
logging.getLogger("pypdfium2").setLevel(logging.WARNING)


# DPI = 150 → A4 renders to ~1240x1754 px. Sonnet 4.5 caps images at 1568 px
# long edge — rendering higher just wastes bytes. See reference/.
DEFAULT_DPI = 150
DEFAULT_JPEG_QUALITY = 85
DEFAULT_MODEL = "claude-sonnet-4-5"
DEFAULT_MAX_TOKENS = 2048

# Sentinel returned by either prompt when a page is blank.
EMPTY_PAGE_SENTINEL = "<empty page>"

PageKind = Literal["text", "drawing"]


# OCR prompt for text-heavy pages — preserves the structure the downstream
# qa_chunker.py regex depends on (Question No., Answer, Drawing No., Clause,
# Rev). Why this exact shape: see reference/claude-vision-ocr.md.
TEXT_OCR_PROMPT = """\
You are extracting text from a single page of a construction tender contract PDF.

Return ONLY the extracted text, preserving:
- Paragraph breaks (blank line between paragraphs).
- Section / Clause headers (keep on their own lines).
- "Question No.:" and "Answer:" markers exactly as printed.
- Drawing No. references (e.g. T/PRJ/CWD/WS/2101A) verbatim.
- Revision markers (Rev A, Revision 0, etc.).
- Tables: render as plain text with column separators (use " | ") and one row per line.

Do NOT:
- Add commentary, summaries, or notes.
- Translate any text (keep English as English).
- Skip handwritten annotations or stamps — transcribe them inline with a [handwritten: ...] marker.
- Describe images / drawings; the next ingest sub-sprint handles vision captioning separately.

If the page is blank or contains no text, return exactly: <empty page>"""


# Backward-compat alias — older imports of OCR_PROMPT continue to work.
OCR_PROMPT = TEXT_OCR_PROMPT


# Drawing prompt — when a page is an engineering drawing, plain OCR
# produces useless "a line" / "an arrow" output. We instead ask for a
# structured *textual representation* of the drawing that:
#   - keeps the qa_chunker.py regex hooks intact (Drawing No., Clause refs)
#   - surfaces labels / dimensions / annotations that are critical for
#     quantity-style queries (e.g. "TSA 总面积是多少?" — the area is often
#     only written on the drawing, not in any body paragraph)
DRAWING_CAPTION_PROMPT = """You are extracting structured information from a single page of an \
engineering drawing from a construction tender contract PDF.

Return ONLY a text representation of the drawing, including:
- Title block: drawing title, Drawing No. (e.g. T/PRJ/CWD/WS/2101A), \
revision (Rev A, Revision 0), scale, date.
- Key labels and dimensions visible on the drawing (numeric callouts, \
area markers, distances, materials).
- Any annotations or handwritten notes — transcribe them inline with a \
[handwritten: ...] marker.
- Drawing No. references to other drawings (verbatim, e.g. T/PRJ/CWD/WS/2101A).
- Section / Clause references visible on the sheet (e.g. Clause 7.3).
- Tabular data on the sheet (legends, schedules, revision history) — \
render as "key: value" lines, one entry per line.

Do NOT:
- Describe visual style ("a blueprint of...", "the drawing shows...").
- Make up content that is not on the page.
- Add commentary, summaries, or interpretation.
- Translate any text (keep English as English).

If the page is blank or contains nothing useful, return exactly: <empty page>"""


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


class ClaudeVisionParser:
    """PDFParser that OCRs scanned PDFs via Claude Vision.

    Implements the PDFParser Protocol — interchangeable with PyPdfParser via
    config (currently CLI --parser flag).

    Dual-prompt routing (Phase 1.7):
      The parser inspects each rendered page with a cheap heuristic
      classifier and chooses TEXT_OCR_PROMPT or DRAWING_CAPTION_PROMPT.
      Set ``auto_classify=False`` to disable routing and always use the
      text OCR prompt (Phase 1.5 backward-compat behaviour).

    Caching: file-level cache in ``cache_dir`` keyed by SHA-256 of the
    rendered JPEG **plus the chosen prompt kind**. Survives across
    process restarts and re-ingest. Delete files in the cache dir to
    force re-OCR.
    """

    backend: ClassVar[str] = "claude-vision"

    def __init__(
        self,
        *,
        cache_dir: Path = Path("data/ocr_cache"),
        dpi: int = DEFAULT_DPI,
        jpeg_quality: int = DEFAULT_JPEG_QUALITY,
        model: str = DEFAULT_MODEL,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        max_pages: int | None = None,
        client: Anthropic | None = None,
        auto_classify: bool = True,
        profile: DomainProfile | None = None,
    ) -> None:
        # Lazily resolve API key — only fail if we actually need to call the API.
        # Tests may inject a mocked client to avoid touching the env.
        self._client = client
        self._cache_dir = cache_dir
        self._dpi = dpi
        self._jpeg_quality = jpeg_quality
        self._model = model
        self._max_tokens = max_tokens
        # max_pages bounds how many pages we OCR — useful for cost-controlled
        # spikes against large PDFs (set None to process the whole document).
        self._max_pages = max_pages
        # When False, behave exactly like Phase 1.5: always use the text
        # OCR prompt regardless of page contents. Used by callers that
        # want deterministic single-prompt behaviour (e.g. eval baselines).
        self._auto_classify = auto_classify
        # Phase 7 SS4: prompts come from the active DomainProfile. None →
        # the construction (contract) constants, byte-for-byte unchanged. The
        # profile name is folded into the cache key (suppressed for contract)
        # so re-OCRing the same page under a different domain re-runs
        # instead of returning the other domain's prompt output.
        self._text_prompt = profile.ocr_text_prompt if profile else TEXT_OCR_PROMPT
        self._drawing_prompt = profile.ocr_drawing_prompt if profile else DRAWING_CAPTION_PROMPT
        self._profile_suffix = model_cache_suffix(
            profile.name if profile else "contract", "contract"
        )

    def _ensure_client(self) -> Anthropic:
        """Lazy-create the Anthropic client so tests can inject mocks freely."""
        if self._client is None:
            # get_anthropic_api_key raises a clear error if not set.
            self._client = Anthropic(api_key=get_anthropic_api_key())
        return self._client

    def parse(self, pdf_path: Path) -> list[ParsedPage]:
        """Render + OCR every page (up to ``max_pages``) and return ParsedPage list."""
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        self._cache_dir.mkdir(parents=True, exist_ok=True)

        # pypdfium2 opens once and we iterate; close on exit.
        pdf = pdfium.PdfDocument(str(pdf_path))
        try:
            total_pages = len(pdf)
            # Bound by max_pages if set; otherwise process the whole PDF.
            page_count = min(total_pages, self._max_pages) if self._max_pages else total_pages
            logger.info(
                "vision_parser.start",
                pdf=pdf_path.name,
                total_pages=total_pages,
                will_process=page_count,
                dpi=self._dpi,
                auto_classify=self._auto_classify,
            )

            pages: list[ParsedPage] = []
            for page_idx in range(page_count):
                page_num = page_idx + 1  # 1-indexed per ParsedPage contract
                text = self._ocr_page(pdf[page_idx], page_num, pdf_path.name)
                pages.append(ParsedPage(page_num=page_num, text=text))
            return pages
        finally:
            pdf.close()

    def _ocr_page(self, page: pdfium.PdfPage, page_num: int, pdf_name: str) -> str:
        """Render one page, check cache, call Vision if miss, return text."""
        # Render to PIL Image. pypdfium2 uses 72 DPI as the unit "1.0"; scale
        # accordingly so we hit our target rasterization density.
        scale = self._dpi / 72.0
        pil_image = page.render(scale=scale).to_pil()

        # JPEG-encode to bytes (used both as the API payload and as the cache key).
        # Quality=85 keeps text legible; lower values introduce JPEG ringing that
        # hurts OCR (see reference/claude-vision-ocr.md "Image format choices").
        buf = io.BytesIO()
        pil_image.save(buf, format="JPEG", quality=self._jpeg_quality)
        jpeg_bytes = buf.getvalue()

        # Decide which prompt to use. When auto_classify is off we
        # preserve the Phase 1.5 single-prompt behaviour exactly.
        # Defensive: if a caller monkey-patches `_classify` with a
        # raising impl (or a future refactor introduces a bug there),
        # we MUST NOT lose the page. Fall back to "text" — the safer
        # prompt — and emit a warning. The module-level
        # `_classify_page` already does its own try/except internally,
        # so this is belt-and-braces for the override path.
        if self._auto_classify:
            try:
                page_kind: PageKind = self._classify(jpeg_bytes)
            except Exception:  # noqa: BLE001
                logger.warning(
                    "vision_parser.classify_raised_fallback_text",
                    pdf=pdf_name,
                    page=page_num,
                )
                page_kind = "text"
        else:
            page_kind = "text"
        prompt = self._drawing_prompt if page_kind == "drawing" else self._text_prompt

        # Cache key: hash of the actual rendered bytes plus the prompt
        # kind — content-addressed AND prompt-addressed. If the page
        # is later reclassified (e.g. tuner adjusts thresholds) we re-OCR
        # rather than returning a stale text/drawing extraction.
        cache_key = hashlib.sha256(jpeg_bytes).hexdigest()
        # E10: fold the model into the key when it isn't the legacy
        # default so switching `--vision-model` re-OCRs rather than
        # returning text produced by a different-fidelity model. The
        # default model keeps the un-suffixed filename so caches written
        # before E10 still hit.
        model_suffix = model_cache_suffix(self._model, DEFAULT_MODEL)
        cache_path = (
            self._cache_dir / f"{cache_key}.{page_kind}{model_suffix}{self._profile_suffix}.txt"
        )

        if cache_path.exists():
            logger.info(
                "vision_parser.cache_hit",
                pdf=pdf_name,
                page=page_num,
                page_kind=page_kind,
                cache_key=cache_key[:12],
            )
            return cache_path.read_text(encoding="utf-8")

        # Cache miss → call the API. Failures don't abort the whole PDF; we log
        # and return empty text so chunker can filter (per PDFParser contract:
        # "Never raise on extraction-quality issues for a single page").
        try:
            text = self._call_vision_api(jpeg_bytes, prompt, page_num, pdf_name, page_kind)
        except Exception as exc:  # noqa: BLE001
            # Why broad except: a single page's API hiccup must not lose the
            # whole batch. We log + return empty per contract. Caller can
            # re-run; cache will pick up the recoveries.
            logger.warning(
                "vision_parser.api_error",
                pdf=pdf_name,
                page=page_num,
                page_kind=page_kind,
                error_type=type(exc).__name__,
            )
            return ""

        # Honour the empty-page sentinel from the prompt — store the canonical
        # empty string (not the sentinel) so chunker treats it like any blank.
        normalised = "" if text.strip() == EMPTY_PAGE_SENTINEL else text

        # Persist cache (even empty results — saves money on truly blank pages).
        cache_path.write_text(normalised, encoding="utf-8")
        return normalised

    def _classify(self, jpeg_bytes: bytes) -> PageKind:
        """Indirection so tests can monkeypatch classification on the instance.

        Defers to the module-level `_classify_page` heuristic. Kept as a
        method (not a direct call) precisely so tests can patch the
        parser's behaviour without touching the module global.
        """
        return _classify_page(jpeg_bytes)

    def _call_vision_api(
        self,
        jpeg_bytes: bytes,
        prompt: str,
        page_num: int,
        pdf_name: str,
        page_kind: PageKind,
    ) -> str:
        """Single Vision API call. Logs only metadata, never the answer body."""
        image_b64 = base64.standard_b64encode(jpeg_bytes).decode("ascii")
        client = self._ensure_client()

        response = client.messages.create(
            model=self._model,
            max_tokens=self._max_tokens,
            messages=[
                {
                    "role": "user",
                    "content": [
                        # Image BEFORE text per Anthropic best practice (see reference/).
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": image_b64,
                            },
                        },
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
        )

        # Extract text from content blocks. The SDK union includes ToolUseBlock
        # etc. which has no .text; guard via duck-typing on the `type` field.
        text_parts: list[str] = []
        for block in response.content:
            block_type = getattr(block, "type", None)
            if block_type == "text":
                text_parts.append(block.text)  # type: ignore[union-attr]  # Why: SDK union widens to non-text blocks; we gated on `type == "text"` immediately above.
        text = "\n".join(text_parts).strip()

        logger.info(
            "vision_parser.ocr_complete",
            pdf=pdf_name,
            page=page_num,
            page_kind=page_kind,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            chars=len(text),
        )

        return text
