"""ClaudeCliVisionParser — PDFParser via ``claude -p`` subprocess + Read tool.

Why this exists:
- Phase 1.5 ``ClaudeVisionParser`` calls the Anthropic SDK directly →
  requires ANTHROPIC_API_KEY → pay-per-token billing.
- Claude Code Max / Pro subscribers want a path that uses their flat
  monthly quota (no marginal $) instead of API tokens.
- Trick that makes this work: render the PDF page to a JPEG on disk,
  then call ``claude -p "extract text from <path>" --allowedTools Read
  --add-dir <renderdir>``. Claude Code uses its built-in Read tool to
  load the image, then Vision-reads it. Round-trip ~10-15s/page (vs
  ~3-5s API direct) but $0 marginal for subscription users.

Architecture:
- Re-uses the same pypdfium2 render path as ``ClaudeVisionParser``
  (DPI 150 + JPEG q=85). The render is local + free.
- Re-uses the same SHA-256 content-addressed cache at
  ``data/ocr_cache/<hash>.text.txt`` so re-runs are zero-cost.
- Calls ``claude -p`` as a subprocess. Auth invisible to us — uses the
  user's existing ``claude login`` OAuth.

Verified end-to-end on 2026-05-29 against the WeChat sample image:
all handwritten annotations + Drawing No. + Q&A markers preserved.

Cost vs ClaudeVisionParser (API direct):
- claude-cli-vision: $0 marginal (subscription); ~10-15s/page; ~40k
  cache_creation tokens overhead (Claude Code system context)
- claude-vision-api: ~$0.015/page (Sonnet) or ~$0.0025 (Haiku);
  ~5s/page; minimal overhead
"""

from __future__ import annotations

import hashlib
import io
import logging
import shutil
from pathlib import Path
from typing import ClassVar

import pypdfium2 as pdfium
import structlog

from jcontract.impls._claude_cli_runner import run_claude_read_image
from jcontract.impls._ocr_cache_key import model_cache_suffix
from jcontract.interfaces import DomainProfile, ParsedPage

logger = structlog.get_logger(__name__)

logging.getLogger("pypdfium2").setLevel(logging.WARNING)

DEFAULT_DPI = 150
DEFAULT_JPEG_QUALITY = 85
# "haiku" is the cheapest subscription quota burn. "sonnet" works too if
# the user prefers higher quality at higher quota cost — set via constructor.
DEFAULT_MODEL = "haiku"
DEFAULT_TIMEOUT_S = 180

EMPTY_PAGE_SENTINEL = "<empty page>"

# Prompt instructs claude to use its Read tool to load the rendered page
# image, then transcribe the text. Mirrors the prompt in
# ``claude_vision_parser.py`` so chunker-side regex hooks (Question No.,
# Drawing No., Clause refs) are preserved.
OCR_PROMPT_TEMPLATE = """Use the Read tool to open the image at: {image_path}

Then extract ALL text from that image. Return ONLY the extracted text, preserving:
- Paragraph breaks (blank line between paragraphs).
- Section / Clause headers on their own lines.
- "Question No.:" and "Answer:" markers exactly as printed.
- Drawing No. references (e.g. T/PRJ/CWD/WS/2101A) verbatim.
- Revision markers (Rev A, Revision 0, etc.).
- Tables: render as plain text with " | " column separators.

Do NOT:
- Add commentary, summaries, or analysis.
- Translate any text (keep English as English).
- Skip handwritten annotations or stamps — transcribe inline with a [handwritten: ...] marker.

If the page is blank or contains no useful text, return exactly: <empty page>"""


class ClaudeCliVisionParser:
    """PDFParser that OCRs scanned PDFs via ``claude -p`` subprocess.

    Uses the caller's existing ``claude login`` OAuth — no API key is
    read, stored, or transmitted by this class. Token usage counts
    against the user's Claude Code subscription quota.

    Drop-in alternative to ``ClaudeVisionParser`` (same Protocol, same
    cache format). Choose via CLI ``--parser claude-cli-vision``.
    """

    backend: ClassVar[str] = "claude-cli-vision"

    def __init__(
        self,
        *,
        cache_dir: Path = Path("data/ocr_cache"),
        render_dir: Path = Path("data/_render_tmp"),
        dpi: int = DEFAULT_DPI,
        jpeg_quality: int = DEFAULT_JPEG_QUALITY,
        model: str = DEFAULT_MODEL,
        max_pages: int | None = None,
        claude_path: str | None = None,
        timeout_s: int = DEFAULT_TIMEOUT_S,
        profile: DomainProfile | None = None,
    ) -> None:
        # Resolve binary at __init__ time so we fail loud if missing.
        resolved = claude_path or shutil.which("claude")
        if not resolved:
            raise RuntimeError(
                "claude CLI not found in PATH. "
                "Install Claude Code (https://docs.claude.com/en/docs/claude-code) "
                "and run `claude login`."
            )
        self._claude_path = resolved
        self._cache_dir = cache_dir
        # Rendered JPEGs go under the project tree (not /tmp) so claude
        # CLI doesn't need --add-dir for arbitrary paths.
        self._render_dir = render_dir
        self._dpi = dpi
        self._jpeg_quality = jpeg_quality
        self._model = model
        self._max_pages = max_pages
        self._timeout_s = timeout_s
        # Phase 7 SS4: for contract / no profile we keep this parser's own
        # historically-tuned OCR_PROMPT_TEMPLATE byte-for-byte. For any
        # other domain we use that profile's neutral ocr_text_prompt
        # (prefixed with the Read-the-image instruction). Profile name folds
        # into the cache key (suppressed for contract) so a re-OCR under a new
        # domain re-runs instead of returning contract's transcription.
        self._ocr_prompt_override = (
            profile.ocr_text_prompt if profile and profile.name != "contract" else None
        )
        self._profile_suffix = model_cache_suffix(
            profile.name if profile else "contract", "contract"
        )

    def parse(self, pdf_path: Path) -> list[ParsedPage]:
        """Render + OCR every page (up to ``max_pages``) and return ParsedPage list."""
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._render_dir.mkdir(parents=True, exist_ok=True)

        pdf = pdfium.PdfDocument(str(pdf_path))
        try:
            total_pages = len(pdf)
            page_count = min(total_pages, self._max_pages) if self._max_pages else total_pages
            logger.info(
                "cli_vision.start",
                pdf=pdf_path.name,
                total_pages=total_pages,
                will_process=page_count,
                model=self._model,
            )

            pages: list[ParsedPage] = []
            for page_idx in range(page_count):
                page_num = page_idx + 1
                text = self._ocr_page(pdf[page_idx], page_num, pdf_path.name)
                pages.append(ParsedPage(page_num=page_num, text=text))
            return pages
        finally:
            pdf.close()

    def _ocr_page(self, page: pdfium.PdfPage, page_num: int, pdf_name: str) -> str:
        """Render one page, check cache, call claude CLI if miss, return text."""
        # Same render path as ClaudeVisionParser — same JPEG bytes → same
        # cache key → cross-parser cache compatibility.
        scale = self._dpi / 72.0
        pil_image = page.render(scale=scale).to_pil()

        buf = io.BytesIO()
        pil_image.save(buf, format="JPEG", quality=self._jpeg_quality)
        jpeg_bytes = buf.getvalue()

        cache_key = hashlib.sha256(jpeg_bytes).hexdigest()
        # Tag with "text" prompt kind to match ClaudeVisionParser's cache
        # layout post-Phase-1.7-ssB (drawing detection cache splits).
        # E10: append the model slug when it isn't the legacy default
        # ("haiku") so `--vision-model sonnet` re-OCRs into its own
        # namespace instead of returning the cached haiku text. The
        # default keeps the un-suffixed filename → existing caches hit.
        model_suffix = model_cache_suffix(self._model, DEFAULT_MODEL)
        cache_path = self._cache_dir / f"{cache_key}.text{model_suffix}{self._profile_suffix}.txt"

        if cache_path.exists():
            logger.info(
                "cli_vision.cache_hit",
                pdf=pdf_name,
                page=page_num,
                cache_key=cache_key[:12],
            )
            return cache_path.read_text(encoding="utf-8")

        # Cache miss → write the rendered JPEG to a per-page file under
        # render_dir so claude CLI can Read it. Naming with cache_key so
        # multiple concurrent renders don't collide.
        image_path = self._render_dir / f"{cache_key}.jpg"
        image_path.write_bytes(jpeg_bytes)

        try:
            text = self._call_claude_cli(image_path, page_num, pdf_name)
        except Exception as exc:  # noqa: BLE001 — per-page errors must not abort batch
            logger.warning(
                "cli_vision.error",
                pdf=pdf_name,
                page=page_num,
                error_type=type(exc).__name__,
            )
            # Clean up the render before bailing.
            image_path.unlink(missing_ok=True)
            return ""

        normalised = "" if text.strip() == EMPTY_PAGE_SENTINEL else text
        cache_path.write_text(normalised, encoding="utf-8")
        # Render is no longer needed once OCR'd — keep the cache, drop the JPEG.
        image_path.unlink(missing_ok=True)
        return normalised

    def _call_claude_cli(self, image_path: Path, page_num: int, pdf_name: str) -> str:
        """Single ``claude -p`` invocation that reads the image and OCRs it.

        Delegates the subprocess + JSON-envelope handling to the shared
        ``run_claude_read_image`` runner (also used by the cli-vision
        captioner, E11); here we own the OCR prompt + usage logging.
        """
        if self._ocr_prompt_override is not None:
            # Non-contract domain: prepend the Read instruction to the profile's
            # neutral OCR prompt.
            prompt = (
                f"Use the Read tool to open the image at: {image_path.resolve()}\n\n"
                f"{self._ocr_prompt_override}"
            )
        else:
            prompt = OCR_PROMPT_TEMPLATE.format(image_path=image_path.resolve())

        data = run_claude_read_image(
            claude_path=self._claude_path,
            render_dir=self._render_dir,
            prompt=prompt,
            model=self._model,
            timeout_s=self._timeout_s,
        )

        text: str = data.get("result", "")

        # Token-usage log only, never logs prompt body or response body.
        usage = data.get("usage", {})
        logger.info(
            "cli_vision.ocr_complete",
            pdf=pdf_name,
            page=page_num,
            input_tokens=usage.get("input_tokens"),
            output_tokens=usage.get("output_tokens"),
            cache_creation=usage.get("cache_creation_input_tokens"),
            cache_read=usage.get("cache_read_input_tokens"),
            metered_cost_usd=data.get("total_cost_usd"),  # 0 for subscription
            chars=len(text),
        )

        return text
