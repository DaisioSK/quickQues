"""DeepSeekV4Parser — PDFParser impl using DeepSeek V4 Vision (OpenAI-compatible).

Why this exists:
- Phase 1.10 enhancement. Fourth PDFParser vendor (after pypdf, claude-vision,
  claude-cli-vision). DeepSeek V4 is natively multimodal and exposes an
  OpenAI-compatible chat-completions endpoint, so we can reach it through
  the standard `openai` Python SDK with `base_url=https://api.deepseek.com`.
- Cost lever: deepseek-v4-flash is roughly 3-5x cheaper per page than Claude
  Sonnet 4.6 Vision, making it the preferred default for the 4100-page DEMO
  full-ingest run (DECISION-1.10.2 in docs/dev-sprint.md, user 2026-05-29).

Architecture:
- Identical render path to ClaudeVisionParser (pypdfium2 -> PIL -> JPEG @ 150
  DPI q=85). Same dual-prompt routing (text OCR vs drawing caption) driven by
  the SAME `_classify_page` heuristic — we import from claude_vision_parser to
  avoid duplicating that calibration work (project_seasee.md §5.3 / N=2 rule).
- API call format: OpenAI chat.completions with content list containing
  `{type: image_url, image_url: {url: "data:image/jpeg;base64,..."}}` then
  `{type: text, text: <prompt>}`. DeepSeek V4 accepts the OpenAI vision
  payload as-is per their public docs.
- Cache layout: `data/ocr_cache/deepseek-v4-<hash>.<kind>.txt` — independent
  prefix from claude vendor cache so a re-OCR with a different vendor does
  not return stale text from the other.

Secret handling: API key read via config.get_deepseek_api_key() — same
path Phase 4 DeepSeek text answerer will reuse. No key in code, no logging.
First-touch upgrade to High-Risk Mode per dev-contract §6.D.

Cost: ~$0.001-0.003 per A4 page on deepseek-v4-flash (vs ~$0.005-0.010 on
deepseek-v4-pro and ~$0.012-0.027 on Claude Sonnet 4.6 Vision). See
reference/deepseek-v4-vision.md for the token math.
"""

from __future__ import annotations

import base64
import hashlib
import logging
from pathlib import Path
from typing import ClassVar

import pypdfium2 as pdfium
import structlog
from openai import OpenAI

from jcontract.config import get_deepseek_api_key
from jcontract.impls._ocr_cache_key import model_cache_suffix
from jcontract.impls._page_classify import _classify_page
from jcontract.impls._pdfium_render import render_page_jpeg
from jcontract.impls.claude_vision_parser import (
    DRAWING_CAPTION_PROMPT,
    EMPTY_PAGE_SENTINEL,
    TEXT_OCR_PROMPT,
)
from jcontract.interfaces import DomainProfile, PageKind, ParsedPage

logger = structlog.get_logger(__name__)

# Suppress noisy pypdfium2 INFO logs during rendering.
logging.getLogger("pypdfium2").setLevel(logging.WARNING)


# DPI = 150 mirrors ClaudeVisionParser; image cap on DeepSeek V4 is not
# publicly nailed down at the byte level, but a 150-DPI A4 (~1240x1754) is
# inside the documented "tens of images per request" envelope and matches
# what we already validated against the calibration fixtures.
DEFAULT_DPI = 150
DEFAULT_JPEG_QUALITY = 85

# DECISION-1.10.2 (2026-05-29, user): default to flash; prototype favours
# cost. Constructor parameter lets callers swap to "deepseek-v4-pro" if
# they spot quality regressions on a given PDF.
DEFAULT_MODEL = "deepseek-v4-flash"
DEFAULT_MAX_TOKENS = 2048

# DeepSeek's published base URL for the OpenAI-compatible endpoint. The
# OpenAI SDK appends `/chat/completions` itself — do not include it here.
# UNCERTAIN-1.10.2 (deferred to user smoke test): some compat shims also
# require a trailing `/v1` path segment. If a 404 surfaces during the
# user's integration test, the fix is to set base_url to
# `https://api.deepseek.com/v1` — same model, same key.
DEEPSEEK_BASE_URL = "https://api.deepseek.com"


class DeepSeekV4Parser:
    """PDFParser that OCRs scanned PDFs via DeepSeek V4 Vision.

    Implements the PDFParser Protocol — interchangeable with ClaudeVisionParser
    via the CLI ``--parser deepseek-v4`` flag.

    Dual-prompt routing: shares the Phase 1.7 heuristic classifier with
    ClaudeVisionParser (we import `_classify_page`), so text-heavy pages get
    TEXT_OCR_PROMPT and drawing pages get DRAWING_CAPTION_PROMPT. Set
    ``auto_classify=False`` to force the text prompt for every page (useful
    when comparing vendors on an equal footing during eval).

    Caching: file-level cache in ``cache_dir`` keyed by ``deepseek-v4-<sha256
    of rendered JPEG>.<kind>.txt``. The `deepseek-v4-` prefix isolates this
    vendor's cache from the Anthropic vendor's; the page-kind suffix means a
    reclassification triggers re-OCR rather than returning stale text.
    """

    backend: ClassVar[str] = "deepseek-v4"
    # Cache filename prefix isolates this vendor from Anthropic in ocr_cache/.
    cache_prefix: ClassVar[str] = "deepseek-v4"

    def __init__(
        self,
        *,
        cache_dir: Path = Path("data/ocr_cache"),
        dpi: int = DEFAULT_DPI,
        jpeg_quality: int = DEFAULT_JPEG_QUALITY,
        model: str = DEFAULT_MODEL,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        max_pages: int | None = None,
        client: OpenAI | None = None,
        auto_classify: bool = True,
        profile: DomainProfile | None = None,
    ) -> None:
        # Lazily resolve API key — only fail if we actually need to call the API.
        # Tests inject a mocked client so they never touch the env.
        self._client = client
        self._cache_dir = cache_dir
        self._dpi = dpi
        self._jpeg_quality = jpeg_quality
        self._model = model
        self._max_tokens = max_tokens
        # max_pages bounds how many pages we OCR — useful for cost-controlled
        # spikes against large PDFs (set None to process the whole document).
        self._max_pages = max_pages
        # Phase 1.5 backward-compat: force the text prompt for every page.
        self._auto_classify = auto_classify
        # Phase 7 SS4: prompts from the active DomainProfile (None → contract
        # constants, unchanged). Profile name folds into the cache key
        # (suppressed for contract) so a re-OCR under a new domain re-runs.
        self._text_prompt = profile.ocr_text_prompt if profile else TEXT_OCR_PROMPT
        self._drawing_prompt = profile.ocr_drawing_prompt if profile else DRAWING_CAPTION_PROMPT
        self._profile_suffix = model_cache_suffix(
            profile.name if profile else "contract", "contract"
        )

    def _ensure_client(self) -> OpenAI:
        """Lazy-create the OpenAI client pointed at DeepSeek's base URL.

        Why lazy: the constructor must not require DEEPSEEK_API_KEY (tests
        inject a mock client; CLI lazy-imports this module). The key check
        only fires on the first real API call.
        """
        if self._client is None:
            # get_deepseek_api_key() raises a clear error if not set, naming
            # only the env var key and never the value.
            self._client = OpenAI(api_key=get_deepseek_api_key(), base_url=DEEPSEEK_BASE_URL)
        return self._client

    def parse(self, pdf_path: Path) -> list[ParsedPage]:
        """Render + OCR every page (up to ``max_pages``) and return ParsedPage list."""
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        self._cache_dir.mkdir(parents=True, exist_ok=True)

        pdf = pdfium.PdfDocument(str(pdf_path))
        try:
            total_pages = len(pdf)
            page_count = min(total_pages, self._max_pages) if self._max_pages else total_pages
            logger.info(
                "deepseek_v4_parser.start",
                pdf=pdf_path.name,
                total_pages=total_pages,
                will_process=page_count,
                model=self._model,
                dpi=self._dpi,
                auto_classify=self._auto_classify,
            )

            pages: list[ParsedPage] = []
            for page_idx in range(page_count):
                page_num = page_idx + 1  # 1-indexed per ParsedPage contract
                pages.append(self._parse_page(pdf[page_idx], page_num, pdf_path.name))
            return pages
        finally:
            pdf.close()

    def _parse_page(self, page: pdfium.PdfPage, page_num: int, pdf_name: str) -> ParsedPage:
        """Render one page, classify, OCR (cache-aware), return ParsedPage.

        Mirrors ClaudeVisionParser._parse_page. Sequential entry point
        (parse loop); concurrent callers render via ``render_pdf_page_jpeg``
        and call ``_ocr_jpeg`` directly — see DECISION-ab3.46.

        ssCL: classification happens ONCE here and is passed down to
        ``_ocr_jpeg`` (prompt routing + cache key) AND recorded on the
        ParsedPage (``page_kind``) so the chunker can emit drawing chunks
        for the --caption lane.
        """
        # Render via the shared serialized helper — JPEG bytes are payload
        # AND cache key (concurrency-deterministic, DECISION-ab3.46).
        jpeg_bytes = render_page_jpeg(page, dpi=self._dpi, jpeg_quality=self._jpeg_quality)
        page_kind = self._page_kind(jpeg_bytes, page_num, pdf_name)
        text = self._ocr_jpeg(jpeg_bytes, page_num, pdf_name, page_kind=page_kind)
        return ParsedPage(page_num=page_num, text=text, page_kind=page_kind)

    def _page_kind(self, jpeg_bytes: bytes, page_num: int, pdf_name: str) -> PageKind:
        """auto_classify-aware classification with the safe-text fallback.

        Single home for the routing decision so the prompt/cache key (in
        ``_ocr_jpeg``) and ``ParsedPage.page_kind`` can never diverge. If
        the classifier raises (user-patched broken impl), fall back to
        "text" — same belt-and-braces as ClaudeVisionParser.
        """
        if not self._auto_classify:
            return "text"
        try:
            return self._classify(jpeg_bytes)
        except Exception:  # noqa: BLE001
            logger.warning(
                "deepseek_v4_parser.classify_raised_fallback_text",
                pdf=pdf_name,
                page=page_num,
            )
            return "text"

    def _ocr_jpeg(
        self,
        jpeg_bytes: bytes,
        page_num: int,
        pdf_name: str,
        page_kind: PageKind | None = None,
    ) -> str:
        """Classify + cache-check + Vision call for pre-rendered JPEG bytes.

        Touches no pdfium state — safe to call from any thread. The
        3-positional-arg signature matches the Claude vendors so cli.py
        batch-ingest can dispatch to any vendor uniformly. ``page_kind``
        lets ``_parse_page`` pass its already-computed classification;
        ``None`` (batch-ingest path) classifies here — same heuristic,
        same bytes, same verdict.
        """
        # Classify before cache lookup so the cache key matches the prompt
        # actually used.
        if page_kind is None:
            page_kind = self._page_kind(jpeg_bytes, page_num, pdf_name)
        prompt = self._drawing_prompt if page_kind == "drawing" else self._text_prompt

        # Cache key: vendor prefix + sha256(jpeg) + kind + profile. Vendor
        # prefix isolates this from Anthropic's cache entries in the same dir
        # (DECISION-1.10.3); profile suffix (suppressed for contract) isolates
        # per-domain prompt output (Phase 7 SS4).
        cache_key = hashlib.sha256(jpeg_bytes).hexdigest()
        cache_path = (
            self._cache_dir
            / f"{self.cache_prefix}-{cache_key}.{page_kind}{self._profile_suffix}.txt"
        )

        if cache_path.exists():
            logger.info(
                "deepseek_v4_parser.cache_hit",
                pdf=pdf_name,
                page=page_num,
                page_kind=page_kind,
                cache_key=cache_key[:12],
            )
            return cache_path.read_text(encoding="utf-8")

        # Cache miss → call the API. Per PDFParser contract a single page's
        # API hiccup MUST NOT abort the batch; we log + return empty string.
        try:
            text = self._call_vision_api(jpeg_bytes, prompt, page_num, pdf_name, page_kind)
        except Exception as exc:  # noqa: BLE001
            # Why broad except: same as ClaudeVisionParser — preserve the
            # PDFParser contract. error_type only, never the message (which
            # could contain the API key for some failure modes).
            logger.warning(
                "deepseek_v4_parser.api_error",
                pdf=pdf_name,
                page=page_num,
                page_kind=page_kind,
                error_type=type(exc).__name__,
            )
            return ""

        # Honour the sentinel — chunker downstream treats empty pages
        # uniformly regardless of which prompt produced them.
        normalised = "" if text.strip() == EMPTY_PAGE_SENTINEL else text

        # Persist cache (even empty — saves money on truly blank pages
        # during a re-ingest).
        cache_path.write_text(normalised, encoding="utf-8")
        return normalised

    def _classify(self, jpeg_bytes: bytes) -> PageKind:
        """Indirection so tests can monkeypatch classification on the instance.

        Defers to claude_vision_parser._classify_page so the calibration
        thresholds stay in one place — the day we re-tune the heuristic,
        both vendors pick up the change automatically (N=2 / §5.3).
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
        """Single Vision API call via OpenAI-compatible chat.completions.

        Logs only metadata (token counts, page kind), never the answer body
        and never the request payload.
        """
        image_b64 = base64.standard_b64encode(jpeg_bytes).decode("ascii")
        client = self._ensure_client()

        # OpenAI vision content list: image entry BEFORE text entry. The
        # `detail: high` hint matters for OCR — `auto` would let the model
        # downscale to ~512 px on its side, which degrades small-font
        # extraction; we already cap at 150 DPI on our side so paying for
        # high-detail is the right trade.
        response = client.chat.completions.create(
            model=self._model,
            max_tokens=self._max_tokens,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_b64}",
                                "detail": "high",
                            },
                        },
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
        )

        # OpenAI response shape: choices[0].message.content is the str (or None
        # in tool-only flows; we never trigger that). Guard defensively.
        content = response.choices[0].message.content if response.choices else None
        text = (content or "").strip()

        # usage may be absent on some compat shims; tolerate None to keep
        # logging cheap. Don't fail OCR for a missing telemetry field.
        usage = response.usage
        logger.info(
            "deepseek_v4_parser.ocr_complete",
            pdf=pdf_name,
            page=page_num,
            page_kind=page_kind,
            model=self._model,
            prompt_tokens=getattr(usage, "prompt_tokens", None),
            completion_tokens=getattr(usage, "completion_tokens", None),
            chars=len(text),
        )

        return text
