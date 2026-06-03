"""ClaudeVisionCaptioner — VisionCaptioner impl using Claude Sonnet 4.6 Vision.

Why this exists:
- Phase 2 ssCaption. Engineering-drawing chunks need a textual handle for
  retrieval — pure OCR yields fragments ("50mm", "Rev A") while the
  drawing is what carries the semantic content ("waterproofing detail of
  pier 3"). The captioner produces a concise Chinese description
  plus an entity list; both get folded into the chunk's indexable text
  by the embedder + BM25 indexer (see schema.chunk_indexable_text).

Architecture:
- Mirrors ClaudeVisionParser's render+cache+API shape but with a
  JSON-output prompt and a separate cache dir (data/caption_cache/) so
  caption results don't collide with OCR results.
- Cache key: SHA-256(image_bytes + model + "caption") — content-addressed
  AND prompt-addressed. A drawing page that's captioned twice (e.g.
  same PDF re-ingested) hits cache on the second call.
- Failure modes (API error, non-JSON output, missing keys) ALL return an
  empty DrawingCaption(caption_zh="", entities=[]) — the Protocol
  contract says callers must not lose the page; downstream the
  IngestPipeline writes chunk.caption="" (distinguishable from None =
  captioner never ran, per DECISION-2.prep.3).

Cost: ~$0.005-0.012 per drawing page on Sonnet 4.5 (smaller output than
OCR — JSON only, no full page text). Cached repeats are $0.

Secret handling: API key via config.get_anthropic_api_key() — same path
ssC + ssOCR use. No key in code, no logging.
"""

from __future__ import annotations

import base64
import hashlib
import io
import logging
from pathlib import Path
from typing import ClassVar

import structlog
from anthropic import Anthropic

from jcontract.config import get_anthropic_api_key
from jcontract.impls._caption_shared import (
    build_caption_prompt,
    parse_caption_payload,
    payload_to_caption,
    read_caption_cache,
    write_caption_cache,
)
from jcontract.interfaces import DomainProfile, DrawingCaption

logger = structlog.get_logger(__name__)

logging.getLogger("pypdfium2").setLevel(logging.WARNING)


DEFAULT_MODEL = "claude-sonnet-4-5"
DEFAULT_MAX_TOKENS = 1024  # caption JSON is tiny vs OCR; 1024 is plenty.

# Cache filename suffix — distinguishes from ocr_cache/*.text.txt and
# ocr_cache/*.drawing.txt. JSON because we round-trip DrawingCaption.
CACHE_SUFFIX = ".caption.json"


class ClaudeVisionCaptioner:
    """VisionCaptioner that generates Chinese captions for drawing pages.

    Implements the VisionCaptioner Protocol — interchangeable with future
    Qwen2-VL / GPT-4o impls via config (no current swap).

    Caching: file-level cache in ``cache_dir`` keyed by SHA-256 of the
    image bytes (the rendered JPEG produced by the parser). Re-running
    ingest on the same drawing page hits the cache for $0.
    """

    backend: ClassVar[str] = "claude-vision-captioner"

    def __init__(
        self,
        *,
        cache_dir: Path = Path("data/caption_cache"),
        model: str = DEFAULT_MODEL,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        client: Anthropic | None = None,
        profile: DomainProfile | None = None,
    ) -> None:
        # Lazy: tests inject a mock client to avoid env access entirely.
        self._client = client
        self._cache_dir = cache_dir
        self._model = model
        self._max_tokens = max_tokens
        # Phase 7 SS4: caption prompt from the active DomainProfile (None →
        # construction default). Non-contract profile name folds into the cache
        # key so domains don't share caption entries.
        self._caption_prompt = profile.caption_prompt if profile else None
        self._profile_name = profile.name if profile and profile.name != "contract" else None

    def _ensure_client(self) -> Anthropic:
        """Lazy-create the Anthropic client; only fail if we actually call the API."""
        if self._client is None:
            self._client = Anthropic(api_key=get_anthropic_api_key())
        return self._client

    def caption(self, image_bytes: bytes, nearby_text: str) -> DrawingCaption:
        """Generate a DrawingCaption from rendered image bytes + nearby OCR text.

        Per Protocol contract: NEVER raises; on any failure returns an
        empty DrawingCaption(caption_zh="", entities=[]) so the caller
        can record "captioner ran but produced nothing" via chunk.caption = "".
        """
        self._cache_dir.mkdir(parents=True, exist_ok=True)

        # Cache key bakes in model to avoid serving a stale answer if the
        # caller swaps Sonnet → Opus mid-project. "caption" suffix
        # distinguishes from ocr_cache (defense in depth — directories
        # already differ, but it's cheap).
        h = hashlib.sha256()
        h.update(image_bytes)
        h.update(self._model.encode("ascii"))
        h.update(b"caption")
        if self._profile_name:  # only non-contract → existing contract cache preserved
            h.update(self._profile_name.encode("utf-8"))
        cache_key = h.hexdigest()
        cache_path = self._cache_dir / f"{cache_key}{CACHE_SUFFIX}"

        cached = read_caption_cache(cache_path)
        if cached is not None:
            logger.info("captioner.cache_hit", cache_key=cache_key[:12])
            return cached

        # Cache miss → API call. Wrap broad except per Protocol contract.
        try:
            payload = self._call_caption_api(image_bytes, nearby_text)
        except Exception as exc:  # noqa: BLE001
            # error_type only, never the message body (could contain key
            # in some failure modes; mirrors ClaudeVisionParser pattern).
            logger.warning("captioner.api_error", error_type=type(exc).__name__)
            return DrawingCaption(caption_zh="", entities=[])

        # Persist cache even on empty result — saves $$ when the same
        # truly-blank page is re-ingested.
        write_caption_cache(cache_path, payload)
        return payload_to_caption(payload)

    def _call_caption_api(self, image_bytes: bytes, nearby_text: str) -> dict[str, object]:
        """Single Vision API call returning a parsed-JSON dict.

        Returns ``{"caption_zh": str, "entities": list}`` on success.
        On non-JSON output or missing keys, returns the empty payload
        ``{"caption_zh": "", "entities": []}`` — same fallback shape the
        caller stores so downstream code only handles one branch.
        """
        image_b64 = base64.standard_b64encode(image_bytes).decode("ascii")
        client = self._ensure_client()

        # Prompt (with tag-wrapped, trimmed nearby text) is built by the
        # shared helper so all three captioners stay calibrated together.
        prompt = build_caption_prompt(nearby_text, self._caption_prompt)

        response = client.messages.create(
            model=self._model,
            max_tokens=self._max_tokens,
            messages=[
                {
                    "role": "user",
                    "content": [
                        # Image before text per Anthropic best practice.
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

        # Extract text from content blocks. SDK union may include
        # ToolUseBlock with no .text; gate on the type tag.
        text_parts: list[str] = []
        for block in response.content:
            block_type = getattr(block, "type", None)
            if block_type == "text":
                # Why type: ignore: union widens beyond TextBlock; we gated
                # on `type == "text"` immediately above.
                text_parts.append(block.text)  # type: ignore[union-attr]
        raw_text = "\n".join(text_parts).strip()

        logger.info(
            "captioner.api_complete",
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            chars=len(raw_text),
        )

        # Shared defensive parse (fence-strip + shape-check + entity
        # coercion) → empty payload on any defect.
        return parse_caption_payload(raw_text)


def render_page_to_jpeg(pdf_path: Path, page_num: int, dpi: int = 150) -> bytes:
    """Render a single PDF page (1-indexed) to JPEG bytes.

    Helper used by IngestPipeline to materialize image bytes for the
    captioner on demand. Kept here (next to the captioner) rather than
    in ingest/pipeline.py so that:
      - the pipeline module stays free of pypdfium2/PIL imports
      - a future caller that wants image_bytes (e.g. a UI thumbnailer)
        can import the helper without dragging in the pipeline

    Returns JPEG q=85 bytes at the requested DPI. ~50ms on a desktop CPU
    for an A4 page; called only for drawing chunks during ingest.
    """
    # Local imports — keeps top-of-file imports lean for callers that
    # only need the captioner class.
    import pypdfium2 as pdfium
    from PIL import Image  # noqa: F401  # ensure Pillow loaded for save()

    pdf = pdfium.PdfDocument(str(pdf_path))
    try:
        # 1-indexed external API ↔ 0-indexed pypdfium2.
        page = pdf[page_num - 1]
        scale = dpi / 72.0
        pil_image = page.render(scale=scale).to_pil()
        buf = io.BytesIO()
        pil_image.save(buf, format="JPEG", quality=85)
        return buf.getvalue()
    finally:
        pdf.close()
