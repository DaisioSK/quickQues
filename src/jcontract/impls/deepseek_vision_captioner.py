"""DeepSeekVisionCaptioner — VisionCaptioner via DeepSeek V4 Vision (OpenAI-compatible).

Enhancement E11 (``p2-ssMultiVendorCaptioner``). Third VisionCaptioner
impl: the API path for users who have a DEEPSEEK_API_KEY (the same key
Phase 1.10's DeepSeekV4Parser uses) rather than a Claude subscription.
Produces the same DrawingCaption JSON via the shared caption helpers.

Architecture (mirrors DeepSeekV4Parser):
- OpenAI SDK pointed at DeepSeek's base URL; reuses get_deepseek_api_key
  and DEEPSEEK_BASE_URL so there's one source of truth for the endpoint.
- OpenAI vision content list: image_url (data URI, detail=high) before
  the text prompt — same shape the parser validated.
- Prompt + parse + cache come from the shared caption helpers.

Secret handling: API key via config.get_deepseek_api_key() — no key in
code, no logging. Cost: ~$0.001-0.003 per drawing page on flash.
"""

from __future__ import annotations

import base64
import hashlib
import logging
from pathlib import Path
from typing import ClassVar

import structlog
from openai import OpenAI

from jcontract.config import get_deepseek_api_key
from jcontract.impls._caption_shared import (
    CACHE_SUFFIX,
    build_caption_prompt,
    parse_caption_payload,
    payload_to_caption,
    read_caption_cache,
    write_caption_cache,
)
from jcontract.impls.deepseek_v4_parser import DEEPSEEK_BASE_URL
from jcontract.interfaces import DomainProfile, DrawingCaption

logger = structlog.get_logger(__name__)

logging.getLogger("pypdfium2").setLevel(logging.WARNING)

# Flash is the cost default (caption output is short); swap to
# "deepseek-v4-pro" via constructor if drawing fidelity regresses.
DEFAULT_MODEL = "deepseek-v4-flash"
DEFAULT_MAX_TOKENS = 1024


class DeepSeekVisionCaptioner:
    """VisionCaptioner that captions drawing pages via DeepSeek V4 Vision."""

    backend: ClassVar[str] = "deepseek-vision-captioner"

    def __init__(
        self,
        *,
        cache_dir: Path = Path("data/caption_cache"),
        model: str = DEFAULT_MODEL,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        client: OpenAI | None = None,
        profile: DomainProfile | None = None,
    ) -> None:
        # Lazy: tests inject a mock client so they never touch the env.
        self._client = client
        self._cache_dir = cache_dir
        self._model = model
        self._max_tokens = max_tokens
        # Phase 7 SS4: caption prompt from the active DomainProfile (None →
        # construction default). Non-contract profile name folds into cache key.
        self._caption_prompt = profile.caption_prompt if profile else None
        self._profile_name = profile.name if profile and profile.name != "contract" else None

    def _ensure_client(self) -> OpenAI:
        """Lazy-create the OpenAI client pointed at DeepSeek; key check fires here."""
        if self._client is None:
            self._client = OpenAI(api_key=get_deepseek_api_key(), base_url=DEEPSEEK_BASE_URL)
        return self._client

    def caption(self, image_bytes: bytes, nearby_text: str) -> DrawingCaption:
        """Caption a drawing page; NEVER raises (returns empty on any failure)."""
        self._cache_dir.mkdir(parents=True, exist_ok=True)

        # Model + backend in the key so DeepSeek captions don't collide
        # with the Claude captioners' cache in the same dir.
        h = hashlib.sha256()
        h.update(image_bytes)
        h.update(self._model.encode("ascii"))
        h.update(self.backend.encode("ascii"))
        if self._profile_name:  # non-contract only → existing cache preserved
            h.update(self._profile_name.encode("utf-8"))
        cache_key = h.hexdigest()
        cache_path = self._cache_dir / f"{cache_key}{CACHE_SUFFIX}"

        cached = read_caption_cache(cache_path)
        if cached is not None:
            logger.info("deepseek_captioner.cache_hit", cache_key=cache_key[:12])
            return cached

        try:
            payload = self._call_caption_api(image_bytes, nearby_text)
        except Exception as exc:  # noqa: BLE001 — Protocol: never raise on one page
            logger.warning("deepseek_captioner.api_error", error_type=type(exc).__name__)
            return DrawingCaption(caption_zh="", entities=[])

        write_caption_cache(cache_path, payload)
        return payload_to_caption(payload)

    def _call_caption_api(self, image_bytes: bytes, nearby_text: str) -> dict[str, object]:
        """Single Vision call via OpenAI-compatible chat.completions → parsed payload."""
        image_b64 = base64.standard_b64encode(image_bytes).decode("ascii")
        client = self._ensure_client()
        prompt = build_caption_prompt(nearby_text, self._caption_prompt)

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

        content = response.choices[0].message.content if response.choices else None
        raw_text = (content or "").strip()

        usage = response.usage
        logger.info(
            "deepseek_captioner.api_complete",
            model=self._model,
            prompt_tokens=getattr(usage, "prompt_tokens", None),
            completion_tokens=getattr(usage, "completion_tokens", None),
            chars=len(raw_text),
        )
        return parse_caption_payload(raw_text)
