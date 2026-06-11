"""OllamaVisionCaptioner — VisionCaptioner via a local Ollama VLM (qwen3-vl).

v4 ssLocalStack (ssLB). Fourth VisionCaptioner impl: the fully-local path —
drawing captions from a VLM served by Ollama on this machine, so caption
generation costs $0 and no page image ever leaves the host (the
confidential-document use case the local stack exists for). Produces the
same DrawingCaption JSON via the shared caption helpers as every other
captioner (N=2 rule: prompt/parse/cache shared, only the backend differs).

Endpoint shape (DECISION-ls.20, live-verified 2026-06-11 on Ollama 0.20.2):
- What: we call Ollama's OpenAI-compatible ``/v1/chat/completions`` with the
  standard vision content list (base64 ``image_url`` data URI + text), via
  the ``openai`` SDK already in the dependency tree.
- Why: both candidate paths were probed live and both work; the OpenAI
  shape keeps this vendor byte-level symmetric with DeepSeekVisionCaptioner
  (same SDK, same message shape, same error taxonomy) and needs no raw-HTTP
  client for the native ``/api/chat`` ``images`` field.
- Context: resolves UNCERTAIN-ls.1; full probe evidence in dev-sprint v4 §13.

Reasoning output (DECISION-ls.11 carry-over): qwen3-vl is a thinking model;
Ollama returns the chain-of-thought in a separate ``message.reasoning``
field and ``message.content`` arrives clean — we read only ``content``.
Inline ``<think>`` blocks are still stripped defensively (shared helper)
because other compat shims inline them, which would break JSON parsing.

Configuration (env, safe local defaults — never a remote address):
  - JCONTRACT_OLLAMA_BASE_URL  default ``http://localhost:11434``
  - JCONTRACT_OLLAMA_VL_MODEL  default ``qwen3-vl:8b``

Secret handling: none — Ollama ignores auth; the SDK gets a non-secret
placeholder key, never read from the environment, never logged.
"""

from __future__ import annotations

import base64
import hashlib
from pathlib import Path
from typing import ClassVar

import structlog
from openai import OpenAI

from jcontract.config import get_ollama_base_url, get_ollama_vl_model
from jcontract.impls._caption_shared import (
    CACHE_SUFFIX,
    build_caption_prompt,
    parse_caption_payload,
    payload_to_caption,
    read_caption_cache,
    write_caption_cache,
)
from jcontract.impls.openai_compat_answerer import _strip_think
from jcontract.interfaces import DomainProfile, DrawingCaption

logger = structlog.get_logger(__name__)

# max_tokens is 8x the DeepSeek captioner's 1024. What: 8192 budget.
# Why: qwen3-vl's chain-of-thought counts against the completion budget on
# this endpoint (DECISION-ls.11 mechanism) and on a REAL drawing page it
# alone exceeds 2048 — the ls.20 probe capped out at max_tokens=2048 with
# ZERO visible content (completion_tokens=2048, content empty), while 8192
# finished at 3081 total. Thinking cannot be switched off through /v1
# (probed extra_body {"think": false}: reasoning still returned, still
# capped), so headroom is the fix; local tokens are free. [DECISION-ls.20]
DEFAULT_MAX_TOKENS = 8192

# Client-side timeout. First call after idle includes loading ~6 GB of
# model into VRAM (tens of seconds) on top of generation — far above
# cloud-API latencies, same rationale as OpenAICompatAnswerer.
DEFAULT_TIMEOUT_S = 300

# The openai SDK refuses an empty api_key; Ollama ignores it entirely.
# A fixed non-secret placeholder — deliberately NOT env-configurable so
# nobody mistakes this vendor for a remote-endpoint client.
_PLACEHOLDER_API_KEY = "ollama"


def _to_openai_base_url(base_url: str) -> str:
    """Normalise an Ollama server URL to its OpenAI-compat ``/v1`` root.

    The env var documents the SERVER address (``http://localhost:11434``,
    matching Ollama's own docs); the OpenAI-compat API lives under ``/v1``.
    Appending here (idempotently) means users paste the address they know
    and never debug a doubled ``/v1/v1`` or a missing suffix.
    """
    trimmed = base_url.rstrip("/")
    return trimmed if trimmed.endswith("/v1") else f"{trimmed}/v1"


class OllamaVisionCaptioner:
    """VisionCaptioner that captions drawing pages via a local Ollama VLM."""

    backend: ClassVar[str] = "ollama-vision-captioner"

    def __init__(
        self,
        *,
        cache_dir: Path = Path("data/caption_cache"),
        model: str | None = None,
        base_url: str | None = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        client: OpenAI | None = None,
        timeout_s: float = DEFAULT_TIMEOUT_S,
        profile: DomainProfile | None = None,
    ) -> None:
        # None → resolve from env lazily so constructing the captioner never
        # reads the environment (tests inject a mock client and stay hermetic;
        # mirrors OpenAICompatAnswerer / DeepSeekVisionCaptioner).
        self._client = client
        self._cache_dir = cache_dir
        self._model = model
        self._base_url = base_url
        self._max_tokens = max_tokens
        self._timeout_s = timeout_s
        # Phase 7 SS4: caption prompt from the active DomainProfile (None →
        # construction default). Non-contract profile name folds into cache key.
        self._caption_prompt = profile.caption_prompt if profile else None
        self._profile_name = profile.name if profile and profile.name != "contract" else None

    def _resolved_model(self) -> str:
        """Model id: explicit constructor arg wins, else env / default."""
        return self._model if self._model is not None else get_ollama_vl_model()

    def _ensure_client(self) -> OpenAI:
        """Lazy-create the OpenAI client pointed at the local Ollama server."""
        if self._client is None:
            self._client = OpenAI(
                base_url=_to_openai_base_url(self._base_url or get_ollama_base_url()),
                api_key=_PLACEHOLDER_API_KEY,
                timeout=self._timeout_s,
            )
        return self._client

    def caption(self, image_bytes: bytes, nearby_text: str) -> DrawingCaption:
        """Caption a drawing page; NEVER raises (returns empty on any failure)."""
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        model = self._resolved_model()

        # Model + backend in the key so local-VLM captions don't collide
        # with the other captioners' cache in the same dir (same recipe as
        # ClaudeCli/DeepSeek captioners — the shared helpers own read/write,
        # each vendor owns its key).
        h = hashlib.sha256()
        h.update(image_bytes)
        h.update(model.encode("utf-8"))
        h.update(self.backend.encode("ascii"))
        if self._profile_name:  # non-contract only → existing cache preserved
            h.update(self._profile_name.encode("utf-8"))
        cache_key = h.hexdigest()
        cache_path = self._cache_dir / f"{cache_key}{CACHE_SUFFIX}"

        cached = read_caption_cache(cache_path)
        if cached is not None:
            logger.info("ollama_captioner.cache_hit", cache_key=cache_key[:12])
            return cached

        try:
            payload = self._call_caption_api(image_bytes, nearby_text, model)
        except Exception as exc:  # noqa: BLE001 — Protocol: never raise on one page
            logger.warning("ollama_captioner.api_error", error_type=type(exc).__name__)
            return DrawingCaption(caption_zh="", entities=[])

        write_caption_cache(cache_path, payload)
        return payload_to_caption(payload)

    def _call_caption_api(
        self, image_bytes: bytes, nearby_text: str, model: str
    ) -> dict[str, object]:
        """One OpenAI-compat vision call against Ollama → parsed payload."""
        image_b64 = base64.standard_b64encode(image_bytes).decode("ascii")
        client = self._ensure_client()
        prompt = build_caption_prompt(nearby_text, self._caption_prompt)

        # Vision content list: image before text, data-URI base64 — the
        # exact shape the DECISION-ls.20 probe validated against Ollama
        # 0.20.2 (and the same one DeepSeekVisionCaptioner sends).
        response = client.chat.completions.create(
            model=model,
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
        # Defensive inline-<think> strip (DECISION-ls.11): Ollama itself
        # keeps reasoning out of content, but other compat shims served at
        # the same base_url inline it, which would fail the JSON parse and
        # silently empty every caption.
        raw_text = _strip_think((content or "").strip())

        usage = getattr(response, "usage", None)
        logger.info(
            "ollama_captioner.api_complete",
            model=model,
            prompt_tokens=getattr(usage, "prompt_tokens", None),
            completion_tokens=getattr(usage, "completion_tokens", None),
            chars=len(raw_text),
        )
        return parse_caption_payload(raw_text)
