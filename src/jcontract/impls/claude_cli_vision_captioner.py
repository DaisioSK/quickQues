"""ClaudeCliVisionCaptioner — VisionCaptioner via ``claude -p`` subprocess.

Enhancement E11 (``p2-ssMultiVendorCaptioner``). Second VisionCaptioner
impl: produces the same DrawingCaption JSON as ClaudeVisionCaptioner but
through the user's Claude Code subscription (``claude login`` OAuth) so it
needs NO ANTHROPIC_API_KEY. This is the default caption backend — it
matches the project's no-key default (claude-cli-vision parser).

Triggered by 2026-05-30 user testing ("图信息几乎 0 理解"): drawing
chunks need a caption to be retrievable, but the only captioner so far
required an API key. This unblocks captioning for subscription users.

Architecture (mirrors ClaudeCliVisionParser):
- Write the rendered image bytes to a file under ``render_dir`` so the
  ``claude`` CLI can Read it (the dir is whitelisted via --add-dir).
- Delegate the subprocess + JSON-envelope handling to the shared
  ``run_claude_read_image`` runner.
- Prompt + parse + cache come from the shared caption helpers, so all
  three captioners stay calibrated together (N=2 upgrade per §5).

Cost: $0 marginal (subscription quota); ~10-15s per drawing page.
Secret handling: NO API key read/stored/transmitted — OAuth via the CLI.
"""

from __future__ import annotations

import hashlib
import logging
import shutil
from pathlib import Path
from typing import ClassVar

import structlog

from jcontract.impls._caption_shared import (
    CACHE_SUFFIX,
    build_caption_prompt,
    parse_caption_payload,
    payload_to_caption,
    read_caption_cache,
    write_caption_cache,
)
from jcontract.impls._claude_cli_runner import run_claude_read_image
from jcontract.interfaces import DomainProfile, DrawingCaption

logger = structlog.get_logger(__name__)

logging.getLogger("pypdfium2").setLevel(logging.WARNING)

# Sonnet (not haiku) is the default here: drawing understanding is the
# whole point of captioning (E11 trigger), and the subscription path has
# no per-token cost. Override via constructor if quota is tight.
DEFAULT_MODEL = "sonnet"
DEFAULT_TIMEOUT_S = 180

# Read instruction prepended to the shared caption prompt — tells Claude
# Code which file to load before producing the caption JSON.
_READ_PREFIX = "Use the Read tool to open the image at: {image_path}\n\n"


class ClaudeCliVisionCaptioner:
    """VisionCaptioner that captions drawing pages via the ``claude`` CLI."""

    backend: ClassVar[str] = "claude-cli-vision-captioner"

    def __init__(
        self,
        *,
        cache_dir: Path = Path("data/caption_cache"),
        render_dir: Path = Path("data/_render_tmp"),
        model: str = DEFAULT_MODEL,
        claude_path: str | None = None,
        timeout_s: int = DEFAULT_TIMEOUT_S,
        profile: DomainProfile | None = None,
    ) -> None:
        # Resolve binary at __init__ so we fail loud if Claude Code is missing.
        resolved = claude_path or shutil.which("claude")
        if not resolved:
            raise RuntimeError(
                "claude CLI not found in PATH. "
                "Install Claude Code (https://docs.claude.com/en/docs/claude-code) "
                "and run `claude login`."
            )
        self._claude_path = resolved
        self._cache_dir = cache_dir
        self._render_dir = render_dir
        self._model = model
        self._timeout_s = timeout_s
        # Phase 7 SS4: caption prompt from the active DomainProfile (None →
        # construction default). Non-contract profile name folds into cache key.
        self._caption_prompt = profile.caption_prompt if profile else None
        self._profile_name = profile.name if profile and profile.name != "contract" else None

    def caption(self, image_bytes: bytes, nearby_text: str) -> DrawingCaption:
        """Caption a drawing page; NEVER raises (returns empty on any failure)."""
        self._cache_dir.mkdir(parents=True, exist_ok=True)

        # Cache key includes model + backend so subscription captions don't
        # collide with the API captioner's cache in the same dir.
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
            logger.info("cli_captioner.cache_hit", cache_key=cache_key[:12])
            return cached

        try:
            payload = self._call_claude_cli(image_bytes, nearby_text, cache_key)
        except Exception as exc:  # noqa: BLE001 — Protocol: never raise on one page
            logger.warning("cli_captioner.error", error_type=type(exc).__name__)
            return DrawingCaption(caption_zh="", entities=[])

        write_caption_cache(cache_path, payload)
        return payload_to_caption(payload)

    def _call_claude_cli(
        self, image_bytes: bytes, nearby_text: str, cache_key: str
    ) -> dict[str, object]:
        """Write the image, run ``claude -p``, parse the caption JSON."""
        self._render_dir.mkdir(parents=True, exist_ok=True)
        image_path = self._render_dir / f"{cache_key}.caption.jpg"
        image_path.write_bytes(image_bytes)

        try:
            prompt = _READ_PREFIX.format(image_path=image_path.resolve())
            prompt += build_caption_prompt(nearby_text, self._caption_prompt)
            data = run_claude_read_image(
                claude_path=self._claude_path,
                render_dir=self._render_dir,
                prompt=prompt,
                model=self._model,
                timeout_s=self._timeout_s,
            )
        finally:
            # Render is transient — drop it whether or not the call succeeded.
            image_path.unlink(missing_ok=True)

        raw_text = str(data.get("result", ""))
        usage = data.get("usage", {})
        usage = usage if isinstance(usage, dict) else {}
        logger.info(
            "cli_captioner.complete",
            input_tokens=usage.get("input_tokens"),
            output_tokens=usage.get("output_tokens"),
            metered_cost_usd=data.get("total_cost_usd"),  # 0 for subscription
            chars=len(raw_text),
        )
        return parse_caption_payload(raw_text)
