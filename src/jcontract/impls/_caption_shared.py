"""Shared caption prompt + JSON-payload parsing for the VisionCaptioner impls.

Enhancement E11 (``p2-ssMultiVendorCaptioner``) adds a second and third
captioner (``ClaudeCliVisionCaptioner`` subscription, ``DeepSeekVisionCaptioner``
API) alongside the original ``ClaudeVisionCaptioner``. All three want the
SAME drawing-caption prompt and the SAME defensive JSON parsing (models
occasionally wrap output in ``` fences or emit the wrong shape). Per
project_guideline.md §5 (N=2 → upgrade the shared piece, don't fork a
parallel wheel) that lives here, consumed by every captioner.

DECISION-2.cap.1 (docs/dev-sprint.md): JSON-only output → stable parsing +
structured entities; any malformed output degrades to an empty caption so
ingest never loses a page (VisionCaptioner Protocol contract).
"""

from __future__ import annotations

import json
from pathlib import Path

import structlog

from jcontract.interfaces import DrawingCaption

logger = structlog.get_logger(__name__)

# Cache filename suffix shared by every captioner — JSON because we
# round-trip the DrawingCaption payload. Lives in caption_cache/ dirs.
CACHE_SUFFIX = ".caption.json"

# Truncate nearby grounding text: the captioner needs the title block +
# a few adjacent paragraphs, not the whole page. Bounds per-call cost.
NEARBY_TEXT_LIMIT = 1500

# Drawing caption prompt — JSON-only output.
#
# Why JSON: DECISION-2.cap.1. Why 80-200 字: short captions don't justify
# the model call; long captions add embedding noise. Why no "这张图"
# preamble: it wastes tokens and dilutes retrieval signal.
CAPTION_PROMPT = """\
You are describing an engineering drawing from a construction tender PDF.

Output a JSON object with EXACTLY these two keys:
  "caption_zh": 80-200 字的中文图说。描述这张图的主题、关键尺寸/材料/构造方式、与其他图的关联。\
直接说内容，不要"这张图""图中显示"之类的废话开头。
  "entities": 字符串列表（list[str]），drawing 上出现的 Drawing No. (如 T/PRJ/CWD/WS/2101A)、\
Clause 引用（如 Clause 7.3）、关键术语和关键尺寸值。

The drawing's nearby OCR'd text is provided for grounding:
<nearby_text>
{nearby_text}
</nearby_text>

Return ONLY a single JSON object. No markdown code fences. No commentary. \
No prose before or after the JSON.
If the image is not actually a drawing or is unreadable, return: \
{{"caption_zh": "", "entities": []}}"""


def build_caption_prompt(nearby_text: str, template: str | None = None) -> str:
    """Fill a caption prompt template with trimmed nearby grounding text.

    The nearby text is wrapped in a ``<nearby_text>`` tag inside the
    prompt template — prompt-injection hardening so OCR'd "ignore all
    instructions" text reads as quoted contract content, not a directive
    (same pattern as answer/prompt.py's context wrapper).

    Phase 7 SS4: ``template`` comes from the active DomainProfile's
    ``caption_prompt``; None → the construction default CAPTION_PROMPT
    (which the contract profile reproduces byte-for-byte).
    """
    tmpl = template if template is not None else CAPTION_PROMPT
    return tmpl.format(nearby_text=(nearby_text or "")[:NEARBY_TEXT_LIMIT])


def parse_caption_payload(raw_text: str) -> dict[str, object]:
    """Parse a model's raw caption output into a normalised payload dict.

    Returns ``{"caption_zh": str, "entities": list}``. On any defect —
    non-JSON, ``` fences, wrong shape, non-list entities — returns the
    empty payload ``{"caption_zh": "", "entities": []}`` so every caller
    handles exactly one branch.
    """
    # Strip one layer of ```json / ``` fences defensively — the prompt
    # forbids them but some models emit them anyway.
    cleaned = raw_text.strip()
    cleaned = cleaned.removeprefix("```json").removeprefix("```").removesuffix("```").strip()

    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        return {"caption_zh": "", "entities": []}

    # Valid JSON of the wrong shape (e.g. a list) → empty.
    if not isinstance(parsed, dict) or "caption_zh" not in parsed:
        return {"caption_zh": "", "entities": []}

    entities = parsed.get("entities", [])
    if not isinstance(entities, list):
        entities = []

    return {"caption_zh": str(parsed.get("caption_zh", "")), "entities": entities}


def payload_to_caption(payload: dict[str, object]) -> DrawingCaption:
    """Build a DrawingCaption from a normalised payload dict."""
    entities = payload.get("entities", [])
    entity_list = [str(e) for e in entities] if isinstance(entities, list) else []
    return DrawingCaption(caption_zh=str(payload.get("caption_zh", "")), entities=entity_list)


def read_caption_cache(cache_path: Path) -> DrawingCaption | None:
    """Return the cached DrawingCaption, or None on miss / corrupt entry.

    A corrupt cache file (truncated write, manual edit) logs a warning and
    returns None so the caller re-fetches rather than crashing the ingest.
    """
    if not cache_path.exists():
        return None
    try:
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise TypeError("cache payload is not a dict")
        return payload_to_caption(payload)
    except (json.JSONDecodeError, TypeError, OSError):
        logger.warning("captioner.cache_corrupt", cache_key=cache_path.stem[:12])
        return None


def write_caption_cache(cache_path: Path, payload: dict[str, object]) -> None:
    """Persist a caption payload (even an empty one — saves $ on re-ingest)."""
    cache_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
