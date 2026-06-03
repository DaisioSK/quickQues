# Claude Vision OCR — Cheatsheet

**Date stamped**: 2026-05-28. Re-fetch from [Anthropic Vision docs](https://platform.claude.com/docs/en/docs/build-with-claude/vision) if reading after 2026-11 — pricing and model snapshots change.

## TL;DR

Claude Vision is well-suited for OCRing scanned PDF pages: high accuracy, structure-preserving, supports messy real-world documents (handwritten annotations, tables, drawings). Worse than dedicated OCR on cost per page; **far better on layout + semantic understanding** (e.g. preserving Q&A structure, table cells, drawing labels).

Use this for j-contract when text-only parsers (pypdf) return empty pages from scanned PDFs.

## Token / cost math

**Image tokens** ≈ `width × height / 750` (clamped to model's max).

For Claude Sonnet 4.6 / 4.5 at **$3 per million input tokens**:

| Page render size | Tokens | Cost / page | Cost / 100 pages |
|---|---|---|---|
| 1000×1000 (1 MP) | ~1334 | ~$0.004 | ~$0.40 |
| 1092×1092 (1.19 MP) | ~1568 (cap on Sonnet) | ~$0.0047 | ~$0.47 |
| 1920×1080 → resized → 1568 long edge | ~1568 (cap) | ~$0.0047 | ~$0.47 |

Larger images are **auto-downscaled** to 1568 px long edge on Sonnet — sending bigger does NOT improve OCR, just wastes upload bandwidth.

For Opus 4.7 at **$5/M tokens**, the cap is 2576 px / ~4784 tokens / ~$0.024 per page. Higher fidelity for tiny text or dense drawings, but 5× the cost. Default to Sonnet unless eval shows Sonnet missing critical details.

**Output tokens** (the extracted text): for an A4 page of contract text, expect 500-1500 output tokens. At ~$15/M output → $0.0075-$0.023 per page.

**Total** per page ≈ **$0.012-$0.027** for Sonnet, including input + output.
For j-contract's 9-part DEMO PDF set (~600 pages total): **~$10-15 for full ingest**.

## Recommended render settings (j-contract)

```python
# pypdfium2 render config for Claude Vision OCR
DPI = 150           # 150 DPI ≈ 1240x1754 px for A4 → auto-downscales to 1568 cap
                    # 200 DPI gives no benefit, just larger upload
FORMAT = "JPEG"     # lossy but small; quality=85 is the sweet spot
QUALITY = 85        # heavy compression (<70) hurts OCR accuracy
```

Don't render above 200 DPI — Sonnet downscales to 1568 px long edge anyway.

## API request shape (Anthropic SDK)

```python
from anthropic import Anthropic
import base64

client = Anthropic()

with open("page.jpg", "rb") as f:
    image_b64 = base64.standard_b64encode(f.read()).decode("utf-8")

response = client.messages.create(
    model="claude-sonnet-4-5",  # alias; pin snapshot for prod
    max_tokens=2048,
    messages=[
        {
            "role": "user",
            "content": [
                # Per docs: place image BEFORE text for best results.
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/jpeg",
                        "data": image_b64,
                    },
                },
                {
                    "type": "text",
                    "text": "<your OCR prompt here>",
                },
            ],
        }
    ],
)
```

**Limits** (Sonnet, single-message):
- Max 100 images per request (200k context model)
- Max 8000×8000 px per image (>20 images → 2000×2000 cap)
- Request size cap: 32 MB total
- Practical batch: 1 page per request — gives best output structure + easy error handling

## OCR prompt template (j-contract default)

```
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

If the page is blank or contains no text, return exactly: <empty page>
```

Why this prompt:
- "Return ONLY the extracted text" — prevents the model from adding "Here's the text from the page..." preamble.
- Explicit preservation list — Drawing No., Clause, Q&A markers are what downstream `qa_chunker.py` regex relies on.
- Sentinel `<empty page>` — lets the parser code detect and skip safely without parsing empty-string ambiguity.

## Image format choices

| Format | Pro | Con |
|---|---|---|
| **JPEG** (q=85) | Small payload, fast upload | Lossy — can hurt OCR on small text. q≥85 is safe. |
| **PNG** | Lossless | 5-10× larger than JPEG; only worth it if pages have very small text or fine line art |
| **WebP** | Smallest at same quality | Anthropic accepts it but less ecosystem support |
| **GIF** | Avoid for OCR — small palette destroys text edges | — |

j-contract default: **JPEG q=85**. Switch to PNG if eval shows OCR errors on small text.

## Pitfalls

1. **Don't send the full PDF as one big request.** Process page-by-page; gives clean error handling, allows caching, fits within 32 MB request cap.
2. **Don't skip caching.** Same page rendered twice = wasted API call. Use SHA-256 of the rendered image bytes as cache key.
3. **Don't downscale before sending if your PDF is already 150-200 DPI.** Sonnet does its own resize; sending smaller does NOT save tokens (token count is computed AFTER resize to the 1568 cap).
4. **Watch for rate limits.** Sonnet has tier-dependent rate limits. For full DEMO ingest (~600 pages) consider rate-limit-aware batching with retry+backoff.
5. **Always set `max_tokens` generous enough for the page.** A dense A4 page can produce 2000+ output tokens. Truncated outputs lose the bottom of the page silently.

## j-contract impl reference

Phase 1.5 implementation lives at `src/jcontract/impls/claude_vision_parser.py` (added in sub-sprint p1.5-ssOCR).

Page rendering: `pypdfium2` (pure-Python pdfium bindings, no system deps).

Cache: `data/ocr_cache/<sha256_of_image>.txt` — survives across re-ingest runs.

## Sources

- [Anthropic Vision docs](https://platform.claude.com/docs/en/docs/build-with-claude/vision) — fetched 2026-05-28, full HTML saved at `reference/_raw/claude-vision-docs.html` (do not commit; large) — relevant excerpt cached above
- [Anthropic pricing](https://claude.com/pricing) — Sonnet 4.6: $3/M input, ~$15/M output (verify before high-volume use)
- Phase 1 dev_log: `FORESHADOW-1.1.1` is what triggered this sub-sprint
