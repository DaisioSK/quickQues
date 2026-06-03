# DeepSeek V4 Vision — OCR via OpenAI-compatible API

> Stamp: 2026-05-29. Refetch if older than 6 months and you're making a cost
> or routing decision based on this file.

## Why this file exists

`src/jcontract/impls/deepseek_v4_parser.py` (Phase 1.10) is the fourth
PDFParser vendor. Whoever touches that file — or batch-ingest cost tuning
— needs the API shape, cost math, and trade-offs vs the other three
vendors in one place.

## Endpoint at a glance

| Field | Value |
|---|---|
| Base URL | `https://api.deepseek.com` (no `/v1` needed; OpenAI SDK appends `/chat/completions`) |
| Auth | `Authorization: Bearer $DEEPSEEK_API_KEY` (managed by openai SDK) |
| Model IDs | `deepseek-v4-flash` (default in our parser) · `deepseek-v4-pro` (quality upgrade) |
| Protocol | OpenAI ChatCompletions (also Anthropic-compatible per official docs) |
| Context | 128K tokens |
| Vision | Native multimodal — image_url content entries accepted alongside text |
| KV cache footprint | ~90 entries / image (vs ~870 on Claude — roughly an order cheaper) |

## Request payload (vision)

```python
client = openai.OpenAI(
    api_key=os.environ["DEEPSEEK_API_KEY"],
    base_url="https://api.deepseek.com",
)

resp = client.chat.completions.create(
    model="deepseek-v4-flash",
    max_tokens=2048,
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{image_b64}",
                        "detail": "high",  # OCR needs full-res scan
                    },
                },
                {"type": "text", "text": OCR_PROMPT},
            ],
        }
    ],
)
text = resp.choices[0].message.content
```

**Notes**

- `detail: "high"` is the difference between "OCR works" and "OCR drops small
  fonts" — `auto` (the default) is allowed to downscale to ~512 px. We
  already render at 150 DPI on our side; pay for high-detail.
- Response shape is OpenAI-standard: `.choices[0].message.content` is `str |
  None`. Compat shims occasionally emit `None` on truncation — guard before
  calling `.strip()`.
- `usage` field is present on the main API but may be absent on some
  redistributor shims; treat as optional telemetry.

## Cost — per-page OCR estimate

Numbers below are estimates for an A4 page rendered at 150 DPI JPEG q=85
(≈ 1240 × 1754 px, ≈ 200 KB), passing the TEXT_OCR_PROMPT through. Refresh
if the model prices change.

| Vendor / Model | Input | Output | Per page (USD) | Source / date |
|---|---|---|---|---|
| `deepseek-v4-flash` | ~1.5K tok | ~0.3K tok | **~$0.001–0.003** | DeepSeek pricing page 2026-05 |
| `deepseek-v4-pro` | ~1.5K tok | ~0.3K tok | ~$0.005–0.010 | same |
| `claude-sonnet-4-5` Vision | ~1.5K tok | ~0.3K tok | ~$0.012–0.027 | reference/claude-vision-ocr.md |
| `claude-cli-vision` (Max sub) | — | — | $0 marginal* | reference/llm-subscription-vs-api.md |
| `pypdf` (text-only) | n/a | n/a | $0 | n/a |

\* Claude Code subscription quota; marginal cost is $0 but quota is
finite. For 4100-page DEMO full-ingest, expect quota exhaustion in <1 PDF.

**Implication**: `deepseek-v4-flash` is the right default for full-batch
ingest of DEMO. ~$5–12 for all 4100 pages vs ~$50–100 on Claude Vision.
Upgrade to `deepseek-v4-pro` on PDFs where flash drops content.

## 4-vendor selection guide

| Situation | Recommended `--parser` |
|---|---|
| Text PDF (not scanned) | `pypdf` (free, instant) |
| One scanned PDF, quality matters | `claude-vision` (best OCR fidelity observed in Phase 1.5) |
| Many scanned PDFs, cost matters | `deepseek-v4` (3-5x cheaper than Claude flash-equivalent) |
| Active Claude Max/Pro subscriber, occasional ingest | `claude-cli-vision` (zero marginal cost via subscription quota) |
| Quality regression on `deepseek-v4-flash` for a specific PDF | Constructor override: `DeepSeekV4Parser(model="deepseek-v4-pro")` |

## Known unknowns (UNCERTAIN)

- **UNCERTAIN-1.10.1**: flash's OCR fidelity vs Claude Sonnet on DEMO
  scan quality. The 8-question dep check passes and the architecture is
  sound, but flash vs Claude head-to-head on the same 5 DEMO pages is
  deferred to the user's integration smoke (DECISION 2026-05-29). Update
  this file with the LESSON once measured.
- **UNCERTAIN-1.10.2**: whether DeepSeek requires the `/v1` suffix on
  base_url in some regions / shims. The parser today omits it (matches
  the public docs). If a 404 surfaces, change `DEEPSEEK_BASE_URL` in
  `deepseek_v4_parser.py` to `https://api.deepseek.com/v1`.

## Sources

- DeepSeek API docs portal — https://api-docs.deepseek.com/ (2026-04-26 V4
  preview release notes mention `deepseek-v4-pro` and `deepseek-v4-flash`
  as drop-in `model=` values)
- MindStudio "DeepSeek V4 Vision: 10x Cheaper Multimodal" blog (cites the
  ~90 vs ~870 KV cache entries per image figure)
- apiyi.com DeepSeek V4 multimodal model guide (chat.completions sample
  showing OpenAI-compatible client + base_url override)

## How to verify when this gets stale

```bash
# Spot-check current model lineup and prices:
curl -s https://api-docs.deepseek.com/ | grep -E "deepseek-v4-(flash|pro|vision)"
# Verify the SDK call shape still works:
DEEPSEEK_API_KEY=sk-... uv run python - <<'PY'
import openai
c = openai.OpenAI(api_key=__import__("os").environ["DEEPSEEK_API_KEY"],
                  base_url="https://api.deepseek.com")
r = c.chat.completions.create(model="deepseek-v4-flash", max_tokens=20,
    messages=[{"role":"user","content":"ping"}])
print(r.choices[0].message.content)
PY
```
