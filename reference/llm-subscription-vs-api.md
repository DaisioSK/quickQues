# Subscription-based LLM Access (Claude Code / Codex CLI) vs. API Keys

**Date stamped**: 2026-05-28. CLI tools update frequently — re-verify flags if reading after 2026-11.

## Two access models

| Mode | Auth | Billing | Setup |
|---|---|---|---|
| **API key** | `ANTHROPIC_API_KEY` env var (or `OPENAI_API_KEY` for codex API) | Per-token, on the API account | Sign up, generate key, set env var |
| **Subscription CLI** | OAuth via the user's account | Subscription quota (flat monthly fee) | Install vendor CLI, run `<vendor> login` |

For j-contract, both paths go through the same `Answerer` Protocol. Three impls today:

- `impls/claude_answerer.py` — Anthropic API direct (per-token)
- `impls/claude_cli_answerer.py` — `claude -p` subprocess (Claude Code subscription)
- `impls/codex_cli_answerer.py` — `codex exec` subprocess (ChatGPT subscription; skeleton)

Wire via `--answerer claude-api | claude-cli | codex-cli` on the `evaluate` command.

## Why the subscription path matters

For development / evaluation iteration:
- A single `evaluate` run = 6 golden cases × 1 LLM call each ≈ $0.07 on API
- Twenty iterations = $1.40 — still cheap, but adds up
- Subscription mode: $0 marginal, just counts against monthly quota

For production / many users:
- API: scales with usage, no upper bound
- Subscription: quota cap; needs orchestration if exceeded

j-contract is currently single-user, so subscription mode is a strict cost win when the user already has a Claude Code Max ($100/mo) or ChatGPT Plus ($20/mo) plan.

## Claude Code CLI (`claude -p`) — confirmed working

### Invocation pattern (what `impls/claude_cli_answerer.py` does)

```bash
claude -p "<user message>" \
  --model sonnet \
  --output-format json \
  --system-prompt "<our citation-strict system prompt>" \
  --permission-mode bypassPermissions \
  --no-session-persistence \
  --setting-sources "" \
  --tools "" \
  --disable-slash-commands
```

### Output shape (--output-format json)

```json
{
  "type": "result",
  "subtype": "success",
  "is_error": false,
  "result": "The model's text response goes here.",
  "stop_reason": "end_turn",
  "usage": {
    "input_tokens": 100,
    "output_tokens": 50,
    "cache_creation_input_tokens": 5000,
    "cache_read_input_tokens": 0
  },
  "total_cost_usd": 0.0082,
  "modelUsage": {...},
  "session_id": "..."
}
```

`total_cost_usd` is reported even for subscription users — it represents the **equivalent API cost**, not actual billing. Useful for tracking quota burn.

### Key gotchas

1. **CLAUDE.md / project context bleeds in by default** — the CLI auto-discovers `CLAUDE.md` files from the current working directory. We override the system prompt entirely (`--system-prompt ...`, REPLACE not `--append-system-prompt`) to neutralize this.
2. **OAuth state lives in the OS keychain** — if `claude login` hasn't been run, calls fail. Our impl returns a canonical fallback rather than raising; user sees "文档中未明确说明。" with low confidence.
3. **--bare mode forces API key auth** — counterintuitive; "minimal mode" disables OAuth/keychain reads. For subscription path, do NOT use `--bare`.
4. **~6000 cache_creation tokens per call** — the CLI loads its own system prompt overhead. Cheap on subscription (cache is reused across calls), but worth knowing for API-key billing.
5. **Model aliases**: `sonnet` / `haiku` / `opus` resolve to the latest snapshot at call time. Pin a specific snapshot (e.g. `claude-sonnet-4-5-20250929`) for reproducibility in production.

### Useful flags we don't yet use

- `--max-budget-usd <N>` — hard cap dollar spend (API-key users only)
- `--fallback-model <m>` — automatic model degradation on overload
- `--json-schema <s>` — structured output validation (would let us drop our post-hoc citation parsing)

## OpenAI Codex CLI (`codex exec`) — skeleton

`codex` is OpenAI's open-source equivalent (https://github.com/openai/codex). j-contract has the impl skeleton (`impls/codex_cli_answerer.py`) but **not validated locally** — the binary is not installed on this dev environment.

### Expected invocation (verify when you install)

```bash
codex exec \
  --model gpt-5 \
  --json \
  --no-color \
  --sandbox read-only \
  --full-auto \
  --system-prompt "<our system prompt>" \
  "<user message>"
```

### Expected output shapes

The impl handles both:
- Single JSON object with `message` or `result` field
- JSONL stream (newline-delimited events); we take the last `message`/`result`/`content`/`text` payload

If neither matches, the answerer falls back gracefully. Add a fixture to `tests/test_codex_cli_answerer.py` when you confirm the actual shape.

## Comparison

| Aspect | claude-api | claude-cli | codex-cli |
|---|---|---|---|
| Marginal cost | per token | $0 (within quota) | $0 (within quota) |
| Auth | `ANTHROPIC_API_KEY` env | `claude login` OAuth | `codex login` OAuth |
| Latency | ~2-5s | ~5-15s (CLI startup + sub-shell) | similar to claude-cli |
| Reliability for batch | High (direct HTTP) | Moderate (subprocess parsing) | Same as claude-cli |
| Reproducibility | Pinnable snapshot id | Alias drift possible | Same as claude-cli |
| Production fit | Recommended | Single-user / dev | Single-user / dev |

## When to use which

- **Dev iteration**: `claude-cli` (no marginal cost; iterate freely)
- **Heavy eval batches**: `claude-cli` if subscription tier allows; else `claude-api` with `--max-budget-usd` set
- **Production deployment**: `claude-api` with pinned model snapshot, real billing visibility, predictable latency
- **Mixed-vendor evaluation**: `codex-cli` to A/B against ChatGPT models (requires user to install + log in)

## Security notes

Neither CLI impl reads or stores secrets in our code:
- `claude` OAuth token lives in the user's OS keychain; the binary accesses it
- `codex` OAuth token similarly

Our code only:
- Resolves the binary path (`shutil.which`)
- Spawns subprocess with program-constructed argv (no `shell=True`)
- Parses stdout JSON

No env var with key material is read. No token persisted in app config. No subprocess output containing key material is logged (we log token counts only, never full bodies).

## Sources

- Claude Code CLI: built-in `claude --help` documentation; tested 2026-05-28 on v2.1.143
- OpenAI Codex CLI: https://github.com/openai/codex (verify version-specific flags when installing)
- Reference for output format and flags: `claude --help | less` (cached version-specific output not committed; re-run as docs evolve)
