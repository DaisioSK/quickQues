"""ClaudeCliJudge — LLM-as-judge via ``claude -p`` subprocess (Enhancement E12).

Grades answer quality (faithfulness, answer relevancy) using the user's
Claude Code subscription — NO API key (matches the project's no-key
defaults: claude-cli answerer / captioner / vision parser).

Why subprocess + JSON: same proven path as ClaudeCliAnswerer; we reuse the
shared ``run_claude_text`` runner. Each grade asks the model for a strict
JSON ``{"score": <0..1>, "reasoning": "<short>"}`` and parses defensively.

Failure contract (see interfaces/judge.py): NEVER raises. Any failure
(call error, non-JSON, missing score) returns ``JudgeScore(nan, ...)`` so
the eval runner excludes it from the aggregate rather than scoring it 0.0
(which would fake a regression).

Secret handling: NONE — OAuth via the `claude` binary, never read here.
"""

from __future__ import annotations

import json
import logging
import math
import shutil
from typing import ClassVar

import structlog

from jcontract.impls._claude_cli_runner import run_claude_text
from jcontract.interfaces import Chunk, JudgeScore

logger = structlog.get_logger(__name__)

logging.getLogger("pypdfium2").setLevel(logging.WARNING)

DEFAULT_MODEL = "sonnet"  # CLI alias → latest Sonnet; judging wants quality.
DEFAULT_TIMEOUT_S = 120
# Cap context fed to the faithfulness judge — title block + a few chunks is
# enough to check grounding; keeps the prompt (and quota) bounded.
_MAX_CONTEXT_CHARS = 6000

_NAN = float("nan")

_FAITHFULNESS_PROMPT = """\
You are a strict RAG evaluator. Decide how FAITHFUL the ANSWER is to the CONTEXT:
does every factual claim in the ANSWER appear in / follow from the CONTEXT?
- 1.0 = fully grounded, no claim outside the context.
- 0.0 = the answer asserts facts absent from the context (hallucination).
Judge grounding ONLY, not whether the answer is correct in the real world.

<context>
{context}
</context>
<answer>
{answer}
</answer>

Output ONLY a JSON object: {{"score": <float 0.0-1.0>, "reasoning": "<one short sentence>"}}.
No markdown fences, no prose before or after."""

_RELEVANCY_PROMPT = """\
You are a strict RAG evaluator. Decide how RELEVANT the ANSWER is to the QUESTION:
does it actually address what was asked? Judge relevancy/on-topic-ness ONLY,
not correctness and not grounding.
- 1.0 = directly answers the question.
- 0.0 = off-topic, evasive, or answers a different question.

<question>
{question}
</question>
<answer>
{answer}
</answer>

Output ONLY a JSON object: {{"score": <float 0.0-1.0>, "reasoning": "<one short sentence>"}}.
No markdown fences, no prose before or after."""


def _format_context(context: list[Chunk]) -> str:
    """Render chunks as ``[file p.N] text`` lines, truncated to a char budget."""
    parts: list[str] = []
    used = 0
    for c in context:
        line = f"[{c.file} p.{c.page}] {c.text}"
        if used + len(line) > _MAX_CONTEXT_CHARS:
            parts.append(line[: _MAX_CONTEXT_CHARS - used])
            break
        parts.append(line)
        used += len(line)
    return "\n\n".join(parts)


def _parse_judge_json(raw_text: str) -> JudgeScore:
    """Parse a model's raw grade into a JudgeScore. NaN score on any defect."""
    cleaned = raw_text.strip()
    cleaned = cleaned.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        return JudgeScore(score=_NAN, reasoning="judge: non-JSON output")
    if not isinstance(parsed, dict) or "score" not in parsed:
        return JudgeScore(score=_NAN, reasoning="judge: unexpected JSON shape")
    try:
        score = float(parsed["score"])
    except (TypeError, ValueError):
        return JudgeScore(score=_NAN, reasoning="judge: non-numeric score")
    # Clamp into [0, 1] — models occasionally emit 0-100 or slight overshoots.
    score = max(0.0, min(1.0, score))
    reasoning = str(parsed.get("reasoning", ""))
    return JudgeScore(score=score, reasoning=reasoning)


class ClaudeCliJudge:
    """LLM-as-judge over the ``claude`` CLI (subscription, zero key)."""

    backend: ClassVar[str] = "claude-cli-judge"

    def __init__(
        self,
        *,
        model: str = DEFAULT_MODEL,
        claude_path: str | None = None,
        timeout_s: int = DEFAULT_TIMEOUT_S,
    ) -> None:
        resolved = claude_path or shutil.which("claude")
        if not resolved:
            raise RuntimeError(
                "claude CLI not found in PATH. "
                "Install Claude Code (https://docs.claude.com/en/docs/claude-code) "
                "and run `claude login`."
            )
        self._claude_path = resolved
        self._model = model
        self._timeout_s = timeout_s

    def faithfulness(self, answer: str, context: list[Chunk]) -> JudgeScore:
        prompt = _FAITHFULNESS_PROMPT.format(context=_format_context(context), answer=answer)
        return self._grade(prompt, metric="faithfulness")

    def answer_relevancy(self, question: str, answer: str) -> JudgeScore:
        prompt = _RELEVANCY_PROMPT.format(question=question, answer=answer)
        return self._grade(prompt, metric="answer_relevancy")

    def _grade(self, prompt: str, *, metric: str) -> JudgeScore:
        """Run one judge call; NEVER raises (NaN JudgeScore on failure)."""
        try:
            data = run_claude_text(
                claude_path=self._claude_path,
                prompt=prompt,
                model=self._model,
                timeout_s=self._timeout_s,
            )
        except Exception as exc:  # noqa: BLE001 — judge must not abort the eval run
            logger.warning("judge.error", metric=metric, error_type=type(exc).__name__)
            return JudgeScore(score=_NAN, reasoning=f"judge unavailable: {type(exc).__name__}")

        raw_text = str(data.get("result", ""))
        result = _parse_judge_json(raw_text)
        logger.info(
            "judge.graded",
            metric=metric,
            score=None if math.isnan(result.score) else result.score,
        )
        return result
