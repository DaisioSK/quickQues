"""Judge Protocol + JudgeScore — Layer 0 (Enhancement E12).

Why this exists:
- The eval pipeline (eval/metrics.py) scores retrieval (recall@k) and a
  couple of weak answer proxies (citation_accuracy, keyword substring).
  To judge ANSWER QUALITY we need an LLM-as-judge: an evaluator that
  reads the question / answer / retrieved context and grades it.
- Two reference-FREE metrics are buildable without a gold answer:
    * faithfulness   — is every claim in the answer supported by the
                       retrieved context? (catches hallucination)
    * answer_relevancy — does the answer actually address the question?
  A reference-based ``correctness(answer, expected_answer)`` is a planned
  N=2 addition once golden answers arrive (FORESHADOW; see DECISION-e12.1).

Default impl: impls/claude_cli_judge.py (subscription, zero API key).
Replacement candidates (N=2): an API judge, or embedding-similarity for
the reference-based correctness baseline (DECISION-e12.1 layering).

Design note — failure handling:
- Judges run offline during eval; a single failed grade must not abort a
  whole run NOR be silently counted as 0.0 (that would fake a regression).
  Impls MUST NOT raise; on any failure they return a JudgeScore whose
  ``score`` is NaN. The eval runner treats a NaN score as "not measured"
  and excludes it from the aggregate (same as a metric simply absent).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from jcontract.interfaces.schema import Chunk


@dataclass(frozen=True)
class JudgeScore:
    """A single graded judgement.

    ``score`` is in [0.0, 1.0] on success, or NaN when the judge could not
    produce a grade (call failed / unparseable output) — see the module
    docstring. ``reasoning`` is a short human-readable justification (or an
    error note when score is NaN).
    """

    score: float
    reasoning: str


class Judge(Protocol):
    """Grade an answer's quality. Impls MUST NOT raise (NaN score on failure)."""

    def faithfulness(self, answer: str, context: list[Chunk]) -> JudgeScore:
        """1.0 = every claim in ``answer`` is supported by ``context``; 0.0 = hallucinated."""
        ...

    def answer_relevancy(self, question: str, answer: str) -> JudgeScore:
        """1.0 = ``answer`` directly addresses ``question``; 0.0 = off-topic/evasive."""
        ...
