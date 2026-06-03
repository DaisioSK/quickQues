"""Evaluation pipeline: golden cases + metrics + runner.

metrics.py: pure functions for Recall@k, citation accuracy, keyword hit rate.
runner.py: orchestrates over EvalCase list, calls injected search/answer
funcs, writes timestamped JSON to data/eval-results/.
golden_cases.jsonl: hand-curated test set (one JSON object per line).
"""
