#!/usr/bin/env bash
# ============================================================
# j-contract three-piece quality gate
# ============================================================
# Per dev-contract/25-domain-quality-gates.md §合流 main 前体检
# and docs/project_guideline.md §7.
#
# Exit 0 only if all four checks pass:
#   1. ruff check
#   2. ruff format --check
#   3. mypy
#   4. pytest
#
# Run before every squash merge to main.
# ============================================================

set -euo pipefail

cd "$(dirname "$0")/.."

echo "==> [1/4] ruff check ."
uv run ruff check .

echo "==> [2/4] ruff format --check ."
uv run ruff format --check .

echo "==> [3/4] mypy ."
uv run mypy .

echo "==> [4/4] pytest"
uv run pytest

echo ""
echo "✓ All gates green."
