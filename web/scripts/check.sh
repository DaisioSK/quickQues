#!/usr/bin/env bash
# Frontend three-piece gate — mirrors scripts/check.sh at the repo root.
#
# Per dev-contract §11.5: each language/stack in this project needs its
# own "lint + typecheck + test" triplet. For web/ that's eslint + tsc +
# vitest.
#
# Run from web/ directory: `bash scripts/check.sh`
# Or from repo root:       `bash web/scripts/check.sh` (auto-cd's below)

set -euo pipefail

# cd into web/ regardless of where this script is invoked from.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WEB_DIR="$(dirname "$SCRIPT_DIR")"
cd "$WEB_DIR"

echo "==> [1/3] eslint"
npm run lint --silent

echo "==> [2/3] tsc --noEmit"
npm run typecheck --silent

echo "==> [3/3] vitest run"
npm run test --silent

echo ""
echo "✓ Frontend gates green."
