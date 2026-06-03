#!/usr/bin/env bash
# Install git hooks tracked in scripts/hooks/ into .git/hooks/.
# Idempotent. Run once after cloning the repo.
set -euo pipefail

cd "$(dirname "$0")/.."

ln -sf ../../scripts/hooks/pre-commit .git/hooks/pre-commit
chmod +x scripts/hooks/pre-commit

echo "✓ Pre-commit hook installed."
