#!/usr/bin/env bash
set -euo pipefail

GEN_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${RETHLAS_PYTHON:-$GEN_DIR/../../.venv/bin/python}"
PROBLEM_FILE="${PROBLEM_FILE:-data/example.md}"
DRY_RUN="${DRY_RUN:-0}"
if [[ "$DRY_RUN" != 0 && "$DRY_RUN" != 1 ]]; then
  printf 'DRY_RUN must be 0 or 1\n' >&2
  exit 1
fi
if [[ ! -x "$PYTHON" ]]; then
  printf 'Install the shared controller package in the dedicated runtime first.\n' >&2
  exit 69
fi
args=(--generation-root "$GEN_DIR" --problem "$PROBLEM_FILE")
if [[ "$DRY_RUN" == 1 ]]; then
  args+=(--dry-run)
fi
# M1 preserves read-only path/log/resume inspection, but deliberately has no
# live Codex path until isolation and a trusted provider broker are available.
exec "$PYTHON" -I -m rethlas_core.cli launch "${args[@]}"
