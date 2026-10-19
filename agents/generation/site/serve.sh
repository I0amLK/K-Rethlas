#!/usr/bin/env bash
set -euo pipefail

SITE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GEN_DIR="$(cd "$SITE_DIR/.." && pwd)"
PYTHON="${RETHLAS_PYTHON:-$GEN_DIR/../../.venv/bin/python}"
PORT="${PORT:-3264}"
SYNC_ONLY="${SYNC_ONLY:-0}"
if [[ "$SYNC_ONLY" != 0 && "$SYNC_ONLY" != 1 ]]; then
  printf 'SYNC_ONLY must be 0 or 1\n' >&2
  exit 1
fi
if [[ "$SYNC_ONLY" == 0 && ! -d "$SITE_DIR/themes/MATbook" ]]; then
  printf 'Theme is absent. Review and run site/setup_theme.sh separately if desired.\n' >&2
  exit 1
fi
# No automatic downloads and no filename-based review promotion. Only previously
# indexed generated pages may be removed; hand-authored content is preserved.
"$PYTHON" -I -m rethlas_core.cli site-sync \
  --results "$GEN_DIR/results" --content "$SITE_DIR/content"
if [[ "$SYNC_ONLY" == 1 ]]; then
  exit 0
fi
cd "$SITE_DIR"
exec zola serve --port "$PORT" --interface 127.0.0.1
