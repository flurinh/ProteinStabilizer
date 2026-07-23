#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENVIRONMENT="${1:-$ROOT/.venv-esmc6b}"
PYTHON="${PYTHON3_12:-python3.12}"

export UV_CACHE_DIR="${UV_CACHE_DIR:-/data/fast/cache/uv}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-/data/fast/cache/pip}"
export TMPDIR="${TMPDIR:-/data/fast/tmp/protein-stabilizer-venv}"
mkdir -p "$UV_CACHE_DIR" "$PIP_CACHE_DIR" "$TMPDIR"

if [[ ! -x "$ENVIRONMENT/bin/python" ]]; then
  if command -v uv >/dev/null 2>&1; then
    uv venv --python "$PYTHON" "$ENVIRONMENT"
  else
    "$PYTHON" -m venv "$ENVIRONMENT"
  fi
fi

if command -v uv >/dev/null 2>&1; then
  uv pip install \
    --python "$ENVIRONMENT/bin/python" \
    -r "$ROOT/requirements-esmc6b-lock.txt"
  uv pip install \
    --python "$ENVIRONMENT/bin/python" \
    -e "$ROOT" \
    --no-deps
else
  "$ENVIRONMENT/bin/pip" install \
    -r "$ROOT/requirements-esmc6b-lock.txt"
  "$ENVIRONMENT/bin/pip" install -e "$ROOT" --no-deps
fi

"$ENVIRONMENT/bin/python" -c \
  'import Bio, torch, transformers; print(f"6B environment ready: torch={torch.__version__}, transformers={transformers.__version__}, biopython={Bio.__version__}")'
