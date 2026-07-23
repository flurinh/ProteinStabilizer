#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BACKBONE="${1:-600m}"
OUTPUT="${2:-$ROOT/artifacts/examples/human_melanopsin/${BACKBONE}_screen.csv}"
FASTA="$ROOT/examples/human_melanopsin/Q9UHM6.fasta"
MASK="$ROOT/examples/human_melanopsin/protected_positions.txt"

export HF_HOME="${HF_HOME:-/data/fast/cache/huggingface}"
export TMPDIR="${TMPDIR:-/data/fast/tmp/protein-stabilizer}"
mkdir -p "$HF_HOME" "$TMPDIR" "$(dirname "$OUTPUT")"

case "$BACKBONE" in
  600m)
    EXECUTABLE="$ROOT/.venv/bin/protein-stabilizer"
    COMMAND="screen-v2"
    MAX_TOKENS="${MAX_TOKENS:-8192}"
    MAX_BATCH_SIZE="${MAX_BATCH_SIZE:-128}"
    ;;
  6b)
    EXECUTABLE="$ROOT/.venv-esmc6b/bin/protein-stabilizer"
    COMMAND="screen-v2-6b"
    MAX_TOKENS="${MAX_TOKENS:-4096}"
    MAX_BATCH_SIZE="${MAX_BATCH_SIZE:-2}"
    ;;
  *)
    echo "usage: $0 [600m|6b] [output.csv]" >&2
    exit 2
    ;;
esac

if [[ ! -x "$EXECUTABLE" ]]; then
  echo "missing executable: $EXECUTABLE" >&2
  if [[ "$BACKBONE" == "6b" ]]; then
    echo "create it with: bash scripts/setup_esmc6b_env.sh" >&2
  fi
  exit 1
fi

cd "$ROOT"
"$EXECUTABLE" "$COMMAND" \
  --fasta "$FASTA" \
  --protected-mask "$MASK" \
  --scan-mode two-stage \
  --rerank-top "${RERANK_TOP:-128}" \
  --max-per-site "${MAX_PER_SITE:-2}" \
  --top "${TOP:-50}" \
  --topology alpha_helical_gpcr \
  --max-tokens "$MAX_TOKENS" \
  --max-batch-size "$MAX_BATCH_SIZE" \
  --output "$OUTPUT"
