#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUTPUT="${1:-$ROOT/artifacts/examples/human_melanopsin/accuracy_screen.csv}"

export HF_HOME="${HF_HOME:-/data/fast/cache/huggingface}"
export TMPDIR="${TMPDIR:-/data/fast/tmp/protein-stabilizer}"
mkdir -p "$HF_HOME" "$TMPDIR" "$(dirname "$OUTPUT")"

cd "$ROOT"
"$ROOT/.venv/bin/protein-stabilizer" screen-accuracy \
  --fasta "$ROOT/examples/human_melanopsin/Q9UHM6.fasta" \
  --uniprot Q9UHM6 \
  --alphafold-min-plddt "${ALPHAFOLD_MIN_PLDDT:-70}" \
  --protected-mask \
    "$ROOT/examples/human_melanopsin/protected_positions.txt" \
  --require-component-agreement \
  --max-per-site "${MAX_PER_SITE:-2}" \
  --top "${TOP:-50}" \
  --max-tokens "${MAX_TOKENS:-8192}" \
  --max-batch-size "${MAX_BATCH_SIZE:-128}" \
  --output "$OUTPUT"
