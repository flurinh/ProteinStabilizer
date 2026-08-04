#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUTPUT="${1:-$ROOT/artifacts/examples/human_melanopsin/accuracy_6b_screen.csv}"
MASKED_CACHE="${MASKED_CACHE:-$ROOT/artifacts/examples/human_melanopsin/accuracy_600m_masked.h5}"
STATE_600M_CACHE="${STATE_600M_CACHE:-$ROOT/embeddings/application/melanopsin_esmc_600m_state_fp32.h5}"
STATE_CACHE="${STATE_CACHE:-$ROOT/embeddings/application/esmc_6b_targets_fp32.h5}"

export HF_HOME="${HF_HOME:-/data/fast/cache/huggingface}"
export TMPDIR="${TMPDIR:-/data/fast/tmp/protein-stabilizer}"
mkdir -p "$HF_HOME" "$TMPDIR" "$(dirname "$OUTPUT")"

POSITION_ARGS=()
if [[ -n "${POSITIONS:-}" ]]; then
  POSITION_ARGS=(--positions "$POSITIONS")
fi

cd "$ROOT"
"$ROOT/.venv/bin/protein-stabilizer" cache-accuracy-target \
  --fasta "$ROOT/examples/human_melanopsin/Q9UHM6.fasta" \
  --protected-mask \
    "$ROOT/examples/human_melanopsin/protected_positions.txt" \
  "${POSITION_ARGS[@]}" \
  --max-tokens "${MASKED_MAX_TOKENS:-8192}" \
  --max-batch-size "${MASKED_MAX_BATCH_SIZE:-128}" \
  --state-output "$STATE_600M_CACHE" \
  --output "$MASKED_CACHE"

"$ROOT/.venv-esmc6b/bin/protein-stabilizer" cache-accuracy-state-6b \
  --fasta "$ROOT/examples/human_melanopsin/Q9UHM6.fasta" \
  --max-tokens "${STATE_MAX_TOKENS:-8192}" \
  --output "$STATE_CACHE"

"$ROOT/.venv-esmc6b/bin/protein-stabilizer" screen-accuracy-6b \
  --fasta "$ROOT/examples/human_melanopsin/Q9UHM6.fasta" \
  --uniprot Q9UHM6 \
  --alphafold-min-plddt "${ALPHAFOLD_MIN_PLDDT:-70}" \
  --protected-mask \
    "$ROOT/examples/human_melanopsin/protected_positions.txt" \
  "${POSITION_ARGS[@]}" \
  --masked-marginals-cache "$MASKED_CACHE" \
  --embedding-cache "$STATE_CACHE" \
  --secondary-state-embedding-cache "$STATE_600M_CACHE" \
  --require-component-agreement \
  --max-per-site "${MAX_PER_SITE:-2}" \
  --top "${TOP:-50}" \
  --max-tokens "${STATE_MAX_TOKENS:-8192}" \
  --max-batch-size 1 \
  --output "$OUTPUT"
