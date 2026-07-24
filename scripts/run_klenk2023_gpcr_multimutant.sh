#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_ROOT="${RUN_ROOT:-/data/fast/tmp/protein-stabilizer/prospective-gpcr/klenk2023}"
RAW="$RUN_ROOT/raw"
BENCHMARK="$RUN_ROOT/klenk2023_gpcr_multimutant.csv"
CHECKPOINT="$ROOT/checkpoints/esmc_6b_accuracy_fp32/promoted_affine/accuracy_ensemble.pt"

export HF_HOME="${HF_HOME:-/data/fast/cache/huggingface}"
export TMPDIR="${TMPDIR:-/data/fast/tmp/protein-stabilizer/tmp}"
mkdir -p "$HF_HOME" "$TMPDIR" "$RUN_ROOT/caches" "$RUN_ROOT/screens"

if [[ ! -f "$BENCHMARK" ]]; then
  echo "missing frozen benchmark: $BENCHMARK" >&2
  echo "run scripts/prepare_klenk2023_gpcr_multimutant.py first" >&2
  exit 1
fi

run_receptor() {
  local accession="$1"
  local positions="$2"
  local fasta="$RAW/$accession.fasta"
  local masked="$RUN_ROOT/caches/${accession}_masked_600m.h5"
  local state="$RUN_ROOT/caches/${accession}_state_6b_fp32.h5"
  local screen="$RUN_ROOT/screens/${accession}_accuracy_6b.csv"

  "$ROOT/.venv/bin/protein-stabilizer" cache-accuracy-target \
    --fasta "$fasta" \
    --positions "$positions" \
    --device "${MASKED_DEVICE:-cuda}" \
    --max-tokens "${MASKED_MAX_TOKENS:-2048}" \
    --max-batch-size "${MASKED_MAX_BATCH_SIZE:-16}" \
    --output "$masked" \
    > "$RUN_ROOT/screens/${accession}_masked_report.json"

  "$ROOT/.venv-esmc6b/bin/protein-stabilizer" cache-accuracy-state-6b \
    --fasta "$fasta" \
    --device "${STATE_DEVICE:-cuda}" \
    --max-tokens "${STATE_MAX_TOKENS:-2048}" \
    --output "$state" \
    > "$RUN_ROOT/screens/${accession}_state_report.json"

  "$ROOT/.venv-esmc6b/bin/protein-stabilizer" screen-accuracy-6b \
    --fasta "$fasta" \
    --uniprot "$accession" \
    --positions "$positions" \
    --device "${STATE_DEVICE:-cuda}" \
    --masked-marginals-cache "$masked" \
    --embedding-cache "$state" \
    --no-require-component-agreement \
    --top 500 \
    --max-batch-size 1 \
    --output "$screen" \
    > "$RUN_ROOT/screens/${accession}_screen_report.json"
}

cd "$ROOT"
run_receptor P20789 "107,155,167,170,239,293,358,365,392"
run_receptor Q03431 "184,188,271,283,294,312,325,341,368,369,373,407,412,427,461"

"$ROOT/.venv/bin/python" \
  "$ROOT/scripts/evaluate_klenk2023_gpcr_multimutant.py" \
  --benchmark "$BENCHMARK" \
  --screen "P20789=$RUN_ROOT/screens/P20789_accuracy_6b.csv" \
  --screen "Q03431=$RUN_ROOT/screens/Q03431_accuracy_6b.csv" \
  --checkpoint "$CHECKPOINT" \
  --output-csv "$RUN_ROOT/klenk2023_gpcr_multimutant_predictions.csv" \
  --output-json "$ROOT/docs/klenk2023_gpcr_multimutant_audit.json"
