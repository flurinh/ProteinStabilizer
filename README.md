# ProteinStabilizer

ProteinStabilizer ranks amino-acid substitutions for stability, with a
GPCR-specific calibration layer. It uses a frozen ESM-C 600M encoder and trains
only compact heads over contextual residue-embedding differences.

Solubility is deliberately not part of the current model.

## Model

For one substitution at residue `i`:

```text
delta_i = ESM-C(mutant)[i] - ESM-C(WT)[i]
predicted ddG = SingleMutationHead(delta_i)
```

Negative predicted ddG is stabilizing. The thermodynamic head was trained on
the published Megascale/cDNA single-mutant split.

For a set of substitutions, the model embeds the WT, every constituent single
mutant, and the complete joint mutant. For each mutation it constructs:

```text
[single_delta_i, joint_delta_i, joint_delta_i - single_delta_i]
```

A DeepSets-style head aggregates these elements by sum, mean, and max. Its
epistasis correction is added to the constituent single predictions. The
operation is permutation invariant and accepts any mutation count, although
the epistasis head has only been trained and evaluated on double mutants.

The final GPCR layer is a low-data ranking calibration trained on the supplied
NTSR1, A2A, and beta-1 adrenergic receptor measurements. Because those endpoints
are residual binding after heating rather than thermodynamic ddG, the GPCR
output is explicitly a ranking score, not kcal/mol or a stability percentage.

## Data and splits

Download and normalize the data:

```bash
python scripts/download_data.py
python scripts/prepare_gpcr_benchmark.py
```

The pipeline uses:

| Dataset | Role | Split policy |
| --- | --- | --- |
| cDNA/Megascale singles | Single-head training | 116 source proteins, with a protein-held-out validation subset |
| cDNA/Megascale single test | External evaluation | 19 held-out proteins; 19,645 mutations |
| Megascale-D | Epistasis training | Published 90/17/20-protein train/validation/test split |
| GPCR workbook | Application calibration | Entire `(protein, mutation site)` groups; assay-balanced 60/20/20 split |

The compact ThermoMPNN-D Zenodo release is checksum verified. The downloader
excludes the 6.6 GB Rosetta sweep because it is not needed by this model.

## Environment and training

ESM-C 600M is provided by `esm==3.2.1.post1`. The model license may require an
authenticated Hugging Face account when the checkpoint is not already cached.

```bash
python -m venv --system-site-packages .venv
.venv/bin/pip install -e '.[esmc,test]'

# Set this to the desired shared or local Hugging Face cache.
export HF_HOME=/data/fast/cache/huggingface

.venv/bin/protein-stabilizer run
```

`requirements-lock.txt` records the exact package versions used to produce the
checked-in heads. The looser `pyproject.toml` bounds are for development.

The `run` command:

1. collects every required WT, single, joint-double, and GPCR sequence;
2. stores final-layer ESM-C residue vectors in a resumable HDF5 cache;
3. materializes row-aligned delta features and provenance manifests;
4. trains the single and epistasis heads;
5. selects the GPCR calibration on the grouped validation split; and
6. evaluates each held-out test split after model selection.

Generated embeddings and features stay under `embeddings/` and `artifacts/`.
The compact trained heads and their metric records are in
`checkpoints/esmc_600m/`.

## Measured performance

The current deterministic run used seed `20260715`:

| Evaluation | Spearman | Pearson | MAE | RMSE |
| --- | ---: | ---: | ---: | ---: |
| Single-mutant protein holdout | 0.754 | 0.762 | 0.582 | 0.786 |
| Double-mutant learned estimate | 0.492 | 0.452 | 0.907 | 1.220 |
| Double-mutant additive baseline | 0.560 | 0.518 | 1.192 | 1.509 |

The epistasis model improves absolute error but the additive score ranks the
double-mutant test set better. Both values are returned at inference.

On the assay-balanced GPCR holdout, the fine-tuned model achieved macro
within-assay Spearman `0.370`, compared with `-0.237` for the uncalibrated
pretrained score. This result covers only 19 held-out mutation sites and is
heterogeneous by receptor/assay, so it should be used as a secondary prior.

## Prediction

Score one or more substitutions:

```bash
.venv/bin/protein-stabilizer predict \
  --fasta target_gpcr.fasta \
  --mutations L72A,A73V
```

For a double mutant the output contains:

- each constituent single prediction;
- the additive ddG;
- the learned epistasis correction; and
- the corrected total ddG.

Screen all 19 substitutions at selected sites, batching sequences according to
the ESM-C token budget:

```bash
.venv/bin/protein-stabilizer screen \
  --fasta target_gpcr.fasta \
  --positions 45-60,72,73,100-120 \
  --output artifacts/target_gpcr_screen.csv
```

When GPCR calibration is available, screening uses a conservative consensus:
75% general-stability percentile and 25% GPCR fine-tune percentile. Raw ddG,
GPCR score, percentiles, and the consensus rank are all retained in the CSV.

## Limitations

- The encoder is sequence-only; membrane topology and structure are not yet
  model inputs.
- The general training proteins are mostly short soluble domains, so GPCR use
  is an extrapolation partially corrected by a very small benchmark.
- GPCR assays can disagree for the same mutation and ligand state.
- More than two mutations are supported architecturally but extrapolate beyond
  epistasis training.
- Predictions are candidates for experimental screening, not evidence that a
  receptor will express, remain functional, or crystallize.

## Sources

- ThermoMPNN-D curated data: https://doi.org/10.5281/zenodo.13345274
- Megascale stability measurements: https://doi.org/10.1038/s41586-023-06328-6
- ESM-C implementation: https://github.com/evolutionaryscale/esm
