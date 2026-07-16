# ProteinStabilizer

ProteinStabilizer ranks amino-acid substitutions for stability, with a
GPCR-specific calibration layer. It uses a frozen ESM-C 600M encoder and trains
only compact heads over contextual residue-embedding differences. Saturation
screening also reads the encoder's masked amino-acid probabilities as a small
sequence-compatibility prior. The current pipeline is intentionally optimized
for application and screening, rather than for introducing a new
protein-stability architecture.

Solubility is deliberately not part of the current model.

## Model

For one substitution at residue `i`:

```text
delta_i = ESM-C(mutant)[i] - ESM-C(WT)[i]
predicted ddG = mean_k SingleMutationHead_k(delta_i)
```

Negative predicted ddG is stabilizing. The base thermodynamic head was trained
on the published Megascale/cDNA single-mutant split. Production scoring averages
five independently initialized heads trained on the same protein-disjoint split;
this adds no embedding work and reduces seed variance. The first member remains
the initialization for a nonlinear copy adapted on experimental ProTherm ddG.
A second copy is trained on MPTherm-Pred delta-Tm, for which positive values are
stabilizing. The ddG and delta-Tm labels are never pooled into one regression
target.

For a saturation scan, each WT site is also masked once. The log-probability
difference
`log P(mutant | masked WT context) - log P(WT | masked WT context)` scores all
19 substitutions from the same ESM-C pass. It is a rank-only sequence prior,
not a trained ddG estimate.

For a set of substitutions, the model embeds the WT, every constituent single
mutant, and the complete joint mutant. For each mutation it constructs:

```text
[single_delta_i, joint_delta_i, joint_delta_i - single_delta_i]
```

A DeepSets-style head aggregates these elements by sum, mean, and max. Its
epistasis correction is added to the constituent single predictions. The
operation is permutation invariant and accepts any mutation count, although
the epistasis head has only been trained and evaluated on double mutants.
Inference also reports additive WT-context masked log odds and a joint-context
pseudo-log-likelihood score, where each mutated site is masked while the other
mutations remain present. Their difference is an additional uncalibrated
interaction diagnostic.

The final GPCR layer is a low-data ranking calibration trained on the supplied
NTSR1, A2A, and beta-1 adrenergic receptor measurements. It combines four
mutation-delta scores (base ddG, delta norm, ProTherm-adapted ddG, and
MPTherm-adapted delta-Tm) through a ridge head. Because the GPCR endpoints are
residual binding after heating rather than thermodynamic ddG, the output is
explicitly a ranking score, not kcal/mol or a stability percentage.
Its new-receptor evidence is weak, so it is reported separately and is not
included in the primary screening consensus.

Two additional membrane-specific transfer experiments are retained as audited
candidates but fail their deployment gates: an mCSM-membrane equilibrium-ddG
adapter and a GPCR-tm delta-Tm adapter. Neither replaces the production score.

## Data and splits

Download and normalize the data:

```bash
python scripts/download_data.py
python scripts/prepare_gpcr_benchmark.py
python scripts/download_transfer_data.py
```

The pipeline uses:

| Dataset | Role | Split policy |
| --- | --- | --- |
| cDNA/Megascale singles | Single-head training | 116 source proteins, with a protein-held-out validation subset |
| cDNA/Megascale single test | External evaluation | 19 held-out proteins; 19,645 mutations |
| Megascale-D | Epistasis training | Published 90/17/20-protein train/validation/test split |
| ProTherm via FireProtDB 2.0 | Experimental ddG transfer | 148/18/18-protein upstream train/validation/test split; 4,504 replicate-aggregated mutations |
| MPTherm-Pred | Membrane-domain delta-Tm auxiliary task | Derived protein-disjoint 633/136/124-row train/validation/test split; 21 rows quarantined from GPCR evaluations |
| mCSM-membrane | Membrane equilibrium-ddG transfer audit | Four-protein development split; 24 forward mutations from unseen alpha-helical proteins 1AFO and 2K73 are test-only |
| GPCR-tm | GPCR delta-Tm transfer audit | 82 development rows after removing substitutions at official test sites; 12-row official test |
| GPCR workbook | Application calibration | Entire `(protein, mutation site)` groups; assay-balanced 60/20/20 split |

The compact ThermoMPNN-D Zenodo release is checksum verified. The downloader
excludes the 6.6 GB Rosetta sweep because it is not needed by this model.
The transfer downloader also pins SHA-256 checksums. It extracts only ProTherm
records from a FireProtDB 2.0 mirror, rejects missing/inconsistent sequences,
aggregates replicate experiments by median, and downweights disagreements.
MPTherm-Pred and GPCR-tm UniProt sequences are cached locally. Proteins longer
than 1,022 residues use a mutation-centered window so ESM-C attention remains
bounded. Synthetic reverse rows in mCSM-membrane are marked, downweighted as
paired observations, and excluded from the forward-mutation test.

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

1. collects every required WT, single, joint-double, transfer, and GPCR sequence;
2. stores final-layer ESM-C residue vectors in a resumable HDF5 cache;
3. materializes row-aligned delta features and provenance manifests;
4. trains the five-member single ensemble, epistasis, and transfer heads;
5. selects the GPCR calibration on the mutation-site-held-out validation split; and
6. evaluates each held-out test split after model selection.

Generated embeddings and features stay under `embeddings/` and `artifacts/`.
The compact trained heads and their metric records are in
`checkpoints/esmc_600m/`. `single_ensemble.pt` is the production scorer;
`single_head.pt` retains its first member for transfer-head initialization and
backward compatibility.

## Measured performance

The current deterministic run used seed `20260715`:

| Evaluation | Spearman | Pearson | MAE | RMSE |
| --- | ---: | ---: | ---: | ---: |
| Single-mutant protein holdout | 0.777 | 0.782 | 0.556 | 0.755 |
| Double-mutant learned estimate | 0.545 | 0.511 | 0.851 | 1.144 |
| Double-mutant additive baseline | 0.601 | 0.562 | 1.135 | 1.441 |
| ProTherm experimental ddG holdout | 0.409 | 0.292 | 1.168 | 1.993 |
| MPTherm delta-Tm holdout | 0.222 | 0.208 | 3.652 | 4.826 |
| Alpha-helical membrane ddG test, frozen ESM-C ensemble | 0.095 | 0.174 | 0.961 | 1.208 |
| Alpha-helical membrane ddG test, rejected adapter | -0.025 | 0.063 | 1.014 | 1.276 |
| GPCR-tm delta-Tm test, MPTherm head | 0.371 | 0.340 | 3.588 | 4.435 |
| GPCR-tm delta-Tm test, rejected GPCR adapter | -0.042 | -0.044 | 3.737 | 4.623 |
| GPCR-tm delta-Tm test, official ThermoMPNN | 0.126 | -0.016 | 3.805 | 4.542 |
| GPCR-tm delta-Tm test, rejected structure/model blend | 0.385 | 0.426 | 3.602 | 4.221 |

The epistasis model improves absolute error but the additive score ranks the
double-mutant test set better. Both values are returned at inference.

On the 26-row, mutation-site-held-out GPCR workbook test, the calibration
achieves macro within-assay Spearman `0.601` and MAE `24.34` percentage points.
That number does not estimate performance on a new receptor. In the more
application-relevant leave-one-receptor-out diagnostic, macro within-assay
Spearman falls to `0.171` (uncalibrated baseline `0.025`), with receptor/assay
values ranging from `-0.121` to `0.515`. Each fold also retrains the MPTherm
head after excluding every row from the held-out receptor. The GPCR score is
therefore retained only as an assay-specific diagnostic.

Additional GPCR-focused transfer, structure, physicochemical, uncertainty,
GPCRdb construct, ProteinGym membrane-expression, and direct C5aR alanine-scan
experiments are recorded in
[`docs/gpcr_model_selection.md`](docs/gpcr_model_selection.md). They were kept
out of trained production heads because their receptor-held-out or
within-receptor ranking did not improve enough to justify added complexity.
The accepted masked-marginal runtime prior and its audit are recorded in
[`docs/esmc_masked_marginal_audit.json`](docs/esmc_masked_marginal_audit.json).

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
- the corrected total ddG;
- additive WT-context masked log odds; and
- joint-context masked pseudo-log-likelihood and interaction log odds.

Screen all 19 substitutions at selected sites, batching sequences according to
the ESM-C token budget:

```bash
.venv/bin/protein-stabilizer screen \
  --fasta target_gpcr.fasta \
  --positions 45-60,72,73,100-120 \
  --output artifacts/target_gpcr_screen.csv
```

The primary scan rank is 80% MPTherm delta-Tm percentile and 20% ESM-C
masked-marginal percentile. The general ddG estimate and assay-specific GPCR
score remain separate CSV columns: use `predicted_stabilizing` to require
agreement with the thermodynamic head when choosing a conservative experimental
set. The consensus is rank-only and can otherwise promote a candidate that the
general ddG head calls destabilizing. `pretrained_ddg_std` records disagreement
across the five heads as an uncertainty flag; it is useful for triage but is
not a calibrated confidence interval.

## Limitations

- The encoder is sequence-only; membrane topology and structure are not model
  inputs. A membrane-only ddG adapter was tested and rejected because it was
  worse than the frozen ESM-C baseline on two unseen alpha-helical proteins.
  Official ThermoMPNN predictions, local structure descriptors, and a compact
  structure/model blend were also evaluated on GPCR-tm and did not pass the
  development plus within-receptor ranking gates. GPCRdb construct positives,
  ProteinGym membrane expression/abundance assays, and a direct C5aR
  alanine-scan classifier also failed the same untouched-test gate.
- ProTherm is heterogeneous and replicate measurements can disagree; the
  normalization records replicate count and spread for every mutation.
- The MPTherm head reaches only Spearman `0.371` on the small leak-free GPCR-tm
  test. Its delta-Tm output is an auxiliary ranking signal, not a calibrated
  universal GPCR stability measurement.
- Masked-marginal log odds measure sequence compatibility, not kcal/mol,
  delta-Tm, expression, activity, or crystallizability. The 20% weight improved
  the independent C5aR scan while preserving the GPCR-tm within-receptor test
  rank, but the evidence is still small.
- GPCR site-held-out macro Spearman `0.601` measures residual-binding thermal
  assays, not thermodynamic GPCR ddG accuracy. New-receptor macro Spearman is
  only `0.171`; high-accuracy GPCR ddG prediction has not been demonstrated.
- GPCR assays can disagree for the same mutation and ligand state.
- More than two mutations are supported architecturally but extrapolate beyond
  epistasis training.
- Predictions are candidates for experimental screening, not evidence that a
  receptor will express, remain functional, or crystallize.

## ESM-C 6B migration

The feature and checkpoint schemas infer the embedding dimension, so the heads
do not assume 1,152 dimensions internally. Moving to ESM-C 6B still requires a
separate embedding cache and complete retraining; 600M and 6B vectors or heads
must never share a cache. The installed open-source ESM package currently
validates the local 600M backend, so a 6B run should be added as a distinct
backend once the 6B checkpoint/API and license are available.

## Sources

- ThermoMPNN-D curated data: https://doi.org/10.5281/zenodo.13345274
- Megascale stability measurements: https://doi.org/10.1038/s41586-023-06328-6
- ESM-C implementation: https://github.com/evolutionaryscale/esm
- FireProtDB 2.0: https://loschmidt.chemi.muni.cz/fireprotdb/
- MPTherm-Pred dataset: https://web.iitm.ac.in/bioinfo2/mpthermpred/dataset_details.html
- mCSM-membrane dataset: https://biosig.lab.uq.edu.au/mcsm_membrane/data
- GPCR-tm dataset: https://biosig.lab.uq.edu.au/gpcr_tm/data
