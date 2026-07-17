# ProteinStabilizer

ProteinStabilizer ranks amino-acid substitutions for stability, with a
GPCR-specific calibration layer. The fast first pass uses a frozen ESM-C 600M
encoder and compact heads over contextual residue-embedding differences. An
optional full ESM-C 6B second pass now supplies a stronger general-ddG estimate
and conservatively reranks GPCR candidates. A separate 6B epistasis head scores
double mutants and accepts larger mutation sets as an explicit extrapolation.
Saturation screening also reads the encoder's masked amino-acid probabilities
as a small sequence-compatibility prior. The current pipeline is intentionally
optimized for application and screening, rather than for introducing a new
protein-stability architecture.

Solubility is deliberately not part of the current model.

## Project dashboard

Open [`docs/model_dashboard.html`](docs/model_dashboard.html) for the current
model roles, training losses, held-out performance, expected ddG error scale,
GPCR screening evidence, and external-method context. It is generated directly
from the checked-in metric and audit JSON files:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python scripts/build_model_dashboard.py
```

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

The optional 6B path repeats the same residue-delta architecture with 2,560
dimensional ESM-C 6B vectors and a newly trained five-member Megascale ensemble.
It does not mix 600M and 6B embeddings or heads. The application output keeps
the 6B general ddG, ProTherm-adapted ddG, and MPTherm delta-Tm as separate
columns. Its bounded GPCR rerank is 60% 600M MPTherm percentile, 15% 600M
masked-marginal percentile, and 25% 6B MPTherm percentile. For combinations,
the 6B constituent ensemble predictions are summed and one permutation-invariant
6B epistasis correction is applied.

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

`requirements-lock.txt` records the exact 600M environment and
`requirements-esmc6b-lock.txt` records the 6B training environment. The looser
`pyproject.toml` bounds are for development.

ESM-C 6B uses `transformers==4.57.6`, which conflicts with the validated 600M
environment. Install it separately:

```bash
python3.12 -m venv .venv-esmc6b
.venv-esmc6b/bin/pip install -e '.[esmc6b]'
```

The 6B checkpoint is downloaded from `biohub/ESMC-6B` on first use unless
`--model` points to an existing snapshot.

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
| ESM-C 600M double-mutant learned estimate | 0.545 | 0.511 | 0.851 | 1.144 |
| ESM-C 600M double-mutant additive baseline | 0.601 | 0.562 | 1.135 | 1.441 |
| ProTherm experimental ddG holdout | 0.409 | 0.292 | 1.168 | 1.993 |
| MPTherm delta-Tm holdout | 0.222 | 0.208 | 3.652 | 4.826 |
| Alpha-helical membrane ddG test, frozen ESM-C ensemble | 0.095 | 0.174 | 0.961 | 1.208 |
| Alpha-helical membrane ddG test, rejected adapter | -0.025 | 0.063 | 1.014 | 1.276 |
| GPCR-tm delta-Tm test, MPTherm head | 0.371 | 0.340 | 3.588 | 4.435 |
| GPCR-tm delta-Tm test, rejected GPCR adapter | -0.042 | -0.044 | 3.737 | 4.623 |
| GPCR-tm delta-Tm test, official ThermoMPNN | 0.126 | -0.016 | 3.805 | 4.542 |
| GPCR-tm delta-Tm test, rejected structure/model blend | 0.385 | 0.426 | 3.602 | 4.221 |
| ESM-C 6B single-mutant protein holdout | 0.818 | 0.816 | 0.516 | 0.705 |
| ESM-C 6B double-mutant learned estimate | 0.623 | 0.592 | 0.741 | 0.983 |
| ESM-C 6B double-mutant additive baseline | 0.615 | 0.586 | 1.142 | 1.468 |
| ESM-C 6B ProTherm protein holdout | 0.481 | 0.348 | 1.133 | 1.979 |
| ESM-C 6B MPTherm protein holdout | 0.274 | 0.278 | 3.517 | 4.719 |

For 600M, the learned epistasis correction improves absolute error but reduces
test-set rank correlation. With 6B, it improves both rank correlation and
absolute error over the corresponding additive baseline. In both paths the
constituent additive value and learned correction are returned separately.

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

Run the same combination directly with the full 6B checkpoints in its separate
environment:

```bash
.venv-esmc6b/bin/protein-stabilizer predict-6b \
  --fasta target_gpcr.fasta \
  --mutations L72A,A73V
```

This uses only 6B embeddings and heads. On the protein-disjoint Megascale-D
double-mutant test, its deployed learned estimate reaches Spearman `0.623`,
MAE `0.741`, and RMSE `0.983`, versus Spearman `0.615`, MAE `1.142`, and RMSE
`1.468` for the 6B additive baseline. These short soluble-protein measurements
support the interaction model but are not direct evidence of GPCR-combination
accuracy.

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

For a slower second-stage 6B pass, rerank the completed 600M CSV in the separate
environment:

```bash
.venv-esmc6b/bin/protein-stabilizer rerank-6b \
  --fasta target_gpcr.fasta \
  --input artifacts/target_gpcr_screen.csv \
  --output artifacts/target_gpcr_screen_6b.csv
```

This adds `esmc6b_pretrained_ddg`, `esmc6b_protherm_calibrated_ddg`,
`esmc6b_mptherm_predicted_delta_tm`, and 6B uncertainty/masked-marginal
diagnostics. It preserves the original score and rank in
`esmc600m_consensus_rank_score` and `esmc600m_rank`, then sorts by the optional
60/15/25 dual-backbone consensus. On GPCR-tm development, the bounded prior
improved macro within-receptor Spearman from `0.086` to `0.117` without
worsening any evaluable receptor. On the independent C5aR scan, average
precision improved from `0.255` to `0.317`, with 15 rather than 13 positives in
the top 50. Exact selection, provenance, and caveats are in
[`docs/esmc6b_full_transfer_audit.json`](docs/esmc6b_full_transfer_audit.json).

## Limitations

- The encoder is sequence-only; membrane topology and structure are not model
  inputs. A membrane-only ddG adapter was tested and rejected because it was
  worse than the frozen ESM-C baseline on two unseen alpha-helical proteins.
  Official ThermoMPNN predictions, local structure descriptors, and a compact
  structure/model blend were also evaluated on GPCR-tm and did not pass the
  development plus within-receptor ranking gates. Sequence-conditioned
  ProteinMPNN logic was additionally discovered with 600M and scaled to the
  selected 6B rank; its development-optimal blend reached macro Spearman
  `0.489` but fell to `-0.500` on the official two-receptor macro check and
  reduced C5aR top-50 recovery from 15 to 11. GPCRdb construct positives,
  ProteinGym membrane expression/abundance assays, and a direct C5aR
  alanine-scan classifier also failed the same untouched-test gate.
- ProTherm is heterogeneous and replicate measurements can disagree; the
  normalization records replicate count and spread for every mutation.
- The MPTherm head reaches only Spearman `0.371` on the small leak-free GPCR-tm
  test. Its delta-Tm output is an auxiliary ranking signal, not a calibrated
  universal GPCR stability measurement.
- The optional 6B rerank is supported by only 82 GPCR-tm development rows, a
  12-row confirmation set, and one independent C5aR scan. The confirmation set
  has been consulted in earlier project stages, so it is not a fresh untouched
  benchmark. Keep the original 600M rank visible when selecting experiments.
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

## ESM-C 6B status

Full ESM-C 6B retraining is complete. Its separate caches contain 136,466
Megascale single-mutant sequences, 125,823 Megascale double-mutant sequences,
and 7,502 focused transfer/GPCR sequences. Its compact application checkpoints
are under `checkpoints/esmc_6b/`. The generic single-mutant holdout improves
from Spearman `0.777` and MAE `0.556` with 600M to `0.818` and `0.516` with 6B.
The deployed double-mutant estimate improves from Spearman `0.545` and MAE
`0.851` with 600M to `0.623` and `0.741` with 6B. ProTherm and MPTherm
protein-held-out correlations also improve.

A 6B-only GPCR rank did not preserve the small official GPCR-tm receptor test,
so 6B does not replace the fast production rank. Instead it is an optional
bounded second-stage prior and a stronger general-ddG diagnostic. The 600M and
6B caches remain strictly separate. The earlier masked-only/DDGemb experiment
is recorded in
[`docs/esmc6b_ddgemb_transfer_audit.json`](docs/esmc6b_ddgemb_transfer_audit.json);
the completed full-transfer stage is in
[`docs/esmc6b_full_transfer_audit.json`](docs/esmc6b_full_transfer_audit.json).
A separate full-6B re-audit of the reconstructed four-receptor Muk et al.
thermostability matrix remained at chance under nested receptor holdout and is
recorded in
[`docs/esmc6b_muk_thermostability_audit.json`](docs/esmc6b_muk_thermostability_audit.json).

## Sources

- ThermoMPNN-D curated data: https://doi.org/10.5281/zenodo.13345274
- Megascale stability measurements: https://doi.org/10.1038/s41586-023-06328-6
- ESM-C implementation: https://github.com/evolutionaryscale/esm
- FireProtDB 2.0: https://loschmidt.chemi.muni.cz/fireprotdb/
- MPTherm-Pred dataset: https://web.iitm.ac.in/bioinfo2/mpthermpred/dataset_details.html
- mCSM-membrane dataset: https://biosig.lab.uq.edu.au/mcsm_membrane/data
- GPCR-tm dataset: https://biosig.lab.uq.edu.au/gpcr_tm/data
- DDGemb S2450, S669, and ptMUL-NR datasets: https://ddgemb.biocomp.unibo.it/datasets/
- Biohub ESM-C 6B checkpoint: https://huggingface.co/biohub/ESMC-6B
