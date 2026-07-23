# ProteinStabilizer

ProteinStabilizer ranks amino-acid substitutions for stability, with a
GPCR-specific calibration layer. The promoted fast path uses frozen ESM-C 600M
embeddings with mutation-direction, ordered local, whole-protein, membrane, and
learned ProteinMPNN context. An optional full ESM-C 6B second pass supplies a
stronger general-ddG estimate and conservatively reranks GPCR candidates.
Permutation-invariant epistasis heads score double mutants and accept larger
mutation sets as an explicit extrapolation.
Saturation screening also reads the encoder's masked amino-acid probabilities
as a small sequence-compatibility prior. The current pipeline is intentionally
optimized for application and screening, rather than for introducing a new
protein-stability architecture.

Solubility is deliberately not part of the current model.

## Project dashboard

Open [`docs/model_dashboard.html`](docs/model_dashboard.html) for the current
model roles, training losses, held-out performance, expected ddG error scale,
the full 19,645-point predicted-versus-experimental kcal/mol scatter, GPCR
screening evidence, and external-method context. It is generated directly from
the checked-in metric and audit JSON files:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python scripts/build_model_dashboard.py
```

The exact v2 splits, training counts, promotion gates, and transfer caveats are
recorded in [`docs/v2_training_report.md`](docs/v2_training_report.md).

## Model

### Full-protein structure experiment

The repository now includes a reproducible SPURS-inspired training path that
embeds each unique WT once, combines full ESM-C residue/global states with a
trainable masked ProteinMPNN encoder through cross-attention, and decodes all
20 amino-acid potentials in one pass. Multiple substitutions use an unordered
additive-plus-epistasis decoder and always expose both components.

The real 600M run used 136,333 single and 114,109 double mutants across 173
proteins, five MMseqs-family folds, and a newly sealed family outer partition.
Warm-started structure fusion improved development OOF MAE only from `0.53131`
to `0.53081` kcal/mol, below the predeclared `0.02` scale gate, so it was not
promoted or scaled to 6B. The selected sequence state-potential model reached
outer-family MAE `0.59824` kcal/mol (95% protein-bootstrap CI
`0.55861–0.64466`) and Spearman `0.72580`. Learned double epistasis also failed
its development gate, so additive prediction is selected while the residual
is retained as a diagnostic.

The durable result and artifact hashes are in
[`docs/full_structure_training_audit.json`](docs/full_structure_training_audit.json).
The generated embedding bank, feature tensors, checkpoints, prediction CSVs,
and real-scale scatterplots remain git-ignored.

```bash
protein-stabilizer embed-full-structure
protein-stabilizer features-full-structure
protein-stabilizer train-full-structure
protein-stabilizer evaluate-full-structure-outer
```

The outer command is intentionally one-shot per output directory and refuses
to overwrite an existing report. Negative ddG means stabilizing.

### Promoted v2 hierarchy

The v2 single-mutant model receives both WT and complete-mutant final-layer
ESM-C states. At the mutation site it uses the center residue, an ordered
radius-four sequence window, and a whole-protein mean. A frozen
sequence-conditioned ProteinMPNN representation and eight membrane/topology
features condition the learned fusion. Structure is missing-masked rather than
imputed when no exact chain match is available.

The directional latent is explicitly antisymmetrized:

```text
odd(WT, mut) = 0.5 * (f(WT, mut) - f(mut, WT))
ddG(WT, mut) = linear_without_bias(odd(WT, mut))
```

Consequently, reversing a mutation negates the prediction exactly and a
self-mutation is exactly zero. Negative project ddG means stabilizing. Separate
bias-free projections predict thermodynamic ddG, delta-Tm, and stabilizer
retrieval; endpoints with different units are never pooled.

Architecture discovery was bounded to the sequence hierarchy versus the same
hierarchy with learned ProteinMPNN fusion. The selected fusion model was then
trained as a five-member ensemble with batch size 256, a 50-epoch floor, and
protein-held-out validation. Its historical 19,645-row generic test reaches
Spearman `0.818`, MAE `0.513` kcal/mol, stabilizer average precision `0.386`,
and 33 stabilizers in the top 50. It passed every prespecified promotion gate
against the retained 600M baseline at the time. Because that test was then
reused for later decisions, it is now a historical continuity set rather than
an untouched estimate.

For double mutants, the v2 head sums the two frozen constituent ddG predictions
and learns a DeepSets-style epistasis correction from constituent and complete
joint-mutant latents. It is exactly permutation invariant and always reports
the additive and epistasis components separately. On the 18,574-row,
protein-held-out test, it improves the additive baseline from MAE `0.917` to
`0.754` kcal/mol and from Spearman `0.562` to `0.581`.

### Retained v1 path

For one substitution at residue `i`, the retained v1 scorer uses:

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

ESM-C 6B uses Biohub's ESM-C-enabled Transformers `4.57.6` fork pinned to
commit `ef32577f55da19a4989cd7b22e004dc43a4998cb`. It conflicts with the
validated 600M environment, so install it separately:

```bash
bash scripts/setup_esmc6b_env.sh
```

The 6B checkpoint is downloaded from `biohub/ESMC-6B` on first use unless
`--model` points to an existing snapshot.

The promoted 6B v2 path is accuracy-first: it loads the checkpoint's native
FP32 tensors, disables TF32, forces the FP32 math attention backend, stores
hierarchy vectors as FP32, and trains the downstream head with strict FP32
matrix multiplication. `embed-v2-context-6b` exposes BF16/FP16 only as explicit
exploratory options; reduced-precision artifacts are not eligible for the
definitive 6B promotion comparison.

The `run` command:

1. collects every required WT, single, joint-double, transfer, and GPCR sequence;
2. stores final-layer ESM-C residue vectors in a resumable HDF5 cache;
3. materializes row-aligned delta features and provenance manifests;
4. trains the five-member single ensemble, epistasis, and transfer heads;
5. selects the GPCR calibration on the mutation-site-held-out validation split; and
6. evaluates each held-out test split after model selection.

### Docker GPU runtimes

The project has two CUDA 12.8 image targets because the validated ESM-C 600M
and 6B dependency stacks conflict. Build the image that matches the backbone;
neither image contains model weights, datasets, embeddings, or trained heads.
Those generated assets remain on the host and are mounted by Compose.

```bash
cp .env.docker.example .env.docker
sed -i "s/^HOST_UID=.*/HOST_UID=$(id -u)/; s/^HOST_GID=.*/HOST_GID=$(id -g)/" \
  .env.docker

docker compose --env-file .env.docker build protein-stabilizer-600m
docker compose --env-file .env.docker build protein-stabilizer-6b
```

Run the human melanopsin two-stage screen with the 6B image:

```bash
docker compose --env-file .env.docker run --rm protein-stabilizer-6b \
  screen-v2-6b \
  --fasta examples/human_melanopsin/Q9UHM6.fasta \
  --protected-mask examples/human_melanopsin/protected_positions.txt \
  --scan-mode two-stage \
  --rerank-top 128 \
  --max-per-site 2 \
  --top 50 \
  --topology alpha_helical_gpcr \
  --output artifacts/examples/human_melanopsin/6b_screen.csv
```

Use service `protein-stabilizer-600m` with command `screen-v2` for the 600M
path. Compose gives the container GPU access and mounts `artifacts/`,
`checkpoints/`, `data/`, and `embeddings/` at their normal project paths. It
also reuses the host Hugging Face cache, so already downloaded ESM-C weights
are not copied into the image or downloaded again. For another machine, edit
the three cache/temp host paths in `.env.docker`; they should point to its
large-volume filesystem.

The equivalent direct builds are:

```bash
docker build --target runtime-600m --build-arg APP_UID="$(id -u)" \
  --build-arg APP_GID="$(id -g)" -t protein-stabilizer:600m .
docker build --file Dockerfile.6b --target runtime-6b \
  --build-arg APP_UID="$(id -u)" \
  --build-arg APP_GID="$(id -g)" -t protein-stabilizer:6b .
```

The host needs an NVIDIA driver, Docker's NVIDIA runtime, and enough GPU memory
for the selected backbone. The strict-FP32 6B screen is intended for the
32 GB-class GPU used by this project. If a PDB is supplied, additionally mount
ProteinMPNN read-only and pass its container path with
`--proteinmpnn-repository`.

Generated embeddings and features stay under `embeddings/` and `artifacts/`.
The compact trained heads and their metric records are in
`checkpoints/esmc_600m/`. `single_ensemble.pt` is the production scorer;
`single_head.pt` retains its first member for transfer-head initialization and
backward compatibility.

## Measured performance

The current deterministic run used seed `20260715`:

| Evaluation | Spearman | Pearson | MAE | RMSE |
| --- | ---: | ---: | ---: | ---: |
| **ESM-C 6B strict-FP32 hierarchy/state fusion, single-mutant protein holdout** | **0.856** | **0.849** | **0.467** | **0.640** |
| **ESM-C 6B strict-FP32 fused double-mutant estimate** | **0.749** | **0.747** | **0.585** | **0.782** |
| ESM-C 6B v2 native-FP32 hierarchy before state fusion | 0.837 | 0.827 | 0.504 | 0.692 |
| ESM-C 6B v2 native-FP32 double before state fusion | 0.689 | 0.707 | 0.620 | 0.831 |
| ESM-C 6B v2 native-FP32 double-mutant additive baseline | 0.611 | 0.601 | 0.959 | 1.230 |
| ESM-C 600M v2 native-FP32 retrain | 0.823 | 0.824 | 0.507 | 0.700 |
| ESM-C 600M v2 hierarchy, single-mutant protein holdout | 0.818 | 0.818 | 0.513 | 0.707 |
| ESM-C 600M v2 double-mutant learned estimate | 0.581 | 0.593 | 0.754 | 0.992 |
| ESM-C 600M v2 double-mutant additive baseline | 0.562 | 0.575 | 0.917 | 1.154 |
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

The strict-FP32 6B hierarchy/state fusion is the definitive generic model. It
recovers 41 true stabilizers in the top 50 and reaches stabilizer average
precision `0.442` on the same frozen test, versus 33 and `0.386` for the
promoted 600M v2 model. Its double-mutant correction improves both absolute
error and rank over its additive baseline. Every multi-mutant path returns the
constituent additive value and learned correction separately.

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

Score one unordered substitution set with the promoted hierarchy/state-potential
fusion:

```bash
.venv/bin/protein-stabilizer predict-v2 \
  --fasta target_gpcr.fasta \
  --mutations L72A,A73V \
  --topology alpha_helical_gpcr \
  --generic-numbering 72=2.50x50,73=2.51x51
```

Add `--pdb receptor.pdb` to enable the learned ProteinMPNN context from a local
structure. For a canonical UniProt entry, `--uniprot Q9UHM6` instead resolves
the current PDB URL through the AlphaFold DB API and caches the structure plus
source hashes under `artifacts/structures/alphafold/`. In both cases, the PDB
must contain exactly one chain matching the FASTA sequence; otherwise the
command fails instead of silently misaligning residues. AlphaFold residues
below `--alphafold-min-plddt 70` are removed from the ProteinMPNN neighborhood
graph and marked structure-missing at those mutation sites. Without either
structure source, structure is explicitly missing-masked.

For the highest-accuracy prediction, run the strict-FP32 6B fusion in the
separate environment. The defaults load the validation-selected state
potential and, for combinations, the promoted permutation-invariant epistasis
head:

```bash
.venv-esmc6b/bin/protein-stabilizer predict-v2-6b \
  --fasta target_gpcr.fasta \
  --mutations L72A,A73V \
  --topology alpha_helical_gpcr
```

The strict 6B path also supports a site-bounded scan:

```bash
.venv-esmc6b/bin/protein-stabilizer screen-v2-6b \
  --fasta target_gpcr.fasta \
  --uniprot Q9UHM6 \
  --positions 72,73,100-110 \
  --topology alpha_helical_gpcr \
  --output artifacts/target_gpcr_v2_6b.csv
```

`--scan-mode exact` is the default and evaluates the full promoted blend.
`--scan-mode state-only` is a fast pre-screen: it embeds only the WT sequence
and scores all 20 amino-acid states per site in one head pass. State-only
scores are explicitly labeled and should be followed by exact rescoring of the
shortlist.

For a receptor-wide experimental suggestion set, use the bounded two-stage
mode and a hard functional mask:

```bash
.venv-esmc6b/bin/protein-stabilizer screen-v2-6b \
  --fasta target_gpcr.fasta \
  --protected-mask target_gpcr.protected.txt \
  --scan-mode two-stage \
  --rerank-top 128 \
  --max-per-site 2 \
  --topology alpha_helical_gpcr \
  --output artifacts/target_gpcr_suggestions.csv
```

The mask is one-based in FASTA coordinates. Each non-comment line contains a
position, range, or comma-separated expression followed optionally by a tab
and its reason; see
[`docs/gpcr_protected_mask.example.txt`](docs/gpcr_protected_mask.example.txt).
`--protected-positions 45,72-76` can add inline exclusions. The two sources are
merged, checked against sequence length, and removed before mutation candidates
or embeddings are generated. The model does not infer that a residue is safe:
include known ligand contacts, activation microswitches, conserved motifs,
disulfides, glycosylation sites, construct boundaries, and partner interfaces
in the target-specific mask.

Two-stage mode performs one WT embedding to score every allowed amino-acid
state, then embeds at most `--rerank-top` mutant sequences for the full promoted
fusion. For a 400-residue unmasked receptor this requests 128 mutant embeddings
instead of 7,600. `--max-per-site` is enforced both when choosing the exact
rerank set and when assembling the final experimental shortlist. The full CSV
retains all allowed state-potential candidates with their scoring stage; the
adjacent `*.shortlist.csv` contains only exact-reranked suggestions. The
provenance-checked application embedding cache under `embeddings/application/`
is reused on identical reruns. The JSON summary reports computed embeddings,
cache hits, avoided mutant embeddings, and the applied protected positions.
The WT backbone is likewise encoded by ProteinMPNN once per command and reused
for all substitutions. MegaScale training already follows the same scalable
pattern: one ProteinMPNN encoding per source protein is cached and indexed by
mutation site, rather than recomputing a structure representation per row.
A pinned, directly runnable human melanopsin example is under
[`examples/human_melanopsin/`](examples/human_melanopsin/README.md).

Convert an exact single-mutant shortlist into a bounded double-mutant design
set without repeatedly embedding the same constituent mutants:

```bash
.venv-esmc6b/bin/protein-stabilizer screen-v2-pairs-6b \
  --fasta target_gpcr.fasta \
  --single-screen artifacts/target_gpcr_suggestions.shortlist.csv \
  --protected-mask target_gpcr.protected.txt \
  --single-limit 20 \
  --single-ddg-ceiling 0 \
  --pair-rerank-top 64 \
  --max-pairs-per-site 4 \
  --topology alpha_helical_gpcr \
  --output artifacts/target_gpcr_pairs.csv
```

Pair generation accepts only suggestion-eligible `exact` or `exact-reranked`
single rows, requires both hierarchy and state-potential components to agree
on the stabilizing direction by default, and reapplies the hard protected
mask. Use `--no-require-component-agreement` only when consuming an exact
screen without both component columns. It prescreens all valid unordered pairs
by the sum of their input single ddGs, then computes fresh constituent
predictions and one joint-sequence embedding for only the bounded rerank set.
The full pair CSV separates input additive prescreen, exact
constituent ddGs, additive ddG, learned epistasis, and corrected total ddG.
The adjacent shortlist is ranked by corrected total and bounds reuse of any
one residue site. The epistasis head was trained only on double mutants; this
command deliberately does not extrapolate it to triples.

Native-FP32 6B inference is deliberately expensive. Use 600M to explore a
broad receptor-wide search and 6B to rescore a bounded set of sites when
turnaround matters.

Run an independent 19-amino-acid scan over selected sites:

```bash
.venv/bin/protein-stabilizer screen-v2 \
  --fasta target_gpcr.fasta \
  --positions 40-350 \
  --topology alpha_helical_gpcr \
  --output artifacts/target_gpcr_v2.csv
```

For GPCR application, negative generic-model ddG is stabilizing. The retained
GPCR reranker remains a separate candidate-ordering signal because it beats
the generic fusion on the independent C5aR scan; its score is not a physical
unit and never overwrites ddG. For a combination, the JSON reports every
constituent prediction, additive ddG, learned epistasis, and corrected total;
sets larger than two are explicitly marked as extrapolations.

The retained v1 application command remains:

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

For the promoted GPCR-specific experimental ordering, run the strict-FP32
`screen-v2-6b` command over exactly the same positions, pin the target's
GPCRdb parent-family alignment, and combine the two CSVs:

```bash
python scripts/download_gpcrdb_evolutionary.py \
  --entry-name c5ar1_human \
  --output data/raw/gpcrdb_evolutionary/c5ar1_human

.venv-esmc6b/bin/protein-stabilizer screen-v2-6b \
  --fasta target_gpcr.fasta \
  --positions 45-60,72,73,100-120 \
  --topology alpha_helical_gpcr \
  --output artifacts/target_gpcr_generic_6b.csv

.venv/bin/protein-stabilizer rank-gpcr-consensus \
  --fasta target_gpcr.fasta \
  --accession P21730 \
  --retained-input artifacts/target_gpcr_screen_6b.csv \
  --generic-6b-input artifacts/target_gpcr_generic_6b.csv \
  --gpcrdb-cache data/raw/gpcrdb_evolutionary/c5ar1_human \
  --output artifacts/target_gpcr_gpcr_consensus.csv
```

Replace the example entry name and accession with the target receptor. The
FASTA must match the pinned GPCRdb canonical sequence, and both input CSVs
must contain the identical mutation set; the command fails on mismatches.
The output retains the signed generic ddG and all source ranks, then adds
`gpcr_screening_rank_score`:

- 50% retained GPCR dual-backbone rank;
- 40% strict-FP32 6B generic favorable-stability percentile; and
- 10% target-excluded GPCRdb family log-odds percentile.

This selection reached development macro Spearman `0.338` (retained `0.115`,
generic 6B `0.311`), official macro Spearman `0.900` (retained `1.000`), and
C5aR AUC `0.718`, AP `0.377`, with 17/34 stabilizers in the top 50 (retained:
`0.696`, `0.317`, and 15/34). It is the current GPCR screening policy, but its
score is only a within-scan rank—not ddG, delta-Tm, percent stability, or
crystallization probability. The 12-row official set and C5aR scan were
consulted during model development, so no untouched GPCR benchmark remains.
Full selection and bootstrap evidence are in
[`docs/gpcr_evolutionary_consensus_audit.json`](docs/gpcr_evolutionary_consensus_audit.json).

## Limitations

- ESM-C itself is sequence-based. The promoted v2 head adds learned frozen
  ProteinMPNN context when an exact structure-chain match exists and explicit
  GPCR/membrane topology priors, but it does not yet distinguish
  lipid-exposed, solvent-exposed, and buried residue surfaces. The older
  post-hoc 6B ProteinMPNN likelihood blend is a different experiment: it
  reached development macro Spearman `0.489` but fell to `-0.500` on the
  official two-receptor check and reduced C5aR top-50 recovery from 15 to 11,
  so that blend remains rejected.
- ProTherm is heterogeneous and replicate measurements can disagree; the
  normalization records replicate count and spread for every mutation.
- The MPTherm head reaches only Spearman `0.371` on the small leak-free GPCR-tm
  test. Its delta-Tm output is an auxiliary ranking signal, not a calibrated
  universal GPCR stability measurement.
- The promoted GPCR consensus is supported by only 82 GPCR-tm development
  rows, a 12-row confirmation set, and one C5aR scan. The confirmation and
  C5aR sets have both been consulted during project development, so neither is
  a fresh untouched benchmark. Its C5aR AP improvement over the retained rank
  has a paired-bootstrap 95% interval of `-0.005` to `0.119`; keep every
  component visible and test a diverse experimental panel.
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

The current ESM-C 6B v2 run is complete in native FP32. The hierarchy cache
contains 259,830 unique sequences and 384,271 local windows. The promoted
hierarchy/state-potential fusion reaches historical protein-held-out Spearman `0.856`, MAE
`0.467` kcal/mol, stabilizer average precision `0.442`, and 41 stabilizers in
the top 50. The 19-protein test was consulted by prior promotion gates and is
therefore a continuity benchmark, not an untouched SOTA estimate; its
protein-bootstrap MAE 95% interval is `0.427–0.509`. The fused native-FP32
double-mutant model reaches Spearman `0.749`,
MAE `0.585`, and exact permutation invariance. Application checkpoints are
under `checkpoints/esmc_6b_state_potential_fp32/`.

The ProTherm transfer diagnostic improves to Spearman `0.452`, but the
receptor-disjoint GPCR delta-Tm and membrane transfers remain weak. The 6B
generic ddG model is therefore the current operational baseline for stability prediction,
while GPCR-specific experimental ordering uses the promoted family consensus
plus experimental judgment—not a claimed calibrated GPCR ddG model. The 600M
and 6B caches remain strictly separate. The earlier masked-only/DDGemb
experiment is recorded in
[`docs/esmc6b_ddgemb_transfer_audit.json`](docs/esmc6b_ddgemb_transfer_audit.json);
the completed full-transfer stage is in
[`docs/esmc6b_full_transfer_audit.json`](docs/esmc6b_full_transfer_audit.json).
A separate full-6B re-audit of the reconstructed four-receptor Muk et al.
thermostability matrix remained at chance under nested receptor holdout and is
recorded in
[`docs/esmc6b_muk_thermostability_audit.json`](docs/esmc6b_muk_thermostability_audit.json).
The current error-floor, split, competitor, and optimization assessment is in
[`docs/stability_optimization_study.md`](docs/stability_optimization_study.md).

## Sources

- ThermoMPNN-D curated data: https://doi.org/10.5281/zenodo.13345274
- Megascale stability measurements: https://doi.org/10.1038/s41586-023-06328-6
- ESM-C implementation: https://github.com/evolutionaryscale/esm
- FireProtDB 2.0: https://loschmidt.chemi.muni.cz/fireprotdb/
- MPTherm-Pred dataset: https://web.iitm.ac.in/bioinfo2/mpthermpred/dataset_details.html
- mCSM-membrane dataset: https://biosig.lab.uq.edu.au/mcsm_membrane/data
- GPCR-tm dataset: https://biosig.lab.uq.edu.au/gpcr_tm/data
- GPCRdb web services: https://docs.gpcrdb.org/web_services.html
- DDGemb S2450, S669, and ptMUL-NR datasets: https://ddgemb.biocomp.unibo.it/datasets/
- Biohub ESM-C 6B checkpoint: https://huggingface.co/biohub/ESMC-6B
