# ProteinStabilizer v2 training report

Date: 2026-07-17
Seed: `20260715`

## Decision

The selected hierarchy passed every prespecified generic promotion gate at
600M and then passed every scale gate at 6B under native-FP32 inference. The 6B
hierarchy is now the preferred highest-accuracy generic single-mutant and
stabilizer-retrieval model. Its double-mutant epistasis head also improves its
frozen additive baseline and is retained.

The native-FP32 600M retrain improves regression over the earlier promoted
600M model (Spearman `0.823` versus `0.818`, MAE `0.507` versus `0.513`) but
has slightly lower stabilizer average precision (`0.379` versus `0.386`).
Accordingly 600M remains a fast development/screening path rather than the
quality ceiling.

The experimental ProTherm, MPTherm, membrane, and GPCR transfer adapters remain
diagnostics. They do not improve enough to replace the retained transfer and
GPCR ranking paths, and the GPCR ranking output is not a physical unit.

For GPCR application, `screen-v2` therefore keeps the previously validated
`0.80` retained MPTherm + `0.20` masked-marginal GPCR rank as the selection key
and reports v2 signed ddG alongside it as orthogonal generic-stability support.
`predict-v2` supplies exact-directional constituent, additive, epistasis, and
total ddG for explicit mutation sets. `predict-v2-6b` and `screen-v2-6b`
provide the definitive native-FP32 generic stability path.

## Single-mutant model

Inputs are frozen final-layer ESM-C embeddings for the WT and complete mutant:

- mutation-site center state;
- ordered radius-four local sequence window;
- whole-protein mean state;
- frozen sequence-conditioned ProteinMPNN features when an exact structure
  chain is available; and
- eight membrane/topology indicators, including GPCR helix and generic-number
  offset when known.

The direction encoder is explicitly antisymmetrized, and every output projection
is bias-free. Forward plus reverse prediction and the self-mutation prediction
therefore have maximum absolute error `0.0` in the recorded audit.

The discovery comparison was limited to the sequence hierarchy and the same
hierarchy with learned ProteinMPNN fusion. The ProteinMPNN-fusion candidate won
on validation and was retrained as a five-member main ensemble.

Training used 104,315 rows from 116 source proteins, batch size 256, a 50-epoch
floor, and protein-held-out validation. Each member saw 5,215,750 examples and
20,400 optimizer steps. Selection used a joint regression, ranking, and
stabilizer-retrieval objective.

### Frozen generic test

The test contains 19,645 mutations from 19 source proteins absent from training
and validation.

| Metric | Retained 600M baseline | 600M v2 |
| --- | ---: | ---: |
| Spearman | 0.777 | 0.818 |
| Pearson | 0.782 | 0.818 |
| MAE, kcal/mol | 0.556 | 0.513 |
| RMSE, kcal/mol | 0.755 | 0.707 |
| Stabilizer average precision | 0.259 | 0.386 |
| Stabilizers in top 50 | 28 | 33 |

Negative project ddG means stabilizing. Average precision and top-50 recovery
use the signed ddG prediction; the auxiliary retrieval projection is also
recorded but is not silently substituted.

## Double mutants

The double-mutant head is a permutation-invariant set model. It returns:

```text
total ddG = frozen constituent single-ddG sum + learned epistasis
```

Training/validation/test contain 85,253/10,282/18,574 mutations from
90/17/20 disjoint proteins. The head trained for the required 50-epoch floor,
16,700 optimizer steps, and 4,262,650 examples.

| Frozen test component | Spearman | Pearson | MAE | RMSE |
| --- | ---: | ---: | ---: | ---: |
| Additive baseline | 0.562 | 0.575 | 0.917 | 1.154 |
| Additive + learned epistasis | 0.581 | 0.593 | 0.754 | 0.992 |
| Epistasis term on measured rows | 0.513 | 0.530 | 0.640 | 0.820 |

The maximum prediction change after swapping the two mutation inputs is `0.0`.
Sets larger than two mutations remain an architectural extrapolation.
One bounded learning-rate check was selected strictly by the prespecified
validation composite (`0.4145` versus `0.4076`); its test metrics above were
read only after that selection. The selection record and both checkpoints are
retained in `checkpoints/esmc_600m_v2/`.

## Transfer diagnostics

Endpoint heads are separate exact-directional adapters. Whole source proteins
or receptors are held out for physical endpoints; the GPCR ranking assay holds
out complete `(protein, mutation site)` groups.

| Dataset | Endpoint | Test rows | Spearman | MAE | Status |
| --- | --- | ---: | ---: | ---: | --- |
| ProTherm | ddG, kcal/mol | 376 | 0.384 | 1.248 | diagnostic |
| mCSM-membrane | ddG, kcal/mol | 24 | -0.099 | 1.215 | rejected |
| MPTherm | delta-Tm, °C | 124 | 0.161 | 3.795 | diagnostic |
| GPCR-tm | delta-Tm, °C | 84 | -0.105 | 3.891 | rejected |
| GPCR workbook | normalized rank score | 26 | 0.219 | 1.117 | diagnostic |

The very small GPCR-tm training partition contains only four receptors after
enforcing receptor-disjoint evaluation. Its weak result is evidence against
claiming calibrated GPCR delta-Tm, not a reason to weaken the split.

## Accuracy-first 6B precision policy

The Biohub ESM-C 6B checkpoint shards contain native FP32 tensors. The
definitive v2 scale run therefore uses FP32 checkpoint inference and storage,
the reference FP32 math attention backend, TF32-disabled matrix
multiplication, FP32 ProteinMPNN vectors, and FP32 downstream latent caches.
BF16/FP16 remains available only for exploratory throughput runs.

A pretraining audit compared the earlier BF16-derived cache with strict FP32
inference on proteins of 62, 315, and 1,022 residues. Local-window relative
RMSE was 2.24%, 2.26%, and 1.77% respectively (cosine similarity 0.99975,
0.99975, and 0.99984). Because the model consumes WT-minus-mutant differences,
the definitive comparison does not assume that shared-vector subtraction will
cancel this quantization error.

## Native-FP32 6B scale result

The scale run retained the exact 600M-selected architecture and trained five
members with batch size 256 and a 50-epoch floor. Each member saw 5,215,750
examples and 20,400 optimizer steps. The frozen test remained the same 19,645
mutations from 19 source proteins.

| Metric | Precision-matched retained 6B | 6B v2 native FP32 |
| --- | ---: | ---: |
| Spearman | 0.818 | **0.837** |
| Pearson | 0.815 | **0.827** |
| MAE, kcal/mol | 0.516 | **0.504** |
| RMSE, kcal/mol | 0.706 | **0.692** |
| Stabilizer average precision | 0.347 | **0.422** |
| Stabilizers in top 50 | 39 | **41** |

Exact forward/reverse and self-mutation errors are both `0.0`. All five
prespecified promotion gates passed.

The native-FP32 6B double-mutant head trained for 16,700 optimizer steps and
4,262,650 examples. Its test result is:

| Frozen test component | Spearman | Pearson | MAE | RMSE |
| --- | ---: | ---: | ---: | ---: |
| Additive baseline | 0.611 | 0.601 | 0.959 | 1.230 |
| Additive + learned epistasis | **0.689** | **0.707** | **0.620** | **0.831** |
| Epistasis term on measured rows | 0.549 | 0.580 | 0.592 | 0.761 |

The maximum prediction change after swapping mutation order is `0.0`.

Transfer adapters were trained only as endpoint-specific diagnostics. ProTherm
reaches Spearman `0.452` and MAE `1.109` kcal/mol. MPTherm reaches Spearman
`0.191`; the GPCR delta-Tm adapter is negative at `-0.259`. The GPCR
crystallization score has overall Spearman `-0.161`, while its macro
within-assay Spearman is `0.427`; it remains rank-only and must not be reported
as kcal/mol or percentage stability. These results do not establish GPCR SOTA
on an independent receptor-disjoint ddG benchmark.

## WT-conditioned state-potential promotion

The final 6B model complements the mutant-aware hierarchy with a
WT-conditioned amino-acid state potential:
`ddG = phi(WT context, mutant) - phi(WT context, WT)`. The context includes
the ordered nine-residue ESM-C window, global WT mean, optional ProteinMPNN
vector, and membrane annotation. Validation selected state weight `0.55`.

On the same frozen 19-protein single-mutant test, the fusion improves Spearman
from `0.837` to `0.856`, Pearson from `0.827` to `0.849`, MAE from `0.504` to
`0.467` kcal/mol, RMSE from `0.692` to `0.640`, and stabilizer average
precision from `0.422` to `0.442`. Self mutation, reversal, and transitivity
errors are at numerical zero.

For doubles, a protein-held-out validation sweep selected state weight `0.50`
for constituent scores. The retrained unordered-set head improves frozen-test
total Spearman from `0.689` to `0.749`, MAE from `0.620` to `0.585` kcal/mol,
and RMSE from `0.831` to `0.782`; epistasis Spearman is `0.556`, and mutation
order changes predictions by exactly `0.0`.

The receptor-excluded MPTherm membrane adapter was rejected because validation
selected membrane weight `0.0`. On C5aR, the generic fusion reaches AUC
`0.689`, AP `0.256`, and 11 stabilizers in the top 50, below the retained GPCR
reranker (`0.696`, `0.317`, and 15/50). The production policy therefore uses
the fusion for signed generic ddG and keeps GPCR reranking separate and
unitless.

## GPCR family-consensus promotion

The application GPCR rank now combines three deliberately separate signals:
50% of the retained dual-backbone GPCR rank, 40% of the strict-FP32 6B
hierarchy/state-potential favorable-stability percentile, and 10% of a
target-excluded GPCRdb parent-family log-odds percentile. The evolutionary
term is `log((mutant count + 0.5) / (WT count + 0.5))`; the target sequence is
removed before counting. It is an orthogonal evolutionary prior, not an
additional trained stability head.

Weights were selected on 82 GPCR-tm development mutations. Macro
within-receptor Spearman is `0.338`, versus `0.115` for the retained GPCR rank
and `0.311` for the strict-6B generic score. A nested leave-one-receptor
reselection diagnostic reaches `0.241`. On the 12-row official confirmation
the selected consensus reaches macro Spearman `0.900`, versus `1.000` for the
retained rank. On the 277-row C5aR scan it improves AUC from `0.696` to
`0.718`, average precision from `0.317` to `0.377`, and top-50 stabilizer
recovery from 15/34 to 17/34. A 10,000-sample paired stratified bootstrap
places the AP difference at median `+0.056`, 95% interval `[-0.005, 0.119]`,
with probability of a positive difference `0.965`.

This is the promoted application ranking policy, not evidence of a calibrated
GPCR ddG model. Evolutionary preference also reflects function, expression,
and phylogeny. The official split and C5aR scan were consulted during earlier
experiments, so no untouched GPCR benchmark remains. Fifty C5aR positions
outside the alignment receive a neutral evolutionary score. Exact evidence is
in `docs/gpcr_evolutionary_consensus_audit.json`.

Two adjacent ideas were rejected. An explicit rigid-body-invariant GPCR bundle
geometry prior produced nested development macro Spearman `0.058`, below the
retained `0.115`. A phenotype-aware latent fit to the small GPCR workbook
looked strong on the development assays but collapsed on C5aR, indicating
assay/receptor overfit. Neither path enters the application model.

## Provenance and artifacts

- ESM-C 600M hierarchy cache:
  `embeddings/esmc_600m/hierarchy_cache.h5`
- ESM-C 600M native-FP32 hierarchy cache:
  `embeddings/esmc_600m/hierarchy_cache_fp32.h5`
- ESM-C 600M native-FP32 cache SHA-256:
  `4d538b8d3c51d157d2634a8f0034088a9d8eb2609f33535c1eaf9e17242d6bde`
- ESM-C checkpoint SHA-256:
  `8ef856e1a237ee3f995442df997a962e70057faadecf38fc0c8561bd3c2f4324`
- Hierarchy request-manifest SHA-256:
  `11bb967be5929c645cf9278064e3765e3a78541a8d564381f173c7b5dbd402fb`
- Selected checkpoint:
  `checkpoints/esmc_600m_v2/hierarchy_selected_ensemble.pt`
- Single report:
  `checkpoints/esmc_600m_v2/hierarchy_ablation.json`
- Multi report:
  `checkpoints/esmc_600m_v2/hierarchy_multi_metrics.json`
- Multi selection:
  `checkpoints/esmc_600m_v2/hierarchy_multi_selection.json`
- Transfer report:
  `checkpoints/esmc_600m_v2/hierarchy_transfer_metrics.json`
- ESM-C 6B native-FP32 hierarchy cache:
  `embeddings/esmc_6b/hierarchy_cache_fp32.h5`
- ESM-C 6B native-FP32 cache SHA-256:
  `d8c0d65ab251619a1e6f77c567057a9aa9392f80c52f2a3bcabceacd7d0ac459`
- ESM-C 6B checkpoint index SHA-256:
  `6846456e20e6ee2c37461f7bfc21d316d69bdaf165b925691afcb39e583244da`
- Selected 6B checkpoint:
  `checkpoints/esmc_6b_v2/hierarchy_selected_ensemble.pt`
- Selected 6B checkpoint SHA-256:
  `be9ff05c745eda74ef615abe04222245e436448079e6159c4a7a3673f47b37f5`
- 6B single, multi, and transfer reports:
  `checkpoints/esmc_6b_v2/hierarchy_scale_report.json`,
  `checkpoints/esmc_6b_v2/hierarchy_multi_metrics.json`, and
  `checkpoints/esmc_6b_v2/hierarchy_transfer_metrics.json`
- GPCR family-consensus selection and confirmation:
  `docs/gpcr_evolutionary_consensus_audit.json`
- Promoted strict-FP32 state-potential single and multi reports:
  `checkpoints/esmc_6b_state_potential_fp32/state_potential_report.json` and
  `checkpoints/esmc_6b_state_potential_fp32/hierarchy_multi_metrics.json`
