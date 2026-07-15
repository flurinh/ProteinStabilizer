# ProteinStabilizer training report

Date: 2026-07-15
Seed: `20260715`

Runtime: Python 3.13, PyTorch 2.10.0+cu128, ESM 3.2.1,
scikit-learn 1.8.0, h5py 3.15.1.

## Outcome

A frozen ESM-C 600M encoder was used to generate a reusable mutation-site
embedding cache. Compact trainable heads now support single substitutions,
additive-plus-epistatic double substitutions, GPCR-specific ranking, direct
prediction, and saturation screening.

## Data and artifacts

- ESM-C sequences: 253,731 unique sequences.
- Cached contextual residue sites: 375,487.
- Embedding cache size: 910,568,366 bytes.
- Embedding runtime: 249.1 seconds on NVIDIA RTX 5090.
- ESM-C checkpoint SHA-256:
  `8ef856e1a237ee3f995442df997a962e70057faadecf38fc0c8561bd3c2f4324`.
- Embedding request manifest SHA-256:
  `e28d4238990c7a1b77692eb4c6f9e6828f02a823a5e9656ca2a6097b7e134f88`.
- Completed embedding cache SHA-256:
  `867416eddb7b8cd256f8a68b128fdc230ee03eca902cf3412a637e1b3a0d302a`.

Training rows:

- single train/validation source: 116,688 rows, 116 proteins;
- single held-out test: 19,645 rows, 19 proteins;
- double train/validation/test: 85,253 / 10,282 / 18,574 rows;
- double rows with measured additive/epistasis targets:
  78,817 / 9,704 / 16,923;
- GPCR fine-tuning: 133 assay rows, 96 unique mutation sites;
- GPCR train/validation/test: 58 / 19 / 19 sites.

## Held-out results

Single-mutant thermodynamic test:

- Spearman: 0.7539
- Pearson: 0.7624
- MAE: 0.5823
- RMSE: 0.7857

Double-mutant thermodynamic test:

- learned total: Spearman 0.4924, Pearson 0.4515, MAE 0.9066, RMSE 1.2197;
- additive baseline: Spearman 0.5600, Pearson 0.5178, MAE 1.1924,
  RMSE 1.5088;
- direct epistasis target: Spearman 0.4944, Pearson 0.4966, MAE 0.6284,
  RMSE 0.8045.

GPCR assay-balanced grouped test:

- 26 assay rows from 19 held-out mutation sites;
- macro within-assay Spearman: 0.370;
- uncalibrated pretrained macro within-assay Spearman: -0.237.

## Checkpoint integrity

- `single_head.pt`:
  `79b37482e558aeb376436efa7de3cb578181db43f0a3038b51fe113de8d73fc8`
- `multi_head.pt`:
  `361bc844d107dc56f84938d614d1ff0aeacd843a73c63abd7ffa6839e881170c`
- `gpcr_calibration.joblib`:
  `db676afc0c73ebad26e7256cbd6a59795e26905fe1d327cf5438773fff266df8`

## Interpretation

The single-mutant model is suitable for candidate prioritization. For multiple
mutations, the learned epistasis term improves calibrated error but reduces
held-out rank correlation relative to simple addition, so applications should
inspect both. GPCR calibration is useful as a secondary ranking prior but is
not yet strong enough to replace the general score or experimental validation.
