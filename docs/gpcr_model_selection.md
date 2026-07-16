# GPCR model-selection audit

This document records compact experiments that were evaluated and deliberately
kept out of the production scorer. The application target is ranking mutations
for a previously unseen GPCR, so a result must improve receptor-held-out or
within-receptor ranking—not merely fit a small pooled test set.

## Evaluation policy

- GPCR-tm model selection uses 82 official training rows after excluding three
  training substitutions at official test mutation sites.
- Candidate choices use leave-one-receptor-out predictions on those 82 rows.
- The official 12-row GPCR-tm test is evaluated only after candidate selection.
- The supplied GPCR workbook is also audited with exact leave-one-receptor-out
  folds. Each fold retrains the MPTherm head after quarantining upstream rows
  from the held-out receptor.
- A more complex candidate is promoted only for a material, consistent gain.
  A small pooled correlation increase is insufficient if within-receptor rank
  or the new-receptor diagnostic becomes worse.

The production reference is the ESM-C 600M delta ensemble plus its MPTherm
transfer head. On the official GPCR-tm test, the MPTherm head reaches Spearman
`0.371`, Pearson `0.340`, MAE `3.588`, and RMSE `4.435`. Its macro Spearman for
the two test receptors having more than one observation is `0.800`.

## Rejected candidates

| Candidate | Selection/evaluation evidence | Decision |
| --- | --- | --- |
| Five-member ProTherm transfer ensemble | ProTherm test Spearman improved from `0.409` to `0.431`, but exact GPCR workbook leave-one-receptor-out macro Spearman decreased from `0.171` to `0.168`. | Keep the single transfer head used by the application calibration. |
| Five-member MPTherm transfer ensemble | MPTherm test Spearman decreased from `0.222` to `0.183`. | Reject. |
| Official ThermoMPNN | GPCR-tm official test: Spearman `0.126`, Pearson `-0.016`, MAE `3.805`, RMSE `4.542`. | Reject as a standalone GPCR score. |
| MPTherm + ThermoMPNN + base ESM-C ridge | Selected on development leave-one-receptor-out macro Spearman `0.379`. Official test Spearman rose slightly from `0.371` to `0.385`, but macro within-receptor Spearman fell from `0.800` to `0.600`, and MAE increased from `3.588` to `3.602`. | Reject the extra checkpoint, PDB input, and graph-model runtime for a marginal pooled gain. |
| Local structure descriptors | Tested relative SASA, nonlocal contact counts, bundle-axis depth, radial position, side-chain orientation, AlphaFold confidence, mutation-property changes, and burial interactions. Development selection preferred a two-score ESM-C model without structure (macro Spearman `0.181`). Its official-test Spearman was `0.357` and within-receptor macro Spearman `0.200`. | Keep the runtime sequence-only. |
| Mutation physicochemical priors | Adding hydropathy, volume, charge, polarity, aromaticity, and backbone-breaker changes reduced the supplied-workbook site test macro Spearman from `0.601` to `0.526` and the receptor-held-out diagnostic. | Reject. Contextual ESM-C deltas already contain the useful part of this signal. |
| Ensemble-disagreement re-ranking | Member standard deviation has weak correlation with generic absolute error (`0.179`) and very weak correlation on the GPCR workbook (`0.079`). | Retain `pretrained_ddg_std` as a caution flag, not a ranking penalty or calibrated interval. |

## ThermoMPNN provenance

The audit used the official ThermoMPNN repository at commit
`2b04fd370e399911b1fa5848112cc9013f084110` and its default checkpoint
(SHA-256 `af449118ccfb4e34971d802321907ed82a8a035ae80f055bf8fc56009a4a838a`).
The 11 GPCR structures came from the GPCR-tm structure archive (SHA-256
`c0aae6f768d03e07221322cbc4b49b2d31494b2e7d406d80f289d6a2518b4ab9`).
Native UniProt residue numbering and WT identities matched all 97 GPCR-tm rows.

## Current application decision

Use the stored ESM-C mutant-minus-WT residue deltas and the compact production
heads for high-throughput screening. Keep the GPCR workbook calibration at its
validated 10% consensus weight. Treat the resulting list as an experimental
prior and test several diverse substitutions rather than relying on one top
prediction. The next materially different model upgrade should be ESM-C 6B or
substantially larger GPCR stability data, not another small adapter selected on
the present benchmarks.
