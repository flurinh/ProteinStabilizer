# GPCR model-selection audit

This document records compact experiments evaluated for the production scorer,
including the one accepted runtime prior and the candidates deliberately kept
out. The application target is ranking mutations for a previously unseen GPCR,
so a result must improve receptor-held-out or within-receptor ranking—not
merely fit a small pooled test set.

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

## Accepted runtime prior

ESM-C masked-marginal log odds add a cheap orthogonal signal without a new
trained head or an additional mutant-sequence forward pass. Each WT site is
masked once and all canonical substitutions are scored as
`log P(mutant | masked WT context) - log P(WT | masked WT context)`.

The selected scan consensus is 80% within-scan MPTherm delta-Tm percentile and
20% within-scan masked-marginal percentile. After correcting an audit that had
computed development percentiles in receptor domains containing test rows,
GPCR-tm development pooled Spearman increases from `0.055` to `0.059`; the
within-receptor macro Spearman remains effectively unchanged (`0.087` to
`0.086`). On the official test, the two multi-row receptors retain macro
Spearman `0.800`. On the independent 277-position C5aR scan, AUC increases from
`0.660` to `0.678`, average precision from `0.245` to `0.255`, and thermostable
substitutions in the top 50 from 12 to 13.

The supplied-workbook association was inconsistent across assay states, so the
masked score is not a universal continuous stability predictor. The
assay-specific GPCR workbook calibration is still reported, but it is excluded
from the primary consensus. Exact metrics and provenance are stored in
[`esmc_masked_marginal_audit.json`](esmc_masked_marginal_audit.json).

## Optional full-6B rerank

The full ESM-C 6B Megascale ensemble improves the generic single-mutant
protein-holdout result from Spearman `0.777` and MAE `0.556` to `0.818` and
`0.516`. Its ProTherm and MPTherm transfer heads also improve their
protein-disjoint holdouts to Spearman `0.481` and `0.274`, respectively.

A 6B-only GPCR rank did not preserve the official GPCR-tm receptor ranking.
The accepted application path therefore retains the validated 600M score and
limits 6B to a 25% second-stage prior:

```text
0.60 * 600M MPTherm percentile
+ 0.15 * 600M masked-marginal percentile
+ 0.25 * 6B MPTherm percentile
```

The 6B weight was selected from the bounded `0.00` to `0.25` development grid;
development and test percentiles were computed in separate within-receptor
domains. Development macro within-receptor Spearman increased from `0.086` to
`0.117`; four receptors improved and six tied. The two evaluable official-test
receptors increased from macro `0.800` to `1.000`, although this 12-row set has
been consulted by earlier project stages and is no longer an untouched
benchmark. On the independent C5aR scan, AUC increased from `0.678` to `0.696`,
average precision from `0.255` to `0.317`, and positives in the top 50 from 13
to 15. A paired stratified bootstrap placed the AP improvement at `+0.062`
with a 95% interval of `+0.005` to `+0.116`.

This is an optional slower reranker, not a replacement for the fast 600M first
pass. It also reports the materially stronger 6B general and ProTherm-adapted
ddG values as separate diagnostics. Exact metrics, cache manifests, checkpoint
hashes, and limitations are in
[`esmc6b_full_transfer_audit.json`](esmc6b_full_transfer_audit.json).

For mutation combinations, a separate 6B permutation-invariant epistasis head
is now available through `predict-6b`. On the protein-held-out Megascale-D
double-mutant test it reaches Spearman `0.623` and MAE `0.741`, versus `0.615`
and `1.142` for the 6B additive baseline and `0.545` and `0.851` for the
existing 600M learned estimate. This supports using the 6B model to prioritize
combinations after singles are selected, but it is not direct evidence for
GPCR double mutants; larger combinations are also outside its training domain.

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
| GPCRdb crystallization-construct positives | A delta-only contrastive head separated held-out construct receptors well (AUC `0.899`, top-1 recovery `0.685`), but official GPCR-tm test Spearman was only `0.126` and within-receptor macro Spearman was `0.000`. | Reject. Construct mutations encode receptor state, ligand, and crystallization-design choices rather than a transferable continuous stability endpoint. |
| ProteinGym membrane expression/abundance transfer | Nine membrane assays gave paper-held-out macro Spearman `0.404` for the best compact head. On GPCR-tm development, adding it reduced leave-one-receptor-out macro Spearman from `0.181` to `0.164`; its standalone official-test Spearman was `0.014`. | Reject. Expression, abundance, surface display, and membrane insertion are useful phenotypes but are not interchangeable with GPCR thermal stability. |
| C5aR experimental Ala/Leu scan | The published scan contributes 34 explicit thermostable substitutions on an otherwise stated exhaustive receptor scan. A delta classifier selected on GPCR-tm development raised leave-one-receptor-out macro Spearman from `0.181` to `0.316`, but official-test Spearman fell to `0.196`, within-receptor macro Spearman fell from `0.800` to `0.600`, and MAE increased from `3.588` to `3.909`. | Reject the single-receptor classifier. Its perfect training separation was overfit and did not transfer consistently. |
| Muk et al. Figure S2 GPCR TM scan | The vector supplement yielded 854 conservative labels after excluding ten gray shared positions: 82 receptor-specific positives and 772 negatives across A2A, β1AR, NTSR1, and AT1R. With 600M, the best leave-one-receptor-out macro AUC was `0.534`; on blind C5aR it recovered only 6 positives in the top 50. A full 6B re-audit reached only `0.522` with a fixed configuration and `0.503` under nested receptor-held-out selection, so C5aR and the official test were not consulted again. | Reject. Larger embeddings do not rescue the incomplete binary labels, and the recoverable scan does not improve new-receptor screening. |
| ESM-C 6B masked marginal alone | The public 6B checkpoint scored all 451 unique sites in 15 seconds on an RTX 5090. It improved the independent C5aR scan to AUC `0.698`, average precision `0.318`, and 16 positives in the top 50, while preserving official-test macro Spearman `0.800`. Its receptor-held-out development macro Spearman was `-0.036`, however. | Do not use the masked-only score. Full 6B delta-head retraining is now complete and is used only through the bounded optional reranker above. |
| Official DDGemb predictions | DDGemb reached development macro within-receptor Spearman `0.272`, but reversed to `-0.600` on the official GPCR-tm test. On C5aR it reached AUC `0.627`, average precision `0.185`, and 12 positives in the top 50. | Reject the score and the public-server dependency. |
| DDGemb S2450 fine-tuning | A five-fold compact delta-head ensemble improved the homology-reduced S669 benchmark from Spearman `0.531` to `0.541` and RMSE `1.413` to `1.404`. Its receptor-held-out GPCR-tm development macro Spearman was `-0.062`. | Keep the downloaded benchmark and embeddings, but do not add another runtime head for a small generic gain that does not transfer to GPCRs. |

## External-data audit

The exact compact metrics and provenance are also stored in
[`gpcr_external_data_audit.json`](gpcr_external_data_audit.json). The ESM-C 6B,
official DDGemb, and S2450/S669 experiments are recorded separately in
[`esmc6b_ddgemb_transfer_audit.json`](esmc6b_ddgemb_transfer_audit.json).

The external-data experiments used only frozen ESM-C mutant-minus-WT residue
deltas. ProteinGym contributed 34,425 rows from nine membrane expression,
abundance, surface-display, and insertion assays; CCR5's mixed binding/surface
assay was evaluated separately rather than silently pooled. The source parquet
SHA-256 was
`22e41fb41ea7da6f857aedf690975b298db06ab472b882aed279161b0369a152`,
and the 40,562-row embedding-request manifest SHA-256 was
`b5dacfdad8de0f4d548181685666360ea7808fe108c12806e9d34fc678624c66`.

The GPCRdb construct audit used repository commit
`705aa63be39752986b858b7442f63026e80b7758` and excluded every receptor
present in GPCR-tm or the supplied workbook before training. This left 24
external GPCRs, 109 sites, and 110 reported stabilizing substitutions. Its
2,071-row site-saturation request manifest SHA-256 was
`52cc40cf3bb9a89701a24204ffd2f40a3fc8ec6636d544d64774dfa82f50e2ec`.

The C5aR audit used Supplementary Table S6 from Muk et al. to identify all 34
reported thermostable substitutions. The paper states that 283 mutants span
Val35 through Leu311, although that inclusive canonical UniProt interval
contains 277 positions. No six undocumented rows were invented. The audit
therefore used the explicit 277-position interval and is not treated as a
production training source. The supplement SHA-256 was
`e93e12e263d77125108e2e8f16ae10f9c3c68586cff01e7208ef69db6bf9e6d8`;
the P21730 sequence SHA-256 was
`dc48c194272465c04ae4727f3710ca1c75b6ca77006ea1730ed45ea75035d730`;
and the ESM-C request manifest SHA-256 was
`2619d749f09cada931bb814030a559c984816a2648203a2ee69cdff0691bab24`.

The larger 1,231-mutant, five-assay GPCR alanine-scan matrix described by Muk
et al. would be a high-value transfer source, but the raw per-mutant table is
not present in the published supplement. The article states that Christopher
Tate provided those measurements. It should only be added if the original
table can be obtained with receptor, mutation, assay state, measured score,
and tested-negative rows intact.

The vector artwork in Supplementary Figure S2 does preserve a conservative
subset of those labels. Receptor-colored cells identify receptor-specific
thermostabilizing TM positions; uncolored or differently colored non-gray
positions provide negatives for the receptor, while the ten gray shared
positions cannot be assigned safely and were omitted. GPCRdb generic-number
mappings produced 854 Ala/Leu substitutions across four receptors (82 positive,
772 negative). Their ESM-C request manifest was
`bf009d79a820fb681204455dc378349b0e4836a32d8c79fbad9744649a7f576d`.
Neither a raw-delta head nor a head built on MPTherm-pretrained latent
coordinates survived whole-receptor transfer or the independent C5aR gate, so
the reconstructed rows are audit evidence rather than production training
data. Repeating the experiment with full 2560-dimensional ESM-C 6B deltas and
compact 6B general/ProTherm/MPTherm latent representations did not help: the
best fixed leave-one-receptor-out macro AUC was `0.522`, and nested
representation selection reached `0.503`. That failed the development gate, so
the independent C5aR scan and official GPCR-tm test were left unconsulted for
the 6B candidate. Exact provenance and fold results are in
[`esmc6b_muk_thermostability_audit.json`](esmc6b_muk_thermostability_audit.json).

## ThermoMPNN provenance

The audit used the official ThermoMPNN repository at commit
`2b04fd370e399911b1fa5848112cc9013f084110` and its default checkpoint
(SHA-256 `af449118ccfb4e34971d802321907ed82a8a035ae80f055bf8fc56009a4a838a`).
The 11 GPCR structures came from the GPCR-tm structure archive (SHA-256
`c0aae6f768d03e07221322cbc4b49b2d31494b2e7d406d80f289d6a2518b4ab9`).
Native UniProt residue numbering and WT identities matched all 97 GPCR-tm rows.

## Current application decision

Use the stored ESM-C mutant-minus-WT residue deltas and compact production heads
for high-throughput screening. Rank the fast first pass with the 80% MPTherm /
20% masked consensus. When GPU time permits, run the separate full-6B command
as a second-stage reranker and retain both the original and reranked positions
in the output. Use the full-6B `predict-6b` command for proposed double mutants,
keeping the additive and epistasis components visible. For a conservative
experimental set, require general-ddG support for some candidates and
deliberately include a smaller number of high-consensus disagreements. Test
several diverse substitutions rather than relying on one top prediction. The
remaining accuracy bottleneck is substantially larger, new-receptor GPCR
stability data, not another adapter fitted to the present small benchmarks.

## External data sources

- ProteinGym: https://github.com/OATML-Markslab/ProteinGym
- GPCRdb construct data: https://github.com/protwis/gpcrdb_data
- GPCR alanine-scan classifier study: https://doi.org/10.1016/j.bpj.2019.10.023
- DDGemb datasets: https://ddgemb.biocomp.unibo.it/datasets/
- ESM-C 6B model: https://huggingface.co/biohub/ESMC-6B
