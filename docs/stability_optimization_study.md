# Stability optimization study

## Decision

The strict-FP32 ESM-C 6B hierarchy/state-potential fusion remains the
operational generic stability model. Its historical MegaScale result is MAE
`0.4670 kcal/mol`, RMSE `0.6402`, Spearman `0.8555`, and stabilizer average
precision `0.442` on 19,645 mutations from 19 protein-held-out proteins. A
protein bootstrap places the MAE 95% interval at `0.4267–0.5092`. This is useful
retrospective evidence, but it is not an untouched SOTA estimate because the
same test proteins were consulted by earlier promotion gates.

The requested `MAE < 0.30 kcal/mol` has not been demonstrated. Reaching it from
the current point estimate would require a `35.8%` reduction, and only one of
the 19 historical test proteins is currently below `0.30`. No tested loss,
calibration, surface, masked-marginal, or pretraining change supports scaling a
new head from ESM-C 600M to 6B yet.

## What the current model learns

The encoder is frozen ESM-C. For each mutation the head receives the contextual
embedding at the mutated residue, a sequence window of radius four, a whole
protein mean embedding, the WT and mutant amino-acid identities, and a
ProteinMPNN-derived structure vector when a structure is available. A second,
thermodynamically constrained head scores all 20 amino-acid states in the same
WT context and forms

`project_ddG(WT→mutant) = phi(mutant | WT context) - phi(WT | WT context)`.

The reverse mutation therefore has the opposite sign when evaluated in the
same context. Negative project ddG means stabilizing. The hierarchy and state
heads are ensembled and blended using protein-held-out development data. The
6B encoder is not fine-tuned; cached final-layer FP32 embeddings make the head
training tractable while preserving a clean 600M-to-6B scaling test.

Multiple mutations are represented as an unordered set. The deployed double
model reports the sum of constituent single-mutant effects separately from a
permutation-invariant learned epistasis correction. This is appropriate for
double-mutant screening; larger combinations and GPCR combinations remain
extrapolations.

## Real optimization runs

All neural runs used native FP32 with TF32 disabled. Main runs used batch size
256 and the complete 104,315-row training split; the surface state-potential
run trained three seeds for 50 epochs each (`15,647,250` row exposures). The
historical test column below is continuity evidence only and was never used to
justify a promotion in this study.

| Candidate | Development result | Historical continuity result | Decision |
|---|---:|---:|---|
| Current 6B hierarchy/state fusion | MAE `0.3709` | MAE `0.4670`, rho `0.8555` | Retain operational baseline |
| 600M Huber/MSE and tail weighting | Best MAE `0.4194` | Best MAE `0.5198` | Reject |
| ProteinMPNN + 13 target-independent surface features, one hierarchy seed | MAE `0.4428→0.4347` | MAE `0.5612→0.5492` | Useful signal, insufficient alone |
| Surface-aware state ensemble, three seeds × 50 epochs | Blend MAE `0.3840` | Blend MAE `0.4860` | Reject; no transfer gain over retained state fusion |
| ESM-C 600M masked amino-acid marginals | Blend MAE `0.3709→0.3573` | Blend MAE `0.4670→0.4736` | Reject; development gain did not transfer |
| Monotone kcal/mol calibration | MAE `0.3709→0.3470` | MAE `0.4670→0.4980` | Reject |
| Family-filtered absolute-ΔG pretraining + paired fine-tune | Random/pretrained blend MAE `0.3769/0.3870` | MAE `0.4891/0.4888` | Reject at 600M; validation worsened |

The masked-marginal cache contains all 136,333 rows / 7,591 unique sites and
was generated in one batched ESM-C pass with checkpoint and source hashes. Raw
masked log odds have Spearman `-0.467` with development ddG and `-0.508` with
historical-test ddG, so the evolutionary prior is real; the failure is
calibration/transfer, not absence of signal.

The official Tsuboyama tables were also re-curated locally with absolute WT and
mutant folding stability, confidence fields, and data-quality metadata. The
legacy cDNA subset used by the current model had stripped these fields. Full
pretraining excluded every current validation/test MMseqs family cluster and
then fine-tuned on the legacy training split. The pretraining corpus contained
222,059 rows from 251 proteins / 152 families. It ran 20 epochs, followed by
paired random-initialized and pretrained 50-epoch fine-tunes. Pretraining
worsened validation blend MAE by `0.0100` and Spearman by `0.0172`. Its
`0.0003` historical-test MAE advantage is not promotion evidence, so the
recipe was not scaled to 6B. Exact metrics and artifact hashes are recorded in
`docs/stability_optimization_audit.json`.

## Noise and split audit

- The historical test has no exact protein, sequence, or mutation overlap with
  training; maximum train identity is `47.22%`.
- The current validation split is not family-clean. Three validation proteins
  have train homologs at `96.88%`, `89.58%`, and `77.27%` identity, accounting
  for `21.47%` of validation rows.
- The legacy cDNA table has no replicate rows or uncertainty columns. Matching
  it back to the full source shows target-version drift of `0.0488` MAE with
  long tails.
- An oracle per-protein residual correction reaches only MAE `0.437`. Only an
  invalid same-site label oracle crosses the target (`0.2919`), which
  demonstrates why site/family leakage must be excluded.
- MegaScale proteins are 30–72 residues; the local GPCRs are 412–483 residues
  and only 10–12% identical to their nearest MegaScale sequence. Generic
  MegaScale accuracy therefore cannot establish GPCR kcal/mol accuracy.

Machine-readable details are in `docs/model_validation_audit.json`. The
read-only audit report used to construct it has SHA-256
`7a7d8fa8e4667d252c35a82078f339ed2997fb9e34d51b8179d6c1e70b19ca9f`.

## External state of the art and transferable ideas

Published results cannot yet be compared numerically to this project's MAE
because the splits and aggregation differ. The strongest directly relevant
architecture is SPURS: frozen ESM2 supplies sequence priors; a trainable,
masked ProteinMPNN path supplies structure features through cross-attention;
one forward pass produces a 20-amino-acid state potential; and a separate
decoder models multi-mutant epistasis. On its >25%-identity-filtered MegaScale
split, SPURS reported median per-protein Spearman `0.83` versus `0.77` for
ThermoMPNN. The published primary metric is not this project's row-micro MAE.

JanusDDG contributes a complementary constraint: combine differential and
contextual WT/mutant embeddings with bidirectional attention while enforcing
antisymmetry and encouraging transitivity. Stability Oracle contributes masked
structural microenvironments and thermodynamic permutations. ThermoMPNN shows
the value of adapting ProteinMPNN rather than treating its 128-dimensional
representation as an immutable side feature. Recent absolute-stability work
also supports jointly supervising WT ΔG, mutant ΔG, and ΔΔG, but current public
results are concentrated on small domains and do not validate long GPCRs.

Primary references:

- [SPURS, Nature Communications (2026)](https://www.nature.com/articles/s41467-025-67609-4)
- [ThermoMPNN, PNAS (2024)](https://pmc.ncbi.nlm.nih.gov/articles/PMC10861915/)
- [Stability Oracle, Nature Communications (2024)](https://www.nature.com/articles/s41467-024-49780-2)
- [JanusDDG, Communications Biology (2026)](https://www.nature.com/articles/s42003-026-09632-9)
- [Tsuboyama MegaScale assay, Nature (2023)](https://www.nature.com/articles/s41586-023-06328-6)
- [MGnify absolute-stability models (2026)](https://pmc.ncbi.nlm.nih.gov/articles/PMC13228446/)
- [IFUM unfolded-state model, Nature Communications (2026)](https://www.nature.com/articles/s41467-026-68637-4)

## Path to a credible SOTA attempt

1. Freeze a new prospective family-clustered and assay-held-out outer set. Test
   metrics must not participate in promotion. Report row-micro MAE,
   macro-protein MAE, and a protein/family bootstrap interval.
2. Rebuild development folds by MMseqs family, preserve source confidence,
   replicate, quality, clipping, assay, and target-derivation metadata, and use
   uncertainty-aware losses only after this curation.
3. Replace static concatenation with a SPURS-style trainable ProteinMPNN →
   ESM-C cross-attention adapter. Decode all 20 amino-acid potentials in one
   pass and keep exact sign/antisymmetry checks.
4. Retain an unordered additive-plus-epistasis decoder for multiple mutations.
   Add Janus-style reverse and cycle-consistency losses using label-preserving
   thermodynamic permutations.
5. Develop the architecture on ESM-C 600M across family folds. Scale the same
   logic to 6B only if the prespecified fold aggregate improves materially
   (suggested gate: at least `0.02 kcal/mol` MAE with no rank/AP regression).
6. For the actual GPCR, acquire a receptor-held-out quantitative mutation
   matrix including tested negatives and uncertainty. Until then, use generic
   ddG only as one ranking signal and validate a small diverse candidate panel
   experimentally; do not label GPCR rank as kcal/mol or crystallization
   probability.

## Full-structure implementation result (2026-07-23)

The proposed architecture has now been implemented and trained end to end in
native FP32. The full MegaScale stage contains 173 unique WT proteins rather
than hundreds of thousands of unique backbones, so ESM-C embeds each WT once
and the training loop gathers all mutation labels from a protein-level forward
pass. The persistent bank contains 136,333 singles and 114,109 doubles.

MMseqs clustering at 25% identity and 80% bidirectional coverage produced 114
families. Five development folds and a newly assigned outer partition hold out
whole families; no mutation rows or families cross partitions. The outer
metrics were not used by either the structure or epistasis promotion gates.

The SPURS-style candidate uses frozen full-protein ESM-C residue queries,
trainable leave-one-residue-out ProteinMPNN decoder states as structure
keys/values, gated cross-attention, and one-pass 20-amino-acid potentials. A
sequence warm start made the structure residual safe: development OOF MAE
improved from `0.53131` to `0.53081` kcal/mol, Spearman from `0.69826` to
`0.69937`, and stabilizer AP from `0.20348` to `0.20410`. The MAE gain was only
`0.00050`, far below the predeclared `0.02` scale gate, so the structure branch
was not promoted to the final checkpoint or scaled to 6B.

On the once-consumed outer family partition, the selected single-mutant model
has MAE `0.59824` kcal/mol (protein-bootstrap 95% CI `0.55861–0.64466`), RMSE
`0.79500`, Spearman `0.72580`, and stabilizer AP `0.23760` across 27,925 rows
and 25 proteins. This is the first clean estimate in the project and does not
support a sub-`0.30` or SOTA claim.

The unordered double-mutant head was also trained and evaluated. Learned
epistasis degraded development OOF MAE from additive `0.86710` to `0.98768`
and Spearman from `0.54540` to `0.49338`, so the development-only gate selected
additive prediction. The learned residual remains in the checkpoint and is
reported separately as a diagnostic; outer results cannot reverse that
selection.

The state-potential parameterization gives exact self and reverse identities.
The maximum three-state cycle residual observed in FP32 was
`4.77e-7` kcal/mol. Reproduction commands, source/checkpoint hashes, metrics,
and generated-artifact hashes are recorded in
`docs/full_structure_training_audit.json`.

This is a product-oriented route: one higher-value architecture experiment and
one defensible evaluation redesign, rather than additional global calibration
or loss sweeps on the already-consulted historical test.
