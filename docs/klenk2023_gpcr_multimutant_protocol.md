# Frozen GPCR multi-mutant transfer check

Status: frozen before model inference on 2026-07-24; consumed by the first
fixed-model evaluation later that day.

The next external check uses the source data from Klenk et al.,
“A Vaccinia-based system for directed evolution of GPCRs in mammalian cells”
([DOI 10.1038/s41467-023-37191-8](https://doi.org/10.1038/s41467-023-37191-8)).
The study reports replicate ligand-binding thermal-shift measurements for
multi-mutant NTR1 and PTH1R variants.

The dataset definition is deliberately fixed before any ProteinStabilizer
predictions are generated. `scripts/prepare_klenk2023_gpcr_multimutant.py`
verifies the publication source-data spreadsheet, supplement, and canonical
UniProt FASTA hashes. It retains only pure-substitution variants whose complete
mutation set is recoverable from the source package:

- five NTR1 variants with three to six substitutions and four ΔTm replicates;
- three PTH1R variants with four, five, or eight substitutions and four ΔTm
  replicates.

PTH1R is the only new-receptor lockbox. It is absent from the project's
GPCR-tm, supplied GPCR workbook, Muk scan, C5aR scan, and MPTherm receptor
sets. NTR1 has already appeared in GPCR and MPTherm development data, so its
five rows are a same-receptor multi-mutant continuity diagnostic only.

The primary PTH1R endpoint is ΔTm measured without mini-Gs. The published
thermal assay removed PTH1R residues 1–170, whereas the application model sees
the canonical receptor sequence; this construct/context mismatch must remain
visible in the result. NTR1 variants all contain R167L, which disrupts the DRY
signaling motif and would be excluded by a functional protected mask. They
test stability prediction, not acceptable application design.

## Frozen evaluation rules

- Use the already promoted checkpoints without fitting, calibration, blend
  selection, or threshold tuning.
- Run the cache-efficient route: one frozen ESM-C 6B WT state per receptor,
  bounded 600M masked-site contexts, and no mutant-sequence embeddings.
- For each variant, sum the fixed constituent expected-ΔΔG predictions. Any
  learned epistasis value for more than two mutations is reported only as an
  out-of-domain diagnostic because the epistasis head was trained on doubles.
- Compare `-predicted ΔΔG` with experimental ΔTm for favorable rank and sign.
  Do not compute MAE or otherwise equate kcal/mol with degrees Celsius.
- Report PTH1R and NTR1 separately. The PTH1R result is too small for a SOTA
  claim, but it can reveal a catastrophic new-receptor direction failure.
- After the first fixed-model result, this dataset is consumed and must not be
  described as untouched.

Run the deterministic preparation step with:

```bash
python scripts/prepare_klenk2023_gpcr_multimutant.py \
  --output-dir \
  /data/fast/tmp/protein-stabilizer/prospective-gpcr/klenk2023
```

Then run the fixed cache-efficient evaluation with:

```bash
MASKED_DEVICE=cuda STATE_DEVICE=cuda \
  bash scripts/run_klenk2023_gpcr_multimutant.sh
```

The first run was executed on CPU because the sole GPU was occupied; both
encoders retained float32 inference and storage. It required two 6B WT
encodings, 24 600M masked-site contexts, and no mutant-sequence embeddings.
Persisted caches make subsequent scoring encoder-free.

## First fixed-model result

The only new receptor, PTH1R, has three experimentally destabilizing variants.
The fixed additive expected-ΔΔG prediction got favorable direction right for
all three and ranked their ΔTm severity at Spearman `0.500`. This is a useful
catastrophic-direction check, not a performance claim: the cohort is tiny,
contains no stabilizing PTH1R variant, and uses a truncated assay construct.

The five prior-seen NTR1 variants are all experimentally stabilizing. Their
additive expected-ΔΔG rank reached Spearman `0.600`, but every predicted
direction was wrong. These variants share a signaling-disrupting DRY-site
mutation and contain three to six substitutions, so the result directly
rejects reliable application use of simple single-mutant sums for evolved
GPCR combinations. The operational policy remains bounded single-mutant
screening with functional masks, followed by experimental combination tests.
Machine-readable results and hashes are in
`docs/klenk2023_gpcr_multimutant_audit.json`.
