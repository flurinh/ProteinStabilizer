# Human melanopsin mutation screen

This example uses reviewed human melanopsin (`Q9UHM6`, `OPN4_HUMAN`) from
UniProtKB release `2026_02`. The canonical sequence has 478 residues. Exact
source metadata and sequence hashes are pinned in `provenance.json`.
The run scripts resolve the current AlphaFold DB PDB through its API, require
an exact match to this sequence, and cache the PDB plus source hashes under
`artifacts/structures/alphafold/Q9UHM6/`. Residues below pLDDT 70 are excluded
from the ProteinMPNN neighborhood graph and treated as structure-missing. Set
`ALPHAFOLD_MIN_PLDDT` to change that threshold explicitly.

The supplied protected mask limits suggestions to the annotated seven-helix
core and excludes the annotated disulfide bond and retinal-linked lysine, plus
the sequence-localized DRY and NPIIY activation motifs. It is deliberately a
minimal starting mask. Add known construct-specific ligand contacts, signaling
interfaces, trafficking determinants, modifications, and experimental
constraints before treating the output as a wet-lab design list.

Run the fast 600M development screen from the repository root:

```bash
bash examples/human_melanopsin/run_screen.sh 600m
```

Run the promoted low-cost general-ddG/state screen:

```bash
bash examples/human_melanopsin/run_accuracy_screen.sh
```

This path uses one full WT ESM-C pass plus batched masked-site contexts; it
never embeds mutant sequences. It reports two deliberately separate outputs:
the ensemble is the expected general-domain ddG, while the exact state
component determines stabilizer rank. By default, the shortlist requires both
components to predict negative ddG. On the supplied 270-site mask, 5,130
substitutions are evaluated in one WT batch plus 16 masked batches.

Run the scaled accuracy model with a strict-FP32 6B WT state and the validated
600M masked prior:

```bash
bash examples/human_melanopsin/run_accuracy_6b.sh
```

The script deliberately uses two environments. First, the 600M runtime writes
target-specific, provenance-checked masked-site and WT-state caches in one
loaded-encoder session. Then the 6B runtime loads the promoted multiscale
accuracy checkpoint, reuses a persistent 6B WT embedding cache, and consumes
both 600M caches without loading the 600M encoder. Expected ΔΔG uses the
fold-0-selected, confirmation-gated monotone calibration over the 6B WT state,
600M WT state, and portable masked/structure prior; candidate order remains
the exact 6B state score. Repeated runs with the same sequence, mask, and
checkpoints require no ESM-C forward passes.

Melanopsin is 478 residues, whereas this quantitative head was trained on
30–72-residue MegaScale proteins with no membrane labels. Its ddG field is
therefore a domain-shift extrapolation, not GPCR-calibrated kcal/mol. Use the
shortlist as orthogonal agreement evidence alongside the strict-FP32 6B GPCR
screen, then validate experimentally.

Run the highest-accuracy strict-FP32 6B screen:

```bash
bash scripts/setup_esmc6b_env.sh  # first run only
bash examples/human_melanopsin/run_screen.sh 6b
```

For a short pipeline smoke test, reduce the expensive exact rerank:

```bash
RERANK_TOP=16 TOP=10 bash examples/human_melanopsin/run_screen.sh 600m
```

After the exact single screen, design bounded double-mutant combinations:

```bash
bash examples/human_melanopsin/run_pairs.sh 6b
```

This admits the best 20 exact, individually stabilizing singles, ranks their
valid unordered pairs by additive single-mutant ddG, and embeds only the best
64 joint sequences for the learned permutation-invariant epistasis correction.
WT and single-mutant embeddings are reused from the application cache. For a
quick integration test:

```bash
PAIR_RERANK_TOP=2 TOP=2 bash examples/human_melanopsin/run_pairs.sh 6b
```

The default mask leaves 270 mutable positions and 5,130 single-substitution
candidates. Two-stage screening embeds the WT once, then embeds at most 128
mutant sequences instead of all 5,130. Identical reruns reuse the application
embedding cache. ProteinMPNN also encodes the WT backbone only once per
command; no mutant structures are predicted.

Default outputs are written below
`artifacts/examples/human_melanopsin/`. The full `*_screen.csv` includes every
allowed state-potential candidate and its scoring stage. The adjacent
`*_screen.shortlist.csv` contains only exact-reranked, site-diverse suggestions.
The pair workflow similarly writes `*_pairs.csv` with every additive-prescreen
pair and `*_pairs.shortlist.csv` with exact epistasis-reranked, site-diverse
double mutants.
Negative exact ddG is the model's stabilizing direction and is reported in
kcal/mol; it is a screening estimate, not proof that function is preserved.
