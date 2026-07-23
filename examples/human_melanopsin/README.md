# Human melanopsin mutation screen

This example uses reviewed human melanopsin (`Q9UHM6`, `OPN4_HUMAN`) from
UniProtKB release `2026_02`. The canonical sequence has 478 residues. Exact
source metadata and sequence hashes are pinned in `provenance.json`.

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

Run the highest-accuracy strict-FP32 6B screen:

```bash
bash scripts/setup_esmc6b_env.sh  # first run only
bash examples/human_melanopsin/run_screen.sh 6b
```

For a short pipeline smoke test, reduce the expensive exact rerank:

```bash
RERANK_TOP=16 TOP=10 bash examples/human_melanopsin/run_screen.sh 600m
```

The default mask leaves 270 mutable positions and 5,130 single-substitution
candidates. Two-stage screening embeds the WT once, then embeds at most 128
mutant sequences instead of all 5,130. Identical reruns reuse the application
embedding cache.

Default outputs are written below
`artifacts/examples/human_melanopsin/`. The full `*_screen.csv` includes every
allowed state-potential candidate and its scoring stage. The adjacent
`*_screen.shortlist.csv` contains only exact-reranked, site-diverse suggestions.
Negative exact ddG is the model's stabilizing direction and is reported in
kcal/mol; it is a screening estimate, not proof that function is preserved.
