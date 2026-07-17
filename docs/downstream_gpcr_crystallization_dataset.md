# Downstream dataset: GPCR construct-stabilizing and crystallization mutations

Status: deferred. This is a downstream data and model track, not part of the
current stability model.

## Objective

Build a GPCR-specific dataset from experimentally solved structures and their
engineered constructs. Compare each deposited receptor construct with its
canonical wild-type sequence and use construct substitutions primarily as weak
evidence for practical receptor stabilization. In GPCR crystallography,
researchers commonly stabilize the receptor in a ligand-bound state and in
detergent so that a homogeneous receptor-compound complex survives purification
and crystallization. Expression, conformational locking, and crystal packing
remain related secondary mechanisms.

The useful distinction is therefore not “stability versus crystallization,”
but measured physical stability versus a weak construct-derived stabilization
label:

- a construct mutation is not a measured delta-delta-G or delta-Tm label;
- many substitutions are deliberately selected for thermostability, detergent
  stability, or conformational homogeneity and should contribute positive
  stabilization evidence;
- some stabilize one ligand/state rather than the receptor generally;
- some help trafficking, expression, detergent tolerance, or crystal packing;
- some are neutral passengers in a successful multi-mutation construct.

The primary weak target should therefore be named a
`construct_stabilization_score`. A broader `crystallization_assistance_score`
can capture other engineering effects. Neither should be labeled in kcal/mol
or degrees Celsius unless the source provides an actual measurement.

## Proposed data sources

- RCSB PDB search results and coordinates, initially restricted to X-ray
  structures of GPCR receptor chains;
- SIFTS mappings between PDB residues and UniProt canonical positions;
- UniProt canonical wild-type sequences and sequence-version provenance;
- GPCRdb construct annotations, receptor state, ligand, and engineered
  mutations where available;
- primary-article annotations that explicitly identify thermostabilizing,
  expression-enhancing, or crystallization mutations.

Cryo-EM structures should be collected separately. They are useful evidence
for constructability but do not share the same crystallization selection
process as X-ray structures.

## Canonical record

Each structure/construct record should retain:

- PDB ID, deposition/release date, experimental method, resolution, chain,
  biological assembly, receptor state, ligand, and source publication;
- UniProt accession, canonical sequence version, receptor family, and species;
- the deposited `SEQRES` sequence, observed `ATOM` sequence, and mapped
  canonical wild-type sequence;
- substitutions, deletions, insertions, truncations, unresolved residues,
  fusion partners, tags, modified residues, and chain breaks as separate
  fields;
- mapping confidence and all source URLs, retrieval dates, source hashes, and
  parser version.

Only substitutions confidently mapped inside the receptor chain should become
mutation examples. Fusion proteins, linkers, affinity tags, signal peptides,
and numbering offsets must not be interpreted as receptor mutations.

## Soft-label design

Treat the task as positive-unlabeled learning rather than ordinary binary
classification.

Use a hierarchy of evidence:

1. **Strong stabilization evidence:** the paper, supplement, or GPCRdb
   explicitly identifies the mutation as thermostabilizing, reports a
   stability screen, or provides delta-Tm or another physical measurement.
   Preserve the experimental value and assay conditions when available.
2. **Moderate construct-stabilization evidence:** an engineered substitution
   recurs across independent structures, is retained across construct
   generations, or appears in multiple ligand states or laboratories.
3. **Weak crystallization evidence:** an otherwise unexplained substitution is
   present in one successful construct. It may assist expression, state
   locking, packing, or purification and should receive less stabilization
   credit.

Within those tiers, an engineered substitution receives stronger positive
weight when one or more of the following hold:

1. the primary paper or GPCRdb explicitly calls it stabilizing or
   crystallization-enabling;
2. it recurs in independently deposited constructs, laboratories, ligand
   states, or related receptors;
3. it is retained across later construct generations;
4. the structure is high quality and the mutation is resolved in the receptor;
5. the construct contains few substitutions, making attribution less
   ambiguous.

Confidence should decrease for large mutation sets, unexplained construct
differences, low-confidence sequence mappings, mutation sites inside fusion
boundaries, and substitutions observed only once without an annotation.
Unmutated residues are unlabeled, not negatives. Explicitly tested failures
would be true negatives if they can be recovered from publications or
supplements.

The model should expose at least:

- a construct-stabilization rank as the primary output;
- a crystallization-assistance rank;
- the evidence/confidence weight behind that rank;
- separate heads for measured/explicit thermostabilization,
  expression/purification assistance, and state locking if enough annotations
  exist.

## Leakage-resistant evaluation

- Hold out whole receptors, and preferably whole receptor families, for the
  primary transfer evaluation.
- Add a temporal split based on PDB release date so later constructs test
  prospective recovery.
- Group all observations of the same `(receptor, mutation site)` together.
- Deduplicate repeated PDB structures of the same construct before splitting.
- Keep constructs derived from the same engineering campaign in one fold.
- Report retrieval metrics such as average precision, enrichment, and recovery
  of annotated construct mutations in the top 20/50, not kcal/mol error.

The existing GPCR-tm and C5aR stability benchmarks remain direct stability
checks. The construct-stabilization score may complement the current stability
rank if it transfers, but its weak label must not be presented as a physical
stability measurement.

## Implementation sequence

1. Query and snapshot all human and non-human GPCR X-ray entries from RCSB.
2. Download coordinates and experimental metadata to generated storage under
   `/data/fast`, with hashes and retrieval manifests.
3. Resolve receptor chains to UniProt through SIFTS and reconcile canonical,
   `SEQRES`, and observed sequences.
4. Extract construct edits while explicitly separating substitutions from
   truncations, fusions, insertions, tags, and missing density.
5. Cross-reference GPCRdb and primary-paper mutation annotations.
6. Collapse duplicate constructs and assign evidence-based soft labels.
7. Build receptor/family-held-out and temporal splits before model fitting.
8. Develop the new scoring logic with frozen ESM-C 600M embeddings, then port
   the identical logic to ESM-C 6B for the production decision.
9. Train a primary construct-stabilization objective and, if the annotations
   support it, secondary crystallization/expression/state objectives.
10. Compare the resulting 6B construct-stabilization score with a small prior
    added to the current 6B GPCR stability rank.

## Promotion gate

Promote this track only if the 6B version improves held-out-receptor or temporal
recovery of independently annotated stabilizing construct mutations. Adding it
to the main stability rank must also preserve or improve GPCR-tm and C5aR.
Until then, report construct stabilization as an experimental, non-physical
score.
