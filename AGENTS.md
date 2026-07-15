# Project invariants

- Use frozen ESM-C final-layer contextual residue embeddings; do not add a
  solubility branch without a new user decision.
- Negative general-model ddG means stabilizing. GPCR calibration is a ranking
  score and must not be labeled as kcal/mol or percentage stability.
- Never random-split mutation rows. Hold out whole source proteins for general
  evaluation and whole `(protein, mutation site)` groups for GPCR evaluation.
- Multiple-mutation inputs are unordered sets. Preserve permutation invariance
  and report additive and epistasis components separately.
- External data, embeddings, and feature tensors are generated artifacts. Keep
  source hashes and ESM-C checkpoint provenance in every cache/checkpoint.
- Run `PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q -p no:cacheprovider`
  and `git diff --check` before committing code changes.
