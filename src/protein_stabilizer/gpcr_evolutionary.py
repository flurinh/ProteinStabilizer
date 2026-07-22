"""GPCR-family evolutionary log odds for mutation screening."""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from .data import Mutation, normalize_sequence
from .embeddings import file_sha256
from .structure import _aligned_structure_indices


GPCR_SCREENING_WEIGHTS = {
    "retained_gpcr_rank": 0.50,
    "generic_6b_stability_rank": 0.40,
    "gpcrdb_family_rank": 0.10,
}


@dataclass(frozen=True)
class GPCRDBEvolutionaryResult:
    """Mutation-aligned evolutionary scores and source provenance."""

    log_odds: np.ndarray
    observed: np.ndarray
    provenance: dict[str, object]


class GPCRDBEvolutionaryCache:
    """Read pinned GPCRdb protein metadata and family alignments.

    The cache is a directory of raw JSON responses.  Protein metadata objects
    identify the GPCRdb entry name for each accession; alignment objects map
    entry names to equal-length aligned sequences and contain ``CONSENSUS``.
    """

    def __init__(self, cache_dir: Path) -> None:
        self.cache_dir = Path(cache_dir).resolve()
        if not self.cache_dir.is_dir():
            raise FileNotFoundError(
                f"GPCRdb evolutionary cache does not exist: {self.cache_dir}"
            )
        self._protein: dict[str, tuple[dict[str, object], Path]] = {}
        self._alignment: dict[str, tuple[dict[str, str], Path]] = {}
        for path in sorted(self.cache_dir.glob("*.json")):
            payload = json.loads(path.read_text(encoding="utf-8"))
            if (
                isinstance(payload, dict)
                and "accession" in payload
                and "entry_name" in payload
                and "sequence" in payload
            ):
                accession = str(payload["accession"])
                if accession in self._protein:
                    raise RuntimeError(
                        f"duplicate GPCRdb metadata for {accession}"
                    )
                self._protein[accession] = (payload, path)
                continue
            if (
                isinstance(payload, dict)
                and "CONSENSUS" in payload
                and payload
                and all(isinstance(value, str) for value in payload.values())
            ):
                lengths = {len(value) for value in payload.values()}
                if len(lengths) != 1:
                    raise RuntimeError(
                        f"{path.name} alignment sequences have unequal lengths"
                    )
                alignment = {
                    str(key): str(value).upper()
                    for key, value in payload.items()
                }
                for entry_name in alignment:
                    if entry_name == "CONSENSUS":
                        continue
                    if entry_name in self._alignment:
                        raise RuntimeError(
                            f"duplicate GPCRdb alignment for {entry_name}"
                        )
                    self._alignment[entry_name] = (alignment, path)
        if not self._protein or not self._alignment:
            raise RuntimeError(
                "GPCRdb evolutionary cache lacks metadata or alignments"
            )

    def canonical_sequence(self, accession: str) -> str:
        """Return the pinned GPCRdb canonical sequence for an accession."""

        try:
            metadata, _ = self._protein[str(accession)]
        except KeyError as error:
            raise KeyError(
                f"GPCRdb cache has no metadata for {accession}"
            ) from error
        return str(metadata["sequence"]).upper()

    def score_mutations(
        self,
        accession: str,
        target_sequence: str,
        mutations: Sequence[Mutation],
        *,
        pseudocount: float = 0.5,
        sequence_scope: str = "family",
    ) -> GPCRDBEvolutionaryResult:
        """Return log P(mutant)/P(WT) from a target-excluded alignment.

        ``family`` uses all entries in the GPCRdb family alignment.
        ``ortholog`` restricts counts to entries sharing the target entry's
        receptor prefix (the part before the species suffix).
        """

        if not math.isfinite(pseudocount) or pseudocount <= 0.0:
            raise ValueError("evolutionary pseudocount must be positive")
        if sequence_scope not in {"family", "ortholog"}:
            raise ValueError(
                "evolutionary sequence scope must be family or ortholog"
            )
        try:
            metadata, metadata_path = self._protein[str(accession)]
        except KeyError as error:
            raise KeyError(
                f"GPCRdb cache has no metadata for {accession}"
            ) from error
        entry_name = str(metadata["entry_name"])
        try:
            alignment, alignment_path = self._alignment[entry_name]
        except KeyError as error:
            raise KeyError(
                f"GPCRdb cache has no family alignment for {entry_name}"
            ) from error
        sequence = str(target_sequence).upper()
        canonical = str(metadata["sequence"]).upper()
        if sequence != canonical:
            raise ValueError(
                f"{accession} target sequence does not match GPCRdb metadata"
            )
        aligned_target = alignment[entry_name]
        column_to_canonical, diagnostics = _aligned_structure_indices(
            sequence, aligned_target
        )
        canonical_to_column = {
            int(canonical_index): column
            for column, canonical_index in enumerate(column_to_canonical)
            if canonical_index >= 0
        }
        sequence_names = [
            name
            for name in alignment
            if name not in {"CONSENSUS", entry_name}
        ]
        if sequence_scope == "ortholog":
            receptor_prefix = entry_name.rsplit("_", 1)[0]
            sequence_names = [
                name
                for name in sequence_names
                if name.rsplit("_", 1)[0] == receptor_prefix
            ]
        if not sequence_names:
            raise ValueError(
                f"{entry_name} has no target-excluded {sequence_scope} sequences"
            )
        log_odds: list[float] = []
        observed: list[bool] = []
        for mutation in mutations:
            canonical_index = mutation.position - 1
            if (
                canonical_index < 0
                or canonical_index >= len(sequence)
                or sequence[canonical_index] != mutation.wt
            ):
                raise ValueError(
                    f"{mutation} does not match the canonical GPCR sequence"
                )
            column = canonical_to_column.get(canonical_index)
            if column is None:
                log_odds.append(0.0)
                observed.append(False)
                continue
            residues = [
                alignment[name][column]
                for name in sequence_names
                if alignment[name][column] not in {"-", ".", "X"}
            ]
            if not residues:
                log_odds.append(0.0)
                observed.append(False)
                continue
            mutant_count = residues.count(mutation.mutant)
            wt_count = residues.count(mutation.wt)
            log_odds.append(
                math.log(
                    (mutant_count + pseudocount)
                    / (wt_count + pseudocount)
                )
            )
            observed.append(True)
        return GPCRDBEvolutionaryResult(
            log_odds=np.asarray(log_odds, dtype=np.float32),
            observed=np.asarray(observed, dtype=bool),
            provenance={
                "schema": "protein-stabilizer.gpcrdb-evolutionary.v1",
                "accession": str(accession),
                "entry_name": entry_name,
                "family": str(metadata.get("family", "")),
                "sequence_scope": sequence_scope,
                "target_excluded": True,
                "alignment_sequences": len(sequence_names),
                "pseudocount": pseudocount,
                "metadata_path": str(metadata_path.resolve()),
                "metadata_sha256": file_sha256(metadata_path),
                "alignment_path": str(alignment_path.resolve()),
                "alignment_sha256": file_sha256(alignment_path),
                "mapped_residues": int(
                    diagnostics["mapped_residues"]
                ),
                "target_coverage": float(
                    diagnostics["target_coverage"]
                ),
            },
        )


def _scan_percentile(score: np.ndarray) -> np.ndarray:
    """Return the percentile convention used to select the GPCR blend."""

    values = np.asarray(score, dtype=np.float64)
    if values.ndim != 1 or not len(values):
        raise ValueError("GPCR scan scores must be a non-empty vector")
    if not np.isfinite(values).all():
        raise ValueError("GPCR scan scores must all be finite")
    return (rankdata(values, method="average") - 0.5) / len(values)


def rank_gpcr_screening_consensus(
    sequence: str,
    accession: str,
    retained_csv: Path,
    generic_6b_csv: Path,
    output_csv: Path,
    gpcrdb_cache: Path,
    *,
    top: int = 50,
) -> dict[str, object]:
    """Combine the promoted GPCR rank signals without rerunning either model.

    ``retained_csv`` is the output of ``rerank-6b`` and ``generic_6b_csv`` is
    the strict-FP32 ``screen-v2-6b`` output for exactly the same mutations.
    The result is a within-scan ranking score, never a thermodynamic unit.
    """

    wt_sequence = normalize_sequence(sequence)
    retained_path = Path(retained_csv)
    generic_path = Path(generic_6b_csv)
    retained = pd.read_csv(retained_path)
    generic = pd.read_csv(generic_path)
    required_retained = {"mutation", "consensus_rank_score"}
    required_generic = {"mutation", "selection_rank_score"}
    missing_retained = sorted(required_retained - set(retained.columns))
    missing_generic = sorted(required_generic - set(generic.columns))
    if missing_retained:
        raise ValueError(
            "retained GPCR scan is missing required columns: "
            f"{missing_retained}"
        )
    if missing_generic:
        raise ValueError(
            "generic 6B scan is missing required columns: "
            f"{missing_generic}"
        )
    if retained.empty or generic.empty:
        raise ValueError("GPCR consensus inputs must contain mutations")
    if retained["mutation"].duplicated().any():
        raise ValueError("retained GPCR scan contains duplicate mutations")
    if generic["mutation"].duplicated().any():
        raise ValueError("generic 6B scan contains duplicate mutations")
    retained_mutations = set(retained["mutation"].astype(str))
    generic_mutations = set(generic["mutation"].astype(str))
    if retained_mutations != generic_mutations:
        only_retained = sorted(retained_mutations - generic_mutations)
        only_generic = sorted(generic_mutations - retained_mutations)
        raise ValueError(
            "GPCR consensus inputs must contain exactly the same mutations; "
            f"only retained={only_retained[:5]}, only generic={only_generic[:5]}"
        )

    frame = retained.copy()
    parsed = [Mutation.parse(value) for value in frame["mutation"].astype(str)]
    for mutation in parsed:
        if mutation.position > len(wt_sequence):
            raise ValueError(f"mutation {mutation} lies outside the sequence")
        if wt_sequence[mutation.position - 1] != mutation.wt:
            raise ValueError(
                f"mutation {mutation} does not match the WT sequence"
            )
    generic_by_mutation = generic.set_index(
        generic["mutation"].astype(str), drop=False
    )
    generic_score = generic_by_mutation.loc[
        frame["mutation"].astype(str), "selection_rank_score"
    ].to_numpy(dtype=np.float64)
    retained_score = frame["consensus_rank_score"].to_numpy(
        dtype=np.float64
    )
    if (
        not np.isfinite(retained_score).all()
        or np.min(retained_score) < 0.0
        or np.max(retained_score) > 1.0
    ):
        raise ValueError(
            "retained consensus_rank_score must be finite and in [0, 1]"
        )

    evolutionary_cache = GPCRDBEvolutionaryCache(gpcrdb_cache)
    evolutionary = evolutionary_cache.score_mutations(
        accession,
        wt_sequence,
        parsed,
    )
    generic_percentile = _scan_percentile(generic_score)
    evolutionary_percentile = _scan_percentile(evolutionary.log_odds)
    weights = GPCR_SCREENING_WEIGHTS
    consensus = (
        weights["retained_gpcr_rank"] * retained_score
        + weights["generic_6b_stability_rank"] * generic_percentile
        + weights["gpcrdb_family_rank"] * evolutionary_percentile
    )

    frame["retained_gpcr_rank_score"] = retained_score
    frame["generic_6b_selection_rank_score"] = generic_score
    frame["generic_6b_stability_percentile"] = generic_percentile
    frame["gpcrdb_family_log_odds"] = evolutionary.log_odds
    frame["gpcrdb_family_observed"] = evolutionary.observed
    frame["gpcrdb_family_percentile"] = evolutionary_percentile
    frame["gpcr_screening_rank_score"] = consensus
    frame = frame.sort_values(
        ["gpcr_screening_rank_score", "mutation"],
        ascending=[False, True],
        kind="stable",
    ).reset_index(drop=True)
    frame["gpcr_screening_rank"] = np.arange(1, len(frame) + 1)
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False, quoting=csv.QUOTE_MINIMAL)
    top_rows = frame.head(max(1, int(top))).to_dict(orient="records")
    return {
        "schema": "protein-stabilizer.gpcr-screening-consensus.v1",
        "output_csv": str(output_path.resolve()),
        "candidate_count": int(len(frame)),
        "ranking_key": "gpcr_screening_rank_score",
        "ranking_policy": (
            "0.50 retained GPCR rank + 0.40 strict-FP32 ESM-C 6B "
            "generic stability percentile + 0.10 target-excluded GPCRdb "
            "family log-odds percentile"
        ),
        "ranking_weights": dict(weights),
        "interpretation": (
            "Higher is more favorable within this mutation scan. The score "
            "is not ddG, delta-Tm, percent stability, or crystallization "
            "probability."
        ),
        "coverage": {
            "evolutionary_observed": int(evolutionary.observed.sum()),
            "evolutionary_neutral": int((~evolutionary.observed).sum()),
        },
        "provenance": {
            "accession": str(accession),
            "retained_csv": str(retained_path.resolve()),
            "retained_csv_sha256": file_sha256(retained_path),
            "generic_6b_csv": str(generic_path.resolve()),
            "generic_6b_csv_sha256": file_sha256(generic_path),
            "gpcrdb": evolutionary.provenance,
        },
        "top_candidates": top_rows,
    }
