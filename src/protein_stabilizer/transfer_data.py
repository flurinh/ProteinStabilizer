"""Normalize external thermodynamic data for transfer learning.

The two targets in this module are intentionally kept separate:

* FireProtDB/ProTherm supplies thermodynamic ddG in kcal/mol, where negative
  values are stabilizing.
* MPTherm-Pred supplies delta-Tm in degrees Celsius, where positive values are
  stabilizing.

They must not be concatenated into a single regression target.
"""

from __future__ import annotations

import hashlib
import json
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .data import AMINO_ACIDS, Mutation, apply_mutations, gpcr_site_splits, normalize_sequence


PROTHERM_SCHEMA = "protein-stabilizer.protherm-ddg.v1"
MPTHERM_SCHEMA = "protein-stabilizer.mptherm-dtm.v1"
MAX_EMBEDDING_SEQUENCE_LENGTH = 1022


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _portable_path(path: Path) -> str:
    parts = Path(path).parts
    if "data" in parts:
        start = max(index for index, value in enumerate(parts) if value == "data")
        return Path(*parts[start:]).as_posix()
    return Path(path).name


def _canonical_substitution(row: pd.Series) -> bool:
    try:
        sequence = normalize_sequence(str(row["sequence"]))
        mutation = Mutation.parse(str(row["mutation"]))
        return (
            mutation.position <= len(sequence)
            and sequence[mutation.position - 1] == mutation.wt
            and int(row["position"]) == mutation.position
            and str(row["wt_residue"]) == mutation.wt
            and str(row["mut_residue"]) == mutation.mutant
            and mutation.wt != mutation.mutant
        )
    except (TypeError, ValueError):
        return False


def mutation_window(
    sequence: str,
    mutation: Mutation,
    *,
    max_length: int = MAX_EMBEDDING_SEQUENCE_LENGTH,
) -> tuple[str, Mutation, int]:
    """Crop a long protein around a mutation and return one-based window offset."""

    sequence = normalize_sequence(sequence)
    if mutation.position > len(sequence) or sequence[mutation.position - 1] != mutation.wt:
        raise ValueError(f"mutation {mutation} does not match sequence")
    if len(sequence) <= max_length:
        return sequence, mutation, 1
    center = mutation.position - 1
    start = max(0, min(center - max_length // 2, len(sequence) - max_length))
    window = sequence[start : start + max_length]
    remapped = Mutation(mutation.wt, mutation.position - start, mutation.mutant)
    return window, remapped, start + 1


def prepare_protherm(
    parquet_paths: Iterable[Path],
    output_csv: Path,
    metadata_json: Path,
) -> dict[str, object]:
    """Extract sequence-consistent ProTherm singles and aggregate replicates."""

    paths = [Path(path) for path in parquet_paths]
    if not paths:
        raise ValueError("no FireProtDB parquet inputs were supplied")
    frames = [pd.read_parquet(path) for path in paths]
    source = pd.concat(frames, ignore_index=True)
    raw_rows = len(source)
    source = source[source["source_dataset"].eq("ProTherm")].copy()
    protherm_rows = len(source)
    source = source[source["ddg"].notna() & source["sequence"].notna()].copy()
    missing_required_rows = protherm_rows - len(source)
    valid = source.apply(_canonical_substitution, axis=1)
    rejected_rows = int((~valid).sum())
    source = source[valid].copy()
    source["sequence"] = source["sequence"].map(normalize_sequence)
    source["mutation"] = source["mutation"].astype(str).str.upper()
    source["position"] = source["position"].astype(int)
    source["ddg"] = source["ddg"].astype(float)

    group_columns = [
        "protein_id",
        "uniprotkb",
        "protein_name",
        "sequence",
        "mutation",
        "wt_residue",
        "position",
        "mut_residue",
        "split",
    ]
    grouped = source.groupby(group_columns, dropna=False, sort=True)["ddg"]
    result = grouped.agg(
        target="median",
        replicate_count="count",
        replicate_mean="mean",
        replicate_std="std",
        replicate_min="min",
        replicate_max="max",
    ).reset_index()
    result["replicate_std"] = result["replicate_std"].fillna(0.0)
    result["replicate_spread"] = result["replicate_max"] - result["replicate_min"]
    # Repeats add evidence, while disagreement reduces it. Clipping prevents a
    # large replicate series from dominating protein-balanced training.
    result["sample_weight"] = np.clip(
        np.sqrt(result["replicate_count"]) / (1.0 + result["replicate_spread"]),
        0.25,
        2.0,
    )
    windows = [
        mutation_window(sequence, Mutation.parse(mutation))
        for sequence, mutation in zip(result["sequence"], result["mutation"], strict=True)
    ]
    result["embedding_mutation"] = [str(value[1]) for value in windows]
    result["embedding_position"] = [value[1].position for value in windows]
    result["embedding_window_start"] = [value[2] for value in windows]
    result["wt_sequence"] = [value[0] for value in windows]
    result["mutant_sequence"] = [
        apply_mutations(value[0], [value[1]]) for value in windows
    ]
    result = result.drop(columns=["sequence"])
    result["target_kind"] = "ddg"
    result["target_units"] = "kcal/mol"
    result["source_dataset"] = "ProTherm via FireProtDB 2.0"
    result = result.rename(
        columns={
            "uniprotkb": "uniprot_id",
            "wt_residue": "wt_aa",
            "mut_residue": "mut_aa",
        }
    )
    result = result.sort_values(
        ["split", "protein_id", "position", "mutation"], kind="stable"
    ).reset_index(drop=True)
    if result.groupby("protein_id")["split"].nunique().max() != 1:
        raise RuntimeError("ProTherm protein leakage across source splits")

    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_csv, index=False)
    metadata = {
        "schema": PROTHERM_SCHEMA,
        "source_files": [
            {"path": _portable_path(path), "sha256": sha256_file(path)} for path in paths
        ],
        "raw_fireprotdb_rows": raw_rows,
        "raw_protherm_rows": protherm_rows,
        "missing_ddg_or_sequence_rows": missing_required_rows,
        "rejected_sequence_inconsistent_rows": rejected_rows,
        "normalized_rows": len(result),
        "proteins": int(result["protein_id"].nunique()),
        "split_rows": {
            str(key): int(value)
            for key, value in result["split"].value_counts().sort_index().items()
        },
        "sign_convention": "negative ddG is stabilizing",
        "replicate_policy": "median target; sqrt(count)/(1+spread) weight clipped to [0.25,2]",
        "embedding_window_length": MAX_EMBEDDING_SEQUENCE_LENGTH,
        "cropped_rows": int((result["embedding_window_start"] > 1).sum()),
        "output": _portable_path(output_csv),
        "output_sha256": sha256_file(output_csv),
    }
    metadata_json = Path(metadata_json)
    metadata_json.parent.mkdir(parents=True, exist_ok=True)
    metadata_json.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    return metadata


def _download_uniprot_fasta(accession: str, cache_dir: Path) -> tuple[str, str | None]:
    target = cache_dir / f"{accession}.fasta"
    if not target.exists():
        url = f"https://rest.uniprot.org/uniprotkb/{accession}.fasta"
        request = urllib.request.Request(url, headers={"User-Agent": "ProteinStabilizer/0.1"})
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                payload = response.read()
        except urllib.error.HTTPError as exc:
            return accession, f"HTTP {exc.code}"
        temporary = target.with_suffix(".fasta.partial")
        temporary.write_bytes(payload)
        temporary.replace(target)
    lines = [
        line.strip()
        for line in target.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith(">")
    ]
    try:
        return accession, normalize_sequence("".join(lines))
    except ValueError as exc:
        return accession, f"invalid sequence: {exc}"


def fetch_uniprot_sequences(
    accessions: Iterable[str], cache_dir: Path, *, workers: int = 8
) -> tuple[dict[str, str], dict[str, str]]:
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    identifiers = sorted(set(str(value) for value in accessions))
    sequences: dict[str, str] = {}
    errors: dict[str, str] = {}
    with ThreadPoolExecutor(max_workers=workers) as executor:
        for accession, value in executor.map(
            lambda identifier: _download_uniprot_fasta(identifier, cache_dir),
            identifiers,
        ):
            if value is None or not set(value).issubset(set(AMINO_ACIDS)):
                errors[accession] = str(value)
            else:
                sequences[accession] = value
    return sequences, errors


def _protein_split(frame: pd.DataFrame, seed: int) -> dict[str, str]:
    """Find a protein-disjoint 70/15/15 split with balanced row counts."""

    identifiers = np.asarray(sorted(frame["protein_id"].unique()), dtype=object)
    rng = np.random.default_rng(seed)
    n_test = max(1, round(0.15 * len(identifiers)))
    n_val = max(1, round(0.15 * len(identifiers)))
    target_rows = 0.15 * len(frame)
    topology_counts = frame["topology"].value_counts().to_dict()
    best: tuple[float, np.ndarray] | None = None
    for _ in range(10_000):
        candidate = rng.permutation(identifiers)
        test = set(candidate[:n_test])
        val = set(candidate[n_test : n_test + n_val])
        test_rows = frame[frame["protein_id"].isin(test)]
        val_rows = frame[frame["protein_id"].isin(val)]
        score = abs(len(test_rows) - target_rows) + abs(len(val_rows) - target_rows)
        for topology, count in topology_counts.items():
            topology_target = 0.15 * count
            score += 0.5 * abs((test_rows["topology"] == topology).sum() - topology_target)
            score += 0.5 * abs((val_rows["topology"] == topology).sum() - topology_target)
        if best is None or score < best[0]:
            best = (float(score), candidate.copy())
    if best is None:
        raise RuntimeError("could not construct an MPTherm protein split")
    chosen = best[1]
    return {
        str(identifier): (
            "test" if index < n_test else "val" if index < n_test + n_val else "train"
        )
        for index, identifier in enumerate(chosen)
    }


def prepare_mptherm(
    source_tab: Path,
    gpcr_csv: Path,
    uniprot_cache_dir: Path,
    output_csv: Path,
    metadata_json: Path,
    *,
    seed: int = 20260715,
) -> dict[str, object]:
    """Attach UniProt sequences and create a protein-held-out delta-Tm set."""

    source_tab = Path(source_tab)
    source = pd.read_csv(source_tab, sep="\t")
    source = source.dropna(
        subset=[
            "UniProt ID",
            "Mutation (based on UniProt numbering)",
            "Experimental ΔTm (in degree celsius)",
        ]
    ).copy()
    sequences, download_errors = fetch_uniprot_sequences(
        source["UniProt ID"].astype(str), uniprot_cache_dir
    )

    gpcr_raw = pd.read_csv(gpcr_csv)
    protein_to_uniprot = dict(
        gpcr_raw[["protein_id", "uniprot_id"]]
        .drop_duplicates()
        .itertuples(index=False, name=None)
    )
    gpcr = gpcr_site_splits(Path(gpcr_csv), seed=seed)
    gpcr["uniprot_id"] = gpcr["protein_id"].map(protein_to_uniprot)
    overlap_by_site = {
        (str(row.uniprot_id), int(row.position)): str(row.split)
        for row in gpcr[["uniprot_id", "position", "split"]]
        .drop_duplicates()
        .itertuples(index=False)
    }

    normalized: list[dict[str, object]] = []
    rejection_reasons: dict[str, int] = {}
    for values in source.to_dict("records"):
        accession = str(values["UniProt ID"])
        mutation_text = str(values["Mutation (based on UniProt numbering)"])
        try:
            mutation = Mutation.parse(mutation_text)
            sequence = sequences[accession]
            window, embedding_mutation, window_start = mutation_window(sequence, mutation)
            mutant_sequence = apply_mutations(window, [embedding_mutation])
        except KeyError:
            reason = "missing_uniprot_sequence"
            rejection_reasons[reason] = rejection_reasons.get(reason, 0) + 1
            continue
        except ValueError:
            reason = "mutation_sequence_mismatch"
            rejection_reasons[reason] = rejection_reasons.get(reason, 0) + 1
            continue
        normalized.append(
            {
                "protein_id": accession,
                "uniprot_id": accession,
                "protein_name": str(values["Protein name"]),
                "mutation": str(mutation),
                "wt_aa": mutation.wt,
                "position": mutation.position,
                "mut_aa": mutation.mutant,
                "embedding_mutation": str(embedding_mutation),
                "embedding_position": embedding_mutation.position,
                "embedding_window_start": window_start,
                "wt_sequence": window,
                "mutant_sequence": mutant_sequence,
                "target": float(values["Experimental ΔTm (in degree celsius)"]),
                "target_kind": "delta_tm",
                "target_units": "degrees Celsius",
                "topology": str(values["Topology"]),
                "function": str(values["Function"]),
                "location": str(
                    values[
                        "Location based on secondary structure and solvent accessibility"
                    ]
                ),
                "pmid": str(values["PubMed Ids"]),
                "mptherm_accession": str(
                    values["MPTherm database accession number"]
                ),
                "upstream_dataset_type": str(values["Dataset type"]),
                "upstream_cv_fold": str(values["10-fold group-wise CV Test*"]),
                "gpcr_overlap_split": overlap_by_site.get(
                    (accession, mutation.position), "none"
                ),
                "source_dataset": "MPTherm-Pred",
            }
        )
    result = pd.DataFrame(normalized)
    split_by_protein = _protein_split(result, seed)
    result["split"] = result["protein_id"].map(split_by_protein)
    # Any MPTherm measurement at a GPCR validation/test site is quarantined.
    # It remains in the normalized file for audit but is never a training row.
    quarantined = result["gpcr_overlap_split"].isin(["val", "test"])
    result.loc[quarantined, "split"] = "quarantine"
    result = result.sort_values(
        ["split", "protein_id", "position", "mutation"], kind="stable"
    ).reset_index(drop=True)
    non_quarantine = result[~result["split"].eq("quarantine")]
    if non_quarantine.groupby("protein_id")["split"].nunique().max() != 1:
        raise RuntimeError("MPTherm protein leakage across derived splits")

    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_csv, index=False)
    metadata = {
        "schema": MPTHERM_SCHEMA,
        "source_file": _portable_path(source_tab),
        "source_sha256": sha256_file(source_tab),
        "source_rows": len(source),
        "normalized_rows": len(result),
        "proteins": int(result["protein_id"].nunique()),
        "download_errors": download_errors,
        "rejection_reasons": rejection_reasons,
        "split_rows": {
            str(key): int(value)
            for key, value in result["split"].value_counts().sort_index().items()
        },
        "topology_rows": {
            str(key): int(value)
            for key, value in result["topology"].value_counts().sort_index().items()
        },
        "gpcr_quarantine_policy": "exclude any site assigned to GPCR validation or test",
        "quarantined_rows": int(quarantined.sum()),
        "sign_convention": "positive delta-Tm is stabilizing",
        "embedding_window_length": MAX_EMBEDDING_SEQUENCE_LENGTH,
        "cropped_rows": int((result["embedding_window_start"] > 1).sum()),
        "output": _portable_path(output_csv),
        "output_sha256": sha256_file(output_csv),
    }
    metadata_json = Path(metadata_json)
    metadata_json.parent.mkdir(parents=True, exist_ok=True)
    metadata_json.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    return metadata
