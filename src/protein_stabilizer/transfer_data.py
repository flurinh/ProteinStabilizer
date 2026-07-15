"""Normalize external thermodynamic data for transfer learning.

The two targets in this module are intentionally kept separate:

* FireProtDB/ProTherm supplies thermodynamic ddG in kcal/mol, where negative
  values are stabilizing.
* MPTherm-Pred and GPCR-tm supply delta-Tm in degrees Celsius, where positive
  values are stabilizing.

They must not be concatenated into a single regression target.
"""

from __future__ import annotations

import hashlib
import json
import tarfile
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
MCSM_MEMBRANE_SCHEMA = "protein-stabilizer.mcsm-membrane-ddg.v1"
GPCR_TM_SCHEMA = "protein-stabilizer.gpcr-tm-dtm.v1"
MAX_EMBEDDING_SEQUENCE_LENGTH = 1022

THREE_TO_ONE = {
    "ALA": "A",
    "ARG": "R",
    "ASN": "N",
    "ASP": "D",
    "CYS": "C",
    "GLN": "Q",
    "GLU": "E",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LEU": "L",
    "LYS": "K",
    "MET": "M",
    "PHE": "F",
    "PRO": "P",
    "SER": "S",
    "THR": "T",
    "TRP": "W",
    "TYR": "Y",
    "VAL": "V",
}

MCSM_TOPOLOGY = {
    "1PY6": "alpha_helical",  # bacteriorhodopsin, seven transmembrane helices
    "2XOV": "alpha_helical",  # GlpG rhomboid protease
    "2K73": "alpha_helical",  # DsbB
    "1AFO": "alpha_helical",  # glycophorin A
    "3GP6": "beta_barrel",  # PagP
    "1QD6": "beta_barrel",  # OmpLA
    "1QJP": "beta_barrel",  # OmpA
}


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


def _pdb_chain_sequence(
    payload: bytes,
    chain: str,
    mutation: Mutation,
) -> tuple[str, Mutation]:
    """Extract an observed PDB-chain sequence and remap its residue number."""

    residues: list[tuple[tuple[int, str], str]] = []
    seen: set[tuple[int, str]] = set()
    for raw_line in payload.decode("ascii", errors="replace").splitlines():
        if not raw_line.startswith("ATOM  ") or len(raw_line) < 27:
            continue
        if raw_line[21].strip() != chain or raw_line[12:16].strip() != "CA":
            continue
        if raw_line[16] not in {" ", "A"}:
            continue
        key = (int(raw_line[22:26]), raw_line[26].strip())
        if key in seen:
            continue
        residue = THREE_TO_ONE.get(raw_line[17:20].strip())
        if residue is None:
            raise ValueError(f"non-canonical PDB residue {raw_line[17:20]!r}")
        seen.add(key)
        residues.append((key, residue))
    if not residues:
        raise ValueError(f"PDB chain {chain!r} has no canonical CA atoms")
    candidates = [
        index
        for index, ((number, insertion), residue) in enumerate(residues)
        if number == mutation.position and insertion == "" and residue == mutation.wt
    ]
    if len(candidates) != 1:
        observed = [
            (index + 1, insertion, residue)
            for index, ((number, insertion), residue) in enumerate(residues)
            if number == mutation.position
        ]
        raise ValueError(
            f"could not map {mutation} on chain {chain}; observed={observed}"
        )
    sequence = normalize_sequence("".join(residue for _, residue in residues))
    remapped = Mutation(mutation.wt, candidates[0] + 1, mutation.mutant)
    return sequence, remapped


def prepare_mcsm_membrane(
    training_csv: Path,
    blind_csv: Path,
    structures_tar_gz: Path,
    output_csv: Path,
    metadata_json: Path,
) -> dict[str, object]:
    """Normalize the mCSM-membrane equilibrium mutation-stability dataset.

    The source convention is ``DG(WT) - DG(mutant)`` (positive is stabilizing).
    The project convention is the opposite, so targets are negated. Rows whose
    PDB filename contains ``.mut.`` are the source authors' hypothetical reverse
    mutations and are explicitly marked rather than treated as measurements.
    """

    inputs = [(Path(training_csv), "train"), (Path(blind_csv), "blind")]
    source = pd.concat(
        [
            pd.read_csv(path, sep="\t").assign(source_split=source_split)
            for path, source_split in inputs
        ],
        ignore_index=True,
    )
    required = {"DDG", "PDB", "MUTATION", "CHAIN", "source_split"}
    if not required.issubset(source.columns):
        raise ValueError(
            f"mCSM-membrane inputs are missing {sorted(required - set(source.columns))}"
        )

    rows: list[dict[str, object]] = []
    with tarfile.open(structures_tar_gz, mode="r:gz") as archive:
        members = {member.name: member for member in archive.getmembers() if member.isfile()}
        for values in source.to_dict("records"):
            filename = str(values["PDB"])
            pdb_id = filename[:4].upper()
            if pdb_id not in MCSM_TOPOLOGY:
                raise ValueError(f"unknown mCSM-membrane protein {pdb_id}")
            member_name = f"pdb_stability/{filename}"
            member = members.get(member_name)
            if member is None:
                raise ValueError(f"missing structure archive member {member_name}")
            handle = archive.extractfile(member)
            if handle is None:
                raise ValueError(f"could not read structure archive member {member_name}")
            mutation = Mutation.parse(str(values["MUTATION"]))
            sequence, embedding_mutation = _pdb_chain_sequence(
                handle.read(), str(values["CHAIN"]), mutation
            )
            mutant_sequence = apply_mutations(sequence, [embedding_mutation])
            is_reverse = ".mut." in filename
            topology = MCSM_TOPOLOGY[pdb_id]
            source_split = str(values["source_split"])
            if topology == "alpha_helical" and source_split == "train":
                split = "train"
            elif topology == "alpha_helical" and source_split == "blind" and not is_reverse:
                split = "test"
            else:
                split = "reference"
            rows.append(
                {
                    "protein_id": pdb_id,
                    "pdb_id": pdb_id,
                    "structure_file": filename,
                    "chain": str(values["CHAIN"]),
                    "mutation": str(mutation),
                    "wt_aa": mutation.wt,
                    "position": mutation.position,
                    "mut_aa": mutation.mutant,
                    "embedding_mutation": str(embedding_mutation),
                    "embedding_position": embedding_mutation.position,
                    "embedding_window_start": 1,
                    "wt_sequence": sequence,
                    "mutant_sequence": mutant_sequence,
                    "source_ddg": float(values["DDG"]),
                    "target": -float(values["DDG"]),
                    "target_kind": "ddg",
                    "target_units": "kcal/mol",
                    "topology": topology,
                    "is_reverse": int(is_reverse),
                    "row_kind": "synthetic_reverse" if is_reverse else "experimental",
                    "source_split": source_split,
                    "split": split,
                    "source_dataset": "mCSM-membrane / Kroncke et al. 2016",
                }
            )
    result = pd.DataFrame(rows)
    pair_columns = ["protein_id", "chain", "position"]
    result["mutation_pair"] = result.apply(
        lambda row: "".join(sorted((str(row["wt_aa"]), str(row["mut_aa"])))),
        axis=1,
    )
    pair_size = result.groupby(pair_columns + ["mutation_pair"])["mutation"].transform(
        "size"
    )
    result["sample_weight"] = np.where(pair_size == 2, 0.5, 1.0)
    result = result.sort_values(
        ["topology", "source_split", "protein_id", "is_reverse", "position", "mutation"],
        kind="stable",
    ).reset_index(drop=True)

    experimental = result[result["is_reverse"].eq(0)]
    if len(experimental) != 223:
        raise RuntimeError(f"expected 223 experimental membrane ddG rows, found {len(experimental)}")
    if len(result) != 404:
        raise RuntimeError(f"expected 404 total mCSM-membrane rows, found {len(result)}")
    if set(experimental["protein_id"]) != set(MCSM_TOPOLOGY):
        raise RuntimeError("mCSM-membrane protein set does not match its publication")

    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_csv, index=False)
    metadata = {
        "schema": MCSM_MEMBRANE_SCHEMA,
        "source_files": [
            {"path": _portable_path(path), "sha256": sha256_file(path)}
            for path, _ in inputs
        ],
        "structure_archive": {
            "path": _portable_path(structures_tar_gz),
            "sha256": sha256_file(structures_tar_gz),
        },
        "rows": len(result),
        "experimental_rows": int((result["is_reverse"] == 0).sum()),
        "synthetic_reverse_rows": int((result["is_reverse"] == 1).sum()),
        "protein_rows": {
            str(key): int(value)
            for key, value in experimental["protein_id"].value_counts().sort_index().items()
        },
        "topology_rows": {
            str(key): int(value)
            for key, value in experimental["topology"].value_counts().sort_index().items()
        },
        "application_training_policy": (
            "source four-protein development split; unseen alpha-helical "
            "1AFO/2K73 evaluation"
        ),
        "test_policy": "experimental forward mutations only; synthetic reverse rows excluded",
        "source_sign_convention": "positive DG(WT)-DG(mutant) is stabilizing",
        "target_sign_convention": "negative project ddG is stabilizing",
        "output": _portable_path(output_csv),
        "output_sha256": sha256_file(output_csv),
    }
    metadata_json = Path(metadata_json)
    metadata_json.parent.mkdir(parents=True, exist_ok=True)
    metadata_json.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    return metadata


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
    gpcr_tm_csv: Path,
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
    gpcr_tm_source = pd.read_csv(gpcr_tm_csv)
    gpcr_tm_test_sites = {
        (str(row.Uniprot), int(row.position))
        for row in gpcr_tm_source[
            gpcr_tm_source["Set"].astype(str).str.lower().eq("test")
        ].itertuples(index=False)
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
                "gpcr_tm_test_site_overlap": int(
                    (accession, mutation.position) in gpcr_tm_test_sites
                ),
                "source_dataset": "MPTherm-Pred",
            }
        )
    result = pd.DataFrame(normalized)
    split_by_protein = _protein_split(result, seed)
    result["split"] = result["protein_id"].map(split_by_protein)
    # Any MPTherm measurement at a GPCR validation/test site is quarantined.
    # It remains in the normalized file for audit but is never a training row.
    benchmark_quarantined = result["gpcr_overlap_split"].isin(["val", "test"])
    gpcr_tm_quarantined = result["gpcr_tm_test_site_overlap"].eq(1)
    quarantined = benchmark_quarantined | gpcr_tm_quarantined
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
        "gpcr_quarantine_policy": (
            "exclude user-benchmark validation/test sites and official GPCR-tm test sites"
        ),
        "quarantined_rows": int(quarantined.sum()),
        "user_benchmark_quarantined_rows": int(benchmark_quarantined.sum()),
        "gpcr_tm_test_quarantined_rows": int(gpcr_tm_quarantined.sum()),
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


def prepare_gpcr_tm(
    source_csv: Path,
    gpcr_csv: Path,
    uniprot_cache_dir: Path,
    output_csv: Path,
    metadata_json: Path,
    *,
    seed: int = 20260715,
) -> dict[str, object]:
    """Normalize the official GPCR-tm experimental delta-Tm dataset.

    The upstream test rows remain untouched for model evaluation. Training
    substitutions at the same protein/site as an upstream test row are marked
    and excluded by the trainer. Separately, all rows at a validation or test
    site in the user GPCR benchmark are marked so an application refit cannot
    leak into the final calibration evaluation.
    """

    source_csv = Path(source_csv)
    source = pd.read_csv(source_csv)
    required = {
        "Uniprot",
        "Mutation",
        "generic_numbering",
        "dTm",
        "Set",
        "wild_type_aa",
        "Mutation_aa",
        "position",
    }
    if not required.issubset(source.columns):
        raise ValueError(
            f"GPCR-tm input is missing {sorted(required - set(source.columns))}"
        )
    sequences, download_errors = fetch_uniprot_sequences(
        source["Uniprot"].astype(str), uniprot_cache_dir
    )

    gpcr_raw = pd.read_csv(gpcr_csv)
    protein_to_uniprot = dict(
        gpcr_raw[["protein_id", "uniprot_id"]]
        .drop_duplicates()
        .itertuples(index=False, name=None)
    )
    gpcr = gpcr_site_splits(Path(gpcr_csv), seed=seed)
    gpcr["uniprot_id"] = gpcr["protein_id"].map(protein_to_uniprot)
    benchmark_site_split = {
        (str(row.uniprot_id), int(row.position)): str(row.split)
        for row in gpcr[["uniprot_id", "position", "split"]]
        .drop_duplicates()
        .itertuples(index=False)
    }
    official_test_sites = {
        (str(row.Uniprot), int(row.position))
        for row in source[source["Set"].astype(str).str.lower().eq("test")].itertuples(
            index=False
        )
    }

    rows: list[dict[str, object]] = []
    rejection_reasons: dict[str, int] = {}
    for values in source.to_dict("records"):
        accession = str(values["Uniprot"])
        try:
            mutation = Mutation.parse(str(values["Mutation"]))
            if (
                mutation.wt != str(values["wild_type_aa"])
                or mutation.mutant != str(values["Mutation_aa"])
                or mutation.position != int(values["position"])
            ):
                raise ValueError("mutation columns disagree")
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
        source_split = str(values["Set"]).strip().lower()
        if source_split not in {"train", "test"}:
            raise ValueError(f"unknown GPCR-tm split {values['Set']!r}")
        site_key = (accession, mutation.position)
        rows.append(
            {
                "protein_id": accession,
                "uniprot_id": accession,
                "mutation": str(mutation),
                "wt_aa": mutation.wt,
                "position": mutation.position,
                "mut_aa": mutation.mutant,
                "embedding_mutation": str(embedding_mutation),
                "embedding_position": embedding_mutation.position,
                "embedding_window_start": window_start,
                "wt_sequence": window,
                "mutant_sequence": mutant_sequence,
                "generic_numbering": str(values["generic_numbering"]),
                "target": float(values["dTm"]),
                "target_kind": "delta_tm",
                "target_units": "degrees Celsius",
                "topology": "alpha_helical_gpcr",
                "sample_weight": 1.0,
                "source_split": source_split,
                "split": source_split,
                "official_test_site_overlap": int(
                    source_split == "train" and site_key in official_test_sites
                ),
                "gpcr_overlap_split": benchmark_site_split.get(site_key, "none"),
                "source_dataset": "GPCR-tm",
            }
        )
    result = pd.DataFrame(rows).sort_values(
        ["split", "protein_id", "position", "mutation"], kind="stable"
    ).reset_index(drop=True)
    if len(result) != 97 or len(result[result["split"].eq("test")]) != 12:
        raise RuntimeError(
            f"expected 97 GPCR-tm rows with 12 test rows, found {len(result)} and "
            f"{int(result['split'].eq('test').sum())}"
        )
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_csv, index=False)
    application_excluded = result["gpcr_overlap_split"].isin(["val", "test"])
    metadata = {
        "schema": GPCR_TM_SCHEMA,
        "source_file": _portable_path(source_csv),
        "source_sha256": sha256_file(source_csv),
        "normalized_rows": len(result),
        "proteins": int(result["protein_id"].nunique()),
        "split_rows": {
            str(key): int(value)
            for key, value in result["split"].value_counts().sort_index().items()
        },
        "download_errors": download_errors,
        "rejection_reasons": rejection_reasons,
        "official_test_site_overlap_rows": int(
            result["official_test_site_overlap"].sum()
        ),
        "benchmark_application_excluded_rows": int(application_excluded.sum()),
        "evaluation_policy": (
            "official 12-row test; exclude training substitutions at official test sites"
        ),
        "application_policy": (
            "refit after evaluation; exclude all user-benchmark validation/test sites"
        ),
        "sign_convention": "positive delta-Tm is stabilizing",
        "output": _portable_path(output_csv),
        "output_sha256": sha256_file(output_csv),
    }
    metadata_json = Path(metadata_json)
    metadata_json.parent.mkdir(parents=True, exist_ok=True)
    metadata_json.write_text(
        json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8"
    )
    return metadata
