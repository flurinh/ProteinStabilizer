"""Row manifests for the hierarchical ESM-C/ProteinMPNN v2 model."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Callable, Sequence

import h5py
import numpy as np
import pandas as pd

from .data import (
    DatasetPaths,
    double_rows,
    gpcr_site_splits,
    sequence_hash,
    single_rows,
    transfer_rows,
)
from .embeddings import HierarchyEmbeddingReader, file_sha256


HIERARCHY_ROW_SCHEMA = "protein-stabilizer.hierarchy-row-features.v1"
HIERARCHY_MULTI_ROW_SCHEMA = "protein-stabilizer.hierarchy-multi-row-features.v1"
MEMBRANE_FEATURE_NAMES = (
    "is_membrane_protein",
    "is_alpha_helical",
    "is_beta_barrel",
    "is_gpcr",
    "is_transmembrane_site",
    "gpcr_helix_scaled",
    "gpcr_x50_offset_scaled",
    "topology_known",
)
_GENERIC_NUMBER = re.compile(r"^(?P<helix>\d+)\.\d+x(?P<offset>\d+)$")


def membrane_topology_features(
    topology: str | None,
    *,
    generic_numbering: str | None = None,
    is_gpcr: bool = False,
) -> np.ndarray:
    """Encode explicit protein- and residue-level membrane annotations.

    Missing site topology stays missing rather than being inferred from the
    endpoint.  GPCR generic numbering supplies a residue-level TM flag, helix,
    and a bounded offset from the conserved x50 position.
    """

    value = "" if topology is None else str(topology).strip().lower()
    membrane = value in {
        "membrane",
        "alpha_helical",
        "beta_barrel",
        "alpha_helical_gpcr",
    } or is_gpcr
    alpha = value in {"alpha_helical", "alpha_helical_gpcr"} or is_gpcr
    beta = value == "beta_barrel"
    gpcr = value == "alpha_helical_gpcr" or is_gpcr
    known = bool(value) and value not in {"unknown", "nan", "none"}
    helix_scaled = 0.0
    offset_scaled = 0.0
    transmembrane_site = 0.0
    if generic_numbering is not None:
        matched = _GENERIC_NUMBER.match(str(generic_numbering).strip())
        if matched is not None:
            helix = int(matched.group("helix"))
            offset = int(matched.group("offset"))
            transmembrane_site = 1.0
            helix_scaled = (helix - 4.0) / 3.0
            offset_scaled = float(np.tanh((offset - 50.0) / 12.0))
            known = True
    return np.asarray(
        [
            float(membrane),
            float(alpha),
            float(beta),
            float(gpcr),
            transmembrane_site,
            helix_scaled,
            offset_scaled,
            float(known),
        ],
        dtype=np.float32,
    )


def _string_dataset(
    handle: h5py.File,
    name: str,
    values: Sequence[str],
) -> None:
    handle.create_dataset(
        name,
        data=np.asarray(values, dtype=object),
        dtype=h5py.string_dtype(encoding="utf-8"),
    )


def _atomic_h5(path: Path, writer: Callable[[h5py.File], None]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".partial")
    partial.unlink(missing_ok=True)
    with h5py.File(partial, "w", libver="latest") as handle:
        writer(handle)
        handle.flush()
    partial.replace(path)


def _protein_validation_split(
    protein_ids: Sequence[str],
    seed: int,
) -> np.ndarray:
    proteins = np.asarray(sorted(set(protein_ids)), dtype=object)
    rng = np.random.default_rng(seed)
    rng.shuffle(proteins)
    validation_count = max(1, round(0.10 * len(proteins)))
    validation = set(proteins[:validation_count].tolist())
    return np.asarray(
        ["val" if protein in validation else "train" for protein in protein_ids],
        dtype=object,
    )


def _load_structure_rows(
    path: Path | None,
    rows: Sequence[dict[str, object]],
    *,
    dimension: int = 128,
) -> tuple[np.ndarray, np.ndarray, dict[str, object] | None]:
    if path is None or not path.is_file():
        return (
            np.zeros((len(rows), dimension), dtype=np.float32),
            np.zeros(len(rows), dtype=bool),
            None,
        )
    with h5py.File(path, "r") as handle:
        protein = np.asarray(handle["protein_id"].asstr()[:])
        mutation = np.asarray(handle["mutation"].asstr()[:])
        position = np.asarray(handle["position"], dtype=np.int32)
        provenance = json.loads(str(handle.attrs["proteinmpnn_provenance"]))
        storage_dtype = np.dtype(
            str(provenance.get("storage_dtype", "float16"))
        )
        if storage_dtype not in {
            np.dtype(np.float16),
            np.dtype(np.float32),
        }:
            raise RuntimeError("unsupported ProteinMPNN storage dtype")
        vectors = np.asarray(handle["proteinmpnn"], dtype=storage_dtype)
        structure_mask = (
            np.asarray(handle["structure_mask"], dtype=bool)
            if "structure_mask" in handle
            else np.ones(len(vectors), dtype=bool)
        )
        source_sha256 = str(handle.attrs["source_sha256"])
        structure_archive_sha256 = str(
            handle.attrs.get("structure_archive_sha256", "")
        )
    expected_protein = np.asarray(
        [str(row["protein_id"]) for row in rows], dtype=object
    )
    expected_mutation = np.asarray(
        [str(row["mutation"]) for row in rows], dtype=object
    )
    expected_position = np.asarray(
        [int(row["position"]) for row in rows], dtype=np.int32
    )
    if (
        vectors.shape != (len(rows), dimension)
        or structure_mask.shape != (len(rows),)
        or not np.array_equal(protein, expected_protein)
        or not np.array_equal(mutation, expected_mutation)
        or not np.array_equal(position, expected_position)
    ):
        raise RuntimeError(f"ProteinMPNN rows do not align with {path.name}")
    return (
        vectors,
        structure_mask,
        {
            "path": str(path),
            "sha256": file_sha256(path),
            "source_sha256": source_sha256,
            "structure_archive_sha256": structure_archive_sha256,
            "proteinmpnn": provenance,
        },
    )


def _write_rows(
    source: Path,
    target: Path,
    rows: Sequence[dict[str, object]],
    cache: HierarchyEmbeddingReader,
    *,
    cache_path: Path,
    cache_sha256: str,
    kind: str,
    target_kind: str,
    favorable_direction: str,
    split: Sequence[str],
    membrane: np.ndarray,
    structure_path: Path | None = None,
) -> dict[str, object]:
    if len(rows) != len(split) or membrane.shape != (
        len(rows),
        len(MEMBRANE_FEATURE_NAMES),
    ):
        raise ValueError("hierarchy row metadata counts differ")
    wt_keys = [
        (sequence_hash(str(row["wt_sequence"])), int(row["position"]))
        for row in rows
    ]
    mutant_keys = [
        (sequence_hash(str(row["mutant_sequence"])), int(row["position"]))
        for row in rows
    ]
    wt_site, wt_sequence = cache.locations(wt_keys)
    mutant_site, mutant_sequence = cache.locations(mutant_keys)
    structure, structure_mask, structure_provenance = _load_structure_rows(
        structure_path, rows
    )

    def write(handle: h5py.File) -> None:
        handle.attrs["schema"] = HIERARCHY_ROW_SCHEMA
        handle.attrs["kind"] = kind
        handle.attrs["target_kind"] = target_kind
        handle.attrs["favorable_direction"] = favorable_direction
        handle.attrs["source"] = str(source)
        handle.attrs["source_sha256"] = file_sha256(source)
        handle.attrs["hierarchy_cache"] = str(cache_path)
        handle.attrs["hierarchy_cache_sha256"] = cache_sha256
        handle.attrs["hierarchy_provenance"] = json.dumps(
            cache.provenance, sort_keys=True, separators=(",", ":")
        )
        handle.attrs["window_radius"] = cache.window_radius
        handle.attrs["membrane_feature_names"] = json.dumps(
            MEMBRANE_FEATURE_NAMES, separators=(",", ":")
        )
        handle.attrs["structure_provenance"] = json.dumps(
            structure_provenance, sort_keys=True, separators=(",", ":")
        )
        handle.create_dataset("wt_site_index", data=wt_site)
        handle.create_dataset("wt_sequence_index", data=wt_sequence)
        handle.create_dataset("mutant_site_index", data=mutant_site)
        handle.create_dataset("mutant_sequence_index", data=mutant_sequence)
        handle.create_dataset(
            "target",
            data=np.asarray([float(row["target"]) for row in rows], dtype=np.float32),
        )
        handle.create_dataset(
            "position",
            data=np.asarray([int(row["position"]) for row in rows], dtype=np.int32),
        )
        handle.create_dataset("membrane", data=membrane.astype(np.float32))
        handle.create_dataset("structure", data=structure)
        handle.create_dataset("structure_mask", data=structure_mask)
        handle.create_dataset(
            "sample_weight",
            data=np.asarray(
                [float(row.get("sample_weight", 1.0)) for row in rows],
                dtype=np.float32,
            ),
        )
        _string_dataset(
            handle, "protein_id", [str(row["protein_id"]) for row in rows]
        )
        _string_dataset(
            handle, "mutation", [str(row["mutation"]) for row in rows]
        )
        _string_dataset(handle, "split", [str(value) for value in split])
        for optional in ("assay_id", "site_id", "uniprot_id"):
            if all(optional in row for row in rows):
                _string_dataset(
                    handle, optional, [str(row[optional]) for row in rows]
                )

    _atomic_h5(target, write)
    split_counts = {
        value: int(sum(str(item) == value for item in split))
        for value in sorted(set(str(item) for item in split))
    }
    return {
        "path": str(target),
        "sha256": file_sha256(target),
        "kind": kind,
        "target_kind": target_kind,
        "favorable_direction": favorable_direction,
        "rows": len(rows),
        "proteins": len({str(row["protein_id"]) for row in rows}),
        "split_counts": split_counts,
        "source_sha256": file_sha256(source),
        "structure_rows": int(structure_mask.sum()),
        "structure_provenance": structure_provenance,
    }


def build_hierarchy_row_features(
    root: Path,
    cache_path: Path,
    output_dir: Path,
    *,
    structure_dir: Path | None = None,
    seed: int = 20260715,
) -> dict[str, object]:
    """Create compact row-to-cache manifests for all staged single tasks."""

    root = Path(root).resolve()
    cache_path = Path(cache_path).resolve()
    output_dir = Path(output_dir).resolve()
    structure_dir = (
        None if structure_dir is None else Path(structure_dir).resolve()
    )
    paths = DatasetPaths(root)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_sha256 = file_sha256(cache_path)
    records: list[dict[str, object]] = []
    with HierarchyEmbeddingReader(cache_path) as cache:
        for split_name in ("train", "test"):
            source = paths.single(split_name)
            rows = list(single_rows(source))
            proteins = [str(row["protein_id"]) for row in rows]
            row_split = (
                _protein_validation_split(proteins, seed)
                if split_name == "train"
                else np.asarray(["test"] * len(rows), dtype=object)
            )
            structure_path = (
                None
                if structure_dir is None
                else structure_dir / f"structure_single_{split_name}.h5"
            )
            records.append(
                _write_rows(
                    source,
                    output_dir / f"single_{split_name}.h5",
                    rows,
                    cache,
                    cache_path=cache_path,
                    cache_sha256=cache_sha256,
                    kind="generic_ddg",
                    target_kind="ddg",
                    favorable_direction="negative",
                    split=row_split.tolist(),
                    membrane=np.zeros(
                        (len(rows), len(MEMBRANE_FEATURE_NAMES)),
                        dtype=np.float32,
                    ),
                    structure_path=structure_path,
                )
            )

        transfer_specs = (
            ("protherm", paths.protherm, "experimental_ddg", "ddg", "negative"),
            ("mptherm", paths.mptherm, "membrane_dtm", "delta_tm", "positive"),
            (
                "mcsm_membrane",
                paths.mcsm_membrane,
                "membrane_ddg",
                "ddg",
                "negative",
            ),
            ("gpcr_tm", paths.gpcr_tm, "gpcr_dtm", "delta_tm", "positive"),
        )
        for filename, source, kind, target_kind, favorable in transfer_specs:
            rows = list(transfer_rows(source))
            membrane = np.stack(
                [
                    membrane_topology_features(
                        str(row.get("topology", "")),
                        generic_numbering=(
                            None
                            if "generic_numbering" not in row
                            else str(row["generic_numbering"])
                        ),
                        is_gpcr=kind == "gpcr_dtm",
                    )
                    for row in rows
                ]
            )
            records.append(
                _write_rows(
                    source,
                    output_dir / f"{filename}.h5",
                    rows,
                    cache,
                    cache_path=cache_path,
                    cache_sha256=cache_sha256,
                    kind=kind,
                    target_kind=target_kind,
                    favorable_direction=favorable,
                    split=[str(row["split"]) for row in rows],
                    membrane=membrane,
                    structure_path=(
                        None
                        if structure_dir is None
                        else structure_dir / f"structure_{filename}.h5"
                    ),
                )
            )

        gpcr_frame = gpcr_site_splits(paths.gpcr, seed=seed)
        gpcr_rows = gpcr_frame.to_dict("records")
        gpcr_membrane = np.stack(
            [
                membrane_topology_features(
                    "alpha_helical_gpcr",
                    is_gpcr=True,
                )
                for _ in gpcr_rows
            ]
        )
        records.append(
            _write_rows(
                paths.gpcr,
                output_dir / "gpcr_rank.h5",
                gpcr_rows,
                cache,
                cache_path=cache_path,
                cache_sha256=cache_sha256,
                kind="gpcr_assay_rank",
                target_kind="stability_delta_percent",
                favorable_direction="positive",
                split=[str(row["split"]) for row in gpcr_rows],
                membrane=gpcr_membrane,
                structure_path=(
                    None
                    if structure_dir is None
                    else structure_dir / "structure_gpcr_rank.h5"
                ),
            )
        )
        hierarchy_provenance = cache.provenance
        window_radius = cache.window_radius
    manifest = {
        "schema": HIERARCHY_ROW_SCHEMA,
        "seed": seed,
        "hierarchy_cache": str(cache_path),
        "hierarchy_cache_sha256": cache_sha256,
        "hierarchy_provenance": hierarchy_provenance,
        "window_radius": window_radius,
        "membrane_feature_names": list(MEMBRANE_FEATURE_NAMES),
        "datasets": records,
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    manifest["manifest_path"] = str(manifest_path)
    return manifest


def build_hierarchy_multi_row_features(
    root: Path,
    cache_path: Path,
    output_dir: Path,
) -> dict[str, object]:
    """Create compact WT/single/joint cache manifests for double mutants."""

    root = Path(root).resolve()
    cache_path = Path(cache_path).resolve()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = DatasetPaths(root)
    cache_sha256 = file_sha256(cache_path)
    full = paths.megascale / "Megascale-D-full.csv"
    full_frame = pd.read_csv(
        full,
        usecols=[
            "description",
            "ddG_ML",
            "ddG_ML_additive",
            "ddG_ML_epi_term",
        ],
    )
    if full_frame["description"].duplicated().any():
        raise RuntimeError("Megascale-D-full descriptions are not unique")
    epistasis = {
        str(row.description): (
            float(row.ddG_ML),
            float(row.ddG_ML_additive),
            float(row.ddG_ML_epi_term),
        )
        for row in full_frame.itertuples(index=False)
    }
    records: list[dict[str, object]] = []
    protein_sets: dict[str, set[str]] = {}
    with HierarchyEmbeddingReader(cache_path) as cache:
        for split_name in ("train", "val", "test"):
            source = paths.double(split_name)
            rows = list(double_rows(source))
            protein_sets[split_name] = {
                str(row["protein_id"]) for row in rows
            }
            wt_keys: list[tuple[str, int]] = []
            single_keys: list[tuple[str, int]] = []
            joint_keys: list[tuple[str, int]] = []
            for row in rows:
                positions = tuple(int(value) for value in row["positions"])
                wt_hash = sequence_hash(str(row["wt_sequence"]))
                joint_hash = sequence_hash(str(row["joint_sequence"]))
                for position, sequence in zip(
                    positions, row["single_sequences"], strict=True
                ):
                    wt_keys.append((wt_hash, position))
                    single_keys.append((sequence_hash(str(sequence)), position))
                    joint_keys.append((joint_hash, position))
            wt_site, wt_sequence = cache.locations(wt_keys)
            single_site, single_sequence = cache.locations(single_keys)
            joint_site, joint_sequence = cache.locations(joint_keys)
            count = len(rows)
            shape = (count, 2)
            matched = [epistasis.get(str(row["description"])) for row in rows]
            true_additive = np.asarray(
                [
                    np.nan if value is None else value[1]
                    for value in matched
                ],
                dtype=np.float32,
            )
            true_epistasis = np.asarray(
                [
                    np.nan if value is None else value[2]
                    for value in matched
                ],
                dtype=np.float32,
            )
            for row, value in zip(rows, matched, strict=True):
                if value is not None and not np.isclose(
                    float(row["target"]), value[0], atol=1e-6
                ):
                    raise RuntimeError(
                        f"double-mutant target mismatch for {row['description']}"
                    )
            target = output_dir / f"double_{split_name}.h5"

            def write(handle: h5py.File) -> None:
                handle.attrs["schema"] = HIERARCHY_MULTI_ROW_SCHEMA
                handle.attrs["kind"] = "double_ddg"
                handle.attrs["split"] = split_name
                handle.attrs["target_kind"] = "ddg"
                handle.attrs["favorable_direction"] = "negative"
                handle.attrs["source"] = str(source)
                handle.attrs["source_sha256"] = file_sha256(source)
                handle.attrs["hierarchy_cache"] = str(cache_path)
                handle.attrs["hierarchy_cache_sha256"] = cache_sha256
                handle.attrs["hierarchy_provenance"] = json.dumps(
                    cache.provenance, sort_keys=True, separators=(",", ":")
                )
                handle.attrs["window_radius"] = cache.window_radius
                handle.create_dataset(
                    "wt_site_index", data=wt_site.reshape(shape)
                )
                handle.create_dataset(
                    "wt_sequence_index", data=wt_sequence.reshape(shape)
                )
                handle.create_dataset(
                    "single_site_index", data=single_site.reshape(shape)
                )
                handle.create_dataset(
                    "single_sequence_index", data=single_sequence.reshape(shape)
                )
                handle.create_dataset(
                    "joint_site_index", data=joint_site.reshape(shape)
                )
                handle.create_dataset(
                    "joint_sequence_index", data=joint_sequence.reshape(shape)
                )
                handle.create_dataset(
                    "target",
                    data=np.asarray(
                        [float(row["target"]) for row in rows],
                        dtype=np.float32,
                    ),
                )
                handle.create_dataset("true_additive", data=true_additive)
                handle.create_dataset("true_epistasis", data=true_epistasis)
                handle.create_dataset(
                    "has_true_epistasis", data=np.isfinite(true_epistasis)
                )
                handle.create_dataset(
                    "structure",
                    data=np.zeros((count, 2, 128), dtype=np.float32),
                )
                handle.create_dataset(
                    "structure_mask", data=np.zeros((count, 2), dtype=bool)
                )
                handle.create_dataset(
                    "membrane",
                    data=np.zeros(
                        (count, 2, len(MEMBRANE_FEATURE_NAMES)),
                        dtype=np.float32,
                    ),
                )
                _string_dataset(
                    handle,
                    "protein_id",
                    [str(row["protein_id"]) for row in rows],
                )
                _string_dataset(
                    handle,
                    "mutation_1",
                    [str(row["mutations"][0]) for row in rows],
                )
                _string_dataset(
                    handle,
                    "mutation_2",
                    [str(row["mutations"][1]) for row in rows],
                )

            _atomic_h5(target, write)
            records.append(
                {
                    "path": str(target),
                    "sha256": file_sha256(target),
                    "split": split_name,
                    "rows": count,
                    "proteins": len(protein_sets[split_name]),
                    "rows_with_true_epistasis": int(
                        np.isfinite(true_epistasis).sum()
                    ),
                    "source_sha256": file_sha256(source),
                    "structure_policy": (
                        "missing-mask; double-mutant protein split does not "
                        "align with the single-mutant structure split"
                    ),
                }
            )
        hierarchy_provenance = cache.provenance
        window_radius = cache.window_radius
    if any(
        protein_sets[left] & protein_sets[right]
        for left, right in (("train", "val"), ("train", "test"), ("val", "test"))
    ):
        raise RuntimeError("double-mutant proteins overlap across splits")
    manifest = {
        "schema": HIERARCHY_MULTI_ROW_SCHEMA,
        "hierarchy_cache": str(cache_path),
        "hierarchy_cache_sha256": cache_sha256,
        "hierarchy_provenance": hierarchy_provenance,
        "window_radius": window_radius,
        "split_integrity": {
            name: len(values) for name, values in protein_sets.items()
        },
        "datasets": records,
    }
    manifest_path = output_dir / "multi_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    manifest["manifest_path"] = str(manifest_path)
    return manifest
