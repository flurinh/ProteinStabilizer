"""Materialize row-aligned embedding-delta feature files."""

from __future__ import annotations

import hashlib
import json
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
)
from .embeddings import ResidueEmbeddingReader, file_sha256


FEATURE_SCHEMA = "protein-stabilizer.embedding-delta-features.v1"


def _string_dataset(handle: h5py.File, name: str, values: Sequence[str]) -> None:
    handle.create_dataset(
        name,
        data=np.asarray(values, dtype=object),
        dtype=h5py.string_dtype(encoding="utf-8"),
    )


def _atomic_h5(path: Path, writer: Callable[[h5py.File], None]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.unlink(missing_ok=True)
    with h5py.File(temporary, "w", libver="latest") as handle:
        writer(handle)
        handle.flush()
    temporary.replace(path)


def _protein_validation_split(protein_ids: Sequence[str], seed: int) -> np.ndarray:
    proteins = np.array(sorted(set(protein_ids)))
    rng = np.random.default_rng(seed)
    rng.shuffle(proteins)
    n_val = max(1, round(0.10 * len(proteins)))
    validation = set(proteins[:n_val])
    return np.asarray(
        ["val" if protein in validation else "train" for protein in protein_ids],
        dtype=object,
    )


def _write_single_features(
    source: Path,
    target: Path,
    cache: ResidueEmbeddingReader,
    *,
    training_source: bool,
    seed: int,
    chunk_size: int,
) -> dict[str, object]:
    rows = list(single_rows(source))

    def write(handle: h5py.File) -> None:
        handle.attrs["schema"] = FEATURE_SCHEMA
        handle.attrs["kind"] = "single"
        handle.attrs["source"] = str(source)
        handle.attrs["source_sha256"] = file_sha256(source)
        handle.attrs["embedding_provenance"] = json.dumps(
            cache.provenance, sort_keys=True, separators=(",", ":")
        )
        count = len(rows)
        delta = handle.create_dataset(
            "delta",
            shape=(count, cache.dimension),
            dtype="f2",
            chunks=(min(chunk_size, count), cache.dimension),
        )
        for start in range(0, count, chunk_size):
            batch = rows[start : start + chunk_size]
            wt_keys = [
                (sequence_hash(str(row["wt_sequence"])), int(row["position"]))
                for row in batch
            ]
            mutant_keys = [
                (sequence_hash(str(row["mutant_sequence"])), int(row["position"]))
                for row in batch
            ]
            values = cache.vectors(mutant_keys).astype(np.float32)
            values -= cache.vectors(wt_keys).astype(np.float32)
            delta[start : start + len(batch)] = values.astype(np.float16)
        handle.create_dataset(
            "target", data=np.asarray([row["target"] for row in rows], dtype=np.float32)
        )
        handle.create_dataset(
            "position",
            data=np.asarray([row["position"] for row in rows], dtype=np.int32),
        )
        protein_ids = [str(row["protein_id"]) for row in rows]
        _string_dataset(handle, "protein_id", protein_ids)
        _string_dataset(handle, "mutation", [str(row["mutation"]) for row in rows])
        split = (
            _protein_validation_split(protein_ids, seed)
            if training_source
            else np.asarray(["test"] * count, dtype=object)
        )
        _string_dataset(handle, "split", split.tolist())

    _atomic_h5(target, write)
    return {
        "path": str(target),
        "kind": "single",
        "rows": len(rows),
        "proteins": len({str(row["protein_id"]) for row in rows}),
        "source_sha256": file_sha256(source),
    }


def _write_double_features(
    source: Path,
    target: Path,
    cache: ResidueEmbeddingReader,
    *,
    split: str,
    epistasis_lookup: dict[str, tuple[float, float, float]],
    chunk_size: int,
) -> dict[str, object]:
    rows = list(double_rows(source))

    def write(handle: h5py.File) -> None:
        handle.attrs["schema"] = FEATURE_SCHEMA
        handle.attrs["kind"] = "double"
        handle.attrs["split"] = split
        handle.attrs["source"] = str(source)
        handle.attrs["source_sha256"] = file_sha256(source)
        handle.attrs["embedding_provenance"] = json.dumps(
            cache.provenance, sort_keys=True, separators=(",", ":")
        )
        count = len(rows)
        shape = (count, 2, cache.dimension)
        single_delta = handle.create_dataset(
            "single_delta",
            shape=shape,
            dtype="f2",
            chunks=(min(chunk_size, count), 2, cache.dimension),
        )
        joint_delta = handle.create_dataset(
            "joint_delta",
            shape=shape,
            dtype="f2",
            chunks=(min(chunk_size, count), 2, cache.dimension),
        )
        for start in range(0, count, chunk_size):
            batch = rows[start : start + chunk_size]
            wt_keys: list[tuple[str, int]] = []
            single_keys: list[tuple[str, int]] = []
            joint_keys: list[tuple[str, int]] = []
            for row in batch:
                positions = tuple(int(value) for value in row["positions"])
                wt_hash = sequence_hash(str(row["wt_sequence"]))
                joint_hash = sequence_hash(str(row["joint_sequence"]))
                for position, sequence in zip(
                    positions, row["single_sequences"], strict=True
                ):
                    wt_keys.append((wt_hash, position))
                    single_keys.append((sequence_hash(str(sequence)), position))
                    joint_keys.append((joint_hash, position))
            wt_values = cache.vectors(wt_keys).astype(np.float32)
            singles = cache.vectors(single_keys).astype(np.float32) - wt_values
            joints = cache.vectors(joint_keys).astype(np.float32) - wt_values
            batch_shape = (len(batch), 2, cache.dimension)
            single_delta[start : start + len(batch)] = singles.reshape(batch_shape).astype(
                np.float16
            )
            joint_delta[start : start + len(batch)] = joints.reshape(batch_shape).astype(
                np.float16
            )
        handle.create_dataset(
            "target", data=np.asarray([row["target"] for row in rows], dtype=np.float32)
        )
        true_additive = np.full(count, np.nan, dtype=np.float32)
        true_epistasis = np.full(count, np.nan, dtype=np.float32)
        for index, row in enumerate(rows):
            matched = epistasis_lookup.get(str(row["description"]))
            if matched is None:
                continue
            matched_target, additive, epistasis = matched
            if not np.isclose(float(row["target"]), matched_target, atol=1e-6):
                raise ValueError(f"double-mutant target mismatch for {row['description']}")
            true_additive[index] = additive
            true_epistasis[index] = epistasis
        handle.create_dataset("true_additive", data=true_additive)
        handle.create_dataset("true_epistasis", data=true_epistasis)
        handle.create_dataset("has_true_epistasis", data=np.isfinite(true_epistasis))
        _string_dataset(
            handle, "protein_id", [str(row["protein_id"]) for row in rows]
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
    return {
        "path": str(target),
        "kind": "double",
        "split": split,
        "rows": len(rows),
        "proteins": len({str(row["protein_id"]) for row in rows}),
        "rows_with_true_epistasis": sum(
            str(row["description"]) in epistasis_lookup for row in rows
        ),
        "source_sha256": file_sha256(source),
    }


def _write_gpcr_features(
    source: Path,
    target: Path,
    cache: ResidueEmbeddingReader,
    *,
    seed: int,
    chunk_size: int,
) -> dict[str, object]:
    frame = gpcr_site_splits(source, seed=seed)
    rows = frame.to_dict("records")

    def write(handle: h5py.File) -> None:
        handle.attrs["schema"] = FEATURE_SCHEMA
        handle.attrs["kind"] = "gpcr"
        handle.attrs["split_seed"] = seed
        handle.attrs["source"] = str(source)
        handle.attrs["source_sha256"] = file_sha256(source)
        handle.attrs["embedding_provenance"] = json.dumps(
            cache.provenance, sort_keys=True, separators=(",", ":")
        )
        count = len(rows)
        delta = handle.create_dataset(
            "delta",
            shape=(count, cache.dimension),
            dtype="f2",
            chunks=(min(chunk_size, count), cache.dimension),
        )
        for start in range(0, count, chunk_size):
            batch = rows[start : start + chunk_size]
            wt_keys = [
                (sequence_hash(str(row["wt_sequence"])), int(row["position"]))
                for row in batch
            ]
            mutant_keys = [
                (sequence_hash(str(row["mutant_sequence"])), int(row["position"]))
                for row in batch
            ]
            values = cache.vectors(mutant_keys).astype(np.float32)
            values -= cache.vectors(wt_keys).astype(np.float32)
            delta[start : start + len(batch)] = values.astype(np.float16)
        handle.create_dataset(
            "target", data=np.asarray([row["target"] for row in rows], dtype=np.float32)
        )
        handle.create_dataset(
            "position",
            data=np.asarray([row["position"] for row in rows], dtype=np.int32),
        )
        for name in ("protein_id", "assay_id", "mutation", "site_id", "split"):
            _string_dataset(handle, name, [str(row[name]) for row in rows])

    _atomic_h5(target, write)
    split_counts = frame["split"].value_counts().sort_index().to_dict()
    return {
        "path": str(target),
        "kind": "gpcr",
        "rows": len(rows),
        "unique_sites": int(frame["site_id"].nunique()),
        "split_counts": {str(key): int(value) for key, value in split_counts.items()},
        "source_sha256": file_sha256(source),
    }


def build_feature_files(
    root: Path,
    cache_path: Path,
    output_dir: Path,
    *,
    seed: int = 20260715,
    chunk_size: int = 4096,
) -> dict[str, object]:
    paths = DatasetPaths(Path(root))
    output_dir = Path(output_dir)
    records: list[dict[str, object]] = []
    full_double_path = paths.megascale / "Megascale-D-full.csv"
    full_double = pd.read_csv(
        full_double_path,
        usecols=["description", "ddG_ML", "ddG_ML_additive", "ddG_ML_epi_term"],
    )
    if full_double["description"].duplicated().any():
        raise ValueError("Megascale-D-full contains duplicate descriptions")
    epistasis_lookup = {
        str(row.description): (
            float(row.ddG_ML),
            float(row.ddG_ML_additive),
            float(row.ddG_ML_epi_term),
        )
        for row in full_double.itertuples(index=False)
    }
    with ResidueEmbeddingReader(cache_path) as cache:
        records.append(
            _write_single_features(
                paths.single("train"),
                output_dir / "single_train.h5",
                cache,
                training_source=True,
                seed=seed,
                chunk_size=chunk_size,
            )
        )
        records.append(
            _write_single_features(
                paths.single("test"),
                output_dir / "single_test.h5",
                cache,
                training_source=False,
                seed=seed,
                chunk_size=chunk_size,
            )
        )
        for split in ("train", "val", "test"):
            records.append(
                _write_double_features(
                    paths.double(split),
                    output_dir / f"double_{split}.h5",
                    cache,
                    split=split,
                    epistasis_lookup=epistasis_lookup,
                    chunk_size=max(256, chunk_size // 2),
                )
            )
        records.append(
            _write_gpcr_features(
                paths.gpcr,
                output_dir / "gpcr.h5",
                cache,
                seed=seed,
                chunk_size=chunk_size,
            )
        )
        embedding_provenance = cache.provenance
    manifest = {
        "schema": FEATURE_SCHEMA,
        "seed": seed,
        "embedding_cache": str(cache_path),
        "embedding_cache_sha256": file_sha256(cache_path),
        "embedding_provenance": embedding_provenance,
        "datasets": records,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    manifest["manifest_path"] = str(manifest_path)
    return manifest
