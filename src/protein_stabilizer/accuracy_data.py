"""Resumable masked-site evidence caches for accuracy training."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from .data import (
    AMINO_ACIDS,
    DatasetPaths,
    normalize_sequence,
    sequence_hash,
)
from .embeddings import (
    ESMCEmbedder,
    HierarchyEmbeddingReader,
    file_sha256,
)


MASKED_MARGINAL_SCHEMA = (
    "protein-stabilizer.esmc-masked-marginal-cache.v1"
)
TARGET_MASKED_MARGINAL_SCHEMA = (
    "protein-stabilizer.target-masked-marginal-cache.v1"
)


def _source_digest(paths: tuple[Path, Path]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.name.encode("utf-8"))
        digest.update(file_sha256(path).encode("ascii"))
    return digest.hexdigest()


def build_masked_marginal_cache(
    root: Path,
    output_path: Path,
    *,
    model_name: str = "esmc_600m",
    device: str = "cuda",
    inference_dtype: str = "float32",
    storage_dtype: str = "float32",
    max_tokens: int = 8192,
    max_batch_size: int = 128,
) -> dict[str, object]:
    """Cache one 20-state masked distribution per MegaScale mutation row."""

    paths = DatasetPaths(Path(root).resolve())
    train_path = paths.single("train").resolve()
    test_path = paths.single("test").resolve()
    if model_name == "esmc_600m":
        embedder = ESMCEmbedder(
            model_name=model_name,
            device=device,
            storage_dtype=storage_dtype,
        )
    else:
        from .esmc6b import ESMC6BEmbedder

        embedder = ESMC6BEmbedder(
            model_name,
            device,
            inference_dtype=inference_dtype,
            storage_dtype=storage_dtype,
        )
    model_provenance = embedder.provenance.canonical_json()
    frames: list[pd.DataFrame] = []
    for split, path in (
        ("development", train_path),
        ("historical_test", test_path),
    ):
        frame = pd.read_csv(path)
        required = {"pdb_id", "wt_seq", "pos1"}
        missing = required - set(frame)
        if missing:
            raise RuntimeError(f"{path} lacks columns {sorted(missing)}")
        frame["source_split"] = split
        frame["source_row"] = np.arange(len(frame), dtype=np.int64)
        frames.append(frame)
    rows = pd.concat(frames, ignore_index=True)
    source_digest = _source_digest((train_path, test_path))
    output = Path(output_path).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    with h5py.File(output, "a", libver="v110") as handle:
        if "log_probabilities" not in handle:
            handle.attrs["schema"] = MASKED_MARGINAL_SCHEMA
            handle.attrs["source_digest"] = source_digest
            handle.attrs["train_sha256"] = file_sha256(train_path)
            handle.attrs["test_sha256"] = file_sha256(test_path)
            handle.attrs["amino_acids"] = json.dumps(AMINO_ACIDS)
            handle.attrs["model_provenance"] = model_provenance
            handle.create_dataset(
                "log_probabilities",
                shape=(len(rows), len(AMINO_ACIDS)),
                dtype="f4",
                fillvalue=np.nan,
            )
            handle.create_dataset(
                "complete",
                shape=(len(rows),),
                dtype="bool",
                fillvalue=False,
            )
            handle.create_dataset(
                "source_split",
                data=rows["source_split"].to_numpy(dtype="S16"),
            )
            handle.create_dataset(
                "source_row",
                data=rows["source_row"].to_numpy(dtype=np.int64),
            )
        expected = {
            "schema": MASKED_MARGINAL_SCHEMA,
            "source_digest": source_digest,
            "train_sha256": file_sha256(train_path),
            "test_sha256": file_sha256(test_path),
            "amino_acids": json.dumps(AMINO_ACIDS),
            "model_provenance": model_provenance,
        }
        for name, value in expected.items():
            if str(handle.attrs.get(name, "")) != value:
                raise RuntimeError(
                    f"masked-marginal cache {name} provenance mismatch"
                )
        if (
            handle["log_probabilities"].shape
            != (len(rows), len(AMINO_ACIDS))
            or handle["complete"].shape != (len(rows),)
            or handle["source_split"].shape != (len(rows),)
            or handle["source_row"].shape != (len(rows),)
        ):
            raise RuntimeError("masked-marginal cache shape mismatch")

        groups = list(rows.groupby(["pdb_id", "wt_seq"], sort=True))
        computed = False
        for group_index, ((protein_id, sequence), group) in enumerate(
            groups, 1
        ):
            row_indices = group.index.to_numpy(dtype=np.int64)
            if bool(np.all(handle["complete"][row_indices])):
                continue
            positions = sorted(
                group["pos1"].astype(int).unique().tolist()
            )
            probability = embedder.masked_marginal_log_probabilities(
                str(sequence),
                positions,
                max_tokens=max_tokens,
                max_batch_size=max_batch_size,
            )
            by_position = {
                position: probability[index]
                for index, position in enumerate(positions)
            }
            handle["log_probabilities"][row_indices] = np.stack(
                [
                    by_position[int(position)]
                    for position in group["pos1"]
                ]
            )
            handle["complete"][row_indices] = True
            computed = True
            handle.flush()
            print(
                f"masked protein={group_index}/{len(groups)} "
                f"id={protein_id} sites={len(positions)} rows={len(group)}",
                flush=True,
            )
        complete_rows = int(np.sum(handle["complete"][:]))
        if computed:
            handle.attrs["complete_rows"] = complete_rows
            handle.attrs["elapsed_seconds_last_run"] = (
                time.monotonic() - started
            )
            handle.flush()
    return {
        "schema": MASKED_MARGINAL_SCHEMA,
        "output": str(output),
        "output_sha256": file_sha256(output),
        "rows": len(rows),
        "complete_rows": complete_rows,
        "model_provenance": json.loads(model_provenance),
        "source_digest": source_digest,
        "elapsed_seconds": time.monotonic() - started,
    }


def build_target_masked_marginal_cache(
    sequence: str,
    positions: tuple[int, ...],
    output_path: Path,
    *,
    model_name: str = "esmc_600m",
    device: str = "cuda",
    max_tokens: int = 8192,
    max_batch_size: int = 128,
    embedder: object | None = None,
) -> dict[str, object]:
    """Persist resumable 20-state masked probabilities for one target."""

    normalized = normalize_sequence(sequence)
    selected = tuple(sorted(set(int(position) for position in positions)))
    if not selected:
        raise ValueError("at least one target position is required")
    if any(position < 1 or position > len(normalized) for position in selected):
        raise ValueError("target masked position is outside the sequence")
    if max_tokens < len(normalized) + 2:
        raise ValueError("max_tokens is smaller than one tokenized sequence")
    if max_batch_size < 1:
        raise ValueError("max_batch_size must be positive")
    output = Path(output_path).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    expected_model_name = model_name
    if embedder is not None:
        expected_model_name = json.loads(
            embedder.provenance.canonical_json()
        ).get("model_name", model_name)
    if output.is_file():
        try:
            _, cached = load_target_masked_marginals(
                output, normalized, selected
            )
        except RuntimeError:
            cached = None
        if cached is not None:
            model = cached["model_provenance"]
            if model.get("model_name") != expected_model_name:
                raise RuntimeError(
                    "completed target masked-marginal cache has an "
                    "incompatible model"
                )
            for numeric in ("inference_dtype", "storage_dtype"):
                if numeric in model and model[numeric] != "float32":
                    raise RuntimeError(
                        "completed target masked-marginal cache has "
                        f"incompatible {numeric}"
                    )
            return {
                "schema": TARGET_MASKED_MARGINAL_SCHEMA,
                "output": str(output),
                "output_sha256": cached["sha256"],
                "sequence_hash": sequence_hash(normalized),
                "sequence_length": len(normalized),
                "positions": len(selected),
                "computed_sites": 0,
                "cache_hits": len(selected),
                "model_provenance": model,
                "elapsed_seconds": time.monotonic() - started,
            }
    active_embedder = embedder or ESMCEmbedder(
        model_name=model_name,
        device=device,
        storage_dtype="float32",
    )
    provenance = active_embedder.provenance.canonical_json()
    batch_size = min(
        max_batch_size,
        max(1, max_tokens // (len(normalized) + 2)),
    )
    computed_sites = 0
    with h5py.File(output, "a", libver="v110") as handle:
        if "log_probabilities" not in handle:
            handle.attrs["schema"] = TARGET_MASKED_MARGINAL_SCHEMA
            handle.attrs["sequence_hash"] = sequence_hash(normalized)
            handle.attrs["sequence_length"] = len(normalized)
            handle.attrs["amino_acids"] = json.dumps(AMINO_ACIDS)
            handle.attrs["model_provenance"] = provenance
            handle.create_dataset(
                "positions",
                data=np.asarray(selected, dtype=np.int64),
            )
            handle.create_dataset(
                "log_probabilities",
                shape=(len(selected), len(AMINO_ACIDS)),
                dtype="f4",
                fillvalue=np.nan,
            )
            handle.create_dataset(
                "complete",
                shape=(len(selected),),
                dtype="bool",
                fillvalue=False,
            )
        expected = {
            "schema": TARGET_MASKED_MARGINAL_SCHEMA,
            "sequence_hash": sequence_hash(normalized),
            "sequence_length": len(normalized),
            "amino_acids": json.dumps(AMINO_ACIDS),
            "model_provenance": provenance,
        }
        for name, value in expected.items():
            if str(handle.attrs.get(name, "")) != str(value):
                raise RuntimeError(
                    f"target masked-marginal cache {name} mismatch"
                )
        if (
            not np.array_equal(
                np.asarray(handle["positions"], dtype=np.int64),
                np.asarray(selected, dtype=np.int64),
            )
            or handle["log_probabilities"].shape
            != (len(selected), len(AMINO_ACIDS))
            or handle["complete"].shape != (len(selected),)
        ):
            raise RuntimeError(
                "target masked-marginal cache position/shape mismatch"
            )
        for start in range(0, len(selected), batch_size):
            stop = min(start + batch_size, len(selected))
            pending = np.flatnonzero(
                ~np.asarray(handle["complete"][start:stop], dtype=bool)
            )
            if not len(pending):
                continue
            rows = pending + start
            batch_positions = tuple(selected[index] for index in rows)
            values = active_embedder.masked_marginal_log_probabilities(
                normalized,
                batch_positions,
                max_tokens=max_tokens,
                max_batch_size=max_batch_size,
            )
            if values.shape != (len(rows), len(AMINO_ACIDS)):
                raise RuntimeError(
                    "target masked-marginal embedder returned wrong shape"
                )
            handle["log_probabilities"][rows] = values
            handle["complete"][rows] = True
            computed_sites += len(rows)
            handle.flush()
        complete_sites = int(np.sum(handle["complete"][:]))
        if complete_sites != len(selected):
            raise RuntimeError("target masked-marginal cache is incomplete")
        if computed_sites:
            handle.attrs["complete_sites"] = complete_sites
            handle.attrs["elapsed_seconds_last_run"] = (
                time.monotonic() - started
            )
            handle.flush()
    return {
        "schema": TARGET_MASKED_MARGINAL_SCHEMA,
        "output": str(output),
        "output_sha256": file_sha256(output),
        "sequence_hash": sequence_hash(normalized),
        "sequence_length": len(normalized),
        "positions": len(selected),
        "computed_sites": computed_sites,
        "cache_hits": len(selected) - computed_sites,
        "model_provenance": json.loads(provenance),
        "elapsed_seconds": time.monotonic() - started,
    }


def load_target_masked_marginals(
    path: Path,
    sequence: str,
    positions: tuple[int, ...],
) -> tuple[np.ndarray, dict[str, object]]:
    """Load requested sites from a provenance-bearing target cache."""

    normalized = normalize_sequence(sequence)
    selected = tuple(int(position) for position in positions)
    if len(selected) != len(set(selected)):
        raise ValueError("target masked positions must be unique")
    resolved = Path(path).resolve()
    with h5py.File(resolved, "r") as handle:
        if handle.attrs.get("schema") != TARGET_MASKED_MARGINAL_SCHEMA:
            raise RuntimeError("target masked-marginal schema mismatch")
        if str(handle.attrs.get("sequence_hash")) != sequence_hash(
            normalized
        ):
            raise RuntimeError(
                "target masked-marginal sequence hash mismatch"
            )
        if str(handle.attrs.get("amino_acids")) != json.dumps(AMINO_ACIDS):
            raise RuntimeError(
                "target masked-marginal amino-acid order mismatch"
            )
        cached_positions = np.asarray(
            handle["positions"], dtype=np.int64
        )
        complete = np.asarray(handle["complete"], dtype=bool)
        values = np.asarray(
            handle["log_probabilities"], dtype=np.float32
        )
        if (
            complete.shape != (len(cached_positions),)
            or values.shape
            != (len(cached_positions), len(AMINO_ACIDS))
        ):
            raise RuntimeError("target masked-marginal cache is malformed")
        by_position = {
            int(position): index
            for index, position in enumerate(cached_positions)
        }
        missing = [
            position for position in selected if position not in by_position
        ]
        if missing:
            raise RuntimeError(
                "target masked-marginal cache lacks requested positions "
                f"{missing[:10]}"
            )
        row = np.asarray(
            [by_position[position] for position in selected],
            dtype=np.int64,
        )
        if not np.all(complete[row]):
            raise RuntimeError(
                "target masked-marginal requested rows are incomplete"
            )
        if not np.isfinite(values[row]).all():
            raise RuntimeError(
                "target masked-marginal requested rows are invalid"
            )
        model_provenance = json.loads(
            str(handle.attrs["model_provenance"])
        )
    return values[row], {
        "path": str(resolved),
        "sha256": file_sha256(resolved),
        "schema": TARGET_MASKED_MARGINAL_SCHEMA,
        "sequence_hash": sequence_hash(normalized),
        "positions": len(selected),
        "model_provenance": model_provenance,
    }


def load_target_state_embeddings(
    path: Path,
    sequence: str,
) -> tuple[np.ndarray, dict[str, object]]:
    """Read every WT residue state from a hierarchy application cache."""

    normalized = normalize_sequence(sequence)
    resolved = Path(path).resolve()
    keys = [
        (sequence_hash(normalized), position)
        for position in range(1, len(normalized) + 1)
    ]
    try:
        with HierarchyEmbeddingReader(resolved) as reader:
            if reader.window_radius != 4:
                raise RuntimeError(
                    "target state embedding cache must use radius four"
                )
            features = reader.features(keys)
            provenance = dict(reader.provenance)
    except KeyError as exc:
        raise RuntimeError(
            "target state embedding cache does not cover every WT residue"
        ) from exc
    window = np.asarray(features["window"], dtype=np.float32)
    window_mask = np.asarray(features["window_mask"], dtype=bool)
    if (
        window.shape[0] != len(normalized)
        or window.shape[1] != 9
        or window_mask.shape != window.shape[:2]
        or not np.all(window_mask[:, 4])
    ):
        raise RuntimeError("target state embedding cache is malformed")
    residue = window[:, 4]
    global_mean = np.asarray(
        features["global_mean"][0], dtype=np.float32
    )
    maximum_global_difference = float(
        np.max(np.abs(residue.mean(axis=0) - global_mean))
    )
    if maximum_global_difference > 2.0e-5:
        raise RuntimeError(
            "target state embedding cache global mean is inconsistent"
        )
    return residue, {
        "path": str(resolved),
        "sha256": file_sha256(resolved),
        "sequence_hash": sequence_hash(normalized),
        "positions": len(normalized),
        "model_provenance": provenance,
        "maximum_global_difference": maximum_global_difference,
    }
