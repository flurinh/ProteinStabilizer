"""Resumable masked-site evidence caches for accuracy training."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from .data import AMINO_ACIDS, DatasetPaths
from .embeddings import ESMCEmbedder, file_sha256


MASKED_MARGINAL_SCHEMA = (
    "protein-stabilizer.esmc-masked-marginal-cache.v1"
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
