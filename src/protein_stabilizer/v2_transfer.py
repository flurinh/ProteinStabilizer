"""Separate experimental ddG, delta-Tm, and GPCR ranking adapters for v2."""

from __future__ import annotations

import copy
import json
import math
import time
from dataclasses import asdict
from pathlib import Path

import h5py
import numpy as np
import torch
from scipy.stats import spearmanr

from .embeddings import file_sha256
from .models import DirectionalAssayConfig, DirectionalAssayHead
from .training import regression_metrics, set_reproducible_seed
from .v2_training import (
    _batch_tensors,
    _load_rows,
    load_hierarchical_ensemble,
)


TRANSFER_REPRESENTATION_SCHEMA = (
    "protein-stabilizer.hierarchical-transfer-representations.v1"
)
TRANSFER_HEAD_SCHEMA = "protein-stabilizer.directional-assay-head.v1"


def _base_representations(
    model: torch.nn.Module,
    data: object,
    *,
    use_structure: bool,
    batch_size: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    latent: list[np.ndarray] = []
    ddg: list[np.ndarray] = []
    indices = np.arange(len(data.target), dtype=np.int64)
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(indices), batch_size):
            selected = indices[start : start + batch_size]
            tensors = _batch_tensors(
                data,
                selected,
                device,
                use_structure=use_structure,
            )
            latent.append(model.latent(**tensors).float().cpu().numpy())
            ddg.append(model(**tensors).float().cpu().numpy())
    return np.concatenate(latent), np.concatenate(ddg)


def build_transfer_representations(
    feature_dir: Path,
    cache_path: Path,
    base_checkpoint: Path,
    output_dir: Path,
    *,
    batch_size: int = 256,
    device: str = "cuda",
) -> dict[str, object]:
    """Cache compact frozen-base inputs for every staged assay endpoint."""

    feature_dir = Path(feature_dir).resolve()
    cache_path = Path(cache_path).resolve()
    base_checkpoint = Path(base_checkpoint).resolve()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    torch_device = torch.device(
        device if device.startswith("cuda") and torch.cuda.is_available() else "cpu"
    )
    if torch_device.type == "cuda":
        torch.set_float32_matmul_precision("highest")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    base, payload = load_hierarchical_ensemble(base_checkpoint, torch_device)
    cache_sha256 = file_sha256(cache_path)
    records: list[dict[str, object]] = []
    specs = (
        ("protherm", "ddg", "kcal/mol", "negative"),
        ("mcsm_membrane", "ddg", "kcal/mol", "negative"),
        ("mptherm", "delta_tm", "degrees Celsius", "positive"),
        ("gpcr_tm", "delta_tm", "degrees Celsius", "positive"),
        (
            "gpcr_rank",
            "stability_delta_percent",
            "rank-only assay score",
            "positive",
        ),
    )
    for name, target_kind, units, favorable in specs:
        row_path = feature_dir / f"{name}.h5"
        data = _load_rows(
            row_path,
            cache_path,
            expected_cache_sha256=cache_sha256,
        )
        latent, base_ddg = _base_representations(
            base,
            data,
            use_structure=bool(payload["use_structure"]),
            batch_size=batch_size,
            device=torch_device,
        )
        output = output_dir / f"{name}.h5"
        partial = output.with_suffix(".h5.partial")
        partial.unlink(missing_ok=True)
        with h5py.File(row_path, "r") as source, h5py.File(
            partial, "w", libver="latest"
        ) as handle:
            handle.attrs["schema"] = TRANSFER_REPRESENTATION_SCHEMA
            handle.attrs["name"] = name
            handle.attrs["target_kind"] = target_kind
            handle.attrs["target_units"] = units
            handle.attrs["favorable_direction"] = favorable
            handle.attrs["row_source"] = str(row_path)
            handle.attrs["row_source_sha256"] = file_sha256(row_path)
            handle.attrs["source_sha256"] = str(source.attrs["source_sha256"])
            handle.attrs["hierarchy_cache_sha256"] = cache_sha256
            handle.attrs["base_checkpoint"] = str(base_checkpoint)
            handle.attrs["base_checkpoint_sha256"] = file_sha256(
                base_checkpoint
            )
            handle.attrs["base_candidate"] = str(payload["candidate"])
            handle.attrs["base_use_structure"] = bool(payload["use_structure"])
            handle.create_dataset("base_latent", data=latent.astype(np.float32))
            handle.create_dataset("base_ddg", data=base_ddg.astype(np.float32))
            handle.create_dataset("membrane", data=data.membrane)
            handle.create_dataset("target", data=data.target)
            handle.create_dataset("sample_weight", data=data.sample_weight)
            string_dtype = h5py.string_dtype(encoding="utf-8")
            handle.create_dataset(
                "protein_id",
                data=np.asarray(data.protein_id, dtype=object),
                dtype=string_dtype,
            )
            handle.create_dataset(
                "split",
                data=np.asarray(data.split, dtype=object),
                dtype=string_dtype,
            )
            for optional in ("assay_id", "site_id", "uniprot_id"):
                if optional in source:
                    handle.create_dataset(
                        optional,
                        data=np.asarray(source[optional].asstr()[:], dtype=object),
                        dtype=string_dtype,
                    )
            handle.flush()
        partial.replace(output)
        records.append(
            {
                "name": name,
                "path": str(output),
                "sha256": file_sha256(output),
                "rows": len(data.target),
                "proteins": len(set(data.protein_id.tolist())),
                "target_kind": target_kind,
                "target_units": units,
                "favorable_direction": favorable,
                "source_sha256": data.provenance["source_sha256"],
            }
        )
        print(f"transfer representations {name}: {len(data.target):,} rows")
    manifest = {
        "schema": TRANSFER_REPRESENTATION_SCHEMA,
        "base_checkpoint": {
            "path": str(base_checkpoint),
            "sha256": file_sha256(base_checkpoint),
            "candidate": payload["candidate"],
            "use_structure": payload["use_structure"],
        },
        "hierarchy_cache": {
            "path": str(cache_path),
            "sha256": cache_sha256,
        },
        "datasets": records,
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    manifest["manifest_path"] = str(manifest_path)
    return manifest


def _load_transfer(path: Path) -> dict[str, object]:
    with h5py.File(path, "r") as handle:
        if handle.attrs.get("schema") != TRANSFER_REPRESENTATION_SCHEMA:
            raise RuntimeError(f"{path} transfer representation schema mismatch")
        value: dict[str, object] = {
            "latent": np.asarray(handle["base_latent"], dtype=np.float32),
            "base_ddg": np.asarray(handle["base_ddg"], dtype=np.float32),
            "membrane": np.asarray(handle["membrane"], dtype=np.float32),
            "target": np.asarray(handle["target"], dtype=np.float32),
            "sample_weight": np.asarray(
                handle["sample_weight"], dtype=np.float32
            ),
            "protein_id": np.asarray(handle["protein_id"].asstr()[:]),
            "split": np.asarray(handle["split"].asstr()[:]),
            "target_kind": str(handle.attrs["target_kind"]),
            "target_units": str(handle.attrs["target_units"]),
            "favorable_direction": str(handle.attrs["favorable_direction"]),
            "source_sha256": str(handle.attrs["source_sha256"]),
            "row_source_sha256": str(handle.attrs["row_source_sha256"]),
            "base_checkpoint_sha256": str(
                handle.attrs["base_checkpoint_sha256"]
            ),
        }
        for optional in ("assay_id", "site_id", "uniprot_id"):
            if optional in handle:
                value[optional] = np.asarray(handle[optional].asstr()[:])
        return value


def _derived_validation_split(
    split: np.ndarray,
    group_id: np.ndarray,
    *,
    seed: int,
) -> np.ndarray:
    """Make evaluation groups disjoint and add group-held-out validation.

    Explicit test assignments are authoritative when a source assigned
    different rows from the same protein or site to both train and test.
    Explicit validation assignments are likewise authoritative over train.
    Rows outside train/validation/test retain their source designation.
    """

    result = np.asarray(split, dtype=object).copy()
    result[result == "validation"] = "val"
    group_id = np.asarray(group_id)
    test_groups = set(group_id[result == "test"].tolist())
    if test_groups:
        belongs_to_test = np.asarray(
            [group in test_groups for group in group_id]
        )
        result[
            np.isin(result, ["train", "val"]) & belongs_to_test
        ] = "test"
    validation_groups = set(group_id[result == "val"].tolist())
    if validation_groups:
        belongs_to_validation = np.asarray(
            [group in validation_groups for group in group_id]
        )
        result[(result == "train") & belongs_to_validation] = "val"
    if np.any(result == "val"):
        return result.astype(str)
    training_groups = np.asarray(
        sorted(set(group_id[result == "train"].tolist())), dtype=object
    )
    if len(training_groups) < 2:
        raise RuntimeError("not enough training groups for validation")
    group_counts = {
        group: int(np.sum((result == "train") & (group_id == group)))
        for group in training_groups
    }
    validation_candidates = np.asarray(
        [group for group in training_groups if group_counts[group] >= 2],
        dtype=object,
    )
    if not len(validation_candidates):
        validation_candidates = training_groups.copy()
    rng = np.random.default_rng(seed)
    rng.shuffle(validation_candidates)
    count = max(1, round(0.20 * len(training_groups)))
    count = min(
        count, len(validation_candidates), len(training_groups) - 1
    )
    heldout = set(validation_candidates[:count].tolist())
    result[
        (result == "train")
        & np.asarray([group in heldout for group in group_id])
    ] = "val"
    return result.astype(str)


def _adapter_predictions(
    model: DirectionalAssayHead,
    data: dict[str, object],
    indices: np.ndarray,
    *,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    output: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(indices), batch_size):
            selected = indices[start : start + batch_size]
            output.append(
                model(
                    torch.from_numpy(
                        np.asarray(data["latent"])[selected].astype(np.float32)
                    ).to(device),
                    torch.from_numpy(
                        np.asarray(data["base_ddg"])[selected].astype(np.float32)
                    ).to(device),
                    torch.from_numpy(
                        np.asarray(data["membrane"])[selected].astype(np.float32)
                    ).to(device),
                )
                .float()
                .cpu()
                .numpy()
            )
    return np.concatenate(output) if output else np.empty(0, dtype=np.float32)


def _macro_spearman(
    target: np.ndarray,
    prediction: np.ndarray,
    group: np.ndarray,
) -> float:
    values: list[float] = []
    for name in sorted(set(group.tolist())):
        selected = group == name
        if selected.sum() < 2:
            continue
        value = float(spearmanr(target[selected], prediction[selected]).statistic)
        if np.isfinite(value):
            values.append(value)
    return float(np.mean(values)) if values else math.nan


def _train_adapter(
    name: str,
    data: dict[str, object],
    *,
    seed: int,
    epochs: int,
    minimum_epochs: int,
    patience: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    device: torch.device,
) -> tuple[DirectionalAssayHead, dict[str, object]]:
    rank_only = str(data["target_units"]) == "rank-only assay score"
    split_group = (
        np.asarray(data["site_id"])
        if rank_only and "site_id" in data
        else np.asarray(data["protein_id"])
    )
    split = _derived_validation_split(
        np.asarray(data["split"]),
        split_group,
        seed=seed,
    )
    allowed = np.isin(split, ["train", "val", "test"])
    train_indices = np.flatnonzero((split == "train") & allowed)
    val_indices = np.flatnonzero((split == "val") & allowed)
    test_indices = np.flatnonzero((split == "test") & allowed)
    protein_id = np.asarray(data["protein_id"])
    if (
        set(split_group[train_indices]) & set(split_group[val_indices])
        or set(split_group[train_indices]) & set(split_group[test_indices])
        or set(split_group[val_indices]) & set(split_group[test_indices])
    ):
        raise RuntimeError(f"{name} evaluation groups overlap across adapter splits")
    config = DirectionalAssayConfig(
        latent_dim=np.asarray(data["latent"]).shape[1],
        membrane_dim=np.asarray(data["membrane"]).shape[1],
    )
    set_reproducible_seed(seed)
    model = DirectionalAssayHead(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=0.05 * learning_rate
    )
    huber = torch.nn.HuberLoss(delta=1.0, reduction="none")
    rng = np.random.default_rng(seed)
    target = np.asarray(data["target"])
    target_scale = max(float(np.std(target[train_indices])), 1e-6)
    training_target = target / target_scale if rank_only else target
    best_state: dict[str, torch.Tensor] | None = None
    best_score = -math.inf
    best_epoch = 0
    stale = 0
    history: list[dict[str, object]] = []
    optimizer_steps = 0
    examples_seen = 0
    for epoch in range(1, epochs + 1):
        model.train()
        shuffled = rng.permutation(train_indices)
        total = 0.0
        seen = 0
        for start in range(0, len(shuffled), batch_size):
            selected = shuffled[start : start + batch_size]
            latent = torch.from_numpy(
                np.asarray(data["latent"])[selected].astype(np.float32)
            ).to(device)
            base_ddg = torch.from_numpy(
                np.asarray(data["base_ddg"])[selected].astype(np.float32)
            ).to(device)
            membrane = torch.from_numpy(
                np.asarray(data["membrane"])[selected].astype(np.float32)
            ).to(device)
            values = torch.from_numpy(
                training_target[selected].astype(np.float32)
            ).to(device)
            weights = torch.from_numpy(
                np.asarray(data["sample_weight"])[selected].astype(np.float32)
            ).to(device)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(latent, base_ddg, membrane)
            loss = (huber(prediction, values) * weights).sum() / weights.sum()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            optimizer_steps += 1
            examples_seen += len(selected)
            total += float(loss.detach()) * len(selected)
            seen += len(selected)
        scheduler.step()
        validation_prediction = _adapter_predictions(
            model,
            data,
            val_indices,
            batch_size=batch_size,
            device=device,
        )
        validation_target = (
            training_target[val_indices]
            if rank_only
            else target[val_indices]
        )
        validation = regression_metrics(
            validation_target, validation_prediction
        )
        score = (
            0.85 * float(validation["spearman"])
            - 0.15 * float(validation["mae"]) / target_scale
        )
        history.append(
            {
                "epoch": epoch,
                "train_huber": total / seen,
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
                "optimizer_steps": optimizer_steps,
                "examples_seen": examples_seen,
                "validation": validation,
                "selection_score": score,
            }
        )
        if score > best_score + 1e-5:
            best_score = score
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
            if epoch >= minimum_epochs and stale >= patience:
                break
    if best_state is None:
        raise RuntimeError(f"{name} adapter has no valid validation checkpoint")
    model.load_state_dict(best_state)
    evaluation: dict[str, object] = {}
    for split_name, indices in (("validation", val_indices), ("test", test_indices)):
        prediction = _adapter_predictions(
            model, data, indices, batch_size=batch_size, device=device
        )
        selected_target = (
            training_target[indices] if rank_only else target[indices]
        )
        metrics = regression_metrics(selected_target, prediction)
        metrics["macro_protein_spearman"] = _macro_spearman(
            selected_target, prediction, protein_id[indices]
        )
        if "assay_id" in data:
            metrics["macro_assay_spearman"] = _macro_spearman(
                selected_target,
                prediction,
                np.asarray(data["assay_id"])[indices],
            )
        evaluation[split_name] = metrics
    reverse_count = min(512, len(test_indices) or len(val_indices))
    reverse_indices = (
        test_indices[:reverse_count]
        if len(test_indices)
        else val_indices[:reverse_count]
    )
    latent = torch.from_numpy(
        np.asarray(data["latent"])[reverse_indices].astype(np.float32)
    ).to(device)
    base_ddg = torch.from_numpy(
        np.asarray(data["base_ddg"])[reverse_indices].astype(np.float32)
    ).to(device)
    membrane = torch.from_numpy(
        np.asarray(data["membrane"])[reverse_indices].astype(np.float32)
    ).to(device)
    model.eval()
    with torch.inference_mode():
        forward = model(latent, base_ddg, membrane)
        reverse = model(-latent, -base_ddg, membrane)
    metrics = {
        "name": name,
        "target_kind": data["target_kind"],
        "target_units": data["target_units"],
        "favorable_direction": data["favorable_direction"],
        "rank_only": rank_only,
        "target_scale": target_scale,
        "split_integrity": {
            "train_rows": int(len(train_indices)),
            "validation_rows": int(len(val_indices)),
            "test_rows": int(len(test_indices)),
            "train_proteins": len(set(protein_id[train_indices])),
            "validation_proteins": len(set(protein_id[val_indices])),
            "test_proteins": len(set(protein_id[test_indices])),
            "group_policy": (
                "whole (protein, mutation site) groups"
                if rank_only
                else "whole source proteins/receptors"
            ),
            "overlap": 0,
        },
        "training": {
            "epoch_budget": epochs,
            "minimum_epochs": minimum_epochs,
            "best_epoch": best_epoch,
            "optimizer_steps": optimizer_steps,
            "examples_seen": examples_seen,
            "history": history,
        },
        "evaluation": evaluation,
        "directional_constraints": {
            "evaluated_rows": reverse_count,
            "maximum_absolute_forward_plus_reverse": float(
                torch.max(torch.abs(forward + reverse)).cpu()
            ),
        },
    }
    return model, metrics


def train_transfer_adapters(
    representation_dir: Path,
    checkpoint_dir: Path,
    *,
    seed: int = 20260715,
    epochs: int = 120,
    minimum_epochs: int = 50,
    patience: int = 12,
    batch_size: int = 256,
    learning_rate: float = 5e-4,
    weight_decay: float = 1e-4,
    device: str = "cuda",
) -> dict[str, object]:
    """Train endpoint-specific heads without mixing units or target meanings."""

    representation_dir = Path(representation_dir).resolve()
    checkpoint_dir = Path(checkpoint_dir).resolve()
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    torch_device = torch.device(
        device if device.startswith("cuda") and torch.cuda.is_available() else "cpu"
    )
    if torch_device.type == "cuda":
        torch.set_float32_matmul_precision("highest")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    results: dict[str, object] = {}
    base_hashes: set[str] = set()
    started = time.monotonic()
    for offset, name in enumerate(
        ("protherm", "mcsm_membrane", "mptherm", "gpcr_tm", "gpcr_rank")
    ):
        data = _load_transfer(representation_dir / f"{name}.h5")
        base_hashes.add(str(data["base_checkpoint_sha256"]))
        model, metrics = _train_adapter(
            name,
            data,
            seed=seed + offset,
            epochs=epochs,
            minimum_epochs=minimum_epochs,
            patience=patience,
            batch_size=batch_size,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            device=torch_device,
        )
        torch.save(
            {
                "schema": TRANSFER_HEAD_SCHEMA,
                "name": name,
                "config": model.config_dict(),
                "state_dict": copy.deepcopy(model.cpu().state_dict()),
                "metrics": metrics,
                "source_sha256": data["source_sha256"],
                "row_source_sha256": data["row_source_sha256"],
                "base_checkpoint_sha256": data["base_checkpoint_sha256"],
            },
            checkpoint_dir / f"{name}_head.pt",
        )
        results[name] = metrics
        print(
            f"transfer {name}: best_epoch={metrics['training']['best_epoch']}",
            flush=True,
        )
    if len(base_hashes) != 1:
        raise RuntimeError("transfer adapters use different base checkpoints")
    report = {
        "schema": "protein-stabilizer.directional-transfer-training.v1",
        "seed": seed,
        "elapsed_seconds": time.monotonic() - started,
        "base_checkpoint_sha256": next(iter(base_hashes)),
        "endpoint_separation": {
            "thermodynamic_ddg": ["protherm", "mcsm_membrane"],
            "assay_delta_tm": ["mptherm", "gpcr_tm"],
            "rank_only_not_physical_units": ["gpcr_rank"],
        },
        "adapters": results,
    }
    (checkpoint_dir / "hierarchy_transfer_metrics.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    return report
