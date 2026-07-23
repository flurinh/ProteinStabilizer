"""Permutation-invariant multi-mutation training over frozen v2 representations."""

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

from .embeddings import HierarchyEmbeddingReader, file_sha256
from .models import (
    HierarchicalEpistasisConfig,
    HierarchicalMultiMutationHead,
)
from .training import regression_metrics, set_reproducible_seed
from .v2_features import HIERARCHY_MULTI_ROW_SCHEMA
from .v2_training import load_hierarchical_ensemble


MULTI_REPRESENTATION_SCHEMA = (
    "protein-stabilizer.hierarchical-multi-representations.v1"
)
MULTI_CHECKPOINT_SCHEMA = "protein-stabilizer.hierarchical-multi-head.v1"


def _indexed_state(
    cache: HierarchyEmbeddingReader,
    site: np.ndarray,
    sequence: np.ndarray,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    features = cache.indexed_features(
        np.asarray(site, dtype=np.int64).reshape(-1),
        np.asarray(sequence, dtype=np.int64).reshape(-1),
    )
    return (
        torch.from_numpy(features["window"].astype(np.float32)).to(device),
        torch.from_numpy(features["window_mask"]).to(device),
        torch.from_numpy(features["global_mean"].astype(np.float32)).to(device),
    )


def build_multi_representations(
    feature_dir: Path,
    cache_path: Path,
    base_checkpoint: Path,
    output_dir: Path,
    *,
    batch_size: int = 256,
    device: str = "cuda",
) -> dict[str, object]:
    """Freeze base single latents/ddG for every WT/single/joint double row."""

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
    base, base_payload = load_hierarchical_ensemble(
        base_checkpoint, torch_device
    )
    latent_dim = int(base.config.latent_dim)
    cache_sha256 = file_sha256(cache_path)
    records: list[dict[str, object]] = []
    with HierarchyEmbeddingReader(cache_path) as cache:
        for split in ("train", "val", "test"):
            source = feature_dir / f"double_{split}.h5"
            with h5py.File(source, "r") as handle:
                if handle.attrs.get("schema") != HIERARCHY_MULTI_ROW_SCHEMA:
                    raise RuntimeError(f"{source} multi-row schema mismatch")
                if str(handle.attrs["hierarchy_cache_sha256"]) != cache_sha256:
                    raise RuntimeError(f"{source} hierarchy cache hash mismatch")
                arrays = {
                    name: np.asarray(handle[name], dtype=np.int64)
                    for name in (
                        "wt_site_index",
                        "wt_sequence_index",
                        "single_site_index",
                        "single_sequence_index",
                        "joint_site_index",
                        "joint_sequence_index",
                    )
                }
                target = np.asarray(handle["target"], dtype=np.float32)
                true_additive = np.asarray(
                    handle["true_additive"], dtype=np.float32
                )
                true_epistasis = np.asarray(
                    handle["true_epistasis"], dtype=np.float32
                )
                has_true_epistasis = np.asarray(
                    handle["has_true_epistasis"], dtype=bool
                )
                # Accuracy-first v2 artifacts retain the native FP32
                # ProteinMPNN representation.  Converting to FP16 here and
                # back to FP32 before the head irreversibly discarded bits.
                structure = np.asarray(handle["structure"], dtype=np.float32)
                structure_mask = np.asarray(
                    handle["structure_mask"], dtype=bool
                )
                membrane = np.asarray(handle["membrane"], dtype=np.float32)
                protein_id = np.asarray(handle["protein_id"].asstr()[:])
                mutation_1 = np.asarray(handle["mutation_1"].asstr()[:])
                mutation_2 = np.asarray(handle["mutation_2"].asstr()[:])
                source_sha256 = str(handle.attrs["source_sha256"])
            count, mutation_count = arrays["wt_site_index"].shape
            if mutation_count != 2:
                raise RuntimeError("v2 multi representation expects double mutants")
            output = output_dir / f"double_{split}.h5"
            partial = output.with_suffix(".h5.partial")
            partial.unlink(missing_ok=True)
            with h5py.File(partial, "w", libver="latest") as handle:
                handle.attrs["schema"] = MULTI_REPRESENTATION_SCHEMA
                handle.attrs["split"] = split
                handle.attrs["row_source"] = str(source)
                handle.attrs["row_source_sha256"] = file_sha256(source)
                handle.attrs["source_sha256"] = source_sha256
                handle.attrs["hierarchy_cache"] = str(cache_path)
                handle.attrs["hierarchy_cache_sha256"] = cache_sha256
                handle.attrs["base_checkpoint"] = str(base_checkpoint)
                handle.attrs["base_checkpoint_sha256"] = file_sha256(
                    base_checkpoint
                )
                handle.attrs["base_candidate"] = str(
                    base_payload["candidate"]
                )
                handle.attrs["base_use_structure"] = bool(
                    base_payload["use_structure"]
                )
                handle.attrs["base_metrics"] = json.dumps(
                    base_payload["metrics"], sort_keys=True, separators=(",", ":")
                )
                single_latent = handle.create_dataset(
                    "single_latent",
                    shape=(count, 2, latent_dim),
                    dtype="f4",
                    chunks=(min(1024, count), 2, latent_dim),
                )
                joint_latent = handle.create_dataset(
                    "joint_latent",
                    shape=(count, 2, latent_dim),
                    dtype="f4",
                    chunks=(min(1024, count), 2, latent_dim),
                )
                single_ddg = handle.create_dataset(
                    "single_ddg",
                    shape=(count, 2),
                    dtype="f4",
                    chunks=(min(4096, count), 2),
                )
                for start in range(0, count, batch_size):
                    stop = min(count, start + batch_size)
                    row_slice = slice(start, stop)
                    flat_count = (stop - start) * 2
                    wt_window, wt_mask, wt_global = _indexed_state(
                        cache,
                        arrays["wt_site_index"][row_slice],
                        arrays["wt_sequence_index"][row_slice],
                        torch_device,
                    )
                    single_window, single_mask, single_global = _indexed_state(
                        cache,
                        arrays["single_site_index"][row_slice],
                        arrays["single_sequence_index"][row_slice],
                        torch_device,
                    )
                    joint_window, joint_mask, joint_global = _indexed_state(
                        cache,
                        arrays["joint_site_index"][row_slice],
                        arrays["joint_sequence_index"][row_slice],
                        torch_device,
                    )
                    if (
                        not torch.equal(wt_mask, single_mask)
                        or not torch.equal(wt_mask, joint_mask)
                    ):
                        raise RuntimeError("multi-mutant hierarchy masks differ")
                    static_structure = torch.from_numpy(
                        structure[row_slice].reshape(flat_count, -1).astype(
                            np.float32
                        )
                    ).to(torch_device)
                    static_mask = torch.from_numpy(
                        structure_mask[row_slice].reshape(flat_count)
                    ).to(torch_device)
                    static_membrane = torch.from_numpy(
                        membrane[row_slice].reshape(flat_count, -1)
                    ).to(torch_device)
                    with torch.inference_mode():
                        single_z = base.latent(
                            wt_window,
                            single_window,
                            wt_mask,
                            wt_global,
                            single_global,
                            structure=static_structure,
                            structure_mask=static_mask,
                            membrane=static_membrane,
                        )
                        joint_z = base.latent(
                            wt_window,
                            joint_window,
                            wt_mask,
                            wt_global,
                            joint_global,
                            structure=static_structure,
                            structure_mask=static_mask,
                            membrane=static_membrane,
                        )
                        values = base(
                            wt_window,
                            single_window,
                            wt_mask,
                            wt_global,
                            single_global,
                            structure=static_structure,
                            structure_mask=static_mask,
                            membrane=static_membrane,
                        )
                    batch_shape = (stop - start, 2, latent_dim)
                    single_latent[row_slice] = (
                        single_z.float().cpu().numpy().reshape(batch_shape)
                    ).astype(np.float32)
                    joint_latent[row_slice] = (
                        joint_z.float().cpu().numpy().reshape(batch_shape)
                    ).astype(np.float32)
                    single_ddg[row_slice] = (
                        values.float().cpu().numpy().reshape(stop - start, 2)
                    )
                handle.create_dataset("target", data=target)
                handle.create_dataset("true_additive", data=true_additive)
                handle.create_dataset("true_epistasis", data=true_epistasis)
                handle.create_dataset(
                    "has_true_epistasis", data=has_true_epistasis
                )
                string_dtype = h5py.string_dtype(encoding="utf-8")
                handle.create_dataset(
                    "protein_id",
                    data=np.asarray(protein_id, dtype=object),
                    dtype=string_dtype,
                )
                handle.create_dataset(
                    "mutation_1",
                    data=np.asarray(mutation_1, dtype=object),
                    dtype=string_dtype,
                )
                handle.create_dataset(
                    "mutation_2",
                    data=np.asarray(mutation_2, dtype=object),
                    dtype=string_dtype,
                )
                handle.flush()
            partial.replace(output)
            records.append(
                {
                    "path": str(output),
                    "sha256": file_sha256(output),
                    "split": split,
                    "rows": count,
                    "proteins": len(set(protein_id.tolist())),
                    "latent_dimension": latent_dim,
                    "source_sha256": source_sha256,
                }
            )
            print(
                f"multi representations {split}: {count:,} rows",
                flush=True,
            )
    manifest = {
        "schema": MULTI_REPRESENTATION_SCHEMA,
        "base_checkpoint": {
            "path": str(base_checkpoint),
            "sha256": file_sha256(base_checkpoint),
            "candidate": base_payload["candidate"],
            "use_structure": base_payload["use_structure"],
        },
        "hierarchy_cache": {
            "path": str(cache_path),
            "sha256": cache_sha256,
            "provenance": cache.provenance,
        },
        "datasets": records,
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    manifest["manifest_path"] = str(manifest_path)
    return manifest


def _load_representation(path: Path) -> dict[str, np.ndarray | str]:
    with h5py.File(path, "r") as handle:
        if handle.attrs.get("schema") != MULTI_REPRESENTATION_SCHEMA:
            raise RuntimeError(f"{path} representation schema mismatch")
        values: dict[str, np.ndarray | str | float] = {
            "single_latent": np.asarray(handle["single_latent"], dtype=np.float32),
            "joint_latent": np.asarray(handle["joint_latent"], dtype=np.float32),
            "single_ddg": np.asarray(handle["single_ddg"], dtype=np.float32),
            "target": np.asarray(handle["target"], dtype=np.float32),
            "true_additive": np.asarray(
                handle["true_additive"], dtype=np.float32
            ),
            "true_epistasis": np.asarray(
                handle["true_epistasis"], dtype=np.float32
            ),
            "has_true_epistasis": np.asarray(
                handle["has_true_epistasis"], dtype=bool
            ),
            "protein_id": np.asarray(handle["protein_id"].asstr()[:]),
            "source_sha256": str(handle.attrs["source_sha256"]),
            "base_checkpoint_sha256": str(
                handle.attrs["base_checkpoint_sha256"]
            ),
            "base_candidate": str(
                handle.attrs.get("base_candidate", "hierarchy_proteinmpnn")
            ),
            "state_potential_checkpoint_sha256": str(
                handle.attrs.get(
                    "state_potential_checkpoint_sha256", ""
                )
            ),
            "state_potential_weight": float(
                handle.attrs.get("state_potential_weight", 0.0)
            ),
        }
        if "hierarchy_ddg" in handle and "state_potential_ddg" in handle:
            values["hierarchy_ddg"] = np.asarray(
                handle["hierarchy_ddg"], dtype=np.float32
            )
            values["state_potential_ddg"] = np.asarray(
                handle["state_potential_ddg"], dtype=np.float32
            )
        return values


def load_hierarchical_multi_checkpoint(
    path: Path,
    device: torch.device,
) -> tuple[HierarchicalMultiMutationHead, dict[str, object]]:
    """Load a trained v2 additive-plus-epistasis head."""

    payload = torch.load(Path(path), map_location=device, weights_only=False)
    if payload.get("schema") != MULTI_CHECKPOINT_SCHEMA:
        raise RuntimeError("hierarchical multi checkpoint schema mismatch")
    model = HierarchicalMultiMutationHead(
        HierarchicalEpistasisConfig(**payload["config"])
    )
    model.load_state_dict(payload["state_dict"])
    return model.to(device).eval(), payload


def _multi_predictions(
    model: HierarchicalMultiMutationHead,
    data: dict[str, np.ndarray | str],
    *,
    batch_size: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    output: list[np.ndarray] = []
    additive: list[np.ndarray] = []
    epistasis: list[np.ndarray] = []
    single_ddg = np.asarray(data["single_ddg"])
    single_latent = np.asarray(data["single_latent"])
    joint_latent = np.asarray(data["joint_latent"])
    with torch.inference_mode():
        for start in range(0, len(single_ddg), batch_size):
            stop = start + batch_size
            values = model(
                torch.from_numpy(single_ddg[start:stop].astype(np.float32)).to(
                    device
                ),
                torch.from_numpy(
                    single_latent[start:stop].astype(np.float32)
                ).to(device),
                torch.from_numpy(
                    joint_latent[start:stop].astype(np.float32)
                ).to(device),
            )
            output.append(values[0].float().cpu().numpy())
            additive.append(values[1].float().cpu().numpy())
            epistasis.append(values[2].float().cpu().numpy())
    return tuple(
        np.concatenate(values) for values in (output, additive, epistasis)
    )


def train_hierarchical_multi_head(
    representation_dir: Path,
    checkpoint_dir: Path,
    *,
    seed: int = 20260715,
    epochs: int = 70,
    minimum_epochs: int = 50,
    patience: int = 8,
    batch_size: int = 256,
    learning_rate: float = 7.5e-4,
    weight_decay: float = 1e-4,
    device: str = "cuda",
    constituent_state_weight: float | None = None,
) -> dict[str, object]:
    """Train the additive-plus-epistasis head on protein-disjoint splits."""

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
    data = {
        split: _load_representation(
            representation_dir / f"double_{split}.h5"
        )
        for split in ("train", "val", "test")
    }
    base_hashes = {
        str(value["base_checkpoint_sha256"]) for value in data.values()
    }
    if len(base_hashes) != 1:
        raise RuntimeError("multi representations use different base checkpoints")
    base_candidates = {
        str(value["base_candidate"]) for value in data.values()
    }
    if len(base_candidates) != 1:
        raise RuntimeError("multi representations use different base candidates")
    state_hashes = {
        str(value["state_potential_checkpoint_sha256"])
        for value in data.values()
    }
    if len(state_hashes) != 1:
        raise RuntimeError(
            "multi representations use different state-potential checkpoints"
        )
    state_weights = {
        float(value["state_potential_weight"])
        for value in data.values()
    }
    if len(state_weights) != 1:
        raise RuntimeError(
            "multi representations use different state-potential weights"
        )
    if constituent_state_weight is not None:
        if not 0.0 <= constituent_state_weight <= 1.0:
            raise ValueError("constituent state weight must be between 0 and 1")
        for split, values in data.items():
            if (
                "hierarchy_ddg" not in values
                or "state_potential_ddg" not in values
            ):
                raise RuntimeError(
                    f"{split} representation lacks constituent ddG components"
                )
            values["single_ddg"] = (
                (1.0 - constituent_state_weight)
                * np.asarray(values["hierarchy_ddg"])
                + constituent_state_weight
                * np.asarray(values["state_potential_ddg"])
            ).astype(np.float32)
        state_weights = {float(constituent_state_weight)}
    protein_sets = {
        split: set(np.asarray(value["protein_id"]).tolist())
        for split, value in data.items()
    }
    if any(
        protein_sets[left] & protein_sets[right]
        for left, right in (("train", "val"), ("train", "test"), ("val", "test"))
    ):
        raise RuntimeError("multi-mutant proteins overlap across splits")
    latent_dim = np.asarray(data["train"]["single_latent"]).shape[-1]
    config = HierarchicalEpistasisConfig(latent_dim=latent_dim)
    set_reproducible_seed(seed)
    model = HierarchicalMultiMutationHead(config).to(torch_device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=learning_rate * 0.05
    )
    huber = torch.nn.HuberLoss(delta=1.0)
    train = data["train"]
    count = len(np.asarray(train["target"]))
    rng = np.random.default_rng(seed)
    best_state: dict[str, torch.Tensor] | None = None
    best_score = -math.inf
    best_epoch = 0
    stale = 0
    history: list[dict[str, object]] = []
    optimizer_steps = 0
    examples_seen = 0
    started = time.monotonic()
    for epoch in range(1, epochs + 1):
        model.train()
        shuffled = rng.permutation(count)
        total = 0.0
        seen = 0
        for start in range(0, count, batch_size):
            indices = shuffled[start : start + batch_size]
            single_ddg = torch.from_numpy(
                np.asarray(train["single_ddg"])[indices].astype(np.float32)
            ).to(torch_device)
            single_latent = torch.from_numpy(
                np.asarray(train["single_latent"])[indices].astype(np.float32)
            ).to(torch_device)
            joint_latent = torch.from_numpy(
                np.asarray(train["joint_latent"])[indices].astype(np.float32)
            ).to(torch_device)
            target = torch.from_numpy(
                np.asarray(train["target"])[indices].astype(np.float32)
            ).to(torch_device)
            true_epistasis = torch.from_numpy(
                np.asarray(train["true_epistasis"])[indices].astype(np.float32)
            ).to(torch_device)
            has_epistasis = torch.from_numpy(
                np.asarray(train["has_true_epistasis"])[indices]
            ).to(torch_device)
            optimizer.zero_grad(set_to_none=True)
            prediction, _, epistasis = model(
                single_ddg, single_latent, joint_latent
            )
            total_loss = huber(prediction, target)
            epistasis_loss = (
                huber(epistasis[has_epistasis], true_epistasis[has_epistasis])
                if torch.any(has_epistasis)
                else torch.zeros((), device=torch_device)
            )
            loss = epistasis_loss + 0.25 * total_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            optimizer_steps += 1
            examples_seen += len(indices)
            total += float(loss.detach()) * len(indices)
            seen += len(indices)
        scheduler.step()
        validation = _multi_predictions(
            model, data["val"], batch_size=batch_size, device=torch_device
        )
        validation_metrics = regression_metrics(
            np.asarray(data["val"]["target"]), validation[0]
        )
        epi_mask = np.asarray(data["val"]["has_true_epistasis"])
        epistasis_metrics = regression_metrics(
            np.asarray(data["val"]["true_epistasis"])[epi_mask],
            validation[2][epi_mask],
        )
        score = (
            0.60 * float(validation_metrics["spearman"])
            + 0.20 * float(epistasis_metrics["spearman"])
            - 0.20 * float(validation_metrics["mae"])
        )
        history.append(
            {
                "epoch": epoch,
                "train_loss": total / seen,
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
                "optimizer_steps": optimizer_steps,
                "examples_seen": examples_seen,
                "validation": validation_metrics,
                "validation_epistasis": epistasis_metrics,
                "selection_score": score,
            }
        )
        print(
            f"multi epoch={epoch:02d} train={total / seen:.4f} "
            f"val_rho={validation_metrics['spearman']:.4f} "
            f"val_mae={validation_metrics['mae']:.4f} "
            f"epi_rho={epistasis_metrics['spearman']:.4f}",
            flush=True,
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
        raise RuntimeError("hierarchical multi training produced no valid model")
    model.load_state_dict(best_state)
    evaluations: dict[str, object] = {}
    for split in ("val", "test"):
        prediction, additive, epistasis = _multi_predictions(
            model, data[split], batch_size=batch_size, device=torch_device
        )
        epi_mask = np.asarray(data[split]["has_true_epistasis"])
        evaluations[split] = {
            "total": regression_metrics(
                np.asarray(data[split]["target"]), prediction
            ),
            "additive": regression_metrics(
                np.asarray(data[split]["target"]), additive
            ),
            "epistasis": regression_metrics(
                np.asarray(data[split]["true_epistasis"])[epi_mask],
                epistasis[epi_mask],
            ),
        }
    permutation_count = min(512, len(np.asarray(data["test"]["target"])))
    singles = torch.from_numpy(
        np.asarray(data["test"]["single_ddg"])[:permutation_count].astype(
            np.float32
        )
    ).to(torch_device)
    single_z = torch.from_numpy(
        np.asarray(data["test"]["single_latent"])[:permutation_count].astype(
            np.float32
        )
    ).to(torch_device)
    joint_z = torch.from_numpy(
        np.asarray(data["test"]["joint_latent"])[:permutation_count].astype(
            np.float32
        )
    ).to(torch_device)
    with torch.inference_mode():
        forward = model(singles, single_z, joint_z)
        reverse_order = torch.tensor([1, 0], device=torch_device)
        permuted = model(
            singles[:, reverse_order],
            single_z[:, reverse_order],
            joint_z[:, reverse_order],
        )
    permutation_error = max(
        float(torch.max(torch.abs(left - right)).cpu())
        for left, right in zip(forward, permuted, strict=True)
    )
    metrics = {
        "schema": "protein-stabilizer.hierarchical-multi-training.v1",
        "seed": seed,
        "elapsed_seconds": time.monotonic() - started,
        "training": {
            "epoch_budget": epochs,
            "minimum_epochs": minimum_epochs,
            "best_epoch": best_epoch,
            "optimizer_steps": optimizer_steps,
            "examples_seen": examples_seen,
            "history": history,
        },
        "evaluation": evaluations,
        "split_integrity": {
            split: len(values) for split, values in protein_sets.items()
        },
        "components": (
            "total = frozen validation-selected hierarchy/state-potential "
            "constituent single ddG sum + learned permutation-invariant "
            "epistasis"
            if next(iter(base_candidates))
            == "hierarchy_state_potential_blend"
            else (
                "total = frozen selected-v2 constituent single ddG sum + "
                "learned permutation-invariant epistasis"
            )
        ),
        "permutation_invariance": {
            "evaluated_rows": permutation_count,
            "maximum_absolute_error": permutation_error,
        },
        "base_checkpoint_sha256": next(iter(base_hashes)),
        "base_candidate": next(iter(base_candidates)),
        "state_potential_checkpoint_sha256": next(iter(state_hashes)),
        "state_potential_weight": next(iter(state_weights)),
        "source_sha256": {
            split: str(value["source_sha256"])
            for split, value in data.items()
        },
    }
    torch.save(
        {
            "schema": MULTI_CHECKPOINT_SCHEMA,
            "config": asdict(config),
            "state_dict": copy.deepcopy(model.cpu().state_dict()),
            "metrics": metrics,
            "base_checkpoint_sha256": next(iter(base_hashes)),
            "base_candidate": next(iter(base_candidates)),
            "state_potential_checkpoint_sha256": next(iter(state_hashes)),
            "state_potential_weight": next(iter(state_weights)),
        },
        checkpoint_dir / "hierarchy_multi_head.pt",
    )
    (checkpoint_dir / "hierarchy_multi_metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8"
    )
    return metrics


def promote_hierarchical_multi_candidate(
    candidate_checkpoint_dir: Path,
    baseline_checkpoint_dir: Path,
) -> dict[str, object]:
    """Apply validation-only gates before replacing the multi-mutant head."""

    candidate_checkpoint_dir = Path(candidate_checkpoint_dir).resolve()
    baseline_checkpoint_dir = Path(baseline_checkpoint_dir).resolve()
    candidate_metrics_path = (
        candidate_checkpoint_dir / "hierarchy_multi_metrics.json"
    )
    baseline_metrics_path = (
        baseline_checkpoint_dir / "hierarchy_multi_metrics.json"
    )
    candidate_checkpoint = (
        candidate_checkpoint_dir / "hierarchy_multi_head.pt"
    )
    baseline_checkpoint = (
        baseline_checkpoint_dir / "hierarchy_multi_head.pt"
    )
    candidate = json.loads(
        candidate_metrics_path.read_text(encoding="utf-8")
    )
    baseline = json.loads(
        baseline_metrics_path.read_text(encoding="utf-8")
    )
    source_match = (
        candidate.get("source_sha256") == baseline.get("source_sha256")
        and candidate.get("split_integrity")
        == baseline.get("split_integrity")
    )
    candidate_validation = candidate["evaluation"]["validation"]
    baseline_validation = baseline["evaluation"]["validation"]
    gates = {
        "identical_sources_and_splits": source_match,
        "validation_total_spearman_improves": (
            float(candidate_validation["total"]["spearman"])
            > float(baseline_validation["total"]["spearman"])
        ),
        "validation_total_mae_improves": (
            float(candidate_validation["total"]["mae"])
            < float(baseline_validation["total"]["mae"])
        ),
        "validation_total_rmse_improves": (
            float(candidate_validation["total"]["rmse"])
            < float(baseline_validation["total"]["rmse"])
        ),
        "validation_epistasis_spearman_not_worse": (
            float(candidate_validation["epistasis"]["spearman"])
            >= float(baseline_validation["epistasis"]["spearman"])
        ),
        "exact_permutation_invariance": (
            float(
                candidate["permutation_invariance"][
                    "maximum_absolute_error"
                ]
            )
            <= 1e-7
        ),
    }
    production_eligible = all(gates.values())
    promotion = {
        "production_eligible": production_eligible,
        "decision": (
            "promote fused multi-mutant head"
            if production_eligible
            else "reject fused multi-mutant head"
        ),
        "gates": gates,
        "baseline": {
            "metrics": str(baseline_metrics_path),
            "metrics_sha256": file_sha256(baseline_metrics_path),
            "checkpoint": str(baseline_checkpoint),
            "checkpoint_sha256": file_sha256(baseline_checkpoint),
            "validation": baseline_validation,
        },
        "candidate_validation": candidate_validation,
        "historical_test": candidate["evaluation"]["test"],
        "policy": (
            "validation metrics and exact permutation invariance only; "
            "historical test is reporting-only"
        ),
    }
    candidate["promotion"] = promotion
    payload = torch.load(
        candidate_checkpoint, map_location="cpu", weights_only=False
    )
    if payload.get("schema") != MULTI_CHECKPOINT_SCHEMA:
        raise RuntimeError("candidate multi checkpoint schema mismatch")
    payload["promotion"] = promotion
    payload["metrics"] = candidate
    torch.save(payload, candidate_checkpoint)
    candidate["artifacts"] = {
        "checkpoint": str(candidate_checkpoint),
        "checkpoint_sha256": file_sha256(candidate_checkpoint),
        "baseline_checkpoint": str(baseline_checkpoint),
        "baseline_checkpoint_sha256": file_sha256(baseline_checkpoint),
    }
    candidate_metrics_path.write_text(
        json.dumps(candidate, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return candidate
