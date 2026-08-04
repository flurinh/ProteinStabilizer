"""Controlled training and ablation for the hierarchical 600M v2 candidate."""

from __future__ import annotations

import copy
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import h5py
import numpy as np
import torch

from .embeddings import HierarchyEmbeddingReader, file_sha256
from .models import (
    HierarchicalDirectionalEnsemble,
    HierarchicalDirectionalMutationHead,
    HierarchicalHeadConfig,
)
from .training import (
    balanced_stability_weights,
    load_scoring_checkpoint,
    regression_metrics,
    set_reproducible_seed,
    stabilizer_retrieval_metrics,
)
from .v2_features import HIERARCHY_ROW_SCHEMA


HIERARCHY_CHECKPOINT_SCHEMA = (
    "protein-stabilizer.hierarchical-directional-ensemble.v1"
)


@dataclass(frozen=True)
class HierarchicalObjectiveConfig:
    stabilizer_threshold: float = -0.5
    neutral_upper_threshold: float = 0.5
    regression_weight: float = 1.0
    retrieval_weight: float = 0.20
    ranking_weight: float = 0.10
    ranking_pairs_per_batch: int = 192
    minimum_ranking_gap: float = 0.25
    maximum_bin_weight: float = 6.0
    huber_delta: float = 1.0
    mse_weight: float = 0.0
    retrieval_source: str = "head"
    selection_metric: str = "composite"
    absolute_stability_weight: float = 0.0


@dataclass
class HierarchyArrays:
    wt_window: np.ndarray
    mutant_window: np.ndarray
    window_mask: np.ndarray
    wt_global: np.ndarray
    mutant_global: np.ndarray
    structure: np.ndarray
    structure_mask: np.ndarray
    membrane: np.ndarray
    target: np.ndarray
    sample_weight: np.ndarray
    protein_id: np.ndarray
    split: np.ndarray
    provenance: dict[str, object]

    def subset(self, indices: np.ndarray) -> "HierarchyArrays":
        return HierarchyArrays(
            wt_window=self.wt_window[indices],
            mutant_window=self.mutant_window[indices],
            window_mask=self.window_mask[indices],
            wt_global=self.wt_global[indices],
            mutant_global=self.mutant_global[indices],
            structure=self.structure[indices],
            structure_mask=self.structure_mask[indices],
            membrane=self.membrane[indices],
            target=self.target[indices],
            sample_weight=self.sample_weight[indices],
            protein_id=self.protein_id[indices],
            split=self.split[indices],
            provenance=self.provenance,
        )


def _load_rows(
    row_path: Path,
    cache_path: Path,
    *,
    expected_cache_sha256: str,
) -> HierarchyArrays:
    with h5py.File(row_path, "r") as handle:
        if handle.attrs.get("schema") != HIERARCHY_ROW_SCHEMA:
            raise RuntimeError(f"{row_path} hierarchy row schema mismatch")
        if str(handle.attrs["hierarchy_cache_sha256"]) != expected_cache_sha256:
            raise RuntimeError(f"{row_path} hierarchy cache hash mismatch")
        if Path(str(handle.attrs["hierarchy_cache"])).resolve() != cache_path.resolve():
            raise RuntimeError(f"{row_path} points to a different hierarchy cache")
        provenance = {
            "row_path": str(row_path),
            "row_sha256": file_sha256(row_path),
            "source": str(handle.attrs["source"]),
            "source_sha256": str(handle.attrs["source_sha256"]),
            "hierarchy_cache": str(cache_path),
            "hierarchy_cache_sha256": expected_cache_sha256,
            "hierarchy_provenance": json.loads(
                str(handle.attrs["hierarchy_provenance"])
            ),
            "window_radius": int(handle.attrs["window_radius"]),
            "structure_provenance": json.loads(
                str(handle.attrs["structure_provenance"])
            ),
        }
        wt_site = np.asarray(handle["wt_site_index"], dtype=np.int64)
        wt_sequence = np.asarray(handle["wt_sequence_index"], dtype=np.int64)
        mutant_site = np.asarray(handle["mutant_site_index"], dtype=np.int64)
        mutant_sequence = np.asarray(
            handle["mutant_sequence_index"], dtype=np.int64
        )
        structure = np.asarray(handle["structure"])
        structure_mask = np.asarray(handle["structure_mask"], dtype=bool)
        membrane = np.asarray(handle["membrane"], dtype=np.float32)
        target = np.asarray(handle["target"], dtype=np.float32)
        sample_weight = np.asarray(handle["sample_weight"], dtype=np.float32)
        protein_id = np.asarray(handle["protein_id"].asstr()[:])
        split = np.asarray(handle["split"].asstr()[:])
    with HierarchyEmbeddingReader(cache_path) as cache:
        wt = cache.indexed_features(wt_site, wt_sequence)
        if np.array_equal(wt_site, mutant_site) and np.array_equal(
            wt_sequence, mutant_sequence
        ):
            # WT-conditioned state-potential pretraining does not consume a
            # separately embedded mutant state. Reuse the same arrays instead
            # of materializing a second multi-gigabyte copy.
            mutant = wt
        else:
            mutant = cache.indexed_features(mutant_site, mutant_sequence)
    if not np.array_equal(wt["window_mask"], mutant["window_mask"]):
        raise RuntimeError("WT and mutant hierarchy window masks differ")
    return HierarchyArrays(
        wt_window=wt["window"],
        mutant_window=mutant["window"],
        window_mask=wt["window_mask"],
        wt_global=wt["global_mean"],
        mutant_global=mutant["global_mean"],
        structure=structure,
        structure_mask=structure_mask,
        membrane=membrane,
        target=target,
        sample_weight=sample_weight,
        protein_id=protein_id,
        split=split,
        provenance=provenance,
    )


def _batch_tensors(
    data: HierarchyArrays,
    indices: np.ndarray,
    device: torch.device,
    *,
    use_structure: bool,
) -> dict[str, torch.Tensor]:
    def floating(array: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(array[indices].astype(np.float32)).to(device)

    structure = floating(data.structure)
    structure_mask = torch.from_numpy(data.structure_mask[indices]).to(device)
    if not use_structure:
        structure.zero_()
        structure_mask.zero_()
    return {
        "wt_window": floating(data.wt_window),
        "mutant_window": floating(data.mutant_window),
        "window_mask": torch.from_numpy(data.window_mask[indices]).to(device),
        "wt_global": floating(data.wt_global),
        "mutant_global": floating(data.mutant_global),
        "structure": structure,
        "structure_mask": structure_mask,
        "membrane": floating(data.membrane),
    }


def _predict(
    model: HierarchicalDirectionalMutationHead | HierarchicalDirectionalEnsemble,
    data: HierarchyArrays,
    indices: np.ndarray,
    *,
    use_structure: bool,
    batch_size: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    ddg: list[np.ndarray] = []
    retrieval: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(indices), batch_size):
            selected = indices[start : start + batch_size]
            outputs = model.predict_heads(
                **_batch_tensors(
                    data,
                    selected,
                    device,
                    use_structure=use_structure,
                )
            )
            ddg.append(outputs["ddg"].float().cpu().numpy())
            retrieval.append(outputs["retrieval"].float().cpu().numpy())
    return (
        np.concatenate(ddg) if ddg else np.empty(0, dtype=np.float32),
        (
            np.concatenate(retrieval)
            if retrieval
            else np.empty(0, dtype=np.float32)
        ),
    )


def _evaluation(
    target: np.ndarray,
    ddg: np.ndarray,
    retrieval: np.ndarray,
    *,
    threshold: float,
) -> dict[str, object]:
    return {
        "regression": regression_metrics(target, ddg),
        "retrieval_from_ddg": stabilizer_retrieval_metrics(
            target, -ddg, threshold=threshold
        ),
        "retrieval_head": stabilizer_retrieval_metrics(
            target, retrieval, threshold=threshold
        ),
    }


def _selection_score(
    metrics: dict[str, object],
    *,
    retrieval_key: str = "retrieval_head",
    selection_metric: str = "composite",
) -> float:
    regression = metrics["regression"]
    if retrieval_key not in {"retrieval_head", "retrieval_from_ddg"}:
        raise ValueError("unknown retrieval metric for model selection")
    retrieval = metrics[retrieval_key]
    spearman = float(regression["spearman"])
    average_precision = float(retrieval["average_precision"])
    mae = float(regression["mae"])
    if not all(np.isfinite(value) for value in (spearman, average_precision, mae)):
        return -math.inf
    if selection_metric == "mae":
        return -mae
    if selection_metric != "composite":
        raise ValueError("selection metric must be 'composite' or 'mae'")
    return 0.65 * spearman + 0.25 * average_precision - 0.10 * mae


def _ranking_pairs(
    protein_id: np.ndarray,
    target: np.ndarray,
    rng: np.random.Generator,
    *,
    count: int,
    minimum_gap: float,
) -> tuple[np.ndarray, np.ndarray]:
    groups = [
        np.flatnonzero(protein_id == protein)
        for protein in sorted(set(protein_id.tolist()))
    ]
    groups = [
        group
        for group in groups
        if len(group) > 1 and float(np.ptp(target[group])) >= minimum_gap
    ]
    stable: list[int] = []
    unstable: list[int] = []
    attempts = 0
    while groups and len(stable) < count and attempts < 20 * max(1, count):
        group = groups[int(rng.integers(len(groups)))]
        left, right = rng.choice(group, size=2, replace=False)
        attempts += 1
        difference = float(target[left] - target[right])
        if abs(difference) < minimum_gap:
            continue
        stable.append(int(left if difference < 0 else right))
        unstable.append(int(right if difference < 0 else left))
    return np.asarray(stable, dtype=np.int64), np.asarray(
        unstable, dtype=np.int64
    )


def _train_candidate(
    name: str,
    train_data: HierarchyArrays,
    test_data: HierarchyArrays,
    *,
    use_structure: bool,
    seed: int,
    epochs: int,
    minimum_epochs: int,
    patience: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    ensemble_size: int,
    device: torch.device,
    objective: HierarchicalObjectiveConfig,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    if minimum_epochs < 1 or minimum_epochs > epochs:
        raise ValueError("minimum epochs must be within the training budget")
    if objective.huber_delta <= 0.0:
        raise ValueError("Huber delta must be positive")
    if objective.mse_weight < 0.0:
        raise ValueError("MSE weight must be non-negative")
    if objective.retrieval_source not in {"head", "ddg"}:
        raise ValueError("retrieval source must be 'head' or 'ddg'")
    if objective.selection_metric not in {"composite", "mae"}:
        raise ValueError("selection metric must be 'composite' or 'mae'")
    retrieval_key = (
        "retrieval_from_ddg"
        if objective.retrieval_source == "ddg"
        else "retrieval_head"
    )
    train_indices = np.flatnonzero(train_data.split == "train")
    validation_indices = np.flatnonzero(train_data.split == "val")
    test_indices = np.arange(len(test_data.target), dtype=np.int64)
    config = HierarchicalHeadConfig(
        embedding_dim=train_data.wt_window.shape[-1],
        window_size=train_data.wt_window.shape[1],
        structure_dim=train_data.structure.shape[1],
        membrane_dim=train_data.membrane.shape[1],
    )
    weights = np.ones(len(train_data.target), dtype=np.float32)
    weights[train_indices] = balanced_stability_weights(
        train_data.target[train_indices],
        stabilizer_threshold=objective.stabilizer_threshold,
        neutral_upper_threshold=objective.neutral_upper_threshold,
        maximum_weight=objective.maximum_bin_weight,
    )
    weights *= train_data.sample_weight
    huber = torch.nn.HuberLoss(
        delta=objective.huber_delta,
        reduction="none",
    )
    binary = torch.nn.BCEWithLogitsLoss(reduction="none")
    members: list[dict[str, object]] = []
    validation_ddg: list[np.ndarray] = []
    validation_retrieval: list[np.ndarray] = []
    test_ddg: list[np.ndarray] = []
    test_retrieval: list[np.ndarray] = []
    started = time.monotonic()
    for member_index in range(ensemble_size):
        member_seed = seed + member_index
        set_reproducible_seed(member_seed)
        model = HierarchicalDirectionalMutationHead(config).to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[
                torch.optim.lr_scheduler.LinearLR(
                    optimizer,
                    start_factor=0.20,
                    end_factor=1.0,
                    total_iters=max(1, min(5, epochs // 10)),
                ),
                torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer,
                    T_max=max(1, epochs - max(1, min(5, epochs // 10))),
                    eta_min=learning_rate * 0.05,
                ),
            ],
            milestones=[max(1, min(5, epochs // 10))],
        )
        rng = np.random.default_rng(member_seed)
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
            totals = {
                "regression": 0.0,
                "mse": 0.0,
                "retrieval": 0.0,
                "ranking": 0.0,
                "total": 0.0,
            }
            seen = 0
            for start in range(0, len(shuffled), batch_size):
                indices = shuffled[start : start + batch_size]
                tensors = _batch_tensors(
                    train_data,
                    indices,
                    device,
                    use_structure=use_structure,
                )
                target = torch.from_numpy(train_data.target[indices]).to(device)
                weight = torch.from_numpy(weights[indices]).to(device)
                label = (target < objective.stabilizer_threshold).float()
                optimizer.zero_grad(set_to_none=True)
                outputs = model.predict_heads(**tensors)
                retrieval_logit = (
                    -(outputs["ddg"] - objective.stabilizer_threshold)
                    if objective.retrieval_source == "ddg"
                    else outputs["retrieval"]
                )
                regression_loss = (
                    huber(outputs["ddg"], target) * weight
                ).sum() / weight.sum()
                mse_loss = (
                    (outputs["ddg"] - target).square() * weight
                ).sum() / weight.sum()
                retrieval_loss = (
                    binary(retrieval_logit, label) * weight
                ).sum() / weight.sum()
                stable, unstable = _ranking_pairs(
                    train_data.protein_id[indices],
                    train_data.target[indices],
                    rng,
                    count=min(objective.ranking_pairs_per_batch, len(indices)),
                    minimum_gap=objective.minimum_ranking_gap,
                )
                if len(stable):
                    stable_tensor = torch.from_numpy(stable).to(device)
                    unstable_tensor = torch.from_numpy(unstable).to(device)
                    ranking_loss = torch.nn.functional.softplus(
                        -(
                            retrieval_logit[stable_tensor]
                            - retrieval_logit[unstable_tensor]
                        )
                    ).mean()
                else:
                    ranking_loss = torch.zeros((), device=device)
                loss = (
                    objective.regression_weight * regression_loss
                    + objective.mse_weight * mse_loss
                    + objective.retrieval_weight * retrieval_loss
                    + objective.ranking_weight * ranking_loss
                )
                loss.backward()
                gradient_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=5.0
                )
                optimizer.step()
                optimizer_steps += 1
                examples_seen += len(indices)
                count = len(indices)
                totals["regression"] += float(regression_loss.detach()) * count
                totals["mse"] += float(mse_loss.detach()) * count
                totals["retrieval"] += float(retrieval_loss.detach()) * count
                totals["ranking"] += float(ranking_loss.detach()) * count
                totals["total"] += float(loss.detach()) * count
                seen += count
            scheduler.step()
            validation_prediction, validation_score = _predict(
                model,
                train_data,
                validation_indices,
                use_structure=use_structure,
                batch_size=batch_size,
                device=device,
            )
            validation = _evaluation(
                train_data.target[validation_indices],
                validation_prediction,
                validation_score,
                threshold=objective.stabilizer_threshold,
            )
            score = _selection_score(
                validation,
                retrieval_key=retrieval_key,
                selection_metric=objective.selection_metric,
            )
            losses = {key: value / seen for key, value in totals.items()}
            history.append(
                {
                    "epoch": epoch,
                    "train_loss": losses,
                    "learning_rate": float(optimizer.param_groups[0]["lr"]),
                    "gradient_norm": float(gradient_norm),
                    "optimizer_steps": optimizer_steps,
                    "examples_seen": examples_seen,
                    "validation": validation,
                    "selection_score": score,
                }
            )
            print(
                f"{name} member={member_index + 1}/{ensemble_size} "
                f"epoch={epoch:02d} train={losses['total']:.4f} "
                f"val_rho={validation['regression']['spearman']:.4f} "
                f"val_mae={validation['regression']['mae']:.4f} "
                f"val_ap={validation[retrieval_key]['average_precision']:.4f}",
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
            raise RuntimeError(f"{name} member {member_index} has no valid model")
        model.load_state_dict(best_state)
        val_ddg, val_retrieval = _predict(
            model,
            train_data,
            validation_indices,
            use_structure=use_structure,
            batch_size=batch_size,
            device=device,
        )
        heldout_ddg, heldout_retrieval = _predict(
            model,
            test_data,
            test_indices,
            use_structure=use_structure,
            batch_size=batch_size,
            device=device,
        )
        validation_ddg.append(val_ddg)
        validation_retrieval.append(val_retrieval)
        test_ddg.append(heldout_ddg)
        test_retrieval.append(heldout_retrieval)
        members.append(
            {
                "config": asdict(config),
                "state_dict": copy.deepcopy(model.cpu().state_dict()),
                "seed": member_seed,
                "best_epoch": best_epoch,
                "optimizer_steps": optimizer_steps,
                "examples_seen": examples_seen,
                "history": history,
            }
        )
    val_ddg = np.mean(validation_ddg, axis=0)
    val_retrieval = np.mean(validation_retrieval, axis=0)
    heldout_ddg = np.mean(test_ddg, axis=0)
    heldout_retrieval = np.mean(test_retrieval, axis=0)
    model_members: list[HierarchicalDirectionalMutationHead] = []
    for payload in members:
        member = HierarchicalDirectionalMutationHead(config)
        member.load_state_dict(payload["state_dict"])
        model_members.append(member)
    ensemble = HierarchicalDirectionalEnsemble(model_members).to(device).eval()
    check_count = min(512, len(test_data.target))
    check = np.arange(check_count, dtype=np.int64)
    tensors = _batch_tensors(
        test_data,
        check,
        device,
        use_structure=use_structure,
    )
    with torch.inference_mode():
        forward = ensemble(**tensors)
        reverse = ensemble(
            tensors["mutant_window"],
            tensors["wt_window"],
            tensors["window_mask"],
            tensors["mutant_global"],
            tensors["wt_global"],
            structure=tensors["structure"],
            structure_mask=tensors["structure_mask"],
            membrane=tensors["membrane"],
        )
        self_prediction = ensemble(
            tensors["wt_window"],
            tensors["wt_window"],
            tensors["window_mask"],
            tensors["wt_global"],
            tensors["wt_global"],
            structure=tensors["structure"],
            structure_mask=tensors["structure_mask"],
            membrane=tensors["membrane"],
        )
    metrics = {
        "candidate": name,
        "use_structure": use_structure,
        "seed": seed,
        "ensemble_size": ensemble_size,
        "minimum_epochs": minimum_epochs,
        "elapsed_seconds": time.monotonic() - started,
        "objective": asdict(objective),
        "selection_rule": (
            "protein-disjoint validation MAE"
            if objective.selection_metric == "mae"
            else (
                "protein-disjoint validation composite = 0.65 Spearman + "
                "0.25 stabilizer AP - 0.10 MAE; "
                f"retrieval metric = {retrieval_key}"
            )
        ),
        "train_rows": int(len(train_indices)),
        "validation_rows": int(len(validation_indices)),
        "test_rows": int(len(test_indices)),
        "validation": _evaluation(
            train_data.target[validation_indices],
            val_ddg,
            val_retrieval,
            threshold=objective.stabilizer_threshold,
        ),
        "validation_selection_score": _selection_score(
            _evaluation(
                train_data.target[validation_indices],
                val_ddg,
                val_retrieval,
                threshold=objective.stabilizer_threshold,
            ),
            retrieval_key=retrieval_key,
            selection_metric=objective.selection_metric,
        ),
        "test": _evaluation(
            test_data.target,
            heldout_ddg,
            heldout_retrieval,
            threshold=objective.stabilizer_threshold,
        ),
        "directional_constraints": {
            "evaluated_rows": check_count,
            "maximum_absolute_forward_plus_reverse": float(
                torch.max(torch.abs(forward + reverse)).cpu()
            ),
            "maximum_absolute_self_ddg": float(
                torch.max(torch.abs(self_prediction)).cpu()
            ),
        },
        "members": [
            {
                "seed": payload["seed"],
                "best_epoch": payload["best_epoch"],
                "optimizer_steps": payload["optimizer_steps"],
                "examples_seen": payload["examples_seen"],
                "history": payload["history"],
            }
            for payload in members
        ],
    }
    return metrics, members


def _baseline_evaluation(
    baseline_checkpoint_dir: Path,
    data: HierarchyArrays,
    indices: np.ndarray,
    *,
    threshold: float,
    batch_size: int,
    device: torch.device,
) -> dict[str, object]:
    model = load_scoring_checkpoint(baseline_checkpoint_dir, device).eval()
    center = data.wt_window.shape[1] // 2
    delta = (
        data.mutant_window[indices, center].astype(np.float32)
        - data.wt_window[indices, center].astype(np.float32)
    )
    prediction: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(delta), batch_size):
            tensor = torch.from_numpy(delta[start : start + batch_size]).to(device)
            prediction.append(model(tensor).float().cpu().numpy())
    ddg = np.concatenate(prediction)
    return _evaluation(
        data.target[indices],
        ddg,
        -ddg,
        threshold=threshold,
    )


def train_hierarchical_single_ablation(
    feature_dir: Path,
    cache_path: Path,
    baseline_checkpoint_dir: Path,
    checkpoint_dir: Path,
    *,
    seed: int = 20260715,
    discovery_epochs: int = 20,
    discovery_patience: int = 4,
    main_epochs: int = 70,
    minimum_main_epochs: int = 50,
    main_patience: int = 8,
    batch_size: int = 256,
    learning_rate: float = 7.5e-4,
    weight_decay: float = 1e-4,
    ensemble_size: int = 3,
    device: str = "cuda",
    objective: HierarchicalObjectiveConfig = HierarchicalObjectiveConfig(),
) -> dict[str, object]:
    """Train the two bounded 600M hierarchy candidates and freeze selection."""

    if ensemble_size < 1:
        raise ValueError("hierarchical ensemble size must be positive")
    feature_dir = Path(feature_dir).resolve()
    cache_path = Path(cache_path).resolve()
    baseline_checkpoint_dir = Path(baseline_checkpoint_dir).resolve()
    checkpoint_dir = Path(checkpoint_dir).resolve()
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    torch_device = torch.device(
        device if device.startswith("cuda") and torch.cuda.is_available() else "cpu"
    )
    if torch_device.type == "cuda":
        torch.set_float32_matmul_precision("highest")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    cache_sha256 = file_sha256(cache_path)
    print("loading hierarchical train tensors into host memory", flush=True)
    train_data = _load_rows(
        feature_dir / "single_train.h5",
        cache_path,
        expected_cache_sha256=cache_sha256,
    )
    print("loading hierarchical test tensors into host memory", flush=True)
    test_data = _load_rows(
        feature_dir / "single_test.h5",
        cache_path,
        expected_cache_sha256=cache_sha256,
    )
    if train_data.provenance["hierarchy_provenance"] != test_data.provenance[
        "hierarchy_provenance"
    ]:
        raise RuntimeError("hierarchical train/test embedding provenance differs")
    train_proteins = set(
        train_data.protein_id[train_data.split == "train"].tolist()
    )
    validation_proteins = set(
        train_data.protein_id[train_data.split == "val"].tolist()
    )
    test_proteins = set(test_data.protein_id.tolist())
    if (
        train_proteins & validation_proteins
        or train_proteins & test_proteins
        or validation_proteins & test_proteins
    ):
        raise RuntimeError("hierarchical evaluation proteins are not disjoint")

    candidates: dict[str, dict[str, object]] = {}
    payloads: dict[str, list[dict[str, object]]] = {}
    for name, use_structure in (
        ("hierarchy_sequence", False),
        ("hierarchy_proteinmpnn", True),
    ):
        metrics, members = _train_candidate(
            name,
            train_data,
            test_data,
            use_structure=use_structure,
            seed=seed,
            epochs=discovery_epochs,
            minimum_epochs=min(8, discovery_epochs),
            patience=discovery_patience,
            batch_size=batch_size,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            ensemble_size=1,
            device=torch_device,
            objective=objective,
        )
        candidates[name] = metrics
        payloads[name] = members
        torch.save(
            {
                "schema": HIERARCHY_CHECKPOINT_SCHEMA,
                "candidate": name,
                "use_structure": use_structure,
                "members": members,
                "objective": asdict(objective),
                "metrics": metrics,
                "train_provenance": train_data.provenance,
                "test_provenance": test_data.provenance,
            },
            checkpoint_dir / f"{name}_ensemble.pt",
        )

    sequence_score = float(
        candidates["hierarchy_sequence"]["validation_selection_score"]
    )
    structure_score = float(
        candidates["hierarchy_proteinmpnn"]["validation_selection_score"]
    )
    selected = (
        "hierarchy_proteinmpnn"
        if structure_score > sequence_score + 0.002
        else "hierarchy_sequence"
    )
    selected_use_structure = selected == "hierarchy_proteinmpnn"
    print(
        f"selected {selected} on protein-disjoint validation; "
        f"starting {ensemble_size}-member main run for at least "
        f"{minimum_main_epochs} epochs",
        flush=True,
    )
    selected_metrics, selected_members = _train_candidate(
        selected,
        train_data,
        test_data,
        use_structure=selected_use_structure,
        seed=seed + 10_000,
        epochs=main_epochs,
        minimum_epochs=minimum_main_epochs,
        patience=main_patience,
        batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        ensemble_size=ensemble_size,
        device=torch_device,
        objective=objective,
    )
    validation_indices = np.flatnonzero(train_data.split == "val")
    baseline = {
        "validation": _baseline_evaluation(
            baseline_checkpoint_dir,
            train_data,
            validation_indices,
            threshold=objective.stabilizer_threshold,
            batch_size=batch_size,
            device=torch_device,
        ),
        "test": _baseline_evaluation(
            baseline_checkpoint_dir,
            test_data,
            np.arange(len(test_data.target), dtype=np.int64),
            threshold=objective.stabilizer_threshold,
            batch_size=batch_size,
            device=torch_device,
        ),
    }
    baseline_validation = baseline["validation"]
    candidate_validation = selected_metrics["validation"]
    selected_retrieval = (
        candidate_validation["retrieval_head"]
        if float(
            selected_metrics["validation"]["retrieval_head"]["average_precision"]
        )
        >= float(
            selected_metrics["validation"]["retrieval_from_ddg"][
                "average_precision"
            ]
        )
        else candidate_validation["retrieval_from_ddg"]
    )
    baseline_retrieval = baseline_validation["retrieval_from_ddg"]
    gates = {
        "validation_spearman_improves": (
            float(candidate_validation["regression"]["spearman"])
            > float(baseline_validation["regression"]["spearman"])
        ),
        "validation_mae_not_worse": (
            float(candidate_validation["regression"]["mae"])
            <= float(baseline_validation["regression"]["mae"])
        ),
        "validation_stabilizer_ap_improves": (
            float(selected_retrieval["average_precision"])
            > float(baseline_retrieval["average_precision"])
        ),
        "validation_top50_not_worse": (
            int(selected_retrieval["hits_at_50"])
            >= int(baseline_retrieval["hits_at_50"])
        ),
        "exact_directionality": (
            float(
                selected_metrics["directional_constraints"][
                    "maximum_absolute_forward_plus_reverse"
                ]
            )
            < 1e-6
            and float(
                selected_metrics["directional_constraints"][
                    "maximum_absolute_self_ddg"
                ]
            )
            < 1e-6
        ),
    }
    promoted = all(gates.values())
    selected_payload = {
        "schema": HIERARCHY_CHECKPOINT_SCHEMA,
        "candidate": selected,
        "use_structure": selected_use_structure,
        "members": selected_members,
        "objective": asdict(objective),
        "metrics": selected_metrics,
        "train_provenance": train_data.provenance,
        "test_provenance": test_data.provenance,
        "promotion": {
            "promoted_to_transfer_and_6b": promoted,
            "gates": gates,
            "policy": (
                "validation metrics and exact algebra only; historical test "
                "is reporting-only"
            ),
        },
    }
    torch.save(selected_payload, checkpoint_dir / "hierarchy_selected_ensemble.pt")
    result = {
        "schema": "protein-stabilizer.hierarchical-single-ablation.v1",
        "seed": seed,
        "objective": asdict(objective),
        "training_policy": {
            "batch_size": batch_size,
            "discovery_epochs": discovery_epochs,
            "discovery_ensemble_size": 1,
            "main_epoch_budget": main_epochs,
            "minimum_main_epochs": minimum_main_epochs,
            "main_patience": main_patience,
            "main_ensemble_size": ensemble_size,
            "schedule": "short linear warmup then cosine decay",
            "gradient_clip_norm": 5.0,
            "reporting": "optimizer steps and examples seen are recorded separately",
        },
        "candidate_limit": [
            "hierarchy_sequence",
            "hierarchy_proteinmpnn",
        ],
        "selection": {
            "rule": (
                "select ProteinMPNN only if its protein-disjoint validation "
                "composite exceeds sequence hierarchy by >0.002"
            ),
            "selected": selected,
            "sequence_validation_score": sequence_score,
            "structure_validation_score": structure_score,
        },
        "baseline": baseline,
        "candidates": candidates,
        "selected_main": selected_metrics,
        "promotion": {
            "promoted_to_transfer_and_6b": promoted,
            "gates": gates,
            "policy": (
                "selected winner must improve validation Spearman and "
                "stabilizer AP, not worsen validation MAE or top-50 recovery, "
                "and satisfy exact directionality; historical test metrics "
                "are reporting-only"
            ),
        },
        "split_integrity": {
            "train_proteins": len(train_proteins),
            "validation_proteins": len(validation_proteins),
            "test_proteins": len(test_proteins),
            "overlap": 0,
        },
        "cache": {
            "path": str(cache_path),
            "sha256": cache_sha256,
            "provenance": train_data.provenance["hierarchy_provenance"],
        },
    }
    (checkpoint_dir / "hierarchy_ablation.json").write_text(
        json.dumps(result, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return result


def load_hierarchical_ensemble(
    path: Path,
    device: str | torch.device = "cpu",
) -> tuple[HierarchicalDirectionalEnsemble, dict[str, object]]:
    payload = torch.load(path, map_location=device, weights_only=False)
    if payload.get("schema") != HIERARCHY_CHECKPOINT_SCHEMA:
        raise RuntimeError("hierarchical checkpoint schema mismatch")
    members: list[HierarchicalDirectionalMutationHead] = []
    for value in payload["members"]:
        model = HierarchicalDirectionalMutationHead(
            HierarchicalHeadConfig(**value["config"])
        )
        model.load_state_dict(value["state_dict"])
        members.append(model)
    return (
        HierarchicalDirectionalEnsemble(members).to(device).eval(),
        payload,
    )


def train_promoted_hierarchical_single(
    feature_dir: Path,
    cache_path: Path,
    baseline_checkpoint_dir: Path,
    discovery_report: Path,
    checkpoint_dir: Path,
    *,
    seed: int = 20260715,
    epochs: int = 70,
    minimum_epochs: int = 50,
    patience: int = 8,
    batch_size: int = 256,
    learning_rate: float = 7.5e-4,
    weight_decay: float = 1e-4,
    ensemble_size: int = 5,
    device: str = "cuda",
    objective: HierarchicalObjectiveConfig = HierarchicalObjectiveConfig(),
) -> dict[str, object]:
    """Scale only the frozen 600M validation winner to a larger backbone."""

    discovery = json.loads(Path(discovery_report).read_text(encoding="utf-8"))
    if not bool(
        discovery.get("promotion", {}).get("promoted_to_transfer_and_6b")
    ):
        raise RuntimeError("600M hierarchy logic did not pass its promotion gate")
    selected = str(discovery["selection"]["selected"])
    if selected not in {"hierarchy_sequence", "hierarchy_proteinmpnn"}:
        raise RuntimeError("unknown promoted hierarchy candidate")
    use_structure = selected == "hierarchy_proteinmpnn"
    feature_dir = Path(feature_dir).resolve()
    cache_path = Path(cache_path).resolve()
    baseline_checkpoint_dir = Path(baseline_checkpoint_dir).resolve()
    checkpoint_dir = Path(checkpoint_dir).resolve()
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    torch_device = torch.device(
        device if device.startswith("cuda") and torch.cuda.is_available() else "cpu"
    )
    if torch_device.type == "cuda":
        torch.set_float32_matmul_precision("highest")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    cache_sha256 = file_sha256(cache_path)
    train_data = _load_rows(
        feature_dir / "single_train.h5",
        cache_path,
        expected_cache_sha256=cache_sha256,
    )
    test_data = _load_rows(
        feature_dir / "single_test.h5",
        cache_path,
        expected_cache_sha256=cache_sha256,
    )
    train_proteins = set(
        train_data.protein_id[train_data.split == "train"].tolist()
    )
    validation_proteins = set(
        train_data.protein_id[train_data.split == "val"].tolist()
    )
    test_proteins = set(test_data.protein_id.tolist())
    if (
        train_proteins & validation_proteins
        or train_proteins & test_proteins
        or validation_proteins & test_proteins
    ):
        raise RuntimeError("promoted hierarchy proteins overlap across splits")
    metrics, members = _train_candidate(
        selected,
        train_data,
        test_data,
        use_structure=use_structure,
        seed=seed,
        epochs=epochs,
        minimum_epochs=minimum_epochs,
        patience=patience,
        batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        ensemble_size=ensemble_size,
        device=torch_device,
        objective=objective,
    )
    validation_indices = np.flatnonzero(train_data.split == "val")
    baseline = {
        "validation": _baseline_evaluation(
            baseline_checkpoint_dir,
            train_data,
            validation_indices,
            threshold=objective.stabilizer_threshold,
            batch_size=batch_size,
            device=torch_device,
        ),
        "test": _baseline_evaluation(
            baseline_checkpoint_dir,
            test_data,
            np.arange(len(test_data.target), dtype=np.int64),
            threshold=objective.stabilizer_threshold,
            batch_size=batch_size,
            device=torch_device,
        ),
    }
    baseline_validation = baseline["validation"]
    candidate_validation = metrics["validation"]
    retrieval_key = (
        "retrieval_head"
        if float(metrics["validation"]["retrieval_head"]["average_precision"])
        >= float(
            metrics["validation"]["retrieval_from_ddg"]["average_precision"]
        )
        else "retrieval_from_ddg"
    )
    selected_retrieval = candidate_validation[retrieval_key]
    baseline_retrieval = baseline_validation["retrieval_from_ddg"]
    gates = {
        "validation_spearman_improves": (
            float(candidate_validation["regression"]["spearman"])
            > float(baseline_validation["regression"]["spearman"])
        ),
        "validation_mae_not_worse": (
            float(candidate_validation["regression"]["mae"])
            <= float(baseline_validation["regression"]["mae"])
        ),
        "validation_stabilizer_ap_improves": (
            float(selected_retrieval["average_precision"])
            > float(baseline_retrieval["average_precision"])
        ),
        "validation_top50_not_worse": (
            int(selected_retrieval["hits_at_50"])
            >= int(baseline_retrieval["hits_at_50"])
        ),
        "exact_directionality": (
            float(
                metrics["directional_constraints"][
                    "maximum_absolute_forward_plus_reverse"
                ]
            )
            < 1e-6
            and float(
                metrics["directional_constraints"][
                    "maximum_absolute_self_ddg"
                ]
            )
            < 1e-6
        ),
    }
    report = {
        "schema": "protein-stabilizer.promoted-hierarchy-training.v1",
        "selected_600m_logic": selected,
        "discovery_report": {
            "path": str(Path(discovery_report).resolve()),
            "sha256": file_sha256(Path(discovery_report)),
        },
        "candidate": metrics,
        "baseline": baseline,
        "promotion_gates": gates,
        "production_eligible": all(gates.values()),
        "promotion_policy": (
            "validation metrics and exact algebra only; historical test is "
            "reporting-only"
        ),
        "cache": {
            "path": str(cache_path),
            "sha256": cache_sha256,
            "provenance": train_data.provenance["hierarchy_provenance"],
        },
        "split_integrity": {
            "train_proteins": len(train_proteins),
            "validation_proteins": len(validation_proteins),
            "test_proteins": len(test_proteins),
            "overlap": 0,
        },
    }
    torch.save(
        {
            "schema": HIERARCHY_CHECKPOINT_SCHEMA,
            "candidate": selected,
            "use_structure": use_structure,
            "members": members,
            "objective": asdict(objective),
            "metrics": metrics,
            "train_provenance": train_data.provenance,
            "test_provenance": test_data.provenance,
            "promotion": {
                "production_eligible": all(gates.values()),
                "gates": gates,
                "policy": (
                    "validation metrics and exact algebra only; historical "
                    "test is reporting-only"
                ),
            },
        },
        checkpoint_dir / "hierarchy_selected_ensemble.pt",
    )
    (checkpoint_dir / "hierarchy_scale_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    return report
