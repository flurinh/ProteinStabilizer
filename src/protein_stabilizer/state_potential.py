"""WT-conditioned amino-acid state-potential training and evaluation."""

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

from .data import AMINO_ACIDS, Mutation
from .embeddings import HierarchyEmbeddingReader, file_sha256
from .models import (
    StatePotentialConfig,
    StatePotentialEnsemble,
    StatePotentialMutationHead,
)
from .training import (
    balanced_stability_weights,
    set_reproducible_seed,
)
from .v2_training import (
    HierarchicalObjectiveConfig,
    HierarchyArrays,
    _batch_tensors,
    _evaluation,
    _load_rows,
    _predict,
    _ranking_pairs,
    _selection_score,
    load_hierarchical_ensemble,
)
from .v2_features import HIERARCHY_MULTI_ROW_SCHEMA
from .v2_multi import MULTI_REPRESENTATION_SCHEMA


STATE_POTENTIAL_CHECKPOINT_SCHEMA = (
    "protein-stabilizer.state-potential-ensemble.v1"
)


@dataclass
class StatePotentialArrays:
    hierarchy: HierarchyArrays
    wt_amino_acid: np.ndarray
    mutant_amino_acid: np.ndarray
    wt_absolute_stability: np.ndarray | None = None
    mutant_absolute_stability: np.ndarray | None = None


def _load_state_rows(
    row_path: Path,
    cache_path: Path,
    *,
    expected_cache_sha256: str,
) -> StatePotentialArrays:
    hierarchy = _load_rows(
        row_path,
        cache_path,
        expected_cache_sha256=expected_cache_sha256,
    )
    amino_acid_index = {
        amino_acid: index for index, amino_acid in enumerate(AMINO_ACIDS)
    }
    with h5py.File(row_path, "r") as handle:
        mutations = [
            Mutation.parse(value) for value in handle["mutation"].asstr()[:]
        ]
        wt_absolute_stability = (
            np.asarray(handle["wt_absolute_stability"], dtype=np.float32)
            if "wt_absolute_stability" in handle
            else None
        )
        mutant_absolute_stability = (
            np.asarray(handle["mutant_absolute_stability"], dtype=np.float32)
            if "mutant_absolute_stability" in handle
            else None
        )
    if (wt_absolute_stability is None) != (mutant_absolute_stability is None):
        raise RuntimeError("absolute-stability supervision must include both states")
    if len(mutations) != len(hierarchy.target):
        raise RuntimeError("state-potential mutation metadata count mismatch")
    return StatePotentialArrays(
        hierarchy=hierarchy,
        wt_amino_acid=np.asarray(
            [amino_acid_index[value.wt] for value in mutations],
            dtype=np.int64,
        ),
        mutant_amino_acid=np.asarray(
            [amino_acid_index[value.mutant] for value in mutations],
            dtype=np.int64,
        ),
        wt_absolute_stability=wt_absolute_stability,
        mutant_absolute_stability=mutant_absolute_stability,
    )


def _state_tensors(
    data: StatePotentialArrays,
    indices: np.ndarray,
    device: torch.device,
    *,
    use_structure: bool,
) -> dict[str, torch.Tensor]:
    hierarchy = _batch_tensors(
        data.hierarchy,
        indices,
        device,
        use_structure=use_structure,
    )
    return {
        "wt_window": hierarchy["wt_window"],
        "window_mask": hierarchy["window_mask"],
        "wt_global": hierarchy["wt_global"],
        "wt_amino_acid": torch.from_numpy(
            data.wt_amino_acid[indices]
        ).to(device),
        "mutant_amino_acid": torch.from_numpy(
            data.mutant_amino_acid[indices]
        ).to(device),
        "structure": hierarchy["structure"],
        "structure_mask": hierarchy["structure_mask"],
        "membrane": hierarchy["membrane"],
    }


def _predict_state(
    model: StatePotentialMutationHead | StatePotentialEnsemble,
    data: StatePotentialArrays,
    indices: np.ndarray,
    *,
    use_structure: bool,
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
                    **_state_tensors(
                        data,
                        selected,
                        device,
                        use_structure=use_structure,
                    )
                )
                .float()
                .cpu()
                .numpy()
            )
    return (
        np.concatenate(output)
        if output
        else np.empty(0, dtype=np.float32)
    )


def _ddg_evaluation(
    target: np.ndarray,
    prediction: np.ndarray,
    *,
    threshold: float,
) -> dict[str, object]:
    return _evaluation(
        target,
        prediction,
        -prediction,
        threshold=threshold,
    )


def _train_member(
    data: StatePotentialArrays,
    *,
    use_structure: bool,
    seed: int,
    epochs: int,
    minimum_epochs: int,
    patience: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    device: torch.device,
    objective: HierarchicalObjectiveConfig,
    initial_state_dict: dict[str, torch.Tensor] | None = None,
    enable_absolute_stability: bool = False,
) -> tuple[dict[str, object], np.ndarray]:
    started = time.monotonic()
    if objective.huber_delta <= 0.0:
        raise ValueError("Huber delta must be positive")
    if objective.mse_weight < 0.0:
        raise ValueError("MSE weight must be non-negative")
    if objective.selection_metric not in {"composite", "mae"}:
        raise ValueError("selection metric must be 'composite' or 'mae'")
    if objective.absolute_stability_weight < 0.0:
        raise ValueError("absolute-stability weight must be non-negative")
    if objective.absolute_stability_weight > 0.0:
        if not enable_absolute_stability:
            raise ValueError(
                "absolute-stability supervision requires the absolute head"
            )
        if (
            data.wt_absolute_stability is None
            or data.mutant_absolute_stability is None
        ):
            raise ValueError(
                "absolute-stability supervision requires WT and mutant targets"
            )
    hierarchy = data.hierarchy
    train_indices = np.flatnonzero(hierarchy.split == "train")
    validation_indices = np.flatnonzero(hierarchy.split == "val")
    config = StatePotentialConfig(
        embedding_dim=hierarchy.wt_window.shape[-1],
        window_size=hierarchy.wt_window.shape[1],
        structure_dim=hierarchy.structure.shape[1],
        membrane_dim=hierarchy.membrane.shape[1],
        absolute_stability_head=enable_absolute_stability,
    )
    weights = np.ones(len(hierarchy.target), dtype=np.float32)
    weights[train_indices] = balanced_stability_weights(
        hierarchy.target[train_indices],
        stabilizer_threshold=objective.stabilizer_threshold,
        neutral_upper_threshold=objective.neutral_upper_threshold,
        maximum_weight=objective.maximum_bin_weight,
    )
    weights *= hierarchy.sample_weight
    set_reproducible_seed(seed)
    rng = np.random.default_rng(seed)
    model = StatePotentialMutationHead(config).to(device)
    if initial_state_dict is not None:
        model.load_state_dict(initial_state_dict)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    warmup = max(1, min(5, epochs // 10))
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[
            torch.optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=0.20,
                end_factor=1.0,
                total_iters=warmup,
            ),
            torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=max(1, epochs - warmup),
                eta_min=learning_rate * 0.05,
            ),
        ],
        milestones=[warmup],
    )
    huber = torch.nn.HuberLoss(
        delta=objective.huber_delta,
        reduction="none",
    )
    binary = torch.nn.BCEWithLogitsLoss(reduction="none")
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
            "absolute_wt": 0.0,
            "absolute_mutant": 0.0,
            "retrieval": 0.0,
            "ranking": 0.0,
            "total": 0.0,
        }
        seen = 0
        for start in range(0, len(shuffled), batch_size):
            indices = shuffled[start : start + batch_size]
            target = torch.from_numpy(hierarchy.target[indices]).to(device)
            weight = torch.from_numpy(weights[indices]).to(device)
            label = (target < objective.stabilizer_threshold).float()
            optimizer.zero_grad(set_to_none=True)
            state_tensors = _state_tensors(
                data,
                indices,
                device,
                use_structure=use_structure,
            )
            if enable_absolute_stability:
                thermodynamic = model.predict_thermodynamic_state(**state_tensors)
                prediction = thermodynamic["ddg"]
            else:
                thermodynamic = None
                prediction = model(**state_tensors)
            retrieval_logit = -(
                prediction - objective.stabilizer_threshold
            )
            regression_loss = (
                huber(prediction, target) * weight
            ).sum() / weight.sum()
            mse_loss = (
                (prediction - target).square() * weight
            ).sum() / weight.sum()
            if (
                thermodynamic is not None
                and data.wt_absolute_stability is not None
                and data.mutant_absolute_stability is not None
                and objective.absolute_stability_weight > 0.0
            ):
                wt_absolute_target = torch.from_numpy(
                    data.wt_absolute_stability[indices]
                ).to(device)
                mutant_absolute_target = torch.from_numpy(
                    data.mutant_absolute_stability[indices]
                ).to(device)
                wt_absolute_loss = (
                    huber(
                        thermodynamic["wt_absolute_stability"],
                        wt_absolute_target,
                    )
                    * weight
                ).sum() / weight.sum()
                mutant_absolute_loss = (
                    huber(
                        thermodynamic["mutant_absolute_stability"],
                        mutant_absolute_target,
                    )
                    * weight
                ).sum() / weight.sum()
            else:
                wt_absolute_loss = torch.zeros((), device=device)
                mutant_absolute_loss = torch.zeros((), device=device)
            retrieval_loss = (
                binary(retrieval_logit, label) * weight
            ).sum() / weight.sum()
            stable, unstable = _ranking_pairs(
                hierarchy.protein_id[indices],
                hierarchy.target[indices],
                rng,
                count=min(
                    objective.ranking_pairs_per_batch,
                    len(indices),
                ),
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
                + objective.absolute_stability_weight
                * (wt_absolute_loss + mutant_absolute_loss)
                + objective.retrieval_weight * retrieval_loss
                + objective.ranking_weight * ranking_loss
            )
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), 5.0
            )
            optimizer.step()
            optimizer_steps += 1
            examples_seen += len(indices)
            count = len(indices)
            totals["regression"] += float(regression_loss.detach()) * count
            totals["mse"] += float(mse_loss.detach()) * count
            totals["absolute_wt"] += float(wt_absolute_loss.detach()) * count
            totals["absolute_mutant"] += (
                float(mutant_absolute_loss.detach()) * count
            )
            totals["retrieval"] += float(retrieval_loss.detach()) * count
            totals["ranking"] += float(ranking_loss.detach()) * count
            totals["total"] += float(loss.detach()) * count
            seen += count
        scheduler.step()
        validation_prediction = _predict_state(
            model,
            data,
            validation_indices,
            use_structure=use_structure,
            batch_size=batch_size,
            device=device,
        )
        validation = _ddg_evaluation(
            hierarchy.target[validation_indices],
            validation_prediction,
            threshold=objective.stabilizer_threshold,
        )
        score = _selection_score(
            validation,
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
            f"state-potential seed={seed} epoch={epoch:02d} "
            f"train={losses['total']:.4f} "
            f"val_rho={validation['regression']['spearman']:.4f} "
            f"val_mae={validation['regression']['mae']:.4f} "
            f"val_ap={validation['retrieval_from_ddg']['average_precision']:.4f}",
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
        raise RuntimeError("state-potential member has no valid checkpoint")
    model.load_state_dict(best_state)
    validation_prediction = _predict_state(
        model,
        data,
        validation_indices,
        use_structure=use_structure,
        batch_size=batch_size,
        device=device,
    )
    payload = {
        "config": asdict(config),
        "state_dict": copy.deepcopy(model.cpu().state_dict()),
        "seed": seed,
        "best_epoch": best_epoch,
        "selection_rule": (
            "protein-disjoint validation MAE"
            if objective.selection_metric == "mae"
            else (
                "protein-disjoint validation composite = 0.65 Spearman + "
                "0.25 stabilizer AP - 0.10 MAE"
            )
        ),
        "optimizer_steps": optimizer_steps,
        "examples_seen": examples_seen,
        "initialized_from_state_dict": initial_state_dict is not None,
        "absolute_stability_head": enable_absolute_stability,
        "absolute_supervision_rows": (
            int(len(hierarchy.target))
            if data.wt_absolute_stability is not None
            else 0
        ),
        "elapsed_seconds": time.monotonic() - started,
        "history": history,
        "validation": _ddg_evaluation(
            hierarchy.target[validation_indices],
            validation_prediction,
            threshold=objective.stabilizer_threshold,
        ),
    }
    return payload, validation_prediction


def _blend_search(
    target: np.ndarray,
    baseline: np.ndarray,
    state_potential: np.ndarray,
    *,
    threshold: float,
    selection_metric: str = "composite",
) -> tuple[float, list[dict[str, object]]]:
    records: list[dict[str, object]] = []
    for state_weight in np.linspace(0.0, 1.0, 21):
        prediction = (
            (1.0 - state_weight) * baseline
            + state_weight * state_potential
        )
        metrics = _ddg_evaluation(
            target,
            prediction,
            threshold=threshold,
        )
        records.append(
            {
                "state_potential_weight": float(state_weight),
                "selection_score": _selection_score(
                    metrics,
                    selection_metric=selection_metric,
                ),
                "metrics": metrics,
            }
        )
    selected = max(
        records,
        key=lambda value: (
            float(value["selection_score"]),
            -float(value["state_potential_weight"]),
        ),
    )
    return float(selected["state_potential_weight"]), records


def _directional_checks(
    model: StatePotentialEnsemble,
    data: StatePotentialArrays,
    *,
    use_structure: bool,
    device: torch.device,
) -> dict[str, object]:
    count = min(512, len(data.hierarchy.target))
    indices = np.arange(count, dtype=np.int64)
    tensors = _state_tensors(
        data,
        indices,
        device,
        use_structure=use_structure,
    )
    model.eval()
    with torch.inference_mode():
        forward = model(**tensors)
        reverse = model(
            tensors["wt_window"],
            tensors["window_mask"],
            tensors["wt_global"],
            tensors["mutant_amino_acid"],
            tensors["wt_amino_acid"],
            structure=tensors["structure"],
            structure_mask=tensors["structure_mask"],
            membrane=tensors["membrane"],
        )
        self_prediction = model(
            tensors["wt_window"],
            tensors["window_mask"],
            tensors["wt_global"],
            tensors["wt_amino_acid"],
            tensors["wt_amino_acid"],
            structure=tensors["structure"],
            structure_mask=tensors["structure_mask"],
            membrane=tensors["membrane"],
        )
        potentials = model.all_potentials(
            tensors["wt_window"],
            tensors["window_mask"],
            tensors["wt_global"],
            structure=tensors["structure"],
            structure_mask=tensors["structure_mask"],
            membrane=tensors["membrane"],
        )
        left = potentials[:, 0] - potentials[:, 1]
        middle = potentials[:, 1] - potentials[:, 2]
        direct = potentials[:, 0] - potentials[:, 2]
    return {
        "evaluated_rows": count,
        "maximum_absolute_forward_plus_reverse": float(
            torch.max(torch.abs(forward + reverse)).cpu()
        ),
        "maximum_absolute_self_ddg": float(
            torch.max(torch.abs(self_prediction)).cpu()
        ),
        "maximum_absolute_transitivity_error": float(
            torch.max(torch.abs(left + middle - direct)).cpu()
        ),
        "all_amino_acid_states_per_forward": len(AMINO_ACIDS),
    }


def train_state_potential_candidate(
    feature_dir: Path,
    cache_path: Path,
    baseline_checkpoint: Path,
    checkpoint_dir: Path,
    *,
    seed: int = 20260718,
    epochs: int = 70,
    minimum_epochs: int = 50,
    patience: int = 8,
    batch_size: int = 256,
    learning_rate: float = 7.5e-4,
    weight_decay: float = 1e-4,
    ensemble_size: int = 3,
    device: str = "cuda",
    objective: HierarchicalObjectiveConfig = HierarchicalObjectiveConfig(),
) -> dict[str, object]:
    """Train and gate a one-WT-forward state-potential candidate at 600M."""

    if ensemble_size < 1:
        raise ValueError("state-potential ensemble size must be positive")
    if minimum_epochs < 1 or minimum_epochs > epochs:
        raise ValueError("minimum epochs must be within the training budget")
    feature_dir = Path(feature_dir).resolve()
    cache_path = Path(cache_path).resolve()
    baseline_checkpoint = Path(baseline_checkpoint).resolve()
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
    train_data = _load_state_rows(
        feature_dir / "single_train.h5",
        cache_path,
        expected_cache_sha256=cache_sha256,
    )
    test_data = _load_state_rows(
        feature_dir / "single_test.h5",
        cache_path,
        expected_cache_sha256=cache_sha256,
    )
    hierarchy = train_data.hierarchy
    train_proteins = set(
        hierarchy.protein_id[hierarchy.split == "train"].tolist()
    )
    validation_proteins = set(
        hierarchy.protein_id[hierarchy.split == "val"].tolist()
    )
    test_proteins = set(test_data.hierarchy.protein_id.tolist())
    if (
        train_proteins & validation_proteins
        or train_proteins & test_proteins
        or validation_proteins & test_proteins
    ):
        raise RuntimeError("state-potential proteins overlap across splits")
    baseline, baseline_payload = load_hierarchical_ensemble(
        baseline_checkpoint,
        torch_device,
    )
    use_structure = bool(baseline_payload["use_structure"])
    validation_indices = np.flatnonzero(hierarchy.split == "val")
    test_indices = np.arange(
        len(test_data.hierarchy.target), dtype=np.int64
    )
    baseline_validation, _ = _predict(
        baseline,
        hierarchy,
        validation_indices,
        use_structure=use_structure,
        batch_size=batch_size,
        device=torch_device,
    )
    baseline_test, _ = _predict(
        baseline,
        test_data.hierarchy,
        test_indices,
        use_structure=use_structure,
        batch_size=batch_size,
        device=torch_device,
    )
    started = time.monotonic()
    members: list[dict[str, object]] = []
    validation_predictions: list[np.ndarray] = []
    for member_index in range(ensemble_size):
        payload, validation_prediction = _train_member(
            train_data,
            use_structure=use_structure,
            seed=seed + member_index,
            epochs=epochs,
            minimum_epochs=minimum_epochs,
            patience=patience,
            batch_size=batch_size,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            device=torch_device,
            objective=objective,
        )
        members.append(payload)
        validation_predictions.append(validation_prediction)
    model_members: list[StatePotentialMutationHead] = []
    for payload in members:
        member = StatePotentialMutationHead(
            StatePotentialConfig(**payload["config"])
        )
        member.load_state_dict(payload["state_dict"])
        model_members.append(member)
    ensemble = StatePotentialEnsemble(model_members).to(torch_device).eval()
    state_validation = np.mean(validation_predictions, axis=0)
    state_test = _predict_state(
        ensemble,
        test_data,
        test_indices,
        use_structure=use_structure,
        batch_size=batch_size,
        device=torch_device,
    )
    state_weight, blend_grid = _blend_search(
        hierarchy.target[validation_indices],
        baseline_validation,
        state_validation,
        threshold=objective.stabilizer_threshold,
        selection_metric=objective.selection_metric,
    )
    blended_validation = (
        (1.0 - state_weight) * baseline_validation
        + state_weight * state_validation
    )
    blended_test = (
        (1.0 - state_weight) * baseline_test + state_weight * state_test
    )
    baseline_metrics = {
        "validation": _ddg_evaluation(
            hierarchy.target[validation_indices],
            baseline_validation,
            threshold=objective.stabilizer_threshold,
        ),
        "test": _ddg_evaluation(
            test_data.hierarchy.target,
            baseline_test,
            threshold=objective.stabilizer_threshold,
        ),
    }
    state_metrics = {
        "validation": _ddg_evaluation(
            hierarchy.target[validation_indices],
            state_validation,
            threshold=objective.stabilizer_threshold,
        ),
        "test": _ddg_evaluation(
            test_data.hierarchy.target,
            state_test,
            threshold=objective.stabilizer_threshold,
        ),
    }
    blended_metrics = {
        "validation": _ddg_evaluation(
            hierarchy.target[validation_indices],
            blended_validation,
            threshold=objective.stabilizer_threshold,
        ),
        "test": _ddg_evaluation(
            test_data.hierarchy.target,
            blended_test,
            threshold=objective.stabilizer_threshold,
        ),
    }
    baseline_validation_metrics = baseline_metrics["validation"]
    blended_validation_metrics = blended_metrics["validation"]
    baseline_retrieval = baseline_validation_metrics["retrieval_from_ddg"]
    blended_retrieval = blended_validation_metrics["retrieval_from_ddg"]
    gates = {
        "validation_selects_nonzero_state_weight": state_weight > 0.0,
        "validation_spearman_improves": (
            float(blended_validation_metrics["regression"]["spearman"])
            > float(baseline_validation_metrics["regression"]["spearman"])
        ),
        "validation_mae_not_worse": (
            float(blended_validation_metrics["regression"]["mae"])
            <= float(baseline_validation_metrics["regression"]["mae"])
        ),
        "validation_stabilizer_ap_improves": (
            float(blended_retrieval["average_precision"])
            > float(baseline_retrieval["average_precision"])
        ),
        "validation_top50_not_worse": (
            int(blended_retrieval["hits_at_50"])
            >= int(baseline_retrieval["hits_at_50"])
        ),
    }
    checks = _directional_checks(
        ensemble,
        train_data,
        use_structure=use_structure,
        device=torch_device,
    )
    gates["exact_state_algebra"] = all(
        float(checks[name]) <= 1e-6
        for name in (
            "maximum_absolute_forward_plus_reverse",
            "maximum_absolute_self_ddg",
            "maximum_absolute_transitivity_error",
        )
    )
    production_eligible = all(gates.values())
    checkpoint_payload = {
        "schema": STATE_POTENTIAL_CHECKPOINT_SCHEMA,
        "candidate": "wt_conditioned_state_potential",
        "members": members,
        "use_structure": use_structure,
        "objective": asdict(objective),
        "state_potential_weight": state_weight,
        "baseline_checkpoint": str(baseline_checkpoint),
        "baseline_checkpoint_sha256": file_sha256(baseline_checkpoint),
        "cache_sha256": cache_sha256,
        "train_provenance": hierarchy.provenance,
        "test_provenance": test_data.hierarchy.provenance,
        "metrics": {
            "baseline": baseline_metrics,
            "state_potential": state_metrics,
            "blended": blended_metrics,
        },
        "promotion": {
            "production_eligible": production_eligible,
            "gates": gates,
            "policy": (
                "validation metrics and exact state algebra only; historical "
                "test is reporting-only"
            ),
        },
    }
    checkpoint_path = checkpoint_dir / "state_potential_ensemble.pt"
    torch.save(checkpoint_payload, checkpoint_path)
    report = {
        "schema": "protein-stabilizer.state-potential-training.v1",
        "candidate": "wt_conditioned_state_potential",
        "elapsed_seconds": time.monotonic() - started,
        "training_policy": {
            "seed": seed,
            "epochs": epochs,
            "minimum_epochs": minimum_epochs,
            "patience": patience,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "ensemble_size": ensemble_size,
            "precision": "native FP32; TF32 disabled",
        },
        "architecture": {
            "conditioning": (
                "WT ESM-C center + local window + global mean + "
                "ProteinMPNN/static membrane context"
            ),
            "mutation_score": "phi(context, mutant) - phi(context, WT)",
            "scan_complexity": "one frozen WT embedding and one 20-state head pass",
            "fixed_amino_acid_priors": (
                "hydropathy, volume, charge, and broad residue classes"
            ),
        },
        "split_integrity": {
            "train_proteins": len(train_proteins),
            "validation_proteins": len(validation_proteins),
            "test_proteins": len(test_proteins),
            "overlap": 0,
        },
        "baseline": baseline_metrics,
        "state_potential": state_metrics,
        "blend_selection": {
            "selected_on": "protein-held-out validation only",
            "state_potential_weight": state_weight,
            "grid": blend_grid,
        },
        "blended": blended_metrics,
        "directional_constraints": checks,
        "promotion": {
            "production_eligible": production_eligible,
            "gates": gates,
            "policy": (
                "validation metrics and exact state algebra only; historical "
                "test is reporting-only"
            ),
        },
        "members": [
            {
                key: value
                for key, value in member.items()
                if key != "state_dict"
            }
            for member in members
        ],
        "artifacts": {
            "checkpoint": str(checkpoint_path),
            "checkpoint_sha256": file_sha256(checkpoint_path),
            "baseline_checkpoint": str(baseline_checkpoint),
            "baseline_checkpoint_sha256": file_sha256(
                baseline_checkpoint
            ),
            "cache": str(cache_path),
            "cache_sha256": cache_sha256,
        },
    }
    report_path = checkpoint_dir / "state_potential_report.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return report


def load_state_potential_ensemble(
    path: Path,
    device: str | torch.device = "cpu",
) -> tuple[StatePotentialEnsemble, dict[str, object]]:
    payload = torch.load(path, map_location=device, weights_only=False)
    if payload.get("schema") != STATE_POTENTIAL_CHECKPOINT_SCHEMA:
        raise RuntimeError("state-potential checkpoint schema mismatch")
    members: list[StatePotentialMutationHead] = []
    for value in payload["members"]:
        model = StatePotentialMutationHead(
            StatePotentialConfig(**value["config"])
        )
        model.load_state_dict(value["state_dict"])
        members.append(model)
    return StatePotentialEnsemble(members).to(device).eval(), payload


def build_state_potential_multi_representations(
    feature_dir: Path,
    cache_path: Path,
    baseline_representation_dir: Path,
    state_checkpoint: Path,
    output_dir: Path,
    *,
    batch_size: int = 256,
    device: str = "cuda",
) -> dict[str, object]:
    """Build fused single/state latents for permutation-invariant multi training."""

    feature_dir = Path(feature_dir).resolve()
    cache_path = Path(cache_path).resolve()
    baseline_representation_dir = Path(
        baseline_representation_dir
    ).resolve()
    state_checkpoint = Path(state_checkpoint).resolve()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    torch_device = torch.device(
        device
        if device.startswith("cuda") and torch.cuda.is_available()
        else "cpu"
    )
    if torch_device.type == "cuda":
        torch.set_float32_matmul_precision("highest")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    state_model, state_payload = load_state_potential_ensemble(
        state_checkpoint, torch_device
    )
    if not bool(
        state_payload.get("promotion", {}).get("production_eligible")
    ):
        raise RuntimeError("state-potential checkpoint is not production eligible")
    cache_sha256 = file_sha256(cache_path)
    if str(state_payload.get("cache_sha256", "")) != cache_sha256:
        raise RuntimeError("state-potential checkpoint hierarchy cache mismatch")
    state_weight = float(state_payload["state_potential_weight"])
    checkpoint_sha256 = file_sha256(state_checkpoint)
    amino_acid_index = {
        amino_acid: index
        for index, amino_acid in enumerate(AMINO_ACIDS)
    }
    records: list[dict[str, object]] = []
    with HierarchyEmbeddingReader(cache_path) as cache:
        for split in ("train", "val", "test"):
            row_path = feature_dir / f"double_{split}.h5"
            baseline_path = (
                baseline_representation_dir / f"double_{split}.h5"
            )
            with (
                h5py.File(row_path, "r") as rows,
                h5py.File(baseline_path, "r") as baseline,
            ):
                if rows.attrs.get("schema") != HIERARCHY_MULTI_ROW_SCHEMA:
                    raise RuntimeError(f"{row_path} multi-row schema mismatch")
                if (
                    str(rows.attrs["hierarchy_cache_sha256"])
                    != cache_sha256
                ):
                    raise RuntimeError(
                        f"{row_path} hierarchy cache hash mismatch"
                    )
                if (
                    baseline.attrs.get("schema")
                    != MULTI_REPRESENTATION_SCHEMA
                ):
                    raise RuntimeError(
                        f"{baseline_path} multi representation schema mismatch"
                    )
                if (
                    str(baseline.attrs["row_source_sha256"])
                    != file_sha256(row_path)
                ):
                    raise RuntimeError(
                        f"{baseline_path} row-source hash mismatch"
                    )
                if (
                    str(baseline.attrs["base_checkpoint_sha256"])
                    != str(state_payload["baseline_checkpoint_sha256"])
                ):
                    raise RuntimeError(
                        f"{baseline_path} hierarchy checkpoint mismatch"
                    )
                count = len(rows["target"])
                mutations = np.stack(
                    [
                        rows["mutation_1"].asstr()[:],
                        rows["mutation_2"].asstr()[:],
                    ],
                    axis=1,
                )
                if not np.array_equal(
                    mutations[:, 0], baseline["mutation_1"].asstr()[:]
                ) or not np.array_equal(
                    mutations[:, 1], baseline["mutation_2"].asstr()[:]
                ):
                    raise RuntimeError(
                        f"{baseline_path} mutation rows are misaligned"
                    )
                wt_site_index = np.asarray(
                    rows["wt_site_index"], dtype=np.int64
                )
                wt_sequence_index = np.asarray(
                    rows["wt_sequence_index"], dtype=np.int64
                )
                structure = np.asarray(rows["structure"], dtype=np.float32)
                structure_mask = np.asarray(
                    rows["structure_mask"], dtype=bool
                )
                membrane = np.asarray(rows["membrane"], dtype=np.float32)
                target = np.asarray(rows["target"], dtype=np.float32)
                true_additive = np.asarray(
                    rows["true_additive"], dtype=np.float32
                )
                true_epistasis = np.asarray(
                    rows["true_epistasis"], dtype=np.float32
                )
                has_true_epistasis = np.asarray(
                    rows["has_true_epistasis"], dtype=bool
                )
                protein_id = rows["protein_id"].asstr()[:]
                source_sha256 = str(rows.attrs["source_sha256"])
                baseline_latent_dim = baseline["single_latent"].shape[-1]
                state_latent_dim = int(state_model.config.latent_dim)
                latent_dim = baseline_latent_dim + state_latent_dim
                output_path = output_dir / f"double_{split}.h5"
                partial = output_path.with_suffix(".h5.partial")
                partial.unlink(missing_ok=True)
                with h5py.File(partial, "w", libver="latest") as output:
                    output.attrs["schema"] = MULTI_REPRESENTATION_SCHEMA
                    output.attrs["split"] = split
                    output.attrs["row_source"] = str(row_path)
                    output.attrs["row_source_sha256"] = file_sha256(
                        row_path
                    )
                    output.attrs["source_sha256"] = source_sha256
                    output.attrs["hierarchy_cache"] = str(cache_path)
                    output.attrs[
                        "hierarchy_cache_sha256"
                    ] = cache_sha256
                    output.attrs["base_checkpoint"] = str(
                        state_checkpoint
                    )
                    output.attrs[
                        "base_checkpoint_sha256"
                    ] = checkpoint_sha256
                    output.attrs[
                        "hierarchy_checkpoint_sha256"
                    ] = str(state_payload["baseline_checkpoint_sha256"])
                    output.attrs[
                        "state_potential_checkpoint_sha256"
                    ] = checkpoint_sha256
                    output.attrs["state_potential_weight"] = state_weight
                    output.attrs[
                        "base_candidate"
                    ] = "hierarchy_state_potential_blend"
                    chunk_rows = min(1024, count)
                    single_latent = output.create_dataset(
                        "single_latent",
                        shape=(count, 2, latent_dim),
                        dtype="f4",
                        chunks=(chunk_rows, 2, latent_dim),
                    )
                    joint_latent = output.create_dataset(
                        "joint_latent",
                        shape=(count, 2, latent_dim),
                        dtype="f4",
                        chunks=(chunk_rows, 2, latent_dim),
                    )
                    single_ddg = output.create_dataset(
                        "single_ddg",
                        shape=(count, 2),
                        dtype="f4",
                        chunks=(min(4096, count), 2),
                    )
                    hierarchy_ddg = output.create_dataset(
                        "hierarchy_ddg",
                        shape=(count, 2),
                        dtype="f4",
                        chunks=(min(4096, count), 2),
                    )
                    state_potential_ddg = output.create_dataset(
                        "state_potential_ddg",
                        shape=(count, 2),
                        dtype="f4",
                        chunks=(min(4096, count), 2),
                    )
                    for start in range(0, count, batch_size):
                        stop = min(count, start + batch_size)
                        selected = slice(start, stop)
                        batch_count = stop - start
                        flat_count = 2 * batch_count
                        features = cache.indexed_features(
                            wt_site_index[selected].reshape(-1),
                            wt_sequence_index[selected].reshape(-1),
                        )
                        parsed = [
                            Mutation.parse(value)
                            for value in mutations[selected].reshape(-1)
                        ]
                        wt_amino_acid = torch.tensor(
                            [
                                amino_acid_index[value.wt]
                                for value in parsed
                            ],
                            dtype=torch.long,
                            device=torch_device,
                        )
                        mutant_amino_acid = torch.tensor(
                            [
                                amino_acid_index[value.mutant]
                                for value in parsed
                            ],
                            dtype=torch.long,
                            device=torch_device,
                        )
                        state_kwargs = {
                            "wt_window": torch.from_numpy(
                                features["window"].astype(np.float32)
                            ).to(torch_device),
                            "window_mask": torch.from_numpy(
                                features["window_mask"]
                            ).to(torch_device),
                            "wt_global": torch.from_numpy(
                                features["global_mean"].astype(np.float32)
                            ).to(torch_device),
                            "wt_amino_acid": wt_amino_acid,
                            "mutant_amino_acid": mutant_amino_acid,
                            "structure": torch.from_numpy(
                                structure[selected]
                                .reshape(flat_count, -1)
                                .astype(np.float32)
                            ).to(torch_device),
                            "structure_mask": torch.from_numpy(
                                structure_mask[selected].reshape(-1)
                            ).to(torch_device),
                            "membrane": torch.from_numpy(
                                membrane[selected]
                                .reshape(flat_count, -1)
                                .astype(np.float32)
                            ).to(torch_device),
                        }
                        with torch.inference_mode():
                            state_z = state_model.latent(
                                **state_kwargs
                            ).float()
                            state_prediction = state_model(
                                **state_kwargs
                            ).float()
                        baseline_single_z = torch.from_numpy(
                            np.asarray(
                                baseline["single_latent"][selected],
                                dtype=np.float32,
                            ).reshape(flat_count, baseline_latent_dim)
                        ).to(torch_device)
                        baseline_joint_z = torch.from_numpy(
                            np.asarray(
                                baseline["joint_latent"][selected],
                                dtype=np.float32,
                            ).reshape(flat_count, baseline_latent_dim)
                        ).to(torch_device)
                        baseline_prediction = torch.from_numpy(
                            np.asarray(
                                baseline["single_ddg"][selected],
                                dtype=np.float32,
                            ).reshape(flat_count)
                        ).to(torch_device)
                        fused_prediction = (
                            (1.0 - state_weight) * baseline_prediction
                            + state_weight * state_prediction
                        )
                        output_shape = (batch_count, 2, latent_dim)
                        single_latent[selected] = (
                            torch.cat([baseline_single_z, state_z], dim=-1)
                            .cpu()
                            .numpy()
                            .reshape(output_shape)
                        )
                        joint_latent[selected] = (
                            torch.cat([baseline_joint_z, state_z], dim=-1)
                            .cpu()
                            .numpy()
                            .reshape(output_shape)
                        )
                        single_ddg[selected] = (
                            fused_prediction.cpu().numpy().reshape(
                                batch_count, 2
                            )
                        )
                        hierarchy_ddg[selected] = (
                            baseline_prediction.cpu().numpy().reshape(
                                batch_count, 2
                            )
                        )
                        state_potential_ddg[selected] = (
                            state_prediction.cpu().numpy().reshape(
                                batch_count, 2
                            )
                        )
                    output.create_dataset("target", data=target)
                    output.create_dataset(
                        "true_additive", data=true_additive
                    )
                    output.create_dataset(
                        "true_epistasis", data=true_epistasis
                    )
                    output.create_dataset(
                        "has_true_epistasis",
                        data=has_true_epistasis,
                    )
                    string_dtype = h5py.string_dtype(encoding="utf-8")
                    output.create_dataset(
                        "protein_id",
                        data=np.asarray(protein_id, dtype=object),
                        dtype=string_dtype,
                    )
                    output.create_dataset(
                        "mutation_1",
                        data=np.asarray(mutations[:, 0], dtype=object),
                        dtype=string_dtype,
                    )
                    output.create_dataset(
                        "mutation_2",
                        data=np.asarray(mutations[:, 1], dtype=object),
                        dtype=string_dtype,
                    )
                    output.flush()
                partial.replace(output_path)
            records.append(
                {
                    "path": str(output_path),
                    "sha256": file_sha256(output_path),
                    "split": split,
                    "rows": count,
                    "proteins": len(set(protein_id.tolist())),
                    "latent_dimension": latent_dim,
                    "source_sha256": source_sha256,
                }
            )
            print(
                f"state-potential multi representations {split}: "
                f"{count:,} rows",
                flush=True,
            )
        provenance = cache.provenance
    manifest = {
        "schema": MULTI_REPRESENTATION_SCHEMA,
        "candidate": "hierarchy_state_potential_blend",
        "state_potential": {
            "path": str(state_checkpoint),
            "sha256": checkpoint_sha256,
            "weight": state_weight,
            "hierarchy_checkpoint_sha256": state_payload[
                "baseline_checkpoint_sha256"
            ],
        },
        "hierarchy_cache": {
            "path": str(cache_path),
            "sha256": cache_sha256,
            "provenance": provenance,
        },
        "baseline_representations": {
            "path": str(baseline_representation_dir),
            "manifest_sha256": file_sha256(
                baseline_representation_dir / "manifest.json"
            ),
        },
        "datasets": records,
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    manifest["manifest_path"] = str(manifest_path)
    return manifest


def _fused_representations(
    state_model: StatePotentialEnsemble,
    baseline: torch.nn.Module,
    data: StatePotentialArrays,
    *,
    state_weight: float,
    use_structure: bool,
    batch_size: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    fused_latent: list[np.ndarray] = []
    blended_ddg: list[np.ndarray] = []
    indices = np.arange(len(data.hierarchy.target), dtype=np.int64)
    state_model.eval()
    baseline.eval()
    with torch.inference_mode():
        for start in range(0, len(indices), batch_size):
            selected = indices[start : start + batch_size]
            state_tensors = _state_tensors(
                data,
                selected,
                device,
                use_structure=use_structure,
            )
            hierarchy_tensors = _batch_tensors(
                data.hierarchy,
                selected,
                device,
                use_structure=use_structure,
            )
            state_latent = state_model.latent(**state_tensors)
            state_ddg = state_model(**state_tensors)
            baseline_latent = baseline.latent(**hierarchy_tensors)
            baseline_ddg = baseline(**hierarchy_tensors)
            fused_latent.append(
                torch.cat([baseline_latent, state_latent], dim=-1)
                .float()
                .cpu()
                .numpy()
            )
            blended_ddg.append(
                (
                    (1.0 - state_weight) * baseline_ddg
                    + state_weight * state_ddg
                )
                .float()
                .cpu()
                .numpy()
            )
    return np.concatenate(fused_latent), np.concatenate(blended_ddg)


def build_state_potential_transfer_representations(
    feature_dir: Path,
    cache_path: Path,
    state_checkpoint: Path,
    output_dir: Path,
    *,
    batch_size: int = 256,
    device: str = "cuda",
) -> dict[str, object]:
    """Cache fused mutant-aware and state-potential transfer features."""

    from .v2_transfer import TRANSFER_REPRESENTATION_SCHEMA

    feature_dir = Path(feature_dir).resolve()
    cache_path = Path(cache_path).resolve()
    state_checkpoint = Path(state_checkpoint).resolve()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    torch_device = torch.device(
        device if device.startswith("cuda") and torch.cuda.is_available() else "cpu"
    )
    if torch_device.type == "cuda":
        torch.set_float32_matmul_precision("highest")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    state_model, payload = load_state_potential_ensemble(
        state_checkpoint, torch_device
    )
    baseline_checkpoint = Path(payload["baseline_checkpoint"]).resolve()
    if file_sha256(baseline_checkpoint) != str(
        payload["baseline_checkpoint_sha256"]
    ):
        raise RuntimeError("state-potential baseline checkpoint hash mismatch")
    baseline, baseline_payload = load_hierarchical_ensemble(
        baseline_checkpoint, torch_device
    )
    use_structure = bool(payload["use_structure"])
    if use_structure != bool(baseline_payload["use_structure"]):
        raise RuntimeError("state-potential and baseline structure policies differ")
    state_weight = float(payload["state_potential_weight"])
    cache_sha256 = file_sha256(cache_path)
    if cache_sha256 != str(payload["cache_sha256"]):
        raise RuntimeError("state-potential hierarchy cache hash mismatch")
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
    records: list[dict[str, object]] = []
    for name, target_kind, units, favorable in specs:
        row_path = feature_dir / f"{name}.h5"
        data = _load_state_rows(
            row_path,
            cache_path,
            expected_cache_sha256=cache_sha256,
        )
        latent, base_ddg = _fused_representations(
            state_model,
            baseline,
            data,
            state_weight=state_weight,
            use_structure=use_structure,
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
            handle.attrs["base_checkpoint"] = str(state_checkpoint)
            handle.attrs["base_checkpoint_sha256"] = file_sha256(
                state_checkpoint
            )
            handle.attrs["base_candidate"] = (
                "hierarchy_plus_wt_state_potential"
            )
            handle.attrs["base_use_structure"] = use_structure
            handle.attrs["state_potential_weight"] = state_weight
            handle.attrs["hierarchy_baseline_checkpoint_sha256"] = str(
                payload["baseline_checkpoint_sha256"]
            )
            handle.create_dataset(
                "base_latent", data=latent.astype(np.float32)
            )
            handle.create_dataset(
                "base_ddg", data=base_ddg.astype(np.float32)
            )
            handle.create_dataset(
                "membrane", data=data.hierarchy.membrane.astype(np.float32)
            )
            handle.create_dataset(
                "target", data=data.hierarchy.target.astype(np.float32)
            )
            handle.create_dataset(
                "sample_weight",
                data=data.hierarchy.sample_weight.astype(np.float32),
            )
            string_dtype = h5py.string_dtype(encoding="utf-8")
            for field, values in (
                ("protein_id", data.hierarchy.protein_id),
                ("split", data.hierarchy.split),
                (
                    "mutation",
                    np.asarray(source["mutation"].asstr()[:]),
                ),
            ):
                handle.create_dataset(
                    field,
                    data=np.asarray(values, dtype=object),
                    dtype=string_dtype,
                )
            for optional in ("assay_id", "site_id", "uniprot_id"):
                if optional in source:
                    handle.create_dataset(
                        optional,
                        data=np.asarray(
                            source[optional].asstr()[:], dtype=object
                        ),
                        dtype=string_dtype,
                    )
            handle.flush()
        partial.replace(output)
        records.append(
            {
                "name": name,
                "path": str(output),
                "sha256": file_sha256(output),
                "rows": len(data.hierarchy.target),
                "proteins": len(
                    set(data.hierarchy.protein_id.tolist())
                ),
                "target_kind": target_kind,
                "target_units": units,
                "favorable_direction": favorable,
                "source_sha256": data.hierarchy.provenance[
                    "source_sha256"
                ],
            }
        )
        print(
            f"state-potential transfer representations {name}: "
            f"{len(data.hierarchy.target):,} rows",
            flush=True,
        )
    manifest = {
        "schema": TRANSFER_REPRESENTATION_SCHEMA,
        "base_checkpoint": {
            "path": str(state_checkpoint),
            "sha256": file_sha256(state_checkpoint),
            "candidate": "hierarchy_plus_wt_state_potential",
            "state_potential_weight": state_weight,
            "use_structure": use_structure,
        },
        "hierarchy_cache": {
            "path": str(cache_path),
            "sha256": cache_sha256,
        },
        "datasets": records,
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    manifest["manifest_path"] = str(manifest_path)
    return manifest
