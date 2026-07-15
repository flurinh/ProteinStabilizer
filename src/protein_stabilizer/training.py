"""Training and held-out evaluation for stability heads."""

from __future__ import annotations

import copy
import json
import math
import random
import time
from dataclasses import asdict
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Callable, Iterable

import h5py
import joblib
import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr, spearmanr
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .models import (
    EpistasisConfig,
    MultiMutationHead,
    SingleHeadConfig,
    SingleMutationEnsemble,
    SingleMutationHead,
)


def set_reproducible_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def regression_metrics(target: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    target = np.asarray(target, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    if target.shape != prediction.shape or target.ndim != 1:
        raise ValueError("metric arrays must have equal one-dimensional shape")
    error = prediction - target
    if len(target) < 2 or np.std(target) == 0 or np.std(prediction) == 0:
        pearson = math.nan
        spearman = math.nan
    else:
        pearson = float(pearsonr(target, prediction).statistic)
        spearman = float(spearmanr(target, prediction).statistic)
    return {
        "n": int(len(target)),
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(np.square(error)))),
        "pearson": pearson,
        "spearman": spearman,
    }


def _torch_predictions(
    model: torch.nn.Module,
    arrays: tuple[np.ndarray, ...],
    predict: Callable[..., torch.Tensor],
    *,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    outputs: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(arrays[0]), batch_size):
            tensors = [
                torch.from_numpy(array[start : start + batch_size].astype(np.float32)).to(
                    device
                )
                for array in arrays
            ]
            outputs.append(predict(*tensors).float().cpu().numpy())
    return np.concatenate(outputs) if outputs else np.empty(0, dtype=np.float32)


def _save_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")


def train_single_head(
    feature_dir: Path,
    checkpoint_dir: Path,
    *,
    seed: int = 20260715,
    epochs: int = 30,
    patience: int = 6,
    batch_size: int = 1024,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    ensemble_size: int = 5,
    device: str = "cuda",
) -> dict[str, object]:
    if ensemble_size < 1:
        raise ValueError("single-head ensemble size must be positive")
    torch_device = torch.device(device if torch.cuda.is_available() else "cpu")
    with h5py.File(Path(feature_dir) / "single_train.h5", "r") as handle:
        delta = np.asarray(handle["delta"], dtype=np.float16)
        target = np.asarray(handle["target"], dtype=np.float32)
        split = np.asarray(handle["split"].asstr()[:])
        embedding_provenance = json.loads(handle.attrs["embedding_provenance"])
        train_source_sha256 = str(handle.attrs["source_sha256"])
    with h5py.File(Path(feature_dir) / "single_test.h5", "r") as handle:
        test_delta = np.asarray(handle["delta"], dtype=np.float16)
        test_target = np.asarray(handle["target"], dtype=np.float32)
        test_source_sha256 = str(handle.attrs["source_sha256"])

    train_indices = np.flatnonzero(split == "train")
    val_indices = np.flatnonzero(split == "val")
    config = SingleHeadConfig(embedding_dim=delta.shape[1])
    started = time.monotonic()
    member_payloads: list[dict[str, object]] = []
    member_metrics: list[dict[str, object]] = []
    val_predictions: list[np.ndarray] = []
    test_predictions: list[np.ndarray] = []
    for member_index in range(ensemble_size):
        member_seed = seed + member_index
        set_reproducible_seed(member_seed)
        model = SingleMutationHead(config).to(torch_device)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
        loss_function = torch.nn.HuberLoss(delta=1.0)
        rng = np.random.default_rng(member_seed)
        best_state: dict[str, torch.Tensor] | None = None
        best_spearman = -math.inf
        best_epoch = 0
        history: list[dict[str, object]] = []
        stale = 0
        for epoch in range(1, epochs + 1):
            model.train()
            shuffled = rng.permutation(train_indices)
            total_loss = 0.0
            seen = 0
            for start in range(0, len(shuffled), batch_size):
                indices = shuffled[start : start + batch_size]
                x = torch.from_numpy(delta[indices].astype(np.float32)).to(torch_device)
                y = torch.from_numpy(target[indices]).to(torch_device)
                optimizer.zero_grad(set_to_none=True)
                prediction = model(x)
                loss = loss_function(prediction, y)
                loss.backward()
                optimizer.step()
                total_loss += float(loss.detach()) * len(indices)
                seen += len(indices)
            validation_prediction = _torch_predictions(
                model,
                (delta[val_indices],),
                lambda x: model(x),
                batch_size=batch_size,
                device=torch_device,
            )
            validation = regression_metrics(target[val_indices], validation_prediction)
            score = validation["spearman"]
            history.append(
                {
                    "epoch": epoch,
                    "train_huber": total_loss / seen,
                    "validation": validation,
                }
            )
            print(
                f"single member={member_index + 1}/{ensemble_size} "
                f"epoch={epoch:02d} train_huber={total_loss / seen:.4f} "
                f"val_rmse={validation['rmse']:.4f} val_spearman={score:.4f}",
                flush=True,
            )
            if np.isfinite(score) and score > best_spearman + 1e-5:
                best_spearman = score
                best_epoch = epoch
                best_state = copy.deepcopy(model.state_dict())
                stale = 0
            else:
                stale += 1
                if stale >= patience:
                    break
        if best_state is None:
            raise RuntimeError(
                f"single-mutant ensemble member {member_index} produced no valid model"
            )
        model.load_state_dict(best_state)
        val_prediction = _torch_predictions(
            model,
            (delta[val_indices],),
            lambda x: model(x),
            batch_size=batch_size,
            device=torch_device,
        )
        test_prediction = _torch_predictions(
            model,
            (test_delta,),
            lambda x: model(x),
            batch_size=batch_size,
            device=torch_device,
        )
        val_predictions.append(val_prediction)
        test_predictions.append(test_prediction)
        member_metrics.append(
            {
                "member": member_index,
                "seed": member_seed,
                "best_epoch": best_epoch,
                "validation": regression_metrics(target[val_indices], val_prediction),
                "test": regression_metrics(test_target, test_prediction),
                "history": history,
            }
        )
        member_payloads.append(
            {
                "config": asdict(config),
                "state_dict": copy.deepcopy(model.cpu().state_dict()),
            }
        )
    val_prediction = np.mean(val_predictions, axis=0)
    test_prediction = np.mean(test_predictions, axis=0)
    metrics = {
        "schema": "protein-stabilizer.single-ensemble-training.v1",
        "seed": seed,
        "ensemble_size": ensemble_size,
        "elapsed_seconds": time.monotonic() - started,
        "train_rows": int(len(train_indices)),
        "validation_rows": int(len(val_indices)),
        "test_rows": int(len(test_target)),
        "validation": regression_metrics(target[val_indices], val_prediction),
        "test": regression_metrics(test_target, test_prediction),
        "members": member_metrics,
        "embedding_provenance": embedding_provenance,
        "train_source_sha256": train_source_sha256,
        "test_source_sha256": test_source_sha256,
    }
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    primary = member_payloads[0]
    torch.save(
        {
            "schema": "protein-stabilizer.single-head.v1",
            "config": primary["config"],
            "state_dict": primary["state_dict"],
            "metrics": member_metrics[0],
        },
        checkpoint_dir / "single_head.pt",
    )
    torch.save(
        {
            "schema": "protein-stabilizer.single-ensemble.v1",
            "members": member_payloads,
            "metrics": metrics,
        },
        checkpoint_dir / "single_ensemble.pt",
    )
    _save_json(checkpoint_dir / "single_metrics.json", metrics)
    return metrics


def load_single_checkpoint(path: Path, device: str | torch.device = "cpu") -> SingleMutationHead:
    payload = torch.load(path, map_location=device, weights_only=False)
    if payload.get("schema") != "protein-stabilizer.single-head.v1":
        raise RuntimeError("single-head checkpoint schema mismatch")
    model = SingleMutationHead(SingleHeadConfig(**payload["config"]))
    model.load_state_dict(payload["state_dict"])
    return model.to(device).eval()


def load_single_ensemble_checkpoint(
    path: Path, device: str | torch.device = "cpu"
) -> SingleMutationEnsemble:
    payload = torch.load(path, map_location=device, weights_only=False)
    if payload.get("schema") != "protein-stabilizer.single-ensemble.v1":
        raise RuntimeError("single-ensemble checkpoint schema mismatch")
    members: list[SingleMutationHead] = []
    for member_payload in payload["members"]:
        member = SingleMutationHead(SingleHeadConfig(**member_payload["config"]))
        member.load_state_dict(member_payload["state_dict"])
        members.append(member)
    return SingleMutationEnsemble(members).to(device).eval()


def load_scoring_checkpoint(
    checkpoint_dir: Path, device: str | torch.device = "cpu"
) -> SingleMutationHead | SingleMutationEnsemble:
    checkpoint_dir = Path(checkpoint_dir)
    ensemble_path = checkpoint_dir / "single_ensemble.pt"
    if ensemble_path.exists():
        return load_single_ensemble_checkpoint(ensemble_path, device)
    return load_single_checkpoint(checkpoint_dir / "single_head.pt", device)


def _base_representation(
    model: SingleMutationHead | SingleMutationEnsemble,
    delta: np.ndarray,
    *,
    batch_size: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    latent = _torch_predictions(
        model,
        (delta,),
        lambda x: model.latent(x),
        batch_size=batch_size,
        device=device,
    )
    base_ddg = _torch_predictions(
        model,
        (delta,),
        lambda x: model(x),
        batch_size=batch_size,
        device=device,
    )
    norm = np.linalg.norm(delta.astype(np.float32), axis=1)
    representation = np.concatenate(
        [latent, base_ddg[:, None], norm[:, None]], axis=1
    ).astype(np.float32)
    return representation, latent, base_ddg


def _load_transfer_features(path: Path) -> dict[str, np.ndarray]:
    with h5py.File(path, "r") as handle:
        result = {
            "delta": np.asarray(handle["delta"], dtype=np.float16),
            "target": np.asarray(handle["target"], dtype=np.float32),
            "sample_weight": np.asarray(handle["sample_weight"], dtype=np.float32),
            "split": np.asarray(handle["split"].asstr()[:]),
            "protein_id": np.asarray(handle["protein_id"].asstr()[:]),
            "topology": np.asarray(handle["topology"].asstr()[:]),
        }
        count = len(result["target"])
        result["is_reverse"] = (
            np.asarray(handle["is_reverse"], dtype=bool)
            if "is_reverse" in handle
            else np.zeros(count, dtype=bool)
        )
        result["source_split"] = (
            np.asarray(handle["source_split"].asstr()[:])
            if "source_split" in handle
            else result["split"].copy()
        )
        result["pdb_id"] = (
            np.asarray(handle["pdb_id"].asstr()[:])
            if "pdb_id" in handle
            else result["protein_id"].copy()
        )
        result["official_test_site_overlap"] = (
            np.asarray(handle["official_test_site_overlap"], dtype=bool)
            if "official_test_site_overlap" in handle
            else np.zeros(count, dtype=bool)
        )
        result["gpcr_overlap_split"] = (
            np.asarray(handle["gpcr_overlap_split"].asstr()[:])
            if "gpcr_overlap_split" in handle
            else np.full(count, "none", dtype=object)
        )
        return result


def _ridge_pipeline(components: int | None, alpha: float, seed: int) -> Pipeline:
    steps: list[tuple[str, object]] = [("scale", StandardScaler())]
    if components is not None:
        steps.append(("pca", PCA(n_components=components, random_state=seed)))
    steps.append(("ridge", Ridge(alpha=alpha)))
    return Pipeline(steps)


def _train_auxiliary_head(
    data: dict[str, np.ndarray],
    base_checkpoint: Path,
    output_checkpoint: Path,
    *,
    target_kind: str,
    validation_name: str,
    seed: int,
    device: torch.device,
    batch_size: int = 256,
    epochs: int = 60,
    patience: int = 10,
) -> dict[str, object]:
    set_reproducible_seed(seed)
    model = load_single_checkpoint(base_checkpoint, device)
    if target_kind == "delta_tm":
        # Stability has opposite signs in the two targets: negative ddG and
        # positive delta-Tm are favorable. This is a useful initialization.
        with torch.no_grad():
            model.output.weight.mul_(-1)
            model.output.bias.mul_(-1)
        # MPTherm is small; adapt only the upper nonlinear block and output.
        for module in list(model.encoder.children())[:4]:
            for parameter in module.parameters():
                parameter.requires_grad_(False)

    train_indices = np.flatnonzero(data["split"] == "train")
    val_indices = np.flatnonzero(data["split"] == validation_name)
    test_indices = np.flatnonzero(data["split"] == "test")
    protein_ids = data["protein_id"]
    counts = pd.Series(protein_ids[train_indices]).value_counts().to_dict()
    median_count = float(np.median(list(counts.values())))
    weights = data["sample_weight"].astype(np.float32).copy()
    weights[train_indices] *= np.asarray(
        [math.sqrt(median_count / counts[str(protein_ids[index])]) for index in train_indices],
        dtype=np.float32,
    )
    if target_kind == "delta_tm":
        weights[train_indices] *= np.where(
            data["topology"][train_indices] == "Membrane", 2.0, 1.0
        )
    weights[train_indices] = np.clip(weights[train_indices], 0.1, 5.0)
    weights[train_indices] /= weights[train_indices].mean()

    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=2e-4, weight_decay=1e-4)
    huber = torch.nn.HuberLoss(
        delta=2.0 if target_kind == "delta_tm" else 1.0, reduction="none"
    )
    rng = np.random.default_rng(seed)
    best_state: dict[str, torch.Tensor] | None = None
    best_score = -math.inf
    best_epoch = 0
    stale = 0
    history: list[dict[str, object]] = []
    for epoch in range(1, epochs + 1):
        model.train()
        shuffled = rng.permutation(train_indices)
        total_loss = 0.0
        total_weight = 0.0
        for start in range(0, len(shuffled), batch_size):
            indices = shuffled[start : start + batch_size]
            x = torch.from_numpy(data["delta"][indices].astype(np.float32)).to(device)
            y = torch.from_numpy(data["target"][indices]).to(device)
            batch_weight = torch.from_numpy(weights[indices]).to(device)
            optimizer.zero_grad(set_to_none=True)
            loss_values = huber(model(x), y)
            loss = (loss_values * batch_weight).sum() / batch_weight.sum()
            loss.backward()
            optimizer.step()
            total_loss += float((loss_values * batch_weight).sum().detach())
            total_weight += float(batch_weight.sum())
        prediction = _torch_predictions(
            model,
            (data["delta"][val_indices],),
            lambda x: model(x),
            batch_size=batch_size,
            device=device,
        )
        validation = regression_metrics(data["target"][val_indices], prediction)
        score = float(validation["spearman"])
        membrane_validation: dict[str, float] | None = None
        if target_kind == "delta_tm":
            membrane = data["topology"][val_indices] == "Membrane"
            if membrane.sum() >= 2:
                membrane_validation = regression_metrics(
                    data["target"][val_indices][membrane], prediction[membrane]
                )
                if np.isfinite(membrane_validation["spearman"]):
                    score = 0.7 * float(membrane_validation["spearman"]) + 0.3 * score
        history.append(
            {
                "epoch": epoch,
                "train_weighted_huber": total_loss / total_weight,
                "validation": validation,
                "membrane_validation": membrane_validation,
                "selection_score": score,
            }
        )
        if np.isfinite(score) and score > best_score + 1e-5:
            best_score = score
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break
    if best_state is None:
        raise RuntimeError(f"{target_kind} auxiliary training produced no valid model")
    model.load_state_dict(best_state)
    val_prediction = _torch_predictions(
        model,
        (data["delta"][val_indices],),
        lambda x: model(x),
        batch_size=batch_size,
        device=device,
    )
    test_prediction = _torch_predictions(
        model,
        (data["delta"][test_indices],),
        lambda x: model(x),
        batch_size=batch_size,
        device=device,
    )
    metrics: dict[str, object] = {
        "target_kind": target_kind,
        "best_epoch": best_epoch,
        "train_rows": int(len(train_indices)),
        "validation_rows": int(len(val_indices)),
        "test_rows": int(len(test_indices)),
        "validation": regression_metrics(data["target"][val_indices], val_prediction),
        "test": regression_metrics(data["target"][test_indices], test_prediction),
        "history": history,
    }
    if target_kind == "delta_tm":
        metrics["test_by_topology"] = {
            str(value): regression_metrics(
                data["target"][test_indices][data["topology"][test_indices] == value],
                test_prediction[data["topology"][test_indices] == value],
            )
            for value in sorted(set(data["topology"][test_indices].tolist()))
        }
    torch.save(
        {
            "schema": "protein-stabilizer.auxiliary-head.v1",
            "target_kind": target_kind,
            "config": model.config_dict(),
            "state_dict": model.cpu().state_dict(),
            "metrics": metrics,
        },
        output_checkpoint,
    )
    return metrics


def load_auxiliary_checkpoint(
    path: Path, device: str | torch.device = "cpu"
) -> SingleMutationHead:
    payload = torch.load(path, map_location=device, weights_only=False)
    if payload.get("schema") != "protein-stabilizer.auxiliary-head.v1":
        raise RuntimeError("auxiliary-head checkpoint schema mismatch")
    model = SingleMutationHead(SingleHeadConfig(**payload["config"]))
    model.load_state_dict(payload["state_dict"])
    return model.to(device).eval()


def membrane_adapter_features(
    feature_kind: str,
    delta: np.ndarray,
    latent: np.ndarray,
    base_ddg: np.ndarray,
) -> np.ndarray:
    """Build the frozen-encoder inputs understood by membrane adapter files."""

    if feature_kind == "raw_delta":
        return delta.astype(np.float32)
    if feature_kind == "base_latent":
        return latent.astype(np.float32)
    if feature_kind == "base_latent_ddg":
        return np.concatenate(
            [latent.astype(np.float32), base_ddg.astype(np.float32)[:, None]], axis=1
        )
    raise ValueError(f"unknown membrane adapter feature kind {feature_kind!r}")


def _metrics_by_protein(
    target: np.ndarray,
    prediction: np.ndarray,
    protein_id: np.ndarray,
) -> dict[str, object]:
    by_protein = {
        str(name): regression_metrics(target[protein_id == name], prediction[protein_id == name])
        for name in sorted(set(protein_id.tolist()))
    }
    finite = [
        float(value["spearman"])
        for value in by_protein.values()
        if np.isfinite(value["spearman"])
    ]
    return {
        "overall": regression_metrics(target, prediction),
        "macro_protein_spearman": float(np.mean(finite)) if finite else math.nan,
        "by_protein": by_protein,
    }


def _fit_membrane_ridge(
    features: np.ndarray,
    target: np.ndarray,
    weights: np.ndarray,
    indices: np.ndarray,
    *,
    alpha: float,
    feature_kind: str,
) -> Ridge:
    model = Ridge(alpha=alpha, fit_intercept=feature_kind != "raw_delta")
    model.fit(features[indices], target[indices], sample_weight=weights[indices])
    return model


def train_membrane_ddg_adapter(
    feature_dir: Path,
    checkpoint_dir: Path,
    *,
    seed: int = 20260715,
    device: str = "cuda",
    batch_size: int = 512,
) -> dict[str, object]:
    """Train a compact alpha-helical membrane ΔΔG transfer head.

    Model selection uses the source publication's four-protein training split
    with leave-one-protein-out validation. DsbB and glycophorin A forward
    measurements remain untouched until the selected model is evaluated.
    """

    set_reproducible_seed(seed)
    feature_dir = Path(feature_dir)
    checkpoint_dir = Path(checkpoint_dir)
    torch_device = torch.device(device if torch.cuda.is_available() else "cpu")
    data = _load_transfer_features(feature_dir / "mcsm_membrane.h5")
    base_model = load_scoring_checkpoint(checkpoint_dir, torch_device)
    _, latent, base_ddg = _base_representation(
        base_model,
        data["delta"],
        batch_size=batch_size,
        device=torch_device,
    )
    feature_arrays = {
        name: membrane_adapter_features(name, data["delta"], latent, base_ddg)
        for name in ("raw_delta", "base_latent", "base_latent_ddg")
    }
    development = np.flatnonzero(data["source_split"] == "train")
    test = np.flatnonzero(
        (data["topology"] == "alpha_helical")
        & (data["split"] == "test")
        & ~data["is_reverse"]
    )
    development_proteins = sorted(set(data["protein_id"][development].tolist()))
    test_proteins = sorted(set(data["protein_id"][test].tolist()))
    if development_proteins != ["1PY6", "1QD6", "2XOV", "3GP6"]:
        raise RuntimeError(f"unexpected membrane development proteins {development_proteins}")
    if test_proteins != ["1AFO", "2K73"]:
        raise RuntimeError(f"unexpected membrane test proteins {test_proteins}")

    candidates: list[dict[str, object]] = []
    for feature_kind, features in feature_arrays.items():
        for alpha in (0.01, 0.1, 1.0, 10.0, 100.0, 1000.0):
            fold_metrics: dict[str, dict[str, float]] = {}
            for held_out in development_proteins:
                fit_indices = development[data["protein_id"][development] != held_out]
                validation_indices = development[
                    (data["protein_id"][development] == held_out)
                    & ~data["is_reverse"][development]
                ]
                model = _fit_membrane_ridge(
                    features,
                    data["target"],
                    data["sample_weight"],
                    fit_indices,
                    alpha=alpha,
                    feature_kind=feature_kind,
                )
                prediction = model.predict(features[validation_indices])
                fold_metrics[held_out] = regression_metrics(
                    data["target"][validation_indices], prediction
                )
            finite = [
                float(value["spearman"])
                for value in fold_metrics.values()
                if np.isfinite(value["spearman"])
            ]
            candidates.append(
                {
                    "feature_kind": feature_kind,
                    "feature_dimension": int(features.shape[1]),
                    "alpha": alpha,
                    "macro_protein_spearman": (
                        float(np.mean(finite)) if finite else math.nan
                    ),
                    "macro_protein_rmse": float(
                        np.mean([float(value["rmse"]) for value in fold_metrics.values()])
                    ),
                    "folds": fold_metrics,
                }
            )
    finite_candidates = [
        value
        for value in candidates
        if np.isfinite(float(value["macro_protein_spearman"]))
    ]
    if not finite_candidates:
        raise RuntimeError("membrane adapter validation produced no finite rank metric")
    best_score = max(float(value["macro_protein_spearman"]) for value in finite_candidates)
    eligible = [
        value
        for value in finite_candidates
        if float(value["macro_protein_spearman"]) >= best_score - 0.01
    ]
    chosen = min(
        eligible,
        key=lambda value: (
            int(value["feature_dimension"]),
            float(value["macro_protein_rmse"]),
            -float(value["alpha"]),
        ),
    )
    feature_kind = str(chosen["feature_kind"])
    alpha = float(chosen["alpha"])
    features = feature_arrays[feature_kind]
    evaluation_model = _fit_membrane_ridge(
        features,
        data["target"],
        data["sample_weight"],
        development,
        alpha=alpha,
        feature_kind=feature_kind,
    )
    test_prediction = evaluation_model.predict(features[test])
    protherm_model = load_auxiliary_checkpoint(
        checkpoint_dir / "protherm_ddg_head.pt", torch_device
    )
    protherm_prediction = _torch_predictions(
        protherm_model,
        (data["delta"][test],),
        lambda x: protherm_model(x),
        batch_size=batch_size,
        device=torch_device,
    )
    metrics = {
        "schema": "protein-stabilizer.membrane-ddg-training.v1",
        "seed": seed,
        "selection": (
            "leave-one-protein-out on the source four-protein training split; "
            "within 0.01 macro Spearman "
            "choose the lowest-dimensional representation"
        ),
        "selected_feature_kind": feature_kind,
        "selected_alpha": alpha,
        "development_rows": int(len(development)),
        "development_experimental_rows": int((~data["is_reverse"][development]).sum()),
        "test_rows": int(len(test)),
        "test_policy": "unseen 1AFO/2K73 experimental forward mutations only",
        "validation_candidates": candidates,
        "test": _metrics_by_protein(
            data["target"][test], test_prediction, data["protein_id"][test]
        ),
        "test_pretrained_baseline": _metrics_by_protein(
            data["target"][test], base_ddg[test], data["protein_id"][test]
        ),
        "test_protherm_baseline": _metrics_by_protein(
            data["target"][test], protherm_prediction, data["protein_id"][test]
        ),
        "sign_convention": "negative project ddG is stabilizing",
    }
    test_macro = float(metrics["test"]["macro_protein_spearman"])
    baseline_macro = float(metrics["test_pretrained_baseline"]["macro_protein_spearman"])
    test_rmse = float(metrics["test"]["overall"]["rmse"])
    baseline_rmse = float(metrics["test_pretrained_baseline"]["overall"]["rmse"])
    passes_gate = test_macro > baseline_macro and test_rmse <= 1.05 * baseline_rmse
    metrics["deployment_gate"] = {
        "passed": passes_gate,
        "policy": (
            "must improve untouched macro protein Spearman and remain within "
            "5% of pretrained RMSE"
        ),
        "decision": (
            "eligible as default membrane score"
            if passes_gate
            else "rejected; retain pretrained ESM-C score as default"
        ),
    }
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "schema": "protein-stabilizer.membrane-ddg-ridge.v1",
            "role": "held-out-evaluation",
            "model": evaluation_model,
            "feature_kind": feature_kind,
            "metrics": metrics,
        },
        checkpoint_dir / "membrane_ddg_eval.joblib",
    )

    deployment = np.flatnonzero(data["topology"] == "alpha_helical")
    deployment_model = _fit_membrane_ridge(
        features,
        data["target"],
        data["sample_weight"],
        deployment,
        alpha=alpha,
        feature_kind=feature_kind,
    )
    deployment_payload = {
        "schema": "protein-stabilizer.membrane-ddg-ridge.v1",
        "role": "application-refit-candidate",
        "model": deployment_model,
        "feature_kind": feature_kind,
        "training_proteins": sorted(set(data["protein_id"][deployment].tolist())),
        "training_rows": int(len(deployment)),
        "deployment_gate": metrics["deployment_gate"],
        "metrics": metrics,
    }
    joblib.dump(deployment_payload, checkpoint_dir / "membrane_ddg_candidate.joblib")
    default_path = checkpoint_dir / "membrane_ddg.joblib"
    if passes_gate:
        deployment_payload["role"] = "application-refit"
        joblib.dump(deployment_payload, default_path)
    else:
        default_path.unlink(missing_ok=True)
    _save_json(checkpoint_dir / "membrane_ddg_metrics.json", metrics)
    return metrics


def gpcr_dtm_adapter_features(
    feature_kind: str,
    delta: np.ndarray,
    latent: np.ndarray,
    base_ddg: np.ndarray,
    mptherm_dtm: np.ndarray,
) -> np.ndarray:
    """Build inputs understood by the compact GPCR delta-Tm adapter."""

    if feature_kind in {"raw_delta", "base_latent", "base_latent_ddg"}:
        return membrane_adapter_features(feature_kind, delta, latent, base_ddg)
    if feature_kind == "mptherm_dtm":
        return mptherm_dtm.astype(np.float32)[:, None]
    if feature_kind == "base_latent_mptherm":
        return np.concatenate(
            [latent.astype(np.float32), mptherm_dtm.astype(np.float32)[:, None]],
            axis=1,
        )
    raise ValueError(f"unknown GPCR delta-Tm feature kind {feature_kind!r}")


def train_gpcr_dtm_adapter(
    feature_dir: Path,
    checkpoint_dir: Path,
    *,
    seed: int = 20260715,
    device: str = "cuda",
    batch_size: int = 512,
) -> dict[str, object]:
    """Fit and gate a small GPCR-specific delta-Tm transfer model."""

    set_reproducible_seed(seed)
    feature_dir = Path(feature_dir)
    checkpoint_dir = Path(checkpoint_dir)
    torch_device = torch.device(device if torch.cuda.is_available() else "cpu")
    data = _load_transfer_features(feature_dir / "gpcr_tm.h5")
    base_model = load_scoring_checkpoint(checkpoint_dir, torch_device)
    _, latent, base_ddg = _base_representation(
        base_model,
        data["delta"],
        batch_size=batch_size,
        device=torch_device,
    )
    mptherm_model = load_auxiliary_checkpoint(
        checkpoint_dir / "mptherm_dtm_head.pt", torch_device
    )
    mptherm_dtm = _torch_predictions(
        mptherm_model,
        (data["delta"],),
        lambda x: mptherm_model(x),
        batch_size=batch_size,
        device=torch_device,
    )
    feature_arrays = {
        name: gpcr_dtm_adapter_features(
            name, data["delta"], latent, base_ddg, mptherm_dtm
        )
        for name in (
            "raw_delta",
            "base_latent",
            "base_latent_ddg",
        )
    }
    development = np.flatnonzero(
        (data["source_split"] == "train")
        & ~data["official_test_site_overlap"]
    )
    test = np.flatnonzero(data["source_split"] == "test")
    if len(development) != 82 or len(test) != 12:
        raise RuntimeError(
            f"unexpected GPCR-tm evaluation split sizes {len(development)}/{len(test)}"
        )
    development_proteins = sorted(set(data["protein_id"][development].tolist()))

    candidates: list[dict[str, object]] = []
    for feature_kind, features in feature_arrays.items():
        for alpha in (0.01, 0.1, 1.0, 10.0, 100.0, 1000.0):
            oof_prediction = np.full(len(development), np.nan, dtype=np.float32)
            fold_metrics: dict[str, dict[str, float]] = {}
            for held_out in development_proteins:
                fit_indices = development[data["protein_id"][development] != held_out]
                validation_indices = development[
                    data["protein_id"][development] == held_out
                ]
                model = _fit_membrane_ridge(
                    features,
                    data["target"],
                    data["sample_weight"],
                    fit_indices,
                    alpha=alpha,
                    feature_kind=feature_kind,
                )
                prediction = model.predict(features[validation_indices])
                oof_prediction[
                    np.flatnonzero(data["protein_id"][development] == held_out)
                ] = prediction
                fold_metrics[held_out] = regression_metrics(
                    data["target"][validation_indices], prediction
                )
            oof = _metrics_by_protein(
                data["target"][development],
                oof_prediction,
                data["protein_id"][development],
            )
            candidates.append(
                {
                    "feature_kind": feature_kind,
                    "feature_dimension": int(features.shape[1]),
                    "alpha": alpha,
                    "oof": oof,
                    "folds": fold_metrics,
                }
            )
    finite_candidates = [
        value
        for value in candidates
        if np.isfinite(float(value["oof"]["macro_protein_spearman"]))
    ]
    if not finite_candidates:
        raise RuntimeError("GPCR delta-Tm adapter produced no finite validation metric")
    best_score = max(
        float(value["oof"]["macro_protein_spearman"])
        for value in finite_candidates
    )
    eligible = [
        value
        for value in finite_candidates
        if float(value["oof"]["macro_protein_spearman"]) >= best_score - 0.01
    ]
    chosen = min(
        eligible,
        key=lambda value: (
            int(value["feature_dimension"]),
            float(value["oof"]["overall"]["rmse"]),
            -float(value["alpha"]),
        ),
    )
    feature_kind = str(chosen["feature_kind"])
    alpha = float(chosen["alpha"])
    features = feature_arrays[feature_kind]
    evaluation_model = _fit_membrane_ridge(
        features,
        data["target"],
        data["sample_weight"],
        development,
        alpha=alpha,
        feature_kind=feature_kind,
    )
    test_prediction = evaluation_model.predict(features[test])
    metrics = {
        "schema": "protein-stabilizer.gpcr-dtm-training.v1",
        "seed": seed,
        "selection": (
            "leave-one-protein-out on official training rows after removing "
            "official-test sites; within 0.01 macro Spearman choose the "
            "lowest-dimensional representation; MPTherm-derived features are "
            "excluded because its source overlaps GPCR-tm training rows"
        ),
        "selected_feature_kind": feature_kind,
        "selected_alpha": alpha,
        "development_rows": int(len(development)),
        "development_proteins": development_proteins,
        "test_rows": int(len(test)),
        "test_policy": "official GPCR-tm test rows; no training substitutions at test sites",
        "validation_candidates": candidates,
        "test": _metrics_by_protein(
            data["target"][test], test_prediction, data["protein_id"][test]
        ),
        "test_mptherm_baseline": _metrics_by_protein(
            data["target"][test], mptherm_dtm[test], data["protein_id"][test]
        ),
        "test_pretrained_baseline": _metrics_by_protein(
            data["target"][test], -base_ddg[test], data["protein_id"][test]
        ),
        "sign_convention": "positive delta-Tm is stabilizing",
    }
    test_spearman = float(metrics["test"]["overall"]["spearman"])
    baseline_spearman = float(
        metrics["test_mptherm_baseline"]["overall"]["spearman"]
    )
    test_rmse = float(metrics["test"]["overall"]["rmse"])
    baseline_rmse = float(metrics["test_mptherm_baseline"]["overall"]["rmse"])
    passes_gate = (
        test_spearman > baseline_spearman and test_rmse <= 1.10 * baseline_rmse
    )
    metrics["deployment_gate"] = {
        "passed": passes_gate,
        "policy": (
            "must improve official-test Spearman over the MPTherm head and remain "
            "within 10% of its RMSE"
        ),
        "decision": (
            "eligible as GPCR delta-Tm feature"
            if passes_gate
            else "rejected; retain the MPTherm delta-Tm feature"
        ),
    }
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "schema": "protein-stabilizer.gpcr-dtm-ridge.v1",
            "role": "held-out-evaluation",
            "model": evaluation_model,
            "feature_kind": feature_kind,
            "metrics": metrics,
        },
        checkpoint_dir / "gpcr_dtm_eval.joblib",
    )
    application = np.flatnonzero(
        ~np.isin(data["gpcr_overlap_split"], ["val", "test"])
    )
    application_model = _fit_membrane_ridge(
        features,
        data["target"],
        data["sample_weight"],
        application,
        alpha=alpha,
        feature_kind=feature_kind,
    )
    application_payload = {
        "schema": "protein-stabilizer.gpcr-dtm-ridge.v1",
        "role": "application-refit-candidate",
        "model": application_model,
        "feature_kind": feature_kind,
        "training_proteins": sorted(set(data["protein_id"][application].tolist())),
        "training_rows": int(len(application)),
        "benchmark_excluded_rows": int(len(data["target"]) - len(application)),
        "deployment_gate": metrics["deployment_gate"],
        "metrics": metrics,
    }
    joblib.dump(application_payload, checkpoint_dir / "gpcr_dtm_candidate.joblib")
    default_path = checkpoint_dir / "gpcr_dtm.joblib"
    if passes_gate:
        application_payload["role"] = "application-refit"
        joblib.dump(application_payload, default_path)
    else:
        default_path.unlink(missing_ok=True)
    _save_json(checkpoint_dir / "gpcr_dtm_metrics.json", metrics)
    return metrics


def train_transfer_heads(
    feature_dir: Path,
    checkpoint_dir: Path,
    *,
    seed: int = 20260715,
    device: str = "cuda",
) -> dict[str, object]:
    """Train thermodynamic transfer heads without mixing endpoint types."""

    feature_dir = Path(feature_dir)
    checkpoint_dir = Path(checkpoint_dir)
    torch_device = torch.device(device if torch.cuda.is_available() else "cpu")
    datasets = {
        "protherm_ddg": (_load_transfer_features(feature_dir / "protherm.h5"), "validation"),
        "mptherm_dtm": (_load_transfer_features(feature_dir / "mptherm.h5"), "val"),
    }
    metrics: dict[str, object] = {
        "schema": "protein-stabilizer.transfer-training.v4",
        "seed": seed,
    }
    for name, (data, validation_name) in datasets.items():
        target_kind = "ddg" if name == "protherm_ddg" else "delta_tm"
        auxiliary = _train_auxiliary_head(
            data,
            checkpoint_dir / "single_head.pt",
            checkpoint_dir / f"{name}_head.pt",
            target_kind=target_kind,
            validation_name=validation_name,
            seed=seed,
            device=torch_device,
        )
        if name == "mptherm_dtm":
            auxiliary["quarantined_rows"] = int(
                (data["split"] == "quarantine").sum()
            )
        metrics[name] = auxiliary
    metrics["mcsm_membrane_ddg"] = train_membrane_ddg_adapter(
        feature_dir,
        checkpoint_dir,
        seed=seed,
        device=device,
    )
    metrics["gpcr_dtm"] = train_gpcr_dtm_adapter(
        feature_dir,
        checkpoint_dir,
        seed=seed,
        device=device,
    )
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    _save_json(checkpoint_dir / "transfer_metrics.json", metrics)
    return metrics


def _load_double(
    path: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    with h5py.File(path, "r") as handle:
        return (
            np.asarray(handle["single_delta"], dtype=np.float16),
            np.asarray(handle["joint_delta"], dtype=np.float16),
            np.asarray(handle["target"], dtype=np.float32),
            np.asarray(handle["true_epistasis"], dtype=np.float32),
            np.asarray(handle["has_true_epistasis"], dtype=bool),
        )


def train_epistasis_head(
    feature_dir: Path,
    checkpoint_dir: Path,
    *,
    seed: int = 20260715,
    epochs: int = 25,
    patience: int = 5,
    batch_size: int = 512,
    learning_rate: float = 8e-4,
    weight_decay: float = 1e-4,
    device: str = "cuda",
) -> dict[str, object]:
    set_reproducible_seed(seed)
    torch_device = torch.device(device if torch.cuda.is_available() else "cpu")
    feature_dir = Path(feature_dir)
    checkpoint_dir = Path(checkpoint_dir)
    train_single, train_joint, train_target, train_true_epi, train_has_epi = _load_double(
        feature_dir / "double_train.h5"
    )
    val_single, val_joint, val_target, val_true_epi, val_has_epi = _load_double(
        feature_dir / "double_val.h5"
    )
    test_single, test_joint, test_target, test_true_epi, test_has_epi = _load_double(
        feature_dir / "double_test.h5"
    )
    single_head = load_single_checkpoint(checkpoint_dir / "single_head.pt", torch_device)
    config = EpistasisConfig(embedding_dim=train_single.shape[-1])
    model = MultiMutationHead(single_head, config).to(torch_device)
    model.freeze_single_head()
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    loss_function = torch.nn.HuberLoss(delta=1.0)
    rng = np.random.default_rng(seed)
    best_state: dict[str, torch.Tensor] | None = None
    best_spearman = -math.inf
    best_epoch = 0
    stale = 0
    history: list[dict[str, object]] = []
    started = time.monotonic()

    for epoch in range(1, epochs + 1):
        model.train()
        model.single_head.eval()
        shuffled = rng.permutation(len(train_target))
        total_loss = 0.0
        seen = 0
        for start in range(0, len(shuffled), batch_size):
            indices = shuffled[start : start + batch_size]
            singles = torch.from_numpy(train_single[indices].astype(np.float32)).to(
                torch_device
            )
            joints = torch.from_numpy(train_joint[indices].astype(np.float32)).to(
                torch_device
            )
            target = torch.from_numpy(train_target[indices]).to(torch_device)
            optimizer.zero_grad(set_to_none=True)
            prediction, _, epistasis = model(singles, joints)
            has_epistasis = torch.from_numpy(train_has_epi[indices]).to(torch_device)
            true_epistasis = torch.from_numpy(train_true_epi[indices]).to(torch_device)
            total_loss_component = loss_function(prediction, target)
            epistasis_loss_component = loss_function(
                epistasis[has_epistasis], true_epistasis[has_epistasis]
            )
            loss = epistasis_loss_component + 0.25 * total_loss_component
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach()) * len(indices)
            seen += len(indices)
        validation_prediction = _torch_predictions(
            model,
            (val_single, val_joint),
            lambda singles, joints: model(singles, joints)[0],
            batch_size=batch_size,
            device=torch_device,
        )
        validation = regression_metrics(val_target, validation_prediction)
        score = validation["spearman"]
        history.append(
            {
                "epoch": epoch,
                "train_huber": total_loss / seen,
                "validation": validation,
            }
        )
        print(
            f"double epoch={epoch:02d} train_huber={total_loss / seen:.4f} "
            f"val_rmse={validation['rmse']:.4f} val_spearman={score:.4f}",
            flush=True,
        )
        if np.isfinite(score) and score > best_spearman + 1e-5:
            best_spearman = score
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break
    if best_state is None:
        raise RuntimeError("epistasis training produced no valid validation model")
    model.load_state_dict(best_state)

    def predictions(
        singles: np.ndarray, joints: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        total = _torch_predictions(
            model,
            (singles, joints),
            lambda x, y: model(x, y)[0],
            batch_size=batch_size,
            device=torch_device,
        )
        additive = _torch_predictions(
            model,
            (singles, joints),
            lambda x, y: model(x, y)[1],
            batch_size=batch_size,
            device=torch_device,
        )
        epistasis = _torch_predictions(
            model,
            (singles, joints),
            lambda x, y: model(x, y)[2],
            batch_size=batch_size,
            device=torch_device,
        )
        return total, additive, epistasis

    val_prediction, val_additive, val_epistasis = predictions(val_single, val_joint)
    test_prediction, test_additive, test_epistasis = predictions(test_single, test_joint)
    scoring_head = load_scoring_checkpoint(checkpoint_dir, torch_device)

    def deployed_additive(deltas: np.ndarray) -> np.ndarray:
        return _torch_predictions(
            scoring_head,
            (deltas,),
            lambda values: scoring_head(
                values.reshape(-1, values.shape[-1])
            ).reshape(values.shape[:2]).sum(dim=1),
            batch_size=batch_size,
            device=torch_device,
        )

    val_deployed_additive = deployed_additive(val_single)
    test_deployed_additive = deployed_additive(test_single)
    val_deployed = val_deployed_additive + val_epistasis
    test_deployed = test_deployed_additive + test_epistasis
    metrics = {
        "schema": "protein-stabilizer.epistasis-training.v2",
        "seed": seed,
        "best_epoch": best_epoch,
        "elapsed_seconds": time.monotonic() - started,
        "train_rows": int(len(train_target)),
        "validation": regression_metrics(val_target, val_prediction),
        "validation_additive_baseline": regression_metrics(val_target, val_additive),
        "validation_true_epistasis": regression_metrics(
            val_true_epi[val_has_epi], val_epistasis[val_has_epi]
        ),
        "test": regression_metrics(test_target, test_prediction),
        "test_additive_baseline": regression_metrics(test_target, test_additive),
        "test_true_epistasis": regression_metrics(
            test_true_epi[test_has_epi], test_epistasis[test_has_epi]
        ),
        "validation_deployed": regression_metrics(val_target, val_deployed),
        "validation_deployed_additive_baseline": regression_metrics(
            val_target, val_deployed_additive
        ),
        "test_deployed": regression_metrics(test_target, test_deployed),
        "test_deployed_additive_baseline": regression_metrics(
            test_target, test_deployed_additive
        ),
        "history": history,
    }
    torch.save(
        {
            "schema": "protein-stabilizer.multi-head.v1",
            "single_config": model.single_head.config_dict(),
            "epistasis_config": asdict(config),
            "state_dict": model.cpu().state_dict(),
            "metrics": metrics,
        },
        checkpoint_dir / "multi_head.pt",
    )
    _save_json(checkpoint_dir / "multi_metrics.json", metrics)
    return metrics


def load_multi_checkpoint(path: Path, device: str | torch.device = "cpu") -> MultiMutationHead:
    payload = torch.load(path, map_location=device, weights_only=False)
    if payload.get("schema") != "protein-stabilizer.multi-head.v1":
        raise RuntimeError("multi-head checkpoint schema mismatch")
    single = SingleMutationHead(SingleHeadConfig(**payload["single_config"]))
    model = MultiMutationHead(single, EpistasisConfig(**payload["epistasis_config"]))
    model.load_state_dict(payload["state_dict"])
    return model.to(device).eval()


def _gpcr_metrics_by_assay(
    target: np.ndarray,
    prediction: np.ndarray,
    assay: np.ndarray,
) -> dict[str, object]:
    by_assay = {
        str(name): regression_metrics(target[assay == name], prediction[assay == name])
        for name in sorted(set(assay.tolist()))
    }
    finite_spearman = [
        metrics["spearman"]
        for metrics in by_assay.values()
        if np.isfinite(metrics["spearman"])
    ]
    return {
        "overall": regression_metrics(target, prediction),
        "macro_within_assay_spearman": (
            float(np.mean(finite_spearman)) if finite_spearman else math.nan
        ),
        "by_assay": by_assay,
    }


def _center_within_assay(
    features: np.ndarray,
    target: np.ndarray,
    assay: np.ndarray,
    indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    centered_features = np.empty_like(features[indices], dtype=np.float32)
    centered_target = np.empty_like(target[indices], dtype=np.float32)
    selected_assays = assay[indices]
    for name in sorted(set(selected_assays.tolist())):
        local = np.flatnonzero(selected_assays == name)
        values = features[indices[local]].astype(np.float32)
        outcomes = target[indices[local]].astype(np.float32)
        centered_features[local] = values - values.mean(axis=0, keepdims=True)
        centered_target[local] = outcomes - outcomes.mean()
    return centered_features, centered_target


def train_gpcr_calibration(
    feature_dir: Path,
    checkpoint_dir: Path,
    *,
    device: str = "cuda",
    batch_size: int = 512,
) -> dict[str, object]:
    feature_dir = Path(feature_dir)
    checkpoint_dir = Path(checkpoint_dir)
    torch_device = torch.device(device if torch.cuda.is_available() else "cpu")
    with h5py.File(feature_dir / "gpcr.h5", "r") as handle:
        delta = np.asarray(handle["delta"], dtype=np.float16)
        target = np.asarray(handle["target"], dtype=np.float32)
        split = np.asarray(handle["split"].asstr()[:])
        assay = np.asarray(handle["assay_id"].asstr()[:])
        protein_id = np.asarray(handle["protein_id"].asstr()[:])
        uniprot_id = np.asarray(handle["uniprot_id"].asstr()[:])
        site = np.asarray(handle["site_id"].asstr()[:])
        split_seed = int(handle.attrs["split_seed"])
        source_sha256 = str(handle.attrs["source_sha256"])
    model = load_scoring_checkpoint(checkpoint_dir, torch_device)
    model.eval()
    representation, latent, base_ddg = _base_representation(
        model,
        delta,
        batch_size=batch_size,
        device=torch_device,
    )
    protherm_model = load_auxiliary_checkpoint(
        checkpoint_dir / "protherm_ddg_head.pt", torch_device
    )
    mptherm_model = load_auxiliary_checkpoint(
        checkpoint_dir / "mptherm_dtm_head.pt", torch_device
    )
    protherm_ddg = _torch_predictions(
        protherm_model,
        (delta,),
        lambda x: protherm_model(x),
        batch_size=batch_size,
        device=torch_device,
    )
    mptherm_dtm = _torch_predictions(
        mptherm_model,
        (delta,),
        lambda x: mptherm_model(x),
        batch_size=batch_size,
        device=torch_device,
    )
    feature_sets = {
        "latent_base": representation,
        "latent_protherm": np.concatenate(
            [representation, protherm_ddg[:, None]], axis=1
        ),
        "latent_mptherm": np.concatenate(
            [representation, mptherm_dtm[:, None]], axis=1
        ),
        "latent_both": np.concatenate(
            [representation, protherm_ddg[:, None], mptherm_dtm[:, None]], axis=1
        ),
        "thermodynamic_scores": np.stack(
            [
                base_ddg,
                representation[:, -1],
                protherm_ddg,
                mptherm_dtm,
            ],
            axis=1,
        ),
    }
    gpcr_dtm: np.ndarray | None = None
    gpcr_dtm_path = checkpoint_dir / "gpcr_dtm.joblib"
    if gpcr_dtm_path.exists():
        gpcr_dtm_payload = joblib.load(gpcr_dtm_path)
        if gpcr_dtm_payload.get("schema") != "protein-stabilizer.gpcr-dtm-ridge.v1":
            raise RuntimeError("GPCR delta-Tm checkpoint schema mismatch")
        gpcr_dtm_features = gpcr_dtm_adapter_features(
            str(gpcr_dtm_payload["feature_kind"]),
            delta,
            latent,
            base_ddg,
            mptherm_dtm,
        )
        gpcr_dtm = gpcr_dtm_payload["model"].predict(gpcr_dtm_features)
        feature_sets.update(
            {
                "latent_gpcr_dtm": np.concatenate(
                    [representation, gpcr_dtm[:, None]], axis=1
                ),
                "latent_gpcr_thermo": np.concatenate(
                    [
                        representation,
                        protherm_ddg[:, None],
                        mptherm_dtm[:, None],
                        gpcr_dtm[:, None],
                    ],
                    axis=1,
                ),
                "thermodynamic_scores": np.stack(
                    [
                        base_ddg,
                        representation[:, -1],
                        protherm_ddg,
                        mptherm_dtm,
                        gpcr_dtm,
                    ],
                    axis=1,
                ),
            }
        )
    membrane_ddg: np.ndarray | None = None
    membrane_path = checkpoint_dir / "membrane_ddg.joblib"
    if membrane_path.exists():
        membrane_payload = joblib.load(membrane_path)
        if membrane_payload.get("schema") != "protein-stabilizer.membrane-ddg-ridge.v1":
            raise RuntimeError("membrane ddG checkpoint schema mismatch")
        membrane_features = membrane_adapter_features(
            str(membrane_payload["feature_kind"]), delta, latent, base_ddg
        )
        membrane_ddg = membrane_payload["model"].predict(membrane_features)
        all_columns = [
            representation,
            protherm_ddg[:, None],
            mptherm_dtm[:, None],
        ]
        score_columns = [
            base_ddg,
            representation[:, -1],
            protherm_ddg,
            mptherm_dtm,
        ]
        if gpcr_dtm is not None:
            all_columns.append(gpcr_dtm[:, None])
            score_columns.append(gpcr_dtm)
        all_columns.append(membrane_ddg[:, None])
        score_columns.append(membrane_ddg)
        feature_sets.update(
            {
                "latent_membrane": np.concatenate(
                    [representation, membrane_ddg[:, None]], axis=1
                ),
                "latent_all": np.concatenate(all_columns, axis=1),
                "thermodynamic_scores": np.stack(score_columns, axis=1),
            }
        )
    train_indices = np.flatnonzero(split == "train")
    val_indices = np.flatnonzero(split == "val")
    test_indices = np.flatnonzero(split == "test")
    if set(site[train_indices]) & set(site[val_indices]) or set(site[train_indices]) & set(
        site[test_indices]
    ) or set(site[val_indices]) & set(site[test_indices]):
        raise RuntimeError("GPCR mutation-site leakage detected")

    development_indices = np.concatenate([train_indices, val_indices])
    candidates: list[dict[str, object]] = []
    # The benchmark has only three receptors. Selecting among high-dimensional
    # latent spaces on 26 validation rows is unstable and does not survive the
    # leave-one-receptor-out audit. Calibrate only the four independently
    # trained thermodynamic scores; all still derive from the same ESM-C deltas.
    calibration_feature_sets = {
        "thermodynamic_scores": feature_sets["thermodynamic_scores"]
    }
    for feature_name, features in calibration_feature_sets.items():
        centered_features, centered_target = _center_within_assay(
            features, target, assay, train_indices
        )
        component_counts: list[int | None] = [None, 2, 4, 8, 16]
        component_counts = [
            value
            for value in component_counts
            if value is None or value <= features.shape[1]
        ]
        for components in component_counts:
            for alpha in (0.1, 1.0, 10.0, 100.0, 1000.0):
                pipeline = _ridge_pipeline(components, alpha, split_seed)
                pipeline.set_params(ridge__fit_intercept=False)
                pipeline.fit(centered_features, centered_target)
                prediction = pipeline.predict(features[val_indices])
                validation = _gpcr_metrics_by_assay(
                    target[val_indices], prediction, assay[val_indices]
                )
                validation_score = float(validation["macro_within_assay_spearman"])
                candidate = {
                    "feature_set": feature_name,
                    "components": components,
                    "alpha": alpha,
                    "effective_dimension": (
                        int(components) if components is not None else int(features.shape[1])
                    ),
                    "validation": validation,
                    "validation_score": validation_score,
                }
                candidates.append(candidate)
    finite_candidates = [
        candidate
        for candidate in candidates
        if np.isfinite(float(candidate["validation_score"]))
    ]
    if not finite_candidates:
        raise RuntimeError("GPCR calibration could not select a ridge penalty")
    best_validation = max(float(value["validation_score"]) for value in finite_candidates)
    # The validation set is necessarily small. Treat models within 0.01
    # Spearman as tied and choose the lowest-dimensional one to control variance.
    eligible = [
        value
        for value in finite_candidates
        if float(value["validation_score"]) >= best_validation - 0.01
    ]
    chosen = min(
        eligible,
        key=lambda value: (
            int(value["effective_dimension"]),
            -float(value["validation_score"]),
            -float(value["alpha"]),
        ),
    )
    selected_validation = chosen["validation"]
    compact_candidates = [
        {
            "feature_set": value["feature_set"],
            "components": value["components"],
            "alpha": value["alpha"],
            "effective_dimension": value["effective_dimension"],
            "validation_score": value["validation_score"],
            "validation_overall_spearman": value["validation"]["overall"]["spearman"],
        }
        for value in candidates
    ]
    best_feature_set = str(chosen["feature_set"])
    best_components = chosen["components"]
    best_alpha = float(chosen["alpha"])
    features = feature_sets[best_feature_set]
    selected = _ridge_pipeline(best_components, best_alpha, split_seed)
    selected.set_params(ridge__fit_intercept=False)
    fit_features, fit_target = _center_within_assay(
        features, target, assay, development_indices
    )
    selected.fit(fit_features, fit_target)
    test_prediction = selected.predict(features[test_indices])
    base_prediction = -base_ddg[test_indices]
    receptor_heldout_prediction = np.full(len(target), np.nan, dtype=np.float32)
    receptor_heldout_exclusions: dict[str, int] = {}
    mptherm_data = _load_transfer_features(feature_dir / "mptherm.h5")
    for held_out in sorted(set(protein_id.tolist())):
        heldout_fit = np.flatnonzero(protein_id != held_out)
        heldout_test = np.flatnonzero(protein_id == held_out)
        heldout_accessions = sorted(set(uniprot_id[heldout_test].tolist()))
        if len(heldout_accessions) != 1:
            raise RuntimeError(
                f"GPCR receptor {held_out} maps to {heldout_accessions}"
            )
        heldout_accession = heldout_accessions[0]
        fold_mptherm = {name: value.copy() for name, value in mptherm_data.items()}
        upstream_heldout = fold_mptherm["protein_id"] == heldout_accession
        receptor_heldout_exclusions[str(held_out)] = int(upstream_heldout.sum())
        fold_mptherm["split"][upstream_heldout] = "quarantine"
        with TemporaryDirectory(prefix="protein-stabilizer-gpcr-holdout-") as temporary:
            fold_checkpoint = Path(temporary) / "mptherm_head.pt"
            _train_auxiliary_head(
                fold_mptherm,
                checkpoint_dir / "single_head.pt",
                fold_checkpoint,
                target_kind="delta_tm",
                validation_name="val",
                seed=split_seed,
                device=torch_device,
            )
            fold_mptherm_model = load_auxiliary_checkpoint(
                fold_checkpoint, torch_device
            )
            fold_mptherm_prediction = _torch_predictions(
                fold_mptherm_model,
                (delta,),
                lambda x: fold_mptherm_model(x),
                batch_size=batch_size,
                device=torch_device,
            )
        heldout_feature_sets = {
            "latent_base": representation,
            "latent_protherm": np.concatenate(
                [representation, protherm_ddg[:, None]], axis=1
            ),
            "latent_mptherm": np.concatenate(
                [representation, fold_mptherm_prediction[:, None]], axis=1
            ),
            "latent_both": np.concatenate(
                [
                    representation,
                    protherm_ddg[:, None],
                    fold_mptherm_prediction[:, None],
                ],
                axis=1,
            ),
            "thermodynamic_scores": np.stack(
                [
                    base_ddg,
                    representation[:, -1],
                    protherm_ddg,
                    fold_mptherm_prediction,
                ],
                axis=1,
            ),
        }
        if best_feature_set not in heldout_feature_sets:
            raise RuntimeError(
                f"receptor-held-out diagnostic cannot build {best_feature_set}"
            )
        heldout_features_full = heldout_feature_sets[best_feature_set]
        heldout_model = _ridge_pipeline(best_components, best_alpha, split_seed)
        heldout_model.set_params(ridge__fit_intercept=False)
        heldout_features, heldout_target = _center_within_assay(
            heldout_features_full, target, assay, heldout_fit
        )
        heldout_model.fit(heldout_features, heldout_target)
        receptor_heldout_prediction[heldout_test] = heldout_model.predict(
            heldout_features_full[heldout_test]
        )
    receptor_heldout = _gpcr_metrics_by_assay(
        target, receptor_heldout_prediction, assay
    )
    receptor_heldout_baseline = _gpcr_metrics_by_assay(target, -base_ddg, assay)
    receptor_macro = float(receptor_heldout["macro_within_assay_spearman"])
    receptor_baseline_macro = float(
        receptor_heldout_baseline["macro_within_assay_spearman"]
    )
    # The endpoint is residual binding after heating, not signed ddG, and the
    # new-receptor diagnostic is weak. Keep this score as a small ranking prior.
    gpcr_weight = 0.10 if receptor_macro > receptor_baseline_macro else 0.0
    general_weight = 1.0 - gpcr_weight
    metrics = {
        "schema": "protein-stabilizer.gpcr-calibration.v5",
        "split_seed": split_seed,
        "source_sha256": source_sha256,
        "train_rows": int(len(train_indices)),
        "validation_rows": int(len(val_indices)),
        "test_rows": int(len(test_indices)),
        "train_sites": int(len(set(site[train_indices]))),
        "validation_sites": int(len(set(site[val_indices]))),
        "test_sites": int(len(set(site[test_indices]))),
        "selection": (
            "thermodynamic-score calibration only; site-held-out validation; "
            "within 0.01 macro Spearman choose the lowest effective dimension"
        ),
        "selected_feature_set": best_feature_set,
        "selected_alpha": best_alpha,
        "selected_components": best_components,
        "selected_validation": selected_validation,
        "selection_candidates": compact_candidates,
        "test": _gpcr_metrics_by_assay(
            target[test_indices], test_prediction, assay[test_indices]
        ),
        "test_pretrained_baseline": _gpcr_metrics_by_assay(
            target[test_indices], base_prediction, assay[test_indices]
        ),
        "leave_one_receptor_out": receptor_heldout,
        "leave_one_receptor_out_pretrained_baseline": receptor_heldout_baseline,
        "leave_one_receptor_out_note": (
            "post-selection diagnostic using the site-held-out-selected model "
            "configuration; each fold retrains the MPTherm head after excluding "
            "every row for the held-out receptor"
        ),
        "leave_one_receptor_out_upstream_rows_excluded": receptor_heldout_exclusions,
    }
    if membrane_ddg is not None:
        metrics["test_membrane_baseline"] = _gpcr_metrics_by_assay(
            target[test_indices], -membrane_ddg[test_indices], assay[test_indices]
        )
    if gpcr_dtm is not None:
        metrics["test_gpcr_dtm_baseline"] = _gpcr_metrics_by_assay(
            target[test_indices], gpcr_dtm[test_indices], assay[test_indices]
        )
    feature_order = {
        "latent_base": ["single_latent", "pretrained_ddg", "delta_l2_norm"],
        "latent_protherm": [
            "single_latent",
            "pretrained_ddg",
            "delta_l2_norm",
            "protherm_ddg",
        ],
        "latent_mptherm": [
            "single_latent",
            "pretrained_ddg",
            "delta_l2_norm",
            "mptherm_delta_tm",
        ],
        "latent_both": [
            "single_latent",
            "pretrained_ddg",
            "delta_l2_norm",
            "protherm_ddg",
            "mptherm_delta_tm",
        ],
        "latent_gpcr_dtm": [
            "single_latent",
            "pretrained_ddg",
            "delta_l2_norm",
            "gpcr_delta_tm",
        ],
        "latent_gpcr_thermo": [
            "single_latent",
            "pretrained_ddg",
            "delta_l2_norm",
            "protherm_ddg",
            "mptherm_delta_tm",
            "gpcr_delta_tm",
        ],
        "latent_membrane": [
            "single_latent",
            "pretrained_ddg",
            "delta_l2_norm",
            "membrane_ddg",
        ],
    }
    thermodynamic_order = [
        "pretrained_ddg",
        "delta_l2_norm",
        "protherm_ddg",
        "mptherm_delta_tm",
    ]
    latent_all_order = [
        "single_latent",
        "pretrained_ddg",
        "delta_l2_norm",
        "protherm_ddg",
        "mptherm_delta_tm",
    ]
    if gpcr_dtm is not None:
        thermodynamic_order.append("gpcr_delta_tm")
        latent_all_order.append("gpcr_delta_tm")
    if membrane_ddg is not None:
        thermodynamic_order.append("membrane_ddg")
        latent_all_order.append("membrane_ddg")
        feature_order["latent_all"] = latent_all_order
    feature_order["thermodynamic_scores"] = thermodynamic_order
    joblib.dump(
        {
            "schema": "protein-stabilizer.gpcr-ridge.v5",
            "pipeline": selected,
            "feature_set": best_feature_set,
            "feature_order": feature_order[best_feature_set],
            "screening_weights": {
                "general_stability": general_weight,
                "gpcr_calibration": gpcr_weight,
            },
            "screening_policy": (
                "thermodynamic-first blend; weak leave-one-receptor-out evidence "
                "limits the GPCR residual-binding prior to 10%"
            ),
            "output": "within-assay GPCR mutation stability ranking score",
            "metrics": metrics,
        },
        checkpoint_dir / "gpcr_calibration.joblib",
    )
    _save_json(checkpoint_dir / "gpcr_metrics.json", metrics)
    return metrics
