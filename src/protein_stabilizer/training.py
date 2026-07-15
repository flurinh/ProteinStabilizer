"""Training and held-out evaluation for stability heads."""

from __future__ import annotations

import copy
import json
import math
import random
import time
from dataclasses import asdict
from pathlib import Path
from typing import Callable, Iterable

import h5py
import joblib
import numpy as np
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
    device: str = "cuda",
) -> dict[str, object]:
    set_reproducible_seed(seed)
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
    model = SingleMutationHead(config).to(torch_device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    loss_function = torch.nn.HuberLoss(delta=1.0)
    rng = np.random.default_rng(seed)
    best_state: dict[str, torch.Tensor] | None = None
    best_spearman = -math.inf
    best_epoch = 0
    history: list[dict[str, object]] = []
    stale = 0
    started = time.monotonic()

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
            f"single epoch={epoch:02d} train_huber={total_loss / seen:.4f} "
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
        raise RuntimeError("single-mutant training produced no valid validation model")
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
    metrics = {
        "schema": "protein-stabilizer.single-training.v1",
        "seed": seed,
        "best_epoch": best_epoch,
        "elapsed_seconds": time.monotonic() - started,
        "train_rows": int(len(train_indices)),
        "validation_rows": int(len(val_indices)),
        "test_rows": int(len(test_target)),
        "validation": regression_metrics(target[val_indices], val_prediction),
        "test": regression_metrics(test_target, test_prediction),
        "history": history,
        "embedding_provenance": embedding_provenance,
        "train_source_sha256": train_source_sha256,
        "test_source_sha256": test_source_sha256,
    }
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema": "protein-stabilizer.single-head.v1",
            "config": asdict(config),
            "state_dict": model.cpu().state_dict(),
            "metrics": metrics,
        },
        checkpoint_dir / "single_head.pt",
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
    metrics = {
        "schema": "protein-stabilizer.epistasis-training.v1",
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
        site = np.asarray(handle["site_id"].asstr()[:])
        split_seed = int(handle.attrs["split_seed"])
        source_sha256 = str(handle.attrs["source_sha256"])
    model = load_single_checkpoint(checkpoint_dir / "single_head.pt", torch_device)
    model.eval()
    latent = _torch_predictions(
        model,
        (delta,),
        lambda x: model.latent(x),
        batch_size=batch_size,
        device=torch_device,
    )
    base_ddg = _torch_predictions(
        model,
        (delta,),
        lambda x: model(x),
        batch_size=batch_size,
        device=torch_device,
    )
    features = np.concatenate(
        [latent, base_ddg[:, None], np.linalg.norm(delta.astype(np.float32), axis=1)[:, None]],
        axis=1,
    )
    train_indices = np.flatnonzero(split == "train")
    val_indices = np.flatnonzero(split == "val")
    test_indices = np.flatnonzero(split == "test")
    if set(site[train_indices]) & set(site[val_indices]) or set(site[train_indices]) & set(
        site[test_indices]
    ) or set(site[val_indices]) & set(site[test_indices]):
        raise RuntimeError("GPCR mutation-site leakage detected")

    train_features, train_target = _center_within_assay(
        features, target, assay, train_indices
    )
    alphas = [0.1, 1.0, 10.0, 100.0, 1000.0]
    component_counts = [2, 4, 8, 16]
    candidates: list[dict[str, object]] = []
    best_alpha: float | None = None
    best_components: int | None = None
    best_score = -math.inf
    best_mae = math.inf
    for components in component_counts:
        for alpha in alphas:
            pipeline = Pipeline(
                [
                    ("scale", StandardScaler()),
                    ("pca", PCA(n_components=components, random_state=split_seed)),
                    ("ridge", Ridge(alpha=alpha, fit_intercept=False)),
                ]
            )
            pipeline.fit(train_features, train_target)
            prediction = pipeline.predict(features[val_indices])
            metrics = _gpcr_metrics_by_assay(
                target[val_indices], prediction, assay[val_indices]
            )
            score = float(metrics["macro_within_assay_spearman"])
            overall_mae = float(metrics["overall"]["mae"])
            candidates.append(
                {
                    "components": components,
                    "alpha": alpha,
                    "validation": metrics,
                }
            )
            comparable = score if np.isfinite(score) else -math.inf
            if comparable > best_score or (
                comparable == best_score and overall_mae < best_mae
            ):
                best_score = comparable
                best_mae = overall_mae
                best_alpha = alpha
                best_components = components
    if best_alpha is None or best_components is None:
        raise RuntimeError("GPCR calibration could not select a ridge penalty")
    selected = Pipeline(
        [
            ("scale", StandardScaler()),
            ("pca", PCA(n_components=best_components, random_state=split_seed)),
            ("ridge", Ridge(alpha=best_alpha, fit_intercept=False)),
        ]
    )
    fit_indices = np.concatenate([train_indices, val_indices])
    fit_features, fit_target = _center_within_assay(
        features, target, assay, fit_indices
    )
    selected.fit(fit_features, fit_target)
    test_prediction = selected.predict(features[test_indices])
    base_prediction = -base_ddg[test_indices]
    metrics = {
        "schema": "protein-stabilizer.gpcr-calibration.v1",
        "split_seed": split_seed,
        "source_sha256": source_sha256,
        "train_rows": int(len(train_indices)),
        "validation_rows": int(len(val_indices)),
        "test_rows": int(len(test_indices)),
        "train_sites": int(len(set(site[train_indices]))),
        "validation_sites": int(len(set(site[val_indices]))),
        "test_sites": int(len(set(site[test_indices]))),
        "selected_alpha": best_alpha,
        "selected_components": best_components,
        "selection_candidates": candidates,
        "test": _gpcr_metrics_by_assay(
            target[test_indices], test_prediction, assay[test_indices]
        ),
        "test_pretrained_baseline": _gpcr_metrics_by_assay(
            target[test_indices], base_prediction, assay[test_indices]
        ),
    }
    joblib.dump(
        {
            "schema": "protein-stabilizer.gpcr-ridge.v1",
            "pipeline": selected,
            "feature_order": ["single_latent", "pretrained_ddg", "delta_l2_norm"],
            "output": "within-assay GPCR mutation stability ranking score",
            "metrics": metrics,
        },
        checkpoint_dir / "gpcr_calibration.joblib",
    )
    _save_json(checkpoint_dir / "gpcr_metrics.json", metrics)
    return metrics
