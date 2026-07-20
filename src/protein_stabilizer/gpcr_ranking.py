"""Leakage-safe membrane pretraining for zero-shot GPCR mutation ranking."""

from __future__ import annotations

import copy
import json
import math
import time
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr

from .embeddings import file_sha256
from .models import DirectionalAssayConfig, DirectionalAssayHead
from .training import regression_metrics, set_reproducible_seed
from .v2_transfer import (
    TRANSFER_REPRESENTATION_SCHEMA,
    _derived_validation_split,
    _macro_spearman,
)


MEMBRANE_RANK_CHECKPOINT_SCHEMA = (
    "protein-stabilizer.zero-shot-membrane-rank-ensemble.v1"
)


def _load_representation(path: Path) -> dict[str, object]:
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
            "mutation": np.asarray(handle["mutation"].asstr()[:]),
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


def _adapter_forward(
    model: DirectionalAssayHead,
    data: dict[str, object],
    indices: np.ndarray,
    device: torch.device,
) -> torch.Tensor:
    return model(
        torch.from_numpy(
            np.asarray(data["latent"])[indices].astype(np.float32)
        ).to(device),
        torch.from_numpy(
            np.asarray(data["base_ddg"])[indices].astype(np.float32)
        ).to(device),
        torch.from_numpy(
            np.asarray(data["membrane"])[indices].astype(np.float32)
        ).to(device),
    )


def _predict(
    model: DirectionalAssayHead,
    data: dict[str, object],
    indices: np.ndarray,
    *,
    target_scale: float,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    output: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(indices), batch_size):
            selected = indices[start : start + batch_size]
            output.append(
                (
                    _adapter_forward(model, data, selected, device)
                    * target_scale
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


def _sample_higher_is_better_pairs(
    eligible_indices: np.ndarray,
    protein_id: np.ndarray,
    target: np.ndarray,
    rng: np.random.Generator,
    *,
    count: int,
    minimum_gap: float,
) -> tuple[np.ndarray, np.ndarray]:
    groups = [
        eligible_indices[protein_id[eligible_indices] == protein]
        for protein in sorted(
            set(protein_id[eligible_indices].tolist())
        )
    ]
    groups = [
        group
        for group in groups
        if len(group) > 1 and float(np.ptp(target[group])) >= minimum_gap
    ]
    higher: list[int] = []
    lower: list[int] = []
    attempts = 0
    while groups and len(higher) < count and attempts < 20 * max(1, count):
        group = groups[int(rng.integers(len(groups)))]
        left, right = rng.choice(group, size=2, replace=False)
        attempts += 1
        difference = float(target[left] - target[right])
        if abs(difference) < minimum_gap:
            continue
        higher.append(int(left if difference > 0 else right))
        lower.append(int(right if difference > 0 else left))
    return np.asarray(higher, dtype=np.int64), np.asarray(
        lower, dtype=np.int64
    )


def _metrics(
    target: np.ndarray,
    prediction: np.ndarray,
    protein_id: np.ndarray,
) -> dict[str, object]:
    result = regression_metrics(target, prediction)
    result["macro_protein_spearman"] = _macro_spearman(
        target, prediction, protein_id
    )
    return result


def _selection_score(
    metrics: dict[str, object],
    *,
    target_scale: float,
) -> float:
    overall = float(metrics["spearman"])
    macro = float(metrics["macro_protein_spearman"])
    mae = float(metrics["mae"])
    if not all(np.isfinite(value) for value in (overall, macro, mae)):
        return -math.inf
    return 0.45 * overall + 0.45 * macro - 0.10 * mae / target_scale


def _train_member(
    data: dict[str, object],
    train_indices: np.ndarray,
    validation_indices: np.ndarray,
    *,
    rank_weight: float,
    seed: int,
    epochs: int,
    minimum_epochs: int,
    patience: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    device: torch.device,
) -> tuple[DirectionalAssayHead, dict[str, object]]:
    if minimum_epochs < 1 or minimum_epochs > epochs:
        raise ValueError("minimum epochs must be within the training budget")
    target = np.asarray(data["target"], dtype=np.float32)
    protein_id = np.asarray(data["protein_id"])
    target_scale = max(float(np.std(target[train_indices])), 1e-6)
    scaled_target = target / target_scale
    config = DirectionalAssayConfig(
        latent_dim=np.asarray(data["latent"]).shape[1],
        membrane_dim=np.asarray(data["membrane"]).shape[1],
    )
    set_reproducible_seed(seed)
    rng = np.random.default_rng(seed)
    model = DirectionalAssayHead(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=epochs,
        eta_min=0.05 * learning_rate,
    )
    huber = torch.nn.HuberLoss(delta=1.0, reduction="none")
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
        total_huber = 0.0
        total_rank = 0.0
        total_loss = 0.0
        seen = 0
        for start in range(0, len(shuffled), batch_size):
            selected = shuffled[start : start + batch_size]
            values = torch.from_numpy(
                scaled_target[selected].astype(np.float32)
            ).to(device)
            weights = torch.from_numpy(
                np.asarray(data["sample_weight"])[selected].astype(
                    np.float32
                )
            ).to(device)
            optimizer.zero_grad(set_to_none=True)
            prediction = _adapter_forward(model, data, selected, device)
            regression_loss = (
                huber(prediction, values) * weights
            ).sum() / weights.sum()
            higher, lower = _sample_higher_is_better_pairs(
                train_indices,
                protein_id,
                target,
                rng,
                count=min(192, len(selected)),
                minimum_gap=1.0,
            )
            if len(higher):
                higher_prediction = _adapter_forward(
                    model, data, higher, device
                )
                lower_prediction = _adapter_forward(
                    model, data, lower, device
                )
                ranking_loss = torch.nn.functional.softplus(
                    -(higher_prediction - lower_prediction)
                ).mean()
            else:
                ranking_loss = torch.zeros((), device=device)
            loss = regression_loss + rank_weight * ranking_loss
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), 5.0
            )
            optimizer.step()
            optimizer_steps += 1
            examples_seen += len(selected)
            count = len(selected)
            total_huber += float(regression_loss.detach()) * count
            total_rank += float(ranking_loss.detach()) * count
            total_loss += float(loss.detach()) * count
            seen += count
        scheduler.step()
        validation_prediction = _predict(
            model,
            data,
            validation_indices,
            target_scale=target_scale,
            batch_size=batch_size,
            device=device,
        )
        validation = _metrics(
            target[validation_indices],
            validation_prediction,
            protein_id[validation_indices],
        )
        score = _selection_score(
            validation, target_scale=target_scale
        )
        history.append(
            {
                "epoch": epoch,
                "train_huber": total_huber / seen,
                "train_ranking": total_rank / seen,
                "train_total": total_loss / seen,
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
                "gradient_norm": float(gradient_norm),
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
        raise RuntimeError("membrane rank member has no valid checkpoint")
    model.load_state_dict(best_state)
    validation_prediction = _predict(
        model,
        data,
        validation_indices,
        target_scale=target_scale,
        batch_size=batch_size,
        device=device,
    )
    payload = {
        "config": model.config_dict(),
        "state_dict": copy.deepcopy(model.cpu().state_dict()),
        "target_scale": target_scale,
        "rank_weight": rank_weight,
        "seed": seed,
        "best_epoch": best_epoch,
        "optimizer_steps": optimizer_steps,
        "examples_seen": examples_seen,
        "history": history,
        "validation": _metrics(
            target[validation_indices],
            validation_prediction,
            protein_id[validation_indices],
        ),
        "validation_selection_score": best_score,
    }
    return model, payload


def _ensemble_predictions(
    members: list[DirectionalAssayHead],
    payloads: list[dict[str, object]],
    data: dict[str, object],
    indices: np.ndarray,
    *,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    predictions = [
        _predict(
            member.to(device),
            data,
            indices,
            target_scale=float(payload["target_scale"]),
            batch_size=batch_size,
            device=device,
        )
        for member, payload in zip(members, payloads, strict=True)
    ]
    return np.mean(predictions, axis=0)


def _gpcr_metrics(
    data: dict[str, object],
    prediction: np.ndarray,
    indices: np.ndarray,
) -> dict[str, object]:
    target = np.asarray(data["target"])[indices]
    selected_prediction = prediction[indices]
    proteins = np.asarray(data["protein_id"])[indices]
    result = _metrics(target, selected_prediction, proteins)
    per_receptor: dict[str, object] = {}
    for protein in sorted(set(proteins.tolist())):
        selected = proteins == protein
        if selected.sum() < 2:
            continue
        statistic = float(
            spearmanr(
                target[selected],
                selected_prediction[selected],
            ).statistic
        )
        if np.isfinite(statistic):
            per_receptor[protein] = {
                "rows": int(selected.sum()),
                "spearman": statistic,
            }
    result["per_receptor"] = per_receptor
    return result


def train_zero_shot_membrane_ranker(
    representation_dir: Path,
    gpcr_tm_source: Path,
    checkpoint_dir: Path,
    *,
    seed: int = 20260718,
    discovery_epochs: int = 40,
    discovery_minimum_epochs: int = 20,
    main_epochs: int = 120,
    minimum_main_epochs: int = 50,
    patience: int = 12,
    batch_size: int = 256,
    learning_rate: float = 5e-4,
    weight_decay: float = 1e-4,
    ensemble_size: int = 5,
    device: str = "cuda",
) -> dict[str, object]:
    """Train on non-benchmark membrane proteins and evaluate GPCR zero-shot."""

    representation_dir = Path(representation_dir).resolve()
    gpcr_tm_source = Path(gpcr_tm_source).resolve()
    checkpoint_dir = Path(checkpoint_dir).resolve()
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    torch_device = torch.device(
        device if device.startswith("cuda") and torch.cuda.is_available() else "cpu"
    )
    if torch_device.type == "cuda":
        torch.set_float32_matmul_precision("highest")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    mptherm = _load_representation(representation_dir / "mptherm.h5")
    gpcr_tm = _load_representation(representation_dir / "gpcr_tm.h5")
    gpcr_rank = _load_representation(representation_dir / "gpcr_rank.h5")
    base_hashes = {
        str(value["base_checkpoint_sha256"])
        for value in (mptherm, gpcr_tm, gpcr_rank)
    }
    if len(base_hashes) != 1:
        raise RuntimeError("membrane-rank representations use different bases")
    benchmark_accessions = set(
        np.asarray(gpcr_tm["protein_id"]).tolist()
    )
    rank_accessions = (
        set(np.asarray(gpcr_rank["uniprot_id"]).tolist())
        if "uniprot_id" in gpcr_rank
        else set(np.asarray(gpcr_rank["protein_id"]).tolist())
    )
    benchmark_accessions |= rank_accessions
    mptherm_protein = np.asarray(mptherm["protein_id"])
    eligible = ~np.isin(
        mptherm_protein,
        np.asarray(sorted(benchmark_accessions), dtype=object),
    )
    source_split = np.asarray(mptherm["split"])
    eligible &= source_split != "quarantine"
    split = _derived_validation_split(
        source_split,
        mptherm_protein,
        seed=seed,
    )
    train_indices = np.flatnonzero(eligible & (split == "train"))
    validation_indices = np.flatnonzero(eligible & (split == "val"))
    test_indices = np.flatnonzero(eligible & (split == "test"))
    split_proteins = [
        set(mptherm_protein[indices].tolist())
        for indices in (train_indices, validation_indices, test_indices)
    ]
    if (
        split_proteins[0] & split_proteins[1]
        or split_proteins[0] & split_proteins[2]
        or split_proteins[1] & split_proteins[2]
    ):
        raise RuntimeError("membrane-rank proteins overlap across splits")
    started = time.monotonic()
    discovery: list[dict[str, object]] = []
    for rank_weight in (0.0, 0.25, 1.0):
        _, payload = _train_member(
            mptherm,
            train_indices,
            validation_indices,
            rank_weight=rank_weight,
            seed=seed,
            epochs=discovery_epochs,
            minimum_epochs=discovery_minimum_epochs,
            patience=max(4, patience // 2),
            batch_size=batch_size,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            device=torch_device,
        )
        discovery.append(
            {
                "rank_weight": rank_weight,
                "best_epoch": payload["best_epoch"],
                "validation": payload["validation"],
                "selection_score": payload[
                    "validation_selection_score"
                ],
            }
        )
    selected_rank_weight = float(
        max(
            discovery,
            key=lambda value: (
                float(value["selection_score"]),
                -float(value["rank_weight"]),
            ),
        )["rank_weight"]
    )
    members: list[DirectionalAssayHead] = []
    member_payloads: list[dict[str, object]] = []
    for member_index in range(ensemble_size):
        model, payload = _train_member(
            mptherm,
            train_indices,
            validation_indices,
            rank_weight=selected_rank_weight,
            seed=seed + 100 + member_index,
            epochs=main_epochs,
            minimum_epochs=minimum_main_epochs,
            patience=patience,
            batch_size=batch_size,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            device=torch_device,
        )
        members.append(model.cpu())
        member_payloads.append(payload)
    validation_prediction = _ensemble_predictions(
        members,
        member_payloads,
        mptherm,
        validation_indices,
        batch_size=batch_size,
        device=torch_device,
    )
    test_prediction = _ensemble_predictions(
        members,
        member_payloads,
        mptherm,
        test_indices,
        batch_size=batch_size,
        device=torch_device,
    )
    gpcr_prediction = _ensemble_predictions(
        members,
        member_payloads,
        gpcr_tm,
        np.arange(len(np.asarray(gpcr_tm["target"])), dtype=np.int64),
        batch_size=batch_size,
        device=torch_device,
    )
    source = pd.read_csv(gpcr_tm_source)
    overlap_by_key = {
        (str(row.protein_id), str(row.mutation)): bool(
            row.official_test_site_overlap
        )
        for row in source.itertuples(index=False)
    }
    overlap = np.asarray(
        [
            overlap_by_key[(str(protein), str(mutation))]
            for protein, mutation in zip(
                np.asarray(gpcr_tm["protein_id"]),
                np.asarray(gpcr_tm["mutation"]),
                strict=True,
            )
        ],
        dtype=bool,
    )
    gpcr_split = np.asarray(gpcr_tm["split"])
    development_indices = np.flatnonzero(
        (gpcr_split == "train") & ~overlap
    )
    official_indices = np.flatnonzero(gpcr_split == "test")
    generic_favorable = -np.asarray(
        gpcr_tm["base_ddg"], dtype=np.float32
    )
    validation_metrics = _metrics(
        np.asarray(mptherm["target"])[validation_indices],
        validation_prediction,
        mptherm_protein[validation_indices],
    )
    test_metrics = _metrics(
        np.asarray(mptherm["target"])[test_indices],
        test_prediction,
        mptherm_protein[test_indices],
    )
    gpcr_metrics = {
        "development": {
            "generic_favorable": _gpcr_metrics(
                gpcr_tm,
                generic_favorable,
                development_indices,
            ),
            "zero_shot_membrane_ranker": _gpcr_metrics(
                gpcr_tm,
                gpcr_prediction,
                development_indices,
            ),
        },
        "official_confirmation": {
            "generic_favorable": _gpcr_metrics(
                gpcr_tm,
                generic_favorable,
                official_indices,
            ),
            "zero_shot_membrane_ranker": _gpcr_metrics(
                gpcr_tm,
                gpcr_prediction,
                official_indices,
            ),
        },
    }
    blend_grid: list[dict[str, object]] = []
    for membrane_weight in np.linspace(0.0, 1.0, 21):
        blended = (
            (1.0 - membrane_weight) * generic_favorable
            + membrane_weight * gpcr_prediction
        )
        development = _gpcr_metrics(
            gpcr_tm, blended, development_indices
        )
        blend_grid.append(
            {
                "membrane_weight": float(membrane_weight),
                "development": development,
            }
        )
    selected_blend = max(
        blend_grid,
        key=lambda value: (
            float(value["development"]["macro_protein_spearman"]),
            float(value["development"]["spearman"]),
            -float(value["membrane_weight"]),
        ),
    )
    selected_membrane_weight = float(
        selected_blend["membrane_weight"]
    )
    selected_prediction = (
        (1.0 - selected_membrane_weight) * generic_favorable
        + selected_membrane_weight * gpcr_prediction
    )
    selected_official = _gpcr_metrics(
        gpcr_tm, selected_prediction, official_indices
    )
    generic_development = gpcr_metrics["development"][
        "generic_favorable"
    ]
    generic_official = gpcr_metrics["official_confirmation"][
        "generic_favorable"
    ]
    promotion_gates = {
        "development_selects_nonzero_membrane_weight": (
            selected_membrane_weight > 0.0
        ),
        "development_macro_improves": (
            float(
                selected_blend["development"][
                    "macro_protein_spearman"
                ]
            )
            > float(generic_development["macro_protein_spearman"])
        ),
        "official_macro_not_worse": (
            float(selected_official["macro_protein_spearman"])
            >= float(generic_official["macro_protein_spearman"])
        ),
    }
    production_eligible = all(promotion_gates.values())
    reverse_count = min(512, len(test_indices))
    reverse_indices = test_indices[:reverse_count]
    member_errors: list[float] = []
    for member, payload in zip(members, member_payloads, strict=True):
        member = member.to(torch_device).eval()
        latent = torch.from_numpy(
            np.asarray(mptherm["latent"])[reverse_indices].astype(
                np.float32
            )
        ).to(torch_device)
        base_ddg = torch.from_numpy(
            np.asarray(mptherm["base_ddg"])[reverse_indices].astype(
                np.float32
            )
        ).to(torch_device)
        membrane = torch.from_numpy(
            np.asarray(mptherm["membrane"])[reverse_indices].astype(
                np.float32
            )
        ).to(torch_device)
        with torch.inference_mode():
            forward = member(latent, base_ddg, membrane)
            reverse = member(-latent, -base_ddg, membrane)
        member_errors.append(
            float(torch.max(torch.abs(forward + reverse)).cpu())
        )
    checkpoint_payload = {
        "schema": MEMBRANE_RANK_CHECKPOINT_SCHEMA,
        "candidate": "zero_shot_membrane_ranker",
        "members": member_payloads,
        "selected_rank_weight": selected_rank_weight,
        "base_checkpoint_sha256": next(iter(base_hashes)),
        "representation_manifest_sha256": file_sha256(
            representation_dir / "manifest.json"
        ),
        "benchmark_accessions_excluded": sorted(benchmark_accessions),
        "metrics": {
            "mptherm_validation": validation_metrics,
            "mptherm_test": test_metrics,
            "gpcr_tm": gpcr_metrics,
        },
        "gpcr_blend_selection": {
            "selected_membrane_weight": selected_membrane_weight,
            "selected_on": (
                "82-row GPCR development macro receptor Spearman; "
                "official 12 rows consulted only after selection"
            ),
            "selected_development": selected_blend["development"],
            "selected_official": selected_official,
        },
        "promotion": {
            "production_eligible": production_eligible,
            "gates": promotion_gates,
        },
    }
    checkpoint_path = checkpoint_dir / "zero_shot_membrane_ranker.pt"
    torch.save(checkpoint_payload, checkpoint_path)
    report = {
        "schema": "protein-stabilizer.zero-shot-membrane-rank-training.v1",
        "candidate": "zero_shot_membrane_ranker",
        "elapsed_seconds": time.monotonic() - started,
        "training_policy": {
            "source": "MPTherm non-benchmark membrane proteins only",
            "target": "delta-Tm; positive is stabilizing",
            "precision": "native FP32; TF32 disabled",
            "discovery_epochs": discovery_epochs,
            "main_epochs": main_epochs,
            "minimum_main_epochs": minimum_main_epochs,
            "ensemble_size": ensemble_size,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "selection": (
                "MPTherm protein-held-out validation composite only"
            ),
        },
        "discovery": discovery,
        "selected_rank_weight": selected_rank_weight,
        "split_integrity": {
            "benchmark_accessions_excluded": sorted(
                benchmark_accessions
            ),
            "excluded_accession_count": len(benchmark_accessions),
            "train_rows": len(train_indices),
            "validation_rows": len(validation_indices),
            "test_rows": len(test_indices),
            "train_proteins": len(split_proteins[0]),
            "validation_proteins": len(split_proteins[1]),
            "test_proteins": len(split_proteins[2]),
            "overlap": 0,
            "gpcr_development_rows": len(development_indices),
            "gpcr_official_rows": len(official_indices),
            "gpcr_training_rows": 0,
        },
        "mptherm": {
            "validation": validation_metrics,
            "test": test_metrics,
        },
        "gpcr_tm": gpcr_metrics,
        "gpcr_blend_selection": {
            "selected_membrane_weight": selected_membrane_weight,
            "selected_on": (
                "82-row GPCR development macro receptor Spearman; "
                "official 12 rows consulted only after selection"
            ),
            "grid": blend_grid,
            "selected_development": selected_blend["development"],
            "selected_official": selected_official,
        },
        "promotion": {
            "production_eligible": production_eligible,
            "decision": (
                "promote membrane adapter"
                if production_eligible
                else "reject membrane adapter; retain generic state-potential fusion"
            ),
            "gates": promotion_gates,
        },
        "directional_constraints": {
            "evaluated_rows": reverse_count,
            "maximum_absolute_forward_plus_reverse": max(
                member_errors, default=0.0
            ),
        },
        "members": [
            {
                key: value
                for key, value in payload.items()
                if key != "state_dict"
            }
            for payload in member_payloads
        ],
        "artifacts": {
            "checkpoint": str(checkpoint_path),
            "checkpoint_sha256": file_sha256(checkpoint_path),
            "representation_manifest": str(
                representation_dir / "manifest.json"
            ),
            "representation_manifest_sha256": file_sha256(
                representation_dir / "manifest.json"
            ),
            "gpcr_tm_source": str(gpcr_tm_source),
            "gpcr_tm_source_sha256": file_sha256(gpcr_tm_source),
        },
    }
    report_path = checkpoint_dir / "zero_shot_membrane_rank_report.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return report
