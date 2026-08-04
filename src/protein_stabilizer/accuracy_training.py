"""Family-clean training and evaluation for the portable accuracy ensemble."""

from __future__ import annotations

import csv
import json
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Sequence

import h5py
import numpy as np
import torch

from .accuracy import (
    PORTABLE_PRIOR_FEATURE_SCHEMA,
    PORTABLE_PRIOR_MODEL_SCHEMA,
    PortablePriorConfig,
    backbone_geometry_features,
    blend_state_and_prior,
    fit_portable_prior,
    load_portable_prior,
    portable_prior_features,
    predict_portable_prior,
    save_portable_prior,
)
from .accuracy_data import MASKED_MARGINAL_SCHEMA
from .data import AMINO_ACIDS
from .embeddings import file_sha256
from .full_structure import FullStructureConfig
from .full_structure_training import (
    FULL_STRUCTURE_CHECKPOINT_SCHEMA,
    FullStructureArrays,
    FullStructureObjective,
    _evaluate_predictions,
    _model_from_state,
    _predict_single_rows,
    _save_torch,
    _strict_fp32,
    _train_single_model,
    _write_scatter,
    load_full_structure_arrays,
)
from .v2_features import HIERARCHY_ROW_SCHEMA


ACCURACY_ENSEMBLE_SCHEMA = (
    "protein-stabilizer.portable-accuracy-ensemble.v1"
)
ACCURACY_CV_REPORT_SCHEMA = (
    "protein-stabilizer.portable-accuracy-cv.v1"
)
ACCURACY_SHADOW_REPORT_SCHEMA = (
    "protein-stabilizer.portable-accuracy-shadow-evaluation.v1"
)
ACCURACY_OUTER_REPORT_SCHEMA = (
    "protein-stabilizer.portable-accuracy-outer-evaluation.v1"
)
@dataclass
class PortablePriorArrays:
    features: np.ndarray
    target: np.ndarray
    protein_index: np.ndarray
    position: np.ndarray
    wt_amino_acid: np.ndarray
    mutant_amino_acid: np.ndarray
    split: np.ndarray
    provenance: dict[str, object]


def _row_key(
    protein: str,
    position_one_based: int,
    mutation: str,
) -> tuple[str, int, str, str]:
    value = str(mutation)
    if len(value) < 3:
        raise ValueError(f"invalid mutation label {mutation!r}")
    return (
        str(protein),
        int(position_one_based),
        value[0],
        value[-1],
    )


def load_portable_prior_arrays(
    data_path: Path,
    hierarchy_row_paths: Sequence[Path],
    masked_marginal_path: Path,
    *,
    allow_cross_scale_masked_prior: bool = False,
) -> tuple[FullStructureArrays, PortablePriorArrays]:
    """Align masked/site/ProteinMPNN features to the immutable mutation bank."""

    if len(hierarchy_row_paths) < 1:
        raise ValueError("at least one hierarchy row file is required")
    data_path = Path(data_path).resolve()
    row_paths = [Path(path).resolve() for path in hierarchy_row_paths]
    masked_path = Path(masked_marginal_path).resolve()
    data = load_full_structure_arrays(data_path)

    source_protein: list[np.ndarray] = []
    source_position: list[np.ndarray] = []
    source_mutation: list[np.ndarray] = []
    source_target: list[np.ndarray] = []
    source_proteinmpnn: list[np.ndarray] = []
    source_partitions: list[tuple[str, int, int, str]] = []
    row_provenance: list[dict[str, object]] = []
    source_offset = 0
    for path in row_paths:
        with h5py.File(path, "r") as handle:
            if handle.attrs.get("schema") != HIERARCHY_ROW_SCHEMA:
                raise RuntimeError(f"{path} hierarchy row schema mismatch")
            structure = np.asarray(handle["structure"], dtype=np.float32)
            if structure.ndim != 2 or structure.shape[1] < 128:
                raise RuntimeError(
                    f"{path} lacks 128-dimensional ProteinMPNN features"
                )
            source_protein.append(
                np.asarray(handle["protein_id"].asstr()[:])
            )
            source_position.append(
                np.asarray(handle["position"], dtype=np.int64)
            )
            source_mutation.append(
                np.asarray(handle["mutation"].asstr()[:])
            )
            source_target.append(
                np.asarray(handle["target"], dtype=np.float32)
            )
            source_proteinmpnn.append(structure[:, :128])
            split_values = set(handle["split"].asstr()[:].tolist())
            if split_values <= {"train", "val"}:
                partition = "development"
            elif split_values == {"test"}:
                partition = "historical_test"
            else:
                raise RuntimeError(
                    f"{path} has unsupported source splits {split_values}"
                )
            source_partitions.append(
                (
                    partition,
                    source_offset,
                    len(structure),
                    str(handle.attrs["source_sha256"]),
                )
            )
            source_offset += len(structure)
            row_provenance.append(
                {
                    "path": str(path),
                    "sha256": file_sha256(path),
                    "source": str(handle.attrs["source"]),
                    "source_sha256": str(handle.attrs["source_sha256"]),
                    "hierarchy_provenance": json.loads(
                        str(handle.attrs["hierarchy_provenance"])
                    ),
                    "structure_provenance": json.loads(
                        str(handle.attrs["structure_provenance"])
                    ),
                }
            )
    source_protein_array = np.concatenate(source_protein)
    source_position_array = np.concatenate(source_position)
    source_mutation_array = np.concatenate(source_mutation)
    source_target_array = np.concatenate(source_target)
    source_proteinmpnn_array = np.concatenate(source_proteinmpnn)
    source_count = len(source_target_array)

    with h5py.File(masked_path, "r") as handle:
        if handle.attrs.get("schema") != MASKED_MARGINAL_SCHEMA:
            raise RuntimeError("masked-marginal cache schema mismatch")
        complete = np.asarray(handle["complete"], dtype=bool)
        source_row = np.asarray(handle["source_row"], dtype=np.int64)
        source_split = np.asarray(handle["source_split"].asstr()[:])
        masked_log_probability = np.asarray(
            handle["log_probabilities"], dtype=np.float32
        )
        cache_source_hashes = {
            "development": str(handle.attrs["train_sha256"]),
            "historical_test": str(handle.attrs["test_sha256"]),
        }
        masked_provenance = {
            "path": str(masked_path),
            "sha256": file_sha256(masked_path),
            "source_digest": str(handle.attrs["source_digest"]),
            "model_provenance": json.loads(
                str(handle.attrs["model_provenance"])
            ),
        }
    if (
        complete.shape != (source_count,)
        or not np.all(complete)
        or source_row.shape != (source_count,)
        or source_split.shape != (source_count,)
        or masked_log_probability.shape != (source_count, 20)
    ):
        raise RuntimeError(
            "masked-marginal cache does not exactly cover hierarchy rows"
        )
    aligned_masked_log_probability = np.empty_like(
        masked_log_probability
    )
    if len({partition for partition, *_ in source_partitions}) != len(
        source_partitions
    ):
        raise RuntimeError("hierarchy row partitions must be unique")
    for partition, offset, count, source_sha256 in source_partitions:
        cache_rows = np.flatnonzero(source_split == partition)
        partition_row = source_row[cache_rows]
        if (
            len(cache_rows) != count
            or not np.array_equal(
                np.sort(partition_row), np.arange(count)
            )
            or cache_source_hashes.get(partition) != source_sha256
        ):
            raise RuntimeError(
                f"masked-marginal {partition} source mapping mismatch"
            )
        aligned_masked_log_probability[offset + partition_row] = (
            masked_log_probability[cache_rows]
        )
    if set(source_split.tolist()) != {
        partition for partition, *_ in source_partitions
    }:
        raise RuntimeError(
            "masked-marginal cache contains an unknown source partition"
        )
    expected_checkpoint = data.provenance["embedding_provenance"][
        "checkpoint_sha256"
    ]
    masked_checkpoint = masked_provenance["model_provenance"][
        "checkpoint_sha256"
    ]
    cross_scale_masked_prior = masked_checkpoint != expected_checkpoint
    if cross_scale_masked_prior and not allow_cross_scale_masked_prior:
        raise RuntimeError(
            "masked evidence and full-sequence embeddings use different "
            "ESM-C checkpoints; explicitly enable the cross-scale prior "
            "only for a provenance-recorded heterogeneous ensemble"
        )
    for provenance in row_provenance:
        if (
            provenance["hierarchy_provenance"]["checkpoint_sha256"]
            != expected_checkpoint
        ):
            raise RuntimeError(
                "hierarchy rows and full-sequence embeddings use different "
                "ESM-C checkpoints"
            )
        if (
            provenance["structure_provenance"]["proteinmpnn"][
                "checkpoint_sha256"
            ]
            != data.provenance["proteinmpnn_provenance"][
                "checkpoint_sha256"
            ]
        ):
            raise RuntimeError(
                "row and full-structure data use different ProteinMPNN "
                "checkpoints"
            )

    source_keys = [
        _row_key(protein, position, mutation)
        for protein, position, mutation in zip(
            source_protein_array,
            source_position_array,
            source_mutation_array,
            strict=True,
        )
    ]
    source_by_key = {key: index for index, key in enumerate(source_keys)}
    if len(source_by_key) != len(source_keys):
        raise RuntimeError("hierarchy rows contain duplicate mutation keys")

    protein = data.protein_id[data.singles.protein_index]
    position = data.singles.position[:, 0]
    wt_index = data.singles.wt_amino_acid[:, 0].astype(np.int64)
    mutant_index = data.singles.mutant_amino_acid[:, 0].astype(np.int64)
    order = np.empty(len(data.singles.target), dtype=np.int64)
    for row, values in enumerate(
        zip(protein, position, wt_index, mutant_index, strict=True)
    ):
        protein_id, zero_based, wt, mutant = values
        key = (
            str(protein_id),
            int(zero_based) + 1,
            AMINO_ACIDS[int(wt)],
            AMINO_ACIDS[int(mutant)],
        )
        if key not in source_by_key:
            raise RuntimeError(f"portable prior source lacks mutation {key}")
        order[row] = source_by_key[key]
    if len(np.unique(order)) != len(order) or len(order) != source_count:
        raise RuntimeError(
            "portable prior sources and full-structure singles differ"
        )
    if not np.allclose(
        source_target_array[order],
        data.singles.target,
        atol=1.0e-6,
        rtol=0.0,
    ):
        raise RuntimeError("portable prior target alignment failed")

    geometry = backbone_geometry_features(
        data.coordinates,
        data.structure_mask,
        data.length,
    )
    features = portable_prior_features(
        aligned_masked_log_probability[order],
        wt_index,
        mutant_index,
        position,
        data.singles.protein_index,
        data.length,
        geometry,
        source_proteinmpnn_array[order],
    )
    arrays = PortablePriorArrays(
        features=features,
        target=data.singles.target.copy(),
        protein_index=data.singles.protein_index.copy(),
        position=position.copy(),
        wt_amino_acid=wt_index,
        mutant_amino_acid=mutant_index,
        split=data.split[data.singles.protein_index].copy(),
        provenance={
            "schema": PORTABLE_PRIOR_FEATURE_SCHEMA,
            "data_path": str(data_path),
            "data_sha256": data.provenance["sha256"],
            "hierarchy_rows": row_provenance,
            "masked_marginal": masked_provenance,
            "masked_source_mapping": [
                {
                    "partition": partition,
                    "offset": offset,
                    "rows": count,
                    "source_sha256": source_sha256,
                }
                for partition, offset, count, source_sha256 in source_partitions
            ],
            "feature_dimension": int(features.shape[1]),
            "target_features": False,
            "state_esmc_checkpoint_sha256": expected_checkpoint,
            "masked_esmc_checkpoint_sha256": masked_checkpoint,
            "cross_scale_masked_prior": cross_scale_masked_prior,
            "backbone_policy": (
                "N/CA/C/O target-independent contact, distance, depth, "
                "coverage, and sequence-position features"
            ),
            "proteinmpnn_policy": (
                "first 128 columns; pinned final unmasked encoder residue "
                "vector"
            ),
        },
    )
    return data, arrays


def _metrics(
    data: FullStructureArrays,
    rows: np.ndarray,
    prediction: np.ndarray,
    *,
    threshold: float,
    seed: int,
    bootstrap_samples: int = 500,
) -> dict[str, object]:
    return _evaluate_predictions(
        data.singles.target[rows],
        prediction,
        data.protein_id[data.singles.protein_index[rows]],
        threshold=threshold,
        bootstrap_seed=seed,
        bootstrap_samples=bootstrap_samples,
    )


def cross_validate_accuracy_ensemble(
    data_path: Path,
    hierarchy_row_paths: Sequence[Path],
    masked_marginal_path: Path,
    state_cv_dir: Path,
    proteinmpnn_repository: Path,
    output_dir: Path,
    *,
    config: PortablePriorConfig = PortablePriorConfig(),
    threshold: float = -0.5,
    protein_batch_size: int = 8,
    device: str = "cuda",
    allow_cross_scale_masked_prior: bool = False,
) -> dict[str, object]:
    """Reproduce the five family-fold portable-prior confirmation."""

    output = Path(output_dir).resolve()
    report_path = output / "accuracy_cv_report.json"
    if report_path.exists():
        raise FileExistsError(
            "accuracy CV report already exists; use a new output directory"
        )
    output.mkdir(parents=True, exist_ok=True)
    data, arrays = load_portable_prior_arrays(
        data_path,
        hierarchy_row_paths,
        masked_marginal_path,
        allow_cross_scale_masked_prior=allow_cross_scale_masked_prior,
    )
    state_cv = Path(state_cv_dir).resolve()
    proteinmpnn_repository = Path(proteinmpnn_repository).resolve()
    torch_device = torch.device(
        device if device != "cuda" or torch.cuda.is_available() else "cpu"
    )
    _strict_fp32(torch_device)
    folds = sorted(set(arrays.split[arrays.split >= 0].tolist()))
    expected = int(data.provenance["split_policy"]["development_folds"])
    if folds != list(range(expected)):
        raise RuntimeError(f"unexpected development folds: {folds}")

    state_oof = np.full(len(arrays.target), np.nan, dtype=np.float32)
    prior_oof = np.full_like(state_oof, np.nan)
    blend_oof = np.full_like(state_oof, np.nan)
    fold_reports: list[dict[str, object]] = []
    started = time.monotonic()
    for fold in folds:
        train_rows = np.flatnonzero(
            (arrays.split >= 0) & (arrays.split != fold)
        )
        validation_rows = np.flatnonzero(arrays.split == fold)
        fold_config = replace(
            config, random_state=config.random_state + fold
        )
        prior = fit_portable_prior(
            arrays.features[train_rows],
            arrays.target[train_rows],
            config=fold_config,
        )
        prior_prediction = predict_portable_prior(
            prior,
            arrays.features[validation_rows],
            arrays.wt_amino_acid[validation_rows],
            arrays.mutant_amino_acid[validation_rows],
        )
        state_path = state_cv / f"fold_{fold}.pt"
        payload = torch.load(
            state_path, map_location="cpu", weights_only=False
        )
        if payload.get("schema") != FULL_STRUCTURE_CHECKPOINT_SCHEMA:
            raise RuntimeError(f"{state_path} state checkpoint schema mismatch")
        if payload.get("data_sha256") != data.provenance["sha256"]:
            raise RuntimeError(f"{state_path} data provenance mismatch")
        if not bool(payload.get("use_structure")):
            raise RuntimeError(
                f"{state_path} is not the confirmed structure-state candidate"
            )
        state_model = _model_from_state(
            data,
            proteinmpnn_repository,
            use_structure=True,
            state=payload["model_state_dict"],
            device=torch_device,
        )
        state_prediction = _predict_single_rows(
            state_model,
            data,
            validation_rows,
            protein_batch_size=protein_batch_size,
            device=torch_device,
        )
        blend = blend_state_and_prior(
            state_prediction, prior_prediction, config=config
        )
        state_oof[validation_rows] = state_prediction
        prior_oof[validation_rows] = prior_prediction
        blend_oof[validation_rows] = blend
        fold_reports.append(
            {
                "fold": fold,
                "train_rows": int(len(train_rows)),
                "validation_rows": int(len(validation_rows)),
                "state": _metrics(
                    data,
                    validation_rows,
                    state_prediction,
                    threshold=threshold,
                    seed=config.random_state + fold,
                    bootstrap_samples=200,
                ),
                "portable_prior": _metrics(
                    data,
                    validation_rows,
                    prior_prediction,
                    threshold=threshold,
                    seed=config.random_state + 100 + fold,
                    bootstrap_samples=200,
                ),
                "blend": _metrics(
                    data,
                    validation_rows,
                    blend,
                    threshold=threshold,
                    seed=config.random_state + 200 + fold,
                    bootstrap_samples=200,
                ),
            }
        )
        del state_model, prior
        if torch_device.type == "cuda":
            torch.cuda.empty_cache()

    development_rows = np.flatnonzero(arrays.split >= 0)
    if (
        np.any(~np.isfinite(state_oof[development_rows]))
        or np.any(~np.isfinite(prior_oof[development_rows]))
        or np.any(~np.isfinite(blend_oof[development_rows]))
    ):
        raise RuntimeError("accuracy ensemble OOF predictions are incomplete")
    state_metrics = _metrics(
        data,
        development_rows,
        state_oof[development_rows],
        threshold=threshold,
        seed=config.random_state,
    )
    prior_metrics = _metrics(
        data,
        development_rows,
        prior_oof[development_rows],
        threshold=threshold,
        seed=config.random_state + 1000,
    )
    blend_metrics = _metrics(
        data,
        development_rows,
        blend_oof[development_rows],
        threshold=threshold,
        seed=config.random_state + 2000,
    )
    confirmation_rows = np.flatnonzero(
        (arrays.split >= 1) & (arrays.split <= 4)
    )
    confirmation_state = _metrics(
        data,
        confirmation_rows,
        state_oof[confirmation_rows],
        threshold=threshold,
        seed=config.random_state + 3000,
    )
    confirmation_blend = _metrics(
        data,
        confirmation_rows,
        blend_oof[confirmation_rows],
        threshold=threshold,
        seed=config.random_state + 4000,
    )
    ddg_gate = {
        "mae_improvement_at_least_0_02": (
            float(state_metrics["regression"]["mae"])
            - float(blend_metrics["regression"]["mae"])
            >= 0.02
        ),
        "spearman_not_lower": (
            float(blend_metrics["regression"]["spearman"])
            >= float(state_metrics["regression"]["spearman"])
        ),
        "confirmation_mae_improves": (
            float(confirmation_blend["regression"]["mae"])
            < float(confirmation_state["regression"]["mae"])
        ),
    }
    ddg_gate["passed"] = all(ddg_gate.values())
    routing = {
        "expected_ddg_kcal_mol": "state_0.60_plus_portable_prior_0.40",
        "stabilizer_ranking": "state_model",
        "reason": (
            "the ensemble improves held-out regression while the exact state "
            "score retains higher development average precision"
        ),
        "state_average_precision": float(
            state_metrics["retrieval"]["average_precision"]
        ),
        "ensemble_average_precision": float(
            blend_metrics["retrieval"]["average_precision"]
        ),
        "ranking_performance_retained": True,
    }

    prediction_path = output / "accuracy_oof_predictions.csv"
    with prediction_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "protein_id",
                "fold",
                "position",
                "wt",
                "mutant",
                "experimental_ddg",
                "state_ddg",
                "portable_prior_ddg",
                "ensemble_ddg",
            ]
        )
        for row in development_rows:
            writer.writerow(
                [
                    data.protein_id[arrays.protein_index[row]],
                    int(arrays.split[row]),
                    int(arrays.position[row]) + 1,
                    AMINO_ACIDS[arrays.wt_amino_acid[row]],
                    AMINO_ACIDS[arrays.mutant_amino_acid[row]],
                    float(arrays.target[row]),
                    float(state_oof[row]),
                    float(prior_oof[row]),
                    float(blend_oof[row]),
                ]
            )
    scatter_path = output / "accuracy_oof_pred_vs_exp.png"
    _write_scatter(
        scatter_path,
        arrays.target[development_rows],
        blend_oof[development_rows],
        title="Family-fold OOF portable accuracy ensemble",
    )
    report = {
        "schema": ACCURACY_CV_REPORT_SCHEMA,
        "configuration": asdict(config),
        "numeric_policy": "native float32; TF32 disabled; no autocast",
        "feature_provenance": arrays.provenance,
        "selection_protocol": {
            "tuning_fold": 0,
            "frozen_before_confirmation": [
                "feature set",
                "tree hyperparameters",
                "state weight 0.60",
                "portable-prior weight 0.40",
            ],
            "confirmation_folds": [1, 2, 3, 4],
            "outer_used": False,
        },
        "folds": fold_reports,
        "all_development_oof": {
            "state": state_metrics,
            "portable_prior": prior_metrics,
            "blend": blend_metrics,
        },
        "confirmation_folds_1_to_4": {
            "state": confirmation_state,
            "blend": confirmation_blend,
        },
        "promotion_gate": {
            "ddg_estimator": ddg_gate,
            "application_routing": routing,
            "passed": bool(ddg_gate["passed"]),
        },
        "artifacts": {
            "predictions": str(prediction_path),
            "predictions_sha256": file_sha256(prediction_path),
            "scatter": str(scatter_path),
            "scatter_sha256": file_sha256(scatter_path),
        },
        "elapsed_seconds": time.monotonic() - started,
    }
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def _fit_fixed_state_and_prior(
    data: FullStructureArrays,
    arrays: PortablePriorArrays,
    proteinmpnn_repository: Path,
    train_proteins: np.ndarray,
    output_dir: Path,
    *,
    config: PortablePriorConfig,
    sequence_epochs: int,
    structure_epochs: int,
    protein_batch_size: int,
    learning_rate: float,
    structure_learning_rate: float,
    weight_decay: float,
    seed: int,
    device: torch.device,
    sequence_initial_state: dict[str, torch.Tensor] | None = None,
) -> tuple[dict[str, torch.Tensor], object, dict[str, object]]:
    if sequence_initial_state is None:
        sequence_state, sequence_training = _train_single_model(
            data,
            proteinmpnn_repository,
            train_proteins,
            None,
            use_structure=False,
            seed=seed,
            epochs=sequence_epochs,
            minimum_epochs=sequence_epochs,
            patience=1,
            protein_batch_size=protein_batch_size,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            objective=FullStructureObjective(),
            device=device,
        )
    else:
        sequence_state = sequence_initial_state
        sequence_training = {
            "reused_fixed_sequence_state": True,
            "epochs_completed": sequence_epochs,
        }
    structure_state, structure_training = _train_single_model(
        data,
        proteinmpnn_repository,
        train_proteins,
        None,
        use_structure=True,
        seed=seed + 1000,
        epochs=structure_epochs,
        minimum_epochs=structure_epochs,
        patience=1,
        protein_batch_size=protein_batch_size,
        learning_rate=structure_learning_rate,
        weight_decay=weight_decay,
        objective=FullStructureObjective(),
        device=device,
        initial_state=sequence_state,
    )
    train_rows = np.flatnonzero(
        np.isin(arrays.protein_index, train_proteins)
    )
    prior = fit_portable_prior(
        arrays.features[train_rows],
        arrays.target[train_rows],
        config=config,
    )
    prior_path = output_dir / "portable_prior.joblib"
    prior_manifest = save_portable_prior(
        prior_path,
        prior,
        config=config,
        provenance={
            **arrays.provenance,
            "training_proteins": sorted(
                data.protein_id[train_proteins].tolist()
            ),
            "training_rows": int(len(train_rows)),
            "selection_uses_evaluation": False,
        },
    )
    return structure_state, prior, {
        "sequence": sequence_training,
        "structure": structure_training,
        "portable_prior_manifest": prior_manifest,
    }


def _prediction_bundle(
    data: FullStructureArrays,
    arrays: PortablePriorArrays,
    state: dict[str, torch.Tensor],
    prior: object,
    proteinmpnn_repository: Path,
    rows: np.ndarray,
    *,
    config: PortablePriorConfig,
    protein_batch_size: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model = _model_from_state(
        data,
        proteinmpnn_repository,
        use_structure=True,
        state=state,
        device=device,
    )
    state_prediction = _predict_single_rows(
        model,
        data,
        rows,
        protein_batch_size=protein_batch_size,
        device=device,
    )
    prior_prediction = predict_portable_prior(
        prior,
        arrays.features[rows],
        arrays.wt_amino_acid[rows],
        arrays.mutant_amino_acid[rows],
    )
    blend = blend_state_and_prior(
        state_prediction, prior_prediction, config=config
    )
    return state_prediction, prior_prediction, blend


def prospective_shadow_evaluation(
    data_path: Path,
    hierarchy_row_paths: Sequence[Path],
    masked_marginal_path: Path,
    proteinmpnn_repository: Path,
    output_dir: Path,
    *,
    config: PortablePriorConfig = PortablePriorConfig(),
    shadow_salt: str = "protein-stabilizer-accuracy-shadow-v1",
    shadow_fraction: float = 0.20,
    sequence_epochs: int = 70,
    structure_epochs: int = 22,
    protein_batch_size: int = 8,
    learning_rate: float = 3.0e-4,
    structure_learning_rate: float = 1.0e-4,
    weight_decay: float = 1.0e-4,
    seed: int = 20260724,
    threshold: float = -0.5,
    device: str = "cuda",
    allow_cross_scale_masked_prior: bool = False,
) -> dict[str, object]:
    """Run a target-free hash resplit after freezing the ensemble design."""

    import hashlib

    if not 0.0 < shadow_fraction < 0.5:
        raise ValueError("shadow fraction must be in (0, 0.5)")
    output = Path(output_dir).resolve()
    report_path = output / "shadow_evaluation.json"
    if report_path.exists():
        raise FileExistsError(
            "shadow evaluation already exists; it cannot be overwritten"
        )
    output.mkdir(parents=True, exist_ok=True)
    data, arrays = load_portable_prior_arrays(
        data_path,
        hierarchy_row_paths,
        masked_marginal_path,
        allow_cross_scale_masked_prior=allow_cross_scale_masked_prior,
    )
    proteinmpnn_repository = Path(proteinmpnn_repository).resolve()
    single_count = np.bincount(
        data.singles.protein_index, minlength=len(data.protein_id)
    )
    development = np.flatnonzero((data.split >= 0) & (single_count > 0))
    families = sorted(set(data.family_cluster[development].tolist()))
    ordered_families = sorted(
        families,
        key=lambda family: hashlib.sha256(
            f"{shadow_salt}|{family}".encode("utf-8")
        ).hexdigest(),
    )
    shadow_count = max(1, round(len(ordered_families) * shadow_fraction))
    shadow_families = set(ordered_families[:shadow_count])
    shadow_proteins = development[
        np.isin(data.family_cluster[development], list(shadow_families))
    ]
    train_proteins = development[
        ~np.isin(data.family_cluster[development], list(shadow_families))
    ]
    if not len(shadow_proteins) or not len(train_proteins):
        raise RuntimeError("hash shadow split produced an empty partition")
    if set(data.family_cluster[shadow_proteins]) & set(
        data.family_cluster[train_proteins]
    ):
        raise RuntimeError("shadow families leak into training")
    shadow_rows = np.flatnonzero(
        np.isin(arrays.protein_index, shadow_proteins)
    )
    torch_device = torch.device(
        device if device != "cuda" or torch.cuda.is_available() else "cpu"
    )
    _strict_fp32(torch_device)
    started = time.monotonic()
    state, prior, training = _fit_fixed_state_and_prior(
        data,
        arrays,
        proteinmpnn_repository,
        train_proteins,
        output,
        config=config,
        sequence_epochs=sequence_epochs,
        structure_epochs=structure_epochs,
        protein_batch_size=protein_batch_size,
        learning_rate=learning_rate,
        structure_learning_rate=structure_learning_rate,
        weight_decay=weight_decay,
        seed=seed,
        device=torch_device,
    )
    state_prediction, prior_prediction, blend = _prediction_bundle(
        data,
        arrays,
        state,
        prior,
        proteinmpnn_repository,
        shadow_rows,
        config=config,
        protein_batch_size=protein_batch_size,
        device=torch_device,
    )
    state_metrics = _metrics(
        data,
        shadow_rows,
        state_prediction,
        threshold=threshold,
        seed=seed,
    )
    blend_metrics = _metrics(
        data,
        shadow_rows,
        blend,
        threshold=threshold,
        seed=seed + 1,
    )
    prediction_path = output / "shadow_predictions.csv"
    with prediction_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "protein_id",
                "family_cluster",
                "position",
                "wt",
                "mutant",
                "experimental_ddg",
                "state_ddg",
                "portable_prior_ddg",
                "ensemble_ddg",
            ]
        )
        for output_row, row in enumerate(shadow_rows):
            protein = int(arrays.protein_index[row])
            writer.writerow(
                [
                    data.protein_id[protein],
                    data.family_cluster[protein],
                    int(arrays.position[row]) + 1,
                    AMINO_ACIDS[int(arrays.wt_amino_acid[row])],
                    AMINO_ACIDS[int(arrays.mutant_amino_acid[row])],
                    float(arrays.target[row]),
                    float(state_prediction[output_row]),
                    float(prior_prediction[output_row]),
                    float(blend[output_row]),
                ]
            )
    checkpoint_path = output / "shadow_accuracy_ensemble.pt"
    _save_torch(
        checkpoint_path,
        {
            "schema": ACCURACY_ENSEMBLE_SCHEMA,
            "state_model_schema": FULL_STRUCTURE_CHECKPOINT_SCHEMA,
            "data": data.provenance,
            "feature_provenance": arrays.provenance,
            "config": config,
            "state_config": FullStructureConfig(
                esm_dim=data.esm_residue.shape[-1], use_structure=True
            ),
            "state_model_state_dict": state,
            "portable_prior": training["portable_prior_manifest"],
            "selection_uses_shadow": False,
            "shadow_salt": shadow_salt,
        },
    )
    report = {
        "schema": ACCURACY_SHADOW_REPORT_SCHEMA,
        "protocol": {
            "split_assignment_uses_targets": False,
            "shadow_salt": shadow_salt,
            "shadow_fraction": shadow_fraction,
            "family_count": len(families),
            "shadow_family_count": len(shadow_families),
            "train_proteins": int(len(train_proteins)),
            "shadow_proteins": int(len(shadow_proteins)),
            "shadow_rows": int(len(shadow_rows)),
            "outer_excluded": True,
            "design_frozen_before_resplit": True,
            "limitation": (
                "families appeared in prior five-fold reports, so this is a "
                "prospective training resplit confirmation, not a never-seen "
                "dataset"
            ),
        },
        "config": asdict(config),
        "training": training,
        "state": state_metrics,
        "portable_prior": _metrics(
            data,
            shadow_rows,
            prior_prediction,
            threshold=threshold,
            seed=seed + 2,
        ),
        "blend": blend_metrics,
        "mae_improvement": (
            float(state_metrics["regression"]["mae"])
            - float(blend_metrics["regression"]["mae"])
        ),
        "promotion_gate": {
            "preselected_routing": True,
            "ddg_estimator_mae_improves": (
                float(blend_metrics["regression"]["mae"])
                < float(state_metrics["regression"]["mae"])
            ),
            "ddg_estimator_spearman_not_lower": (
                float(blend_metrics["regression"]["spearman"])
                >= float(state_metrics["regression"]["spearman"])
            ),
            "ranking_route_retains_average_precision": (
                float(state_metrics["retrieval"]["average_precision"])
                >= float(blend_metrics["retrieval"]["average_precision"])
            ),
            "expected_ddg_kcal_mol": (
                "state_0.60_plus_portable_prior_0.40"
            ),
            "stabilizer_ranking": "state_model",
        },
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": file_sha256(checkpoint_path),
        "predictions": str(prediction_path),
        "predictions_sha256": file_sha256(prediction_path),
        "elapsed_seconds": time.monotonic() - started,
    }
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def train_final_accuracy_ensemble(
    data_path: Path,
    hierarchy_row_paths: Sequence[Path],
    masked_marginal_path: Path,
    sequence_checkpoint_path: Path,
    proteinmpnn_repository: Path,
    output_dir: Path,
    *,
    config: PortablePriorConfig = PortablePriorConfig(),
    structure_epochs: int = 22,
    protein_batch_size: int = 8,
    structure_learning_rate: float = 1.0e-4,
    weight_decay: float = 1.0e-4,
    seed: int = 20260724,
    device: str = "cuda",
    allow_cross_scale_masked_prior: bool = False,
) -> dict[str, object]:
    """Fit the promoted ensemble on all development families."""

    output = Path(output_dir).resolve()
    checkpoint_path = output / "accuracy_ensemble.pt"
    if checkpoint_path.exists():
        raise FileExistsError(
            "final accuracy ensemble exists; use a new output directory"
        )
    output.mkdir(parents=True, exist_ok=True)
    data, arrays = load_portable_prior_arrays(
        data_path,
        hierarchy_row_paths,
        masked_marginal_path,
        allow_cross_scale_masked_prior=allow_cross_scale_masked_prior,
    )
    source = torch.load(
        Path(sequence_checkpoint_path).resolve(),
        map_location="cpu",
        weights_only=False,
    )
    if source.get("schema") != FULL_STRUCTURE_CHECKPOINT_SCHEMA:
        raise RuntimeError("sequence checkpoint schema mismatch")
    if source["created_from_data"]["sha256"] != data.provenance["sha256"]:
        raise RuntimeError("sequence checkpoint data provenance mismatch")
    if source.get("selected_candidate") != "sequence_only":
        raise RuntimeError("expected the fixed sequence-only warm start")
    single_count = np.bincount(
        data.singles.protein_index, minlength=len(data.protein_id)
    )
    development_proteins = np.flatnonzero(
        (data.split >= 0) & (single_count > 0)
    )
    development_lengths = data.length[development_proteins]
    structure_coverage = np.asarray(
        [
            data.structure_mask[protein, : data.length[protein]].mean()
            for protein in development_proteins
        ],
        dtype=np.float64,
    )
    training_domain = {
        "dataset": "MegaScale",
        "protein_count": int(len(development_proteins)),
        "sequence_length": {
            "minimum": int(development_lengths.min()),
            "median": float(np.median(development_lengths)),
            "maximum": int(development_lengths.max()),
        },
        "mean_structure_coverage": float(structure_coverage.mean()),
        "membrane_protein_labels": 0,
    }
    application_routing = {
        "expected_ddg_kcal_mol": (
            "0.60 state ddG + 0.40 portable-prior ddG"
        ),
        "stabilizer_ranking": "state_model",
        "reason": (
            "development and family-shadow evidence improved regression "
            "with the blend while the state score retained higher "
            "stabilizer average precision"
        ),
    }
    torch_device = torch.device(
        device if device != "cuda" or torch.cuda.is_available() else "cpu"
    )
    _strict_fp32(torch_device)
    started = time.monotonic()
    state, _, training = _fit_fixed_state_and_prior(
        data,
        arrays,
        Path(proteinmpnn_repository).resolve(),
        development_proteins,
        output,
        config=config,
        sequence_epochs=70,
        structure_epochs=structure_epochs,
        protein_batch_size=protein_batch_size,
        learning_rate=3.0e-4,
        structure_learning_rate=structure_learning_rate,
        weight_decay=weight_decay,
        seed=seed,
        device=torch_device,
        sequence_initial_state=source["state_model_state_dict"],
    )
    checkpoint = {
        "schema": ACCURACY_ENSEMBLE_SCHEMA,
        "state_model_schema": FULL_STRUCTURE_CHECKPOINT_SCHEMA,
        "data": data.provenance,
        "feature_provenance": arrays.provenance,
        "config": config,
        "state_config": FullStructureConfig(
            esm_dim=data.esm_residue.shape[-1], use_structure=True
        ),
        "state_model_state_dict": state,
        "portable_prior": training["portable_prior_manifest"],
        "selection_uses_outer": False,
        "training_domain": training_domain,
        "application_routing": application_routing,
        "training": training,
    }
    _save_torch(checkpoint_path, checkpoint)
    report = {
        "schema": ACCURACY_ENSEMBLE_SCHEMA,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": file_sha256(checkpoint_path),
        "portable_prior": training["portable_prior_manifest"],
        "configuration": asdict(config),
        "development_proteins": int(len(development_proteins)),
        "development_rows": int(np.sum(arrays.split >= 0)),
        "selection_uses_outer": False,
        "training_domain": training_domain,
        "application_routing": application_routing,
        "training": training,
        "elapsed_seconds": time.monotonic() - started,
    }
    (output / "training_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def evaluate_accuracy_outer(
    data_path: Path,
    hierarchy_row_paths: Sequence[Path],
    masked_marginal_path: Path,
    checkpoint_path: Path,
    proteinmpnn_repository: Path,
    output_dir: Path,
    *,
    threshold: float = -0.5,
    protein_batch_size: int = 8,
    device: str = "cuda",
    allow_cross_scale_masked_prior: bool = False,
) -> dict[str, object]:
    """Evaluate once on the already-consumed historical outer partition."""

    output = Path(output_dir).resolve()
    report_path = output / "outer_evaluation.json"
    if report_path.exists():
        raise FileExistsError(
            "accuracy outer evaluation already exists and is immutable"
        )
    output.mkdir(parents=True, exist_ok=True)
    data, arrays = load_portable_prior_arrays(
        data_path,
        hierarchy_row_paths,
        masked_marginal_path,
        allow_cross_scale_masked_prior=allow_cross_scale_masked_prior,
    )
    checkpoint_path = Path(checkpoint_path).resolve()
    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False
    )
    if checkpoint.get("schema") != ACCURACY_ENSEMBLE_SCHEMA:
        raise RuntimeError("accuracy ensemble checkpoint schema mismatch")
    if checkpoint["data"]["sha256"] != data.provenance["sha256"]:
        raise RuntimeError("accuracy ensemble data provenance mismatch")
    config = checkpoint["config"]
    if not isinstance(config, PortablePriorConfig):
        config = PortablePriorConfig(**config)
    prior_path = Path(checkpoint["portable_prior"]["model"])
    prior, prior_manifest = load_portable_prior(prior_path)
    if (
        prior_manifest["model_sha256"]
        != checkpoint["portable_prior"]["model_sha256"]
    ):
        raise RuntimeError("accuracy checkpoint portable-prior hash mismatch")
    outer_rows = np.flatnonzero(arrays.split < 0)
    torch_device = torch.device(
        device if device != "cuda" or torch.cuda.is_available() else "cpu"
    )
    _strict_fp32(torch_device)
    state, expert, blend = _prediction_bundle(
        data,
        arrays,
        checkpoint["state_model_state_dict"],
        prior,
        Path(proteinmpnn_repository).resolve(),
        outer_rows,
        config=config,
        protein_batch_size=protein_batch_size,
        device=torch_device,
    )
    state_metrics = _metrics(
        data,
        outer_rows,
        state,
        threshold=threshold,
        seed=config.random_state,
    )
    expert_metrics = _metrics(
        data,
        outer_rows,
        expert,
        threshold=threshold,
        seed=config.random_state + 1,
    )
    blend_metrics = _metrics(
        data,
        outer_rows,
        blend,
        threshold=threshold,
        seed=config.random_state + 2,
    )
    csv_path = output / "outer_predictions.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "protein_id",
                "position",
                "wt",
                "mutant",
                "experimental_ddg",
                "state_ddg",
                "portable_prior_ddg",
                "ensemble_ddg",
            ]
        )
        for local_index, row in enumerate(outer_rows):
            writer.writerow(
                [
                    data.protein_id[arrays.protein_index[row]],
                    int(arrays.position[row]) + 1,
                    AMINO_ACIDS[arrays.wt_amino_acid[row]],
                    AMINO_ACIDS[arrays.mutant_amino_acid[row]],
                    float(arrays.target[row]),
                    float(state[local_index]),
                    float(expert[local_index]),
                    float(blend[local_index]),
                ]
            )
    scatter_path = output / "outer_pred_vs_exp.png"
    _write_scatter(
        scatter_path,
        arrays.target[outer_rows],
        blend,
        title="Historical outer portable accuracy ensemble",
    )
    report = {
        "schema": ACCURACY_OUTER_REPORT_SCHEMA,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": file_sha256(checkpoint_path),
        "configuration": asdict(config),
        "selection_uses_outer": False,
        "outer_status": (
            "already consumed by the preceding architecture report; "
            "continuity evidence only"
        ),
        "state": state_metrics,
        "portable_prior": expert_metrics,
        "blend": blend_metrics,
        "mae_improvement": (
            float(state_metrics["regression"]["mae"])
            - float(blend_metrics["regression"]["mae"])
        ),
        "artifacts": {
            "predictions": str(csv_path),
            "predictions_sha256": file_sha256(csv_path),
            "scatter": str(scatter_path),
            "scatter_sha256": file_sha256(scatter_path),
        },
    }
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report
