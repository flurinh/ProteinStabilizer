"""Portable structural-prior features and regression for single mutations."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import joblib
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor

from .data import AMINO_ACIDS
from .embeddings import file_sha256


PORTABLE_PRIOR_FEATURE_SCHEMA = (
    "protein-stabilizer.portable-structural-prior-features.v1"
)
PORTABLE_PRIOR_MODEL_SCHEMA = (
    "protein-stabilizer.portable-structural-prior-model.v1"
)
AFFINE_DDG_CALIBRATION_SCHEMA = (
    "protein-stabilizer.affine-ddg-calibration.v1"
)
MULTISCALE_AFFINE_DDG_CALIBRATION_SCHEMA = (
    "protein-stabilizer.multiscale-affine-ddg-calibration.v1"
)
PROTEINMPNN_MULTISCALE_DDG_CALIBRATION_SCHEMA = (
    "protein-stabilizer.proteinmpnn-multiscale-ddg-calibration.v1"
)
MASKED_CHEMISTRY_DIMENSION = 70
BACKBONE_GEOMETRY_DIMENSION = 18
PROTEINMPNN_DIMENSION = 128
PORTABLE_PRIOR_DIMENSION = (
    MASKED_CHEMISTRY_DIMENSION
    + BACKBONE_GEOMETRY_DIMENSION
    + PROTEINMPNN_DIMENSION
)


@dataclass(frozen=True)
class PortablePriorConfig:
    """Frozen configuration selected before confirmation folds 1--4."""

    n_estimators: int = 600
    max_features: float = 0.8
    min_samples_leaf: int = 10
    random_state: int = 20260724
    state_weight: float = 0.60
    expert_weight: float = 0.40

    def __post_init__(self) -> None:
        if self.n_estimators < 1:
            raise ValueError("portable prior requires at least one tree")
        if not 0.0 < self.max_features <= 1.0:
            raise ValueError("max_features must be in (0, 1]")
        if self.min_samples_leaf < 1:
            raise ValueError("min_samples_leaf must be positive")
        if not np.isclose(self.state_weight + self.expert_weight, 1.0):
            raise ValueError("state and expert weights must sum to one")
        if min(self.state_weight, self.expert_weight) < 0.0:
            raise ValueError("ensemble weights cannot be negative")


def backbone_geometry_features(
    coordinates: np.ndarray,
    structure_mask: np.ndarray,
    length: np.ndarray,
) -> np.ndarray:
    """Build target-independent residue geometry from N/CA/C/O backbones.

    The output contains four local contact counts, three nonlocal contact
    counts, five nearest-neighbour distances, mean CA distance, radial depth,
    local coordinate coverage, relative sequence position, terminal distance,
    and an explicit structure-known flag.
    """

    coordinate = np.asarray(coordinates, dtype=np.float32)
    mask = np.asarray(structure_mask, dtype=bool)
    lengths = np.asarray(length, dtype=np.int64)
    if coordinate.ndim != 4 or coordinate.shape[2:] != (4, 3):
        raise ValueError("coordinates must have shape [protein, length, 4, 3]")
    if mask.shape != coordinate.shape[:2]:
        raise ValueError("structure mask shape does not match coordinates")
    if lengths.shape != (len(coordinate),):
        raise ValueError("length must contain one value per protein")
    if np.any(lengths < 1) or np.any(lengths > coordinate.shape[1]):
        raise ValueError("protein length is outside the coordinate tensor")
    if np.any(~np.isfinite(coordinate[mask])):
        raise ValueError("active structure coordinates must be finite")

    result = np.zeros(
        (*coordinate.shape[:2], BACKBONE_GEOMETRY_DIMENSION),
        dtype=np.float32,
    )
    for protein_index, protein_length in enumerate(lengths):
        count = int(protein_length)
        active = mask[protein_index, :count]
        active_index = np.flatnonzero(active)
        if not len(active_index):
            continue
        alpha_carbon = coordinate[protein_index, :count, 1]
        points = alpha_carbon[active_index]
        distance = np.linalg.norm(
            points[:, None, :] - points[None, :, :],
            axis=-1,
        )
        sequence_distance = np.abs(
            active_index[:, None] - active_index[None, :]
        )
        centroid = points.mean(axis=0)
        radius = float(
            np.sqrt(np.mean(np.sum((points - centroid) ** 2, axis=1)))
        )
        radius = max(radius, 1.0e-6)
        row_index = np.arange(len(active_index))
        for local_index, residue_index in enumerate(active_index):
            nonself = row_index != local_index
            nonlocal_residue = sequence_distance[local_index] > 2
            neighbour_distance = np.sort(distance[local_index, nonself])
            nearest = np.full(5, 20.0, dtype=np.float32)
            nearest[: min(5, len(neighbour_distance))] = (
                neighbour_distance[:5]
            )
            window = active[
                max(0, residue_index - 4) : min(count, residue_index + 5)
            ]
            values: Sequence[float] = (
                *(
                    float(
                        np.sum(
                            (distance[local_index] < threshold) & nonself
                        )
                    )
                    for threshold in (6.0, 8.0, 10.0, 12.0)
                ),
                *(
                    float(
                        np.sum(
                            (distance[local_index] < threshold)
                            & nonlocal_residue
                        )
                    )
                    for threshold in (8.0, 10.0, 12.0)
                ),
                *nearest.tolist(),
                (
                    float(distance[local_index, nonself].mean())
                    if np.any(nonself)
                    else 0.0
                ),
                float(
                    np.linalg.norm(alpha_carbon[residue_index] - centroid)
                    / radius
                ),
                float(window.mean()),
                float(residue_index / max(1, count - 1)),
                float(
                    min(residue_index, count - 1 - residue_index)
                    / max(1, count - 1)
                ),
                1.0,
            )
            result[protein_index, residue_index] = np.asarray(
                values, dtype=np.float32
            )
    return result


def masked_chemistry_features(
    masked_log_probabilities: np.ndarray,
    wt_amino_acid: np.ndarray,
    mutant_amino_acid: np.ndarray,
    position_one_based: np.ndarray,
    protein_length: np.ndarray,
) -> np.ndarray:
    """Encode masked ESM-C state evidence and mutation direction."""

    log_probability = np.asarray(
        masked_log_probabilities, dtype=np.float32
    )
    wt = np.asarray(wt_amino_acid, dtype=np.int64)
    mutant = np.asarray(mutant_amino_acid, dtype=np.int64)
    position = np.asarray(position_one_based, dtype=np.float32)
    length = np.asarray(protein_length, dtype=np.float32)
    rows = len(log_probability)
    if log_probability.shape != (rows, len(AMINO_ACIDS)):
        raise ValueError("masked log probabilities must have shape [rows, 20]")
    for name, value in (
        ("WT amino acid", wt),
        ("mutant amino acid", mutant),
        ("position", position),
        ("length", length),
    ):
        if value.shape != (rows,):
            raise ValueError(f"{name} must have one value per mutation row")
    if np.any((wt < 0) | (wt >= len(AMINO_ACIDS))):
        raise ValueError("WT amino-acid index is outside the canonical alphabet")
    if np.any((mutant < 0) | (mutant >= len(AMINO_ACIDS))):
        raise ValueError(
            "mutant amino-acid index is outside the canonical alphabet"
        )
    if np.any(length < 1) or np.any(position < 1) or np.any(position > length):
        raise ValueError("mutation position is outside the protein sequence")
    if np.any(~np.isfinite(log_probability)):
        raise ValueError("masked log probabilities must be finite")

    row = np.arange(rows)
    wt_log_probability = log_probability[row, wt]
    mutant_log_probability = log_probability[row, mutant]
    probability = np.exp(log_probability)
    entropy = -np.sum(probability * log_probability, axis=1)
    wt_rank = (
        np.sum(log_probability > wt_log_probability[:, None], axis=1) / 19.0
    )
    mutant_rank = (
        np.sum(
            log_probability > mutant_log_probability[:, None], axis=1
        )
        / 19.0
    )
    one_hot = np.eye(len(AMINO_ACIDS), dtype=np.float32)
    relative_position = position / length
    terminal_distance = np.minimum(position - 1.0, length - position) / length
    result = np.column_stack(
        [
            mutant_log_probability - wt_log_probability,
            wt_log_probability,
            mutant_log_probability,
            entropy,
            np.max(log_probability, axis=1),
            wt_rank,
            mutant_rank,
            relative_position,
            terminal_distance,
            np.log(length),
            one_hot[wt],
            one_hot[mutant],
            one_hot[mutant] - one_hot[wt],
        ]
    ).astype(np.float32)
    if result.shape != (rows, MASKED_CHEMISTRY_DIMENSION):
        raise RuntimeError("masked chemistry feature dimension drifted")
    return result


def portable_prior_features(
    masked_log_probabilities: np.ndarray,
    wt_amino_acid: np.ndarray,
    mutant_amino_acid: np.ndarray,
    position_zero_based: np.ndarray,
    protein_index: np.ndarray,
    protein_length: np.ndarray,
    geometry: np.ndarray,
    proteinmpnn_residue: np.ndarray,
) -> np.ndarray:
    """Assemble the frozen 216-dimensional portable prior feature vector."""

    position = np.asarray(position_zero_based, dtype=np.int64)
    protein = np.asarray(protein_index, dtype=np.int64)
    lengths = np.asarray(protein_length, dtype=np.int64)
    geometry_array = np.asarray(geometry, dtype=np.float32)
    proteinmpnn = np.asarray(proteinmpnn_residue, dtype=np.float32)
    rows = len(position)
    if protein.shape != (rows,):
        raise ValueError("protein index must have one value per mutation row")
    if np.any((protein < 0) | (protein >= len(lengths))):
        raise ValueError("mutation references an unknown protein")
    if np.any((position < 0) | (position >= lengths[protein])):
        raise ValueError("mutation position is outside its protein")
    expected_prefix = (len(lengths), geometry_array.shape[1])
    if geometry_array.shape != (
        *expected_prefix,
        BACKBONE_GEOMETRY_DIMENSION,
    ):
        raise ValueError("backbone geometry tensor has the wrong shape")
    if proteinmpnn.shape == (
        len(lengths),
        geometry_array.shape[1],
        PROTEINMPNN_DIMENSION,
    ):
        proteinmpnn = proteinmpnn[protein, position]
    elif proteinmpnn.shape != (rows, PROTEINMPNN_DIMENSION):
        raise ValueError(
            "ProteinMPNN features must be per-residue or row-aligned"
        )
    chemistry = masked_chemistry_features(
        masked_log_probabilities,
        wt_amino_acid,
        mutant_amino_acid,
        position + 1,
        lengths[protein],
    )
    result = np.column_stack(
        [
            chemistry,
            geometry_array[protein, position],
            proteinmpnn,
        ]
    ).astype(np.float32)
    if result.shape != (rows, PORTABLE_PRIOR_DIMENSION):
        raise RuntimeError("portable prior feature dimension drifted")
    if np.any(~np.isfinite(result)):
        raise ValueError("portable prior features must be finite")
    return result


def fit_portable_prior(
    features: np.ndarray,
    target: np.ndarray,
    *,
    config: PortablePriorConfig = PortablePriorConfig(),
) -> ExtraTreesRegressor:
    """Fit the deterministic complementary expert."""

    feature = np.asarray(features, dtype=np.float32)
    values = np.asarray(target, dtype=np.float32)
    if feature.ndim != 2 or feature.shape[1] != PORTABLE_PRIOR_DIMENSION:
        raise ValueError("portable prior training feature dimension mismatch")
    if values.shape != (len(feature),):
        raise ValueError("portable prior target shape mismatch")
    if np.any(~np.isfinite(feature)) or np.any(~np.isfinite(values)):
        raise ValueError("portable prior training data must be finite")
    model = ExtraTreesRegressor(
        n_estimators=config.n_estimators,
        max_features=config.max_features,
        min_samples_leaf=config.min_samples_leaf,
        n_jobs=-1,
        random_state=config.random_state,
    )
    model.fit(feature, values)
    return model


def predict_portable_prior(
    model: ExtraTreesRegressor,
    features: np.ndarray,
    wt_amino_acid: np.ndarray,
    mutant_amino_acid: np.ndarray,
) -> np.ndarray:
    """Predict forward WT-conditioned ddG and force self mutations to zero."""

    feature = np.asarray(features, dtype=np.float32)
    wt = np.asarray(wt_amino_acid, dtype=np.int64)
    mutant = np.asarray(mutant_amino_acid, dtype=np.int64)
    if feature.ndim != 2 or feature.shape[1] != PORTABLE_PRIOR_DIMENSION:
        raise ValueError("portable prior prediction feature dimension mismatch")
    if wt.shape != (len(feature),) or mutant.shape != (len(feature),):
        raise ValueError("amino-acid arrays must align with prediction rows")
    prediction = np.asarray(model.predict(feature), dtype=np.float32)
    prediction[wt == mutant] = 0.0
    return prediction


def blend_state_and_prior(
    state_prediction: np.ndarray,
    prior_prediction: np.ndarray,
    *,
    config: PortablePriorConfig = PortablePriorConfig(),
) -> np.ndarray:
    """Apply the frozen 60/40 accuracy ensemble."""

    state = np.asarray(state_prediction, dtype=np.float32)
    prior = np.asarray(prior_prediction, dtype=np.float32)
    if state.shape != prior.shape:
        raise ValueError("state and portable-prior predictions must align")
    if np.any(~np.isfinite(state)) or np.any(~np.isfinite(prior)):
        raise ValueError("accuracy ensemble predictions must be finite")
    return (
        config.state_weight * state + config.expert_weight * prior
    ).astype(np.float32)


def apply_affine_ddg_calibration(
    state_prediction: np.ndarray,
    prior_prediction: np.ndarray,
    calibration: dict[str, object],
    *,
    self_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Apply a selected monotone affine calibration to the two components."""

    if calibration.get("schema") != AFFINE_DDG_CALIBRATION_SCHEMA:
        raise RuntimeError("affine ddG calibration schema mismatch")
    state = np.asarray(state_prediction, dtype=np.float32)
    prior = np.asarray(prior_prediction, dtype=np.float32)
    if state.shape != prior.shape:
        raise ValueError("state and portable-prior predictions must align")
    coefficient = calibration.get("coefficients")
    if not isinstance(coefficient, dict):
        raise RuntimeError("affine ddG calibration lacks coefficients")
    state_scale = float(coefficient["state"])
    prior_scale = float(coefficient["portable_prior"])
    intercept = float(coefficient["intercept"])
    if (
        state_scale < 0.0
        or prior_scale < 0.0
        or not np.isfinite([state_scale, prior_scale, intercept]).all()
    ):
        raise RuntimeError("affine ddG calibration is not monotone/finite")
    result = (
        state_scale * state + prior_scale * prior + intercept
    ).astype(np.float32)
    if self_mask is not None:
        mask = np.asarray(self_mask, dtype=bool)
        if mask.shape != result.shape:
            raise ValueError("self-mutation mask must align with predictions")
        result[mask] = 0.0
    if np.any(~np.isfinite(result)):
        raise RuntimeError("affine ddG calibration produced invalid values")
    return result


def apply_multiscale_affine_ddg_calibration(
    primary_state_prediction: np.ndarray,
    secondary_state_prediction: np.ndarray,
    prior_prediction: np.ndarray,
    calibration: dict[str, object],
    *,
    self_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Apply the selected monotone 6B/600M/portable-prior calibration."""

    if (
        calibration.get("schema")
        != MULTISCALE_AFFINE_DDG_CALIBRATION_SCHEMA
    ):
        raise RuntimeError("multiscale affine ddG calibration schema mismatch")
    primary = np.asarray(primary_state_prediction, dtype=np.float32)
    secondary = np.asarray(secondary_state_prediction, dtype=np.float32)
    prior = np.asarray(prior_prediction, dtype=np.float32)
    if primary.shape != secondary.shape or primary.shape != prior.shape:
        raise ValueError(
            "primary, secondary, and portable-prior predictions must align"
        )
    coefficient = calibration.get("coefficients")
    if not isinstance(coefficient, dict):
        raise RuntimeError(
            "multiscale affine ddG calibration lacks coefficients"
        )
    primary_scale = float(coefficient["primary_state"])
    secondary_scale = float(coefficient["secondary_state"])
    prior_scale = float(coefficient["portable_prior"])
    intercept = float(coefficient["intercept"])
    values = np.asarray(
        [primary_scale, secondary_scale, prior_scale, intercept],
        dtype=np.float64,
    )
    if np.any(values[:3] < 0.0) or not np.isfinite(values).all():
        raise RuntimeError(
            "multiscale affine ddG calibration is not monotone/finite"
        )
    result = (
        primary_scale * primary
        + secondary_scale * secondary
        + prior_scale * prior
        + intercept
    ).astype(np.float32)
    if self_mask is not None:
        mask = np.asarray(self_mask, dtype=bool)
        if mask.shape != result.shape:
            raise ValueError("self-mutation mask must align with predictions")
        result[mask] = 0.0
    if np.any(~np.isfinite(result)):
        raise RuntimeError(
            "multiscale affine ddG calibration produced invalid values"
        )
    return result


def apply_proteinmpnn_multiscale_ddg_calibration(
    primary_state_prediction: np.ndarray,
    secondary_state_prediction: np.ndarray,
    prior_prediction: np.ndarray,
    proteinmpnn_prediction: np.ndarray,
    calibration: dict[str, object],
    *,
    self_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Apply the selected 6B/600M/prior/ProteinMPNN calibration."""

    if (
        calibration.get("schema")
        != PROTEINMPNN_MULTISCALE_DDG_CALIBRATION_SCHEMA
    ):
        raise RuntimeError(
            "ProteinMPNN multiscale ddG calibration schema mismatch"
        )
    primary = np.asarray(primary_state_prediction, dtype=np.float32)
    secondary = np.asarray(secondary_state_prediction, dtype=np.float32)
    prior = np.asarray(prior_prediction, dtype=np.float32)
    proteinmpnn = np.asarray(proteinmpnn_prediction, dtype=np.float32)
    if not (
        primary.shape
        == secondary.shape
        == prior.shape
        == proteinmpnn.shape
    ):
        raise ValueError(
            "primary, secondary, portable-prior, and ProteinMPNN "
            "predictions must align"
        )
    coefficient = calibration.get("coefficients")
    if not isinstance(coefficient, dict):
        raise RuntimeError(
            "ProteinMPNN multiscale ddG calibration lacks coefficients"
        )
    primary_scale = float(coefficient["primary_state"])
    secondary_scale = float(coefficient["secondary_state"])
    prior_scale = float(coefficient["portable_prior"])
    proteinmpnn_scale = float(
        coefficient["proteinmpnn_leave_one_out"]
    )
    intercept = float(coefficient["intercept"])
    values = np.asarray(
        [
            primary_scale,
            secondary_scale,
            prior_scale,
            proteinmpnn_scale,
            intercept,
        ],
        dtype=np.float64,
    )
    if np.any(values[:4] < 0.0) or not np.isfinite(values).all():
        raise RuntimeError(
            "ProteinMPNN multiscale ddG calibration is not monotone/finite"
        )
    result = (
        primary_scale * primary
        + secondary_scale * secondary
        + prior_scale * prior
        + proteinmpnn_scale * proteinmpnn
        + intercept
    ).astype(np.float32)
    if self_mask is not None:
        mask = np.asarray(self_mask, dtype=bool)
        if mask.shape != result.shape:
            raise ValueError("self-mutation mask must align with predictions")
        result[mask] = 0.0
    if np.any(~np.isfinite(result)):
        raise RuntimeError(
            "ProteinMPNN multiscale ddG calibration produced invalid values"
        )
    return result


def save_portable_prior(
    path: Path,
    model: ExtraTreesRegressor,
    *,
    config: PortablePriorConfig,
    provenance: dict[str, object],
) -> dict[str, object]:
    """Write an atomic joblib artifact plus a hash-bound JSON manifest."""

    output = Path(path).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_suffix(output.suffix + ".partial")
    partial.unlink(missing_ok=True)
    joblib.dump(model, partial, compress=3)
    partial.replace(output)
    manifest = {
        "schema": PORTABLE_PRIOR_MODEL_SCHEMA,
        "model": str(output),
        "model_sha256": file_sha256(output),
        "feature_schema": PORTABLE_PRIOR_FEATURE_SCHEMA,
        "feature_dimension": PORTABLE_PRIOR_DIMENSION,
        "config": asdict(config),
        "sign_convention": "negative ddG means stabilizing",
        "semantics": (
            "forward WT-conditioned empirical correction; self mutations "
            "are forced to zero; the state-model component retains exact "
            "thermodynamic algebra"
        ),
        "provenance": provenance,
    }
    manifest_path = output.with_suffix(output.suffix + ".json")
    manifest_partial = manifest_path.with_suffix(
        manifest_path.suffix + ".partial"
    )
    manifest_partial.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    manifest_partial.replace(manifest_path)
    return manifest


def load_portable_prior(
    path: Path,
) -> tuple[ExtraTreesRegressor, dict[str, object]]:
    """Load a portable-prior artifact after schema and hash validation."""

    model_path = Path(path).resolve()
    manifest_path = model_path.with_suffix(model_path.suffix + ".json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != PORTABLE_PRIOR_MODEL_SCHEMA:
        raise RuntimeError("portable prior model schema mismatch")
    if manifest.get("feature_schema") != PORTABLE_PRIOR_FEATURE_SCHEMA:
        raise RuntimeError("portable prior feature schema mismatch")
    if int(manifest.get("feature_dimension", -1)) != PORTABLE_PRIOR_DIMENSION:
        raise RuntimeError("portable prior feature dimension mismatch")
    if manifest.get("model_sha256") != file_sha256(model_path):
        raise RuntimeError("portable prior model hash mismatch")
    model = joblib.load(model_path)
    if not isinstance(model, ExtraTreesRegressor):
        raise RuntimeError("portable prior artifact has the wrong model type")
    return model, manifest
