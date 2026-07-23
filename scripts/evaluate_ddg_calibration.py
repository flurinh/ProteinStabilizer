#!/usr/bin/env python3
"""Select and gate a monotone odd ddG calibration on validation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from protein_stabilizer.embeddings import file_sha256
from protein_stabilizer.state_potential import (
    _load_state_rows,
    _predict_state,
    load_state_potential_ensemble,
)
from protein_stabilizer.training import regression_metrics
from protein_stabilizer.v2_training import (
    _predict,
    load_hierarchical_ensemble,
)


ROOT = Path(__file__).resolve().parents[1]


def _apply(
    prediction: np.ndarray,
    *,
    scale: float,
    power: float,
) -> np.ndarray:
    values = np.asarray(prediction, dtype=np.float64)
    return scale * np.sign(values) * np.abs(values) ** power


def _weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    order = np.argsort(values)
    sorted_values = values[order]
    cumulative = np.cumsum(weights[order])
    return float(sorted_values[np.searchsorted(cumulative, cumulative[-1] / 2)])


def _candidate_scales(
    transformed: np.ndarray,
    target: np.ndarray,
) -> list[float]:
    denominator = float(np.dot(transformed, transformed))
    rmse_scale = float(np.dot(transformed, target) / denominator)
    nonzero = np.abs(transformed) > 1e-12
    mae_scale = _weighted_median(
        target[nonzero] / transformed[nonzero],
        np.abs(transformed[nonzero]),
    )
    values = {
        1.0,
        rmse_scale,
        mae_scale,
        0.5 * (rmse_scale + mae_scale),
    }
    return sorted(value for value in values if value > 0.0)


def _selection_metrics(
    target: np.ndarray,
    prediction: np.ndarray,
    baseline: dict[str, float],
) -> dict[str, float]:
    metrics = regression_metrics(target, prediction)
    metrics["normalized_error"] = 0.5 * (
        float(metrics["mae"]) / float(baseline["mae"])
        + float(metrics["rmse"]) / float(baseline["rmse"])
    )
    return metrics


def _select_calibration(
    target: np.ndarray,
    prediction: np.ndarray,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    baseline = regression_metrics(target, prediction)
    candidates: list[dict[str, object]] = []
    for power in np.linspace(0.70, 1.50, 81):
        transformed = np.sign(prediction) * np.abs(prediction) ** power
        for scale in _candidate_scales(transformed, target):
            calibrated = scale * transformed
            metrics = _selection_metrics(target, calibrated, baseline)
            candidates.append(
                {
                    "scale": float(scale),
                    "power": float(power),
                    "metrics": metrics,
                }
            )
    selected = min(
        candidates,
        key=lambda value: (
            float(value["metrics"]["normalized_error"]),
            abs(float(value["power"]) - 1.0),
            abs(float(value["scale"]) - 1.0),
        ),
    )
    return selected, candidates


def _predict_split(
    row_path: Path,
    cache_path: Path,
    baseline: torch.nn.Module,
    state: torch.nn.Module,
    *,
    cache_sha256: str,
    use_structure: bool,
    state_weight: float,
    split: str,
    batch_size: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    data = _load_state_rows(
        row_path,
        cache_path,
        expected_cache_sha256=cache_sha256,
    )
    indices = np.flatnonzero(data.hierarchy.split == split)
    baseline_prediction, _ = _predict(
        baseline,
        data.hierarchy,
        indices,
        use_structure=use_structure,
        batch_size=batch_size,
        device=device,
    )
    state_prediction = _predict_state(
        state,
        data,
        indices,
        use_structure=use_structure,
        batch_size=batch_size,
        device=device,
    )
    prediction = (
        (1.0 - state_weight) * baseline_prediction
        + state_weight * state_prediction
    )
    return data.hierarchy.target[indices], prediction


def evaluate(args: argparse.Namespace) -> dict[str, object]:
    device = torch.device(
        args.device
        if args.device.startswith("cuda") and torch.cuda.is_available()
        else "cpu"
    )
    if device.type == "cuda":
        torch.set_float32_matmul_precision("highest")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    cache_sha256 = file_sha256(args.cache)
    baseline, baseline_payload = load_hierarchical_ensemble(
        args.baseline_checkpoint, device
    )
    state, state_payload = load_state_potential_ensemble(
        args.state_checkpoint, device
    )
    if str(state_payload["cache_sha256"]) != cache_sha256:
        raise RuntimeError("state checkpoint hierarchy cache mismatch")
    if file_sha256(args.baseline_checkpoint) != str(
        state_payload["baseline_checkpoint_sha256"]
    ):
        raise RuntimeError("state checkpoint baseline mismatch")
    use_structure = bool(state_payload["use_structure"])
    if bool(baseline_payload["use_structure"]) != use_structure:
        raise RuntimeError("baseline/state structure policy mismatch")
    state_weight = float(state_payload["state_potential_weight"])
    validation_target, validation_prediction = _predict_split(
        args.features / "single_train.h5",
        args.cache,
        baseline,
        state,
        cache_sha256=cache_sha256,
        use_structure=use_structure,
        state_weight=state_weight,
        split="val",
        batch_size=args.batch_size,
        device=device,
    )
    selected, candidates = _select_calibration(
        validation_target, validation_prediction
    )
    test_target, test_prediction = _predict_split(
        args.features / "single_test.h5",
        args.cache,
        baseline,
        state,
        cache_sha256=cache_sha256,
        use_structure=use_structure,
        state_weight=state_weight,
        split="test",
        batch_size=args.batch_size,
        device=device,
    )
    scale = float(selected["scale"])
    power = float(selected["power"])
    validation_calibrated = _apply(
        validation_prediction, scale=scale, power=power
    )
    test_calibrated = _apply(
        test_prediction, scale=scale, power=power
    )
    validation_before = regression_metrics(
        validation_target, validation_prediction
    )
    validation_after = regression_metrics(
        validation_target, validation_calibrated
    )
    test_before = regression_metrics(test_target, test_prediction)
    test_after = regression_metrics(test_target, test_calibrated)
    maximum = float(max(np.abs(test_prediction).max(), 1.0))
    grid = np.linspace(-maximum, maximum, 10001)
    transformed_grid = _apply(grid, scale=scale, power=power)
    monotone = bool(np.all(np.diff(transformed_grid) >= 0.0))
    reverse_error = float(
        np.max(
            np.abs(
                _apply(test_prediction, scale=scale, power=power)
                + _apply(-test_prediction, scale=scale, power=power)
            )
        )
    )
    promotion_gates = {
        "validation_mae_improves": (
            float(validation_after["mae"])
            < float(validation_before["mae"])
        ),
        "validation_rmse_improves": (
            float(validation_after["rmse"])
            < float(validation_before["rmse"])
        ),
        "strictly_monotone": monotone,
        "exact_reversal": reverse_error <= 1e-12,
    }
    report = {
        "schema": "protein-stabilizer.ddg-calibration-audit.v1",
        "decision": {
            "status": (
                "promoted"
                if all(promotion_gates.values())
                else "rejected"
            ),
            "promotion_gates": promotion_gates,
            "policy": (
                "validation error and exact calibration algebra only; "
                "historical test metrics are reporting-only"
            ),
        },
        "calibration": {
            "formula": "scale * sign(raw_ddg) * abs(raw_ddg) ** power",
            "scale": scale,
            "power": power,
            "selected_on": "protein-held-out validation only",
            "candidate_powers": 81,
            "candidate_parameterizations": len(candidates),
            "preserves_zero": True,
            "preserves_order": monotone,
            "maximum_forward_plus_reverse": reverse_error,
        },
        "validation": {
            "rows": len(validation_target),
            "before": validation_before,
            "after": validation_after,
        },
        "frozen_test": {
            "rows": len(test_target),
            "before": test_before,
            "after": test_after,
        },
        "provenance": {
            "features": str(args.features.resolve()),
            "validation_features_sha256": file_sha256(
                args.features / "single_train.h5"
            ),
            "test_features_sha256": file_sha256(
                args.features / "single_test.h5"
            ),
            "cache_sha256": cache_sha256,
            "baseline_checkpoint_sha256": file_sha256(
                args.baseline_checkpoint
            ),
            "state_checkpoint_sha256": file_sha256(
                args.state_checkpoint
            ),
            "precision": "native FP32; TF32 disabled",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--features",
        type=Path,
        default=ROOT / "artifacts/features_v2_6b_fp32",
    )
    parser.add_argument(
        "--cache",
        type=Path,
        default=ROOT / "embeddings/esmc_6b/hierarchy_cache_fp32.h5",
    )
    parser.add_argument(
        "--baseline-checkpoint",
        type=Path,
        default=(
            ROOT / "checkpoints/esmc_6b_v2/hierarchy_selected_ensemble.pt"
        ),
    )
    parser.add_argument(
        "--state-checkpoint",
        type=Path,
        default=(
            ROOT
            / "checkpoints/esmc_6b_state_potential_fp32/"
            "state_potential_ensemble.pt"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "docs/ddg_calibration_audit.json",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=256)
    return parser


if __name__ == "__main__":
    print(json.dumps(evaluate(build_parser().parse_args()), indent=2))
