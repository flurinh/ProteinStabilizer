#!/usr/bin/env python3
"""Export exact held-out experimental/predicted ddG pairs for the dashboard."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
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


def export_accuracy(args: argparse.Namespace) -> dict[str, object]:
    """Export the current family-OOF hybrid scatter without test selection."""

    if args.max_points < 1:
        raise ValueError("max-points must be positive")
    source = Path(args.accuracy_predictions).resolve()
    frame = pd.read_csv(source)
    required = {
        "protein_id",
        "fold",
        "experimental_ddg",
        "ensemble_ddg",
    }
    missing = required - set(frame)
    if missing:
        raise RuntimeError(
            f"accuracy prediction table lacks {sorted(missing)}"
        )
    target = frame["experimental_ddg"].to_numpy(dtype=np.float32)
    prediction_column = next(
        (
            name
            for name in (
                "multiscale_calibrated_ddg",
                "calibrated_ddg",
                "ensemble_ddg",
            )
            if name in frame
        )
    )
    prediction = frame[prediction_column].to_numpy(dtype=np.float32)
    if (
        len(target) == 0
        or not np.isfinite(target).all()
        or not np.isfinite(prediction).all()
    ):
        raise RuntimeError("accuracy scatter contains invalid values")
    metrics = regression_metrics(target, prediction)
    scale_audit_path = (
        ROOT / "docs/esmc6b_accuracy_scale_audit.json"
    ).resolve()
    scale_audit = json.loads(
        scale_audit_path.read_text(encoding="utf-8")
    )
    calibration_audit_path = (
        ROOT
        / (
            "docs/esmc6b_multiscale_accuracy_audit.json"
            if prediction_column == "multiscale_calibrated_ddg"
            else "docs/esmc6b_accuracy_calibration_audit.json"
        )
    ).resolve()
    if prediction_column in {
        "calibrated_ddg",
        "multiscale_calibrated_ddg",
    }:
        if not calibration_audit_path.is_file():
            raise FileNotFoundError(calibration_audit_path)
        calibration_audit = json.loads(
            calibration_audit_path.read_text(encoding="utf-8")
        )
        candidate_key = (
            "candidate"
            if prediction_column == "multiscale_calibrated_ddg"
            else "calibrated"
        )
        calibrated_metrics = calibration_audit["metrics"][
            "all_development_oof"
        ][candidate_key]
        metrics["protein_cluster_bootstrap_mae_95_ci"] = (
            calibrated_metrics[
                "protein_cluster_bootstrap_mae_95_ci"
            ]
        )
        prediction_description = calibration_audit["calibration"]["formula"]
    else:
        calibration_audit = None
        metrics["protein_cluster_bootstrap_mae_95_ci"] = (
            scale_audit["development_oof"]["blend"][
                "protein_cluster_bootstrap_mae_95_ci"
            ]
        )
        prediction_description = (
            "0.60 6B state + 0.40 600M portable prior"
        )
    slope, intercept = np.polyfit(target, prediction, deg=1)
    axis_min = float(min(target.min(), prediction.min()))
    axis_max = float(max(target.max(), prediction.max()))
    padding = 0.04 * (axis_max - axis_min)
    maximum = min(int(args.max_points), len(target))
    rng = np.random.default_rng(int(args.seed))
    plotted = np.sort(
        rng.choice(len(target), size=maximum, replace=False)
    )
    report = {
        "schema": "protein-stabilizer.ddg-scatter.v2",
        "split": (
            "five family-held-out MegaScale development folds; "
            "historical outer excluded"
        ),
        "units": "kcal/mol",
        "sign_convention": "negative ddG is stabilizing",
        "evaluated_rows": len(target),
        "rows": len(plotted),
        "metrics": metrics,
        "calibration": {
            "fit": "predicted = slope * experimental + intercept",
            "slope": float(slope),
            "intercept": float(intercept),
        },
        "axes": {
            "minimum": axis_min - padding,
            "maximum": axis_max + padding,
            "equal_scale": True,
            "clipped": False,
        },
        "experimental_ddg": np.round(target[plotted], 5).tolist(),
        "predicted_ddg": np.round(prediction[plotted], 5).tolist(),
        "sampling": {
            "method": "uniform rows without replacement",
            "seed": int(args.seed),
            "maximum_points": int(args.max_points),
            "metrics_use_all_rows": True,
        },
        "provenance": {
            "predictions": str(source),
            "predictions_sha256": file_sha256(source),
            "scale_audit": str(
                scale_audit_path
            ),
            "scale_audit_schema": scale_audit["schema"],
            "prediction_column": prediction_column,
            "prediction": prediction_description,
            "precision": "native FP32 state features; TF32 disabled",
            "outer_used": False,
            "calibration_audit": (
                None
                if calibration_audit is None
                else str(calibration_audit_path)
            ),
            "calibration_audit_sha256": (
                None
                if calibration_audit is None
                else file_sha256(calibration_audit_path)
            ),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, separators=(",", ":")),
        encoding="utf-8",
    )
    return {
        key: value
        for key, value in report.items()
        if key not in {"experimental_ddg", "predicted_ddg"}
    }


def export(args: argparse.Namespace) -> dict[str, object]:
    if args.accuracy_predictions is not None:
        return export_accuracy(args)
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
    test_data = _load_state_rows(
        args.features / "single_test.h5",
        args.cache,
        expected_cache_sha256=cache_sha256,
    )
    indices = np.arange(
        len(test_data.hierarchy.target), dtype=np.int64
    )
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
    baseline_prediction, _ = _predict(
        baseline,
        test_data.hierarchy,
        indices,
        use_structure=use_structure,
        batch_size=args.batch_size,
        device=device,
    )
    state_prediction = _predict_state(
        state,
        test_data,
        indices,
        use_structure=use_structure,
        batch_size=args.batch_size,
        device=device,
    )
    state_weight = float(state_payload["state_potential_weight"])
    prediction = (
        (1.0 - state_weight) * baseline_prediction
        + state_weight * state_prediction
    ).astype(np.float32)
    target = test_data.hierarchy.target.astype(np.float32)
    metrics = regression_metrics(target, prediction)
    slope, intercept = np.polyfit(target, prediction, deg=1)
    axis_min = float(min(target.min(), prediction.min()))
    axis_max = float(max(target.max(), prediction.max()))
    padding = 0.04 * (axis_max - axis_min)
    report = {
        "schema": "protein-stabilizer.ddg-scatter.v1",
        "split": "frozen MegaScale protein-held-out test",
        "units": "kcal/mol",
        "sign_convention": "negative ddG is stabilizing",
        "rows": len(target),
        "metrics": metrics,
        "calibration": {
            "fit": "predicted = slope * experimental + intercept",
            "slope": float(slope),
            "intercept": float(intercept),
        },
        "axes": {
            "minimum": axis_min - padding,
            "maximum": axis_max + padding,
            "equal_scale": True,
            "clipped": False,
        },
        "experimental_ddg": np.round(target, 5).tolist(),
        "predicted_ddg": np.round(prediction, 5).tolist(),
        "provenance": {
            "features": str((args.features / "single_test.h5").resolve()),
            "features_sha256": file_sha256(
                args.features / "single_test.h5"
            ),
            "cache": str(args.cache.resolve()),
            "cache_sha256": cache_sha256,
            "baseline_checkpoint": str(
                args.baseline_checkpoint.resolve()
            ),
            "baseline_checkpoint_sha256": file_sha256(
                args.baseline_checkpoint
            ),
            "state_checkpoint": str(args.state_checkpoint.resolve()),
            "state_checkpoint_sha256": file_sha256(
                args.state_checkpoint
            ),
            "state_potential_weight": state_weight,
            "precision": "native FP32; TF32 disabled",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, separators=(",", ":")),
        encoding="utf-8",
    )
    return {
        key: value
        for key, value in report.items()
        if key not in {"experimental_ddg", "predicted_ddg"}
    }


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
        default=ROOT / "docs/generic_ddg_scatter.json",
    )
    parser.add_argument(
        "--accuracy-predictions",
        type=Path,
        help=(
            "export current family-OOF hybrid predictions instead of "
            "rerunning the historical state model"
        ),
    )
    parser.add_argument("--max-points", type=int, default=25000)
    parser.add_argument("--seed", type=int, default=20260724)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=256)
    return parser


if __name__ == "__main__":
    print(json.dumps(export(build_parser().parse_args()), indent=2))
