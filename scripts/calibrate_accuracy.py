#!/usr/bin/env python3
"""Select and promote a target-independent affine ddG calibration."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.optimize import minimize

from protein_stabilizer.accuracy import (
    AFFINE_DDG_CALIBRATION_SCHEMA,
    apply_affine_ddg_calibration,
)
from protein_stabilizer.embeddings import file_sha256
from protein_stabilizer.full_structure_training import (
    _evaluate_predictions,
    _save_torch,
)


ROOT = Path(__file__).resolve().parents[1]


def _read(path: Path, *, require_fold: bool) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {
        "protein_id",
        "experimental_ddg",
        "state_ddg",
        "portable_prior_ddg",
        "ensemble_ddg",
    }
    if require_fold:
        required.add("fold")
    missing = required - set(frame)
    if missing:
        raise RuntimeError(f"{path} lacks columns {sorted(missing)}")
    numeric = frame[
        [
            "experimental_ddg",
            "state_ddg",
            "portable_prior_ddg",
            "ensemble_ddg",
        ]
    ].to_numpy(dtype=np.float64)
    if not np.isfinite(numeric).all():
        raise RuntimeError(f"{path} contains invalid predictions")
    return frame


def _compact_metrics(
    frame: pd.DataFrame,
    prediction: np.ndarray,
    *,
    seed: int,
) -> dict[str, object]:
    metrics = _evaluate_predictions(
        frame["experimental_ddg"].to_numpy(dtype=np.float32),
        np.asarray(prediction, dtype=np.float32),
        frame["protein_id"].astype(str).to_numpy(),
        threshold=-0.5,
        bootstrap_seed=seed,
        bootstrap_samples=500,
    )
    metrics.pop("per_protein", None)
    return metrics


def _suggestion_metrics(
    frame: pd.DataFrame,
    prediction: np.ndarray,
) -> dict[str, object]:
    target = frame["experimental_ddg"].to_numpy(dtype=np.float32)
    state = frame["state_ddg"].to_numpy(dtype=np.float32)
    expected = np.asarray(prediction, dtype=np.float32)
    eligible = (state < 0.0) & (expected < 0.0)
    eligible_index = np.flatnonzero(eligible)
    order = eligible_index[
        np.argsort(state[eligible_index], kind="stable")
    ]
    positive = target <= -0.5
    result: dict[str, object] = {
        "eligible_rows": int(eligible.sum()),
        "ranker": "state_ddg",
        "eligibility": "state_ddg < 0 and expected_ddg < 0",
    }
    for count in (20, 50, 100, 500):
        selected = order[:count]
        hits = int(positive[selected].sum())
        result[f"hits_at_{count}"] = hits
        result[f"precision_at_{count}"] = (
            float(hits / len(selected)) if len(selected) else 0.0
        )
    return result


def _prediction(
    frame: pd.DataFrame,
    calibration: dict[str, object],
) -> np.ndarray:
    return apply_affine_ddg_calibration(
        frame["state_ddg"].to_numpy(dtype=np.float32),
        frame["portable_prior_ddg"].to_numpy(dtype=np.float32),
        calibration,
    )


def calibrate(args: argparse.Namespace) -> dict[str, object]:
    oof_path = Path(args.oof_predictions).resolve()
    shadow_path = Path(args.shadow_predictions).resolve()
    source_checkpoint = Path(args.source_checkpoint).resolve()
    output = Path(args.output).resolve()
    report_path = Path(args.report).resolve()
    calibrated_checkpoint = output / "accuracy_ensemble.pt"
    if calibrated_checkpoint.exists():
        raise FileExistsError(
            "calibrated accuracy checkpoint exists; use a new output"
        )
    oof = _read(oof_path, require_fold=True)
    if sorted(oof["fold"].astype(int).unique().tolist()) != [0, 1, 2, 3, 4]:
        raise RuntimeError("accuracy OOF table must contain folds 0--4")
    shadow = _read(shadow_path, require_fold=False)
    tuning = oof["fold"].to_numpy(dtype=np.int64) == 0
    confirmation = ~tuning
    state = oof["state_ddg"].to_numpy(dtype=np.float64)
    prior = oof["portable_prior_ddg"].to_numpy(dtype=np.float64)
    target = oof["experimental_ddg"].to_numpy(dtype=np.float64)
    design = np.column_stack([state[tuning], prior[tuning]])
    tuning_target = target[tuning]
    optimization = minimize(
        lambda value: float(
            np.mean(
                np.abs(
                    design @ value[:2]
                    + value[2]
                    - tuning_target
                )
            )
        ),
        x0=np.asarray([0.60, 0.40, 0.0], dtype=np.float64),
        method="Powell",
        bounds=[(0.0, 2.0), (0.0, 2.0), (-1.0, 1.0)],
        options={"maxiter": 1000, "xtol": 1.0e-9, "ftol": 1.0e-10},
    )
    if not optimization.success:
        raise RuntimeError(
            f"affine calibration optimization failed: {optimization.message}"
        )
    state_scale, prior_scale, intercept = (
        float(value) for value in optimization.x
    )
    formula = (
        f"{state_scale:.12f} * state_ddg + "
        f"{prior_scale:.12f} * portable_prior_ddg "
        f"{intercept:+.12f}"
    )
    calibration = {
        "schema": AFFINE_DDG_CALIBRATION_SCHEMA,
        "coefficients": {
            "state": state_scale,
            "portable_prior": prior_scale,
            "intercept": intercept,
        },
        "formula": formula,
        "selection": {
            "objective": "micro MAE",
            "tuning_fold": 0,
            "confirmation_folds": [1, 2, 3, 4],
            "outer_used": False,
            "monotone_nonnegative_component_scales": True,
            "bounds": {
                "component_scales": [0.0, 2.0],
                "intercept_kcal_mol": [-1.0, 1.0],
            },
        },
        "self_mutation_policy": "force exactly zero at application time",
    }

    calibrated_oof = _prediction(oof, calibration)
    calibrated_shadow = _prediction(shadow, calibration)
    baseline_confirmation = oof.loc[confirmation].reset_index(drop=True)
    calibrated_confirmation = calibrated_oof[confirmation]
    baseline_oof_prediction = oof["ensemble_ddg"].to_numpy(
        dtype=np.float32
    )
    baseline_shadow_prediction = shadow["ensemble_ddg"].to_numpy(
        dtype=np.float32
    )
    metrics = {
        "tuning_fold_0": {
            "baseline": _compact_metrics(
                oof.loc[tuning].reset_index(drop=True),
                baseline_oof_prediction[tuning],
                seed=args.seed,
            ),
            "calibrated": _compact_metrics(
                oof.loc[tuning].reset_index(drop=True),
                calibrated_oof[tuning],
                seed=args.seed + 1,
            ),
        },
        "confirmation_folds_1_to_4": {
            "baseline": _compact_metrics(
                baseline_confirmation,
                baseline_oof_prediction[confirmation],
                seed=args.seed + 2,
            ),
            "calibrated": _compact_metrics(
                baseline_confirmation,
                calibrated_confirmation,
                seed=args.seed + 3,
            ),
            "baseline_suggestions": _suggestion_metrics(
                baseline_confirmation,
                baseline_oof_prediction[confirmation],
            ),
            "calibrated_suggestions": _suggestion_metrics(
                baseline_confirmation,
                calibrated_confirmation,
            ),
        },
        "all_development_oof": {
            "baseline": _compact_metrics(
                oof, baseline_oof_prediction, seed=args.seed + 4
            ),
            "calibrated": _compact_metrics(
                oof, calibrated_oof, seed=args.seed + 5
            ),
        },
        "family_shadow": {
            "baseline": _compact_metrics(
                shadow, baseline_shadow_prediction, seed=args.seed + 6
            ),
            "calibrated": _compact_metrics(
                shadow, calibrated_shadow, seed=args.seed + 7
            ),
            "baseline_suggestions": _suggestion_metrics(
                shadow, baseline_shadow_prediction
            ),
            "calibrated_suggestions": _suggestion_metrics(
                shadow, calibrated_shadow
            ),
        },
    }
    confirmation_metrics = metrics["confirmation_folds_1_to_4"]
    shadow_metrics = metrics["family_shadow"]
    confirmation_before = confirmation_metrics["baseline"]["regression"]
    confirmation_after = confirmation_metrics["calibrated"]["regression"]
    shadow_before = shadow_metrics["baseline"]["regression"]
    shadow_after = shadow_metrics["calibrated"]["regression"]
    gates = {
        "confirmation_mae_improves_by_0_001": (
            confirmation_before["mae"] - confirmation_after["mae"]
            >= 0.001
        ),
        "confirmation_rmse_improves": (
            confirmation_after["rmse"] < confirmation_before["rmse"]
        ),
        "confirmation_spearman_not_lower": (
            confirmation_after["spearman"]
            >= confirmation_before["spearman"]
        ),
        "confirmation_direction_not_lower": (
            confirmation_after["direction_accuracy"]
            >= confirmation_before["direction_accuracy"]
        ),
        "confirmation_top50_hits_retained": (
            confirmation_metrics["calibrated_suggestions"]["hits_at_50"]
            >= confirmation_metrics["baseline_suggestions"]["hits_at_50"]
        ),
        "shadow_mae_improves_by_0_001": (
            shadow_before["mae"] - shadow_after["mae"] >= 0.001
        ),
        "shadow_rmse_improves": (
            shadow_after["rmse"] < shadow_before["rmse"]
        ),
        "shadow_spearman_not_lower": (
            shadow_after["spearman"] >= shadow_before["spearman"]
        ),
        "shadow_top50_hits_retained": (
            shadow_metrics["calibrated_suggestions"]["hits_at_50"]
            >= shadow_metrics["baseline_suggestions"]["hits_at_50"]
        ),
        "outer_excluded": True,
    }
    gates["passed"] = all(gates.values())

    output.mkdir(parents=True, exist_ok=True)
    calibrated_predictions = oof.copy()
    calibrated_predictions["calibrated_ddg"] = calibrated_oof
    calibrated_prediction_path = output / "calibrated_oof_predictions.csv"
    calibrated_predictions.to_csv(calibrated_prediction_path, index=False)
    checkpoint_sha256 = None
    if gates["passed"]:
        payload = torch.load(
            source_checkpoint, map_location="cpu", weights_only=False
        )
        prior_record = payload["portable_prior"]
        recorded_prior = Path(prior_record["model"])
        source_prior = (
            source_checkpoint.parent / recorded_prior.name
            if (source_checkpoint.parent / recorded_prior.name).is_file()
            else recorded_prior
        )
        if not source_prior.is_file():
            raise FileNotFoundError(source_prior)
        target_prior = output / source_prior.name
        shutil.copy2(source_prior, target_prior)
        source_manifest = source_prior.with_suffix(
            source_prior.suffix + ".json"
        )
        target_manifest = target_prior.with_suffix(
            target_prior.suffix + ".json"
        )
        manifest = json.loads(
            source_manifest.read_text(encoding="utf-8")
        )
        manifest["model"] = str(target_prior)
        target_manifest.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        payload["portable_prior"] = dict(
            prior_record, model=str(target_prior)
        )
        payload["ddg_calibration"] = calibration
        payload["calibration_selection"] = {
            "oof_predictions_sha256": file_sha256(oof_path),
            "shadow_predictions_sha256": file_sha256(shadow_path),
            "gates": gates,
            "selection_uses_outer": False,
        }
        payload["application_routing"] = {
            "expected_ddg_kcal_mol": formula,
            "stabilizer_ranking": "state_model",
            "reason": (
                "the monotone affine calibration improves confirmation and "
                "family-shadow magnitude metrics while state-ranked top-50 "
                "hits are retained"
            ),
        }
        _save_torch(calibrated_checkpoint, payload)
        checkpoint_sha256 = file_sha256(calibrated_checkpoint)

    report = {
        "schema": "protein-stabilizer.accuracy-calibration-audit.v1",
        "created_date": "2026-07-24",
        "calibration": calibration,
        "metrics": metrics,
        "promotion_gates": gates,
        "decision": {
            "promoted": bool(gates["passed"]),
            "expected_ddg": "calibrated affine state/prior estimate",
            "stabilizer_ranking": "unchanged 6B state score",
            "sub_0_3_supported": False,
            "gpcr_quantitative_accuracy_established": False,
        },
        "artifacts": {
            "oof_predictions": str(oof_path),
            "oof_predictions_sha256": file_sha256(oof_path),
            "shadow_predictions": str(shadow_path),
            "shadow_predictions_sha256": file_sha256(shadow_path),
            "source_checkpoint": str(source_checkpoint),
            "source_checkpoint_sha256": file_sha256(source_checkpoint),
            "calibrated_predictions": str(calibrated_prediction_path),
            "calibrated_predictions_sha256": file_sha256(
                calibrated_prediction_path
            ),
            "calibrated_checkpoint": (
                str(calibrated_checkpoint) if gates["passed"] else None
            ),
            "calibrated_checkpoint_sha256": checkpoint_sha256,
        },
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--oof-predictions",
        type=Path,
        default=(
            ROOT
            / "checkpoints/esmc_6b_accuracy_fp32/cv_hybrid/"
            "accuracy_oof_predictions.csv"
        ),
    )
    parser.add_argument(
        "--shadow-predictions",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--source-checkpoint",
        type=Path,
        default=(
            ROOT
            / "checkpoints/esmc_6b_accuracy_fp32/promoted_hybrid/"
            "accuracy_ensemble.pt"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            ROOT
            / "checkpoints/esmc_6b_accuracy_fp32/promoted_affine"
        ),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=ROOT / "docs/esmc6b_accuracy_calibration_audit.json",
    )
    parser.add_argument("--seed", type=int, default=20260724)
    return parser


if __name__ == "__main__":
    result = calibrate(build_parser().parse_args())
    print(
        json.dumps(
            {
                "schema": result["schema"],
                "calibration": result["calibration"],
                "promotion_gates": result["promotion_gates"],
                "decision": result["decision"],
                "artifacts": result["artifacts"],
            },
            indent=2,
        )
    )
