#!/usr/bin/env python3
"""Fit and promote the target-independent 6B/600M accuracy calibration."""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.optimize import minimize

from protein_stabilizer.accuracy import (
    AFFINE_DDG_CALIBRATION_SCHEMA,
    MULTISCALE_AFFINE_DDG_CALIBRATION_SCHEMA,
    apply_affine_ddg_calibration,
    apply_multiscale_affine_ddg_calibration,
)
from protein_stabilizer.accuracy_predictor import (
    SECONDARY_ACCURACY_STATE_SCHEMA,
)
from protein_stabilizer.accuracy_training import ACCURACY_ENSEMBLE_SCHEMA
from protein_stabilizer.data import AMINO_ACIDS
from protein_stabilizer.embeddings import file_sha256
from protein_stabilizer.full_structure_training import (
    _evaluate_predictions,
    _model_from_state,
    _predict_single_rows,
    _save_torch,
    load_full_structure_arrays,
)


ROOT = Path(__file__).resolve().parents[1]
KEYS = ["protein_id", "position", "wt", "mutant"]


def _read(path: Path, *, require_fold: bool) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {
        *KEYS,
        "experimental_ddg",
        "state_ddg",
        "portable_prior_ddg",
    }
    if require_fold:
        required.add("fold")
    missing = required - set(frame)
    if missing:
        raise RuntimeError(f"{path} lacks columns {sorted(missing)}")
    numeric = frame[
        ["experimental_ddg", "state_ddg", "portable_prior_ddg"]
    ].to_numpy(dtype=np.float64)
    if not np.isfinite(numeric).all():
        raise RuntimeError(f"{path} contains invalid predictions")
    order = (["fold"] if require_fold else []) + KEYS
    return frame.sort_values(order).reset_index(drop=True)


def _assert_aligned(
    primary: pd.DataFrame,
    secondary: pd.DataFrame,
    *,
    require_fold: bool,
) -> None:
    keys = (["fold"] if require_fold else []) + KEYS
    if len(primary) != len(secondary) or not primary[keys].equals(
        secondary[keys]
    ):
        raise RuntimeError("primary and secondary prediction keys differ")
    maximum_target_error = float(
        np.max(
            np.abs(
                primary["experimental_ddg"].to_numpy(dtype=np.float64)
                - secondary["experimental_ddg"].to_numpy(dtype=np.float64)
            )
        )
    )
    if maximum_target_error > 1.0e-6:
        raise RuntimeError(
            "primary and secondary prediction targets differ by "
            f"{maximum_target_error}"
        )


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
    expected_ddg: np.ndarray,
) -> dict[str, object]:
    target = frame["experimental_ddg"].to_numpy(dtype=np.float32)
    primary = frame["state_ddg"].to_numpy(dtype=np.float32)
    expected = np.asarray(expected_ddg, dtype=np.float32)
    eligible_index = np.flatnonzero((primary < 0.0) & (expected < 0.0))
    order = eligible_index[
        np.argsort(primary[eligible_index], kind="stable")
    ]
    positive = target <= -0.5
    result: dict[str, object] = {
        "eligible_rows": int(len(eligible_index)),
        "ranker": "primary_6b_state_ddg",
        "eligibility": "primary_state_ddg < 0 and expected_ddg < 0",
    }
    for count in (20, 50, 100, 500):
        selected = order[:count]
        hits = int(positive[selected].sum())
        result[f"hits_at_{count}"] = hits
        result[f"precision_at_{count}"] = (
            float(hits / len(selected)) if len(selected) else 0.0
        )
    return result


def _secondary_shadow_predictions(
    shadow: pd.DataFrame,
    checkpoint_path: Path,
    proteinmpnn_repository: Path,
    *,
    device: str,
    protein_batch_size: int,
) -> tuple[np.ndarray, dict[str, object]]:
    payload = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False
    )
    if payload.get("schema") != ACCURACY_ENSEMBLE_SCHEMA:
        raise RuntimeError("secondary shadow checkpoint schema mismatch")
    data = load_full_structure_arrays(Path(payload["data"]["path"]))
    lookup = {
        (
            str(data.protein_id[int(protein)]),
            int(position) + 1,
            AMINO_ACIDS[int(wt)],
            AMINO_ACIDS[int(mutant)],
        ): row
        for row, (protein, position, wt, mutant) in enumerate(
            zip(
                data.singles.protein_index,
                data.singles.position[:, 0],
                data.singles.wt_amino_acid[:, 0],
                data.singles.mutant_amino_acid[:, 0],
                strict=True,
            )
        )
    }
    keys = list(
        zip(
            shadow["protein_id"].astype(str),
            shadow["position"].astype(int),
            shadow["wt"].astype(str),
            shadow["mutant"].astype(str),
            strict=True,
        )
    )
    try:
        rows = np.asarray([lookup[key] for key in keys], dtype=np.int64)
    except KeyError as exc:
        raise RuntimeError(
            f"secondary shadow data lacks mutation {exc.args[0]}"
        ) from exc
    if len(np.unique(rows)) != len(rows):
        raise RuntimeError("secondary shadow mapping contains duplicate rows")
    target_error = float(
        np.max(
            np.abs(
                data.singles.target[rows]
                - shadow["experimental_ddg"].to_numpy(dtype=np.float32)
            )
        )
    )
    if target_error > 1.0e-6:
        raise RuntimeError(
            f"secondary shadow target alignment differs by {target_error}"
        )
    torch_device = torch.device(
        device
        if not device.startswith("cuda") or torch.cuda.is_available()
        else "cpu"
    )
    model = _model_from_state(
        data,
        proteinmpnn_repository,
        use_structure=True,
        state=payload["state_model_state_dict"],
        device=torch_device,
    )
    prediction = _predict_single_rows(
        model,
        data,
        rows,
        protein_batch_size=protein_batch_size,
        device=torch_device,
    )
    return prediction, {
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": file_sha256(checkpoint_path),
        "data": payload["data"],
        "rows": int(len(rows)),
        "proteins": int(
            len(np.unique(data.singles.protein_index[rows]))
        ),
        "device": str(torch_device),
    }


def _copy_prior(
    payload: dict[str, object],
    source_checkpoint: Path,
    output: Path,
) -> None:
    prior_record = payload["portable_prior"]
    recorded_prior = Path(prior_record["model"])
    local_prior = source_checkpoint.parent / recorded_prior.name
    source_prior = local_prior if local_prior.is_file() else recorded_prior
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
    manifest = json.loads(source_manifest.read_text(encoding="utf-8"))
    manifest["model"] = str(target_prior)
    target_manifest.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    payload["portable_prior"] = dict(
        prior_record,
        model=str(target_prior),
    )


def calibrate(args: argparse.Namespace) -> dict[str, object]:
    started = time.monotonic()
    paths = {
        name: Path(value).resolve()
        for name, value in {
            "primary_oof": args.primary_oof,
            "secondary_oof": args.secondary_oof,
            "primary_shadow": args.primary_shadow,
            "primary_checkpoint": args.primary_checkpoint,
            "secondary_shadow_checkpoint": (
                args.secondary_shadow_checkpoint
            ),
            "secondary_final_checkpoint": args.secondary_final_checkpoint,
        }.items()
    }
    output = Path(args.output).resolve()
    report_path = Path(args.report).resolve()
    checkpoint_path = output / "accuracy_ensemble.pt"
    if checkpoint_path.exists():
        raise FileExistsError(
            "multiscale accuracy checkpoint exists; use a new output"
        )
    if report_path.exists():
        raise FileExistsError(
            "multiscale accuracy audit exists; use a new report path"
        )
    output.mkdir(parents=True, exist_ok=True)

    primary_oof = _read(paths["primary_oof"], require_fold=True)
    secondary_oof = _read(paths["secondary_oof"], require_fold=True)
    _assert_aligned(primary_oof, secondary_oof, require_fold=True)
    folds = sorted(primary_oof["fold"].astype(int).unique().tolist())
    if folds != [0, 1, 2, 3, 4]:
        raise RuntimeError(f"unexpected OOF folds {folds}")
    primary_shadow = _read(
        paths["primary_shadow"], require_fold=False
    )
    secondary_shadow, secondary_shadow_provenance = (
        _secondary_shadow_predictions(
            primary_shadow,
            paths["secondary_shadow_checkpoint"],
            Path(args.proteinmpnn_repository).resolve(),
            device=args.device,
            protein_batch_size=args.protein_batch_size,
        )
    )

    primary_payload = torch.load(
        paths["primary_checkpoint"],
        map_location="cpu",
        weights_only=False,
    )
    if primary_payload.get("schema") != ACCURACY_ENSEMBLE_SCHEMA:
        raise RuntimeError("primary checkpoint schema mismatch")
    current_calibration = primary_payload.get("ddg_calibration")
    if (
        not isinstance(current_calibration, dict)
        or current_calibration.get("schema")
        != AFFINE_DDG_CALIBRATION_SCHEMA
    ):
        raise RuntimeError(
            "primary checkpoint lacks the retained affine calibration"
        )

    target = primary_oof["experimental_ddg"].to_numpy(dtype=np.float64)
    design = np.column_stack(
        [
            primary_oof["state_ddg"].to_numpy(dtype=np.float64),
            secondary_oof["state_ddg"].to_numpy(dtype=np.float64),
            primary_oof["portable_prior_ddg"].to_numpy(dtype=np.float64),
        ]
    )
    tuning = primary_oof["fold"].to_numpy(dtype=np.int64) == 0
    confirmation = ~tuning
    optimization = minimize(
        lambda value: float(
            np.mean(
                np.abs(
                    design[tuning] @ value[:3]
                    + value[3]
                    - target[tuning]
                )
            )
        ),
        x0=np.asarray([0.5, 0.1, 0.5, 0.0], dtype=np.float64),
        method="Powell",
        bounds=[
            (0.0, 2.0),
            (0.0, 2.0),
            (0.0, 2.0),
            (-1.0, 1.0),
        ],
        options={"maxiter": 1000, "xtol": 1.0e-9, "ftol": 1.0e-10},
    )
    if not optimization.success:
        raise RuntimeError(
            f"multiscale calibration failed: {optimization.message}"
        )
    primary_scale, secondary_scale, prior_scale, intercept = (
        float(value) for value in optimization.x
    )
    formula = (
        f"{primary_scale:.12f} * primary_state_ddg + "
        f"{secondary_scale:.12f} * secondary_state_ddg + "
        f"{prior_scale:.12f} * portable_prior_ddg "
        f"{intercept:+.12f}"
    )
    calibration = {
        "schema": MULTISCALE_AFFINE_DDG_CALIBRATION_SCHEMA,
        "coefficients": {
            "primary_state": primary_scale,
            "secondary_state": secondary_scale,
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
    current_oof = apply_affine_ddg_calibration(
        primary_oof["state_ddg"].to_numpy(dtype=np.float32),
        primary_oof["portable_prior_ddg"].to_numpy(dtype=np.float32),
        current_calibration,
    )
    candidate_oof = apply_multiscale_affine_ddg_calibration(
        primary_oof["state_ddg"].to_numpy(dtype=np.float32),
        secondary_oof["state_ddg"].to_numpy(dtype=np.float32),
        primary_oof["portable_prior_ddg"].to_numpy(dtype=np.float32),
        calibration,
    )
    current_shadow = apply_affine_ddg_calibration(
        primary_shadow["state_ddg"].to_numpy(dtype=np.float32),
        primary_shadow["portable_prior_ddg"].to_numpy(dtype=np.float32),
        current_calibration,
    )
    candidate_shadow = apply_multiscale_affine_ddg_calibration(
        primary_shadow["state_ddg"].to_numpy(dtype=np.float32),
        secondary_shadow,
        primary_shadow["portable_prior_ddg"].to_numpy(dtype=np.float32),
        calibration,
    )

    confirmation_frame = primary_oof.loc[confirmation].reset_index(drop=True)
    metrics = {
        "tuning_fold_0": {
            "retained": _compact_metrics(
                primary_oof.loc[tuning].reset_index(drop=True),
                current_oof[tuning],
                seed=args.seed,
            ),
            "candidate": _compact_metrics(
                primary_oof.loc[tuning].reset_index(drop=True),
                candidate_oof[tuning],
                seed=args.seed + 1,
            ),
        },
        "confirmation_folds_1_to_4": {
            "retained": _compact_metrics(
                confirmation_frame,
                current_oof[confirmation],
                seed=args.seed + 2,
            ),
            "candidate": _compact_metrics(
                confirmation_frame,
                candidate_oof[confirmation],
                seed=args.seed + 3,
            ),
            "retained_suggestions": _suggestion_metrics(
                confirmation_frame, current_oof[confirmation]
            ),
            "candidate_suggestions": _suggestion_metrics(
                confirmation_frame, candidate_oof[confirmation]
            ),
        },
        "all_development_oof": {
            "retained": _compact_metrics(
                primary_oof, current_oof, seed=args.seed + 4
            ),
            "candidate": _compact_metrics(
                primary_oof, candidate_oof, seed=args.seed + 5
            ),
        },
        "family_shadow": {
            "retained": _compact_metrics(
                primary_shadow, current_shadow, seed=args.seed + 6
            ),
            "candidate": _compact_metrics(
                primary_shadow, candidate_shadow, seed=args.seed + 7
            ),
            "retained_suggestions": _suggestion_metrics(
                primary_shadow, current_shadow
            ),
            "candidate_suggestions": _suggestion_metrics(
                primary_shadow, candidate_shadow
            ),
        },
    }

    confirmation_metrics = metrics["confirmation_folds_1_to_4"]
    shadow_metrics = metrics["family_shadow"]
    confirmation_before = confirmation_metrics["retained"]["regression"]
    confirmation_after = confirmation_metrics["candidate"]["regression"]
    confirmation_retrieval_before = confirmation_metrics["retained"][
        "retrieval"
    ]
    confirmation_retrieval_after = confirmation_metrics["candidate"][
        "retrieval"
    ]
    shadow_before = shadow_metrics["retained"]["regression"]
    shadow_after = shadow_metrics["candidate"]["regression"]
    shadow_retrieval_before = shadow_metrics["retained"]["retrieval"]
    shadow_retrieval_after = shadow_metrics["candidate"]["retrieval"]
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
        "confirmation_average_precision_not_lower": (
            confirmation_retrieval_after["average_precision"]
            >= confirmation_retrieval_before["average_precision"]
        ),
        "confirmation_top50_hits_retained": (
            confirmation_metrics["candidate_suggestions"]["hits_at_50"]
            >= confirmation_metrics["retained_suggestions"]["hits_at_50"]
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
        "shadow_direction_not_lower": (
            shadow_after["direction_accuracy"]
            >= shadow_before["direction_accuracy"]
        ),
        "shadow_average_precision_not_lower": (
            shadow_retrieval_after["average_precision"]
            >= shadow_retrieval_before["average_precision"]
        ),
        "shadow_top50_hits_retained": (
            shadow_metrics["candidate_suggestions"]["hits_at_50"]
            >= shadow_metrics["retained_suggestions"]["hits_at_50"]
        ),
        "outer_excluded": True,
    }
    gates["passed"] = all(gates.values())

    oof_output = primary_oof.copy()
    oof_output["secondary_state_ddg"] = secondary_oof["state_ddg"]
    oof_output["retained_calibrated_ddg"] = current_oof
    oof_output["multiscale_calibrated_ddg"] = candidate_oof
    oof_path = output / "multiscale_oof_predictions.csv"
    oof_output.to_csv(oof_path, index=False)
    shadow_output = primary_shadow.copy()
    shadow_output["secondary_state_ddg"] = secondary_shadow
    shadow_output["retained_calibrated_ddg"] = current_shadow
    shadow_output["multiscale_calibrated_ddg"] = candidate_shadow
    shadow_path = output / "multiscale_shadow_predictions.csv"
    shadow_output.to_csv(shadow_path, index=False)

    checkpoint_sha256 = None
    if gates["passed"]:
        secondary_payload = torch.load(
            paths["secondary_final_checkpoint"],
            map_location="cpu",
            weights_only=False,
        )
        if secondary_payload.get("schema") != ACCURACY_ENSEMBLE_SCHEMA:
            raise RuntimeError("secondary final checkpoint schema mismatch")
        payload = dict(primary_payload)
        _copy_prior(payload, paths["primary_checkpoint"], output)
        payload["ddg_calibration"] = calibration
        payload["secondary_state"] = {
            "schema": SECONDARY_ACCURACY_STATE_SCHEMA,
            "role": "ESM-C 600M full-WT structure-aware state potential",
            "source_checkpoint": str(
                paths["secondary_final_checkpoint"]
            ),
            "source_checkpoint_sha256": file_sha256(
                paths["secondary_final_checkpoint"]
            ),
            "state_config": secondary_payload["state_config"],
            "state_model_state_dict": secondary_payload[
                "state_model_state_dict"
            ],
            "embedding_provenance": secondary_payload["data"][
                "embedding_provenance"
            ],
            "proteinmpnn_provenance": secondary_payload["data"][
                "proteinmpnn_provenance"
            ],
            "data_sha256": secondary_payload["data"]["sha256"],
            "source_sha256": secondary_payload["data"]["source_sha256"],
            "split_policy": secondary_payload["data"]["split_policy"],
        }
        payload["calibration_selection"] = {
            "schema": (
                "protein-stabilizer.multiscale-calibration-selection.v1"
            ),
            "primary_oof_sha256": file_sha256(paths["primary_oof"]),
            "secondary_oof_sha256": file_sha256(
                paths["secondary_oof"]
            ),
            "primary_shadow_sha256": file_sha256(
                paths["primary_shadow"]
            ),
            "secondary_shadow_checkpoint_sha256": file_sha256(
                paths["secondary_shadow_checkpoint"]
            ),
            "gates": gates,
            "selection_uses_outer": False,
        }
        payload["application_routing"] = {
            "expected_ddg_kcal_mol": formula,
            "stabilizer_ranking": "primary_6b_state_ddg",
            "reason": (
                "the multiscale affine estimator improves family-held-out "
                "magnitude, rank correlation, direction, and retrieval; "
                "the exact 6B state remains the primary candidate order"
            ),
        }
        _save_torch(checkpoint_path, payload)
        checkpoint_sha256 = file_sha256(checkpoint_path)

    report = {
        "schema": (
            "protein-stabilizer.multiscale-accuracy-calibration-audit.v1"
        ),
        "objective": (
            "improve expected general-domain ddG with complementary cached "
            "6B and 600M WT state models without additional mutant encodings"
        ),
        "decision": (
            "promoted"
            if gates["passed"]
            else "rejected; retained affine checkpoint remains default"
        ),
        "calibration": calibration,
        "metrics": metrics,
        "gates": gates,
        "protocol": {
            "fit": "tuning fold 0 only",
            "confirmation": "family folds 1-4",
            "shadow": (
                "pre-existing target-free family shadow; consulted only "
                "after coefficients were frozen"
            ),
            "outer_used": False,
            "primary_ranker_changed": False,
            "mutation_sequence_embeddings": 0,
            "secondary_shadow": secondary_shadow_provenance,
        },
        "artifacts": {
            "checkpoint": (
                str(checkpoint_path) if gates["passed"] else None
            ),
            "checkpoint_sha256": checkpoint_sha256,
            "oof_predictions": str(oof_path),
            "oof_predictions_sha256": file_sha256(oof_path),
            "shadow_predictions": str(shadow_path),
            "shadow_predictions_sha256": file_sha256(shadow_path),
            "inputs": {
                name: {
                    "path": str(path),
                    "sha256": file_sha256(path),
                }
                for name, path in paths.items()
            },
        },
        "limitations": [
            "The family shadow was consulted during promotion and is not a "
            "new untouched benchmark.",
            "MegaScale training proteins are 30-72 residues and have no "
            "membrane labels; GPCR kcal/mol accuracy remains unvalidated.",
            "This calibration improves the current model but does not support "
            "a sub-0.30 kcal/mol claim.",
        ],
        "elapsed_seconds": time.monotonic() - started,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--primary-oof",
        type=Path,
        default=(
            ROOT
            / "checkpoints/esmc_6b_accuracy_fp32/cv_hybrid/"
            "accuracy_oof_predictions.csv"
        ),
    )
    parser.add_argument(
        "--secondary-oof",
        type=Path,
        default=(
            ROOT
            / "checkpoints/esmc_600m_accuracy_fp32/cv/"
            "accuracy_oof_predictions.csv"
        ),
    )
    parser.add_argument(
        "--primary-shadow",
        type=Path,
        default=(
            ROOT
            / "checkpoints/esmc_6b_accuracy_fp32/"
            "shadow_hybrid_components/shadow_predictions.csv"
        ),
    )
    parser.add_argument(
        "--primary-checkpoint",
        type=Path,
        default=(
            ROOT
            / "checkpoints/esmc_6b_accuracy_fp32/promoted_affine/"
            "accuracy_ensemble.pt"
        ),
    )
    parser.add_argument(
        "--secondary-shadow-checkpoint",
        type=Path,
        default=(
            ROOT
            / "checkpoints/esmc_600m_accuracy_fp32/shadow/"
            "shadow_accuracy_ensemble.pt"
        ),
    )
    parser.add_argument(
        "--secondary-final-checkpoint",
        type=Path,
        default=(
            ROOT
            / "checkpoints/esmc_600m_accuracy_fp32/"
            "accuracy_ensemble.pt"
        ),
    )
    parser.add_argument(
        "--proteinmpnn-repository",
        type=Path,
        default=Path(
            "/data/fast/tmp/protein-stabilizer/upstreams/ProteinMPNN"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            ROOT
            / "checkpoints/esmc_6b_accuracy_fp32/"
            "promoted_multiscale_affine"
        ),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=ROOT / "docs/esmc6b_multiscale_accuracy_audit.json",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--protein-batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260724)
    return parser


def main() -> None:
    report = calibrate(build_parser().parse_args())
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
