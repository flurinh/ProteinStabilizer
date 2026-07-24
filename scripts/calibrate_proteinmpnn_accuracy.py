#!/usr/bin/env python3
"""Add a frozen ProteinMPNN leave-one-out potential to the ddG estimator."""

from __future__ import annotations

import argparse
import json
import shutil
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.optimize import minimize

from protein_stabilizer.accuracy import (
    MULTISCALE_AFFINE_DDG_CALIBRATION_SCHEMA,
    PROTEINMPNN_MULTISCALE_DDG_CALIBRATION_SCHEMA,
    apply_multiscale_affine_ddg_calibration,
    apply_proteinmpnn_multiscale_ddg_calibration,
)
from protein_stabilizer.accuracy_predictor import (
    SECONDARY_ACCURACY_STATE_SCHEMA,
)
from protein_stabilizer.accuracy_training import ACCURACY_ENSEMBLE_SCHEMA
from protein_stabilizer.data import AMINO_ACIDS
from protein_stabilizer.embeddings import file_sha256
from protein_stabilizer.full_structure import (
    MaskedProteinMPNNEncoder,
    load_trainable_proteinmpnn,
)
from protein_stabilizer.full_structure_training import (
    _evaluate_predictions,
    load_full_structure_arrays,
)


ROOT = Path(__file__).resolve().parents[1]
SHRINKAGE_TO_NEW_CANDIDATE = 0.50


def _strict_fp32(device: torch.device) -> None:
    if device.type == "cuda":
        torch.set_float32_matmul_precision("highest")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False


def _read(path: Path, *, shadow: bool) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {
        "protein_id",
        "position",
        "wt",
        "mutant",
        "experimental_ddg",
        "state_ddg",
        "secondary_state_ddg",
        "portable_prior_ddg",
        "multiscale_calibrated_ddg",
    }
    if not shadow:
        required.add("fold")
    missing = required - set(frame)
    if missing:
        raise RuntimeError(f"{path} lacks columns {sorted(missing)}")
    numeric = frame[
        [
            "experimental_ddg",
            "state_ddg",
            "secondary_state_ddg",
            "portable_prior_ddg",
            "multiscale_calibrated_ddg",
        ]
    ].to_numpy(dtype=np.float64)
    if not np.isfinite(numeric).all():
        raise RuntimeError(f"{path} contains invalid predictions")
    return frame.reset_index(drop=True)


def _proteinmpnn_potentials(
    data_path: Path,
    repository: Path,
    output_path: Path,
    *,
    device: torch.device,
    protein_batch_size: int,
) -> tuple[np.ndarray, dict[str, object]]:
    data = load_full_structure_arrays(data_path)
    proteinmpnn, module, provenance = load_trainable_proteinmpnn(repository)
    encoder = MaskedProteinMPNNEncoder(
        proteinmpnn, module
    ).to(device)
    encoder.eval()
    maximum_length = data.esm_residue.shape[1]
    potential = np.zeros(
        (len(data.protein_id), maximum_length, len(AMINO_ACIDS)),
        dtype=np.float32,
    )
    with torch.inference_mode():
        for start in range(0, len(data.protein_id), protein_batch_size):
            proteins = np.arange(
                start,
                min(start + protein_batch_size, len(data.protein_id)),
                dtype=np.int64,
            )
            coordinate = torch.from_numpy(
                np.ascontiguousarray(
                    data.coordinates[proteins], dtype=np.float32
                )
            ).to(device)
            sequence = torch.from_numpy(
                np.ascontiguousarray(data.sequence_tokens[proteins])
            ).to(device)
            mask = torch.from_numpy(
                np.ascontiguousarray(data.structure_mask[proteins])
            ).to(device)
            representation = encoder(coordinate, sequence, mask)
            log_probability = torch.log_softmax(
                proteinmpnn.W_out(representation[..., :128]),
                dim=-1,
            )[..., : len(AMINO_ACIDS)]
            potential[proteins] = (
                log_probability.float().cpu().numpy()
            )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        schema=np.asarray(
            "protein-stabilizer.proteinmpnn-leave-one-out-potential.v1"
        ),
        protein_id=data.protein_id,
        log_probability=potential,
        proteinmpnn_checkpoint_sha256=np.asarray(
            provenance.checkpoint_sha256
        ),
        proteinmpnn_commit=np.asarray(provenance.commit),
    )
    return potential, {
        "schema": (
            "protein-stabilizer.proteinmpnn-leave-one-out-potential.v1"
        ),
        "policy": (
            "frozen pretrained ProteinMPNN; every residue sees the full WT "
            "sequence except its own identity; all 20 states in one "
            "structure pass"
        ),
        "sign_convention": (
            "ddG prior = log P(WT|context) - log P(mutant|context); "
            "negative means the mutant is more compatible"
        ),
        "proteinmpnn": asdict(provenance),
        "data_path": str(data_path),
        "data_sha256": data.provenance["sha256"],
        "cache_path": str(output_path),
        "cache_sha256": file_sha256(output_path),
    }


def _frame_prior(
    frame: pd.DataFrame,
    potential: np.ndarray,
    protein_id: np.ndarray,
) -> np.ndarray:
    protein = {str(value): index for index, value in enumerate(protein_id)}
    amino_acid = {
        value: index for index, value in enumerate(AMINO_ACIDS)
    }
    result = np.asarray(
        [
            potential[
                protein[str(row.protein_id)],
                int(row.position) - 1,
                amino_acid[str(row.wt)],
            ]
            - potential[
                protein[str(row.protein_id)],
                int(row.position) - 1,
                amino_acid[str(row.mutant)],
            ]
            for row in frame.itertuples()
        ],
        dtype=np.float32,
    )
    if not np.isfinite(result).all():
        raise RuntimeError("ProteinMPNN prior contains invalid values")
    return result


def _metrics(
    frame: pd.DataFrame,
    prediction: np.ndarray,
    mask: np.ndarray,
    *,
    seed: int,
) -> dict[str, object]:
    result = _evaluate_predictions(
        frame["experimental_ddg"].to_numpy(dtype=np.float32)[mask],
        np.asarray(prediction, dtype=np.float32)[mask],
        frame["protein_id"].astype(str).to_numpy()[mask],
        threshold=-0.5,
        bootstrap_seed=seed,
        bootstrap_samples=500,
    )
    result.pop("per_protein", None)
    return result


def _suggestions(
    frame: pd.DataFrame,
    expected_ddg: np.ndarray,
    *,
    rank_by_expected: bool,
) -> dict[str, object]:
    state = frame["state_ddg"].to_numpy(dtype=np.float32)
    target = frame["experimental_ddg"].to_numpy(dtype=np.float32)
    expected = np.asarray(expected_ddg, dtype=np.float32)
    eligible = np.flatnonzero((state < 0.0) & (expected < 0.0))
    rank = expected if rank_by_expected else state
    order = eligible[np.argsort(rank[eligible], kind="stable")]
    result: dict[str, object] = {
        "eligible_rows": int(len(eligible)),
        "ranker": (
            "proteinmpnn_augmented_expected_ddg"
            if rank_by_expected
            else "primary_6b_state_ddg"
        ),
    }
    for count in (20, 50, 100, 500):
        selected = order[:count]
        hits = int(np.sum(target[selected] <= -0.5))
        result[f"hits_at_{count}"] = hits
        result[f"precision_at_{count}"] = (
            float(hits / len(selected)) if len(selected) else 0.0
        )
    return result


def _copy_prior(
    payload: dict[str, object],
    checkpoint_path: Path,
    output: Path,
) -> None:
    record = payload["portable_prior"]
    source = Path(record["model"])
    local = checkpoint_path.parent / source.name
    source = local if local.is_file() else source
    manifest = source.with_suffix(source.suffix + ".json")
    if (
        not source.is_file()
        or not manifest.is_file()
        or file_sha256(source) != record["model_sha256"]
    ):
        raise RuntimeError("retained portable-prior artifact is unavailable")
    destination = output / source.name
    shutil.copy2(source, destination)
    shutil.copy2(
        manifest, destination.with_suffix(destination.suffix + ".json")
    )
    payload["portable_prior"] = {
        **record,
        "model": str(destination),
        "model_sha256": file_sha256(destination),
    }


def _save_checkpoint(path: Path, payload: dict[str, object]) -> None:
    partial = path.with_suffix(path.suffix + ".partial")
    partial.unlink(missing_ok=True)
    torch.save(payload, partial)
    partial.replace(path)


def run(args: argparse.Namespace) -> dict[str, object]:
    started = time.monotonic()
    output = args.output.resolve()
    report_path = args.audit.resolve()
    if output.exists() or report_path.exists():
        raise FileExistsError(
            "ProteinMPNN calibration outputs already exist; use fresh paths"
        )
    output.mkdir(parents=True)
    device = torch.device(
        args.device
        if not args.device.startswith("cuda") or torch.cuda.is_available()
        else "cpu"
    )
    _strict_fp32(device)
    oof = _read(args.oof.resolve(), shadow=False)
    shadow = _read(args.shadow.resolve(), shadow=True)
    checkpoint_path = args.checkpoint.resolve()
    payload = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False
    )
    if payload.get("schema") != ACCURACY_ENSEMBLE_SCHEMA:
        raise RuntimeError("retained accuracy checkpoint schema mismatch")
    retained_calibration = payload.get("ddg_calibration")
    if (
        not isinstance(retained_calibration, dict)
        or retained_calibration.get("schema")
        != MULTISCALE_AFFINE_DDG_CALIBRATION_SCHEMA
    ):
        raise RuntimeError("retained checkpoint is not multiscale calibrated")
    secondary = payload.get("secondary_state")
    if (
        not isinstance(secondary, dict)
        or secondary.get("schema") != SECONDARY_ACCURACY_STATE_SCHEMA
    ):
        raise RuntimeError("retained checkpoint lacks its secondary state")

    potential_path = output / "proteinmpnn_leave_one_out_fp32.npz"
    potential, prior_provenance = _proteinmpnn_potentials(
        args.data.resolve(),
        args.proteinmpnn_repository.resolve(),
        potential_path,
        device=device,
        protein_batch_size=args.protein_batch_size,
    )
    data = load_full_structure_arrays(args.data.resolve())
    proteinmpnn_oof = _frame_prior(oof, potential, data.protein_id)
    proteinmpnn_shadow = _frame_prior(
        shadow, potential, data.protein_id
    )

    retained_oof = apply_multiscale_affine_ddg_calibration(
        oof["state_ddg"].to_numpy(dtype=np.float32),
        oof["secondary_state_ddg"].to_numpy(dtype=np.float32),
        oof["portable_prior_ddg"].to_numpy(dtype=np.float32),
        retained_calibration,
    )
    retained_shadow = apply_multiscale_affine_ddg_calibration(
        shadow["state_ddg"].to_numpy(dtype=np.float32),
        shadow["secondary_state_ddg"].to_numpy(dtype=np.float32),
        shadow["portable_prior_ddg"].to_numpy(dtype=np.float32),
        retained_calibration,
    )
    if (
        np.max(
            np.abs(
                retained_oof
                - oof["multiscale_calibrated_ddg"].to_numpy(
                    dtype=np.float32
                )
            )
        )
        > 1.0e-5
    ):
        raise RuntimeError("retained OOF calibration reconstruction failed")

    design = np.column_stack(
        [
            oof["state_ddg"].to_numpy(dtype=np.float64),
            oof["secondary_state_ddg"].to_numpy(dtype=np.float64),
            oof["portable_prior_ddg"].to_numpy(dtype=np.float64),
            proteinmpnn_oof.astype(np.float64),
        ]
    )
    target = oof["experimental_ddg"].to_numpy(dtype=np.float64)
    tuning = oof["fold"].to_numpy(dtype=np.int64) == 0
    confirmation = ~tuning
    retained_coefficient = retained_calibration["coefficients"]
    retained_vector = np.asarray(
        [
            retained_coefficient["primary_state"],
            retained_coefficient["secondary_state"],
            retained_coefficient["portable_prior"],
            0.0,
            retained_coefficient["intercept"],
        ],
        dtype=np.float64,
    )
    optimization = minimize(
        lambda value: float(
            np.mean(
                np.abs(
                    design[tuning] @ value[:4]
                    + value[4]
                    - target[tuning]
                )
            )
        ),
        x0=np.asarray(
            [
                retained_vector[0],
                retained_vector[1],
                retained_vector[2],
                0.08,
                -0.14,
            ],
            dtype=np.float64,
        ),
        method="Powell",
        bounds=[(0.0, 2.0)] * 4 + [(-1.0, 1.0)],
        options={"maxiter": 2000, "xtol": 1.0e-10, "ftol": 1.0e-11},
    )
    if not optimization.success:
        raise RuntimeError(
            f"ProteinMPNN calibration failed: {optimization.message}"
        )
    raw_vector = np.asarray(optimization.x, dtype=np.float64)
    selected_vector = (
        (1.0 - SHRINKAGE_TO_NEW_CANDIDATE) * retained_vector
        + SHRINKAGE_TO_NEW_CANDIDATE * raw_vector
    )
    names = [
        "primary_state",
        "secondary_state",
        "portable_prior",
        "proteinmpnn_leave_one_out",
        "intercept",
    ]
    coefficient = {
        name: float(value)
        for name, value in zip(names, selected_vector)
    }
    formula = (
        f"{coefficient['primary_state']:.12f} * primary_state_ddg + "
        f"{coefficient['secondary_state']:.12f} * secondary_state_ddg + "
        f"{coefficient['portable_prior']:.12f} * portable_prior_ddg + "
        f"{coefficient['proteinmpnn_leave_one_out']:.12f} * "
        "proteinmpnn_leave_one_out_ddg "
        f"{coefficient['intercept']:+.12f}"
    )
    calibration = {
        "schema": PROTEINMPNN_MULTISCALE_DDG_CALIBRATION_SCHEMA,
        "coefficients": coefficient,
        "formula": formula,
        "selection": {
            "objective": "micro MAE",
            "raw_fit": "tuning family fold 0 only",
            "raw_coefficients": {
                name: float(value)
                for name, value in zip(names, raw_vector)
            },
            "shrinkage_to_raw_candidate": (
                SHRINKAGE_TO_NEW_CANDIDATE
            ),
            "shrinkage_policy": (
                "post-hoc conservative blend with retained estimator; "
                "consumed family shadow was consulted"
            ),
            "confirmation_folds": [1, 2, 3, 4],
            "outer_used": False,
        },
        "self_mutation_policy": "force exactly zero at application time",
    }
    candidate_oof = apply_proteinmpnn_multiscale_ddg_calibration(
        design[:, 0],
        design[:, 1],
        design[:, 2],
        design[:, 3],
        calibration,
    )
    shadow_design = np.column_stack(
        [
            shadow["state_ddg"].to_numpy(dtype=np.float32),
            shadow["secondary_state_ddg"].to_numpy(dtype=np.float32),
            shadow["portable_prior_ddg"].to_numpy(dtype=np.float32),
            proteinmpnn_shadow,
        ]
    )
    candidate_shadow = apply_proteinmpnn_multiscale_ddg_calibration(
        shadow_design[:, 0],
        shadow_design[:, 1],
        shadow_design[:, 2],
        shadow_design[:, 3],
        calibration,
    )
    all_oof = np.ones(len(oof), dtype=bool)
    all_shadow = np.ones(len(shadow), dtype=bool)
    metrics = {
        "tuning_fold_0": {
            "retained": _metrics(
                oof, retained_oof, tuning, seed=args.seed
            ),
            "candidate": _metrics(
                oof, candidate_oof, tuning, seed=args.seed + 1
            ),
        },
        "confirmation_folds_1_to_4": {
            "retained": _metrics(
                oof, retained_oof, confirmation, seed=args.seed + 2
            ),
            "candidate": _metrics(
                oof, candidate_oof, confirmation, seed=args.seed + 3
            ),
            "retained_suggestions": _suggestions(
                oof.loc[confirmation].reset_index(drop=True),
                retained_oof[confirmation],
                rank_by_expected=False,
            ),
            "candidate_suggestions": _suggestions(
                oof.loc[confirmation].reset_index(drop=True),
                candidate_oof[confirmation],
                rank_by_expected=True,
            ),
        },
        "all_development_oof": {
            "retained": _metrics(
                oof, retained_oof, all_oof, seed=args.seed + 4
            ),
            "candidate": _metrics(
                oof, candidate_oof, all_oof, seed=args.seed + 5
            ),
        },
        "family_shadow": {
            "retained": _metrics(
                shadow, retained_shadow, all_shadow, seed=args.seed + 6
            ),
            "candidate": _metrics(
                shadow, candidate_shadow, all_shadow, seed=args.seed + 7
            ),
            "retained_suggestions": _suggestions(
                shadow, retained_shadow, rank_by_expected=False
            ),
            "candidate_suggestions": _suggestions(
                shadow, candidate_shadow, rank_by_expected=True
            ),
        },
    }
    confirmation_before = metrics["confirmation_folds_1_to_4"][
        "retained"
    ]
    confirmation_after = metrics["confirmation_folds_1_to_4"][
        "candidate"
    ]
    shadow_before = metrics["family_shadow"]["retained"]
    shadow_after = metrics["family_shadow"]["candidate"]
    gates = {
        "confirmation_mae_improves_by_0_005": (
            confirmation_before["regression"]["mae"]
            - confirmation_after["regression"]["mae"]
            >= 0.005
        ),
        "confirmation_rmse_improves": (
            confirmation_after["regression"]["rmse"]
            < confirmation_before["regression"]["rmse"]
        ),
        "confirmation_spearman_not_lower": (
            confirmation_after["regression"]["spearman"]
            >= confirmation_before["regression"]["spearman"]
        ),
        "confirmation_direction_not_lower": (
            confirmation_after["regression"]["direction_accuracy"]
            >= confirmation_before["regression"]["direction_accuracy"]
        ),
        "confirmation_average_precision_not_lower": (
            confirmation_after["retrieval"]["average_precision"]
            >= confirmation_before["retrieval"]["average_precision"]
        ),
        "confirmation_top50_hits_not_lower": (
            metrics["confirmation_folds_1_to_4"][
                "candidate_suggestions"
            ]["hits_at_50"]
            >= metrics["confirmation_folds_1_to_4"][
                "retained_suggestions"
            ]["hits_at_50"]
        ),
        "shadow_mae_improves_by_0_001": (
            shadow_before["regression"]["mae"]
            - shadow_after["regression"]["mae"]
            >= 0.001
        ),
        "shadow_rmse_improves": (
            shadow_after["regression"]["rmse"]
            < shadow_before["regression"]["rmse"]
        ),
        "shadow_spearman_not_lower": (
            shadow_after["regression"]["spearman"]
            >= shadow_before["regression"]["spearman"]
        ),
        "shadow_direction_not_lower": (
            shadow_after["regression"]["direction_accuracy"]
            >= shadow_before["regression"]["direction_accuracy"]
        ),
        "shadow_average_precision_not_lower": (
            shadow_after["retrieval"]["average_precision"]
            >= shadow_before["retrieval"]["average_precision"]
        ),
        "shadow_top50_hits_not_lower": (
            metrics["family_shadow"]["candidate_suggestions"][
                "hits_at_50"
            ]
            >= metrics["family_shadow"]["retained_suggestions"][
                "hits_at_50"
            ]
        ),
        "outer_excluded": True,
    }
    gates["passed"] = all(gates.values())

    oof_output = oof.copy()
    oof_output["proteinmpnn_leave_one_out_ddg"] = proteinmpnn_oof
    oof_output["retained_expected_ddg"] = retained_oof
    oof_output["proteinmpnn_expected_ddg"] = candidate_oof
    oof_path = output / "proteinmpnn_oof_predictions.csv"
    oof_output.to_csv(oof_path, index=False)
    shadow_output = shadow.copy()
    shadow_output[
        "proteinmpnn_leave_one_out_ddg"
    ] = proteinmpnn_shadow
    shadow_output["retained_expected_ddg"] = retained_shadow
    shadow_output["proteinmpnn_expected_ddg"] = candidate_shadow
    shadow_path = output / "proteinmpnn_shadow_predictions.csv"
    shadow_output.to_csv(shadow_path, index=False)

    promoted_checkpoint = output / "accuracy_ensemble.pt"
    checkpoint_sha256 = None
    if gates["passed"]:
        promoted = dict(payload)
        _copy_prior(promoted, checkpoint_path, output)
        promoted["ddg_calibration"] = calibration
        promoted["proteinmpnn_leave_one_out_prior"] = prior_provenance
        promoted["calibration_selection"] = {
            "schema": (
                "protein-stabilizer.proteinmpnn-calibration-selection.v1"
            ),
            "retained_checkpoint_sha256": file_sha256(checkpoint_path),
            "oof_input_sha256": file_sha256(args.oof.resolve()),
            "shadow_input_sha256": file_sha256(args.shadow.resolve()),
            "gates": gates,
            "selection_uses_outer": False,
            "selection_uses_consumed_shadow": True,
        }
        promoted["application_routing"] = {
            "expected_ddg_kcal_mol": formula,
            "stabilizer_ranking": "proteinmpnn_augmented_expected_ddg",
            "reason": (
                "the shrunk ProteinMPNN potential improves confirmation "
                "and consumed-shadow magnitude, direction, rank, average "
                "precision, and top-50 retrieval"
            ),
        }
        _save_checkpoint(promoted_checkpoint, promoted)
        checkpoint_sha256 = file_sha256(promoted_checkpoint)

    report = {
        "schema": (
            "protein-stabilizer.proteinmpnn-accuracy-calibration-audit.v1"
        ),
        "objective": (
            "improve single-mutant ddG and early stabilizer retrieval with "
            "one frozen ProteinMPNN structure pass and no new ESM-C calls"
        ),
        "decision": "promoted" if gates["passed"] else "rejected",
        "calibration": calibration,
        "metrics": metrics,
        "gates": gates,
        "protocol": {
            "raw_fit": "designated tuning family fold 0 only",
            "confirmation": "family folds 1-4",
            "shadow": (
                "previously consumed family shadow; consulted for post-hoc "
                "50% shrinkage and promotion, not an untouched benchmark"
            ),
            "outer_used": False,
            "primary_ranker_changed": True,
            "esm_forward_passes_added": 0,
            "proteinmpnn_structure_passes_added_per_target": 1,
            "mutant_sequence_embeddings": 0,
        },
        "proteinmpnn_prior": prior_provenance,
        "artifacts": {
            "checkpoint": (
                str(promoted_checkpoint) if gates["passed"] else None
            ),
            "checkpoint_sha256": checkpoint_sha256,
            "oof_predictions": str(oof_path),
            "oof_predictions_sha256": file_sha256(oof_path),
            "shadow_predictions": str(shadow_path),
            "shadow_predictions_sha256": file_sha256(shadow_path),
            "inputs": {
                "retained_checkpoint": {
                    "path": str(checkpoint_path),
                    "sha256": file_sha256(checkpoint_path),
                },
                "oof": {
                    "path": str(args.oof.resolve()),
                    "sha256": file_sha256(args.oof.resolve()),
                },
                "shadow": {
                    "path": str(args.shadow.resolve()),
                    "sha256": file_sha256(args.shadow.resolve()),
                },
                "data": {
                    "path": str(args.data.resolve()),
                    "sha256": file_sha256(args.data.resolve()),
                },
            },
        },
        "limitations": [
            "The shadow was consulted and is not an untouched benchmark.",
            "MegaScale proteins are 30-72 residues and lack membrane labels; "
            "GPCR kcal/mol accuracy remains unvalidated.",
            "The result improves the current model but does not support a "
            "sub-0.30 kcal/mol claim.",
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
        "--checkpoint",
        type=Path,
        default=(
            ROOT
            / "checkpoints/esmc_6b_accuracy_fp32/"
            "promoted_multiscale_affine/accuracy_ensemble.pt"
        ),
    )
    parser.add_argument(
        "--oof",
        type=Path,
        default=(
            ROOT
            / "checkpoints/esmc_6b_accuracy_fp32/"
            "promoted_multiscale_affine/multiscale_oof_predictions.csv"
        ),
    )
    parser.add_argument(
        "--shadow",
        type=Path,
        default=(
            ROOT
            / "checkpoints/esmc_6b_accuracy_fp32/"
            "promoted_multiscale_affine/multiscale_shadow_predictions.csv"
        ),
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=ROOT / "artifacts/full_structure/megascale_6b_fp32.h5",
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
            "promoted_proteinmpnn_affine"
        ),
    )
    parser.add_argument(
        "--audit",
        type=Path,
        default=ROOT / "docs/esmc6b_proteinmpnn_accuracy_audit.json",
    )
    parser.add_argument("--protein-batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20260724)
    parser.add_argument("--device", default="cuda")
    return parser


if __name__ == "__main__":
    print(json.dumps(run(build_parser().parse_args()), indent=2))
