#!/usr/bin/env python3
"""Run a bounded 600M loss ablation without selecting on the frozen test set."""

from __future__ import annotations

import argparse
import gc
import json
import time
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import torch

from protein_stabilizer.embeddings import file_sha256
from protein_stabilizer.v2_training import (
    HierarchicalObjectiveConfig,
    _load_rows,
    _train_candidate,
)


ROOT = Path(__file__).resolve().parents[1]


def _summary(metrics: dict[str, object], split: str) -> dict[str, float]:
    evaluation = metrics[split]
    regression = evaluation["regression"]
    retrieval = evaluation["retrieval_head"]
    summary = {
        "mae": float(regression["mae"]),
        "rmse": float(regression["rmse"]),
        "pearson": float(regression["pearson"]),
        "spearman": float(regression["spearman"]),
        "direction_accuracy": float(regression["direction_accuracy"]),
        "stabilizer_average_precision": float(retrieval["average_precision"]),
    }
    if split == "validation":
        summary["selection_score"] = float(metrics["validation_selection_score"])
    return summary


def _passes_validation_gate(
    candidate: dict[str, float],
    baseline: dict[str, float],
) -> tuple[bool, dict[str, bool]]:
    gates = {
        "composite_improves_by_0.002": (
            candidate["selection_score"] > baseline["selection_score"] + 0.002
        ),
        "spearman_not_materially_worse": (
            candidate["spearman"] >= baseline["spearman"] - 0.002
        ),
        "stabilizer_ap_not_materially_worse": (
            candidate["stabilizer_average_precision"]
            >= baseline["stabilizer_average_precision"] - 0.002
        ),
        "mae_not_materially_worse": (
            candidate["mae"] <= baseline["mae"] + 0.005
        ),
        "rmse_not_materially_worse": (
            candidate["rmse"] <= baseline["rmse"] + 0.005
        ),
    }
    return all(gates.values()), gates


def evaluate(args: argparse.Namespace) -> dict[str, object]:
    feature_dir = args.features.resolve()
    cache_path = args.cache.resolve()
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device(
        args.device
        if args.device.startswith("cuda") and torch.cuda.is_available()
        else "cpu"
    )
    if device.type == "cuda":
        torch.set_float32_matmul_precision("highest")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

    cache_sha256 = file_sha256(cache_path)
    train_data = _load_rows(
        feature_dir / "single_train.h5",
        cache_path,
        expected_cache_sha256=cache_sha256,
    )
    test_data = _load_rows(
        feature_dir / "single_test.h5",
        cache_path,
        expected_cache_sha256=cache_sha256,
    )
    train_proteins = set(
        train_data.protein_id[train_data.split == "train"].tolist()
    )
    validation_proteins = set(
        train_data.protein_id[train_data.split == "val"].tolist()
    )
    test_proteins = set(test_data.protein_id.tolist())
    if (
        train_proteins & validation_proteins
        or train_proteins & test_proteins
        or validation_proteins & test_proteins
    ):
        raise RuntimeError("loss ablation proteins overlap across splits")

    baseline = HierarchicalObjectiveConfig()
    objectives = {
        "baseline": baseline,
        "moderate_tail_balance": replace(baseline, maximum_bin_weight=3.0),
        "wide_huber": replace(baseline, huber_delta=2.0),
        "moderate_tail_balance_wide_huber": replace(
            baseline,
            maximum_bin_weight=3.0,
            huber_delta=2.0,
        ),
        "moderate_tail_balance_wide_huber_retrieval_0.35": replace(
            baseline,
            maximum_bin_weight=3.0,
            huber_delta=2.0,
            retrieval_weight=0.35,
        ),
        "moderate_tail_balance_wide_huber_retrieval_0.50": replace(
            baseline,
            maximum_bin_weight=3.0,
            huber_delta=2.0,
            retrieval_weight=0.50,
        ),
    }
    started = time.monotonic()
    records: dict[str, dict[str, object]] = {}
    hidden_test_summaries: dict[str, dict[str, float]] = {}
    for name, objective in objectives.items():
        metrics, members = _train_candidate(
            f"loss_ablation_{name}",
            train_data,
            test_data,
            use_structure=True,
            seed=args.seed,
            epochs=args.epochs,
            minimum_epochs=args.minimum_epochs,
            patience=args.patience,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            ensemble_size=1,
            device=device,
            objective=objective,
        )
        records[name] = {
            "objective": asdict(objective),
            "elapsed_seconds": float(metrics["elapsed_seconds"]),
            "validation": _summary(metrics, "validation"),
            "best_epoch": int(metrics["members"][0]["best_epoch"]),
            "optimizer_steps": int(metrics["members"][0]["optimizer_steps"]),
            "examples_seen": int(metrics["members"][0]["examples_seen"]),
            "directional_constraints": metrics["directional_constraints"],
        }
        # The training primitive computes test predictions, but they remain hidden
        # until after the validation-only selection has been frozen below.
        hidden_test_summaries[name] = _summary(metrics, "test")
        del members, metrics
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    selected_name = max(
        records,
        key=lambda name: (
            float(records[name]["validation"]["selection_score"]),
            name == "baseline",
        ),
    )
    baseline_validation = records["baseline"]["validation"]
    selected_validation = records[selected_name]["validation"]
    passes, gates = _passes_validation_gate(
        selected_validation,
        baseline_validation,
    )
    promoted = selected_name != "baseline" and passes
    frozen_selection = selected_name if promoted else "baseline"

    report = {
        "schema": "protein-stabilizer.ddg-loss-ablation.v1",
        "decision": {
            "selected_by_validation": selected_name,
            "promoted": promoted,
            "frozen_selection": frozen_selection,
            "validation_gates": gates,
            "policy": (
                "Select on protein-disjoint validation only; require >0.002 "
                "composite improvement and no material regression in Spearman, "
                "stabilizer AP, MAE, or RMSE. Reveal frozen test metrics only "
                "for the final selected objective."
            ),
        },
        "candidates": records,
        "frozen_test": {
            "objective": frozen_selection,
            "metrics": hidden_test_summaries[frozen_selection],
        },
        "training": {
            "seed": args.seed,
            "epochs": args.epochs,
            "minimum_epochs": args.minimum_epochs,
            "patience": args.patience,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "ensemble_size": 1,
            "device": str(device),
            "precision": "strict float32; TF32 disabled",
            "elapsed_seconds": time.monotonic() - started,
        },
        "split_integrity": {
            "train_rows": int(np.count_nonzero(train_data.split == "train")),
            "validation_rows": int(
                np.count_nonzero(train_data.split == "val")
            ),
            "test_rows": int(len(test_data.target)),
            "train_proteins": len(train_proteins),
            "validation_proteins": len(validation_proteins),
            "test_proteins": len(test_proteins),
            "protein_overlap": 0,
        },
        "provenance": {
            "feature_dir": str(feature_dir),
            "single_train_sha256": file_sha256(
                feature_dir / "single_train.h5"
            ),
            "single_test_sha256": file_sha256(feature_dir / "single_test.h5"),
            "cache": str(cache_path),
            "cache_sha256": cache_sha256,
            "embedding": train_data.provenance["hierarchy_provenance"],
        },
    }
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--features",
        type=Path,
        default=ROOT / "artifacts/features_v2_600m_fp32",
    )
    parser.add_argument(
        "--cache",
        type=Path,
        default=ROOT / "embeddings/esmc_600m/hierarchy_cache_fp32.h5",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "docs/ddg_loss_ablation_audit.json",
    )
    parser.add_argument("--seed", type=int, default=20260722)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--minimum-epochs", type=int, default=8)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=7.5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--device", default="cuda")
    return parser


if __name__ == "__main__":
    print(json.dumps(evaluate(build_parser().parse_args()), indent=2))
