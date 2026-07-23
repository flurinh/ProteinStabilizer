#!/usr/bin/env python3
"""Ablate direct ddG-aligned stabilizer ranking at ESM-C 600M."""

from __future__ import annotations

import argparse
import gc
import json
import time
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import torch

from evaluate_ddg_loss_ablation import _passes_validation_gate
from protein_stabilizer.embeddings import file_sha256
from protein_stabilizer.v2_training import (
    HierarchicalObjectiveConfig,
    _load_rows,
    _selection_score,
    _train_candidate,
)


ROOT = Path(__file__).resolve().parents[1]


def _summary(metrics: dict[str, object], split: str) -> dict[str, float]:
    evaluation = metrics[split]
    regression = evaluation["regression"]
    from_ddg = evaluation["retrieval_from_ddg"]
    head = evaluation["retrieval_head"]
    return {
        "selection_score": float(
            _selection_score(evaluation, retrieval_key="retrieval_from_ddg")
        ),
        "mae": float(regression["mae"]),
        "rmse": float(regression["rmse"]),
        "pearson": float(regression["pearson"]),
        "spearman": float(regression["spearman"]),
        "direction_accuracy": float(regression["direction_accuracy"]),
        "stabilizer_average_precision": float(from_ddg["average_precision"]),
        "stabilizers_in_top_50": int(from_ddg["hits_at_50"]),
        "auxiliary_head_average_precision": float(head["average_precision"]),
    }


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
        raise RuntimeError("aligned-retrieval proteins overlap across splits")

    baseline_objective = HierarchicalObjectiveConfig()
    objectives = {
        "auxiliary_retrieval_head": baseline_objective,
        "ddg_aligned_retrieval": replace(
            baseline_objective,
            retrieval_source="ddg",
        ),
    }
    started = time.monotonic()
    records: dict[str, dict[str, object]] = {}
    test_records: dict[str, dict[str, float]] = {}
    for name, objective in objectives.items():
        metrics, members = _train_candidate(
            name,
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
            ensemble_size=args.ensemble_size,
            device=device,
            objective=objective,
        )
        records[name] = {
            "objective": asdict(objective),
            "validation": _summary(metrics, "validation"),
            "members": [
                {
                    "seed": int(member["seed"]),
                    "best_epoch": int(member["best_epoch"]),
                    "optimizer_steps": int(member["optimizer_steps"]),
                    "examples_seen": int(member["examples_seen"]),
                }
                for member in metrics["members"]
            ],
            "directional_constraints": metrics["directional_constraints"],
        }
        test_records[name] = _summary(metrics, "test")
        del metrics, members
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    baseline_name = "auxiliary_retrieval_head"
    candidate_name = "ddg_aligned_retrieval"
    validation_passes, validation_gates = _passes_validation_gate(
        records[candidate_name]["validation"],
        records[baseline_name]["validation"],
    )
    promoted = validation_passes
    report = {
        "schema": "protein-stabilizer.ddg-aligned-retrieval-ablation.v1",
        "decision": {
            "validation_selected": validation_passes,
            "promoted_to_full_training": promoted,
            "frozen_selection": candidate_name if promoted else baseline_name,
            "validation_gates": validation_gates,
            "policy": (
                "Select the loss/early-stopping signal on protein-disjoint "
                "validation using the same -ddG ranking consumed by screening; "
                "historical test metrics are reporting-only."
            ),
        },
        "candidates": records,
        "frozen_test": test_records if validation_passes else None,
        "training": {
            "seed": args.seed,
            "epochs": args.epochs,
            "minimum_epochs": args.minimum_epochs,
            "patience": args.patience,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "ensemble_size": args.ensemble_size,
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
        default=ROOT / "docs/ddg_aligned_retrieval_audit.json",
    )
    parser.add_argument("--seed", type=int, default=20260722)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--minimum-epochs", type=int, default=8)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=7.5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--ensemble-size", type=int, default=3)
    parser.add_argument("--device", default="cuda")
    return parser


if __name__ == "__main__":
    print(json.dumps(evaluate(build_parser().parse_args()), indent=2))
