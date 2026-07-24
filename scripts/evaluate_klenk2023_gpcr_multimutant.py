#!/usr/bin/env python3
"""Evaluate fixed additive GPCR multi-mutant predictions against delta-Tm.

The model prediction is a sum of constituent single-substitution predictions.
Experimental delta-Tm and predicted delta-delta-G have different units, so this
script reports favorable-direction agreement and rank correlation, never MAE.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import date
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr


REQUIRED_SCREEN_COLUMNS = {
    "mutation",
    "expected_general_ddg",
    "state_ddg",
    "portable_prior_ddg",
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _finite_correlation(
    target: np.ndarray,
    prediction: np.ndarray,
    *,
    method: str,
) -> dict[str, float | None]:
    if target.size < 3 or np.unique(target).size < 2:
        return {"statistic": None, "pvalue": None}
    if np.unique(prediction).size < 2:
        return {"statistic": None, "pvalue": None}
    result = (
        spearmanr(target, prediction)
        if method == "spearman"
        else pearsonr(target, prediction)
    )
    statistic = float(result.statistic)
    pvalue = float(result.pvalue)
    return {
        "statistic": statistic if math.isfinite(statistic) else None,
        "pvalue": pvalue if math.isfinite(pvalue) else None,
    }


def _metrics(frame: pd.DataFrame, prediction: str) -> dict[str, object]:
    target = frame["experimental_delta_tm_mean_c"].to_numpy(dtype=np.float64)
    score = frame[prediction].to_numpy(dtype=np.float64)
    target_sign = np.sign(target)
    predicted_sign = np.sign(score)
    nonzero = (target_sign != 0) & (predicted_sign != 0)
    return {
        "rows": int(len(frame)),
        "spearman": _finite_correlation(target, score, method="spearman"),
        "pearson": _finite_correlation(target, score, method="pearson"),
        "favorable_sign_accuracy": (
            float(np.mean(target_sign[nonzero] == predicted_sign[nonzero]))
            if nonzero.any()
            else None
        ),
        "sign_evaluable_rows": int(nonzero.sum()),
    }


def _parse_screen_argument(value: str) -> tuple[str, Path]:
    accession, separator, raw_path = value.partition("=")
    if not separator or not accession.strip() or not raw_path.strip():
        raise argparse.ArgumentTypeError(
            "--screen must have the form UNIPROT=/path/to/screen.csv"
        )
    return accession.strip().upper(), Path(raw_path).expanduser().resolve()


def evaluate(
    benchmark_path: Path,
    screen_paths: dict[str, Path],
    output_csv: Path,
    output_json: Path,
    *,
    checkpoint: Path,
) -> dict[str, object]:
    benchmark_path = Path(benchmark_path).resolve()
    benchmark = pd.read_csv(benchmark_path)
    screens: dict[str, pd.DataFrame] = {}
    for accession, path in screen_paths.items():
        frame = pd.read_csv(path)
        missing = sorted(REQUIRED_SCREEN_COLUMNS - set(frame.columns))
        if missing:
            raise ValueError(f"{path} is missing screen columns: {missing}")
        if frame["mutation"].duplicated().any():
            raise ValueError(f"{path} contains duplicate mutation rows")
        screens[accession] = frame.set_index("mutation", drop=False)

    expected_accessions = set(benchmark["uniprot_id"].astype(str))
    if set(screens) != expected_accessions:
        raise ValueError(
            "screen accessions do not match benchmark: "
            f"expected {sorted(expected_accessions)}, got {sorted(screens)}"
        )

    rows: list[dict[str, object]] = []
    components = (
        "expected_general_ddg",
        "state_ddg",
        "portable_prior_ddg",
    )
    for source in benchmark.to_dict(orient="records"):
        accession = str(source["uniprot_id"])
        mutations = str(source["mutation_set"]).split(",")
        screen = screens[accession]
        missing_mutations = sorted(set(mutations) - set(screen.index))
        if missing_mutations:
            raise ValueError(
                f"{accession} screen is missing mutations: {missing_mutations}"
            )
        selected = screen.loc[mutations]
        row = dict(source)
        for component in components:
            additive = float(selected[component].astype(float).sum())
            row[f"additive_{component}"] = additive
            row[f"favorable_{component}"] = -additive
        row["constituent_expected_general_ddg"] = "|".join(
            f"{mutation}:{float(selected.loc[mutation, 'expected_general_ddg']):.9g}"
            for mutation in mutations
        )
        rows.append(row)

    result = pd.DataFrame(rows)
    output_csv = Path(output_csv).resolve()
    output_json = Path(output_json).resolve()
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_csv, index=False)

    by_receptor: dict[str, object] = {}
    for receptor, subset in result.groupby("protein_id", sort=True):
        by_receptor[str(receptor)] = {
            "benchmark_role": str(subset["benchmark_role"].iloc[0]),
            "experimental_delta_tm_range_c": [
                float(subset["experimental_delta_tm_mean_c"].min()),
                float(subset["experimental_delta_tm_mean_c"].max()),
            ],
            "calibrated_expected_ddg": _metrics(
                subset,
                "favorable_expected_general_ddg",
            ),
            "state": _metrics(subset, "favorable_state_ddg"),
            "portable_prior": _metrics(
                subset,
                "favorable_portable_prior_ddg",
            ),
        }

    unique_sites = {
        (
            str(row["uniprot_id"]),
            int(mutation[1:-1]),
        )
        for row in rows
        for mutation in str(row["mutation_set"]).split(",")
    }
    pth1r = by_receptor.get("PTH1R_HUMAN", {})
    ntr1 = by_receptor.get("NTR1_RAT", {})
    pth1r_expected = pth1r.get("calibrated_expected_ddg", {})
    ntr1_expected = ntr1.get("calibrated_expected_ddg", {})
    checkpoint = Path(checkpoint).resolve()
    audit: dict[str, object] = {
        "schema": "protein-stabilizer.klenk2023-gpcr-multimutant-audit.v1",
        "created_date": date.today().isoformat(),
        "objective": (
            "Fixed-model transfer check on quantitative evolved GPCR "
            "multi-mutants without equating degrees Celsius and kcal/mol."
        ),
        "benchmark": {
            "path": str(benchmark_path),
            "sha256": file_sha256(benchmark_path),
            "rows": int(len(result)),
            "receptors": int(result["protein_id"].nunique()),
            "status": (
                "consumed by this fixed-model evaluation; no longer untouched"
            ),
        },
        "prediction": {
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": file_sha256(checkpoint),
            "combination_rule": (
                "sum fixed constituent single-mutant predictions"
            ),
            "sign_mapping": (
                "favorable prediction = negative predicted delta-delta-G; "
                "favorable experiment = positive delta-Tm"
            ),
            "unit_policy": (
                "rank and direction only; no cross-unit MAE or calibration"
            ),
            "gpcr_labels_used_for_fitting": False,
            "epistasis": (
                "not used; the learned epistasis head was trained only on "
                "double mutants, while these variants contain 3-8 mutations"
            ),
        },
        "embedding_cost": {
            "cold_cache_six_b_full_wt_sequences": int(
                result["uniprot_id"].nunique()
            ),
            "cold_cache_six_b_mutant_sequence_passes": 0,
            "cold_cache_six_hundred_m_masked_site_contexts": len(
                unique_sites
            ),
            "substitutions_scored": int(
                sum(len(frame) for frame in screens.values())
            ),
            "repeat_encoder_cost": (
                "zero ESM-C forward passes from persisted receptor caches"
            ),
        },
        "results_by_receptor": by_receptor,
        "rows": [
            {
                "protein_id": str(row["protein_id"]),
                "uniprot_id": str(row["uniprot_id"]),
                "variant": str(row["variant"]),
                "benchmark_role": str(row["benchmark_role"]),
                "mutation_count": int(row["mutation_count"]),
                "mutation_set": str(row["mutation_set"]),
                "experimental_delta_tm_mean_c": float(
                    row["experimental_delta_tm_mean_c"]
                ),
                "experimental_delta_tm_sd_c": float(
                    row["experimental_delta_tm_sd_c"]
                ),
                "additive_expected_general_ddg": float(
                    row["additive_expected_general_ddg"]
                ),
                "favorable_expected_general_ddg": float(
                    row["favorable_expected_general_ddg"]
                ),
                "additive_state_ddg": float(row["additive_state_ddg"]),
                "additive_portable_prior_ddg": float(
                    row["additive_portable_prior_ddg"]
                ),
            }
            for row in rows
        ],
        "decision": {
            "new_receptor_pth1r": (
                "Passed the catastrophic direction gate for all 3 variants "
                f"and ranked delta-Tm with Spearman "
                f"{pth1r_expected.get('spearman', {}).get('statistic')}; "
                "too few rows, only destabilizing variants, and a different "
                "construct context prevent a performance or SOTA claim."
            ),
            "prior_receptor_ntr1": (
                "Failed favorable direction for all 5 variants despite "
                f"Spearman "
                f"{ntr1_expected.get('spearman', {}).get('statistic')}; "
                "the additive single-mutant rule does not recover these "
                "evolved, signaling-disrupted stabilizing combinations."
            ),
            "application_policy": (
                "Retain the promoted model for bounded single-mutant "
                "screening. Do not treat sums over 3-8 mutations as reliable "
                "GPCR stabilization predictions; screen constituents, apply "
                "functional masks, then test combinations experimentally."
            ),
            "promotion": False,
            "limitation": (
                "Every variant within a receptor has the same experimental "
                "direction, so sign accuracy is a coarse catastrophic-failure "
                "gate rather than a balanced classification estimate."
            ),
        },
        "provenance": {
            "screens": {
                accession: {
                    "path": str(path),
                    "sha256": file_sha256(path),
                    "rows": int(len(screens[accession])),
                }
                for accession, path in sorted(screen_paths.items())
            },
            "result_csv": {
                "path": str(output_csv),
                "sha256": file_sha256(output_csv),
            },
        },
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return audit


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument(
        "--screen",
        action="append",
        type=_parse_screen_argument,
        required=True,
        help="repeat UNIPROT=/path/to/screen.csv for every receptor",
    )
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(
            "checkpoints/esmc_6b_accuracy_fp32/promoted_affine/"
            "accuracy_ensemble.pt"
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    screen_paths = dict(args.screen)
    if len(screen_paths) != len(args.screen):
        raise ValueError("each --screen accession must be unique")
    audit = evaluate(
        args.benchmark,
        screen_paths,
        args.output_csv,
        args.output_json,
        checkpoint=args.checkpoint,
    )
    print(json.dumps(audit, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
