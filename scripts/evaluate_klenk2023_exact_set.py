#!/usr/bin/env python3
"""Run a fixed promoted mutation-set head on the consumed Klenk GPCR panel.

This is an extrapolation diagnostic, not a fitting or promotion dataset: every
variant has 3-8 substitutions while the epistasis head was trained on doubles.
The request planner embeds each WT, unique constituent single, and complete
joint sequence once, then reuses a provenance-locked hierarchy cache.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
from datetime import date
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

from protein_stabilizer.alphafold import fetch_alphafold_structure
from protein_stabilizer.data import (
    EmbeddingRequest,
    Mutation,
    apply_mutations,
    normalize_sequence,
)
from protein_stabilizer.v2_predictor import (
    _embed_requests_cached,
    predict_hierarchical_mutations,
)


ROOT = Path(__file__).resolve().parents[1]
SCALE_CONFIG = {
    "600m": {
        "model": "esmc_600m",
        "checkpoint_dir": ROOT / "checkpoints/esmc_600m_v2",
        "state_checkpoint": None,
        "runtime": ROOT / ".venv/bin/python",
    },
    "6b": {
        "model": "biohub/ESMC-6B",
        "checkpoint_dir": ROOT / "checkpoints/esmc_6b_v2",
        "state_checkpoint": (
            ROOT
            / "checkpoints/esmc_6b_state_potential_fp32"
            / "state_potential_ensemble.pt"
        ),
        "runtime": ROOT / ".venv-esmc6b/bin/python",
    },
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _mutations(value: str) -> tuple[Mutation, ...]:
    return tuple(
        sorted(
            (Mutation.parse(item) for item in str(value).split(",")),
            key=lambda mutation: mutation.position,
        )
    )


def build_request_plan(frame: pd.DataFrame) -> dict[str, list[EmbeddingRequest]]:
    """Return one deduplicated cold-cache request plan per receptor."""

    plans: dict[str, list[EmbeddingRequest]] = {}
    for accession, receptor_rows in frame.groupby("uniprot_id", sort=True):
        sequences = {
            normalize_sequence(value)
            for value in receptor_rows["wt_sequence"].astype(str)
        }
        if len(sequences) != 1:
            raise ValueError(f"{accession} has inconsistent WT sequences")
        sequence = sequences.pop()
        parsed_sets = [
            _mutations(value)
            for value in receptor_rows["mutation_set"].astype(str)
        ]
        unique_mutations = sorted(
            {mutation for values in parsed_sets for mutation in values},
            key=lambda mutation: (mutation.position, mutation.mutant),
        )
        positions = tuple(
            sorted({mutation.position for mutation in unique_mutations})
        )
        requests = [EmbeddingRequest(sequence, positions)]
        requests.extend(
            EmbeddingRequest(
                apply_mutations(sequence, [mutation]),
                (mutation.position,),
            )
            for mutation in unique_mutations
        )
        requests.extend(
            EmbeddingRequest(
                apply_mutations(sequence, values),
                tuple(mutation.position for mutation in values),
            )
            for values in parsed_sets
            if len(values) > 1
        )
        deduplicated: list[EmbeddingRequest] = []
        seen: set[tuple[str, tuple[int, ...]]] = set()
        for request in requests:
            key = (request.sequence_hash, request.positions)
            if key not in seen:
                deduplicated.append(request)
                seen.add(key)
        plans[str(accession)] = deduplicated
    return plans


def _correlation(
    target: np.ndarray,
    prediction: np.ndarray,
    *,
    method: str,
) -> dict[str, float | None]:
    if (
        target.size < 3
        or np.unique(target).size < 2
        or np.unique(prediction).size < 2
    ):
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


def _metrics(
    frame: pd.DataFrame,
    prediction_column: str,
) -> dict[str, object]:
    target = frame["experimental_delta_tm_mean_c"].to_numpy(dtype=np.float64)
    prediction = -frame[prediction_column].to_numpy(dtype=np.float64)
    target_sign = np.sign(target)
    prediction_sign = np.sign(prediction)
    evaluable = (target_sign != 0) & (prediction_sign != 0)
    return {
        "rows": int(len(frame)),
        "spearman": _correlation(target, prediction, method="spearman"),
        "pearson": _correlation(target, prediction, method="pearson"),
        "favorable_sign_accuracy": (
            float(np.mean(target_sign[evaluable] == prediction_sign[evaluable]))
            if evaluable.any()
            else None
        ),
        "sign_evaluable_rows": int(evaluable.sum()),
    }


def summarize_predictions(
    benchmark: pd.DataFrame,
    predictions: Sequence[dict[str, object]],
) -> tuple[pd.DataFrame, dict[str, object]]:
    indexed = {
        (str(value["uniprot_id"]), str(value["variant"])): value
        for value in predictions
    }
    rows: list[dict[str, object]] = []
    for source in benchmark.to_dict(orient="records"):
        key = (str(source["uniprot_id"]), str(source["variant"]))
        if key not in indexed:
            raise ValueError(f"missing exact-set prediction for {key}")
        prediction = indexed[key]
        row = dict(source)
        row.update(
            {
                "exact_set_total_ddg": float(prediction["total_ddg"]),
                "same_model_additive_ddg": float(
                    prediction["additive_ddg"]
                ),
                "learned_epistasis_ddg": float(
                    prediction["epistasis_ddg"]
                ),
            }
        )
        rows.append(row)
    result = pd.DataFrame(rows)
    by_receptor: dict[str, object] = {}
    for receptor, subset in result.groupby("protein_id", sort=True):
        by_receptor[str(receptor)] = {
            "benchmark_role": str(subset["benchmark_role"].iloc[0]),
            "same_model_additive": _metrics(
                subset,
                "same_model_additive_ddg",
            ),
            "exact_set_total": _metrics(
                subset,
                "exact_set_total_ddg",
            ),
            "epistasis_summary": {
                "mean_ddg": float(subset["learned_epistasis_ddg"].mean()),
                "minimum_ddg": float(
                    subset["learned_epistasis_ddg"].min()
                ),
                "maximum_ddg": float(
                    subset["learned_epistasis_ddg"].max()
                ),
            },
        }
    return result, by_receptor


def _embedder(scale: str, model: str, device: str):
    if scale == "600m":
        from protein_stabilizer.embeddings import ESMCEmbedder

        return ESMCEmbedder(
            model_name=model,
            device=device,
            storage_dtype="float32",
        )
    from protein_stabilizer.esmc6b import ESMC6BEmbedder

    return ESMC6BEmbedder(
        model,
        device,
        inference_dtype="float32",
        storage_dtype="float32",
    )


def evaluate(args: argparse.Namespace) -> dict[str, object]:
    config = SCALE_CONFIG[args.scale]
    benchmark_path = Path(args.benchmark).resolve()
    benchmark = pd.read_csv(benchmark_path)
    plans = build_request_plan(benchmark)
    run_root = Path(args.run_root).resolve()
    run_root.mkdir(parents=True, exist_ok=True)
    model = str(config["model"])
    embedder = _embedder(args.scale, model, args.device)

    cache_by_accession: dict[str, Path] = {}
    precompute: dict[str, dict[str, object]] = {}
    for accession, requests in plans.items():
        cache_path = run_root / "caches" / (
            f"{accession}_exact_set_{args.scale}_fp32.h5"
        )
        _, stats = _embed_requests_cached(
            embedder,
            requests,
            window_radius=4,
            max_tokens=args.max_tokens,
            max_batch_size=args.max_batch_size,
            cache_path=cache_path,
        )
        cache_by_accession[accession] = cache_path
        precompute[accession] = {
            **stats,
            "cache": str(cache_path),
            "cache_sha256": file_sha256(cache_path),
        }

    structures: dict[str, object] = {}
    for accession, receptor_rows in benchmark.groupby(
        "uniprot_id", sort=True
    ):
        sequence = normalize_sequence(str(receptor_rows["wt_sequence"].iloc[0]))
        structures[str(accession)] = fetch_alphafold_structure(
            str(accession),
            sequence,
            args.alphafold_cache,
            min_plddt=args.alphafold_min_plddt,
        )

    predictions: list[dict[str, object]] = []
    for source in benchmark.to_dict(orient="records"):
        accession = str(source["uniprot_id"])
        structure = structures[accession]
        prediction = predict_hierarchical_mutations(
            str(source["wt_sequence"]),
            str(source["mutation_set"]).split(","),
            Path(config["checkpoint_dir"]),
            model_name=model,
            device=args.device,
            topology="alpha_helical_gpcr",
            pdb_path=structure.pdb_path,
            proteinmpnn_repository=args.proteinmpnn_repository,
            structure_residue_mask=structure.residue_mask,
            structure_source_provenance=structure.provenance,
            max_tokens=args.max_tokens,
            max_batch_size=args.max_batch_size,
            embedder=embedder,
            state_potential_checkpoint=(
                None
                if config["state_checkpoint"] is None
                else Path(config["state_checkpoint"])
            ),
            embedding_cache=cache_by_accession[accession],
        )
        if prediction["model"]["multi_checkpoint_sha256"] is None:
            raise RuntimeError(
                f"{args.scale} produced no exact-set epistasis checkpoint"
            )
        predictions.append(
            {
                "uniprot_id": accession,
                "protein_id": str(source["protein_id"]),
                "variant": str(source["variant"]),
                "mutation_set": str(source["mutation_set"]),
                **prediction,
            }
        )

    result, by_receptor = summarize_predictions(benchmark, predictions)
    output_csv = Path(args.output_csv).resolve()
    output_json = Path(args.output_json).resolve()
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_csv, index=False)
    first_model = predictions[0]["model"]
    audit = {
        "schema": "protein-stabilizer.klenk2023-exact-set-audit.v1",
        "created_date": date.today().isoformat(),
        "scale": args.scale,
        "objective": (
            "Post-consumption diagnostic of the fixed promoted "
            "permutation-invariant mutation-set model on 3-8-mutation GPCR "
            "variants."
        ),
        "benchmark": {
            "path": str(benchmark_path),
            "sha256": file_sha256(benchmark_path),
            "rows": int(len(benchmark)),
            "status": (
                "already consumed by additive evaluation; no fitting, "
                "selection, or promotion decision may use this diagnostic"
            ),
        },
        "model": {
            **first_model,
            "numeric_policy": "float32 inference and storage",
            "training_domain": (
                "MegaScale double mutants from short soluble proteins"
            ),
            "mutation_count_domain": 2,
            "evaluated_mutation_count_range": [
                int(benchmark["mutation_count"].min()),
                int(benchmark["mutation_count"].max()),
            ],
        },
        "embedding_cost": {
            "request_plan": {
                accession: len(requests)
                for accession, requests in plans.items()
            },
            "precompute": precompute,
            "cold_cache_unique_sequence_requests": int(
                sum(len(requests) for requests in plans.values())
            ),
            "prediction_stage_computed": int(
                sum(
                    int(value["embedding_cost"][
                        "sequence_embeddings_computed"
                    ])
                    for value in predictions
                )
            ),
            "prediction_stage_cache_hits": int(
                sum(
                    int(value["embedding_cost"]["cache_hits"])
                    for value in predictions
                )
            ),
            "mutant_sequence_requests": int(
                sum(len(requests) - 1 for requests in plans.values())
            ),
        },
        "results_by_receptor": by_receptor,
        "rows": [
            {
                "protein_id": str(row["protein_id"]),
                "uniprot_id": str(row["uniprot_id"]),
                "variant": str(row["variant"]),
                "mutation_count": int(row["mutation_count"]),
                "experimental_delta_tm_mean_c": float(
                    row["experimental_delta_tm_mean_c"]
                ),
                "same_model_additive_ddg": float(
                    row["same_model_additive_ddg"]
                ),
                "learned_epistasis_ddg": float(
                    row["learned_epistasis_ddg"]
                ),
                "exact_set_total_ddg": float(
                    row["exact_set_total_ddg"]
                ),
            }
            for row in result.to_dict(orient="records")
        ],
        "decision": {
            "production_promotion": False,
            "reason": (
                "This benchmark was already consumed and all mutation counts "
                "extrapolate beyond double-mutant training."
            ),
            "allowed_use": (
                "diagnose whether exact joint context is a plausible bounded "
                "combination reranker; require a new receptor-held-out "
                "quantitative panel before promotion"
            ),
        },
        "provenance": {
            "output_csv": str(output_csv),
            "output_csv_sha256": file_sha256(output_csv),
            "runtime": str(config["runtime"]),
            "alphafold_cache": str(Path(args.alphafold_cache).resolve()),
        },
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    del embedder
    gc.collect()
    return audit


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scale", choices=sorted(SCALE_CONFIG), required=True)
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument(
        "--run-root",
        type=Path,
        default=Path(
            "/data/fast/tmp/protein-stabilizer/prospective-gpcr/klenk2023"
        ),
    )
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--max-batch-size", type=int, default=2)
    parser.add_argument(
        "--alphafold-cache",
        type=Path,
        default=ROOT / "artifacts/structures/alphafold",
    )
    parser.add_argument("--alphafold-min-plddt", type=float, default=70.0)
    parser.add_argument(
        "--proteinmpnn-repository",
        type=Path,
        default=Path(
            "/data/fast/tmp/protein-stabilizer/upstreams/ProteinMPNN"
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    report = evaluate(args)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
