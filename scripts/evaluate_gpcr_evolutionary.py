"""Select and confirm a GPCR-family evolutionary screening consensus."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score

from protein_stabilizer.data import Mutation
from protein_stabilizer.embeddings import file_sha256
from protein_stabilizer.gpcr_evolutionary import (
    GPCRDBEvolutionaryCache,
)


ROOT = Path(__file__).resolve().parents[1]
WEIGHT_STEP = 0.05


def _within_group_percentile(
    score: np.ndarray,
    group: np.ndarray,
) -> np.ndarray:
    output = np.empty(len(score), dtype=np.float64)
    for name in sorted(set(group.tolist())):
        selected = group == name
        output[selected] = (
            rankdata(score[selected], method="average") - 0.5
        ) / int(selected.sum())
    return output


def _macro_spearman(
    target: np.ndarray,
    score: np.ndarray,
    group: np.ndarray,
) -> tuple[float, dict[str, object]]:
    per_group: dict[str, object] = {}
    finite: list[float] = []
    for name in sorted(set(group.tolist())):
        selected = group == name
        if selected.sum() < 2:
            continue
        value = (
            math.nan
            if np.ptp(score[selected]) <= 0.0
            else float(
                spearmanr(
                    target[selected], score[selected]
                ).statistic
            )
        )
        per_group[name] = {
            "rows": int(selected.sum()),
            "spearman": value,
        }
        if np.isfinite(value):
            finite.append(value)
    return (
        float(np.mean(finite)) if finite else math.nan,
        per_group,
    )


def _weight_grid() -> list[tuple[float, float, float]]:
    steps = round(1.0 / WEIGHT_STEP)
    return [
        (
            retained / steps,
            generic / steps,
            evolutionary / steps,
        )
        for retained in range(steps + 1)
        for generic in range(steps + 1 - retained)
        for evolutionary in [steps - retained - generic]
    ]


def _select_weights(
    target: np.ndarray,
    group: np.ndarray,
    retained: np.ndarray,
    generic: np.ndarray,
    evolutionary: np.ndarray,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    candidates: list[dict[str, object]] = []
    for retained_weight, generic_weight, evolutionary_weight in (
        _weight_grid()
    ):
        score = (
            retained_weight * retained
            + generic_weight * generic
            + evolutionary_weight * evolutionary
        )
        macro, _ = _macro_spearman(target, score, group)
        pooled = float(spearmanr(target, score).statistic)
        candidates.append(
            {
                "retained_weight": retained_weight,
                "generic_6b_weight": generic_weight,
                "evolutionary_weight": evolutionary_weight,
                "macro_spearman": macro,
                "pooled_spearman": pooled,
            }
        )
    selected = max(
        candidates,
        key=lambda value: (
            float(value["macro_spearman"]),
            float(value["pooled_spearman"]),
            float(value["retained_weight"]),
            -float(value["evolutionary_weight"]),
        ),
    )
    return selected, candidates


def _blend(
    selected: dict[str, object],
    retained: np.ndarray,
    generic: np.ndarray,
    evolutionary: np.ndarray,
) -> np.ndarray:
    return (
        float(selected["retained_weight"]) * retained
        + float(selected["generic_6b_weight"]) * generic
        + float(selected["evolutionary_weight"]) * evolutionary
    )


def _nested_blend(
    target: np.ndarray,
    group: np.ndarray,
    retained: np.ndarray,
    generic: np.ndarray,
    evolutionary: np.ndarray,
) -> tuple[np.ndarray, list[dict[str, object]]]:
    prediction = np.zeros(len(target), dtype=np.float64)
    folds: list[dict[str, object]] = []
    for heldout in sorted(set(group.tolist())):
        train = group != heldout
        test = ~train
        selected, _ = _select_weights(
            target[train],
            group[train],
            retained[train],
            generic[train],
            evolutionary[train],
        )
        prediction[test] = _blend(
            selected,
            retained[test],
            generic[test],
            evolutionary[test],
        )
        folds.append(
            {
                "heldout_receptor": heldout,
                "rows": int(test.sum()),
                "selected": selected,
            }
        )
    return prediction, folds


def _c5ar_metrics(
    target: np.ndarray,
    score: np.ndarray,
) -> dict[str, object]:
    order = np.argsort(-score)
    return {
        "rows": len(target),
        "positives": int(target.sum()),
        "roc_auc": float(roc_auc_score(target, score)),
        "average_precision": float(
            average_precision_score(target, score)
        ),
        "positives_in_top20": int(target[order[:20]].sum()),
        "positives_in_top50": int(target[order[:50]].sum()),
    }


def _paired_stratified_bootstrap(
    target: np.ndarray,
    candidate: np.ndarray,
    baseline: np.ndarray,
    *,
    seed: int,
    samples: int,
) -> dict[str, object]:
    positive = np.flatnonzero(target > 0.5)
    negative = np.flatnonzero(target <= 0.5)
    rng = np.random.default_rng(seed)
    auc_difference: list[float] = []
    ap_difference: list[float] = []
    for _ in range(samples):
        selected = np.concatenate(
            [
                rng.choice(positive, size=len(positive), replace=True),
                rng.choice(negative, size=len(negative), replace=True),
            ]
        )
        selected_target = target[selected]
        auc_difference.append(
            float(
                roc_auc_score(selected_target, candidate[selected])
                - roc_auc_score(selected_target, baseline[selected])
            )
        )
        ap_difference.append(
            float(
                average_precision_score(
                    selected_target, candidate[selected]
                )
                - average_precision_score(
                    selected_target, baseline[selected]
                )
            )
        )

    def summary(values: list[float]) -> dict[str, object]:
        array = np.asarray(values)
        return {
            "ci95": [
                float(value)
                for value in np.quantile(array, [0.025, 0.975])
            ],
            "median": float(np.median(array)),
            "probability_positive": float(np.mean(array > 0.0)),
        }

    return {
        "seed": seed,
        "samples": samples,
        "roc_auc_difference": summary(auc_difference),
        "average_precision_difference": summary(ap_difference),
    }


def _score_gpcr_tm(
    source: pd.DataFrame,
    cache: GPCRDBEvolutionaryCache,
) -> tuple[pd.DataFrame, dict[str, object]]:
    frames: list[pd.DataFrame] = []
    provenance: dict[str, object] = {}
    for protein, frame in source.groupby("protein_id", sort=True):
        sequence_values = frame["wt_sequence"].unique()
        if len(sequence_values) != 1:
            raise RuntimeError(f"{protein} has inconsistent WT sequences")
        result = cache.score_mutations(
            str(protein),
            str(sequence_values[0]),
            [Mutation.parse(value) for value in frame["mutation"]],
            sequence_scope="family",
            pseudocount=0.5,
        )
        frames.append(
            pd.DataFrame(
                {
                    "protein_id": frame["protein_id"].to_numpy(),
                    "mutation": frame["mutation"].to_numpy(),
                    "evolutionary_log_odds": result.log_odds,
                    "evolutionary_observed": result.observed,
                }
            )
        )
        provenance[str(protein)] = result.provenance
    return pd.concat(frames, ignore_index=True), provenance


def evaluate(args: argparse.Namespace) -> dict[str, object]:
    cache = GPCRDBEvolutionaryCache(args.gpcrdb_cache)
    source = pd.read_csv(args.gpcr_tm_source)
    evolutionary_frame, gpcrdb_provenance = _score_gpcr_tm(
        source, cache
    )
    components = pd.read_csv(args.gpcr_tm_components)
    with h5py.File(args.gpcr_tm_representations, "r") as handle:
        generic = pd.DataFrame(
            {
                "protein_id": handle["protein_id"].asstr()[:],
                "mutation": handle["mutation"].asstr()[:],
                "generic_score": -np.asarray(
                    handle["base_ddg"], dtype=np.float32
                ),
            }
        )
        base_checkpoint_sha256 = str(
            handle.attrs["base_checkpoint_sha256"]
        )
    frame = components.merge(
        generic,
        on=["protein_id", "mutation"],
        validate="one_to_one",
    ).merge(
        evolutionary_frame,
        on=["protein_id", "mutation"],
        validate="one_to_one",
    )
    development = frame["domain"].eq("development").to_numpy()
    official = frame["domain"].eq("official").to_numpy()
    if int(development.sum()) != 82 or int(official.sum()) != 12:
        raise RuntimeError("GPCR-tm development/official row contract changed")
    target = frame["target"].to_numpy(dtype=np.float64)
    group = frame["protein_id"].to_numpy(dtype=str)
    retained = frame["rank6b"].to_numpy(dtype=np.float64)
    generic_rank = np.zeros(len(frame), dtype=np.float64)
    evolutionary_rank = np.zeros(len(frame), dtype=np.float64)
    for selected_domain in (development, official):
        generic_rank[selected_domain] = _within_group_percentile(
            frame.loc[selected_domain, "generic_score"].to_numpy(),
            group[selected_domain],
        )
        evolutionary_rank[selected_domain] = _within_group_percentile(
            frame.loc[
                selected_domain, "evolutionary_log_odds"
            ].to_numpy(),
            group[selected_domain],
        )
    selected, candidates = _select_weights(
        target[development],
        group[development],
        retained[development],
        generic_rank[development],
        evolutionary_rank[development],
    )
    development_candidate = _blend(
        selected,
        retained[development],
        generic_rank[development],
        evolutionary_rank[development],
    )
    official_candidate = _blend(
        selected,
        retained[official],
        generic_rank[official],
        evolutionary_rank[official],
    )
    nested_prediction, nested_folds = _nested_blend(
        target[development],
        group[development],
        retained[development],
        generic_rank[development],
        evolutionary_rank[development],
    )
    retained_development_macro, retained_development_per = (
        _macro_spearman(
            target[development],
            retained[development],
            group[development],
        )
    )
    generic_development_macro, generic_development_per = (
        _macro_spearman(
            target[development],
            generic_rank[development],
            group[development],
        )
    )
    candidate_development_macro, candidate_development_per = (
        _macro_spearman(
            target[development],
            development_candidate,
            group[development],
        )
    )
    nested_development_macro, nested_development_per = (
        _macro_spearman(
            target[development],
            nested_prediction,
            group[development],
        )
    )
    retained_official_macro, retained_official_per = _macro_spearman(
        target[official], retained[official], group[official]
    )
    candidate_official_macro, candidate_official_per = _macro_spearman(
        target[official], official_candidate, group[official]
    )
    c5ar_scores = pd.read_csv(args.c5ar_scores)
    c5ar_mutations = [
        Mutation.parse(value) for value in c5ar_scores["mutation"]
    ]
    c5ar_result = cache.score_mutations(
        args.c5ar_accession,
        cache.canonical_sequence(args.c5ar_accession),
        c5ar_mutations,
        sequence_scope="family",
        pseudocount=0.5,
    )
    c5ar_group = np.asarray(
        [args.c5ar_accession] * len(c5ar_scores)
    )
    c5ar_evolutionary = _within_group_percentile(
        c5ar_result.log_odds, c5ar_group
    )
    c5ar_generic = _within_group_percentile(
        c5ar_scores["candidate_score"].to_numpy(dtype=np.float64),
        c5ar_group,
    )
    c5ar_retained = c5ar_scores["retained_rank6b"].to_numpy(
        dtype=np.float64
    )
    c5ar_candidate = _blend(
        selected,
        c5ar_retained,
        c5ar_generic,
        c5ar_evolutionary,
    )
    c5ar_target = c5ar_scores["target"].to_numpy(dtype=np.float64)
    c5ar_retained_metrics = _c5ar_metrics(
        c5ar_target, c5ar_retained
    )
    c5ar_candidate_metrics = _c5ar_metrics(
        c5ar_target, c5ar_candidate
    )
    bootstrap = _paired_stratified_bootstrap(
        c5ar_target,
        c5ar_candidate,
        c5ar_retained,
        seed=args.seed,
        samples=args.bootstrap_samples,
    )
    promotion_gates = {
        "development_beats_retained": (
            candidate_development_macro > retained_development_macro
        ),
        "development_beats_generic_6b": (
            candidate_development_macro > generic_development_macro
        ),
        "official_within_0_10_of_retained": (
            candidate_official_macro
            >= retained_official_macro - 0.10 - 1e-12
        ),
        "c5ar_auc_improves": (
            float(c5ar_candidate_metrics["roc_auc"])
            > float(c5ar_retained_metrics["roc_auc"])
        ),
        "c5ar_average_precision_improves": (
            float(c5ar_candidate_metrics["average_precision"])
            > float(c5ar_retained_metrics["average_precision"])
        ),
        "c5ar_top50_improves": (
            int(c5ar_candidate_metrics["positives_in_top50"])
            > int(c5ar_retained_metrics["positives_in_top50"])
        ),
    }
    report = {
        "schema": "protein-stabilizer.gpcr-evolutionary-consensus.v1",
        "decision": {
            "status": (
                "promoted_gpcr_screening_consensus"
                if all(promotion_gates.values())
                else "rejected_for_production"
            ),
            "promotion_gates": promotion_gates,
            "interpretation": (
                "The output is a within-scan GPCR mutation ranking score, "
                "not delta-delta-G, delta-Tm, or percent stability."
            ),
        },
        "development_selection": {
            "rows": int(development.sum()),
            "receptors": len(set(group[development].tolist())),
            "selected": selected,
            "weight_grid_step": WEIGHT_STEP,
            "weight_candidates": len(candidates),
            "retained_macro_spearman": retained_development_macro,
            "generic_6b_macro_spearman": generic_development_macro,
            "candidate_macro_spearman": candidate_development_macro,
            "nested_reselected_macro_spearman": (
                nested_development_macro
            ),
            "per_receptor": {
                "retained": retained_development_per,
                "generic_6b": generic_development_per,
                "candidate": candidate_development_per,
                "nested_reselected": nested_development_per,
            },
            "nested_folds": nested_folds,
        },
        "frozen_confirmation": {
            "official_gpcr_tm": {
                "rows": int(official.sum()),
                "retained_macro_spearman": retained_official_macro,
                "candidate_macro_spearman": candidate_official_macro,
                "retained_per_receptor": retained_official_per,
                "candidate_per_receptor": candidate_official_per,
            },
            "c5ar": {
                "retained": c5ar_retained_metrics,
                "candidate": c5ar_candidate_metrics,
                "paired_stratified_bootstrap": bootstrap,
            },
        },
        "evolutionary_prior": {
            "formula": (
                "log((family mutant count + 0.5) / "
                "(family WT count + 0.5))"
            ),
            "target_sequence_excluded": True,
            "scope": "GPCRdb family alignment",
            "gpcr_tm_observed_rows": int(
                frame["evolutionary_observed"].sum()
            ),
            "gpcr_tm_rows": len(frame),
            "c5ar_observed_rows": int(c5ar_result.observed.sum()),
            "c5ar_rows": len(c5ar_result.observed),
        },
        "provenance": {
            "gpcr_tm_source": {
                "path": str(args.gpcr_tm_source.resolve()),
                "sha256": file_sha256(args.gpcr_tm_source),
            },
            "gpcr_tm_components_sha256": file_sha256(
                args.gpcr_tm_components
            ),
            "gpcr_tm_representations_sha256": file_sha256(
                args.gpcr_tm_representations
            ),
            "base_checkpoint_sha256": base_checkpoint_sha256,
            "gpcrdb": gpcrdb_provenance,
            "c5ar_scores_sha256": file_sha256(args.c5ar_scores),
            "c5ar_gpcrdb": c5ar_result.provenance,
        },
        "limitations": [
            "The 12-row official GPCR-tm split and C5aR scan have been "
            "consulted by prior project experiments; no untouched GPCR "
            "benchmark remains.",
            "C5aR is one binary thermostability scan and is not a calibrated "
            "thermodynamic endpoint.",
            "Evolutionary preference mixes folding, expression, function, "
            "and phylogeny; it is an orthogonal ranking prior, not stability "
            "physics.",
            "Fifty C5aR scan positions lie outside the family alignment and "
            "receive a neutral evolutionary prior.",
        ],
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
        "--gpcrdb-cache",
        type=Path,
        default=ROOT / "data/raw/gpcrdb_evolutionary",
    )
    parser.add_argument(
        "--gpcr-tm-source",
        type=Path,
        default=ROOT / "data/curated/gpcr_tm_dtm.csv",
    )
    parser.add_argument(
        "--gpcr-tm-components",
        type=Path,
        default=Path(
            "/data/fast/tmp/protein-stabilizer/proteinmpnn-gpcr/"
            "esmc6b_proteinmpnn_gpcr_tm_components.csv"
        ),
    )
    parser.add_argument(
        "--gpcr-tm-representations",
        type=Path,
        default=(
            ROOT
            / "artifacts/features_v2_transfer_6b_state_potential_fp32/"
            "gpcr_tm.h5"
        ),
    )
    parser.add_argument(
        "--c5ar-scores",
        type=Path,
        default=Path(
            "/data/fast/tmp/protein-stabilizer/proteinmpnn-gpcr/"
            "esmc6b_state_potential_c5ar_scores.csv"
        ),
    )
    parser.add_argument("--c5ar-accession", default="P21730")
    parser.add_argument("--seed", type=int, default=20260722)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "docs/gpcr_evolutionary_consensus_audit.json",
    )
    return parser


if __name__ == "__main__":
    print(
        json.dumps(
            evaluate(build_parser().parse_args()),
            indent=2,
            sort_keys=True,
        )
    )
