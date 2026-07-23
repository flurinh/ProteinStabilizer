#!/usr/bin/env python3
"""Build the self-contained model-status dashboard from recorded metrics."""

from __future__ import annotations

import json
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _load(relative: str) -> dict[str, object]:
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


def _load_optional(relative: str) -> dict[str, object] | None:
    path = ROOT / relative
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def _round(value: float, digits: int = 3) -> float:
    return round(float(value), digits)


def _final_losses(payload: dict[str, object]) -> list[float]:
    members = payload["members"]
    return [
        float(member["history"][-1]["train_huber"])
        for member in members
    ]


def dashboard_data() -> dict[str, object]:
    single_600m = _load("checkpoints/esmc_600m/single_metrics.json")
    single_6b = _load("checkpoints/esmc_6b/single_metrics.json")
    multi_600m = _load("checkpoints/esmc_600m/multi_metrics.json")
    multi_6b = _load("checkpoints/esmc_6b/multi_metrics.json")
    transfer_600m = _load("checkpoints/esmc_600m/transfer_metrics.json")
    transfer_6b = _load("checkpoints/esmc_6b/transfer_metrics.json")
    gpcr_dtm = _load("checkpoints/esmc_600m/gpcr_dtm_metrics.json")
    gpcr_full = _load("docs/esmc6b_full_transfer_audit.json")
    ddgemb = _load("docs/esmc6b_ddgemb_transfer_audit.json")
    structure_scale = _load("docs/proteinmpnn_esmc6b_scale_audit.json")
    hierarchy_v2 = _load(
        "checkpoints/esmc_600m_v2/hierarchy_ablation.json"
    )
    hierarchy_transfer = _load(
        "checkpoints/esmc_600m_v2/hierarchy_transfer_metrics.json"
    )
    hierarchy_multi = _load(
        "checkpoints/esmc_600m_v2/hierarchy_multi_metrics.json"
    )
    hierarchy_6b = _load_optional(
        "checkpoints/esmc_6b_v2/hierarchy_scale_report.json"
    )
    hierarchy_6b_multi = _load_optional(
        "checkpoints/esmc_6b_v2/hierarchy_multi_metrics.json"
    )
    hierarchy_6b_transfer = _load_optional(
        "checkpoints/esmc_6b_v2/hierarchy_transfer_metrics.json"
    )
    hierarchy_600m_fp32 = _load_optional(
        "checkpoints/esmc_600m_v2_fp32/hierarchy_scale_report.json"
    )
    state_6b = _load_optional(
        "checkpoints/esmc_6b_state_potential_fp32/state_potential_report.json"
    )
    state_6b_multi = _load_optional(
        "checkpoints/esmc_6b_state_potential_fp32/hierarchy_multi_metrics.json"
    )
    zero_shot_gpcr = _load_optional(
        "checkpoints/esmc_6b_state_potential_fp32/"
        "zero_shot_membrane_rank_report.json"
    )
    c5ar_state = _load_optional(
        "docs/esmc6b_state_potential_c5ar_audit.json"
    )
    evolutionary_gpcr = _load_optional(
        "docs/gpcr_evolutionary_consensus_audit.json"
    )
    ddg_scatter = _load_optional("docs/generic_ddg_scatter.json")
    ddg_calibration = _load_optional("docs/ddg_calibration_audit.json")
    ddg_loss_ablation = _load_optional("docs/ddg_loss_ablation_audit.json")
    ddg_aligned_retrieval = _load_optional(
        "docs/ddg_aligned_retrieval_audit.json"
    )
    validation_audit = _load_optional("docs/model_validation_audit.json")
    stability_optimization = _load_optional(
        "docs/stability_optimization_audit.json"
    )
    full_structure = _load_optional(
        "docs/full_structure_training_audit.json"
    )

    test_600m = single_600m["test"]
    test_6b = single_6b["test"]
    double_600m = multi_600m["test_deployed"]
    double_6b = multi_6b["test_deployed"]
    protherm_6b = transfer_6b["protherm_ddg"]["test"]
    mptherm_600m = transfer_600m["mptherm_dtm"]["test"]
    mptherm_6b = transfer_6b["mptherm_dtm"]["test"]
    gpcr_test = gpcr_dtm["test_mptherm_baseline"]
    gpcr_rank = gpcr_full["gpcr_optional_rerank"]
    c5ar = gpcr_rank["c5ar_independent_scan"]
    structure_selection = structure_scale["development_policy"]["scale_selection"]
    structure_confirmation = structure_scale["frozen_confirmation"]
    small_structure_prior = structure_scale["post_hoc_small_prior_diagnostic"]
    s669 = ddgemb["s2450_transfer"]["S669_results"]
    hierarchy_main = hierarchy_v2["selected_main"]
    hierarchy_test = hierarchy_main["test"]["regression"]
    hierarchy_retrieval = hierarchy_main["test"]["retrieval_from_ddg"]
    hierarchy_losses = [
        float(member["history"][-1]["train_loss"]["total"])
        for member in hierarchy_main["members"]
    ]
    hierarchy_adapters = hierarchy_transfer["adapters"]
    hierarchy_multi_test = hierarchy_multi["evaluation"]["test"]

    loss_600m = _final_losses(single_600m)
    loss_6b = _final_losses(single_6b)

    data = {
        "status": {
            "headline": (
                "600M v2 promoted on the frozen generic test; "
                "GPCR calibration remains diagnostic"
            ),
            "fast_path": "ESM-C 600M v2 hierarchy",
            "best_ddg": "600M v2 ≈ retained 6B baseline",
            "gpcr_path": (
                "Retained GPCR consensus + v2 signed-ddG support"
            ),
            "training": "600M single, transfer, and multi complete; 6B next",
            "structure_audit": (
                "Learned ProteinMPNN fusion passed at 600M; the older "
                "post-hoc 6B likelihood blend remains rejected"
            ),
        },
        "expected_ddg": {
            "generic_mae": _round(hierarchy_test["mae"]),
            "generic_rmse": _round(hierarchy_test["rmse"]),
            "retained_6b_mae": _round(test_6b["mae"]),
            "experimental_mae": _round(protherm_6b["mae"]),
            "experimental_rmse": _round(protherm_6b["rmse"]),
            "s669_mae": _round(s669["existing_protherm_head"]["mae"]),
            "s669_rmse": _round(s669["existing_protherm_head"]["rmse"]),
            "interpretation": (
                "For a new GPCR, treat about ±1 kcal/mol as an operational "
                "error scale, not a calibrated confidence interval. Use the "
                "score primarily to rank candidates."
            ),
        },
        "losses": [
            {
                "model": "600M v2 hierarchy ensemble",
                "objective": "joint Huber + rank + retrieval",
                "members": 5,
                "final_median": _round(
                    statistics.median(hierarchy_losses), 4
                ),
                "final_range": [
                    _round(min(hierarchy_losses), 4),
                    _round(max(hierarchy_losses), 4),
                ],
                "note": (
                    "50-epoch floor; checkpoints selected by protein-held-out "
                    "validation, not final train loss."
                ),
            },
            {
                "model": "600M v2 double epistasis",
                "objective": "total + supervised epistasis Huber",
                "members": 1,
                "final_median": _round(
                    hierarchy_multi["training"]["history"][-1]["train_loss"],
                    4,
                ),
                "final_range": None,
                "note": (
                    "Best validation checkpoint was epoch "
                    f"{hierarchy_multi['training']['best_epoch']}; training "
                    "continued through the 50-epoch floor."
                ),
            },
            {
                "model": "600M single ensemble",
                "objective": "Huber",
                "members": 5,
                "final_median": _round(statistics.median(loss_600m), 4),
                "final_range": [
                    _round(min(loss_600m), 4),
                    _round(max(loss_600m), 4),
                ],
                "note": "Training loss only; early-stopped checkpoints are selected by validation.",
            },
            {
                "model": "6B single ensemble",
                "objective": "Huber",
                "members": 5,
                "final_median": _round(statistics.median(loss_6b), 4),
                "final_range": [
                    _round(min(loss_6b), 4),
                    _round(max(loss_6b), 4),
                ],
                "note": "Lower than 600M, consistent with its better held-out metrics.",
            },
            {
                "model": "600M double epistasis",
                "objective": "Huber",
                "members": 1,
                "final_median": _round(multi_600m["history"][-1]["train_huber"], 4),
                "final_range": None,
                "note": "Improves MAE but loses rank versus the additive baseline.",
            },
            {
                "model": "6B double epistasis",
                "objective": "Huber",
                "members": 1,
                "final_median": _round(multi_6b["history"][-1]["train_huber"], 4),
                "final_range": None,
                "note": "Improves both MAE and rank versus the 6B additive baseline.",
            },
        ],
        "models": [
            {
                "name": "ESM-C 600M v2 hierarchy",
                "role": "Promoted generic screen + stabilizer retrieval",
                "status": "production",
                "target": (
                    "ΔΔG, kcal/mol · AP "
                    f"{_round(hierarchy_retrieval['average_precision'])} · "
                    f"top-50 {int(hierarchy_retrieval['hits_at_50'])}"
                ),
                "test": "Same frozen Megascale protein holdout",
                "rows": int(hierarchy_main["test_rows"]),
                "spearman": _round(hierarchy_test["spearman"]),
                "pearson": _round(hierarchy_test["pearson"]),
                "mae": _round(hierarchy_test["mae"]),
                "rmse": _round(hierarchy_test["rmse"]),
            },
            {
                "name": "ESM-C 600M v2 double",
                "role": "Constituent sum + learned epistasis",
                "status": "production",
                "target": (
                    "ΔΔG, kcal/mol · additive MAE "
                    f"{_round(hierarchy_multi_test['additive']['mae'])}"
                ),
                "test": "Megascale-D protein holdout",
                "rows": int(hierarchy_multi_test["total"]["n"]),
                "spearman": _round(
                    hierarchy_multi_test["total"]["spearman"]
                ),
                "pearson": _round(
                    hierarchy_multi_test["total"]["pearson"]
                ),
                "mae": _round(hierarchy_multi_test["total"]["mae"]),
                "rmse": _round(hierarchy_multi_test["total"]["rmse"]),
            },
            {
                "name": "ESM-C 600M single",
                "role": "Fast production first pass",
                "status": "production",
                "target": "ΔΔG, kcal/mol",
                "test": "Megascale protein holdout",
                "rows": int(single_600m["test_rows"]),
                "spearman": _round(test_600m["spearman"]),
                "pearson": _round(test_600m["pearson"]),
                "mae": _round(test_600m["mae"]),
                "rmse": _round(test_600m["rmse"]),
            },
            {
                "name": "ESM-C 6B single",
                "role": "Best generic ΔΔG estimate",
                "status": "production",
                "target": "ΔΔG, kcal/mol",
                "test": "Megascale protein holdout",
                "rows": int(single_6b["test_rows"]),
                "spearman": _round(test_6b["spearman"]),
                "pearson": _round(test_6b["pearson"]),
                "mae": _round(test_6b["mae"]),
                "rmse": _round(test_6b["rmse"]),
            },
            {
                "name": "ESM-C 6B ProTherm head",
                "role": "Experimental thermodynamic diagnostic",
                "status": "production",
                "target": "ΔΔG, kcal/mol",
                "test": "ProTherm protein holdout",
                "rows": int(transfer_6b["protherm_ddg"]["test_rows"]),
                "spearman": _round(protherm_6b["spearman"]),
                "pearson": _round(protherm_6b["pearson"]),
                "mae": _round(protherm_6b["mae"]),
                "rmse": _round(protherm_6b["rmse"]),
            },
            {
                "name": "ESM-C 6B double",
                "role": "Additive + learned epistasis",
                "status": "production",
                "target": "ΔΔG, kcal/mol",
                "test": "Megascale-D protein holdout",
                "rows": int(double_6b["n"]),
                "spearman": _round(double_6b["spearman"]),
                "pearson": _round(double_6b["pearson"]),
                "mae": _round(double_6b["mae"]),
                "rmse": _round(double_6b["rmse"]),
            },
            {
                "name": "ESM-C 600M double",
                "role": "Fallback combination estimate",
                "status": "limited",
                "target": "ΔΔG, kcal/mol",
                "test": "Megascale-D protein holdout",
                "rows": int(double_600m["n"]),
                "spearman": _round(double_600m["spearman"]),
                "pearson": _round(double_600m["pearson"]),
                "mae": _round(double_600m["mae"]),
                "rmse": _round(double_600m["rmse"]),
            },
        ],
        "transfer": [
            {
                "model": "600M v2 ProTherm diagnostic",
                "endpoint": "ΔΔG, kcal/mol",
                "spearman": _round(
                    hierarchy_adapters["protherm"]["evaluation"]["test"][
                        "spearman"
                    ]
                ),
                "mae": _round(
                    hierarchy_adapters["protherm"]["evaluation"]["test"]["mae"]
                ),
                "rmse": _round(
                    hierarchy_adapters["protherm"]["evaluation"]["test"][
                        "rmse"
                    ]
                ),
                "rows": int(
                    hierarchy_adapters["protherm"]["evaluation"]["test"]["n"]
                ),
            },
            {
                "model": "600M v2 MPTherm diagnostic",
                "endpoint": "ΔTm, °C",
                "spearman": _round(
                    hierarchy_adapters["mptherm"]["evaluation"]["test"][
                        "spearman"
                    ]
                ),
                "mae": _round(
                    hierarchy_adapters["mptherm"]["evaluation"]["test"]["mae"]
                ),
                "rmse": _round(
                    hierarchy_adapters["mptherm"]["evaluation"]["test"]["rmse"]
                ),
                "rows": int(
                    hierarchy_adapters["mptherm"]["evaluation"]["test"]["n"]
                ),
            },
            {
                "model": "600M v2 GPCR ΔTm diagnostic",
                "endpoint": "ΔTm, °C",
                "spearman": _round(
                    hierarchy_adapters["gpcr_tm"]["evaluation"]["test"][
                        "spearman"
                    ]
                ),
                "mae": _round(
                    hierarchy_adapters["gpcr_tm"]["evaluation"]["test"]["mae"]
                ),
                "rmse": _round(
                    hierarchy_adapters["gpcr_tm"]["evaluation"]["test"]["rmse"]
                ),
                "rows": int(
                    hierarchy_adapters["gpcr_tm"]["evaluation"]["test"]["n"]
                ),
            },
            {
                "model": "600M MPTherm",
                "endpoint": "ΔTm, °C",
                "spearman": _round(mptherm_600m["spearman"]),
                "mae": _round(mptherm_600m["mae"]),
                "rmse": _round(mptherm_600m["rmse"]),
                "rows": int(mptherm_600m["n"]),
            },
            {
                "model": "6B MPTherm",
                "endpoint": "ΔTm, °C",
                "spearman": _round(mptherm_6b["spearman"]),
                "mae": _round(mptherm_6b["mae"]),
                "rmse": _round(mptherm_6b["rmse"]),
                "rows": int(mptherm_6b["n"]),
            },
            {
                "model": "600M MPTherm on GPCR-tm",
                "endpoint": "ΔTm, °C",
                "spearman": _round(gpcr_test["overall"]["spearman"]),
                "mae": _round(gpcr_test["overall"]["mae"]),
                "rmse": _round(gpcr_test["overall"]["rmse"]),
                "rows": int(gpcr_test["overall"]["n"]),
            },
        ],
        "gpcr": {
            "development": {
                "rows": int(gpcr_rank["development"]["rows"]),
                "fast_macro_spearman": _round(
                    gpcr_rank["development"][
                        "baseline_macro_within_receptor_spearman"
                    ]
                ),
                "rerank_macro_spearman": _round(
                    gpcr_rank["development"][
                        "candidate_macro_within_receptor_spearman"
                    ]
                ),
            },
            "official_test": {
                "rows": int(gpcr_rank["official_test_confirmation"]["rows"]),
                "fast_macro_spearman": _round(
                    gpcr_rank["official_test_confirmation"][
                        "baseline_macro_within_receptor_spearman"
                    ]
                ),
                "rerank_macro_spearman": _round(
                    gpcr_rank["official_test_confirmation"][
                        "candidate_macro_within_receptor_spearman"
                    ]
                ),
            },
            "c5ar": {
                "rows": int(c5ar["rows"]),
                "positives": int(c5ar["reported_thermostable_substitutions"]),
                "fast_auc": _round(c5ar["baseline"]["roc_auc"]),
                "rerank_auc": _round(
                    c5ar["candidate_end_to_end_runtime_path"]["roc_auc"]
                ),
                "fast_ap": _round(c5ar["baseline"]["average_precision"]),
                "rerank_ap": _round(
                    c5ar["candidate_end_to_end_runtime_path"]["average_precision"]
                ),
                "fast_top50": int(c5ar["baseline"]["positives_in_top50"]),
                "rerank_top50": int(
                    c5ar["candidate_end_to_end_runtime_path"]["positives_in_top50"]
                ),
            },
        },
        "structure_scale": {
            "status": structure_scale["decision"]["status"],
            "selected_weight": float(structure_selection["selected_weight"]),
            "development_baseline": _round(
                structure_selection[
                    "sequence_baseline_macro_spearman_recomputed"
                ]
            ),
            "development_candidate": _round(
                structure_selection["selected_macro_spearman"]
            ),
            "development_nested": _round(
                structure_selection[
                    "nested_leave_one_receptor_out_macro_spearman"
                ]
            ),
            "official_baseline": _round(
                structure_confirmation["official_gpcr_tm"][
                    "sequence_baseline_macro_within_receptor_spearman"
                ]
            ),
            "official_candidate": _round(
                structure_confirmation["official_gpcr_tm"][
                    "candidate_macro_within_receptor_spearman"
                ]
            ),
            "c5ar_baseline_ap": _round(
                structure_confirmation["c5ar"]["sequence_baseline"][
                    "average_precision"
                ]
            ),
            "c5ar_candidate_ap": _round(
                structure_confirmation["c5ar"]["candidate"]["average_precision"]
            ),
            "c5ar_baseline_top50": int(
                structure_confirmation["c5ar"]["sequence_baseline"][
                    "positives_in_top50"
                ]
            ),
            "c5ar_candidate_top50": int(
                structure_confirmation["c5ar"]["candidate"][
                    "positives_in_top50"
                ]
            ),
            "small_prior_weight": float(small_structure_prior["weight"]),
            "small_prior_ap": _round(
                small_structure_prior["c5ar"]["average_precision"]
            ),
            "small_prior_top50": int(
                small_structure_prior["c5ar"]["positives_in_top50"]
            ),
            "small_prior_ap_ci": [
                _round(value)
                for value in small_structure_prior["c5ar"][
                    "paired_stratified_bootstrap_10000"
                ]["average_precision_difference_ci95"]
            ],
        },
        "same_project_comparisons": [
            {
                "model": "Our 600M v2 hierarchy",
                "input": "Sequence + learned structure context",
                "benchmark": "Frozen generic protein holdout",
                "metric": (
                    "ρ 0.818 · MAE 0.513 · stabilizer AP 0.386 · "
                    "top-50 33"
                ),
                "decision": "Promoted; every prespecified gate passed",
                "tone": "good",
            },
            {
                "model": "Our 600M MPTherm head",
                "input": "Sequence",
                "benchmark": "GPCR-tm official test",
                "metric": "Spearman 0.371 · MAE 3.588 °C",
                "decision": "Fast production signal",
                "tone": "good",
            },
            {
                "model": "Official ThermoMPNN",
                "input": "Structure",
                "benchmark": "Same GPCR-tm test",
                "metric": "Spearman 0.126 · MAE 3.805 °C",
                "decision": "Rejected locally",
                "tone": "bad",
            },
            {
                "model": "Official DDGemb",
                "input": "Sequence",
                "benchmark": "Same GPCR-tm test",
                "metric": "Spearman −0.371 · macro ρ −0.600",
                "decision": "Rejected locally",
                "tone": "bad",
            },
            {
                "model": "Our optional 6B rank",
                "input": "Sequence",
                "benchmark": "C5aR independent scan",
                "metric": "AUC 0.696 · AP 0.318 · top-50 15/34",
                "decision": "Optional reranker",
                "tone": "good",
            },
            {
                "model": "6B rank + ProteinMPNN",
                "input": "Sequence + structure",
                "benchmark": "Frozen official / C5aR checks",
                "metric": "macro ρ −0.500 · AP 0.242 · top-50 11/34",
                "decision": "Rejected after 6B scale-up",
                "tone": "bad",
            },
        ],
        "literature_context": [
            {
                "model": "SPURS",
                "input": "Sequence + structure",
                "reported": (
                    "MegaScale median protein ρ 0.83 vs ThermoMPNN 0.77; "
                    ">25% identity filtering"
                ),
                "fit": (
                    "Best architectural lead: ESM/ProteinMPNN cross-attention "
                    "+ 20-state decoder; metric is not our row-micro MAE"
                ),
                "url": "https://www.nature.com/articles/s41467-025-67609-4",
            },
            {
                "model": "JanusDDG",
                "input": "Sequence",
                "reported": (
                    "WT/mutant two-front attention with antisymmetry and "
                    "transitivity constraints"
                ),
                "fit": (
                    "Relevant for reverse and multi-mutant consistency; not "
                    "tested locally on MegaScale or GPCR"
                ),
                "url": "https://www.nature.com/articles/s42003-026-09632-9",
            },
            {
                "model": "Stability Oracle",
                "input": "Structure",
                "reported": "T2837+TP: AUROC 0.83, precision 0.70, recall 0.69",
                "fit": "Strong stabilizer retrieval; not tested locally on GPCR-tm",
                "url": "https://www.nature.com/articles/s41467-024-49780-2",
            },
            {
                "model": "DDGemb",
                "input": "Sequence",
                "reported": "Published S669 leader; our GPCR run reversed on held-out test",
                "fit": "Useful generic comparator, poor local GPCR transfer",
                "url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC11783275/",
            },
            {
                "model": "ThermoMPNN",
                "input": "Structure",
                "reported": "Published transfer model trained on Megascale stability data",
                "fit": "Fast, but weaker than our MPTherm signal on the exact GPCR test",
                "url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC10402116/",
            },
            {
                "model": "IFUM",
                "input": "Sequence + folded/unfolded structure states",
                "reported": (
                    "Absolute ΔG MegaScale-common PCC 0.78, RMSE 1.16 "
                    "kcal/mol"
                ),
                "fit": (
                    "Supports explicit state supervision; absolute ΔG is a "
                    "different endpoint and >200-aa proteins need caution"
                ),
                "url": "https://www.nature.com/articles/s41467-026-68637-4",
            },
            {
                "model": "FoldX",
                "input": "Structure",
                "reported": "Literature error spread ~1.0–1.78 kcal/mol; r ~0.29–0.73",
                "fit": "Valuable orthogonal physics baseline; not run in this project",
                "url": "https://spj.science.org/doi/10.1016/j.csbj.2018.01.002",
            },
            {
                "model": "Rosetta Cartesian ΔΔG",
                "input": "Structure",
                "reported": "Literature performance is dataset/protocol dependent",
                "fit": "Useful when backbone relaxation matters; not run locally",
                "url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC9729920/",
            },
        ],
        "sources": [
            "checkpoints/esmc_600m_v2/hierarchy_ablation.json",
            "checkpoints/esmc_600m_v2/hierarchy_transfer_metrics.json",
            "checkpoints/esmc_600m_v2/hierarchy_multi_metrics.json",
            "checkpoints/esmc_600m/single_metrics.json",
            "checkpoints/esmc_600m/multi_metrics.json",
            "checkpoints/esmc_600m/transfer_metrics.json",
            "checkpoints/esmc_600m/gpcr_dtm_metrics.json",
            "checkpoints/esmc_6b/single_metrics.json",
            "checkpoints/esmc_6b/multi_metrics.json",
            "checkpoints/esmc_6b/transfer_metrics.json",
            "docs/esmc6b_full_transfer_audit.json",
            "docs/esmc6b_ddgemb_transfer_audit.json",
            "docs/proteinmpnn_esmc6b_scale_audit.json",
        ],
    }
    if hierarchy_6b is not None:
        candidate = hierarchy_6b["candidate"]
        regression = candidate["test"]["regression"]
        validation = candidate["validation"]
        retrieval_key = (
            "retrieval_head"
            if float(validation["retrieval_head"]["average_precision"])
            >= float(validation["retrieval_from_ddg"]["average_precision"])
            else "retrieval_from_ddg"
        )
        retrieval = candidate["test"][retrieval_key]
        eligible = bool(hierarchy_6b["production_eligible"])
        losses = [
            float(member["history"][-1]["train_loss"]["total"])
            for member in candidate["members"]
        ]
        data["losses"].insert(
            0,
            {
                "model": "6B v2 hierarchy ensemble · native FP32",
                "objective": "joint Huber + rank + retrieval",
                "members": len(losses),
                "final_median": _round(statistics.median(losses), 4),
                "final_range": [
                    _round(min(losses), 4),
                    _round(max(losses), 4),
                ],
                "note": (
                    "Native FP32 ESM-C/cache/head; TF32 disabled; "
                    "protein-held-out checkpoint selection."
                ),
            },
        )
        data["models"].insert(
            0,
            {
                "name": "ESM-C 6B v2 hierarchy · native FP32",
                "role": (
                    "Highest-accuracy generic screen"
                    if eligible
                    else "Frozen scale diagnostic"
                ),
                "status": "production" if eligible else "limited",
                "target": (
                    f"ΔΔG, kcal/mol · AP "
                    f"{_round(retrieval['average_precision'])} · "
                    f"top-50 {int(retrieval['hits_at_50'])}"
                ),
                "test": "Same frozen Megascale protein holdout",
                "rows": int(candidate["test_rows"]),
                "spearman": _round(regression["spearman"]),
                "pearson": _round(regression["pearson"]),
                "mae": _round(regression["mae"]),
                "rmse": _round(regression["rmse"]),
            },
        )
        data["same_project_comparisons"].insert(
            0,
            {
                "model": "Our 6B v2 hierarchy · native FP32",
                "input": "Sequence + learned structure context",
                "benchmark": "Frozen generic protein holdout",
                "metric": (
                    f"ρ {_round(regression['spearman'])} · "
                    f"MAE {_round(regression['mae'])} · "
                    f"stabilizer AP {_round(retrieval['average_precision'])} · "
                    f"top-50 {int(retrieval['hits_at_50'])}"
                ),
                "decision": (
                    "Highest-accuracy production path; every scale gate passed"
                    if eligible
                    else "Did not pass all scale gates"
                ),
                "tone": "good" if eligible else "bad",
            },
        )
        data["status"]["headline"] = (
            "6B v2 native-FP32 passed every generic promotion gate; "
            "GPCR calibration remains diagnostic"
            if eligible
            else "6B v2 native-FP32 hierarchy did not pass every promotion gate"
        )
        data["status"]["best_ddg"] = (
            "6B v2 native FP32"
            if eligible
            else "600M v2 / retained 6B baseline"
        )
        data["status"]["training"] = (
            "6B v2 native-FP32 single, transfer, and multi complete"
        )
        data["expected_ddg"]["generic_mae"] = _round(regression["mae"])
        data["expected_ddg"]["generic_rmse"] = _round(regression["rmse"])
        data["sources"].append(
            "checkpoints/esmc_6b_v2/hierarchy_scale_report.json"
        )
    if hierarchy_6b_multi is not None:
        multi_test = hierarchy_6b_multi["evaluation"]["test"]
        multi_training = hierarchy_6b_multi["training"]
        data["losses"].insert(
            1,
            {
                "model": "6B v2 double epistasis · native FP32",
                "objective": "total + supervised epistasis Huber",
                "members": 1,
                "final_median": _round(
                    multi_training["history"][-1]["train_loss"], 4
                ),
                "final_range": None,
                "note": (
                    f"Validation selected epoch {multi_training['best_epoch']}; "
                    f"training continued through the "
                    f"{multi_training['minimum_epochs']}-epoch floor."
                ),
            },
        )
        data["models"].insert(
            1,
            {
                "name": "ESM-C 6B v2 double · native FP32",
                "role": "Constituent sum + permutation-invariant epistasis",
                "status": "production",
                "target": (
                    "ΔΔG, kcal/mol · epistasis ρ "
                    f"{_round(multi_test['epistasis']['spearman'])}"
                ),
                "test": "Megascale-D protein holdout",
                "rows": int(multi_test["total"]["n"]),
                "spearman": _round(multi_test["total"]["spearman"]),
                "pearson": _round(multi_test["total"]["pearson"]),
                "mae": _round(multi_test["total"]["mae"]),
                "rmse": _round(multi_test["total"]["rmse"]),
            },
        )
        data["sources"].append(
            "checkpoints/esmc_6b_v2/hierarchy_multi_metrics.json"
        )
    if hierarchy_6b_transfer is not None:
        transfer_rows = []
        for key, label, endpoint in (
            ("protherm", "6B v2 ProTherm diagnostic", "ΔΔG, kcal/mol"),
            ("mptherm", "6B v2 MPTherm diagnostic", "ΔTm, °C"),
            ("gpcr_tm", "6B v2 GPCR ΔTm diagnostic", "ΔTm, °C"),
            (
                "gpcr_rank",
                "6B v2 GPCR crystallization rank diagnostic",
                "macro-assay rank · overall ρ -0.161 · not physical units",
            ),
        ):
            metrics = hierarchy_6b_transfer["adapters"][key]["evaluation"][
                "test"
            ]
            transfer_rows.append(
                {
                    "model": label,
                    "endpoint": endpoint,
                    "spearman": _round(
                        metrics.get("macro_assay_spearman", metrics["spearman"])
                    ),
                    "mae": _round(metrics["mae"]),
                    "rmse": _round(metrics["rmse"]),
                    "rows": int(metrics["n"]),
                }
            )
        data["transfer"] = transfer_rows + data["transfer"]
        data["sources"].append(
            "checkpoints/esmc_6b_v2/hierarchy_transfer_metrics.json"
        )
    if hierarchy_600m_fp32 is not None:
        fast_candidate = hierarchy_600m_fp32["candidate"]
        fast_regression = fast_candidate["test"]["regression"]
        fast_retrieval = fast_candidate["test"]["retrieval_from_ddg"]
        improves_regression = (
            float(fast_regression["spearman"])
            > float(hierarchy_test["spearman"])
            and float(fast_regression["mae"]) < float(hierarchy_test["mae"])
        )
        improves_retrieval = (
            float(fast_retrieval["average_precision"])
            >= float(hierarchy_retrieval["average_precision"])
        )
        data["models"].insert(
            2 if hierarchy_6b_multi is not None else 1,
            {
                "name": "ESM-C 600M v2 retrain · native FP32",
                "role": (
                    "Better fast regression; retained 600M v2 has higher AP"
                    if improves_regression and not improves_retrieval
                    else "Native-FP32 fast-path candidate"
                ),
                "status": (
                    "production"
                    if improves_regression and improves_retrieval
                    else "limited"
                ),
                "target": (
                    f"ΔΔG, kcal/mol · AP "
                    f"{_round(fast_retrieval['average_precision'])} · "
                    f"top-50 {int(fast_retrieval['hits_at_50'])}"
                ),
                "test": "Same frozen Megascale protein holdout",
                "rows": int(fast_candidate["test_rows"]),
                "spearman": _round(fast_regression["spearman"]),
                "pearson": _round(fast_regression["pearson"]),
                "mae": _round(fast_regression["mae"]),
                "rmse": _round(fast_regression["rmse"]),
            },
        )
        data["sources"].append(
            "checkpoints/esmc_600m_v2_fp32/hierarchy_scale_report.json"
        )
    if state_6b is not None:
        blended = state_6b["blended"]["test"]
        regression = blended["regression"]
        retrieval = blended["retrieval_from_ddg"]
        state_weight = state_6b["blend_selection"][
            "state_potential_weight"
        ]
        losses = [
            float(member["history"][-1]["train_loss"]["total"])
            for member in state_6b["members"]
        ]
        data["status"].update(
            {
                "headline": (
                    "Strict-FP32 6B hierarchy/state fusion is the promoted "
                    "generic model; retained GPCR reranker remains best on C5aR"
                ),
                "best_ddg": "6B hierarchy + WT-conditioned state potential",
                "training": (
                    "600M logic validation and strict-FP32 6B single/double "
                    "scale-up complete"
                ),
                "gpcr_path": (
                    "Promoted fused ΔΔG + retained GPCR-specific reranker"
                ),
            }
        )
        data["expected_ddg"]["generic_mae"] = _round(regression["mae"])
        data["expected_ddg"]["generic_rmse"] = _round(regression["rmse"])
        data["losses"].insert(
            0,
            {
                "model": "6B WT-conditioned state-potential ensemble",
                "objective": "Huber + rank + stabilizer retrieval",
                "members": len(losses),
                "final_median": _round(statistics.median(losses), 4),
                "final_range": [
                    _round(min(losses), 4),
                    _round(max(losses), 4),
                ],
                "note": (
                    f"Validation selected state weight {state_weight}; strict "
                    "FP32 and TF32 disabled."
                ),
            },
        )
        data["models"].insert(
            0,
            {
                "name": "ESM-C 6B hierarchy/state fusion · strict FP32",
                "role": "Highest-accuracy generic stabilizing-mutation screen",
                "status": "production",
                "target": (
                    f"ΔΔG, kcal/mol · AP "
                    f"{_round(retrieval['average_precision'])} · "
                    f"top-50 {int(retrieval['hits_at_50'])}"
                ),
                "test": "Frozen Megascale protein holdout",
                "rows": int(regression["n"]),
                "spearman": _round(regression["spearman"]),
                "pearson": _round(regression["pearson"]),
                "mae": _round(regression["mae"]),
                "rmse": _round(regression["rmse"]),
            },
        )
        data["same_project_comparisons"].insert(
            0,
            {
                "model": "Our 6B hierarchy/state fusion · strict FP32",
                "input": "WT/mutant context + WT-conditioned 20-state potential",
                "benchmark": "Frozen generic protein holdout",
                "metric": (
                    f"ρ {_round(regression['spearman'])} · "
                    f"MAE {_round(regression['mae'])} · "
                    f"AP {_round(retrieval['average_precision'])} · "
                    f"top-50 {int(retrieval['hits_at_50'])}"
                ),
                "decision": "Promoted; every frozen single-mutant gate passed",
                "tone": "good",
            },
        )
        data["sources"].append(
            "checkpoints/esmc_6b_state_potential_fp32/"
            "state_potential_report.json"
        )
    if state_6b_multi is not None:
        multi_test = state_6b_multi["evaluation"]["test"]
        eligible = bool(
            state_6b_multi.get("promotion", {}).get(
                "production_eligible", False
            )
        )
        data["models"].insert(
            1,
            {
                "name": "ESM-C 6B fused double-mutant model",
                "role": "Unordered constituent set + learned epistasis",
                "status": "production" if eligible else "limited",
                "target": (
                    "ΔΔG, kcal/mol · epistasis ρ "
                    f"{_round(multi_test['epistasis']['spearman'])}"
                ),
                "test": "Megascale-D protein holdout",
                "rows": int(multi_test["total"]["n"]),
                "spearman": _round(multi_test["total"]["spearman"]),
                "pearson": _round(multi_test["total"]["pearson"]),
                "mae": _round(multi_test["total"]["mae"]),
                "rmse": _round(multi_test["total"]["rmse"]),
            },
        )
        data["sources"].append(
            "checkpoints/esmc_6b_state_potential_fp32/"
            "hierarchy_multi_metrics.json"
        )
    if zero_shot_gpcr is not None:
        selected = zero_shot_gpcr["gpcr_blend_selection"]
        data["gpcr"]["development"]["fused_macro_spearman"] = _round(
            selected["selected_development"]["macro_protein_spearman"]
        )
        data["gpcr"]["official_test"]["fused_macro_spearman"] = _round(
            selected["selected_official"]["macro_protein_spearman"]
        )
        data["status"]["structure_audit"] = (
            "Membrane-only adapter rejected: validation selected weight 0; "
            "generic fusion retained"
        )
        data["sources"].append(
            "checkpoints/esmc_6b_state_potential_fp32/"
            "zero_shot_membrane_rank_report.json"
        )
    if c5ar_state is not None:
        candidate = c5ar_state["evaluation"]["candidate"]
        data["gpcr"]["c5ar"].update(
            {
                "fusion_auc": _round(candidate["roc_auc"]),
                "fusion_ap": _round(candidate["average_precision"]),
                "fusion_top50": int(candidate["positives_in_top50"]),
            }
        )
        data["same_project_comparisons"].insert(
            1,
            {
                "model": "6B generic fusion on C5aR",
                "input": "Sequence + WT-conditioned state potential",
                "benchmark": "Independent C5aR saturation scan",
                "metric": (
                    f"AUC {_round(candidate['roc_auc'])} · "
                    f"AP {_round(candidate['average_precision'])} · "
                    f"top-50 {int(candidate['positives_in_top50'])}/34"
                ),
                "decision": (
                    "Improves fast MPTherm but does not replace retained "
                    "GPCR reranker"
                ),
                "tone": "bad",
            },
        )
        data["sources"].append(
            "docs/esmc6b_state_potential_c5ar_audit.json"
        )
    if evolutionary_gpcr is not None:
        development = evolutionary_gpcr["development_selection"]
        confirmation = evolutionary_gpcr["frozen_confirmation"]
        official = confirmation["official_gpcr_tm"]
        c5_candidate = confirmation["c5ar"]["candidate"]
        bootstrap = confirmation["c5ar"]["paired_stratified_bootstrap"]
        weights = development["selected"]
        data["status"].update(
            {
                "headline": (
                    "Strict-FP32 6B hierarchy/state fusion leads generic "
                    "ΔΔG; GPCRdb evolutionary consensus is the promoted "
                    "GPCR screening rank"
                ),
                "gpcr_path": (
                    "50% retained GPCR rank + 40% strict-FP32 6B generic "
                    "stability + 10% target-excluded GPCR family prior"
                ),
            }
        )
        data["gpcr"]["development"].update(
            {
                "generic_6b_macro_spearman": _round(
                    development["generic_6b_macro_spearman"]
                ),
                "consensus_macro_spearman": _round(
                    development["candidate_macro_spearman"]
                ),
                "nested_consensus_macro_spearman": _round(
                    development["nested_reselected_macro_spearman"]
                ),
            }
        )
        data["gpcr"]["official_test"]["consensus_macro_spearman"] = _round(
            official["candidate_macro_spearman"]
        )
        data["gpcr"]["c5ar"].update(
            {
                "consensus_auc": _round(c5_candidate["roc_auc"]),
                "consensus_ap": _round(
                    c5_candidate["average_precision"]
                ),
                "consensus_top50": int(
                    c5_candidate["positives_in_top50"]
                ),
                "consensus_ap_difference_ci95": [
                    _round(value)
                    for value in bootstrap[
                        "average_precision_difference"
                    ]["ci95"]
                ],
                "consensus_ap_probability_positive": _round(
                    bootstrap["average_precision_difference"][
                        "probability_positive"
                    ],
                    4,
                ),
            }
        )
        data["same_project_comparisons"].insert(
            0,
            {
                "model": "Promoted GPCR evolutionary consensus",
                "input": (
                    "Retained GPCR rank + strict-6B stability + "
                    "target-excluded family alignment"
                ),
                "benchmark": (
                    "GPCR-tm development/confirmation + C5aR scan"
                ),
                "metric": (
                    f"dev macro ρ {_round(development['candidate_macro_spearman'])} · "
                    f"official {_round(official['candidate_macro_spearman'])} · "
                    f"C5 AUC {_round(c5_candidate['roc_auc'])}/"
                    f"AP {_round(c5_candidate['average_precision'])}/"
                    f"top-50 {int(c5_candidate['positives_in_top50'])}/34"
                ),
                "decision": (
                    "Promoted for rank-only GPCR screening; no untouched "
                    "GPCR benchmark remains"
                ),
                "tone": "good",
            },
        )
        data["gpcr"]["consensus_weights"] = {
            "retained": float(weights["retained_weight"]),
            "generic_6b": float(weights["generic_6b_weight"]),
            "evolutionary": float(weights["evolutionary_weight"]),
        }
        data["sources"].append(
            "docs/gpcr_evolutionary_consensus_audit.json"
        )
    if hierarchy_6b is not None and state_6b is not None:
        main_training = hierarchy_6b["candidate"]
        main_members = main_training["members"]
        state_members = state_6b["members"]
        main_steps = sum(
            int(member["optimizer_steps"])
            for member in main_members
        )
        state_steps = sum(
            int(member["optimizer_steps"])
            for member in state_members
        )
        main_seconds = float(main_training["elapsed_seconds"])
        state_seconds = float(state_6b["elapsed_seconds"])
        data["training_scale"] = {
            "dataset_rows": (
                int(main_training["train_rows"])
                + int(main_training["validation_rows"])
                + int(main_training["test_rows"])
            ),
            "train_rows": int(main_training["train_rows"]),
            "validation_rows": int(
                main_training["validation_rows"]
            ),
            "test_rows": int(main_training["test_rows"]),
            "batch_size": int(
                state_6b["training_policy"]["batch_size"]
            ),
            "epochs_per_member": len(
                main_members[0]["history"]
            ),
            "ensemble_members": len(main_members),
            "steps_per_member": int(
                main_members[0]["optimizer_steps"]
            ),
            "examples_per_member": int(
                main_members[0]["examples_seen"]
            ),
            "main_optimizer_steps": main_steps,
            "state_optimizer_steps": state_steps,
            "total_optimizer_steps": main_steps + state_steps,
            "main_minutes": _round(main_seconds / 60.0, 1),
            "state_minutes": _round(state_seconds / 60.0, 1),
            "total_minutes": _round(
                (main_seconds + state_seconds) / 60.0, 1
            ),
            "encoder_policy": (
                "ESM-C embeddings are frozen and cached; these runtimes "
                "train the downstream heads, not the 6B encoder"
            ),
        }
        data["status"]["training"] = (
            f"{data['training_scale']['dataset_rows']:,} MegaScale rows · "
            f"{data['training_scale']['total_optimizer_steps']:,} optimizer "
            f"updates · {data['training_scale']['total_minutes']:.1f} min "
            "for the two five-member 6B stages"
        )
    if ddg_scatter is not None:
        data["ddg_scatter"] = ddg_scatter
        data["sources"].append("docs/generic_ddg_scatter.json")
    optimization_audits: list[dict[str, object]] = []
    if ddg_calibration is not None:
        validation = ddg_calibration["validation"]
        frozen = ddg_calibration["frozen_test"]
        optimization_audits.append(
            {
                "candidate": "Odd monotone kcal/mol calibration",
                "validation": (
                    f"MAE {_round(validation['before']['mae'])}→"
                    f"{_round(validation['after']['mae'])}; RMSE "
                    f"{_round(validation['before']['rmse'])}→"
                    f"{_round(validation['after']['rmse'])}"
                ),
                "frozen_test": (
                    f"MAE {_round(frozen['before']['mae'])}→"
                    f"{_round(frozen['after']['mae'])}; RMSE "
                    f"{_round(frozen['before']['rmse'])}→"
                    f"{_round(frozen['after']['rmse'])}"
                ),
                "decision": "Rejected; test kcal/mol error regressed",
                "tone": "bad",
            }
        )
        data["sources"].append("docs/ddg_calibration_audit.json")
    if ddg_loss_ablation is not None:
        decision = ddg_loss_ablation["decision"]
        selected_name = decision["selected_by_validation"]
        selected = ddg_loss_ablation["candidates"][selected_name]["validation"]
        baseline = ddg_loss_ablation["candidates"]["baseline"]["validation"]
        optimization_audits.append(
            {
                "candidate": "600M loss/tail-balance ablation",
                "validation": (
                    f"best MAE {_round(baseline['mae'])}→"
                    f"{_round(selected['mae'])}; ρ "
                    f"{_round(baseline['spearman'])}→"
                    f"{_round(selected['spearman'])}; AP "
                    f"{_round(baseline['stabilizer_average_precision'])}→"
                    f"{_round(selected['stabilizer_average_precision'])}"
                ),
                "frozen_test": "Baseline retained after validation gate failure",
                "decision": "Rejected; stabilizer AP regressed",
                "tone": "bad",
            }
        )
        data["sources"].append("docs/ddg_loss_ablation_audit.json")
    if ddg_aligned_retrieval is not None:
        candidates = ddg_aligned_retrieval["candidates"]
        baseline = candidates["auxiliary_retrieval_head"]["validation"]
        selected = candidates["ddg_aligned_retrieval"]["validation"]
        frozen = ddg_aligned_retrieval.get("frozen_test")
        if frozen:
            frozen_baseline = frozen["auxiliary_retrieval_head"]
            frozen_selected = frozen["ddg_aligned_retrieval"]
            frozen_text = (
                f"ρ {_round(frozen_baseline['spearman'])}→"
                f"{_round(frozen_selected['spearman'])}; MAE "
                f"{_round(frozen_baseline['mae'])}→"
                f"{_round(frozen_selected['mae'])}; AP "
                f"{_round(frozen_baseline['stabilizer_average_precision'])}→"
                f"{_round(frozen_selected['stabilizer_average_precision'])}"
            )
        else:
            frozen_text = "Not opened; validation gate failed"
        optimization_audits.append(
            {
                "candidate": "ddG-aligned retrieval/ranking",
                "validation": (
                    f"ρ {_round(baseline['spearman'])}→"
                    f"{_round(selected['spearman'])}; MAE "
                    f"{_round(baseline['mae'])}→"
                    f"{_round(selected['mae'])}; AP "
                    f"{_round(baseline['stabilizer_average_precision'])}→"
                    f"{_round(selected['stabilizer_average_precision'])}"
                ),
                "frozen_test": frozen_text,
                "decision": "Rejected; frozen rank and kcal/mol error regressed",
                "tone": "bad",
            }
        )
        data["sources"].append("docs/ddg_aligned_retrieval_audit.json")
    if validation_audit is not None:
        estimate = validation_audit["current_historical_estimate"]
        interval = estimate["protein_cluster_bootstrap_mae_ci95"]
        data["status"].update(
            {
                "headline": (
                    "The strict-FP32 6B fusion remains the operational "
                    "baseline; its 0.467 kcal/mol result is historical, and "
                    "a new sealed family-held-out lockbox is required"
                ),
                "best_ddg": "6B hierarchy/state fusion · historical estimate",
                "structure_audit": (
                    "Sub-0.30 is not supported by current evidence; only "
                    "1/19 historical test proteins is below that MAE"
                ),
            }
        )
        data["expected_ddg"].update(
            {
                "generic_mae_ci95": [_round(value, 4) for value in interval],
                "target_mae": 0.3,
                "relative_reduction_needed": _round(
                    estimate["relative_mae_reduction_needed"], 4
                ),
            }
        )
        if data["models"]:
            data["models"][0]["status"] = "limited"
            data["models"][0]["test"] = (
                "Historical MegaScale protein holdout; no longer untouched"
            )
            data["models"][0]["role"] = (
                "Highest-accuracy operational generic screen; not a current "
                "SOTA claim"
            )
        for row in data["same_project_comparisons"]:
            if row["model"] == (
                "Our 6B hierarchy/state fusion · strict FP32"
            ):
                row.update(
                    {
                        "benchmark": (
                            "Historical generic protein holdout; no longer "
                            "untouched"
                        ),
                        "decision": (
                            "Retained operationally; requires a new sealed "
                            "lockbox for a SOTA claim"
                        ),
                        "tone": "bad",
                    }
                )
        optimization_audits.insert(
            0,
            {
                "candidate": "Sub-0.30 kcal/mol objective",
                "validation": (
                    "Current 6B MAE 0.467; protein-bootstrap 95% CI "
                    f"{_round(interval[0], 4)}–{_round(interval[1], 4)}"
                ),
                "frozen_test": (
                    "No untouched test remains; validation has 21.47% of "
                    "rows in three high-identity train-homolog groups"
                ),
                "decision": "Not demonstrated; freeze a prospective lockbox",
                "tone": "bad",
            },
        )
        data["sources"].append("docs/model_validation_audit.json")
    if stability_optimization is not None:
        optimization_audits.extend(
            experiment["dashboard"]
            for experiment in stability_optimization["experiments"]
        )
        data["optimization_training"] = stability_optimization[
            "training_scale"
        ]
        recent = data["optimization_training"]
        data["status"]["training"] += (
            f" · latest 600M audit {recent['core_optimizer_steps']:,} "
            f"updates / {recent['core_row_exposures'] / 1e6:.2f}M row "
            "exposures"
        )
        data["sources"].append("docs/stability_optimization_audit.json")
    if full_structure is not None:
        outer = full_structure["sealed_outer_evaluation"]
        single = outer["single"]
        double = outer["double_selected_additive"]
        development = full_structure["development_oof"]
        sequence_oof = development["sequence_only"]
        structure_oof = development["warm_started_full_structure"]
        structure_gate = development["structure_promotion_gate"]
        double_oof = development["double_mutants"]
        training = full_structure["training"]
        data["status"].update(
            {
                "headline": (
                    "New family-held-out outer result: 0.598 kcal/mol MAE; "
                    "historical 6B remains operational, while full "
                    "ProteinMPNN fusion and learned epistasis were not promoted"
                ),
                "best_ddg": (
                    "6B hierarchy/state fusion remains operational; 600M "
                    "full-protein run supplies the clean evidence estimate"
                ),
                "training": (
                    "173 WT proteins embedded once · 136,333 singles + "
                    "114,109 doubles · five family folds + sealed outer · "
                    "historical 204,000 optimizer updates retained for "
                    "continuity"
                ),
                "structure_audit": (
                    "Sub-0.30 is not supported; warm-started trainable "
                    "ProteinMPNN fusion improved OOF MAE by "
                    f"{_round(structure_gate['mae_improvement'], 4)}, below "
                    "the 0.02 scale gate"
                ),
            }
        )
        data["expected_ddg"].update(
            {
                "clean_family_mae": _round(single["mae"]),
                "clean_family_rmse": _round(single["rmse"]),
                "clean_family_mae_ci95": [
                    _round(value, 4)
                    for value in single[
                        "protein_bootstrap_mae_95_ci"
                    ]
                ],
                "interpretation": (
                    "The clean family-held-out estimate is about 0.60 "
                    "kcal/mol MAE (95% protein-bootstrap CI 0.559–0.645). "
                    "For a new GPCR, continue to treat roughly ±1 kcal/mol "
                    "as the operational error scale and prioritize ranking "
                    "plus experimental confirmation."
                ),
            }
        )
        data["losses"].insert(
            0,
            {
                "model": "600M full-protein state potential",
                "objective": "Huber + rank + stabilizer retrieval",
                "members": 1,
                "final_median": _round(
                    training["final_sequence_train_objective"], 4
                ),
                "final_range": None,
                "note": (
                    "70 final epochs after five family-clustered folds; "
                    "native FP32 and TF32 disabled. Structure refinement was "
                    "evaluated OOF but did not clear its promotion gate."
                ),
            },
        )
        data["models"].insert(
            2,
            {
                "name": "ESM-C 600M full-protein state potential",
                "role": (
                    "Leakage-clean evidence baseline; one-pass 20-state "
                    "screening"
                ),
                "status": "limited",
                "target": (
                    "ΔΔG, kcal/mol · AP "
                    f"{_round(single['stabilizer_average_precision'])}"
                ),
                "test": (
                    "New sealed MMseqs-family outer partition; selected "
                    "without outer metrics"
                ),
                "rows": int(single["rows"]),
                "spearman": _round(single["spearman"]),
                "pearson": _round(single["pearson"]),
                "mae": _round(single["mae"]),
                "rmse": _round(single["rmse"]),
            },
        )
        data["models"].insert(
            3,
            {
                "name": "600M full-protein additive doubles",
                "role": (
                    "Permutation-invariant additive prediction; learned "
                    "epistasis rejected on development OOF"
                ),
                "status": "limited",
                "target": "ΔΔG, kcal/mol · additive + reported residual",
                "test": "Same sealed MMseqs-family outer partition",
                "rows": int(double["rows"]),
                "spearman": _round(double["spearman"]),
                "pearson": _round(double["pearson"]),
                "mae": _round(double["mae"]),
                "rmse": _round(double["rmse"]),
            },
        )
        data["same_project_comparisons"].insert(
            0,
            {
                "model": "Trainable full ProteinMPNN → ESM-C fusion",
                "input": (
                    "Full WT ESM-C residues + raw backbone + masked "
                    "ProteinMPNN"
                ),
                "benchmark": (
                    "Five MMseqs-family development folds; outer excluded "
                    "from promotion"
                ),
                "metric": (
                    f"sequence MAE {_round(sequence_oof['mae'], 4)} → "
                    f"structure {_round(structure_oof['mae'], 4)}; "
                    f"ρ {_round(sequence_oof['spearman'], 4)} → "
                    f"{_round(structure_oof['spearman'], 4)}"
                ),
                "decision": (
                    "Not scaled to 6B: positive change was much smaller than "
                    "the predeclared 0.02 kcal/mol gate"
                ),
                "tone": "bad",
            },
        )
        optimization_audits.insert(
            0,
            {
                "candidate": "Full trainable structure architecture",
                "validation": (
                    f"family OOF MAE {sequence_oof['mae']:.4f}→"
                    f"{structure_oof['mae']:.4f}; AP "
                    f"{sequence_oof['stabilizer_average_precision']:.4f}→"
                    f"{structure_oof['stabilizer_average_precision']:.4f}"
                ),
                "frozen_test": (
                    f"sealed outer MAE {single['mae']:.4f}, ρ "
                    f"{single['spearman']:.4f}; 95% CI "
                    f"{single['protein_bootstrap_mae_95_ci'][0]:.4f}–"
                    f"{single['protein_bootstrap_mae_95_ci'][1]:.4f}"
                ),
                "decision": (
                    "Structure not promoted; additive selected for doubles "
                    f"(OOF MAE {double_oof['additive_mae']:.4f})"
                ),
                "tone": "bad",
            },
        )
        data["sources"].append(
            "docs/full_structure_training_audit.json"
        )
    data["optimization_audits"] = optimization_audits
    return data


HTML_TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>ProteinStabilizer · Model status</title>
  <style>
    :root {
      color-scheme: dark;
      --bg: #07110f;
      --panel: #0d1b18;
      --panel-2: #11231f;
      --ink: #edf7f2;
      --muted: #92aaa1;
      --line: #213a33;
      --green: #60d394;
      --lime: #b8f26b;
      --amber: #f4bc5c;
      --red: #ff7b72;
      --cyan: #67d8de;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background:
        radial-gradient(circle at 75% -10%, #173d31 0, transparent 34rem),
        linear-gradient(180deg, #081511 0, var(--bg) 45rem);
      color: var(--ink);
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont,
        "Segoe UI", sans-serif;
    }
    main { width: min(1320px, calc(100% - 36px)); margin: 0 auto; padding: 32px 0 70px; }
    header { display: grid; grid-template-columns: 1.5fr 1fr; gap: 24px; align-items: end; }
    .eyebrow { color: var(--green); font-size: 12px; font-weight: 800; letter-spacing: .16em; text-transform: uppercase; }
    h1 { margin: 8px 0 10px; font-size: clamp(34px, 6vw, 72px); letter-spacing: -.055em; line-height: .95; }
    h2 { margin: 0 0 18px; font-size: 20px; letter-spacing: -.02em; }
    h3 { margin: 0 0 5px; font-size: 16px; }
    p { color: var(--muted); line-height: 1.55; }
    .status-line { font-size: 18px; max-width: 820px; }
    .stamp { justify-self: end; text-align: right; color: var(--muted); font-size: 13px; line-height: 1.6; }
    .stamp b { color: var(--ink); }
    .grid { display: grid; gap: 16px; }
    .kpis { grid-template-columns: repeat(4, 1fr); margin: 28px 0 16px; }
    .two { grid-template-columns: 1.1fr .9fr; }
    .three { grid-template-columns: repeat(3, 1fr); }
    .panel, .kpi {
      border: 1px solid var(--line);
      background: linear-gradient(145deg, rgba(17,35,31,.94), rgba(10,23,20,.96));
      border-radius: 18px;
      padding: 20px;
      box-shadow: 0 18px 55px rgba(0,0,0,.17);
    }
    .kpi .label { color: var(--muted); font-size: 12px; text-transform: uppercase; letter-spacing: .08em; }
    .kpi .value { margin: 10px 0 3px; font-size: 30px; font-weight: 800; letter-spacing: -.04em; }
    .kpi .note { color: var(--muted); font-size: 12px; }
    .accent { color: var(--lime); }
    .warn { color: var(--amber); }
    .section { margin-top: 16px; }
    .model-list { display: grid; grid-template-columns: repeat(5, 1fr); gap: 12px; }
    .model {
      border: 1px solid var(--line);
      border-radius: 14px;
      padding: 15px;
      background: rgba(6, 18, 15, .48);
    }
    .tag { display: inline-flex; padding: 4px 8px; border-radius: 999px; font-size: 10px; font-weight: 800; text-transform: uppercase; letter-spacing: .08em; }
    .tag.production { color: #07110f; background: var(--green); }
    .tag.limited { color: #271900; background: var(--amber); }
    .model .role { min-height: 40px; font-size: 12px; color: var(--muted); }
    .metrics { display: grid; grid-template-columns: 1fr 1fr; gap: 7px; margin-top: 13px; }
    .metric { border-top: 1px solid var(--line); padding-top: 8px; }
    .metric span { display: block; color: var(--muted); font-size: 10px; text-transform: uppercase; }
    .metric b { font-size: 18px; }
    .bar-row { display: grid; grid-template-columns: 148px 1fr 48px; align-items: center; gap: 10px; margin: 11px 0; font-size: 12px; }
    .bar-track { height: 10px; background: #07110f; border-radius: 99px; overflow: hidden; }
    .bar { height: 100%; border-radius: inherit; background: linear-gradient(90deg, var(--green), var(--lime)); }
    .bar.alt { background: linear-gradient(90deg, #3aa7ad, var(--cyan)); }
    table { width: 100%; border-collapse: collapse; font-size: 13px; }
    th { color: var(--muted); text-align: left; font-weight: 600; padding: 9px 10px; border-bottom: 1px solid var(--line); }
    td { padding: 11px 10px; border-bottom: 1px solid rgba(33,58,51,.7); vertical-align: top; }
    tr:last-child td { border-bottom: 0; }
    a { color: var(--cyan); text-decoration: none; }
    a:hover { text-decoration: underline; }
    .callout { border-left: 3px solid var(--amber); padding: 5px 0 5px 16px; }
    .callout strong { color: var(--ink); }
    .loss-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; }
    .loss { background: rgba(6,18,15,.45); border: 1px solid var(--line); border-radius: 12px; padding: 13px; }
    .loss b { font-size: 22px; }
    .loss small { color: var(--muted); display: block; line-height: 1.4; margin-top: 6px; }
    .training-facts { display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px; margin-bottom: 14px; }
    .training-fact { background: rgba(6,18,15,.45); border: 1px solid var(--line); border-radius: 12px; padding: 13px; }
    .training-fact span { color: var(--muted); display: block; font-size: 10px; letter-spacing: .08em; text-transform: uppercase; }
    .training-fact b { display: block; font-size: 22px; margin-top: 5px; }
    .training-fact small { color: var(--muted); display: block; line-height: 1.35; margin-top: 4px; }
    .scatter-grid { display: grid; grid-template-columns: minmax(0, 1fr) 260px; gap: 20px; align-items: center; }
    .scatter-canvas { width: 100%; height: 560px; display: block; border-radius: 12px; background: #07110f; }
    .scatter-stats { display: grid; gap: 10px; }
    .scatter-stat { border-top: 1px solid var(--line); padding-top: 10px; }
    .scatter-stat span { color: var(--muted); display: block; font-size: 11px; text-transform: uppercase; }
    .scatter-stat b { font-size: 22px; }
    .foot { margin-top: 16px; color: var(--muted); font-size: 12px; }
    .legend { display: flex; gap: 14px; flex-wrap: wrap; color: var(--muted); font-size: 11px; }
    .dot { width: 8px; height: 8px; display: inline-block; border-radius: 50%; margin-right: 5px; }
    @media (max-width: 1050px) {
      .kpis, .model-list, .training-facts { grid-template-columns: repeat(2, 1fr); }
      .two, .three, header, .scatter-grid { grid-template-columns: 1fr; }
      .stamp { justify-self: start; text-align: left; }
    }
    @media (max-width: 620px) {
      main { width: min(100% - 22px, 1320px); padding-top: 22px; }
      .kpis, .model-list, .loss-grid, .training-facts { grid-template-columns: 1fr; }
      .scatter-canvas { height: 410px; }
      .bar-row { grid-template-columns: 110px 1fr 42px; }
      .table-wrap { overflow-x: auto; }
    }
  </style>
</head>
<body>
<main>
  <header>
    <div>
      <div class="eyebrow">ProteinStabilizer · model oversight</div>
      <h1>What is ready,<br>what is believable.</h1>
      <p class="status-line" id="headline"></p>
    </div>
    <div class="stamp" id="stamp"></div>
  </header>

  <section class="grid kpis" id="kpis"></section>

  <section class="panel section">
    <h2>Deployed and retained models</h2>
    <div class="model-list" id="models"></div>
  </section>

  <section class="grid two section">
    <div class="panel">
      <h2>Held-out rank performance</h2>
      <div id="rank-bars"></div>
      <div class="legend">
        <span><i class="dot" style="background:var(--green)"></i>protein-held-out</span>
        <span>Spearman ρ; higher is better</span>
      </div>
    </div>
    <div class="panel">
      <h2>Training scale and objective</h2>
      <div class="training-facts" id="training-facts"></div>
      <div class="loss-grid" id="losses"></div>
      <p class="foot">Huber values are optimization diagnostics. They are not kcal/mol and should not be compared with another model’s published loss.</p>
    </div>
  </section>

  <section class="panel section">
    <h2>Latest optimization gates</h2>
    <p>Rows below preserve prior optimization evidence. New promotions require family-clustered development plus a newly sealed outer lockbox; the historical test is continuity-only.</p>
    <div class="table-wrap"><table id="optimization-audits"></table></div>
  </section>

  <section class="panel section">
    <h2>Held-out ΔΔG: predicted versus experimental</h2>
    <div class="scatter-grid">
      <canvas class="scatter-canvas" id="ddg-scatter"></canvas>
      <div>
        <div class="scatter-stats" id="scatter-stats"></div>
        <p class="foot">Every point is one mutation from the historical protein-held-out MegaScale test. Both axes use the same unclipped kcal/mol scale; the diagonal is perfect calibration. Negative values are stabilizing. This set has been consulted by prior promotion gates and is not an untouched lockbox.</p>
      </div>
    </div>
  </section>

  <section class="grid two section">
    <div class="panel">
      <h2>Expected ΔΔG accuracy</h2>
      <div class="callout" id="ddg-callout"></div>
      <div id="ddg-bars"></div>
      <p class="foot">Negative project ΔΔG means stabilizing. The GPCR consensus is a percentile rank and is never labeled kcal/mol.</p>
    </div>
    <div class="panel">
      <h2>GPCR screening evidence</h2>
      <div id="gpcr"></div>
    </div>
  </section>

  <section class="panel section">
    <h2>Exact local comparisons</h2>
    <p>These are the comparisons that matter most because they were run inside this project. Rows still differ in endpoint; the benchmark and units are explicit.</p>
    <div class="table-wrap"><table id="local-comparison"></table></div>
  </section>

  <section class="grid two section">
    <div class="panel">
      <h2>ΔTm transfer heads</h2>
      <div class="table-wrap"><table id="transfer"></table></div>
      <p class="foot">ΔTm is measured in °C and cannot be compared numerically with ΔΔG in kcal/mol.</p>
    </div>
    <div class="panel">
      <h2>My assessment</h2>
      <div class="callout">
        <strong>Use the consensus to choose GPCR experiments, not to believe a GPCR decimal.</strong>
        <p>The strict-FP32 6B hierarchy/state fusion is the best signed generic ΔΔG model. GPCR screening adds the retained receptor rank and a small target-excluded GPCRdb family prior. This improves the local GPCR checks, but it remains a within-scan ordering score rather than calibrated stability or crystallization probability.</p>
      </div>
      <p><strong>Practical target:</strong> rank a broad single-mutant scan, retain general-ΔΔG support, and test a diverse panel rather than only near-duplicate top hits.</p>
    </div>
  </section>

  <section class="panel section">
    <h2>How this compares with external stability models</h2>
    <p>Literature metrics below are context, not a leaderboard: training sets, splits, sign conventions, structures, and endpoints differ.</p>
    <div class="table-wrap"><table id="literature"></table></div>
  </section>

  <p class="foot" id="sources"></p>
</main>
<script>
const DATA = __DASHBOARD_DATA__;
const fmt = (v, n=3) => Number(v).toFixed(n);
const table = (headers, rows) =>
  `<thead><tr>${headers.map(x => `<th>${x}</th>`).join("")}</tr></thead>` +
  `<tbody>${rows.map(row => `<tr>${row.map(x => `<td>${x}</td>`).join("")}</tr>`).join("")}</tbody>`;
const bar = (label, value, max=1, alt=false) =>
  `<div class="bar-row"><span>${label}</span><div class="bar-track"><div class="bar ${alt ? "alt" : ""}" style="width:${Math.max(0, Math.min(100, 100 * value / max))}%"></div></div><b>${fmt(value)}</b></div>`;

document.querySelector("#headline").textContent = DATA.status.headline;
document.querySelector("#stamp").innerHTML =
  `<b>Training</b> ${DATA.status.training}<br>` +
  `<b>Fast path</b> ${DATA.status.fast_path}<br>` +
  `<b>Best ΔΔG</b> ${DATA.status.best_ddg}<br>` +
  `<b>GPCR path</b> ${DATA.status.gpcr_path}<br>` +
  `<span class="warn">${DATA.status.structure_audit}</span>`;

const e = DATA.expected_ddg;
document.querySelector("#kpis").innerHTML = [
  ["Best generic ρ", fmt(DATA.models[0].spearman), DATA.models[0].test],
  ["Typical |ΔΔG error|", `${fmt(e.generic_mae, 2)} kcal/mol`, e.generic_mae_ci95 ? `protein-bootstrap 95% CI ${fmt(e.generic_mae_ci95[0], 2)}–${fmt(e.generic_mae_ci95[1], 2)}` : "MegaScale holdout"],
  ["Experimental transfer", `${fmt(e.experimental_mae, 2)} kcal/mol`, "ProTherm holdout"],
  ["C5aR GPCR AP", fmt(DATA.gpcr.c5ar.consensus_ap ?? DATA.gpcr.c5ar.rerank_ap), `${DATA.gpcr.c5ar.consensus_top50 ?? DATA.gpcr.c5ar.rerank_top50}/34 positives in top 50`],
].map(([label, value, note]) => `<div class="kpi"><div class="label">${label}</div><div class="value accent">${value}</div><div class="note">${note}</div></div>`).join("");

document.querySelector("#models").innerHTML = DATA.models.map(m => `
  <article class="model">
    <span class="tag ${m.status}">${m.status}</span>
    <h3 style="margin-top:12px">${m.name}</h3>
    <div class="role">${m.role}<br>${m.test} · n=${m.rows.toLocaleString()}</div>
    <div class="metrics">
      <div class="metric"><span>Spearman</span><b>${fmt(m.spearman)}</b></div>
      <div class="metric"><span>MAE</span><b>${fmt(m.mae)}</b></div>
      <div class="metric"><span>Pearson</span><b>${fmt(m.pearson)}</b></div>
      <div class="metric"><span>RMSE</span><b>${fmt(m.rmse)}</b></div>
    </div>
    <p class="foot">${m.target}</p>
  </article>`).join("");

document.querySelector("#rank-bars").innerHTML = DATA.models.map((m, i) => bar(m.name.replace("ESM-C ", ""), m.spearman, 1, i % 2)).join("");
const t = DATA.training_scale;
const recent = DATA.optimization_training;
document.querySelector("#training-facts").innerHTML = t ? [
  ["MegaScale rows", t.dataset_rows.toLocaleString(), `${t.train_rows.toLocaleString()} train · ${t.validation_rows.toLocaleString()} validation · ${t.test_rows.toLocaleString()} test`],
  ["Batch / epochs", `${t.batch_size} / ${t.epochs_per_member}`, `${(t.examples_per_member / 1e6).toFixed(2)}M row exposures per member`],
  ["Optimizer updates", t.total_optimizer_steps.toLocaleString(), `${t.main_optimizer_steps.toLocaleString()} main + ${t.state_optimizer_steps.toLocaleString()} state`],
  ["Measured training", `${fmt(t.total_minutes, 1)} min`, `${fmt(t.main_minutes, 1)} main + ${fmt(t.state_minutes, 1)} state`],
  ...(recent ? [
    ["Latest 600M study", `${(recent.core_row_exposures / 1e6).toFixed(2)}M`, `${recent.core_optimizer_steps.toLocaleString()} updates · FP32 / TF32 off`],
    ["Full pretraining", recent.pretraining_rows.toLocaleString(), `${recent.pretraining_epochs} epochs + two ${recent.fine_tuning_epochs}-epoch paired fine-tunes`],
  ] : []),
].map(([label, value, note]) => `<div class="training-fact"><span>${label}</span><b>${value}</b><small>${note}</small></div>`).join("") : "";
document.querySelector("#losses").innerHTML = DATA.losses.map(l => `
  <div class="loss"><h3>${l.model}</h3><b>${fmt(l.final_median, 4)}</b>
  <small>final train ${l.objective}${l.final_range ? ` · range ${fmt(l.final_range[0],4)}–${fmt(l.final_range[1],4)}` : ""}<br>${l.note}</small></div>`).join("");

document.querySelector("#optimization-audits").innerHTML = table(
  ["Candidate", "Protein-held-out validation", "Historical confirmation", "Decision"],
  DATA.optimization_audits.map(r => [r.candidate, r.validation, r.frozen_test, `<span class="${r.tone === "good" ? "accent" : "warn"}">${r.decision}</span>`])
);

const scatter = DATA.ddg_scatter;
if (scatter) {
  const m = scatter.metrics;
  document.querySelector("#scatter-stats").innerHTML = [
    ["Held-out mutations", scatter.rows.toLocaleString()],
    ["Pearson r", fmt(m.pearson)],
    ["Spearman ρ", fmt(m.spearman)],
    ["MAE", `${fmt(m.mae)} kcal/mol`],
    ["RMSE", `${fmt(m.rmse)} kcal/mol`],
    ...(e.generic_mae_ci95 ? [["MAE protein CI", `${fmt(e.generic_mae_ci95[0])}–${fmt(e.generic_mae_ci95[1])}`]] : []),
    ["Calibration", `slope ${fmt(scatter.calibration.slope)} · intercept ${fmt(scatter.calibration.intercept)}`],
  ].map(([label, value]) => `<div class="scatter-stat"><span>${label}</span><b>${value}</b></div>`).join("");

  const canvas = document.querySelector("#ddg-scatter");
  const drawScatter = () => {
    const rect = canvas.getBoundingClientRect();
    const ratio = window.devicePixelRatio || 1;
    canvas.width = Math.max(1, Math.round(rect.width * ratio));
    canvas.height = Math.max(1, Math.round(rect.height * ratio));
    const ctx = canvas.getContext("2d");
    ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
    const width = rect.width;
    const height = rect.height;
    const margin = {left: 68, right: 22, top: 24, bottom: 62};
    const plotWidth = width - margin.left - margin.right;
    const plotHeight = height - margin.top - margin.bottom;
    const minimum = scatter.axes.minimum;
    const maximum = scatter.axes.maximum;
    const span = maximum - minimum;
    const x = value => margin.left + (value - minimum) / span * plotWidth;
    const y = value => margin.top + (maximum - value) / span * plotHeight;
    ctx.clearRect(0, 0, width, height);
    ctx.fillStyle = "#07110f";
    ctx.fillRect(0, 0, width, height);
    ctx.font = "11px ui-sans-serif, system-ui, sans-serif";
    ctx.textAlign = "center";
    ctx.textBaseline = "top";
    for (let index = 0; index <= 8; index++) {
      const value = minimum + span * index / 8;
      const px = x(value);
      const py = y(value);
      ctx.strokeStyle = "rgba(146,170,161,.14)";
      ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(px, margin.top); ctx.lineTo(px, margin.top + plotHeight); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(margin.left, py); ctx.lineTo(margin.left + plotWidth, py); ctx.stroke();
      ctx.fillStyle = "#92aaa1";
      ctx.fillText(value.toFixed(1), px, margin.top + plotHeight + 9);
      ctx.textAlign = "right";
      ctx.textBaseline = "middle";
      ctx.fillText(value.toFixed(1), margin.left - 9, py);
      ctx.textAlign = "center";
      ctx.textBaseline = "top";
    }
    if (minimum <= 0 && maximum >= 0) {
      ctx.strokeStyle = "rgba(244,188,92,.38)";
      ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(x(0), margin.top); ctx.lineTo(x(0), margin.top + plotHeight); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(margin.left, y(0)); ctx.lineTo(margin.left + plotWidth, y(0)); ctx.stroke();
    }
    ctx.save();
    ctx.beginPath();
    ctx.rect(margin.left, margin.top, plotWidth, plotHeight);
    ctx.clip();
    ctx.fillStyle = "rgba(103,216,222,.14)";
    for (let index = 0; index < scatter.rows; index++) {
      ctx.fillRect(x(scatter.experimental_ddg[index]) - .8, y(scatter.predicted_ddg[index]) - .8, 1.6, 1.6);
    }
    ctx.strokeStyle = "rgba(184,242,107,.88)";
    ctx.lineWidth = 2;
    ctx.setLineDash([7, 5]);
    ctx.beginPath(); ctx.moveTo(x(minimum), y(minimum)); ctx.lineTo(x(maximum), y(maximum)); ctx.stroke();
    ctx.setLineDash([]);
    const fitStart = scatter.calibration.slope * minimum + scatter.calibration.intercept;
    const fitStop = scatter.calibration.slope * maximum + scatter.calibration.intercept;
    ctx.strokeStyle = "rgba(244,188,92,.88)";
    ctx.lineWidth = 2;
    ctx.beginPath(); ctx.moveTo(x(minimum), y(fitStart)); ctx.lineTo(x(maximum), y(fitStop)); ctx.stroke();
    ctx.restore();
    ctx.fillStyle = "#edf7f2";
    ctx.font = "12px ui-sans-serif, system-ui, sans-serif";
    ctx.textAlign = "center";
    ctx.fillText("Experimental ΔΔG (kcal/mol)", margin.left + plotWidth / 2, height - 25);
    ctx.save();
    ctx.translate(18, margin.top + plotHeight / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.fillText("Predicted ΔΔG (kcal/mol)", 0, 0);
    ctx.restore();
    ctx.textAlign = "left";
    ctx.fillStyle = "#b8f26b"; ctx.fillText("— perfect", margin.left + 8, margin.top + 8);
    ctx.fillStyle = "#f4bc5c"; ctx.fillText("— fitted", margin.left + 78, margin.top + 8);
  };
  drawScatter();
  new ResizeObserver(drawScatter).observe(canvas);
}

document.querySelector("#ddg-callout").innerHTML =
  `<strong>Operational expectation: roughly ±1 kcal/mol for a new GPCR.</strong><p>${e.interpretation}</p>`;
document.querySelector("#ddg-bars").innerHTML =
  bar("600M v2 Megascale MAE", e.generic_mae, 1.5) +
  bar("Retained 6B MAE", e.retained_6b_mae, 1.5, true) +
  bar("6B ProTherm MAE", e.experimental_mae, 1.5, true) +
  bar("S669 local MAE", e.s669_mae, 1.5);

const g = DATA.gpcr;
const s = DATA.structure_scale;
document.querySelector("#gpcr").innerHTML =
  `<h3>Within-receptor rank</h3>` +
  bar("Fast development", g.development.fast_macro_spearman, 1) +
  bar("6B development", g.development.rerank_macro_spearman, 1, true) +
  (g.development.consensus_macro_spearman == null ? "" :
    bar("Promoted consensus development", g.development.consensus_macro_spearman, 1, true)) +
  bar("Fast official test", g.official_test.fast_macro_spearman, 1) +
  bar("6B official test", g.official_test.rerank_macro_spearman, 1, true) +
  (g.official_test.consensus_macro_spearman == null ? "" :
    bar("Promoted consensus official", g.official_test.consensus_macro_spearman, 1, true)) +
  `<h3 style="margin-top:20px">Independent C5aR scan</h3>` +
  bar("Fast AUC", g.c5ar.fast_auc, 1) +
  bar("6B rerank AUC", g.c5ar.rerank_auc, 1, true) +
  (g.c5ar.consensus_auc == null ? "" :
    bar("Promoted consensus AUC", g.c5ar.consensus_auc, 1, true)) +
  bar("Fast AP", g.c5ar.fast_ap, 1) +
  bar("6B rerank AP", g.c5ar.rerank_ap, 1, true) +
  (g.c5ar.consensus_ap == null ? "" :
    bar("Promoted consensus AP", g.c5ar.consensus_ap, 1, true)) +
  `<p class="foot">${g.c5ar.rows} substitutions · ${g.c5ar.positives} reported thermostable. The promoted 50/40/10 rank reaches top-50 ${g.c5ar.consensus_top50 ?? g.c5ar.rerank_top50}/34; its paired AP-difference interval is ${g.c5ar.consensus_ap_difference_ci95 ? `[${fmt(g.c5ar.consensus_ap_difference_ci95[0])}, ${fmt(g.c5ar.consensus_ap_difference_ci95[1])}]` : "not available"}. The 12-row official test and C5aR scan have both been consulted, so neither is untouched.</p>` +
  `<h3 style="margin-top:20px">ProteinMPNN logic scaled to 6B</h3>` +
  bar("Development", s.development_candidate, 1) +
  bar("Official check", s.official_candidate, 1, true) +
  bar("C5aR AP", s.c5ar_candidate_ap, 1) +
  `<p class="foot">The development-selected ${Math.round(100*s.selected_weight)}% structure blend rose from ρ ${fmt(s.development_baseline)} to ${fmt(s.development_candidate)}, but official macro ρ fell from ${fmt(s.official_baseline)} to ${fmt(s.official_candidate)} and C5aR top-50 recovery fell ${s.c5ar_baseline_top50}→${s.c5ar_candidate_top50}. Rejected. A post-hoc ${Math.round(100*s.small_prior_weight)}% prior reached AP ${fmt(s.small_prior_ap)}, but kept ${s.small_prior_top50} top-50 positives and its AP difference CI [${fmt(s.small_prior_ap_ci[0])}, ${fmt(s.small_prior_ap_ci[1])}] includes zero.</p>`;

document.querySelector("#local-comparison").innerHTML = table(
  ["Model", "Input", "Benchmark", "Measured result", "Decision"],
  DATA.same_project_comparisons.map(r => [r.model, r.input, r.benchmark, r.metric, `<span class="${r.tone === "good" ? "accent" : "warn"}">${r.decision}</span>`])
);
document.querySelector("#transfer").innerHTML = table(
  ["Model", "Endpoint", "ρ", "MAE", "RMSE", "n"],
  DATA.transfer.map(r => [r.model, r.endpoint, fmt(r.spearman), fmt(r.mae), fmt(r.rmse), r.rows])
);
document.querySelector("#literature").innerHTML = table(
  ["Model", "Input", "Published context", "Relevance here"],
  DATA.literature_context.map(r => [`<a href="${r.url}">${r.model}</a>`, r.input, r.reported, r.fit])
);
document.querySelector("#sources").textContent = "Generated from: " + DATA.sources.join(" · ");
</script>
</body>
</html>
"""


def build(output: Path) -> None:
    data = json.dumps(dashboard_data(), separators=(",", ":"), ensure_ascii=False)
    html = HTML_TEMPLATE.replace("__DASHBOARD_DATA__", data)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(html, encoding="utf-8")


if __name__ == "__main__":
    build(ROOT / "docs/model_dashboard.html")
