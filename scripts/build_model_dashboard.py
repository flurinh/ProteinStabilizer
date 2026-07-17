#!/usr/bin/env python3
"""Build the self-contained model-status dashboard from recorded metrics."""

from __future__ import annotations

import json
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _load(relative: str) -> dict[str, object]:
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


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

    loss_600m = _final_losses(single_600m)
    loss_6b = _final_losses(single_6b)

    return {
        "status": {
            "headline": "Ready for mutation screening; GPCR magnitudes remain uncertain",
            "fast_path": "ESM-C 600M",
            "best_ddg": "ESM-C 6B",
            "gpcr_path": "600M rank + optional 6B rerank",
            "training": "Complete",
            "structure_audit": "ProteinMPNN logic scaled to 6B; rejected for inconsistent transfer",
        },
        "expected_ddg": {
            "generic_mae": _round(test_6b["mae"]),
            "generic_rmse": _round(test_6b["rmse"]),
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
    .foot { margin-top: 16px; color: var(--muted); font-size: 12px; }
    .legend { display: flex; gap: 14px; flex-wrap: wrap; color: var(--muted); font-size: 11px; }
    .dot { width: 8px; height: 8px; display: inline-block; border-radius: 50%; margin-right: 5px; }
    @media (max-width: 1050px) {
      .kpis, .model-list { grid-template-columns: repeat(2, 1fr); }
      .two, .three, header { grid-template-columns: 1fr; }
      .stamp { justify-self: start; text-align: left; }
    }
    @media (max-width: 620px) {
      main { width: min(100% - 22px, 1320px); padding-top: 22px; }
      .kpis, .model-list, .loss-grid { grid-template-columns: 1fr; }
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
      <h2>Training objective</h2>
      <div class="loss-grid" id="losses"></div>
      <p class="foot">Huber values are optimization diagnostics. They are not kcal/mol and should not be compared with another model’s published loss.</p>
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
        <strong>Use it now for screening, not for believing a decimal.</strong>
        <p>The 6B model is genuinely better on protein-disjoint generic stability. The GPCR evidence is directionally useful but small, so receptor ranking is more trustworthy than absolute ΔΔG. ProteinMPNN structure likelihood was scaled from the 600M development workflow to the 6B rank; its large development gain did not transfer, so the production path remains sequence-only.</p>
      </div>
      <p><strong>Practical target:</strong> rank a broad single-mutant scan, inspect the 6B rerank, retain general-ΔΔG support, and test a diverse panel rather than only near-duplicate top hits.</p>
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
  ["Best generic ρ", fmt(DATA.models[1].spearman), "6B · protein-held-out"],
  ["Typical |ΔΔG error|", `${fmt(e.generic_mae, 2)} kcal/mol`, "Megascale holdout"],
  ["Experimental transfer", `${fmt(e.experimental_mae, 2)} kcal/mol`, "ProTherm holdout"],
  ["C5aR rerank AP", fmt(DATA.gpcr.c5ar.rerank_ap), `${DATA.gpcr.c5ar.rerank_top50}/34 positives in top 50`],
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
document.querySelector("#losses").innerHTML = DATA.losses.map(l => `
  <div class="loss"><h3>${l.model}</h3><b>${fmt(l.final_median, 4)}</b>
  <small>final train ${l.objective}${l.final_range ? ` · range ${fmt(l.final_range[0],4)}–${fmt(l.final_range[1],4)}` : ""}<br>${l.note}</small></div>`).join("");

document.querySelector("#ddg-callout").innerHTML =
  `<strong>Operational expectation: roughly ±1 kcal/mol for a new GPCR.</strong><p>${e.interpretation}</p>`;
document.querySelector("#ddg-bars").innerHTML =
  bar("6B Megascale MAE", e.generic_mae, 1.5) +
  bar("6B ProTherm MAE", e.experimental_mae, 1.5, true) +
  bar("S669 local MAE", e.s669_mae, 1.5);

const g = DATA.gpcr;
const s = DATA.structure_scale;
document.querySelector("#gpcr").innerHTML =
  `<h3>Within-receptor rank</h3>` +
  bar("Fast development", g.development.fast_macro_spearman, 1) +
  bar("6B development", g.development.rerank_macro_spearman, 1, true) +
  bar("Fast official test", g.official_test.fast_macro_spearman, 1) +
  bar("6B official test", g.official_test.rerank_macro_spearman, 1, true) +
  `<h3 style="margin-top:20px">Independent C5aR scan</h3>` +
  bar("Fast AUC", g.c5ar.fast_auc, 1) +
  bar("6B rerank AUC", g.c5ar.rerank_auc, 1, true) +
  bar("Fast AP", g.c5ar.fast_ap, 1) +
  bar("6B rerank AP", g.c5ar.rerank_ap, 1, true) +
  `<p class="foot">${g.c5ar.rows} substitutions · ${g.c5ar.positives} reported thermostable. The 12-row official test is confirmatory, not untouched.</p>` +
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
