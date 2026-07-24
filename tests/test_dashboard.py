from __future__ import annotations

import argparse
import json

import pandas as pd
import pytest

from scripts.build_model_dashboard import build, dashboard_data
from scripts.export_ddg_scatter import export_accuracy


def test_model_dashboard_is_generated_from_recorded_metrics(tmp_path) -> None:
    data = dashboard_data()
    output = tmp_path / "model_dashboard.html"
    build(output)
    html = output.read_text(encoding="utf-8")

    assert "__DASHBOARD_DATA__" not in html
    assert "ESM-C 6B hierarchy/state fusion · strict FP32" in html
    assert data["models"][0]["name"] == (
        "ESM-C 6B/600M calibrated accuracy ensemble"
    )
    assert data["models"][0]["spearman"] == 0.78
    assert data["models"][0]["status"] == "limited"
    assert "strict-FP32 6B state" in data["status"]["headline"]
    assert data["expected_ddg"]["generic_mae"] == 0.467
    assert data["expected_ddg"]["generic_mae_ci95"] == [0.4267, 0.5092]
    assert data["expected_ddg"]["target_mae"] == 0.3
    assert data["expected_ddg"]["development_oof_mae"] == 0.4681
    assert data["expected_ddg"]["family_shadow_mae"] == 0.4583
    assert data["expected_ddg"]["family_shadow_mae_ci95"] == [
        0.4141,
        0.5104,
    ]
    assert data["expected_ddg"]["family_shadow_spearman"] == 0.7796
    assert data["training_scale"]["dataset_rows"] == 136333
    assert data["training_scale"]["train_rows"] == 104315
    assert data["training_scale"]["batch_size"] == 256
    assert data["training_scale"]["epochs_per_member"] == 50
    assert data["training_scale"]["examples_per_member"] == 5215750
    assert data["training_scale"]["total_optimizer_steps"] == 204000
    assert data["training_scale"]["total_minutes"] == 55.5
    assert data["optimization_training"]["core_optimizer_steps"] == 115880
    assert data["optimization_training"]["core_row_exposures"] == 29630170
    assert data["ddg_scatter"]["evaluated_rows"] == 108408
    assert data["ddg_scatter"]["rows"] == 25000
    assert data["ddg_scatter"]["units"] == "kcal/mol"
    assert data["ddg_scatter"]["axes"]["clipped"] is False
    assert data["ddg_scatter"]["metrics"]["mae"] == pytest.approx(
        0.4680908
    )
    assert any(
        row["name"] == "ESM-C 6B fused double-mutant model"
        and row["spearman"] == 0.749
        for row in data["models"]
    )
    assert any(
        row["model"] == "6B v2 GPCR crystallization rank diagnostic"
        for row in data["transfer"]
    )
    assert data["gpcr"]["c5ar"]["rerank_ap"] == 0.318
    assert data["gpcr"]["c5ar"]["fusion_ap"] == 0.256
    assert data["gpcr"]["development"]["consensus_macro_spearman"] == 0.338
    assert data["gpcr"]["official_test"]["consensus_macro_spearman"] == 0.9
    assert data["gpcr"]["c5ar"]["consensus_auc"] == 0.718
    assert data["gpcr"]["c5ar"]["consensus_ap"] == 0.377
    assert data["gpcr"]["c5ar"]["consensus_top50"] == 17
    assert data["gpcr"]["accuracy_transfer"] == {
        "rows": 97,
        "receptors": 11,
        "pooled_spearman": 0.029,
        "macro_spearman": 0.244,
        "clean_mptherm_macro_spearman": 0.264,
        "best_nested_fusion_macro_spearman": 0.219,
        "decision": "rejected; retain existing GPCR rank",
    }
    assert data["gpcr"]["multimutant_transfer"] == {
        "pth1r_rows": 3,
        "pth1r_spearman": 0.5,
        "pth1r_sign_accuracy": 1.0,
        "ntr1_rows": 5,
        "ntr1_spearman": 0.6,
        "ntr1_sign_accuracy": 0.0,
        "six_b_wt_sequences": 2,
        "masked_sites": 24,
        "decision": (
            "No promotion: PTH1R passes direction on 3/3 destabilizing "
            "variants, while additive NTR1 direction fails 0/5 stabilizing "
            "variants."
        ),
    }
    assert "Promoted GPCR evolutionary consensus" in html
    assert "Fixed GPCR multi-mutant transfer" in html
    assert "Experimental ΔΔG (kcal/mol)" in html
    assert "108,408 five-fold OOF mutations" in html
    assert any(
        row["candidate"] == "Sub-0.30 kcal/mol objective"
        for row in data["optimization_audits"]
    )
    assert any(
        row["candidate"] == "Portable masked/geometry accuracy ensemble"
        for row in data["optimization_audits"]
    )
    assert any(
        row["candidate"] == "Strict-FP32 6B accuracy scale-up"
        for row in data["optimization_audits"]
    )
    assert data["optimization_audits"][0]["candidate"] == (
        "Monotone affine ΔΔG calibration"
    )
    assert any(
        row["candidate"]
        == "6B expected-ΔΔG transfer to quantitative GPCR ΔTm"
        for row in data["optimization_audits"]
    )
    assert any(
        row["candidate"]
        == "Additive 6B/600M GPCR multi-mutant transfer"
        for row in data["optimization_audits"]
    )
    assert any(
        row["name"] == "ESM-C 600M portable accuracy ensemble"
        and row["mae"] == 0.476
        and row["spearman"] == 0.762
        for row in data["models"]
    )
    assert any(row["model"] == "SPURS" for row in data["literature_context"])
    assert any(
        row["model"] == "JanusDDG" for row in data["literature_context"]
    )
    assert data["structure_scale"]["status"] == "rejected_for_production"
    assert data["structure_scale"]["development_candidate"] == 0.489
    assert data["structure_scale"]["official_candidate"] == -0.5
    assert data["structure_scale"]["c5ar_candidate_top50"] == 11
    assert "sub-0.30 and quantitative gpcr accuracy remain" in html.lower()
    assert (
        "monotone calibration transfers from tuning fold 0"
        in html.lower()
    )
    assert "docs/accuracy_optimization_audit.json" in data["sources"]
    assert "docs/esmc6b_accuracy_scale_audit.json" in data["sources"]
    assert "docs/esmc6b_accuracy_scatter.json" in data["sources"]
    assert (
        "docs/esmc6b_accuracy_calibration_audit.json"
        in data["sources"]
    )
    assert (
        "docs/esmc6b_accuracy_gpcr_transfer_audit.json"
        in data["sources"]
    )
    assert "docs/klenk2023_gpcr_multimutant_audit.json" in data["sources"]


def test_accuracy_scatter_uses_all_rows_for_metrics_and_bounded_plot(
    tmp_path,
) -> None:
    predictions = tmp_path / "predictions.csv"
    pd.DataFrame(
        {
            "protein_id": ["a", "a", "b", "b"],
            "fold": [0, 0, 1, 1],
            "experimental_ddg": [-1.0, 0.0, 1.0, 2.0],
            "ensemble_ddg": [-0.5, 0.1, 0.8, 1.5],
            "calibrated_ddg": [3.0, 4.0, 5.0, 6.0],
        }
    ).to_csv(predictions, index=False)
    output = tmp_path / "scatter.json"
    report = export_accuracy(
        argparse.Namespace(
            accuracy_predictions=predictions,
            max_points=3,
            seed=7,
            output=output,
        )
    )
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert report["evaluated_rows"] == 4
    assert report["rows"] == 3
    assert report["metrics"]["n"] == 4
    assert len(payload["experimental_ddg"]) == 3
    assert min(payload["predicted_ddg"]) >= 3.0
    assert payload["provenance"]["prediction_column"] == "calibrated_ddg"
    assert payload["provenance"]["outer_used"] is False
