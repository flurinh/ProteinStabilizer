from __future__ import annotations

import pytest

from scripts.build_model_dashboard import build, dashboard_data


def test_model_dashboard_is_generated_from_recorded_metrics(tmp_path) -> None:
    data = dashboard_data()
    output = tmp_path / "model_dashboard.html"
    build(output)
    html = output.read_text(encoding="utf-8")

    assert "__DASHBOARD_DATA__" not in html
    assert "ESM-C 6B hierarchy/state fusion · strict FP32" in html
    assert data["models"][0]["name"] == (
        "ESM-C 6B hierarchy/state fusion · strict FP32"
    )
    assert data["models"][0]["spearman"] == 0.856
    assert data["models"][0]["status"] == "limited"
    assert "historical" in data["status"]["headline"]
    assert data["expected_ddg"]["generic_mae"] == 0.467
    assert data["expected_ddg"]["generic_mae_ci95"] == [0.4267, 0.5092]
    assert data["expected_ddg"]["target_mae"] == 0.3
    assert data["expected_ddg"]["family_shadow_mae"] == 0.4763
    assert data["expected_ddg"]["family_shadow_mae_ci95"] == [
        0.4317,
        0.5227,
    ]
    assert data["expected_ddg"]["family_shadow_spearman"] == 0.7623
    assert data["training_scale"]["dataset_rows"] == 136333
    assert data["training_scale"]["train_rows"] == 104315
    assert data["training_scale"]["batch_size"] == 256
    assert data["training_scale"]["epochs_per_member"] == 50
    assert data["training_scale"]["examples_per_member"] == 5215750
    assert data["training_scale"]["total_optimizer_steps"] == 204000
    assert data["training_scale"]["total_minutes"] == 55.5
    assert data["optimization_training"]["core_optimizer_steps"] == 115880
    assert data["optimization_training"]["core_row_exposures"] == 29630170
    assert data["ddg_scatter"]["rows"] == 19645
    assert data["ddg_scatter"]["units"] == "kcal/mol"
    assert data["ddg_scatter"]["axes"]["clipped"] is False
    assert data["ddg_scatter"]["metrics"]["mae"] == pytest.approx(
        0.4670035
    )
    assert data["models"][1]["name"] == (
        "ESM-C 6B fused double-mutant model"
    )
    assert data["models"][1]["spearman"] == 0.749
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
    assert "Promoted GPCR evolutionary consensus" in html
    assert "Experimental ΔΔG (kcal/mol)" in html
    assert "204,000 optimizer" in html
    assert any(
        row["candidate"] == "Sub-0.30 kcal/mol objective"
        for row in data["optimization_audits"]
    )
    assert any(
        row["candidate"] == "Portable masked/geometry accuracy ensemble"
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
    assert "Sub-0.30 is not supported" in html
    assert "strict-FP32 6B scaling is pending GPU headroom" in html
    assert "docs/accuracy_optimization_audit.json" in data["sources"]
