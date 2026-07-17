from __future__ import annotations

from scripts.build_model_dashboard import build, dashboard_data


def test_model_dashboard_is_generated_from_recorded_metrics(tmp_path) -> None:
    data = dashboard_data()
    output = tmp_path / "model_dashboard.html"
    build(output)
    html = output.read_text(encoding="utf-8")

    assert "__DASHBOARD_DATA__" not in html
    assert "Ready for mutation screening" in html
    assert data["models"][1]["name"] == "ESM-C 6B single"
    assert data["models"][1]["spearman"] == 0.818
    assert data["expected_ddg"]["generic_mae"] == 0.516
    assert data["gpcr"]["c5ar"]["rerank_ap"] == 0.318
    assert data["structure_scale"]["status"] == "rejected_for_production"
    assert data["structure_scale"]["development_candidate"] == 0.489
    assert data["structure_scale"]["official_candidate"] == -0.5
    assert data["structure_scale"]["c5ar_candidate_top50"] == 11
    assert "ProteinMPNN logic scaled to 6B" in html
