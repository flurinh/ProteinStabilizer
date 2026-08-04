from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "evaluate_klenk2023_exact_set.py"


def _module():
    spec = importlib.util.spec_from_file_location(
        "evaluate_klenk2023_exact_set", SCRIPT
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_request_plan_deduplicates_shared_constituents() -> None:
    module = _module()
    frame = pd.DataFrame(
        [
            {
                "uniprot_id": "P12345",
                "wt_sequence": "ACDE",
                "mutation_set": "A1V,C2D",
            },
            {
                "uniprot_id": "P12345",
                "wt_sequence": "ACDE",
                "mutation_set": "A1V,E4W",
            },
        ]
    )
    plan = module.build_request_plan(frame)["P12345"]

    assert len(plan) == 6
    assert plan[0].sequence == "ACDE"
    assert plan[0].positions == (1, 2, 4)
    assert len({request.sequence_hash for request in plan[1:4]}) == 3


def test_summary_keeps_additive_and_epistasis_components_separate() -> None:
    module = _module()
    benchmark = pd.DataFrame(
        [
            {
                "uniprot_id": "P12345",
                "protein_id": "receptor",
                "variant": "v1",
                "benchmark_role": "new_receptor_lockbox",
                "mutation_count": 3,
                "experimental_delta_tm_mean_c": 2.0,
            },
            {
                "uniprot_id": "P12345",
                "protein_id": "receptor",
                "variant": "v2",
                "benchmark_role": "new_receptor_lockbox",
                "mutation_count": 3,
                "experimental_delta_tm_mean_c": 1.0,
            },
            {
                "uniprot_id": "P12345",
                "protein_id": "receptor",
                "variant": "v3",
                "benchmark_role": "new_receptor_lockbox",
                "mutation_count": 3,
                "experimental_delta_tm_mean_c": -1.0,
            },
        ]
    )
    predictions = [
        {
            "uniprot_id": "P12345",
            "variant": "v1",
            "additive_ddg": 1.0,
            "epistasis_ddg": -2.0,
            "total_ddg": -1.0,
        },
        {
            "uniprot_id": "P12345",
            "variant": "v2",
            "additive_ddg": 0.5,
            "epistasis_ddg": -1.0,
            "total_ddg": -0.5,
        },
        {
            "uniprot_id": "P12345",
            "variant": "v3",
            "additive_ddg": -0.5,
            "epistasis_ddg": 1.5,
            "total_ddg": 1.0,
        },
    ]

    result, metrics = module.summarize_predictions(benchmark, predictions)

    assert result["learned_epistasis_ddg"].tolist() == [-2.0, -1.0, 1.5]
    exact = metrics["receptor"]["exact_set_total"]
    additive = metrics["receptor"]["same_model_additive"]
    assert exact["favorable_sign_accuracy"] == 1.0
    assert additive["favorable_sign_accuracy"] == 0.0
