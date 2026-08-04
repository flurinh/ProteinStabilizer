from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "evaluate_klenk2023_gpcr_multimutant.py"


def _module():
    spec = importlib.util.spec_from_file_location(
        "evaluate_klenk2023_gpcr_multimutant", SCRIPT
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_evaluation_sums_unordered_constituents_and_keeps_units_separate(
    tmp_path: Path,
) -> None:
    module = _module()
    benchmark = pd.DataFrame(
        [
            {
                "protein_id": "receptor",
                "uniprot_id": "P12345",
                "variant": "v1",
                "mutation_set": "A1V,C3D",
                "mutation_count": 2,
                "experimental_delta_tm_mean_c": 2.0,
                "experimental_delta_tm_sd_c": 0.2,
                "benchmark_role": "new_receptor_lockbox",
            }
        ]
    )
    screen = pd.DataFrame(
        [
            {
                "mutation": "C3D",
                "expected_general_ddg": -0.25,
                "state_ddg": -0.5,
                "portable_prior_ddg": 0.1,
            },
            {
                "mutation": "A1V",
                "expected_general_ddg": -0.75,
                "state_ddg": -0.2,
                "portable_prior_ddg": -0.3,
            },
        ]
    )
    benchmark_path = tmp_path / "benchmark.csv"
    screen_path = tmp_path / "screen.csv"
    checkpoint = tmp_path / "checkpoint.pt"
    benchmark.to_csv(benchmark_path, index=False)
    screen.to_csv(screen_path, index=False)
    checkpoint.write_bytes(b"fixed")

    audit = module.evaluate(
        benchmark_path,
        {"P12345": screen_path},
        tmp_path / "result.csv",
        tmp_path / "audit.json",
        checkpoint=checkpoint,
    )

    row = audit["rows"][0]
    assert row["additive_expected_general_ddg"] == -1.0
    assert row["favorable_expected_general_ddg"] == 1.0
    assert audit["prediction"]["unit_policy"].startswith("rank and direction")
    assert "mae" not in audit["results_by_receptor"]["receptor"]
