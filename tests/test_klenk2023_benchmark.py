from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "prepare_klenk2023_gpcr_multimutant.py"


def _module():
    spec = importlib.util.spec_from_file_location(
        "prepare_klenk2023_gpcr_multimutant", SCRIPT
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_frozen_variant_cohort_is_role_separated_and_complete() -> None:
    module = _module()
    variants = module.VARIANTS
    assert len(variants) == 8
    assert {
        value.variant
        for value in variants
        if value.benchmark_role == "new_receptor_lockbox"
    } == {"P34_05", "P34_13", "P14_12"}
    assert {
        value.variant
        for value in variants
        if value.benchmark_role == "prior_receptor_continuity"
    } == {"N8", "N12", "N13", "N21", "N23"}
    assert {len(value.mutations) for value in variants} == {3, 4, 5, 6, 8}


def test_canonical_mutation_set_sorts_and_validates_wild_type() -> None:
    module = _module()
    assert module.canonical_mutation_set(
        ("C3A", "A1V"),
        "ABC",
    ) == ("A1V", "C3A")

    try:
        module.canonical_mutation_set(("C2A",), "ABC")
    except ValueError as error:
        assert "disagrees with canonical sequence" in str(error)
    else:
        raise AssertionError("WT mismatch was not rejected")
