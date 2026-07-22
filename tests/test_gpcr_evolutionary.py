from __future__ import annotations

import json
import math
from pathlib import Path

import pandas as pd
import pytest

from protein_stabilizer.cli import build_parser
from protein_stabilizer.data import Mutation
from protein_stabilizer.gpcr_evolutionary import (
    GPCRDBEvolutionaryCache,
    GPCR_SCREENING_WEIGHTS,
    rank_gpcr_screening_consensus,
)


def _write_gpcrdb_cache(path: Path) -> Path:
    path.mkdir()
    metadata = {
        "accession": "TEST1",
        "entry_name": "test_human",
        "family": "001_001_001_001",
        "sequence": "ACDE",
    }
    alignment = {
        "test_human": "AC-DE",
        "test_mouse": "AA-DE",
        "test_rat": "AA-DE",
        "other_species": "AD-DE",
        "CONSENSUS": "AA-DE",
    }
    (path / "metadata.json").write_text(
        json.dumps(metadata), encoding="utf-8"
    )
    (path / "alignment.json").write_text(
        json.dumps(alignment), encoding="utf-8"
    )
    return path


def test_gpcrdb_evolutionary_log_odds_exclude_target(tmp_path: Path) -> None:
    cache = GPCRDBEvolutionaryCache(
        _write_gpcrdb_cache(tmp_path / "gpcrdb")
    )
    result = cache.score_mutations(
        "TEST1",
        "ACDE",
        [Mutation.parse("C2A"), Mutation.parse("E4A")],
    )

    assert result.log_odds[0] == pytest.approx(math.log(5.0))
    assert result.log_odds[1] == pytest.approx(math.log(0.5 / 3.5))
    assert result.observed.tolist() == [True, True]
    assert result.provenance["target_excluded"] is True
    assert result.provenance["alignment_sequences"] == 3
    assert len(str(result.provenance["alignment_sha256"])) == 64

    with pytest.raises(ValueError, match="does not match GPCRdb metadata"):
        cache.score_mutations("TEST1", "ACDA", [Mutation.parse("C2A")])
    with pytest.raises(ValueError, match="sequence scope"):
        cache.score_mutations(
            "TEST1",
            "ACDE",
            [Mutation.parse("C2A")],
            sequence_scope="species",
        )


def test_gpcr_consensus_merges_exact_mutation_sets(tmp_path: Path) -> None:
    cache = _write_gpcrdb_cache(tmp_path / "gpcrdb")
    retained_path = tmp_path / "retained.csv"
    generic_path = tmp_path / "generic.csv"
    output_path = tmp_path / "consensus.csv"
    pd.DataFrame(
        {
            "mutation": ["A1C", "C2A", "D3A"],
            "consensus_rank_score": [0.2, 0.8, 0.5],
            "rank": [3, 1, 2],
        }
    ).to_csv(retained_path, index=False)
    pd.DataFrame(
        {
            "mutation": ["D3A", "A1C", "C2A"],
            "selection_rank_score": [2.0, 3.0, 1.0],
            "ddg": [-2.0, -3.0, -1.0],
        }
    ).to_csv(generic_path, index=False)

    report = rank_gpcr_screening_consensus(
        "ACDE",
        "TEST1",
        retained_path,
        generic_path,
        output_path,
        cache,
        top=2,
    )
    output = pd.read_csv(output_path)

    assert report["ranking_weights"] == GPCR_SCREENING_WEIGHTS
    assert report["candidate_count"] == 3
    assert report["coverage"] == {
        "evolutionary_observed": 3,
        "evolutionary_neutral": 0,
    }
    assert output.iloc[0]["mutation"] == "C2A"
    assert output["gpcr_screening_rank"].tolist() == [1, 2, 3]
    assert output.set_index("mutation").loc[
        "A1C", "generic_6b_selection_rank_score"
    ] == pytest.approx(3.0)
    assert "not ddG" in report["interpretation"]

    mismatched = pd.read_csv(generic_path).iloc[:2]
    mismatched.to_csv(generic_path, index=False)
    with pytest.raises(ValueError, match="exactly the same mutations"):
        rank_gpcr_screening_consensus(
            "ACDE",
            "TEST1",
            retained_path,
            generic_path,
            output_path,
            cache,
        )


def test_gpcr_consensus_cli_contract() -> None:
    args = build_parser().parse_args(
        [
            "rank-gpcr-consensus",
            "--sequence",
            "ACDE",
            "--accession",
            "TEST1",
            "--retained-input",
            "retained.csv",
            "--generic-6b-input",
            "generic.csv",
        ]
    )

    assert args.command == "rank-gpcr-consensus"
    assert args.output.name == "gpcr_consensus.csv"
    assert args.gpcrdb_cache.name == "gpcrdb_evolutionary"
