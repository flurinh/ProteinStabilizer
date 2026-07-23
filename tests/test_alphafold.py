from __future__ import annotations

import io
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from protein_stabilizer.alphafold import fetch_alphafold_structure
from protein_stabilizer.cli import build_parser
from protein_stabilizer.data import Mutation
from protein_stabilizer import v2_predictor


PDB_BYTES = b"""HEADER    TEST ALPHAFOLD MODEL
ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00 95.00           C
ATOM      2  CA  CYS A   2       1.000   0.000   0.000  1.00 65.00           C
ATOM      3  CA  ASP A   3       2.000   0.000   0.000  1.00 80.00           C
TER
END
"""


class _Response(io.BytesIO):
    pass


class _Opener:
    def __init__(self, responses: dict[str, bytes]) -> None:
        self.responses = responses
        self.calls: list[str] = []

    def __call__(self, request: object, *, timeout: float) -> _Response:
        del timeout
        url = str(getattr(request, "full_url"))
        self.calls.append(url)
        if url not in self.responses:
            raise AssertionError(f"unexpected request: {url}")
        return _Response(self.responses[url])


def _api_payload(sequence: str = "ACD") -> bytes:
    return json.dumps(
        [
            {
                "entryId": "AF-Q9TEST-F1",
                "uniprotAccession": "Q9TEST",
                "latestVersion": 6,
                "sequence": sequence,
                "sequenceChecksum": "checksum",
                "modelCreatedDate": "2025-08-01T00:00:00Z",
                "globalMetricValue": 80.0,
                "pdbUrl": (
                    "https://alphafold.ebi.ac.uk/files/"
                    "AF-Q9TEST-F1-model_v6.pdb"
                ),
            }
        ]
    ).encode("utf-8")


def test_alphafold_fetch_validates_sequence_and_masks_plddt(
    tmp_path: Path,
) -> None:
    api_url = "https://alphafold.ebi.ac.uk/api/prediction/Q9TEST"
    pdb_url = (
        "https://alphafold.ebi.ac.uk/files/"
        "AF-Q9TEST-F1-model_v6.pdb"
    )
    opener = _Opener({api_url: _api_payload(), pdb_url: PDB_BYTES})
    structure = fetch_alphafold_structure(
        "q9test",
        "ACD",
        tmp_path,
        min_plddt=70.0,
        opener=opener,
    )
    assert structure.pdb_path.name == "AF-Q9TEST-F1-model_v6.pdb"
    assert structure.residue_mask.tolist() == [True, False, True]
    assert structure.provenance["latest_version"] == 6
    assert structure.provenance["cache_status"] == "downloaded"
    assert structure.provenance["confidence"]["masked_residue_count"] == 1
    assert (tmp_path / "Q9TEST" / "provenance.json").is_file()

    cached_opener = _Opener({api_url: _api_payload()})
    cached = fetch_alphafold_structure(
        "Q9TEST",
        "ACD",
        tmp_path,
        min_plddt=90.0,
        opener=cached_opener,
    )
    assert cached.residue_mask.tolist() == [True, False, False]
    assert cached.provenance["cache_status"] == "validated-cache"
    assert cached_opener.calls == [api_url]

    def unavailable(request: object, *, timeout: float) -> _Response:
        del request, timeout
        raise OSError("offline")

    offline = fetch_alphafold_structure(
        "Q9TEST",
        "ACD",
        tmp_path,
        min_plddt=70.0,
        opener=unavailable,
    )
    assert offline.provenance["cache_status"] == "offline-cache"
    assert offline.residue_mask.tolist() == [True, False, True]


def test_alphafold_fetch_rejects_nonmatching_sequence(tmp_path: Path) -> None:
    api_url = "https://alphafold.ebi.ac.uk/api/prediction/Q9TEST"
    opener = _Opener({api_url: _api_payload("ACE")})
    with pytest.raises(ValueError, match="no exact sequence match"):
        fetch_alphafold_structure(
            "Q9TEST",
            "ACD",
            tmp_path,
            opener=opener,
        )


def test_structure_annotation_marks_low_confidence_sites_missing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pdb_path = tmp_path / "model.pdb"
    pdb_path.write_bytes(PDB_BYTES)
    seen_mask: list[bool] = []

    class FakeEmbedder:
        provenance = SimpleNamespace(
            canonical_json=lambda: json.dumps({"model": "fake"})
        )

        def __init__(self, repository: Path, *, device: str) -> None:
            del repository, device

        def encode(
            self,
            path: Path,
            sequence: str,
            *,
            residue_mask: np.ndarray,
        ) -> np.ndarray:
            del path
            assert sequence == "ACD"
            seen_mask.extend(residue_mask.tolist())
            return np.ones((3, 128), dtype=np.float32)

    monkeypatch.setattr(
        v2_predictor,
        "ProteinMPNNBackboneEmbedder",
        FakeEmbedder,
    )
    structure, mask, _, provenance = v2_predictor._annotation_rows(
        "ACD",
        [Mutation.parse("A1C"), Mutation.parse("C2A")],
        topology=None,
        generic_numbering=None,
        pdb_path=pdb_path,
        proteinmpnn_repository=tmp_path,
        structure_residue_mask=np.asarray([True, False, True]),
        structure_source_provenance={"source": "test"},
        device="cpu",
    )
    assert seen_mask == [True, False, True]
    assert mask.tolist() == [True, False]
    assert np.all(structure[0] == 1.0)
    assert np.all(structure[1] == 0.0)
    assert provenance is not None
    assert provenance["masked_residue_count"] == 1
    assert provenance["source"] == {"source": "test"}


def test_structure_cli_uses_mutually_exclusive_sources() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "screen-v2-6b",
            "--sequence",
            "ACD",
            "--uniprot",
            "Q9TEST",
        ]
    )
    assert args.uniprot == "Q9TEST"
    assert args.pdb is None
    assert args.alphafold_min_plddt == pytest.approx(70.0)
    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "screen-v2-6b",
                "--sequence",
                "ACD",
                "--uniprot",
                "Q9TEST",
                "--pdb",
                "model.pdb",
            ]
        )
