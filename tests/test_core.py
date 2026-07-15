from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from protein_stabilizer.data import (
    EmbeddingRequest,
    Mutation,
    apply_mutations,
    gpcr_site_splits,
    merge_embedding_requests,
    reconstruct_double,
    sequence_hash,
)
from protein_stabilizer.cli import _parse_positions
from protein_stabilizer.embeddings import (
    ESMCProvenance,
    ResidueEmbeddingReader,
    ResidueEmbeddingWriter,
)
from protein_stabilizer.models import (
    EpistasisConfig,
    MultiMutationHead,
    SingleHeadConfig,
    SingleMutationHead,
)


def test_mutation_application_and_double_reconstruction() -> None:
    wt = "ACDEFGHIK"
    assert apply_mutations(wt, [Mutation.parse("C2W")]) == "AWDEFGHIK"
    joint = apply_mutations(wt, [Mutation.parse("C2W"), Mutation.parse("H7A")])
    recovered, single1, single2 = reconstruct_double(joint, "C2W", "H7A")
    assert recovered == wt
    assert single1 == "AWDEFGHIK"
    assert single2 == "ACDEFGAIK"
    with pytest.raises(ValueError, match="expects"):
        apply_mutations(wt, [Mutation.parse("W2A")])


def test_embedding_requests_are_deduplicated_by_sequence_and_site() -> None:
    requests = merge_embedding_requests(
        [
            EmbeddingRequest("ACDE", (1, 2)),
            EmbeddingRequest("ACDE", (2, 4)),
            EmbeddingRequest("FGHI", (3,)),
        ]
    )
    by_sequence = {request.sequence: request.positions for request in requests}
    assert by_sequence == {"ACDE": (1, 2, 4), "FGHI": (3,)}


def test_residue_cache_round_trip_and_resume(tmp_path: Path) -> None:
    provenance = ESMCProvenance(
        model_name="fake",
        embedding_dimension=4,
        package_version="test",
        checkpoint_path="/fake/model",
        checkpoint_sha256="0" * 64,
    )
    cache_path = tmp_path / "cache.h5"
    requests = [EmbeddingRequest("ACDE", (1, 4)), EmbeddingRequest("FGHI", (2,))]
    matrices = [
        np.asarray([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=np.float16),
        np.asarray([[9, 10, 11, 12]], dtype=np.float16),
    ]
    with ResidueEmbeddingWriter(cache_path, provenance) as writer:
        writer.append(requests, matrices)
        assert writer.missing_requests(requests) == []
    with ResidueEmbeddingReader(cache_path) as reader:
        values = reader.vectors(
            [
                (sequence_hash("FGHI"), 2),
                (sequence_hash("ACDE"), 4),
                (sequence_hash("ACDE"), 1),
                (sequence_hash("ACDE"), 4),
            ]
        )
    np.testing.assert_array_equal(values, np.asarray([matrices[1][0], matrices[0][1], matrices[0][0], matrices[0][1]]))


def test_multi_mutation_head_is_permutation_invariant() -> None:
    torch.manual_seed(7)
    single = SingleMutationHead(
        SingleHeadConfig(embedding_dim=8, hidden_dim=12, latent_dim=6, dropout=0.0)
    )
    model = MultiMutationHead(
        single,
        EpistasisConfig(
            embedding_dim=8,
            element_hidden_dim=12,
            element_dim=6,
            set_hidden_dim=10,
            dropout=0.0,
        ),
    ).eval()
    single_delta = torch.randn(4, 3, 8)
    joint_delta = torch.randn(4, 3, 8)
    permutation = torch.tensor([2, 0, 1])
    prediction, additive, epistasis = model(single_delta, joint_delta)
    permuted = model(single_delta[:, permutation], joint_delta[:, permutation])
    torch.testing.assert_close(prediction, permuted[0])
    torch.testing.assert_close(additive, permuted[1])
    torch.testing.assert_close(epistasis, permuted[2])


def test_gpcr_split_holds_out_complete_sites(tmp_path: Path) -> None:
    sequence = "ACDEFGHIKLMNPQRSTVWY"
    rows: list[dict[str, object]] = []
    for protein, assay in (("P1", "a"), ("P2", "b")):
        for position in range(1, 9):
            wt = sequence[position - 1]
            mutant = "A" if wt != "A" else "C"
            for replicate_assay in (assay, assay + "2"):
                rows.append(
                    {
                        "protein_id": protein,
                        "assay_id": replicate_assay,
                        "mutation": f"{wt}{position}{mutant}",
                        "sequence": sequence,
                        "is_wt": 0,
                        "stability_delta_percent": float(position),
                    }
                )
    path = tmp_path / "gpcr.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    split = gpcr_site_splits(path, seed=11)
    assert set(split["split"]) == {"train", "val", "test"}
    assert split.groupby("site_id")["split"].nunique().max() == 1
    assert split.groupby(["protein_id", "split"]).size().gt(0).all()


def test_position_expression_parser() -> None:
    assert _parse_positions("1-3,8,10-11") == [1, 2, 3, 8, 10, 11]
    assert _parse_positions(None) is None
    with pytest.raises(ValueError, match="range"):
        _parse_positions("5-2")
