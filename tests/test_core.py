from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from protein_stabilizer.data import (
    AMINO_ACIDS,
    EmbeddingRequest,
    Mutation,
    apply_mutations,
    gpcr_site_splits,
    merge_embedding_requests,
    reconstruct_double,
    sequence_hash,
    transfer_rows,
)
from protein_stabilizer.cli import _parse_positions, build_parser
from protein_stabilizer.embeddings import (
    ESMCEmbedder,
    ESMCProvenance,
    ResidueEmbeddingReader,
    ResidueEmbeddingWriter,
)
from protein_stabilizer.models import (
    EpistasisConfig,
    MultiMutationHead,
    SingleHeadConfig,
    SingleMutationEnsemble,
    SingleMutationHead,
)
from protein_stabilizer.training import (
    gpcr_dtm_adapter_features,
    membrane_adapter_features,
)
from protein_stabilizer.predictor import (
    _masked_marginal_mutation_scores,
    _percentile_ranks,
    _thermostability_consensus,
    dual_backbone_thermostability_consensus,
)
from protein_stabilizer.transfer_data import _pdb_chain_sequence, mutation_window


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


def test_single_mutation_ensemble_averages_predictions_and_latents() -> None:
    config = SingleHeadConfig(
        embedding_dim=8, hidden_dim=12, latent_dim=6, dropout=0.0
    )
    torch.manual_seed(17)
    members = [SingleMutationHead(config), SingleMutationHead(config)]
    ensemble = SingleMutationEnsemble(members).eval()
    delta = torch.randn(5, 8)
    expected_prediction = torch.stack([member(delta) for member in members]).mean(0)
    expected_latent = members[0].latent(delta)
    assert ensemble.member_predictions(delta).shape == (2, 5)
    torch.testing.assert_close(ensemble(delta), expected_prediction)
    torch.testing.assert_close(ensemble.latent(delta), expected_latent)


def test_masked_marginal_scores_follow_canonical_amino_acid_order() -> None:
    log_probabilities = np.zeros((2, 20), dtype=np.float32)
    log_probabilities[0, AMINO_ACIDS.index("C")] = 1.25
    log_probabilities[0, AMINO_ACIDS.index("A")] = -0.75
    log_probabilities[1, AMINO_ACIDS.index("W")] = 0.5
    log_probabilities[1, AMINO_ACIDS.index("Y")] = 0.2
    scores = _masked_marginal_mutation_scores(
        log_probabilities,
        [Mutation.parse("A1C"), Mutation.parse("Y2W")],
    )
    np.testing.assert_allclose(scores, [2.0, 0.3])


def test_esmc_masked_marginals_batch_sites_at_residue_token_offsets() -> None:
    class FakeTokenizer:
        mask_token_id = 3

        def __init__(self) -> None:
            self.amino_acid_ids = {
                amino_acid: index + 4
                for index, amino_acid in enumerate(AMINO_ACIDS)
            }

        def convert_tokens_to_ids(self, token: str) -> int:
            return self.amino_acid_ids[token]

        def __call__(
            self,
            sequences: list[str],
            *,
            add_special_tokens: bool,
            padding: bool,
            truncation: bool,
            return_tensors: str,
        ) -> dict[str, torch.Tensor]:
            assert add_special_tokens and padding and not truncation
            assert return_tensors == "pt"
            rows = [
                [1, *(self.amino_acid_ids[residue] for residue in sequence), 2]
                for sequence in sequences
            ]
            return {
                "input_ids": torch.tensor(rows),
                "attention_mask": torch.ones((len(rows), len(rows[0])), dtype=torch.long),
            }

    class FakeModel:
        def __init__(self) -> None:
            self.tokenizer = FakeTokenizer()
            self.calls: list[torch.Tensor] = []

        def __call__(self, *, sequence_tokens: torch.Tensor) -> SimpleNamespace:
            self.calls.append(sequence_tokens.clone())
            batch, length = sequence_tokens.shape
            logits = torch.zeros((batch, length, 24), dtype=torch.float32)
            for row in range(batch):
                masked_position = int(
                    torch.nonzero(
                        sequence_tokens[row] == self.tokenizer.mask_token_id,
                        as_tuple=False,
                    ).item()
                )
                values = (
                    torch.arange(len(AMINO_ACIDS), dtype=torch.float32)
                    * masked_position
                    / 10
                )
                ids = list(self.tokenizer.amino_acid_ids.values())
                logits[row, masked_position, ids] = values
            return SimpleNamespace(sequence_logits=logits)

    embedder = object.__new__(ESMCEmbedder)
    embedder.device = torch.device("cpu")
    embedder.model = FakeModel()
    scores = embedder.masked_marginal_log_probabilities(
        "ACDE",
        [1, 3, 4],
        max_tokens=12,
        max_batch_size=8,
    )
    assert scores.shape == (3, len(AMINO_ACIDS))
    assert len(embedder.model.calls) == 2
    assert embedder.model.calls[0][0, 1].item() == 3
    assert embedder.model.calls[0][1, 3].item() == 3
    assert embedder.model.calls[1][0, 4].item() == 3
    for row, position in enumerate((1, 3, 4)):
        logits = torch.zeros(24, dtype=torch.float32)
        amino_acid_ids = list(
            embedder.model.tokenizer.amino_acid_ids.values()
        )
        logits[amino_acid_ids] = (
            torch.arange(len(AMINO_ACIDS), dtype=torch.float32) * position / 10
        )
        expected = torch.log_softmax(logits, dim=0)[amino_acid_ids]
        np.testing.assert_allclose(scores[row], expected.numpy(), rtol=1e-6)


def test_percentile_ranks_average_ties_without_order_bias() -> None:
    values = np.asarray([2.0, 1.0, 2.0, 3.0], dtype=np.float32)
    np.testing.assert_allclose(
        _percentile_ranks(values),
        [0.5, 0.0, 0.5, 1.0],
    )


def test_thermostability_consensus_is_eighty_twenty_rank_blend() -> None:
    mptherm = np.asarray([0.0, 2.0, 1.0], dtype=np.float32)
    masked = np.asarray([3.0, 1.0, 2.0], dtype=np.float32)
    expected = 0.8 * _percentile_ranks(mptherm) + 0.2 * _percentile_ranks(
        masked
    )
    np.testing.assert_allclose(
        _thermostability_consensus(mptherm, masked), expected
    )


def test_dual_backbone_consensus_uses_selected_gpcr_rank_blend() -> None:
    mptherm_600m = np.asarray([0.0, 2.0, 1.0], dtype=np.float32)
    masked_600m = np.asarray([3.0, 1.0, 2.0], dtype=np.float32)
    mptherm_6b = np.asarray([1.0, 0.0, 3.0], dtype=np.float32)
    expected = (
        0.60 * _percentile_ranks(mptherm_600m)
        + 0.15 * _percentile_ranks(masked_600m)
        + 0.25 * _percentile_ranks(mptherm_6b)
    )
    np.testing.assert_allclose(
        dual_backbone_thermostability_consensus(
            mptherm_600m,
            masked_600m,
            mptherm_6b,
        ),
        expected,
    )
    with pytest.raises(ValueError, match="equal shape"):
        dual_backbone_thermostability_consensus(
            mptherm_600m,
            masked_600m[:2],
            mptherm_6b,
        )


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


def test_esmc6b_multiple_mutation_cli_contract() -> None:
    args = build_parser().parse_args(
        [
            "predict-6b",
            "--sequence",
            "ACDE",
            "--mutations",
            "A1C,E4W",
        ]
    )
    assert args.command == "predict-6b"
    assert args.model == "biohub/ESMC-6B"
    assert args.mutations == "A1C,E4W"


def test_long_transfer_sequence_is_cropped_and_remapped() -> None:
    sequence = "A" * 1200 + "C" + "D" * 1200
    window, mutation, start = mutation_window(
        sequence, Mutation.parse("C1201W"), max_length=100
    )
    assert len(window) == 100
    assert start == 1151
    assert mutation == Mutation.parse("C51W")
    assert apply_mutations(window, [mutation])[50] == "W"


def test_transfer_reader_preserves_source_mutation_numbering(tmp_path: Path) -> None:
    frame = pd.DataFrame(
        [
            {
                "protein_id": "P1",
                "mutation": "C1201W",
                "position": 1201,
                "embedding_mutation": "C51W",
                "embedding_position": 51,
                "wt_sequence": "A" * 50 + "C" + "D" * 49,
                "mutant_sequence": "A" * 50 + "W" + "D" * 49,
                "target": -1.5,
                "target_kind": "ddg",
                "split": "train",
            }
        ]
    )
    path = tmp_path / "transfer.csv"
    frame.to_csv(path, index=False)
    row = next(transfer_rows(path))
    assert row["mutation"] == "C1201W"
    assert row["source_position"] == 1201
    assert row["embedding_mutation"] == "C51W"
    assert row["position"] == 51


def test_pdb_numbering_is_remapped_to_embedding_sequence() -> None:
    def atom(serial: int, residue: str, number: int) -> str:
        return (
            f"ATOM  {serial:5d}  CA  {residue:>3s} A{number:4d}    "
            "   0.000   0.000   0.000  1.00  0.00           C  "
        )

    payload = "\n".join(
        [atom(1, "ALA", 5), atom(2, "CYS", 6), atom(3, "ASP", 7)]
    ).encode("ascii")
    sequence, remapped = _pdb_chain_sequence(payload, "A", Mutation.parse("C6W"))
    assert sequence == "ACD"
    assert remapped == Mutation.parse("C2W")


def test_membrane_adapter_feature_schemas() -> None:
    delta = np.arange(12, dtype=np.float32).reshape(3, 4)
    latent = np.arange(6, dtype=np.float32).reshape(3, 2)
    ddg = np.asarray([1.0, 2.0, 3.0], dtype=np.float32)
    assert membrane_adapter_features("raw_delta", delta, latent, ddg).shape == (3, 4)
    assert membrane_adapter_features("base_latent", delta, latent, ddg).shape == (3, 2)
    combined = membrane_adapter_features("base_latent_ddg", delta, latent, ddg)
    assert combined.shape == (3, 3)
    np.testing.assert_array_equal(combined[:, -1], ddg)


def test_gpcr_dtm_adapter_feature_schemas() -> None:
    delta = np.arange(12, dtype=np.float32).reshape(3, 4)
    latent = np.arange(6, dtype=np.float32).reshape(3, 2)
    ddg = np.asarray([1.0, 2.0, 3.0], dtype=np.float32)
    dtm = np.asarray([4.0, 5.0, 6.0], dtype=np.float32)
    score = gpcr_dtm_adapter_features("mptherm_dtm", delta, latent, ddg, dtm)
    assert score.shape == (3, 1)
    np.testing.assert_array_equal(score[:, 0], dtm)
    combined = gpcr_dtm_adapter_features(
        "base_latent_mptherm", delta, latent, ddg, dtm
    )
    assert combined.shape == (3, 3)
    np.testing.assert_array_equal(combined[:, -1], dtm)
