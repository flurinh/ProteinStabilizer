from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace

import h5py
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
from protein_stabilizer.cli import (
    _parse_generic_numbering,
    _parse_positions,
    _protected_mask,
    build_parser,
)
from protein_stabilizer.embeddings import (
    ESMCEmbedder,
    ESMCProvenance,
    HierarchicalEmbedding,
    HierarchyEmbeddingReader,
    HierarchyEmbeddingWriter,
    ResidueEmbeddingReader,
    ResidueEmbeddingWriter,
    file_sha256,
)
from protein_stabilizer.esmc6b import ESMC6BEmbedder
from protein_stabilizer.gpcr_ranking import (
    _sample_higher_is_better_pairs,
)
from protein_stabilizer.structure import _aligned_structure_indices
from protein_stabilizer.models import (
    DirectionalAssayConfig,
    DirectionalAssayHead,
    DirectionalSingleMutationEnsemble,
    DirectionalSingleMutationHead,
    EpistasisConfig,
    HierarchicalDirectionalMutationHead,
    HierarchicalEpistasisConfig,
    HierarchicalHeadConfig,
    HierarchicalMultiMutationHead,
    MultiMutationHead,
    SingleHeadConfig,
    SingleMutationEnsemble,
    SingleMutationHead,
    StatePotentialConfig,
    StatePotentialMutationHead,
)
from protein_stabilizer.training import (
    balanced_stability_weights,
    evaluate_directional_ablation,
    gpcr_dtm_adapter_features,
    load_directional_single_ensemble_checkpoint,
    membrane_adapter_features,
    stabilizer_retrieval_metrics,
    train_directional_single_head,
)
from protein_stabilizer.predictor import (
    _masked_marginal_mutation_scores,
    _percentile_ranks,
    _thermostability_consensus,
    dual_backbone_thermostability_consensus,
)
from protein_stabilizer.transfer_data import _pdb_chain_sequence, mutation_window
from protein_stabilizer.v2_features import (
    MEMBRANE_FEATURE_NAMES,
    membrane_topology_features,
)
from protein_stabilizer.v2_multi import MULTI_CHECKPOINT_SCHEMA
from protein_stabilizer.v2_predictor import (
    predict_hierarchical_mutations,
    screen_hierarchical_double_mutants,
    screen_hierarchical_single_mutants,
)
from protein_stabilizer.v2_training import (
    HIERARCHY_CHECKPOINT_SCHEMA,
    HierarchicalObjectiveConfig,
    HierarchyArrays,
    _train_candidate,
)
from protein_stabilizer.v2_transfer import _derived_validation_split


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


def test_hierarchy_cache_round_trip_preserves_ordered_windows(tmp_path: Path) -> None:
    provenance = ESMCProvenance(
        model_name="fake",
        embedding_dimension=3,
        package_version="test",
        checkpoint_path="/fake/model",
        checkpoint_sha256="0" * 64,
    )
    request = EmbeddingRequest("ACDE", (1, 3))
    hierarchy = HierarchicalEmbedding(
        global_mean=np.asarray([1, 2, 3], dtype=np.float16),
        windows=np.arange(2 * 5 * 3, dtype=np.float16).reshape(2, 5, 3),
        window_mask=np.asarray(
            [[False, False, True, True, True], [True, True, True, True, False]]
        ),
    )
    path = tmp_path / "hierarchy.h5"
    with HierarchyEmbeddingWriter(path, provenance, window_radius=2) as writer:
        writer.append([request], [hierarchy])
        assert writer.missing_requests([request]) == []
    with HierarchyEmbeddingReader(path) as reader:
        values = reader.features(
            [
                (request.sequence_hash, 3),
                (request.sequence_hash, 1),
                (request.sequence_hash, 3),
            ]
        )
    np.testing.assert_array_equal(
        values["window"], hierarchy.windows[[1, 0, 1]]
    )
    np.testing.assert_array_equal(
        values["window_mask"], hierarchy.window_mask[[1, 0, 1]]
    )
    np.testing.assert_array_equal(
        values["global_mean"],
        np.repeat(hierarchy.global_mean[None], 3, axis=0),
    )
    np.testing.assert_array_equal(values["sequence_length"], [4, 4, 4])


def test_hierarchy_cache_preserves_float32_storage(tmp_path: Path) -> None:
    provenance = ESMCProvenance(
        model_name="fake-fp32",
        embedding_dimension=2,
        package_version="test",
        checkpoint_path="/fake/model",
        checkpoint_sha256="1" * 64,
        inference_dtype="float32",
        storage_dtype="float32",
    )
    request = EmbeddingRequest("ACDE", (2,))
    hierarchy = HierarchicalEmbedding(
        global_mean=np.asarray([1.00001, 2.00001], dtype=np.float32),
        windows=np.asarray(
            [[[3.00001, 4.00001], [5.00001, 6.00001], [7.00001, 8.00001]]],
            dtype=np.float32,
        ),
        window_mask=np.ones((1, 3), dtype=bool),
    )
    path = tmp_path / "hierarchy-fp32.h5"
    with HierarchyEmbeddingWriter(path, provenance, window_radius=1) as writer:
        writer.append([request], [hierarchy])
    with HierarchyEmbeddingReader(path) as reader:
        values = reader.features([(request.sequence_hash, 2)])
    assert values["window"].dtype == np.float32
    assert values["global_mean"].dtype == np.float32
    np.testing.assert_array_equal(values["window"][0], hierarchy.windows[0])
    np.testing.assert_array_equal(values["global_mean"][0], hierarchy.global_mean)


def test_hierarchy_dense_row_reader_preserves_order_and_duplicates(
    tmp_path: Path,
) -> None:
    path = tmp_path / "dense.h5"
    values = np.arange(20_000, dtype=np.int32).reshape(10_000, 2)
    indices = np.concatenate(
        [
            np.arange(0, 10_000, 2, dtype=np.int64),
            np.asarray([8, 2, 8], dtype=np.int64),
        ]
    )
    with h5py.File(path, "w") as handle:
        dataset = handle.create_dataset(
            "values", data=values, chunks=(16, 2)
        )
        observed = HierarchyEmbeddingReader._rows(dataset, indices)
    np.testing.assert_array_equal(observed, values[indices])


def test_membrane_topology_features_keep_site_annotations_explicit() -> None:
    gpcr = membrane_topology_features(
        "alpha_helical_gpcr",
        generic_numbering="6.38x38",
        is_gpcr=True,
    )
    assert len(gpcr) == len(MEMBRANE_FEATURE_NAMES) == 8
    np.testing.assert_array_equal(gpcr[:5], [1, 1, 0, 1, 1])
    assert gpcr[5] == pytest.approx(2 / 3)
    assert gpcr[6] < 0
    assert gpcr[7] == 1

    whole_protein_only = membrane_topology_features("Membrane")
    np.testing.assert_array_equal(
        whole_protein_only,
        [1, 0, 0, 0, 0, 0, 0, 1],
    )
    unknown = membrane_topology_features(None)
    np.testing.assert_array_equal(unknown, np.zeros(8))


def test_esmc_hierarchy_encodes_global_and_padded_local_context() -> None:
    class FakeTokenizer:
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
            maximum = max(len(sequence) for sequence in sequences) + 2
            rows = [
                [1, *range(2, len(sequence) + 2), 99]
                + [0] * (maximum - len(sequence) - 2)
                for sequence in sequences
            ]
            masks = [
                [1] * (len(sequence) + 2)
                + [0] * (maximum - len(sequence) - 2)
                for sequence in sequences
            ]
            return {
                "input_ids": torch.tensor(rows),
                "attention_mask": torch.tensor(masks),
            }

    class FakeModel:
        tokenizer = FakeTokenizer()

        def __call__(self, *, sequence_tokens: torch.Tensor) -> SimpleNamespace:
            values = torch.stack(
                [
                    sequence_tokens.float(),
                    sequence_tokens.float() + 10,
                    sequence_tokens.float() + 20,
                ],
                dim=-1,
            )
            return SimpleNamespace(embeddings=values)

    embedder = object.__new__(ESMCEmbedder)
    embedder.device = torch.device("cpu")
    embedder.dimension = 3
    embedder.storage_dtype = np.dtype(np.float32)
    embedder.model = FakeModel()
    request = EmbeddingRequest("ACDE", (1, 3, 4))
    result = embedder.encode_hierarchy([request], window_radius=2)[0]
    np.testing.assert_allclose(result.global_mean, [3.5, 13.5, 23.5])
    assert result.windows.shape == (3, 5, 3)
    np.testing.assert_array_equal(
        result.window_mask,
        [
            [False, False, True, True, True],
            [True, True, True, True, False],
            [True, True, True, False, False],
        ],
    )
    np.testing.assert_allclose(result.windows[0, 2], [2, 12, 22])
    np.testing.assert_allclose(result.windows[2, 2], [5, 15, 25])


def test_esmc6b_hierarchy_uses_the_same_residue_offsets() -> None:
    class FakeModel:
        def __call__(self, **encoded: torch.Tensor) -> SimpleNamespace:
            tokens = encoded["input_ids"].float()
            values = torch.stack([tokens, tokens + 10], dim=-1)
            return SimpleNamespace(last_hidden_state=values)

    embedder = object.__new__(ESMC6BEmbedder)
    embedder.device = torch.device("cpu")
    embedder.dimension = 2
    embedder.storage_dtype = np.dtype(np.float32)
    embedder.strict_fp32 = True
    embedder.model = FakeModel()
    embedder._tokenize = lambda sequences: {
        "input_ids": torch.tensor(
            [[1, *range(2, len(sequence) + 2), 99] for sequence in sequences]
        )
    }
    result = embedder.encode_hierarchy(
        [EmbeddingRequest("ACDE", (1, 4))], window_radius=1
    )[0]
    np.testing.assert_allclose(result.global_mean, [3.5, 13.5])
    np.testing.assert_array_equal(
        result.window_mask,
        [[False, True, True], [True, True, False]],
    )
    np.testing.assert_allclose(result.windows[0, 1], [2, 12])
    np.testing.assert_allclose(result.windows[1, 1], [5, 15])


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


def test_directional_head_is_exactly_antisymmetric_and_zero_at_self() -> None:
    config = SingleHeadConfig(
        embedding_dim=8, hidden_dim=12, latent_dim=6, dropout=0.0
    )
    torch.manual_seed(23)
    members = [
        DirectionalSingleMutationHead(config),
        DirectionalSingleMutationHead(config),
    ]
    model = DirectionalSingleMutationEnsemble(members).eval()
    delta = torch.randn(7, 8)
    torch.testing.assert_close(model(delta), -model(-delta), atol=1e-7, rtol=1e-7)
    torch.testing.assert_close(
        model(torch.zeros_like(delta)), torch.zeros(7), atol=1e-7, rtol=0
    )
    assert model.stabilizer_logit(delta).shape == (7,)
    assert model.member_stabilizer_logits(delta).shape == (2, 7)


def test_hierarchical_head_swaps_complete_states_exactly() -> None:
    torch.manual_seed(29)
    config = HierarchicalHeadConfig(
        embedding_dim=8,
        window_size=5,
        structure_dim=4,
        membrane_dim=3,
        state_dim=10,
        context_dim=6,
        hidden_dim=14,
        latent_dim=7,
    )
    model = HierarchicalDirectionalMutationHead(config).eval()
    batch = 6
    wt_window = torch.randn(batch, 5, 8)
    mutant_window = torch.randn(batch, 5, 8)
    window_mask = torch.tensor(
        [[False, True, True, True, True], [True] * 5] * 3,
        dtype=torch.bool,
    )
    wt_global = torch.randn(batch, 8)
    mutant_global = torch.randn(batch, 8)
    structure = torch.randn(batch, 4)
    structure_mask = torch.tensor([0, 1, 1, 0, 1, 1], dtype=torch.bool)
    membrane = torch.randn(batch, 3)

    forward = model.predict_heads(
        wt_window,
        mutant_window,
        window_mask,
        wt_global,
        mutant_global,
        structure=structure,
        structure_mask=structure_mask,
        membrane=membrane,
    )
    reverse = model.predict_heads(
        mutant_window,
        wt_window,
        window_mask,
        mutant_global,
        wt_global,
        structure=structure,
        structure_mask=structure_mask,
        membrane=membrane,
    )
    for task in ("ddg", "dtm", "retrieval"):
        torch.testing.assert_close(
            forward[task], -reverse[task], atol=1e-7, rtol=1e-7
        )

    self_prediction = model(
        wt_window,
        wt_window,
        window_mask,
        wt_global,
        wt_global,
        structure=structure,
        structure_mask=structure_mask,
        membrane=membrane,
    )
    torch.testing.assert_close(
        self_prediction, torch.zeros(batch), atol=1e-7, rtol=0
    )


def test_hierarchical_multi_head_is_permutation_invariant() -> None:
    torch.manual_seed(37)
    model = HierarchicalMultiMutationHead(
        HierarchicalEpistasisConfig(
            latent_dim=7,
            element_hidden_dim=11,
            element_dim=5,
            set_hidden_dim=9,
        )
    ).eval()
    single_ddg = torch.randn(4, 3)
    single_latent = torch.randn(4, 3, 7)
    joint_latent = torch.randn(4, 3, 7)
    mask = torch.tensor(
        [
            [True, True, False],
            [True, True, True],
            [True, False, True],
            [False, True, True],
        ]
    )
    permutation = torch.tensor([2, 0, 1])
    forward = model(single_ddg, single_latent, joint_latent, mask)
    permuted = model(
        single_ddg[:, permutation],
        single_latent[:, permutation],
        joint_latent[:, permutation],
        mask[:, permutation],
    )
    for value, expected in zip(forward, permuted, strict=True):
        torch.testing.assert_close(value, expected)
    torch.testing.assert_close(
        forward[1], (single_ddg * mask).sum(dim=1)
    )
    torch.testing.assert_close(forward[0], forward[1] + forward[2])


def test_directional_assay_heads_keep_endpoint_signs_separate() -> None:
    torch.manual_seed(39)
    model = DirectionalAssayHead(
        DirectionalAssayConfig(
            latent_dim=7,
            membrane_dim=3,
            context_dim=5,
            hidden_dim=11,
            output_latent_dim=6,
        )
    ).eval()
    latent = torch.randn(8, 7)
    ddg = torch.randn(8)
    membrane = torch.randn(8, 3)
    forward = model(latent, ddg, membrane)
    reverse = model(-latent, -ddg, membrane)
    torch.testing.assert_close(forward, -reverse, atol=1e-7, rtol=1e-7)
    torch.testing.assert_close(
        model(torch.zeros_like(latent), torch.zeros_like(ddg), membrane),
        torch.zeros(8),
        atol=1e-7,
        rtol=0,
    )


def test_transfer_validation_derivation_holds_out_whole_proteins() -> None:
    split = np.asarray(
        ["train"] * 12 + ["test"] * 4 + ["quarantine"] * 2
    )
    protein = np.asarray(
        ["a"] * 4 + ["b"] * 4 + ["c"] * 4 + ["d"] * 4 + ["e"] * 2
    )
    derived = _derived_validation_split(split, protein, seed=43)
    train_proteins = set(protein[derived == "train"])
    validation_proteins = set(protein[derived == "val"])
    test_proteins = set(protein[derived == "test"])
    assert train_proteins
    assert validation_proteins
    assert not train_proteins & validation_proteins
    assert not train_proteins & test_proteins
    assert not validation_proteins & test_proteins
    assert np.all(derived[split == "quarantine"] == "quarantine")


def test_transfer_split_promotes_overlapping_groups_to_test() -> None:
    split = np.asarray(
        ["train", "train", "train", "train", "test", "reference"]
    )
    protein = np.asarray(["a", "a", "b", "c", "a", "d"])
    derived = _derived_validation_split(split, protein, seed=43)
    assert np.all(derived[protein == "a"] == "test")
    assert set(protein[derived == "train"]).isdisjoint(
        set(protein[derived == "val"])
    )
    assert set(protein[derived == "train"]).isdisjoint(
        set(protein[derived == "test"])
    )
    assert derived[-1] == "reference"


def test_hierarchical_training_records_steps_and_examples() -> None:
    rng = np.random.default_rng(41)

    def arrays(rows: int, split: list[str]) -> HierarchyArrays:
        wt_window = rng.normal(size=(rows, 5, 8)).astype(np.float16)
        mutant_window = wt_window.copy()
        mutation_signal = np.resize(
            np.asarray([-1.5, -0.8, -0.2, 0.3, 0.9, 1.5], dtype=np.float32),
            rows,
        )
        mutant_window[:, 2, 0] += mutation_signal
        wt_global = rng.normal(size=(rows, 8)).astype(np.float16)
        mutant_global = wt_global.copy()
        mutant_global[:, 0] += 0.25 * mutation_signal
        return HierarchyArrays(
            wt_window=wt_window,
            mutant_window=mutant_window,
            window_mask=np.ones((rows, 5), dtype=bool),
            wt_global=wt_global,
            mutant_global=mutant_global,
            structure=rng.normal(size=(rows, 4)).astype(np.float16),
            structure_mask=np.ones(rows, dtype=bool),
            membrane=np.zeros((rows, 3), dtype=np.float32),
            target=mutation_signal,
            sample_weight=np.ones(rows, dtype=np.float32),
            protein_id=np.asarray(
                [f"p{index // 4}" for index in range(rows)]
            ),
            split=np.asarray(split),
            provenance={},
        )

    train = arrays(24, ["train"] * 16 + ["val"] * 8)
    test = arrays(8, ["test"] * 8)
    metrics, members = _train_candidate(
        "synthetic",
        train,
        test,
        use_structure=True,
        seed=41,
        epochs=2,
        minimum_epochs=1,
        patience=1,
        batch_size=8,
        learning_rate=1e-3,
        weight_decay=1e-4,
        ensemble_size=1,
        device=torch.device("cpu"),
        objective=HierarchicalObjectiveConfig(
            ranking_pairs_per_batch=4,
            huber_delta=2.0,
            mse_weight=0.1,
            retrieval_source="ddg",
        ),
    )
    assert metrics["ensemble_size"] == 1
    assert metrics["directional_constraints"][
        "maximum_absolute_forward_plus_reverse"
    ] < 1e-6
    assert members[0]["optimizer_steps"] >= 2
    assert members[0]["examples_seen"] >= 16
    assert members[0]["history"][0]["train_loss"]["mse"] >= 0.0


def test_tail_balancing_and_retrieval_metrics_prioritize_stabilizers() -> None:
    target = np.asarray([-2.0, -1.0, -0.2, 0.2, 1.0, 2.0], dtype=np.float32)
    weights = balanced_stability_weights(target, maximum_weight=100.0)
    bins = (target < -0.5, (target >= -0.5) & (target <= 0.5), target > 0.5)
    totals = [float(weights[mask].sum()) for mask in bins]
    np.testing.assert_allclose(totals, [2.0, 2.0, 2.0], rtol=1e-6)
    metrics = stabilizer_retrieval_metrics(
        target,
        np.asarray([6.0, 5.0, 1.0, 0.0, -1.0, -2.0]),
        top_ks=(2,),
    )
    assert metrics["positives"] == 2
    assert metrics["average_precision"] == pytest.approx(1.0)
    assert metrics["auroc"] == pytest.approx(1.0)
    assert metrics["precision_at_2"] == pytest.approx(1.0)
    assert metrics["recall_at_2"] == pytest.approx(1.0)


def test_directional_training_checkpoint_keeps_provenance(tmp_path: Path) -> None:
    feature_dir = tmp_path / "features"
    checkpoint_dir = tmp_path / "checkpoints"
    baseline_dir = tmp_path / "baseline"
    feature_dir.mkdir()
    baseline_dir.mkdir()
    rng = np.random.default_rng(31)
    provenance = {
        "model_name": "fake-esmc",
        "checkpoint_sha256": "a" * 64,
        "embedding_dimension": 8,
    }

    def write_features(
        path: Path,
        rows: int,
        proteins: list[str],
        split: list[str],
        source_hash: str,
    ) -> None:
        target = np.tile(
            np.asarray([-1.5, -0.8, -0.1, 0.3, 1.0, 1.8], dtype=np.float32),
            math.ceil(rows / 6),
        )[:rows]
        with h5py.File(path, "w") as handle:
            handle.create_dataset("delta", data=rng.normal(size=(rows, 8)).astype("f2"))
            handle.create_dataset("target", data=target)
            handle.create_dataset(
                "protein_id",
                data=np.asarray(proteins, dtype=object),
                dtype=h5py.string_dtype("utf-8"),
            )
            handle.create_dataset(
                "split",
                data=np.asarray(split, dtype=object),
                dtype=h5py.string_dtype("utf-8"),
            )
            handle.attrs["embedding_provenance"] = json.dumps(provenance)
            handle.attrs["source_sha256"] = source_hash

    write_features(
        feature_dir / "single_train.h5",
        18,
        ["train-a"] * 6 + ["train-b"] * 6 + ["validation"] * 6,
        ["train"] * 12 + ["val"] * 6,
        "b" * 64,
    )
    write_features(
        feature_dir / "single_test.h5",
        6,
        ["test"] * 6,
        ["test"] * 6,
        "c" * 64,
    )
    metrics = train_directional_single_head(
        feature_dir,
        checkpoint_dir,
        seed=31,
        epochs=2,
        patience=2,
        batch_size=6,
        ensemble_size=1,
        device="cpu",
    )
    assert metrics["embedding_provenance"] == provenance
    assert metrics["train_source_sha256"] == "b" * 64
    assert metrics["test_source_sha256"] == "c" * 64
    assert metrics["train_proteins"] == 2
    assert metrics["validation_proteins"] == 1
    assert (
        metrics["directional_constraints"][
            "maximum_absolute_forward_plus_reverse"
        ]
        < 1e-6
    )
    loaded = load_directional_single_ensemble_checkpoint(
        checkpoint_dir / "directional_single_ensemble.pt"
    )
    delta = torch.randn(3, 8)
    torch.testing.assert_close(loaded(delta), -loaded(-delta), atol=1e-7, rtol=1e-7)
    baseline_config = SingleHeadConfig(
        embedding_dim=8, hidden_dim=12, latent_dim=6, dropout=0.0
    )
    baseline = SingleMutationHead(baseline_config)
    torch.save(
        {
            "schema": "protein-stabilizer.single-ensemble.v1",
            "members": [
                {
                    "config": baseline.config_dict(),
                    "state_dict": baseline.state_dict(),
                }
            ],
            "metrics": {},
        },
        baseline_dir / "single_ensemble.pt",
    )
    ablation = evaluate_directional_ablation(
        feature_dir,
        baseline_dir,
        checkpoint_dir,
        device="cpu",
        batch_size=6,
    )
    assert ablation["validation"]["rows"] == 6
    assert ablation["test"]["rows"] == 6
    assert len(ablation["candidate_checkpoint"]["sha256"]) == 64
    assert ablation["directional_constraints"]["maximum_absolute_self_ddg"] < 1e-6


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


def test_protected_mask_combines_inline_and_reasoned_file(
    tmp_path: Path,
) -> None:
    path = tmp_path / "protected.txt"
    path.write_text(
        "# one-based hard exclusions\n"
        "2-3\tDRY motif\n"
        "5 ligand contact # inline comment\n",
        encoding="utf-8",
    )
    mask = _protected_mask("ACDEFG", "1,3", path)
    assert mask == {
        1: "command-line protected mask",
        2: "DRY motif",
        3: "command-line protected mask; DRY motif",
        5: "ligand contact",
    }
    with pytest.raises(ValueError, match="outside"):
        _protected_mask("ACDEFG", "7", None)


def test_human_melanopsin_example_is_pinned_and_runnable() -> None:
    example = Path(__file__).resolve().parents[1] / "examples/human_melanopsin"
    fasta_path = example / "Q9UHM6.fasta"
    fasta_bytes = fasta_path.read_bytes()
    sequence = "".join(
        line.strip()
        for line in fasta_bytes.decode("ascii").splitlines()
        if line and not line.startswith(">")
    )
    provenance = json.loads(
        (example / "provenance.json").read_text(encoding="utf-8")
    )
    assert provenance["accession"] == "Q9UHM6"
    assert provenance["uniprot_id"] == "OPN4_HUMAN"
    assert len(sequence) == provenance["sequence_length"] == 478
    assert hashlib.sha256(fasta_bytes).hexdigest() == provenance["fasta_sha256"]
    assert (
        hashlib.sha256(sequence.encode("ascii")).hexdigest()
        == provenance["sequence_sha256"]
    )

    protected = _protected_mask(
        sequence,
        None,
        example / "protected_positions.txt",
    )
    assert len(protected) == 208
    assert len(sequence) - len(protected) == 270
    assert sequence[142] == "C" and sequence[220] == "C"
    assert sequence[166:169] == "DRY"
    assert sequence[339] == "K"
    assert sequence[345:350] == "NPIIY"
    assert all(position in protected for position in (143, 167, 340, 350, 478))
    assert (example / "run_screen.sh").stat().st_mode & 0o100


def test_v2_application_cli_contract() -> None:
    args = build_parser().parse_args(
        [
            "predict-v2",
            "--sequence",
            "ACDE",
            "--mutations",
            "A1C,E4W",
            "--topology",
            "alpha_helical_gpcr",
            "--generic-numbering",
            "1=1.50x50,4=2.50x50",
        ]
    )
    assert args.command == "predict-v2"
    assert args.checkpoints.name == "esmc_600m_v2_fp32"
    assert args.state_potential_checkpoint.name == (
        "state_potential_ensemble.pt"
    )
    assert _parse_generic_numbering(args.generic_numbering) == {
        1: "1.50x50",
        4: "2.50x50",
    }
    with pytest.raises(ValueError, match="duplicate"):
        _parse_generic_numbering("1=1.50x50,1=1.51x51")


def test_v2_application_predicts_unordered_sets_and_screens(
    tmp_path: Path,
) -> None:
    config = HierarchicalHeadConfig(
        embedding_dim=8,
        structure_dim=128,
        membrane_dim=8,
        state_dim=6,
        context_dim=4,
        hidden_dim=9,
        latent_dim=5,
    )
    base = HierarchicalDirectionalMutationHead(config).eval()
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    torch.save(
        {
            "schema": HIERARCHY_CHECKPOINT_SCHEMA,
            "candidate": "hierarchy_proteinmpnn",
            "use_structure": True,
            "members": [
                {
                    "config": base.config_dict(),
                    "state_dict": base.state_dict(),
                }
            ],
        },
        checkpoint_dir / "hierarchy_selected_ensemble.pt",
    )
    state = StatePotentialMutationHead(
        StatePotentialConfig(
            embedding_dim=8,
            window_size=9,
            structure_dim=128,
            membrane_dim=8,
            state_dim=7,
            amino_acid_dim=5,
            hidden_dim=11,
            latent_dim=6,
        )
    ).eval()
    state_path = checkpoint_dir / "state_potential_ensemble.pt"
    torch.save(
        {
            "schema": "protein-stabilizer.state-potential-ensemble.v1",
            "candidate": "wt_conditioned_state_potential",
            "members": [
                {
                    "config": state.config_dict(),
                    "state_dict": state.state_dict(),
                }
            ],
            "state_potential_weight": 0.55,
            "baseline_checkpoint_sha256": file_sha256(
                checkpoint_dir / "hierarchy_selected_ensemble.pt"
            ),
            "promotion": {"production_eligible": True},
        },
        state_path,
    )
    multi = HierarchicalMultiMutationHead(
        HierarchicalEpistasisConfig(
            latent_dim=5,
            element_hidden_dim=7,
            element_dim=4,
            set_hidden_dim=6,
        )
    ).eval()
    torch.save(
        {
            "schema": MULTI_CHECKPOINT_SCHEMA,
            "config": multi.config_dict(),
            "state_dict": multi.state_dict(),
        },
        checkpoint_dir / "hierarchy_multi_head.pt",
    )

    class FakeEmbedder:
        provenance = ESMCProvenance(
            model_name="fake-application",
            embedding_dimension=8,
            package_version="test",
            checkpoint_path="/fake/application",
            checkpoint_sha256="a" * 64,
        )

        def encode_hierarchy(
            self,
            requests: list[EmbeddingRequest],
            *,
            window_radius: int,
        ) -> list[HierarchicalEmbedding]:
            values: list[HierarchicalEmbedding] = []
            width = 2 * window_radius + 1
            for request in requests:
                sequence_value = float(
                    sum((index + 1) * ord(aa) for index, aa in enumerate(request.sequence))
                    % 31
                )
                windows = np.stack(
                    [
                        np.full(
                            (width, 8),
                            sequence_value + position / 10,
                            dtype=np.float16,
                        )
                        for position in request.positions
                    ]
                )
                values.append(
                    HierarchicalEmbedding(
                        global_mean=np.full(
                            8, sequence_value / 10, dtype=np.float16
                        ),
                        windows=windows,
                        window_mask=np.ones(
                            (len(request.positions), width), dtype=bool
                        ),
                    )
                )
            return values

        def masked_marginal_log_probabilities(
            self,
            sequence: str,
            positions: list[int] | tuple[int, ...],
            **_: object,
        ) -> np.ndarray:
            return np.stack(
                [
                    np.linspace(-1, 1, len(AMINO_ACIDS), dtype=np.float32)
                    + position / 100
                    for position in positions
                ]
            )

    forward = predict_hierarchical_mutations(
        "ACDE",
        ["A1W", "E4F"],
        checkpoint_dir,
        device="cpu",
        topology="alpha_helical_gpcr",
        generic_numbering={1: "1.50x50", 4: "2.50x50"},
        embedder=FakeEmbedder(),
    )
    reverse_order = predict_hierarchical_mutations(
        "ACDE",
        ["E4F", "A1W"],
        checkpoint_dir,
        device="cpu",
        embedder=FakeEmbedder(),
    )
    assert [row["mutation"] for row in forward["mutations"]] == [
        "A1W",
        "E4F",
    ]
    assert forward["total_ddg"] == pytest.approx(
        forward["additive_ddg"] + forward["epistasis_ddg"]
    )
    assert forward["total_ddg"] == pytest.approx(
        reverse_order["total_ddg"]
    )
    assert "permutation-invariant" in forward["mutation_set_policy"]

    fused = predict_hierarchical_mutations(
        "ACDE",
        ["A1W"],
        checkpoint_dir,
        device="cpu",
        embedder=FakeEmbedder(),
        state_potential_checkpoint=state_path,
    )
    fused_row = fused["mutations"][0]
    assert fused_row["ddg"] == pytest.approx(
        0.45 * fused_row["hierarchy_ddg"]
        + 0.55 * fused_row["state_potential_ddg"],
        abs=1e-6,
    )
    assert fused["model"]["state_potential_weight"] == pytest.approx(0.55)

    screen = screen_hierarchical_single_mutants(
        "ACDE",
        checkpoint_dir,
        tmp_path / "screen.csv",
        positions=[1],
        device="cpu",
        top=5,
        embedder=FakeEmbedder(),
    )
    assert screen["rows"] == 19
    assert len(screen["top"]) == 5
    assert (tmp_path / "screen.csv").is_file()
    assert [row["stabilizer_rank"] for row in screen["top"]] == [1, 2, 3, 4, 5]

    state_screen = screen_hierarchical_single_mutants(
        "ACDE",
        checkpoint_dir,
        tmp_path / "state_screen.csv",
        positions=[1],
        device="cpu",
        top=5,
        embedder=FakeEmbedder(),
        state_potential_checkpoint=state_path,
        scan_mode="state-only",
    )
    assert state_screen["scan_mode"] == "state-only"
    assert state_screen["model"]["state_potential_weight"] == pytest.approx(1.0)
    assert "state_potential_ddg" in state_screen["top"][0]
    assert "hierarchy_ddg" not in state_screen["top"][0]

    application_cache = tmp_path / "application_embeddings.h5"
    two_stage = screen_hierarchical_single_mutants(
        "ACDE",
        checkpoint_dir,
        tmp_path / "two_stage.csv",
        positions=[1, 2, 3],
        protected_positions=[2],
        protected_reasons={2: "ligand contact"},
        device="cpu",
        top=2,
        rerank_top=2,
        max_per_site=1,
        embedder=FakeEmbedder(),
        state_potential_checkpoint=state_path,
        scan_mode="two-stage",
        embedding_cache=application_cache,
    )
    assert two_stage["rows"] == 38
    assert two_stage["positions"] == 2
    assert two_stage["protected_mask"]["position_ranges"] == ["2"]
    assert two_stage["protected_mask"]["reasons"] == [
        {"reason": "ligand contact", "position_ranges": ["2"]}
    ]
    assert two_stage["embedding_cost"]["sequence_embedding_requests"] == 3
    assert two_stage["embedding_cost"]["sequence_embeddings_computed"] == 3
    assert two_stage["embedding_cost"]["mutant_embeddings_avoided"] == 36
    assert len({row["position"] for row in two_stage["top"]}) == 2
    assert all(row["score_stage"] == "exact-reranked" for row in two_stage["top"])
    for row in two_stage["top"]:
        assert row["ddg"] == pytest.approx(
            0.45 * row["hierarchy_ddg"]
            + 0.55 * row["state_potential_ddg"],
            abs=1e-6,
        )
    assert (tmp_path / "two_stage.shortlist.csv").is_file()
    assert 2 not in set(pd.read_csv(tmp_path / "two_stage.csv")["position"])

    cached_two_stage = screen_hierarchical_single_mutants(
        "ACDE",
        checkpoint_dir,
        tmp_path / "two_stage_cached.csv",
        positions=[1, 2, 3],
        protected_positions=[2],
        device="cpu",
        top=2,
        rerank_top=2,
        max_per_site=1,
        embedder=FakeEmbedder(),
        state_potential_checkpoint=state_path,
        scan_mode="two-stage",
        embedding_cache=application_cache,
    )
    assert cached_two_stage["embedding_cost"]["sequence_embeddings_computed"] == 0
    assert cached_two_stage["embedding_cost"]["cache_hits"] == 3

    pair_screen = screen_hierarchical_double_mutants(
        "ACDE",
        tmp_path / "two_stage.shortlist.csv",
        checkpoint_dir,
        tmp_path / "pairs.csv",
        device="cpu",
        top=1,
        single_limit=2,
        single_ddg_ceiling=None,
        require_component_agreement=False,
        pair_rerank_top=1,
        max_pairs_per_site=1,
        protected_positions=[2],
        protected_reasons={2: "ligand contact"},
        embedder=FakeEmbedder(),
        state_potential_checkpoint=state_path,
        embedding_cache=application_cache,
    )
    assert pair_screen["rows"] == 1
    assert pair_screen["exact_pairs"] == 1
    assert pair_screen["embedding_cost"]["sequence_embedding_requests"] == 4
    assert pair_screen["embedding_cost"]["sequence_embeddings_computed"] == 1
    assert pair_screen["embedding_cost"]["cache_hits"] == 3
    assert pair_screen["embedding_cost"]["joint_pair_embeddings_requested"] == 1
    assert pair_screen["protected_mask"]["position_ranges"] == ["2"]
    pair = pair_screen["top"][0]
    assert pair["score_stage"] == "exact-pair-reranked"
    assert pair["suggestion_eligible"] is True
    assert pair["position_1"] < pair["position_2"]
    assert pair["total_ddg"] == pytest.approx(
        pair["additive_ddg"] + pair["epistasis_ddg"]
    )
    assert (tmp_path / "pairs.csv").is_file()
    assert (tmp_path / "pairs.shortlist.csv").is_file()

    reversed_singles = tmp_path / "two_stage_reversed.csv"
    pd.read_csv(tmp_path / "two_stage.shortlist.csv").iloc[::-1].to_csv(
        reversed_singles, index=False
    )
    reversed_pair_screen = screen_hierarchical_double_mutants(
        "ACDE",
        reversed_singles,
        checkpoint_dir,
        tmp_path / "pairs_reversed.csv",
        device="cpu",
        top=1,
        single_limit=2,
        single_ddg_ceiling=None,
        require_component_agreement=False,
        pair_rerank_top=1,
        max_pairs_per_site=1,
        protected_positions=[2],
        embedder=FakeEmbedder(),
        state_potential_checkpoint=state_path,
        embedding_cache=application_cache,
    )
    assert reversed_pair_screen["top"][0]["mutation_set"] == pair["mutation_set"]
    assert reversed_pair_screen["top"][0]["total_ddg"] == pytest.approx(
        pair["total_ddg"]
    )
    assert (
        reversed_pair_screen["embedding_cost"]["sequence_embeddings_computed"]
        == 0
    )

    legacy_dir = tmp_path / "legacy"
    legacy_dir.mkdir()
    legacy = SingleMutationHead(
        SingleHeadConfig(
            embedding_dim=8,
            hidden_dim=7,
            latent_dim=4,
            dropout=0.0,
        )
    ).eval()
    torch.save(
        {
            "schema": "protein-stabilizer.auxiliary-head.v1",
            "config": legacy.config_dict(),
            "state_dict": legacy.state_dict(),
        },
        legacy_dir / "mptherm_dtm_head.pt",
    )
    gpcr_screen = screen_hierarchical_single_mutants(
        "ACDE",
        checkpoint_dir,
        tmp_path / "gpcr_screen.csv",
        positions=[1],
        device="cpu",
        top=3,
        legacy_checkpoint_dir=legacy_dir,
        embedder=FakeEmbedder(),
    )
    assert "0.80 retained MPTherm" in gpcr_screen["ranking_policy"]
    assert "retained_mptherm_delta_tm" in gpcr_screen["top"][0]
    assert "masked_marginal_log_odds" in gpcr_screen["top"][0]


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


def test_esmc6b_v2_cache_defaults_to_native_fp32() -> None:
    development_args = build_parser().parse_args(["embed-v2-context"])
    assert development_args.storage_dtype == "float32"
    args = build_parser().parse_args(["embed-v2-context-6b"])
    assert args.inference_dtype == "float32"
    assert args.storage_dtype == "float32"
    assert args.cache.name == "hierarchy_cache_fp32.h5"
    prediction_args = build_parser().parse_args(
        [
            "predict-v2-6b",
            "--sequence",
            "ACDE",
            "--mutations",
            "A1C",
        ]
    )
    assert prediction_args.checkpoints.name == "esmc_6b_v2"
    assert prediction_args.state_potential_checkpoint.parent.name == (
        "esmc_6b_state_potential_fp32"
    )
    assert prediction_args.max_batch_size == 2
    screen_args = build_parser().parse_args(
        ["screen-v2-6b", "--sequence", "ACDE"]
    )
    assert screen_args.checkpoints.name == "esmc_6b_v2"
    assert screen_args.output.name == "screen_v2_6b.csv"
    assert screen_args.scan_mode == "exact"
    assert screen_args.rerank_top == 128
    assert screen_args.embedding_cache.name == "esmc_6b_targets_fp32.h5"
    assert screen_args.state_potential_checkpoint.parent.name == (
        "esmc_6b_state_potential_fp32"
    )
    staged_args = build_parser().parse_args(
        [
            "screen-v2-6b",
            "--sequence",
            "ACDE",
            "--scan-mode",
            "two-stage",
            "--protected-positions",
            "2-3",
            "--max-per-site",
            "2",
        ]
    )
    assert staged_args.scan_mode == "two-stage"
    assert staged_args.protected_positions == "2-3"
    assert staged_args.max_per_site == 2
    pair_args = build_parser().parse_args(
        [
            "screen-v2-pairs-6b",
            "--sequence",
            "ACDE",
            "--single-screen",
            "singles.csv",
        ]
    )
    assert pair_args.checkpoints.name == "esmc_6b_v2"
    assert pair_args.output.name == "screen_v2_pairs_6b.csv"
    assert pair_args.single_limit == 20
    assert pair_args.single_ddg_ceiling == pytest.approx(0.0)
    assert pair_args.require_component_agreement is True
    assert pair_args.pair_rerank_top == 64
    assert pair_args.max_pairs_per_site == 4
    assert pair_args.embedding_cache.name == "esmc_6b_targets_fp32.h5"
    assert pair_args.max_batch_size == 2
    structure_args = build_parser().parse_args(["structure-v2"])
    assert structure_args.storage_dtype == "float32"
    potential_args = build_parser().parse_args(["train-state-potential"])
    assert potential_args.cache.name == "hierarchy_cache_fp32.h5"
    assert potential_args.features.name == "features_v2_600m_fp32"
    assert potential_args.minimum_epochs == 50
    gpcr_args = build_parser().parse_args(["train-zero-shot-gpcr"])
    assert gpcr_args.representations.name == (
        "features_v2_transfer_6b_state_potential_fp32"
    )
    assert gpcr_args.minimum_main_epochs == 50


def test_state_potential_scores_all_amino_acids_with_exact_algebra() -> None:
    config = StatePotentialConfig(
        embedding_dim=6,
        window_size=3,
        structure_dim=4,
        membrane_dim=2,
        state_dim=8,
        amino_acid_dim=5,
        hidden_dim=12,
        latent_dim=7,
    )
    model = StatePotentialMutationHead(config).eval()
    generator = torch.Generator().manual_seed(7)
    wt_window = torch.randn(4, 3, 6, generator=generator)
    window_mask = torch.ones(4, 3, dtype=torch.bool)
    wt_global = torch.randn(4, 6, generator=generator)
    structure = torch.randn(4, 4, generator=generator)
    structure_mask = torch.ones(4, dtype=torch.bool)
    membrane = torch.randn(4, 2, generator=generator)
    wt = torch.tensor([0, 1, 2, 3], dtype=torch.long)
    mutant = torch.tensor([4, 5, 6, 7], dtype=torch.long)
    inputs = {
        "wt_window": wt_window,
        "window_mask": window_mask,
        "wt_global": wt_global,
        "structure": structure,
        "structure_mask": structure_mask,
        "membrane": membrane,
    }
    with torch.inference_mode():
        potentials = model.all_potentials(**inputs)
        forward = model(
            **inputs,
            wt_amino_acid=wt,
            mutant_amino_acid=mutant,
        )
        reverse = model(
            **inputs,
            wt_amino_acid=mutant,
            mutant_amino_acid=wt,
        )
        self_prediction = model(
            **inputs,
            wt_amino_acid=wt,
            mutant_amino_acid=wt,
        )
    assert potentials.shape == (4, len(AMINO_ACIDS))
    torch.testing.assert_close(
        potentials.mean(dim=-1),
        torch.zeros(4),
        atol=1e-6,
        rtol=0,
    )
    torch.testing.assert_close(forward, -reverse, atol=0, rtol=0)
    torch.testing.assert_close(
        self_prediction,
        torch.zeros_like(self_prediction),
        atol=0,
        rtol=0,
    )
    cycle = (
        potentials[:, 1] - potentials[:, 0]
        + potentials[:, 2] - potentials[:, 1]
        - (potentials[:, 2] - potentials[:, 0])
    )
    torch.testing.assert_close(
        cycle,
        torch.zeros_like(cycle),
        atol=1e-6,
        rtol=0,
    )


def test_absolute_state_head_is_consistent_with_signed_ddg() -> None:
    model = StatePotentialMutationHead(
        StatePotentialConfig(
            embedding_dim=6,
            window_size=3,
            structure_dim=4,
            membrane_dim=2,
            state_dim=8,
            amino_acid_dim=5,
            hidden_dim=12,
            latent_dim=7,
            absolute_stability_head=True,
        )
    ).eval()
    generator = torch.Generator().manual_seed(11)
    inputs = {
        "wt_window": torch.randn(4, 3, 6, generator=generator),
        "window_mask": torch.ones(4, 3, dtype=torch.bool),
        "wt_global": torch.randn(4, 6, generator=generator),
        "wt_amino_acid": torch.tensor([0, 1, 2, 3], dtype=torch.long),
        "mutant_amino_acid": torch.tensor([4, 5, 6, 7], dtype=torch.long),
        "structure": torch.randn(4, 4, generator=generator),
        "structure_mask": torch.ones(4, dtype=torch.bool),
        "membrane": torch.randn(4, 2, generator=generator),
    }
    with torch.inference_mode():
        outputs = model.predict_thermodynamic_state(**inputs)
        direct = model(**inputs)
    torch.testing.assert_close(outputs["ddg"], direct)
    torch.testing.assert_close(
        outputs["mutant_absolute_stability"]
        - outputs["wt_absolute_stability"],
        -outputs["ddg"],
        atol=1e-6,
        rtol=0,
    )


def test_membrane_rank_pairs_follow_positive_delta_tm_direction() -> None:
    indices = np.arange(6, dtype=np.int64)
    proteins = np.asarray(["A", "A", "A", "B", "B", "B"])
    target = np.asarray([-2.0, 0.0, 3.0, -1.0, 1.0, 4.0])
    higher, lower = _sample_higher_is_better_pairs(
        indices,
        proteins,
        target,
        np.random.default_rng(3),
        count=32,
        minimum_gap=1.0,
    )
    assert len(higher) == 32
    assert np.all(target[higher] > target[lower])
    assert np.all(proteins[higher] == proteins[lower])


def test_truncated_gapped_structure_alignment_preserves_canonical_numbering() -> None:
    mapping, diagnostics = _aligned_structure_indices(
        "ACDEFGHIKLMN",
        "ACD--GHIK",
    )
    assert mapping.tolist() == [0, 1, 2, -1, -1, 5, 6, 7, 8]
    assert diagnostics["identity"] == 1.0
    assert diagnostics["mapped_residues"] == 7
    assert diagnostics["target_coverage"] == pytest.approx(7 / 12)


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
