from __future__ import annotations

import json

import h5py
import numpy as np
import pandas as pd
import pytest

from protein_stabilizer import accuracy_data
from protein_stabilizer.accuracy import (
    AFFINE_DDG_CALIBRATION_SCHEMA,
    BACKBONE_GEOMETRY_DIMENSION,
    MASKED_CHEMISTRY_DIMENSION,
    PORTABLE_PRIOR_DIMENSION,
    PROTEINMPNN_DIMENSION,
    PortablePriorConfig,
    apply_affine_ddg_calibration,
    backbone_geometry_features,
    blend_state_and_prior,
    fit_portable_prior,
    load_portable_prior,
    masked_chemistry_features,
    portable_prior_features,
    predict_portable_prior,
    save_portable_prior,
)
from protein_stabilizer.accuracy_data import (
    build_masked_marginal_cache,
    build_target_masked_marginal_cache,
    load_target_masked_marginals,
    load_target_state_embeddings,
)
from protein_stabilizer.cli import build_parser
from protein_stabilizer.data import EmbeddingRequest
from protein_stabilizer.embeddings import (
    ESMCProvenance,
    HierarchicalEmbedding,
    HierarchyEmbeddingWriter,
    file_sha256,
)
from protein_stabilizer.structure import ProteinMPNNBackboneEmbedder
from protein_stabilizer.accuracy_predictor import (
    _validate_runtime_provenance,
)


def _coordinates() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    coordinates = np.zeros((2, 5, 4, 3), dtype=np.float32)
    coordinates[0, :, :, 0] = np.arange(5)[:, None] * 3.8
    coordinates[1, :, :, 1] = np.arange(5)[:, None] * 3.8
    mask = np.array(
        [
            [True, True, False, True, True],
            [False, False, False, False, False],
        ]
    )
    return coordinates, mask, np.array([5, 3])


def test_backbone_geometry_is_target_independent_and_mask_aware() -> None:
    coordinates, mask, length = _coordinates()
    geometry = backbone_geometry_features(coordinates, mask, length)
    assert geometry.shape == (2, 5, BACKBONE_GEOMETRY_DIMENSION)
    assert np.isfinite(geometry).all()
    assert np.all(geometry[0, mask[0], -1] == 1.0)
    assert np.count_nonzero(geometry[0, 2]) == 0
    assert np.count_nonzero(geometry[1]) == 0
    np.testing.assert_array_equal(
        geometry,
        backbone_geometry_features(coordinates, mask, length),
    )


def test_masked_chemistry_and_portable_feature_dimensions() -> None:
    log_probability = np.log(
        np.tile(np.arange(1, 21, dtype=np.float32), (3, 1))
        / np.arange(1, 21, dtype=np.float32).sum()
    )
    wt = np.array([0, 2, 4])
    mutant = np.array([1, 2, 7])
    chemistry = masked_chemistry_features(
        log_probability,
        wt,
        mutant,
        np.array([1, 2, 3]),
        np.array([5, 5, 5]),
    )
    assert chemistry.shape == (3, MASKED_CHEMISTRY_DIMENSION)
    assert np.isfinite(chemistry).all()

    coordinates, mask, length = _coordinates()
    geometry = backbone_geometry_features(coordinates, mask, length)
    proteinmpnn = np.zeros(
        (2, 5, PROTEINMPNN_DIMENSION), dtype=np.float32
    )
    features = portable_prior_features(
        log_probability,
        wt,
        mutant,
        np.array([0, 1, 2]),
        np.array([0, 0, 0]),
        length,
        geometry,
        proteinmpnn,
    )
    assert features.shape == (3, PORTABLE_PRIOR_DIMENSION)
    assert np.isfinite(features).all()


def test_portable_prior_round_trip_and_self_constraint(tmp_path) -> None:
    rng = np.random.default_rng(7)
    features = rng.normal(size=(80, PORTABLE_PRIOR_DIMENSION)).astype(
        np.float32
    )
    target = (
        0.5 * features[:, 0] - 0.25 * features[:, 4]
    ).astype(np.float32)
    config = PortablePriorConfig(
        n_estimators=8,
        max_features=0.75,
        min_samples_leaf=2,
        random_state=11,
    )
    model = fit_portable_prior(features, target, config=config)
    wt = np.array([1, 2, 3])
    mutant = np.array([1, 7, 9])
    prediction = predict_portable_prior(
        model, features[:3], wt, mutant
    )
    assert prediction[0] == 0.0
    assert np.isfinite(prediction).all()

    path = tmp_path / "portable_prior.joblib"
    manifest = save_portable_prior(
        path,
        model,
        config=config,
        provenance={"fixture": True},
    )
    loaded, loaded_manifest = load_portable_prior(path)
    assert loaded_manifest == manifest
    np.testing.assert_allclose(
        loaded.predict(features[:4]),
        model.predict(features[:4]),
    )
    manifest_path = path.with_suffix(path.suffix + ".json")
    tampered = json.loads(manifest_path.read_text(encoding="utf-8"))
    tampered["model_sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(RuntimeError, match="hash mismatch"):
        load_portable_prior(path)


def test_accuracy_blend_uses_frozen_weights() -> None:
    state = np.array([-1.0, 0.5], dtype=np.float32)
    prior = np.array([-0.5, -0.5], dtype=np.float32)
    blended = blend_state_and_prior(state, prior)
    np.testing.assert_allclose(blended, [-0.8, 0.1], atol=1.0e-7)
    with pytest.raises(ValueError, match="must align"):
        blend_state_and_prior(state, prior[:1])


def test_affine_ddg_calibration_is_monotone_and_self_zeroed() -> None:
    calibration = {
        "schema": AFFINE_DDG_CALIBRATION_SCHEMA,
        "coefficients": {
            "state": 0.5,
            "portable_prior": 0.25,
            "intercept": -0.1,
        },
    }
    prediction = apply_affine_ddg_calibration(
        np.array([-1.0, 0.5], dtype=np.float32),
        np.array([-0.4, 0.2], dtype=np.float32),
        calibration,
        self_mask=np.array([False, True]),
    )
    np.testing.assert_allclose(prediction, [-0.7, 0.0], atol=1.0e-7)

    invalid = {
        **calibration,
        "coefficients": {
            **calibration["coefficients"],
            "state": -0.1,
        },
    }
    with pytest.raises(RuntimeError, match="not monotone"):
        apply_affine_ddg_calibration(
            np.array([0.0], dtype=np.float32),
            np.array([0.0], dtype=np.float32),
            invalid,
        )
    with pytest.raises(RuntimeError, match="schema mismatch"):
        apply_affine_ddg_calibration(
            np.array([0.0], dtype=np.float32),
            np.array([0.0], dtype=np.float32),
            {**calibration, "schema": "wrong"},
        )
    with pytest.raises(ValueError, match="mask must align"):
        apply_affine_ddg_calibration(
            np.array([0.0], dtype=np.float32),
            np.array([0.0], dtype=np.float32),
            calibration,
            self_mask=np.array([True, False]),
        )


def test_masked_cache_is_resumable_and_noop_is_hash_stable(
    tmp_path, monkeypatch
) -> None:
    source = (
        tmp_path
        / "data"
        / "processed"
        / "thermompnn_d"
        / "Megascale"
        / "csv"
    )
    source.mkdir(parents=True)
    pd.DataFrame(
        {
            "pdb_id": ["p1", "p1"],
            "wt_seq": ["ACD", "ACD"],
            "pos1": [1, 2],
        }
    ).to_csv(source / "cdna1_train.csv", index=False)
    pd.DataFrame(
        {
            "pdb_id": ["p2"],
            "wt_seq": ["EFG"],
            "pos1": [3],
        }
    ).to_csv(source / "cdna1_test.csv", index=False)

    class Provenance:
        def canonical_json(self) -> str:
            return json.dumps(
                {
                    "model_name": "fixture",
                    "checkpoint_sha256": "a" * 64,
                },
                sort_keys=True,
                separators=(",", ":"),
            )

    class Embedder:
        calls = 0

        def __init__(self, *args, **kwargs) -> None:
            self.provenance = Provenance()

        def masked_marginal_log_probabilities(
            self, sequence, positions, **kwargs
        ) -> np.ndarray:
            Embedder.calls += 1
            value = np.arange(20, dtype=np.float32)
            return np.tile(value, (len(positions), 1))

    monkeypatch.setattr(accuracy_data, "ESMCEmbedder", Embedder)
    output = tmp_path / "masked.h5"
    first = build_masked_marginal_cache(tmp_path, output)
    assert first["complete_rows"] == 3
    assert Embedder.calls == 2
    before = file_sha256(output)
    second = build_masked_marginal_cache(tmp_path, output)
    assert second["complete_rows"] == 3
    assert Embedder.calls == 2
    assert file_sha256(output) == before
    with h5py.File(output, "r") as handle:
        assert np.all(handle["complete"][:])
        assert handle["log_probabilities"].shape == (3, 20)


def test_target_masked_cache_is_resumable_and_subset_addressable(
    tmp_path,
    monkeypatch,
) -> None:
    class Provenance:
        checkpoint_sha256 = "a" * 64

        def canonical_json(self) -> str:
            return json.dumps(
                {
                    "model_name": "fixture",
                    "checkpoint_sha256": self.checkpoint_sha256,
                },
                sort_keys=True,
                separators=(",", ":"),
            )

    class Embedder:
        def __init__(self) -> None:
            self.provenance = Provenance()
            self.calls = 0

        def masked_marginal_log_probabilities(
            self, sequence, positions, **kwargs
        ) -> np.ndarray:
            self.calls += 1
            return np.stack(
                [
                    np.arange(20, dtype=np.float32) + position
                    for position in positions
                ]
            )

    embedder = Embedder()
    output = tmp_path / "target_masked.h5"
    first = build_target_masked_marginal_cache(
        "ACDE",
        (1, 3, 4),
        output,
        max_batch_size=2,
        embedder=embedder,
    )
    assert first["computed_sites"] == 3
    assert embedder.calls == 2
    before = file_sha256(output)
    second = build_target_masked_marginal_cache(
        "ACDE",
        (4, 3, 1),
        output,
        max_batch_size=2,
        embedder=embedder,
    )
    assert second["computed_sites"] == 0
    assert second["cache_hits"] == 3
    assert embedder.calls == 2
    assert file_sha256(output) == before
    monkeypatch.setattr(
        accuracy_data,
        "ESMCEmbedder",
        lambda **kwargs: pytest.fail("completed cache loaded the encoder"),
    )
    cache_only = build_target_masked_marginal_cache(
        "ACDE",
        (1, 3, 4),
        output,
        model_name="fixture",
    )
    assert cache_only["computed_sites"] == 0

    values, provenance = load_target_masked_marginals(
        output, "ACDE", (4, 1)
    )
    np.testing.assert_array_equal(
        values[0], np.arange(20, dtype=np.float32) + 4
    )
    np.testing.assert_array_equal(
        values[1], np.arange(20, dtype=np.float32) + 1
    )
    assert provenance["model_provenance"]["model_name"] == "fixture"
    with pytest.raises(RuntimeError, match="lacks requested positions"):
        load_target_masked_marginals(output, "ACDE", (2,))
    with pytest.raises(RuntimeError, match="sequence hash mismatch"):
        load_target_masked_marginals(output, "ACDF", (1,))


def test_cross_scale_accuracy_cli_contract() -> None:
    parser = build_parser()
    cache = parser.parse_args(
        [
            "cache-accuracy-target",
            "--sequence",
            "ACDE",
            "--output",
            "masked.h5",
        ]
    )
    assert cache.command == "cache-accuracy-target"
    state_cache = parser.parse_args(
        ["cache-accuracy-state-6b", "--sequence", "ACDE"]
    )
    assert state_cache.command == "cache-accuracy-state-6b"
    assert state_cache.model == "biohub/ESMC-6B"
    screen = parser.parse_args(
        [
            "screen-accuracy-6b",
            "--sequence",
            "ACDE",
            "--masked-marginals-cache",
            "masked.h5",
        ]
    )
    assert screen.command == "screen-accuracy-6b"
    assert screen.model == "biohub/ESMC-6B"
    assert screen.max_batch_size == 1
    assert screen.accuracy_checkpoint.name == "accuracy_ensemble.pt"
    assert screen.accuracy_checkpoint.parent.name == "promoted_affine"


def test_target_state_cache_reconstructs_full_wt_residues(tmp_path) -> None:
    sequence = "ACDE"
    dimension = 3
    residues = np.arange(
        len(sequence) * dimension, dtype=np.float32
    ).reshape(len(sequence), dimension)
    windows = np.zeros((len(sequence), 9, dimension), dtype=np.float32)
    mask = np.zeros((len(sequence), 9), dtype=bool)
    windows[:, 4] = residues
    mask[:, 4] = True
    provenance = ESMCProvenance(
        model_name="fixture-6b",
        embedding_dimension=dimension,
        package_version="fixture",
        checkpoint_path="/fixture",
        checkpoint_sha256="b" * 64,
        inference_dtype="float32",
        storage_dtype="float32",
    )
    path = tmp_path / "state.h5"
    request = EmbeddingRequest(sequence, (1, 2, 3, 4))
    with HierarchyEmbeddingWriter(
        path, provenance, window_radius=4
    ) as writer:
        writer.append(
            [request],
            [
                HierarchicalEmbedding(
                    global_mean=residues.mean(axis=0),
                    windows=windows,
                    window_mask=mask,
                )
            ],
        )
    loaded, cache = load_target_state_embeddings(path, sequence)
    np.testing.assert_array_equal(loaded, residues)
    assert cache["positions"] == len(sequence)
    assert cache["model_provenance"]["checkpoint_sha256"] == "b" * 64
    with pytest.raises(RuntimeError, match="does not cover every WT"):
        load_target_state_embeddings(path, "ACDEF")


def test_accuracy_runtime_rejects_numeric_provenance_drift() -> None:
    esmc = {
        "checkpoint_sha256": "a" * 64,
        "embedding_dimension": 2560,
        "inference_dtype": "float32",
        "storage_dtype": "float32",
        "residue_policy": "final",
    }
    proteinmpnn = {"checkpoint_sha256": "b" * 64}
    checkpoint = {
        "feature_provenance": {
            "hierarchy_rows": [
                {
                    "hierarchy_provenance": dict(esmc),
                    "structure_provenance": {
                        "proteinmpnn": dict(proteinmpnn)
                    },
                }
            ],
            "masked_marginal": {
                "model_provenance": dict(esmc)
            },
        },
        "training": {
            "structure": {
                "proteinmpnn_provenance": dict(proteinmpnn)
            }
        },
    }

    class StaticProvenance:
        def canonical_json(self) -> str:
            return json.dumps(proteinmpnn)

    class StaticEmbedder:
        provenance = StaticProvenance()

    _validate_runtime_provenance(
        checkpoint,
        dict(esmc),
        dict(esmc),
        StaticEmbedder(),
        dict(proteinmpnn),
    )
    reduced = dict(esmc, storage_dtype="float16")
    with pytest.raises(RuntimeError, match="storage_dtype"):
        _validate_runtime_provenance(
            checkpoint,
            reduced,
            dict(esmc),
            StaticEmbedder(),
            dict(proteinmpnn),
        )


def test_exact_backbone_coordinates_follow_target_chain(tmp_path) -> None:
    class Parser:
        @staticmethod
        def parse_PDB(path, ca_only=False):
            coordinates = {
                f"{atom}_chain_A": np.full(
                    (3, 3), index, dtype=np.float32
                )
                for index, atom in enumerate(("N", "CA", "C", "O"))
            }
            return [
                {
                    "seq_chain_A": "ACD",
                    "coords_chain_A": coordinates,
                    "name": "fixture",
                }
            ]

    embedder = ProteinMPNNBackboneEmbedder.__new__(
        ProteinMPNNBackboneEmbedder
    )
    embedder.module = Parser()
    coordinates, mask, chain = embedder.backbone_coordinates(
        tmp_path / "fixture.pdb", "ACD"
    )
    assert coordinates.shape == (3, 4, 3)
    assert mask.tolist() == [True, True, True]
    assert chain == "A"
