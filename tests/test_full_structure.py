from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import torch
from torch import nn

from protein_stabilizer.data import AMINO_ACIDS, sequence_hash
from protein_stabilizer.embeddings import HIERARCHY_CACHE_SCHEMA
from protein_stabilizer.full_structure import (
    FullStructureConfig,
    FullStructureEpistasisConfig,
    FullStructureEpistasisHead,
    FullStructureStateModel,
)
from protein_stabilizer.full_structure_data import (
    FULL_STRUCTURE_DATA_SCHEMA,
    MutationRow,
    ProteinRecord,
    _cluster_split_assignments,
    build_full_structure_from_hierarchy_cache,
    canonical_protein_id,
)
from protein_stabilizer.full_structure_training import (
    load_full_structure_arrays,
)


class DummyStructureEncoder(nn.Module):
    output_dim = 6

    def forward(
        self,
        coordinates: torch.Tensor,
        sequence_tokens: torch.Tensor,
        structure_mask: torch.Tensor,
    ) -> torch.Tensor:
        coordinate = coordinates.mean(dim=(2, 3), keepdim=False).unsqueeze(-1)
        token = sequence_tokens.to(coordinates.dtype).unsqueeze(-1) / 20.0
        values = torch.cat(
            [coordinate, token, coordinate + token], dim=-1
        ).repeat(1, 1, 2)
        return values * structure_mask.unsqueeze(-1)


def _state_model() -> FullStructureStateModel:
    torch.manual_seed(4)
    return FullStructureStateModel(
        DummyStructureEncoder(),
        FullStructureConfig(
            esm_dim=8,
            structure_dim=6,
            fusion_dim=8,
            attention_heads=2,
            local_kernel_size=3,
            topology_dim=4,
            dropout=0.0,
        ),
    ).eval()


def test_structure_mask_does_not_remove_sequence_queries() -> None:
    model = _state_model()
    esm = torch.randn(2, 5, 8)
    global_mean = esm.mean(dim=1)
    coordinates = torch.randn(2, 5, 4, 3)
    tokens = torch.randint(0, 20, (2, 5))
    sequence_mask = torch.tensor(
        [[True, True, True, True, True], [True, True, True, False, False]]
    )
    structure_mask = torch.tensor(
        [[True, False, True, True, True], [False, False, False, False, False]]
    )
    potential, representation = model.all_potentials(
        esm,
        global_mean,
        coordinates,
        tokens,
        sequence_mask,
        structure_mask=structure_mask,
    )
    assert torch.isfinite(potential).all()
    assert torch.isfinite(representation).all()
    assert torch.linalg.vector_norm(representation[0, 1]) > 0
    assert torch.linalg.vector_norm(representation[1, 0]) > 0
    assert torch.count_nonzero(representation[1, 3:]) == 0


def test_state_potentials_enforce_direction_and_cycles() -> None:
    model = _state_model()
    potential = torch.randn(2, 4, 20)
    protein = torch.tensor([0, 1])
    position = torch.tensor([1, 2])
    a = torch.tensor([2, 6])
    b = torch.tensor([9, 13])
    c = torch.tensor([17, 4])
    forward = model.mutation_ddg(potential, protein, position, a, b)
    reverse = model.mutation_ddg(potential, protein, position, b, a)
    self_change = model.mutation_ddg(potential, protein, position, a, a)
    cycle = (
        model.mutation_ddg(potential, protein, position, a, b)
        + model.mutation_ddg(potential, protein, position, b, c)
        + model.mutation_ddg(potential, protein, position, c, a)
    )
    torch.testing.assert_close(forward, -reverse)
    torch.testing.assert_close(self_change, torch.zeros_like(self_change))
    torch.testing.assert_close(cycle, torch.zeros_like(cycle), atol=1e-6, rtol=0)


def test_epistasis_decoder_is_permutation_invariant() -> None:
    torch.manual_seed(8)
    head = FullStructureEpistasisHead(
        FullStructureEpistasisConfig(
            representation_dim=8,
            amino_acid_dim=4,
            mutation_dim=12,
            hidden_dim=16,
            dropout=0.0,
        )
    ).eval()
    potentials = torch.randn(2, 6, 20)
    representations = torch.randn(2, 6, 8)
    protein = torch.tensor([0, 1])
    position = torch.tensor([[1, 4], [0, 5]])
    wt = torch.tensor([[2, 8], [1, 11]])
    mutant = torch.tensor([[7, 3], [15, 9]])
    mask = torch.ones((2, 2), dtype=torch.bool)
    original = head(
        potentials,
        representations,
        protein,
        position,
        wt,
        mutant,
        mask,
    )
    permuted = head(
        potentials,
        representations,
        protein,
        position.flip(1),
        wt.flip(1),
        mutant.flip(1),
        mask,
    )
    for left, right in zip(original, permuted, strict=True):
        torch.testing.assert_close(left, right)


def test_family_assignment_never_splits_a_cluster() -> None:
    proteins = tuple(
        ProteinRecord(f"p{index}", "ACDEFGHIK")
        for index in range(8)
    )
    families = {
        "p0": "a",
        "p1": "a",
        "p2": "b",
        "p3": "c",
        "p4": "d",
        "p5": "e",
        "p6": "f",
        "p7": "g",
    }
    rows = tuple(
        MutationRow(
            protein_id=protein.protein_id,
            positions=(1,),
            wt_amino_acids=(1,),
            mutant_amino_acids=(2,),
            target=float(index),
            source_partition="ignored",
        )
        for index, protein in enumerate(proteins)
    )
    assignment = _cluster_split_assignments(
        proteins,
        families,
        rows,
        (),
        seed=7,
        outer_fraction=0.20,
        folds=3,
    )
    assert assignment["p0"] == assignment["p1"]
    assert set(assignment.values()) == {-1, 0, 1, 2}
    repeated = _cluster_split_assignments(
        proteins,
        families,
        rows,
        (),
        seed=7,
        outer_fraction=0.20,
        folds=3,
    )
    assert repeated == assignment


def test_canonical_protein_id_handles_double_dataset_suffix() -> None:
    assert canonical_protein_id("1ABC.pdb") == "1abc"
    assert canonical_protein_id("design_001") == "design_001"


def _strings(
    group: h5py.Group,
    name: str,
    values: list[str],
) -> None:
    group.create_dataset(
        name,
        data=np.asarray(values, dtype=object),
        dtype=h5py.string_dtype(encoding="utf-8"),
    )


def test_reconstruct_full_structure_from_hierarchy_windows(
    tmp_path: Path,
) -> None:
    source_path = tmp_path / "source.h5"
    cache_path = tmp_path / "hierarchy.h5"
    output_path = tmp_path / "scaled.h5"
    sequences = ["ACD", "EFG", "HIK"]
    tokens = np.asarray(
        [[AMINO_ACIDS.index(value) for value in sequence] for sequence in sequences],
        dtype=np.int8,
    )
    with h5py.File(source_path, "w") as handle:
        handle.attrs["schema"] = FULL_STRUCTURE_DATA_SCHEMA
        handle.attrs["embedding_source"] = "source-embedding.h5"
        handle.attrs["embedding_source_sha256"] = "source-embedding"
        handle.attrs["embedding_provenance"] = json.dumps(
            {"checkpoint_sha256": "small", "embedding_dimension": 2}
        )
        handle.attrs["structure_archive"] = "structures.tar.gz"
        handle.attrs["structure_archive_sha256"] = "structures"
        handle.attrs["proteinmpnn_provenance"] = json.dumps(
            {"checkpoint_sha256": "proteinmpnn"}
        )
        handle.attrs["source_sha256"] = json.dumps({"rows.csv": "rows"})
        handle.attrs["split_policy"] = json.dumps(
            {"development_folds": 2, "outer_value": -1}
        )
        proteins = handle.create_group("proteins")
        _strings(proteins, "protein_id", ["p0", "p1", "p2"])
        _strings(proteins, "sequence", sequences)
        _strings(proteins, "family_cluster", ["f0", "f1", "f2"])
        _strings(proteins, "structure_chain", ["A", "A", "A"])
        proteins.create_dataset("length", data=np.asarray([3, 3, 3]))
        proteins.create_dataset("split", data=np.asarray([0, 1, 0]))
        proteins.create_dataset(
            "esm_residue", data=np.zeros((3, 3, 2), dtype=np.float32)
        )
        proteins.create_dataset(
            "esm_global", data=np.zeros((3, 2), dtype=np.float32)
        )
        proteins.create_dataset(
            "coordinates", data=np.zeros((3, 3, 4, 3), dtype=np.float32)
        )
        proteins.create_dataset("sequence_tokens", data=tokens)
        proteins.create_dataset(
            "sequence_mask", data=np.ones((3, 3), dtype=bool)
        )
        proteins.create_dataset(
            "structure_mask", data=np.ones((3, 3), dtype=bool)
        )
        singles = handle.create_group("singles")
        singles.create_dataset("protein_index", data=np.asarray([0, 1]))
        singles.create_dataset(
            "position", data=np.asarray([[0], [1]], dtype=np.int16)
        )
        singles.create_dataset(
            "wt_amino_acid",
            data=np.asarray(
                [[AMINO_ACIDS.index("A")], [AMINO_ACIDS.index("F")]]
            ),
        )
        singles.create_dataset(
            "mutant_amino_acid", data=np.asarray([[1], [2]])
        )
        singles.create_dataset(
            "target", data=np.asarray([-0.2, 0.4], dtype=np.float32)
        )
        _strings(
            singles,
            "source_partition",
            ["historical_train", "historical_train"],
        )
        doubles = handle.create_group("doubles")
        doubles.create_dataset("protein_index", data=np.asarray([1, 2]))
        doubles.create_dataset(
            "position", data=np.asarray([[0, 1], [0, 1]], dtype=np.int16)
        )
        doubles.create_dataset(
            "wt_amino_acid",
            data=np.asarray(
                [
                    [AMINO_ACIDS.index("E"), AMINO_ACIDS.index("F")],
                    [AMINO_ACIDS.index("H"), AMINO_ACIDS.index("I")],
                ]
            ),
        )
        doubles.create_dataset(
            "mutant_amino_acid", data=np.asarray([[1, 2], [2, 3]])
        )
        doubles.create_dataset(
            "target", data=np.asarray([0.1, -0.3], dtype=np.float32)
        )
        _strings(
            doubles,
            "source_partition",
            ["historical_train", "historical_train"],
        )

    residue = np.arange(2 * 3 * 4, dtype=np.float32).reshape(2, 3, 4)
    windows = np.zeros((3, 3, 4), dtype=np.float32)
    masks = np.zeros((3, 3), dtype=bool)
    windows[0, 1:] = residue[0, :2]
    masks[0, 1:] = True
    windows[1, :2] = residue[0, 1:]
    masks[1, :2] = True
    windows[2] = residue[1]
    masks[2] = True
    provenance = {
        "model_name": "strict-6b-test",
        "checkpoint_sha256": "large",
        "embedding_dimension": 4,
        "inference_dtype": "float32",
        "storage_dtype": "float32",
    }
    with h5py.File(cache_path, "w") as handle:
        handle.attrs["schema"] = HIERARCHY_CACHE_SCHEMA
        handle.attrs["window_radius"] = 1
        handle.attrs["model_provenance"] = json.dumps(provenance)
        handle.create_dataset(
            "sequence_hashes",
            data=np.asarray(
                [sequence_hash(sequence).encode("ascii") for sequence in sequences],
                dtype="S64",
            ),
        )
        _strings(handle, "sequences", sequences)
        handle.create_dataset(
            "sequence_length", data=np.asarray([3, 3, 3])
        )
        global_mean = np.zeros((3, 4), dtype=np.float32)
        global_mean[:2] = residue.mean(axis=1)
        handle.create_dataset("global_mean", data=global_mean)
        handle.create_dataset(
            "site_sequence_index", data=np.asarray([0, 0, 1])
        )
        handle.create_dataset("site_position", data=np.asarray([1, 3, 2]))
        handle.create_dataset("window_embeddings", data=windows)
        handle.create_dataset("window_mask", data=masks)

    report = build_full_structure_from_hierarchy_cache(
        source_path, cache_path, output_path
    )
    arrays = load_full_structure_arrays(output_path)
    assert report["proteins"] == 2
    assert report["single_rows"] == 2
    assert report["double_rows"] == 1
    assert report["embedding_dimension"] == 4
    assert report["maximum_overlap_absolute_difference"] == 0.0
    assert arrays.protein_id.tolist() == ["p0", "p1"]
    np.testing.assert_array_equal(arrays.esm_residue, residue)
    np.testing.assert_array_equal(arrays.esm_global, residue.mean(axis=1))
    assert arrays.singles.protein_index.tolist() == [0, 1]
    assert arrays.doubles.protein_index.tolist() == [1]
    with h5py.File(output_path, "r") as handle:
        recorded = json.loads(handle.attrs["embedding_reconstruction"])
        assert recorded["single_mutant_proteins_only"] is True
        assert (
            json.loads(handle.attrs["embedding_provenance"])
            ["checkpoint_sha256"]
            == "large"
        )
