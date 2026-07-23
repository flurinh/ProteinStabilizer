from __future__ import annotations

import numpy as np
import torch
from torch import nn

from protein_stabilizer.full_structure import (
    FullStructureConfig,
    FullStructureEpistasisConfig,
    FullStructureEpistasisHead,
    FullStructureStateModel,
)
from protein_stabilizer.full_structure_data import (
    MutationRow,
    ProteinRecord,
    _cluster_split_assignments,
    canonical_protein_id,
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
