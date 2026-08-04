"""Trainable full-sequence ESM-C/ProteinMPNN stability architecture."""

from __future__ import annotations

import importlib
import json
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from types import ModuleType

import torch
from torch import nn

from .data import AMINO_ACIDS
from .embeddings import file_sha256
from .structure import (
    PROTEINMPNN_COMMIT,
    PROTEINMPNN_MODEL_NAME,
    ProteinMPNNProvenance,
)


FULL_STRUCTURE_MODEL_SCHEMA = "protein-stabilizer.full-structure-model.v1"


def load_trainable_proteinmpnn(
    repository: Path,
    *,
    model_name: str = PROTEINMPNN_MODEL_NAME,
) -> tuple[nn.Module, ModuleType, ProteinMPNNProvenance]:
    """Load the pinned pretrained ProteinMPNN without freezing its weights."""

    repository = Path(repository).resolve()
    source = repository / "protein_mpnn_utils.py"
    checkpoint_path = (
        repository / "vanilla_model_weights" / f"{model_name}.pt"
    )
    if not source.is_file() or not checkpoint_path.is_file():
        raise FileNotFoundError(
            "ProteinMPNN repository must contain protein_mpnn_utils.py and "
            f"vanilla_model_weights/{model_name}.pt"
        )
    commit = subprocess.run(
        ["git", "-C", str(repository), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if commit != PROTEINMPNN_COMMIT:
        raise RuntimeError(
            f"ProteinMPNN commit mismatch: expected {PROTEINMPNN_COMMIT}, "
            f"got {commit}"
        )
    repository_string = str(repository)
    if repository_string not in sys.path:
        sys.path.insert(0, repository_string)
    module = importlib.import_module("protein_mpnn_utils")
    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False
    )
    hidden_dimension = 128
    model = module.ProteinMPNN(
        ca_only=False,
        num_letters=21,
        node_features=hidden_dimension,
        edge_features=hidden_dimension,
        hidden_dim=hidden_dimension,
        num_encoder_layers=3,
        num_decoder_layers=3,
        augment_eps=0.0,
        k_neighbors=checkpoint["num_edges"],
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    provenance = ProteinMPNNProvenance(
        repository="https://github.com/dauparas/ProteinMPNN",
        commit=commit,
        model_name=model_name,
        checkpoint_path=str(checkpoint_path),
        checkpoint_sha256=file_sha256(checkpoint_path),
        source_sha256=file_sha256(source),
        hidden_dimension=hidden_dimension,
        residue_policy=(
            "trainable-nonautoregressive-full-sequence-decoder-states-v1"
        ),
        storage_dtype="float32",
    )
    return model, module, provenance


class MaskedProteinMPNNEncoder(nn.Module):
    """Expose trainable leave-one-residue-out ProteinMPNN decoder states."""

    def __init__(
        self,
        model: nn.Module,
        module: ModuleType,
    ) -> None:
        super().__init__()
        self.model = model
        self._gather_nodes = module.gather_nodes
        self._cat_neighbors_nodes = module.cat_neighbors_nodes
        self.output_dim = 4 * int(model.hidden_dim)

    def forward(
        self,
        coordinates: torch.Tensor,
        sequence_tokens: torch.Tensor,
        residue_mask: torch.Tensor,
    ) -> torch.Tensor:
        if coordinates.ndim != 4 or coordinates.shape[2:] != (4, 3):
            raise ValueError(
                "ProteinMPNN coordinates must have shape [batch, length, 4, 3]"
            )
        if sequence_tokens.shape != coordinates.shape[:2]:
            raise ValueError("ProteinMPNN sequence-token shape mismatch")
        if (
            residue_mask.shape != coordinates.shape[:2]
            or residue_mask.dtype != torch.bool
        ):
            raise ValueError(
                "ProteinMPNN residue mask must be boolean [batch, length]"
            )
        mask = residue_mask.to(dtype=coordinates.dtype)
        batch_size, length = mask.shape
        residue_index = torch.arange(
            length, device=coordinates.device, dtype=torch.long
        ).unsqueeze(0).expand(batch_size, -1)
        chain_encoding = torch.ones_like(residue_index)

        edges, edge_index = self.model.features(
            coordinates,
            mask,
            residue_index,
            chain_encoding,
        )
        hidden = torch.zeros(
            (batch_size, length, edges.shape[-1]),
            device=edges.device,
            dtype=edges.dtype,
        )
        edge_hidden = self.model.W_e(edges)
        attend = self._gather_nodes(
            mask.unsqueeze(-1), edge_index
        ).squeeze(-1)
        attend = mask.unsqueeze(-1) * attend
        for layer in self.model.encoder_layers:
            hidden, edge_hidden = layer(
                hidden,
                edge_hidden,
                edge_index,
                mask,
                attend,
            )

        amino_acid = self.model.W_s(sequence_tokens)
        sequence_edges = self._cat_neighbors_nodes(
            amino_acid, edge_hidden, edge_index
        )
        encoder_edges = self._cat_neighbors_nodes(
            torch.zeros_like(amino_acid),
            edge_hidden,
            edge_index,
        )
        encoder_context = self._cat_neighbors_nodes(
            hidden, encoder_edges, edge_index
        )

        # Every position can see every other sequence identity, but not its
        # own. This is the one-shot masked decoding policy used by SPURS.
        leave_one_out = (
            1.0
            - torch.eye(
                length,
                device=coordinates.device,
                dtype=coordinates.dtype,
            )
        ).unsqueeze(0).expand(batch_size, -1, -1)
        backward = torch.gather(
            leave_one_out, 2, edge_index
        ).unsqueeze(-1)
        mask_1d = mask.view(batch_size, length, 1, 1)
        backward = mask_1d * backward
        forward = mask_1d * (1.0 - backward)
        encoder_forward = forward * encoder_context

        decoder_states: list[torch.Tensor] = []
        for layer in self.model.decoder_layers:
            sequence_context = self._cat_neighbors_nodes(
                hidden, sequence_edges, edge_index
            )
            combined = backward * sequence_context + encoder_forward
            hidden = layer(hidden, combined, mask)
            decoder_states.append(hidden)
        if len(decoder_states) != 3:
            raise RuntimeError("expected three ProteinMPNN decoder layers")
        result = torch.cat(
            [
                decoder_states[-1],
                amino_acid,
                decoder_states[-2],
                decoder_states[-3],
            ],
            dim=-1,
        )
        return result * residue_mask.unsqueeze(-1)


@dataclass(frozen=True)
class FullStructureConfig:
    esm_dim: int = 1152
    structure_dim: int = 512
    fusion_dim: int = 256
    attention_heads: int = 8
    local_kernel_size: int = 9
    topology_dim: int = 8
    dropout: float = 0.10
    use_structure: bool = True


class FullStructureStateModel(nn.Module):
    """One-pass per-residue 20-state potential with trainable structure fusion.

    Frozen final-layer ESM-C residue embeddings supply local and whole-protein
    sequence queries. A trainable non-autoregressive ProteinMPNN path supplies
    full-sequence structure keys and values. Potential differences enforce
    exact sign reversal and cycle consistency in a shared WT context.
    """

    def __init__(
        self,
        structure_encoder: nn.Module,
        config: FullStructureConfig = FullStructureConfig(),
    ) -> None:
        super().__init__()
        if config.local_kernel_size < 1 or config.local_kernel_size % 2 != 1:
            raise ValueError("local kernel size must be a positive odd number")
        if config.fusion_dim % config.attention_heads:
            raise ValueError("fusion dimension must divide attention heads")
        if config.structure_dim < 1 or config.esm_dim < 1:
            raise ValueError("encoder dimensions must be positive")
        self.config = config
        self.structure_encoder = structure_encoder
        self.sequence_projection = nn.Sequential(
            nn.LayerNorm(config.esm_dim),
            nn.Linear(config.esm_dim, config.fusion_dim),
            nn.GELU(),
        )
        self.local_projection = nn.Conv1d(
            config.fusion_dim,
            config.fusion_dim,
            kernel_size=config.local_kernel_size,
            padding=config.local_kernel_size // 2,
            groups=config.fusion_dim,
        )
        self.global_projection = nn.Sequential(
            nn.LayerNorm(config.esm_dim),
            nn.Linear(config.esm_dim, config.fusion_dim),
            nn.GELU(),
        )
        self.structure_projection = nn.Sequential(
            nn.LayerNorm(config.structure_dim),
            nn.Linear(config.structure_dim, config.fusion_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
        )
        self.cross_attention = nn.MultiheadAttention(
            config.fusion_dim,
            config.attention_heads,
            dropout=config.dropout,
            batch_first=True,
        )
        # Structure is a residual refinement of an already useful frozen-ESM
        # path. Starting near zero enables leakage-free sequence warm starts
        # without allowing an untrained structural adapter to erase them.
        self.cross_gate = nn.Parameter(torch.tensor(-4.0))
        self.topology_projection = nn.Sequential(
            nn.LayerNorm(config.topology_dim),
            nn.Linear(config.topology_dim, config.fusion_dim),
            nn.GELU(),
        )
        self.fusion_norm = nn.LayerNorm(config.fusion_dim)
        self.fusion_ffn = nn.Sequential(
            nn.Linear(config.fusion_dim, 2 * config.fusion_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(2 * config.fusion_dim, config.fusion_dim),
        )
        self.potential_decoder = nn.Sequential(
            nn.LayerNorm(config.fusion_dim),
            nn.Linear(config.fusion_dim, 2 * config.fusion_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(2 * config.fusion_dim, len(AMINO_ACIDS)),
        )

    def encode(
        self,
        esm_residue: torch.Tensor,
        esm_global: torch.Tensor,
        coordinates: torch.Tensor,
        sequence_tokens: torch.Tensor,
        sequence_mask: torch.Tensor,
        *,
        structure_mask: torch.Tensor | None = None,
        topology: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if esm_residue.ndim != 3:
            raise ValueError("ESM-C residues must have shape [batch, length, dim]")
        batch_size, length, dimension = esm_residue.shape
        if dimension != self.config.esm_dim:
            raise ValueError("ESM-C residue dimension mismatch")
        if esm_global.shape != (batch_size, self.config.esm_dim):
            raise ValueError("ESM-C global dimension mismatch")
        if (
            sequence_mask.shape != (batch_size, length)
            or sequence_mask.dtype != torch.bool
        ):
            raise ValueError("sequence mask must be boolean [batch, length]")
        if not torch.all(sequence_mask.any(dim=1)):
            raise ValueError("every protein must contain at least one residue")
        if structure_mask is None:
            structure_mask = sequence_mask
        if (
            structure_mask.shape != (batch_size, length)
            or structure_mask.dtype != torch.bool
        ):
            raise ValueError("structure mask must be boolean [batch, length]")
        if torch.any(structure_mask & ~sequence_mask):
            raise ValueError("structure mask cannot include sequence padding")

        sequence = self.sequence_projection(esm_residue)
        sequence = sequence * sequence_mask.unsqueeze(-1)
        local = self.local_projection(sequence.transpose(1, 2)).transpose(1, 2)
        global_context = self.global_projection(esm_global).unsqueeze(1)
        query = sequence + local + global_context

        has_structure = structure_mask.any(dim=1, keepdim=True).unsqueeze(-1)
        if self.config.use_structure:
            # MultiheadAttention cannot consume an example whose every key is
            # padding. Unmask one zero-valued sentinel for those examples and
            # suppress the resulting attention update with ``has_structure``.
            attention_mask = structure_mask.clone()
            no_structure = ~structure_mask.any(dim=1)
            if torch.any(no_structure):
                attention_mask[no_structure, 0] = True
            raw_structure = self.structure_encoder(
                coordinates,
                sequence_tokens,
                structure_mask,
            )
            if raw_structure.shape != (
                batch_size,
                length,
                self.config.structure_dim,
            ):
                raise ValueError("trainable ProteinMPNN feature shape mismatch")
            structure = self.structure_projection(raw_structure)
            structure = structure * structure_mask.unsqueeze(-1)
            attended, _ = self.cross_attention(
                query,
                structure,
                structure,
                key_padding_mask=~attention_mask,
                need_weights=False,
            )
            query = query + torch.sigmoid(self.cross_gate) * attended * has_structure

        if topology is not None:
            if topology.shape != (
                batch_size,
                length,
                self.config.topology_dim,
            ):
                raise ValueError("per-residue topology dimension mismatch")
            query = query + self.topology_projection(topology)
        fused = self.fusion_norm(query)
        fused = fused + self.fusion_ffn(fused)
        return fused * sequence_mask.unsqueeze(-1)

    def all_potentials(
        self,
        esm_residue: torch.Tensor,
        esm_global: torch.Tensor,
        coordinates: torch.Tensor,
        sequence_tokens: torch.Tensor,
        sequence_mask: torch.Tensor,
        *,
        structure_mask: torch.Tensor | None = None,
        topology: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        representation = self.encode(
            esm_residue,
            esm_global,
            coordinates,
            sequence_tokens,
            sequence_mask,
            structure_mask=structure_mask,
            topology=topology,
        )
        potential = self.potential_decoder(representation)
        potential = potential - potential.mean(dim=-1, keepdim=True)
        potential = potential * sequence_mask.unsqueeze(-1)
        return potential, representation

    @staticmethod
    def mutation_ddg(
        potentials: torch.Tensor,
        protein_index: torch.Tensor,
        position: torch.Tensor,
        wt_amino_acid: torch.Tensor,
        mutant_amino_acid: torch.Tensor,
    ) -> torch.Tensor:
        if not (
            protein_index.shape
            == position.shape
            == wt_amino_acid.shape
            == mutant_amino_acid.shape
        ):
            raise ValueError("mutation index tensors must have equal shapes")
        state = potentials[protein_index, position]
        row = torch.arange(len(state), device=state.device)
        return state[row, mutant_amino_acid] - state[row, wt_amino_acid]

    def config_dict(self) -> dict[str, object]:
        return asdict(self.config)


@dataclass(frozen=True)
class FullStructureEpistasisConfig:
    representation_dim: int = 256
    amino_acid_dim: int = 32
    mutation_dim: int = 192
    hidden_dim: int = 384
    dropout: float = 0.10


class FullStructureEpistasisHead(nn.Module):
    """Permutation-invariant additive-plus-epistasis mutation-set decoder."""

    def __init__(
        self,
        config: FullStructureEpistasisConfig = (
            FullStructureEpistasisConfig()
        ),
    ) -> None:
        super().__init__()
        self.config = config
        self.amino_acid = nn.Embedding(
            len(AMINO_ACIDS), config.amino_acid_dim
        )
        input_dim = config.representation_dim + 2 * config.amino_acid_dim
        self.mutation_encoder = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, config.mutation_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.mutation_dim, config.mutation_dim),
            nn.GELU(),
        )
        aggregate_dim = 2 * config.mutation_dim + 1
        self.epistasis = nn.Sequential(
            nn.LayerNorm(aggregate_dim),
            nn.Linear(aggregate_dim, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, 1, bias=False),
        )

    def forward(
        self,
        potentials: torch.Tensor,
        representations: torch.Tensor,
        protein_index: torch.Tensor,
        positions: torch.Tensor,
        wt_amino_acids: torch.Tensor,
        mutant_amino_acids: torch.Tensor,
        mutation_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if positions.ndim != 2 or positions.shape != mutation_mask.shape:
            raise ValueError("mutation sets require [batch, mutations] tensors")
        if mutation_mask.dtype != torch.bool:
            raise ValueError("mutation-set mask must be boolean")
        if not (
            wt_amino_acids.shape
            == mutant_amino_acids.shape
            == positions.shape
        ):
            raise ValueError("mutation-set amino-acid shapes differ")
        batch_size, mutation_count = positions.shape
        if protein_index.shape != (batch_size,):
            raise ValueError("one protein index is required per mutation set")
        expanded_protein = protein_index.unsqueeze(1).expand(-1, mutation_count)
        site_state = potentials[expanded_protein, positions]
        row = torch.arange(batch_size, device=positions.device).unsqueeze(1)
        column = torch.arange(
            mutation_count, device=positions.device
        ).unsqueeze(0)
        single = (
            site_state[row, column, mutant_amino_acids]
            - site_state[row, column, wt_amino_acids]
        )
        single = single * mutation_mask
        additive = single.sum(dim=1)

        site = representations[expanded_protein, positions]
        mutation = self.mutation_encoder(
            torch.cat(
                [
                    site,
                    self.amino_acid(wt_amino_acids),
                    self.amino_acid(mutant_amino_acids),
                ],
                dim=-1,
            )
        )
        mutation = mutation * mutation_mask.unsqueeze(-1)
        summed = mutation.sum(dim=1)
        maximum = mutation.masked_fill(
            ~mutation_mask.unsqueeze(-1), -torch.inf
        ).max(dim=1).values
        maximum = torch.where(
            torch.isfinite(maximum), maximum, torch.zeros_like(maximum)
        )
        count = mutation_mask.sum(dim=1, keepdim=True).to(mutation.dtype)
        epistasis = self.epistasis(
            torch.cat([summed, maximum, count], dim=-1)
        ).squeeze(-1)
        epistasis = epistasis * (count.squeeze(-1) > 1)
        return additive + epistasis, additive, epistasis

    def config_dict(self) -> dict[str, object]:
        return asdict(self.config)


def checkpoint_provenance_json(
    provenance: ProteinMPNNProvenance,
) -> str:
    """Stable JSON helper shared by training and checkpoint validation."""

    return json.dumps(
        asdict(provenance), sort_keys=True, separators=(",", ":")
    )
