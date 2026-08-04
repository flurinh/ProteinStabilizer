"""Small trainable heads over frozen ESM-C mutation deltas."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import nn

from .data import AMINO_ACIDS


@dataclass(frozen=True)
class SingleHeadConfig:
    embedding_dim: int = 1152
    hidden_dim: int = 512
    latent_dim: int = 128
    dropout: float = 0.10


class SingleMutationHead(nn.Module):
    """Predict ddG from one contextual mutant-minus-WT residue vector."""

    def __init__(self, config: SingleHeadConfig = SingleHeadConfig()) -> None:
        super().__init__()
        self.config = config
        self.encoder = nn.Sequential(
            nn.LayerNorm(config.embedding_dim),
            nn.Linear(config.embedding_dim, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.latent_dim),
            nn.GELU(),
            nn.LayerNorm(config.latent_dim),
        )
        self.output = nn.Linear(config.latent_dim, 1)

    def latent(self, delta: torch.Tensor) -> torch.Tensor:
        if delta.shape[-1] != self.config.embedding_dim:
            raise ValueError("single-mutant delta dimension mismatch")
        return self.encoder(delta)

    def forward(self, delta: torch.Tensor) -> torch.Tensor:
        return self.output(self.latent(delta)).squeeze(-1)

    def config_dict(self) -> dict[str, object]:
        return asdict(self.config)


class SingleMutationEnsemble(nn.Module):
    """Average predictions while retaining the primary member's latent space."""

    def __init__(self, members: list[SingleMutationHead]) -> None:
        super().__init__()
        if not members:
            raise ValueError("single-mutation ensemble requires at least one member")
        config = members[0].config
        if any(member.config != config for member in members[1:]):
            raise ValueError("single-mutation ensemble member configurations differ")
        self.members = nn.ModuleList(members)
        self.config = config

    def latent(self, delta: torch.Tensor) -> torch.Tensor:
        # Independently trained latent coordinates are not aligned, so their
        # element-wise mean has no stable meaning. Keep the primary member's
        # coordinate system for backward-compatible downstream calibrations.
        return self.members[0].latent(delta)

    def member_predictions(self, delta: torch.Tensor) -> torch.Tensor:
        """Return one prediction row per independently trained member."""

        return torch.stack([member(delta) for member in self.members])

    def forward(self, delta: torch.Tensor) -> torch.Tensor:
        return self.member_predictions(delta).mean(dim=0)

    def config_dict(self) -> dict[str, object]:
        return asdict(self.config)


class DirectionalSingleMutationHead(nn.Module):
    """Physics-constrained signed ddG and stabilizer-retrieval head.

    The input is the contextual residue difference ``mutant - WT``.  Reversing
    the mutation therefore negates the input exactly.  Antisymmetrizing a
    shared encoder makes the thermodynamic prediction an odd function:

    ``ddG(delta) = -ddG(-delta)`` and ``ddG(0) = 0``.

    A separate retrieval logit shares the directional latent representation
    but is not interpreted as kcal/mol.  It is trained to rank the rare
    stabilizing tail, where a larger value means "more likely stabilizing".
    """

    def __init__(self, config: SingleHeadConfig = SingleHeadConfig()) -> None:
        super().__init__()
        self.config = config
        self.encoder = nn.Sequential(
            nn.LayerNorm(config.embedding_dim),
            nn.Linear(config.embedding_dim, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.latent_dim),
            nn.GELU(),
            nn.LayerNorm(config.latent_dim),
        )
        # The directional latent is exactly zero for a self-mutation, so a
        # bias-free regression projection preserves the zero-ddG constraint.
        self.output = nn.Linear(config.latent_dim, 1, bias=False)
        self.retrieval_output = nn.Linear(config.latent_dim, 1)

    def latent(self, delta: torch.Tensor) -> torch.Tensor:
        if delta.shape[-1] != self.config.embedding_dim:
            raise ValueError("single-mutant delta dimension mismatch")
        return 0.5 * (self.encoder(delta) - self.encoder(-delta))

    def forward(self, delta: torch.Tensor) -> torch.Tensor:
        return self.output(self.latent(delta)).squeeze(-1)

    def stabilizer_logit(self, delta: torch.Tensor) -> torch.Tensor:
        """Return an uncalibrated retrieval score; larger is more stabilizing."""

        return self.retrieval_output(self.latent(delta)).squeeze(-1)

    def config_dict(self) -> dict[str, object]:
        return asdict(self.config)


class DirectionalSingleMutationEnsemble(nn.Module):
    """Average independently trained directional heads."""

    def __init__(self, members: list[DirectionalSingleMutationHead]) -> None:
        super().__init__()
        if not members:
            raise ValueError("directional ensemble requires at least one member")
        config = members[0].config
        if any(member.config != config for member in members[1:]):
            raise ValueError("directional ensemble member configurations differ")
        self.members = nn.ModuleList(members)
        self.config = config

    def latent(self, delta: torch.Tensor) -> torch.Tensor:
        return self.members[0].latent(delta)

    def member_predictions(self, delta: torch.Tensor) -> torch.Tensor:
        return torch.stack([member(delta) for member in self.members])

    def member_stabilizer_logits(self, delta: torch.Tensor) -> torch.Tensor:
        return torch.stack([member.stabilizer_logit(delta) for member in self.members])

    def forward(self, delta: torch.Tensor) -> torch.Tensor:
        return self.member_predictions(delta).mean(dim=0)

    def stabilizer_logit(self, delta: torch.Tensor) -> torch.Tensor:
        return self.member_stabilizer_logits(delta).mean(dim=0)

    def config_dict(self) -> dict[str, object]:
        return asdict(self.config)


@dataclass(frozen=True)
class HierarchicalHeadConfig:
    """Frozen-encoder dimensions for the v2 contextual mutation head."""

    embedding_dim: int = 1152
    window_size: int = 9
    structure_dim: int = 128
    membrane_dim: int = 8
    state_dim: int = 192
    context_dim: int = 96
    hidden_dim: int = 384
    latent_dim: int = 128


class HierarchicalDirectionalMutationHead(nn.Module):
    """Exact directional head over complete WT and mutant contextual states.

    The ESM-C encoder remains frozen.  Each state is summarized from the
    mutation-site residue, an ordered local window, and the whole-protein mean.
    A frozen ProteinMPNN residue vector and engineered membrane/topology
    features condition the *difference* between mutant and WT states.

    The final latent is explicitly antisymmetrized, so swapping the complete
    WT and mutant inputs negates every signed output exactly in evaluation
    mode, including at self mutations.  Separate projections retain the
    distinct semantics of thermodynamic ddG, assay-specific delta-Tm, and
    stabilizer retrieval.
    """

    def __init__(
        self,
        config: HierarchicalHeadConfig = HierarchicalHeadConfig(),
    ) -> None:
        super().__init__()
        if config.window_size < 1 or config.window_size % 2 != 1:
            raise ValueError("hierarchical window size must be a positive odd number")
        if min(
            config.embedding_dim,
            config.state_dim,
            config.context_dim,
            config.hidden_dim,
            config.latent_dim,
        ) < 1:
            raise ValueError("hierarchical head dimensions must be positive")
        if config.structure_dim < 0 or config.membrane_dim < 0:
            raise ValueError("context dimensions cannot be negative")
        self.config = config
        self.center_index = config.window_size // 2
        self.position_embedding = nn.Parameter(
            torch.empty(config.window_size, config.state_dim)
        )
        nn.init.normal_(self.position_embedding, std=0.02)
        self.residue_projection = nn.Sequential(
            nn.LayerNorm(config.embedding_dim),
            nn.Linear(config.embedding_dim, config.state_dim),
            nn.GELU(),
        )
        self.global_projection = nn.Sequential(
            nn.LayerNorm(config.embedding_dim),
            nn.Linear(config.embedding_dim, config.state_dim),
            nn.GELU(),
        )
        self.state_projection = nn.Sequential(
            nn.LayerNorm(4 * config.state_dim),
            nn.Linear(4 * config.state_dim, 2 * config.state_dim),
            nn.GELU(),
            nn.Linear(2 * config.state_dim, config.state_dim),
        )
        static_input_dim = config.structure_dim + config.membrane_dim + 1
        self.context_projection = nn.Sequential(
            nn.LayerNorm(static_input_dim),
            nn.Linear(static_input_dim, config.context_dim),
            nn.GELU(),
            nn.Linear(config.context_dim, config.state_dim),
        )
        odd_dim = 4 * config.state_dim
        self.directional_encoder = nn.Sequential(
            nn.LayerNorm(odd_dim),
            nn.Linear(odd_dim, config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, config.latent_dim),
            nn.GELU(),
            nn.LayerNorm(config.latent_dim),
        )
        self.ddg_output = nn.Linear(config.latent_dim, 1, bias=False)
        self.dtm_output = nn.Linear(config.latent_dim, 1, bias=False)
        self.retrieval_output = nn.Linear(config.latent_dim, 1, bias=False)

    def _validate_state(
        self,
        window: torch.Tensor,
        window_mask: torch.Tensor,
        global_mean: torch.Tensor,
    ) -> None:
        expected_window = (
            window.shape[0],
            self.config.window_size,
            self.config.embedding_dim,
        )
        if window.ndim != 3 or tuple(window.shape) != expected_window:
            raise ValueError(
                "hierarchical windows must have shape "
                f"[batch, {self.config.window_size}, {self.config.embedding_dim}]"
            )
        if window_mask.shape != window.shape[:2] or window_mask.dtype != torch.bool:
            raise ValueError("hierarchical window mask must be boolean [batch, window]")
        if global_mean.shape != (window.shape[0], self.config.embedding_dim):
            raise ValueError("hierarchical global-mean dimension mismatch")
        if not torch.all(window_mask[:, self.center_index]):
            raise ValueError("every hierarchical window must contain its center residue")

    def encode_state(
        self,
        window: torch.Tensor,
        window_mask: torch.Tensor,
        global_mean: torch.Tensor,
    ) -> torch.Tensor:
        """Encode one complete sequence state without mixing mutation direction."""

        self._validate_state(window, window_mask, global_mean)
        tokens = self.residue_projection(window)
        tokens = tokens + self.position_embedding.unsqueeze(0)
        mask = window_mask.unsqueeze(-1)
        masked = tokens * mask
        count = mask.sum(dim=1).clamp_min(1)
        mean = masked.sum(dim=1) / count
        maximum = tokens.masked_fill(~mask, -torch.inf).max(dim=1).values
        center = tokens[:, self.center_index]
        global_state = self.global_projection(global_mean)
        return self.state_projection(
            torch.cat([center, mean, maximum, global_state], dim=-1)
        )

    def _context(
        self,
        batch_size: int,
        *,
        structure: torch.Tensor | None,
        structure_mask: torch.Tensor | None,
        membrane: torch.Tensor | None,
        reference: torch.Tensor,
    ) -> torch.Tensor:
        if structure is None:
            structure = reference.new_zeros((batch_size, self.config.structure_dim))
        if structure.shape != (batch_size, self.config.structure_dim):
            raise ValueError("ProteinMPNN structure feature dimension mismatch")
        if structure_mask is None:
            structure_mask = reference.new_zeros((batch_size, 1))
        elif structure_mask.ndim == 1:
            structure_mask = structure_mask.unsqueeze(-1)
        if structure_mask.shape != (batch_size, 1):
            raise ValueError("structure mask must have shape [batch] or [batch, 1]")
        if membrane is None:
            membrane = reference.new_zeros((batch_size, self.config.membrane_dim))
        if membrane.shape != (batch_size, self.config.membrane_dim):
            raise ValueError("membrane/topology feature dimension mismatch")
        static = torch.cat(
            [
                structure.to(dtype=reference.dtype),
                membrane.to(dtype=reference.dtype),
                structure_mask.to(dtype=reference.dtype),
            ],
            dim=-1,
        )
        return self.context_projection(static)

    def latent(
        self,
        wt_window: torch.Tensor,
        mutant_window: torch.Tensor,
        window_mask: torch.Tensor,
        wt_global: torch.Tensor,
        mutant_global: torch.Tensor,
        *,
        structure: torch.Tensor | None = None,
        structure_mask: torch.Tensor | None = None,
        membrane: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if mutant_window.shape != wt_window.shape:
            raise ValueError("WT and mutant hierarchy windows must have equal shape")
        if mutant_global.shape != wt_global.shape:
            raise ValueError("WT and mutant global states must have equal shape")
        wt_state = self.encode_state(wt_window, window_mask, wt_global)
        mutant_state = self.encode_state(
            mutant_window, window_mask, mutant_global
        )
        difference = mutant_state - wt_state
        symmetric = 0.5 * (mutant_state + wt_state)
        context = self._context(
            len(difference),
            structure=structure,
            structure_mask=structure_mask,
            membrane=membrane,
            reference=difference,
        )
        odd = torch.cat(
            [
                difference,
                difference * symmetric,
                difference * torch.sigmoid(context),
                difference * torch.tanh(context),
            ],
            dim=-1,
        )
        return 0.5 * (
            self.directional_encoder(odd) - self.directional_encoder(-odd)
        )

    def predict_heads(
        self,
        wt_window: torch.Tensor,
        mutant_window: torch.Tensor,
        window_mask: torch.Tensor,
        wt_global: torch.Tensor,
        mutant_global: torch.Tensor,
        *,
        structure: torch.Tensor | None = None,
        structure_mask: torch.Tensor | None = None,
        membrane: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        latent = self.latent(
            wt_window,
            mutant_window,
            window_mask,
            wt_global,
            mutant_global,
            structure=structure,
            structure_mask=structure_mask,
            membrane=membrane,
        )
        return {
            "ddg": self.ddg_output(latent).squeeze(-1),
            "dtm": self.dtm_output(latent).squeeze(-1),
            "retrieval": self.retrieval_output(latent).squeeze(-1),
        }

    def forward(
        self,
        wt_window: torch.Tensor,
        mutant_window: torch.Tensor,
        window_mask: torch.Tensor,
        wt_global: torch.Tensor,
        mutant_global: torch.Tensor,
        *,
        structure: torch.Tensor | None = None,
        structure_mask: torch.Tensor | None = None,
        membrane: torch.Tensor | None = None,
        task: str = "ddg",
    ) -> torch.Tensor:
        outputs = self.predict_heads(
            wt_window,
            mutant_window,
            window_mask,
            wt_global,
            mutant_global,
            structure=structure,
            structure_mask=structure_mask,
            membrane=membrane,
        )
        if task not in outputs:
            raise ValueError(f"unknown hierarchical prediction task {task!r}")
        return outputs[task]

    def config_dict(self) -> dict[str, object]:
        return asdict(self.config)


class HierarchicalDirectionalEnsemble(nn.Module):
    """Average signed hierarchical outputs from independent members."""

    def __init__(
        self,
        members: list[HierarchicalDirectionalMutationHead],
    ) -> None:
        super().__init__()
        if not members:
            raise ValueError("hierarchical ensemble requires at least one member")
        config = members[0].config
        if any(member.config != config for member in members[1:]):
            raise ValueError("hierarchical ensemble member configurations differ")
        self.members = nn.ModuleList(members)
        self.config = config

    def latent(self, *args: torch.Tensor, **kwargs: torch.Tensor) -> torch.Tensor:
        return self.members[0].latent(*args, **kwargs)

    def predict_heads(
        self,
        *args: torch.Tensor,
        **kwargs: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        member_outputs = [
            member.predict_heads(*args, **kwargs) for member in self.members
        ]
        return {
            task: torch.stack([output[task] for output in member_outputs]).mean(0)
            for task in ("ddg", "dtm", "retrieval")
        }

    def forward(
        self,
        *args: torch.Tensor,
        task: str = "ddg",
        **kwargs: torch.Tensor,
    ) -> torch.Tensor:
        outputs = self.predict_heads(*args, **kwargs)
        if task not in outputs:
            raise ValueError(f"unknown hierarchical prediction task {task!r}")
        return outputs[task]

    def config_dict(self) -> dict[str, object]:
        return asdict(self.config)


@dataclass(frozen=True)
class StatePotentialConfig:
    """Dimensions for a WT-conditioned, all-amino-acid energy head."""

    embedding_dim: int = 1152
    window_size: int = 9
    structure_dim: int = 128
    membrane_dim: int = 8
    state_dim: int = 192
    amino_acid_dim: int = 64
    hidden_dim: int = 256
    latent_dim: int = 128
    absolute_stability_head: bool = False


def _amino_acid_descriptors() -> torch.Tensor:
    """Return compact physicochemical priors in ``AMINO_ACIDS`` order.

    The first two columns are normalized Kyte-Doolittle hydropathy and
    approximate side-chain volume.  The remaining columns encode charge and
    broad residue classes.  Identity remains learnable; these fixed values
    provide a low-data interpolation prior, particularly for membrane sites.
    """

    hydropathy = {
        "A": 1.8,
        "C": 2.5,
        "D": -3.5,
        "E": -3.5,
        "F": 2.8,
        "G": -0.4,
        "H": -3.2,
        "I": 4.5,
        "K": -3.9,
        "L": 3.8,
        "M": 1.9,
        "N": -3.5,
        "P": -1.6,
        "Q": -3.5,
        "R": -4.5,
        "S": -0.8,
        "T": -0.7,
        "V": 4.2,
        "W": -0.9,
        "Y": -1.3,
    }
    volume = {
        "A": 88.6,
        "C": 108.5,
        "D": 111.1,
        "E": 138.4,
        "F": 189.9,
        "G": 60.1,
        "H": 153.2,
        "I": 166.7,
        "K": 168.6,
        "L": 166.7,
        "M": 162.9,
        "N": 114.1,
        "P": 112.7,
        "Q": 143.8,
        "R": 173.4,
        "S": 89.0,
        "T": 116.1,
        "V": 140.0,
        "W": 227.8,
        "Y": 193.6,
    }
    charged = {"D": -1.0, "E": -1.0, "K": 1.0, "R": 1.0}
    polar = set("CDEHKNQRSTY")
    aromatic = set("FHWY")
    aliphatic = set("AILMV")
    sulfur = set("CM")
    rows = []
    for amino_acid in AMINO_ACIDS:
        rows.append(
            [
                hydropathy[amino_acid] / 4.5,
                (volume[amino_acid] - 140.0) / 50.0,
                charged.get(amino_acid, 0.0),
                float(amino_acid in polar),
                float(amino_acid in aromatic),
                float(amino_acid in aliphatic),
                float(amino_acid == "G"),
                float(amino_acid == "P"),
                float(amino_acid in sulfur),
            ]
        )
    return torch.tensor(rows, dtype=torch.float32)


class StatePotentialMutationHead(nn.Module):
    """Score all 20 amino-acid states from one frozen WT representation.

    A mutation is the difference between two state potentials,
    ``phi(context, mutant) - phi(context, WT)``.  This makes self-mutations
    exactly zero and reversal on the same background exactly antisymmetric,
    while allowing a complete saturation scan from one ESM-C encoding.
    """

    def __init__(
        self,
        config: StatePotentialConfig = StatePotentialConfig(),
    ) -> None:
        super().__init__()
        if config.window_size < 1 or config.window_size % 2 != 1:
            raise ValueError("state-potential window size must be a positive odd number")
        if min(
            config.embedding_dim,
            config.state_dim,
            config.amino_acid_dim,
            config.hidden_dim,
            config.latent_dim,
        ) < 1:
            raise ValueError("state-potential dimensions must be positive")
        self.config = config
        self.center_index = config.window_size // 2
        self.position_embedding = nn.Parameter(
            torch.empty(config.window_size, config.state_dim)
        )
        nn.init.normal_(self.position_embedding, std=0.02)
        self.residue_projection = nn.Sequential(
            nn.LayerNorm(config.embedding_dim),
            nn.Linear(config.embedding_dim, config.state_dim),
            nn.GELU(),
        )
        self.global_projection = nn.Sequential(
            nn.LayerNorm(config.embedding_dim),
            nn.Linear(config.embedding_dim, config.state_dim),
            nn.GELU(),
        )
        self.state_projection = nn.Sequential(
            nn.LayerNorm(4 * config.state_dim),
            nn.Linear(4 * config.state_dim, 2 * config.state_dim),
            nn.GELU(),
            nn.Linear(2 * config.state_dim, config.state_dim),
        )
        static_dim = config.structure_dim + config.membrane_dim + 1
        self.static_projection = nn.Sequential(
            nn.LayerNorm(static_dim),
            nn.Linear(static_dim, config.state_dim),
            nn.GELU(),
            nn.Linear(config.state_dim, config.state_dim),
        )
        self.context_projection = nn.Sequential(
            nn.LayerNorm(4 * config.state_dim),
            nn.Linear(4 * config.state_dim, 2 * config.state_dim),
            nn.GELU(),
            nn.Linear(2 * config.state_dim, config.state_dim),
            nn.LayerNorm(config.state_dim),
        )
        descriptors = _amino_acid_descriptors()
        self.register_buffer("amino_acid_descriptors", descriptors)
        self.amino_acid_embedding = nn.Embedding(
            len(AMINO_ACIDS), config.amino_acid_dim
        )
        self.descriptor_projection = nn.Sequential(
            nn.LayerNorm(descriptors.shape[1]),
            nn.Linear(descriptors.shape[1], config.amino_acid_dim),
            nn.GELU(),
        )
        self.context_to_hidden = nn.Linear(config.state_dim, config.hidden_dim)
        self.context_interaction = nn.Linear(config.state_dim, config.hidden_dim)
        self.amino_acid_affine = nn.Linear(
            config.amino_acid_dim, 2 * config.hidden_dim
        )
        self.amino_acid_interaction = nn.Linear(
            config.amino_acid_dim, config.hidden_dim
        )
        self.potential_encoder = nn.Sequential(
            nn.LayerNorm(config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, config.latent_dim),
            nn.GELU(),
            nn.LayerNorm(config.latent_dim),
        )
        self.potential_output = nn.Linear(config.latent_dim, 1, bias=False)
        if config.absolute_stability_head:
            self.absolute_baseline = nn.Sequential(
                nn.LayerNorm(config.embedding_dim),
                nn.Linear(config.embedding_dim, config.state_dim),
                nn.GELU(),
                nn.Linear(config.state_dim, 1),
            )
        else:
            self.absolute_baseline = None

    def _validate_state(
        self,
        wt_window: torch.Tensor,
        window_mask: torch.Tensor,
        wt_global: torch.Tensor,
    ) -> None:
        expected = (
            wt_window.shape[0],
            self.config.window_size,
            self.config.embedding_dim,
        )
        if wt_window.ndim != 3 or tuple(wt_window.shape) != expected:
            raise ValueError(
                "state-potential windows must have shape "
                f"[batch, {self.config.window_size}, {self.config.embedding_dim}]"
            )
        if window_mask.shape != wt_window.shape[:2] or window_mask.dtype != torch.bool:
            raise ValueError("state-potential mask must be boolean [batch, window]")
        if wt_global.shape != (len(wt_window), self.config.embedding_dim):
            raise ValueError("state-potential global-mean dimension mismatch")
        if not torch.all(window_mask[:, self.center_index]):
            raise ValueError("every state-potential window must contain its center")

    def encode_context(
        self,
        wt_window: torch.Tensor,
        window_mask: torch.Tensor,
        wt_global: torch.Tensor,
        *,
        structure: torch.Tensor | None = None,
        structure_mask: torch.Tensor | None = None,
        membrane: torch.Tensor | None = None,
    ) -> torch.Tensor:
        self._validate_state(wt_window, window_mask, wt_global)
        tokens = self.residue_projection(wt_window)
        tokens = tokens + self.position_embedding.unsqueeze(0)
        mask = window_mask.unsqueeze(-1)
        masked = tokens * mask
        count = mask.sum(dim=1).clamp_min(1)
        mean = masked.sum(dim=1) / count
        maximum = tokens.masked_fill(~mask, -torch.inf).max(dim=1).values
        center = tokens[:, self.center_index]
        global_state = self.global_projection(wt_global)
        state = self.state_projection(
            torch.cat([center, mean, maximum, global_state], dim=-1)
        )
        batch_size = len(state)
        if structure is None:
            structure = state.new_zeros((batch_size, self.config.structure_dim))
        if structure.shape != (batch_size, self.config.structure_dim):
            raise ValueError("state-potential structure feature dimension mismatch")
        if structure_mask is None:
            structure_mask = state.new_zeros((batch_size, 1))
        elif structure_mask.ndim == 1:
            structure_mask = structure_mask.unsqueeze(-1)
        if structure_mask.shape != (batch_size, 1):
            raise ValueError("state-potential structure mask has invalid shape")
        if membrane is None:
            membrane = state.new_zeros((batch_size, self.config.membrane_dim))
        if membrane.shape != (batch_size, self.config.membrane_dim):
            raise ValueError("state-potential membrane feature dimension mismatch")
        static = self.static_projection(
            torch.cat(
                [
                    structure.to(dtype=state.dtype),
                    membrane.to(dtype=state.dtype),
                    structure_mask.to(dtype=state.dtype),
                ],
                dim=-1,
            )
        )
        return self.context_projection(
            torch.cat(
                [
                    state,
                    static,
                    state * torch.sigmoid(static),
                    state * torch.tanh(static),
                ],
                dim=-1,
            )
        )

    def all_candidate_latents(
        self,
        wt_window: torch.Tensor,
        window_mask: torch.Tensor,
        wt_global: torch.Tensor,
        *,
        structure: torch.Tensor | None = None,
        structure_mask: torch.Tensor | None = None,
        membrane: torch.Tensor | None = None,
    ) -> torch.Tensor:
        context = self.encode_context(
            wt_window,
            window_mask,
            wt_global,
            structure=structure,
            structure_mask=structure_mask,
            membrane=membrane,
        )
        descriptors = self.amino_acid_descriptors.to(dtype=context.dtype)
        amino_acid = self.amino_acid_embedding.weight + self.descriptor_projection(
            descriptors
        )
        scale, shift = self.amino_acid_affine(amino_acid).chunk(2, dim=-1)
        base = self.context_to_hidden(context).unsqueeze(1)
        interaction = self.context_interaction(context).unsqueeze(1)
        hidden = (
            base * (1.0 + torch.tanh(scale).unsqueeze(0))
            + shift.unsqueeze(0)
            + interaction
            * torch.tanh(self.amino_acid_interaction(amino_acid)).unsqueeze(0)
        )
        return self.potential_encoder(hidden)

    def all_potentials(
        self,
        wt_window: torch.Tensor,
        window_mask: torch.Tensor,
        wt_global: torch.Tensor,
        *,
        structure: torch.Tensor | None = None,
        structure_mask: torch.Tensor | None = None,
        membrane: torch.Tensor | None = None,
    ) -> torch.Tensor:
        latent = self.all_candidate_latents(
            wt_window,
            window_mask,
            wt_global,
            structure=structure,
            structure_mask=structure_mask,
            membrane=membrane,
        )
        potential = self.potential_output(latent).squeeze(-1)
        return potential - potential.mean(dim=-1, keepdim=True)

    @staticmethod
    def _gather(values: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
        if index.ndim != 1 or index.shape[0] != values.shape[0]:
            raise ValueError("amino-acid indices must have shape [batch]")
        if index.dtype != torch.long:
            raise ValueError("amino-acid indices must use torch.long")
        return values.gather(1, index.unsqueeze(-1)).squeeze(-1)

    def latent(
        self,
        wt_window: torch.Tensor,
        window_mask: torch.Tensor,
        wt_global: torch.Tensor,
        wt_amino_acid: torch.Tensor,
        mutant_amino_acid: torch.Tensor,
        *,
        structure: torch.Tensor | None = None,
        structure_mask: torch.Tensor | None = None,
        membrane: torch.Tensor | None = None,
    ) -> torch.Tensor:
        candidates = self.all_candidate_latents(
            wt_window,
            window_mask,
            wt_global,
            structure=structure,
            structure_mask=structure_mask,
            membrane=membrane,
        )
        batch = torch.arange(len(candidates), device=candidates.device)
        return (
            candidates[batch, mutant_amino_acid]
            - candidates[batch, wt_amino_acid]
        )

    def forward(
        self,
        wt_window: torch.Tensor,
        window_mask: torch.Tensor,
        wt_global: torch.Tensor,
        wt_amino_acid: torch.Tensor,
        mutant_amino_acid: torch.Tensor,
        *,
        structure: torch.Tensor | None = None,
        structure_mask: torch.Tensor | None = None,
        membrane: torch.Tensor | None = None,
    ) -> torch.Tensor:
        potential = self.all_potentials(
            wt_window,
            window_mask,
            wt_global,
            structure=structure,
            structure_mask=structure_mask,
            membrane=membrane,
        )
        return self._gather(potential, mutant_amino_acid) - self._gather(
            potential, wt_amino_acid
        )

    def predict_thermodynamic_state(
        self,
        wt_window: torch.Tensor,
        window_mask: torch.Tensor,
        wt_global: torch.Tensor,
        wt_amino_acid: torch.Tensor,
        mutant_amino_acid: torch.Tensor,
        *,
        structure: torch.Tensor | None = None,
        structure_mask: torch.Tensor | None = None,
        membrane: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Predict a thermodynamically consistent WT/mutant state triple.

        Project ΔΔG is defined as ``ΔG_wt - ΔG_mutant`` so negative values
        indicate stabilization. The absolute predictions therefore satisfy
        ``mutant_dG - wt_dG == -ddG`` by construction.
        """

        if self.absolute_baseline is None:
            raise RuntimeError("absolute-stability head is not enabled")
        potential = self.all_potentials(
            wt_window,
            window_mask,
            wt_global,
            structure=structure,
            structure_mask=structure_mask,
            membrane=membrane,
        )
        wt_potential = self._gather(potential, wt_amino_acid)
        mutant_potential = self._gather(potential, mutant_amino_acid)
        ddg = mutant_potential - wt_potential
        baseline = self.absolute_baseline(wt_global).squeeze(-1)
        return {
            "ddg": ddg,
            "wt_absolute_stability": baseline,
            "mutant_absolute_stability": baseline - ddg,
        }

    def predict_heads(self, *args: torch.Tensor, **kwargs: torch.Tensor) -> dict[str, torch.Tensor]:
        ddg = self.forward(*args, **kwargs)
        return {"ddg": ddg, "retrieval": -ddg}

    def config_dict(self) -> dict[str, object]:
        return asdict(self.config)


class StatePotentialEnsemble(nn.Module):
    """Average independently trained WT-conditioned state potentials."""

    def __init__(self, members: list[StatePotentialMutationHead]) -> None:
        super().__init__()
        if not members:
            raise ValueError("state-potential ensemble requires at least one member")
        config = members[0].config
        if any(member.config != config for member in members[1:]):
            raise ValueError("state-potential ensemble member configurations differ")
        self.members = nn.ModuleList(members)
        self.config = config

    def all_potentials(self, *args: torch.Tensor, **kwargs: torch.Tensor) -> torch.Tensor:
        return torch.stack(
            [member.all_potentials(*args, **kwargs) for member in self.members]
        ).mean(dim=0)

    def latent(self, *args: torch.Tensor, **kwargs: torch.Tensor) -> torch.Tensor:
        return torch.stack(
            [member.latent(*args, **kwargs) for member in self.members]
        ).mean(dim=0)

    def forward(self, *args: torch.Tensor, **kwargs: torch.Tensor) -> torch.Tensor:
        return torch.stack(
            [member(*args, **kwargs) for member in self.members]
        ).mean(dim=0)

    def predict_thermodynamic_state(
        self, *args: torch.Tensor, **kwargs: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        outputs = [
            member.predict_thermodynamic_state(*args, **kwargs)
            for member in self.members
        ]
        return {
            key: torch.stack([output[key] for output in outputs]).mean(dim=0)
            for key in (
                "ddg",
                "wt_absolute_stability",
                "mutant_absolute_stability",
            )
        }

    def predict_heads(self, *args: torch.Tensor, **kwargs: torch.Tensor) -> dict[str, torch.Tensor]:
        ddg = self.forward(*args, **kwargs)
        return {"ddg": ddg, "retrieval": -ddg}

    def config_dict(self) -> dict[str, object]:
        return asdict(self.config)


@dataclass(frozen=True)
class DirectionalAssayConfig:
    latent_dim: int = 128
    membrane_dim: int = 8
    context_dim: int = 64
    hidden_dim: int = 192
    output_latent_dim: int = 64


class DirectionalAssayHead(nn.Module):
    """Small exact-odd adapter for a distinct experimental endpoint."""

    def __init__(
        self,
        config: DirectionalAssayConfig = DirectionalAssayConfig(),
    ) -> None:
        super().__init__()
        self.config = config
        self.context = nn.Sequential(
            nn.LayerNorm(config.membrane_dim),
            nn.Linear(config.membrane_dim, config.context_dim),
            nn.GELU(),
            nn.Linear(config.context_dim, config.latent_dim),
        )
        input_dim = 3 * config.latent_dim + 1
        self.encoder = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, config.output_latent_dim),
            nn.GELU(),
            nn.LayerNorm(config.output_latent_dim),
        )
        self.output = nn.Linear(config.output_latent_dim, 1, bias=False)

    def latent(
        self,
        base_latent: torch.Tensor,
        base_ddg: torch.Tensor,
        membrane: torch.Tensor,
    ) -> torch.Tensor:
        if base_latent.ndim != 2 or base_latent.shape[1] != self.config.latent_dim:
            raise ValueError("assay base latent dimension mismatch")
        if base_ddg.shape != (len(base_latent),):
            raise ValueError("assay base ddG must have shape [batch]")
        if membrane.shape != (len(base_latent), self.config.membrane_dim):
            raise ValueError("assay membrane feature dimension mismatch")
        context = self.context(membrane)
        odd = torch.cat(
            [
                base_latent,
                base_latent * torch.sigmoid(context),
                base_latent * torch.tanh(context),
                base_ddg.unsqueeze(-1),
            ],
            dim=-1,
        )
        return 0.5 * (self.encoder(odd) - self.encoder(-odd))

    def forward(
        self,
        base_latent: torch.Tensor,
        base_ddg: torch.Tensor,
        membrane: torch.Tensor,
    ) -> torch.Tensor:
        return self.output(
            self.latent(base_latent, base_ddg, membrane)
        ).squeeze(-1)

    def config_dict(self) -> dict[str, object]:
        return asdict(self.config)


@dataclass(frozen=True)
class EpistasisConfig:
    embedding_dim: int = 1152
    element_hidden_dim: int = 512
    element_dim: int = 128
    set_hidden_dim: int = 256
    dropout: float = 0.10


class MultiMutationHead(nn.Module):
    """Additive singles plus a permutation-invariant contextual epistasis term."""

    def __init__(
        self,
        single_head: SingleMutationHead,
        config: EpistasisConfig = EpistasisConfig(),
    ) -> None:
        super().__init__()
        if single_head.config.embedding_dim != config.embedding_dim:
            raise ValueError("single and epistasis embedding dimensions differ")
        self.single_head = single_head
        self.config = config
        self.element = nn.Sequential(
            nn.LayerNorm(3 * config.embedding_dim),
            nn.Linear(3 * config.embedding_dim, config.element_hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.element_hidden_dim, config.element_dim),
            nn.GELU(),
        )
        self.set_head = nn.Sequential(
            nn.LayerNorm(3 * config.element_dim + 1),
            nn.Linear(3 * config.element_dim + 1, config.set_hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.set_hidden_dim, 1),
        )

    def freeze_single_head(self) -> None:
        for parameter in self.single_head.parameters():
            parameter.requires_grad_(False)
        self.single_head.eval()

    def forward(
        self,
        single_delta: torch.Tensor,
        joint_delta: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if single_delta.shape != joint_delta.shape or single_delta.ndim != 3:
            raise ValueError("multi-mutant deltas must both have shape [batch, mutations, dim]")
        if single_delta.shape[-1] != self.config.embedding_dim:
            raise ValueError("multi-mutant delta dimension mismatch")
        batch, mutation_count, _ = single_delta.shape
        if mask is None:
            mask = torch.ones(
                (batch, mutation_count), dtype=torch.bool, device=single_delta.device
            )
        if mask.shape != (batch, mutation_count) or mask.dtype != torch.bool:
            raise ValueError("mutation mask must be boolean [batch, mutations]")
        counts = mask.sum(dim=1)
        if torch.any(counts == 0):
            raise ValueError("each example must contain at least one mutation")

        single_predictions = self.single_head(
            single_delta.reshape(batch * mutation_count, -1)
        ).reshape(batch, mutation_count)
        additive = (single_predictions * mask).sum(dim=1)

        interaction_input = torch.cat(
            [single_delta, joint_delta, joint_delta - single_delta], dim=-1
        )
        elements = self.element(interaction_input)
        mask_values = mask.unsqueeze(-1)
        masked = elements * mask_values
        summed = masked.sum(dim=1)
        mean = summed / counts.unsqueeze(-1)
        max_values = elements.masked_fill(~mask_values, -torch.inf).max(dim=1).values
        count_feature = torch.log1p(counts.float()).unsqueeze(-1)
        epistasis = self.set_head(
            torch.cat([summed, mean, max_values, count_feature], dim=-1)
        ).squeeze(-1)
        return additive + epistasis, additive, epistasis

    def config_dict(self) -> dict[str, object]:
        return asdict(self.config)


@dataclass(frozen=True)
class HierarchicalEpistasisConfig:
    latent_dim: int = 128
    element_hidden_dim: int = 256
    element_dim: int = 96
    set_hidden_dim: int = 192


class HierarchicalMultiMutationHead(nn.Module):
    """Permutation-invariant additive-plus-epistasis v2 combination head.

    ``single_latent`` represents each WT-to-single mutation. ``joint_latent``
    represents the same site in the complete joint-mutant context.  The
    thermodynamic single predictions are summed explicitly and returned
    separately from the learned interaction correction.
    """

    def __init__(
        self,
        config: HierarchicalEpistasisConfig = HierarchicalEpistasisConfig(),
    ) -> None:
        super().__init__()
        self.config = config
        self.element = nn.Sequential(
            nn.LayerNorm(3 * config.latent_dim),
            nn.Linear(3 * config.latent_dim, config.element_hidden_dim),
            nn.GELU(),
            nn.Linear(config.element_hidden_dim, config.element_dim),
            nn.GELU(),
        )
        self.set_head = nn.Sequential(
            nn.LayerNorm(3 * config.element_dim + 1),
            nn.Linear(
                3 * config.element_dim + 1,
                config.set_hidden_dim,
            ),
            nn.GELU(),
            nn.Linear(config.set_hidden_dim, 1),
        )

    def forward(
        self,
        single_ddg: torch.Tensor,
        single_latent: torch.Tensor,
        joint_latent: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if single_latent.shape != joint_latent.shape or single_latent.ndim != 3:
            raise ValueError(
                "hierarchical mutation latents must have shape "
                "[batch, mutations, latent]"
            )
        batch, mutation_count, latent_dim = single_latent.shape
        if latent_dim != self.config.latent_dim:
            raise ValueError("hierarchical epistasis latent dimension mismatch")
        if single_ddg.shape != (batch, mutation_count):
            raise ValueError("single ddG values must have shape [batch, mutations]")
        if mask is None:
            mask = torch.ones(
                (batch, mutation_count),
                dtype=torch.bool,
                device=single_latent.device,
            )
        if mask.shape != (batch, mutation_count) or mask.dtype != torch.bool:
            raise ValueError("mutation mask must be boolean [batch, mutations]")
        counts = mask.sum(dim=1)
        if torch.any(counts == 0):
            raise ValueError("each example must contain at least one mutation")

        additive = (single_ddg * mask).sum(dim=1)
        interactions = self.element(
            torch.cat(
                [
                    single_latent,
                    joint_latent,
                    joint_latent - single_latent,
                ],
                dim=-1,
            )
        )
        mask_values = mask.unsqueeze(-1)
        masked = interactions * mask_values
        summed = masked.sum(dim=1)
        mean = summed / counts.unsqueeze(-1)
        maximum = interactions.masked_fill(~mask_values, -torch.inf).max(dim=1).values
        count_feature = torch.log1p(counts.float()).unsqueeze(-1)
        epistasis = self.set_head(
            torch.cat([summed, mean, maximum, count_feature], dim=-1)
        ).squeeze(-1)
        return additive + epistasis, additive, epistasis

    def config_dict(self) -> dict[str, object]:
        return asdict(self.config)
