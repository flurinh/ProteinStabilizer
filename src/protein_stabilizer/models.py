"""Small trainable heads over frozen ESM-C mutation deltas."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import nn


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
