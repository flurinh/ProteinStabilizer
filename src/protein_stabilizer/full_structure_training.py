"""Training and sealed evaluation for the full ESM-C/ProteinMPNN model."""

from __future__ import annotations

import copy
import csv
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import h5py
import numpy as np
import torch
from torch import nn

from .embeddings import file_sha256
from .full_structure import (
    FULL_STRUCTURE_MODEL_SCHEMA,
    FullStructureConfig,
    FullStructureEpistasisConfig,
    FullStructureEpistasisHead,
    FullStructureStateModel,
    MaskedProteinMPNNEncoder,
    load_trainable_proteinmpnn,
)
from .full_structure_data import FULL_STRUCTURE_DATA_SCHEMA
from .training import (
    balanced_stability_weights,
    regression_metrics,
    set_reproducible_seed,
    stabilizer_retrieval_metrics,
)
from .v2_training import _ranking_pairs


FULL_STRUCTURE_CHECKPOINT_SCHEMA = (
    "protein-stabilizer.full-structure-checkpoint.v1"
)
FULL_STRUCTURE_TRAINING_SCHEMA = (
    "protein-stabilizer.full-structure-training.v1"
)
FULL_STRUCTURE_OUTER_EVALUATION_SCHEMA = (
    "protein-stabilizer.full-structure-outer-evaluation.v1"
)


@dataclass(frozen=True)
class FullStructureObjective:
    stabilizer_threshold: float = -0.5
    neutral_upper_threshold: float = 0.5
    huber_delta: float = 1.0
    regression_weight: float = 1.0
    retrieval_weight: float = 0.10
    ranking_weight: float = 0.10
    minimum_ranking_gap: float = 0.25
    ranking_pairs_per_batch: int = 256
    maximum_bin_weight: float = 6.0


@dataclass
class MutationArrays:
    protein_index: np.ndarray
    position: np.ndarray
    wt_amino_acid: np.ndarray
    mutant_amino_acid: np.ndarray
    target: np.ndarray
    source_partition: np.ndarray


@dataclass
class FullStructureArrays:
    protein_id: np.ndarray
    sequence: np.ndarray
    family_cluster: np.ndarray
    split: np.ndarray
    length: np.ndarray
    esm_residue: np.ndarray
    esm_global: np.ndarray
    coordinates: np.ndarray
    sequence_tokens: np.ndarray
    sequence_mask: np.ndarray
    structure_mask: np.ndarray
    singles: MutationArrays
    doubles: MutationArrays
    provenance: dict[str, object]


def _load_mutations(group: h5py.Group) -> MutationArrays:
    return MutationArrays(
        protein_index=np.asarray(group["protein_index"], dtype=np.int64),
        position=np.asarray(group["position"], dtype=np.int64),
        wt_amino_acid=np.asarray(
            group["wt_amino_acid"], dtype=np.int64
        ),
        mutant_amino_acid=np.asarray(
            group["mutant_amino_acid"], dtype=np.int64
        ),
        target=np.asarray(group["target"], dtype=np.float32),
        source_partition=np.asarray(group["source_partition"].asstr()[:]),
    )


def load_full_structure_arrays(path: Path) -> FullStructureArrays:
    """Load the compact per-protein bank and validate mutation references."""

    path = Path(path).resolve()
    with h5py.File(path, "r") as handle:
        if handle.attrs.get("schema") != FULL_STRUCTURE_DATA_SCHEMA:
            raise RuntimeError("full-structure data schema mismatch")
        proteins = handle["proteins"]
        values = FullStructureArrays(
            protein_id=np.asarray(proteins["protein_id"].asstr()[:]),
            sequence=np.asarray(proteins["sequence"].asstr()[:]),
            family_cluster=np.asarray(
                proteins["family_cluster"].asstr()[:]
            ),
            split=np.asarray(proteins["split"], dtype=np.int8),
            length=np.asarray(proteins["length"], dtype=np.int32),
            esm_residue=np.asarray(
                proteins["esm_residue"], dtype=np.float32
            ),
            esm_global=np.asarray(
                proteins["esm_global"], dtype=np.float32
            ),
            coordinates=np.asarray(
                proteins["coordinates"], dtype=np.float32
            ),
            sequence_tokens=np.asarray(
                proteins["sequence_tokens"], dtype=np.int64
            ),
            sequence_mask=np.asarray(
                proteins["sequence_mask"], dtype=bool
            ),
            structure_mask=np.asarray(
                proteins["structure_mask"], dtype=bool
            ),
            singles=_load_mutations(handle["singles"]),
            doubles=_load_mutations(handle["doubles"]),
            provenance={
                "path": str(path),
                "sha256": file_sha256(path),
                "embedding_source": str(
                    handle.attrs["embedding_source"]
                ),
                "embedding_source_sha256": str(
                    handle.attrs["embedding_source_sha256"]
                ),
                "embedding_provenance": json.loads(
                    str(handle.attrs["embedding_provenance"])
                ),
                "structure_archive": str(
                    handle.attrs["structure_archive"]
                ),
                "structure_archive_sha256": str(
                    handle.attrs["structure_archive_sha256"]
                ),
                "proteinmpnn_provenance": json.loads(
                    str(handle.attrs["proteinmpnn_provenance"])
                ),
                "source_sha256": json.loads(
                    str(handle.attrs["source_sha256"])
                ),
                "split_policy": json.loads(
                    str(handle.attrs["split_policy"])
                ),
            },
        )
    protein_count, maximum_length, dimension = values.esm_residue.shape
    if values.esm_global.shape != (protein_count, dimension):
        raise RuntimeError("global ESM-C tensor shape mismatch")
    if values.coordinates.shape != (
        protein_count,
        maximum_length,
        4,
        3,
    ):
        raise RuntimeError("coordinate tensor shape mismatch")
    if values.sequence_tokens.shape != (protein_count, maximum_length):
        raise RuntimeError("sequence-token tensor shape mismatch")
    if values.sequence_mask.shape != (protein_count, maximum_length):
        raise RuntimeError("sequence-mask tensor shape mismatch")
    if np.any(values.structure_mask & ~values.sequence_mask):
        raise RuntimeError("structure mask includes sequence padding")
    for name, mutations, count in (
        ("singles", values.singles, 1),
        ("doubles", values.doubles, 2),
    ):
        if mutations.position.shape != (len(mutations.target), count):
            raise RuntimeError(f"{name} position shape mismatch")
        if np.any(mutations.protein_index < 0) or np.any(
            mutations.protein_index >= protein_count
        ):
            raise RuntimeError(f"{name} contains invalid protein indices")
        lengths = values.length[mutations.protein_index, None]
        if np.any(mutations.position < 0) or np.any(
            mutations.position >= lengths
        ):
            raise RuntimeError(f"{name} contains invalid positions")
        sequence_states = values.sequence_tokens[
            mutations.protein_index[:, None], mutations.position
        ]
        if not np.array_equal(
            sequence_states, mutations.wt_amino_acid
        ):
            raise RuntimeError(
                f"{name} WT amino acids do not match protein sequences"
            )
    return values


def _strict_fp32(device: torch.device) -> None:
    if device.type == "cuda":
        torch.set_float32_matmul_precision("highest")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False


def _create_state_model(
    proteinmpnn_repository: Path,
    *,
    esm_dimension: int,
    use_structure: bool,
    device: torch.device,
) -> tuple[FullStructureStateModel, dict[str, object]]:
    proteinmpnn, module, provenance = load_trainable_proteinmpnn(
        proteinmpnn_repository
    )
    encoder = MaskedProteinMPNNEncoder(proteinmpnn, module)
    model = FullStructureStateModel(
        encoder,
        FullStructureConfig(
            esm_dim=esm_dimension,
            structure_dim=encoder.output_dim,
            use_structure=use_structure,
        ),
    ).to(device)
    if not use_structure:
        for parameter in model.structure_encoder.parameters():
            parameter.requires_grad_(False)
    return model, asdict(provenance)


def _protein_batch(
    model: FullStructureStateModel,
    data: FullStructureArrays,
    proteins: np.ndarray,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    floating = lambda array: torch.from_numpy(  # noqa: E731
        np.ascontiguousarray(array[proteins], dtype=np.float32)
    ).to(device)
    return model.all_potentials(
        floating(data.esm_residue),
        floating(data.esm_global),
        floating(data.coordinates),
        torch.from_numpy(
            np.ascontiguousarray(data.sequence_tokens[proteins])
        ).to(device),
        torch.from_numpy(
            np.ascontiguousarray(data.sequence_mask[proteins])
        ).to(device),
        structure_mask=torch.from_numpy(
            np.ascontiguousarray(data.structure_mask[proteins])
        ).to(device),
    )


def _row_lookup(
    mutations: MutationArrays,
    protein_count: int,
) -> tuple[np.ndarray, ...]:
    return tuple(
        np.flatnonzero(mutations.protein_index == protein)
        for protein in range(protein_count)
    )


def _batch_rows(
    lookup: tuple[np.ndarray, ...],
    proteins: np.ndarray,
) -> np.ndarray:
    selected = [lookup[int(protein)] for protein in proteins]
    selected = [rows for rows in selected if len(rows)]
    return (
        np.concatenate(selected)
        if selected
        else np.empty(0, dtype=np.int64)
    )


def _single_predictions_for_batch(
    model: FullStructureStateModel,
    potentials: torch.Tensor,
    rows: np.ndarray,
    proteins: np.ndarray,
    mutations: MutationArrays,
    device: torch.device,
) -> torch.Tensor:
    local = np.full(
        int(np.max(proteins)) + 1 if len(proteins) else 0,
        -1,
        dtype=np.int64,
    )
    local[proteins] = np.arange(len(proteins))
    protein_index = local[mutations.protein_index[rows]]
    if np.any(protein_index < 0):
        raise RuntimeError("row-to-protein batch mapping failed")
    return model.mutation_ddg(
        potentials,
        torch.from_numpy(protein_index).to(device),
        torch.from_numpy(mutations.position[rows, 0]).to(device),
        torch.from_numpy(mutations.wt_amino_acid[rows, 0]).to(device),
        torch.from_numpy(
            mutations.mutant_amino_acid[rows, 0]
        ).to(device),
    )


def _evaluate_predictions(
    target: np.ndarray,
    prediction: np.ndarray,
    protein_id: np.ndarray,
    *,
    threshold: float,
    bootstrap_seed: int = 20260723,
    bootstrap_samples: int = 2000,
) -> dict[str, object]:
    regression = regression_metrics(target, prediction)
    retrieval = stabilizer_retrieval_metrics(
        target, -prediction, threshold=threshold
    )
    proteins = sorted(set(protein_id.tolist()))
    per_protein = {
        protein: regression_metrics(
            target[protein_id == protein],
            prediction[protein_id == protein],
        )
        for protein in proteins
    }
    rng = np.random.default_rng(bootstrap_seed)
    bootstrap = np.empty(bootstrap_samples, dtype=np.float64)
    indices = {
        protein: np.flatnonzero(protein_id == protein)
        for protein in proteins
    }
    for sample in range(bootstrap_samples):
        drawn = rng.choice(proteins, size=len(proteins), replace=True)
        selected = np.concatenate([indices[protein] for protein in drawn])
        bootstrap[sample] = np.mean(
            np.abs(prediction[selected] - target[selected])
        )
    finite_spearman = [
        float(metrics["spearman"])
        for metrics in per_protein.values()
        if np.isfinite(float(metrics["spearman"]))
    ]
    return {
        "regression": regression,
        "retrieval": retrieval,
        "macro_protein_mae": float(
            np.mean(
                [
                    float(metrics["mae"])
                    for metrics in per_protein.values()
                ]
            )
        ),
        "macro_protein_spearman": (
            float(np.mean(finite_spearman))
            if finite_spearman
            else math.nan
        ),
        "protein_cluster_bootstrap_mae_95_ci": [
            float(np.quantile(bootstrap, 0.025)),
            float(np.quantile(bootstrap, 0.975)),
        ],
        "protein_count": len(proteins),
        "per_protein": per_protein,
    }


def _predict_single_rows(
    model: FullStructureStateModel,
    data: FullStructureArrays,
    row_indices: np.ndarray,
    *,
    protein_batch_size: int,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    prediction = np.full(len(data.singles.target), np.nan, dtype=np.float32)
    selected_proteins = np.unique(
        data.singles.protein_index[row_indices]
    )
    requested = np.zeros(len(data.singles.target), dtype=bool)
    requested[row_indices] = True
    lookup = _row_lookup(data.singles, len(data.protein_id))
    with torch.inference_mode():
        for start in range(0, len(selected_proteins), protein_batch_size):
            proteins = selected_proteins[
                start : start + protein_batch_size
            ]
            potentials, _ = _protein_batch(
                model, data, proteins, device
            )
            rows = _batch_rows(lookup, proteins)
            rows = rows[requested[rows]]
            prediction[rows] = (
                _single_predictions_for_batch(
                    model,
                    potentials,
                    rows,
                    proteins,
                    data.singles,
                    device,
                )
                .float()
                .cpu()
                .numpy()
            )
    if np.any(~np.isfinite(prediction[row_indices])):
        raise RuntimeError("single prediction did not cover requested rows")
    return prediction[row_indices]


def _train_single_model(
    data: FullStructureArrays,
    proteinmpnn_repository: Path,
    train_proteins: np.ndarray,
    validation_proteins: np.ndarray | None,
    *,
    use_structure: bool,
    seed: int,
    epochs: int,
    minimum_epochs: int,
    patience: int,
    protein_batch_size: int,
    learning_rate: float,
    weight_decay: float,
    objective: FullStructureObjective,
    device: torch.device,
    initial_state: dict[str, torch.Tensor] | None = None,
) -> tuple[dict[str, torch.Tensor], dict[str, object]]:
    if minimum_epochs < 1 or minimum_epochs > epochs:
        raise ValueError("minimum epochs must be within the epoch budget")
    set_reproducible_seed(seed)
    model, proteinmpnn_provenance = _create_state_model(
        proteinmpnn_repository,
        esm_dimension=data.esm_residue.shape[-1],
        use_structure=use_structure,
        device=device,
    )
    if initial_state is not None:
        model.load_state_dict(initial_state)
        if use_structure:
            model.cross_gate.data.fill_(-4.0)
    train_rows = np.flatnonzero(
        np.isin(data.singles.protein_index, train_proteins)
    )
    if not len(train_rows):
        raise ValueError("single-model training partition is empty")
    weights = np.ones(len(data.singles.target), dtype=np.float32)
    weights[train_rows] = balanced_stability_weights(
        data.singles.target[train_rows],
        stabilizer_threshold=objective.stabilizer_threshold,
        neutral_upper_threshold=objective.neutral_upper_threshold,
        maximum_weight=objective.maximum_bin_weight,
    )
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    warmup = max(1, min(5, epochs // 10))
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[
            torch.optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=0.20,
                end_factor=1.0,
                total_iters=warmup,
            ),
            torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=max(1, epochs - warmup),
                eta_min=learning_rate * 0.05,
            ),
        ],
        milestones=[warmup],
    )
    lookup = _row_lookup(data.singles, len(data.protein_id))
    rng = np.random.default_rng(seed)
    huber = nn.HuberLoss(delta=objective.huber_delta, reduction="none")
    best_state: dict[str, torch.Tensor] | None = None
    best_score = math.inf
    best_epoch = 0
    stale = 0
    history: list[dict[str, object]] = []
    started = time.monotonic()

    for epoch in range(1, epochs + 1):
        model.train()
        ordered = np.asarray(train_proteins, dtype=np.int64).copy()
        rng.shuffle(ordered)
        totals = {
            "loss": 0.0,
            "regression": 0.0,
            "retrieval": 0.0,
            "ranking": 0.0,
            "rows": 0,
        }
        for start in range(0, len(ordered), protein_batch_size):
            proteins = ordered[start : start + protein_batch_size]
            rows = _batch_rows(lookup, proteins)
            if not len(rows):
                continue
            optimizer.zero_grad(set_to_none=True)
            potentials, _ = _protein_batch(model, data, proteins, device)
            prediction = _single_predictions_for_batch(
                model,
                potentials,
                rows,
                proteins,
                data.singles,
                device,
            )
            target = torch.from_numpy(data.singles.target[rows]).to(device)
            row_weight = torch.from_numpy(weights[rows]).to(device)
            regression_loss = (
                huber(prediction, target) * row_weight
            ).sum() / row_weight.sum()
            stabilizer = (
                target < objective.stabilizer_threshold
            ).to(prediction.dtype)
            retrieval_loss = nn.functional.binary_cross_entropy_with_logits(
                -prediction, stabilizer
            )
            stable, unstable = _ranking_pairs(
                data.singles.protein_index[rows],
                data.singles.target[rows],
                rng,
                count=min(
                    objective.ranking_pairs_per_batch, len(rows)
                ),
                minimum_gap=objective.minimum_ranking_gap,
            )
            if len(stable):
                ranking_loss = nn.functional.softplus(
                    prediction[
                        torch.from_numpy(stable).to(device)
                    ]
                    - prediction[
                        torch.from_numpy(unstable).to(device)
                    ]
                ).mean()
            else:
                ranking_loss = torch.zeros((), device=device)
            loss = (
                objective.regression_weight * regression_loss
                + objective.retrieval_weight * retrieval_loss
                + objective.ranking_weight * ranking_loss
            )
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            count = len(rows)
            totals["loss"] += float(loss.detach()) * count
            totals["regression"] += (
                float(regression_loss.detach()) * count
            )
            totals["retrieval"] += (
                float(retrieval_loss.detach()) * count
            )
            totals["ranking"] += float(ranking_loss.detach()) * count
            totals["rows"] += count
        scheduler.step()
        record: dict[str, object] = {
            "epoch": epoch,
            "learning_rate": float(scheduler.get_last_lr()[0]),
            "train": {
                name: float(totals[name]) / max(1, totals["rows"])
                for name in (
                    "loss",
                    "regression",
                    "retrieval",
                    "ranking",
                )
            },
        }
        if validation_proteins is not None:
            validation_rows = np.flatnonzero(
                np.isin(
                    data.singles.protein_index, validation_proteins
                )
            )
            validation_prediction = _predict_single_rows(
                model,
                data,
                validation_rows,
                protein_batch_size=protein_batch_size,
                device=device,
            )
            validation_target = data.singles.target[validation_rows]
            validation = {
                "regression": regression_metrics(
                    validation_target, validation_prediction
                ),
                "retrieval": stabilizer_retrieval_metrics(
                    validation_target,
                    -validation_prediction,
                    threshold=objective.stabilizer_threshold,
                ),
            }
            score = float(validation["regression"]["mae"])
            record["validation"] = validation
            if score < best_score - 1e-5:
                best_score = score
                best_epoch = epoch
                best_state = {
                    name: value.detach().cpu().clone()
                    for name, value in model.state_dict().items()
                }
                stale = 0
            elif epoch >= minimum_epochs:
                stale += 1
            print(
                f"single use_structure={use_structure} epoch={epoch:03d} "
                f"train={record['train']['loss']:.4f} "
                f"val_mae={score:.4f} "
                f"val_rho={validation['regression']['spearman']:.4f}",
                flush=True,
            )
        else:
            best_epoch = epoch
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
            print(
                f"single final use_structure={use_structure} "
                f"epoch={epoch:03d} train={record['train']['loss']:.4f}",
                flush=True,
            )
        history.append(record)
        if (
            validation_proteins is not None
            and epoch >= minimum_epochs
            and stale >= patience
        ):
            break
    if best_state is None:
        raise RuntimeError("single-model training produced no checkpoint")
    return best_state, {
        "use_structure": use_structure,
        "seed": seed,
        "epochs_completed": len(history),
        "best_epoch": best_epoch,
        "best_validation_mae": (
            best_score if validation_proteins is not None else None
        ),
        "elapsed_seconds": time.monotonic() - started,
        "history": history,
        "proteinmpnn_provenance": proteinmpnn_provenance,
        "warm_started": initial_state is not None,
    }


def _model_from_state(
    data: FullStructureArrays,
    proteinmpnn_repository: Path,
    *,
    use_structure: bool,
    state: dict[str, torch.Tensor],
    device: torch.device,
) -> FullStructureStateModel:
    model, _ = _create_state_model(
        proteinmpnn_repository,
        esm_dimension=data.esm_residue.shape[-1],
        use_structure=use_structure,
        device=device,
    )
    model.load_state_dict(state)
    return model


def _all_state_outputs(
    model: FullStructureStateModel,
    data: FullStructureArrays,
    *,
    protein_batch_size: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    potentials = np.zeros(
        (
            len(data.protein_id),
            data.esm_residue.shape[1],
            20,
        ),
        dtype=np.float32,
    )
    representation = np.zeros(
        (
            len(data.protein_id),
            data.esm_residue.shape[1],
            model.config.fusion_dim,
        ),
        dtype=np.float32,
    )
    model.eval()
    with torch.inference_mode():
        for start in range(
            0, len(data.protein_id), protein_batch_size
        ):
            proteins = np.arange(
                start,
                min(start + protein_batch_size, len(data.protein_id)),
                dtype=np.int64,
            )
            state, encoded = _protein_batch(
                model, data, proteins, device
            )
            potentials[proteins] = state.float().cpu().numpy()
            representation[proteins] = encoded.float().cpu().numpy()
    return potentials, representation


def _known_epistasis_targets(
    singles: MutationArrays,
    doubles: MutationArrays,
) -> tuple[np.ndarray, np.ndarray]:
    values: dict[tuple[int, int, int, int], list[float]] = {}
    for index in range(len(singles.target)):
        key = (
            int(singles.protein_index[index]),
            int(singles.position[index, 0]),
            int(singles.wt_amino_acid[index, 0]),
            int(singles.mutant_amino_acid[index, 0]),
        )
        values.setdefault(key, []).append(float(singles.target[index]))
    means = {key: float(np.mean(target)) for key, target in values.items()}
    target = np.zeros(len(doubles.target), dtype=np.float32)
    mask = np.zeros(len(doubles.target), dtype=bool)
    for index in range(len(doubles.target)):
        keys = [
            (
                int(doubles.protein_index[index]),
                int(doubles.position[index, mutation]),
                int(doubles.wt_amino_acid[index, mutation]),
                int(doubles.mutant_amino_acid[index, mutation]),
            )
            for mutation in range(2)
        ]
        if all(key in means for key in keys):
            target[index] = float(doubles.target[index]) - sum(
                means[key] for key in keys
            )
            mask[index] = True
    return target, mask


def _epistasis_forward(
    head: FullStructureEpistasisHead,
    potentials: torch.Tensor,
    representations: torch.Tensor,
    rows: np.ndarray,
    mutations: MutationArrays,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    count = mutations.position.shape[1]
    return head(
        potentials,
        representations,
        torch.from_numpy(mutations.protein_index[rows]).to(device),
        torch.from_numpy(mutations.position[rows]).to(device),
        torch.from_numpy(mutations.wt_amino_acid[rows]).to(device),
        torch.from_numpy(
            mutations.mutant_amino_acid[rows]
        ).to(device),
        torch.ones((len(rows), count), dtype=torch.bool, device=device),
    )


def _predict_double_rows(
    head: FullStructureEpistasisHead,
    potentials: np.ndarray,
    representations: np.ndarray,
    mutations: MutationArrays,
    rows: np.ndarray,
    *,
    row_batch_size: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    head.eval()
    state = torch.from_numpy(potentials).to(device)
    encoded = torch.from_numpy(representations).to(device)
    outputs: list[list[np.ndarray]] = [[], [], []]
    with torch.inference_mode():
        for start in range(0, len(rows), row_batch_size):
            selected = rows[start : start + row_batch_size]
            values = _epistasis_forward(
                head,
                state,
                encoded,
                selected,
                mutations,
                device,
            )
            for output, value in zip(outputs, values, strict=True):
                output.append(value.float().cpu().numpy())
    return tuple(
        np.concatenate(output)
        if output
        else np.empty(0, dtype=np.float32)
        for output in outputs
    )


def _train_epistasis_head(
    data: FullStructureArrays,
    potentials: np.ndarray,
    representations: np.ndarray,
    train_rows: np.ndarray,
    validation_rows: np.ndarray | None,
    *,
    seed: int,
    epochs: int,
    minimum_epochs: int,
    patience: int,
    row_batch_size: int,
    learning_rate: float,
    weight_decay: float,
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], dict[str, object]]:
    config = FullStructureEpistasisConfig(
        representation_dim=representations.shape[-1]
    )
    set_reproducible_seed(seed)
    head = FullStructureEpistasisHead(config).to(device)
    optimizer = torch.optim.AdamW(
        head.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    huber = nn.HuberLoss(delta=1.0)
    state = torch.from_numpy(potentials).to(device)
    encoded = torch.from_numpy(representations).to(device)
    known_target, known_mask = _known_epistasis_targets(
        data.singles, data.doubles
    )
    rng = np.random.default_rng(seed)
    best_state: dict[str, torch.Tensor] | None = None
    best_score = math.inf
    best_epoch = 0
    stale = 0
    history: list[dict[str, object]] = []
    started = time.monotonic()
    for epoch in range(1, epochs + 1):
        head.train()
        ordered = train_rows.copy()
        rng.shuffle(ordered)
        total_loss = 0.0
        seen = 0
        for start in range(0, len(ordered), row_batch_size):
            rows = ordered[start : start + row_batch_size]
            optimizer.zero_grad(set_to_none=True)
            total, _, epistasis = _epistasis_forward(
                head,
                state,
                encoded,
                rows,
                data.doubles,
                device,
            )
            target = torch.from_numpy(data.doubles.target[rows]).to(device)
            loss = huber(total, target)
            supervised = known_mask[rows]
            if np.any(supervised):
                selected = torch.from_numpy(
                    np.flatnonzero(supervised)
                ).to(device)
                epistasis_target = torch.from_numpy(
                    known_target[rows][supervised]
                ).to(device)
                loss = loss + 0.25 * huber(
                    epistasis[selected], epistasis_target
                )
            loss.backward()
            nn.utils.clip_grad_norm_(head.parameters(), 1.0)
            optimizer.step()
            total_loss += float(loss.detach()) * len(rows)
            seen += len(rows)
        record: dict[str, object] = {
            "epoch": epoch,
            "train_loss": total_loss / max(1, seen),
        }
        if validation_rows is not None:
            total, additive, epistasis = _predict_double_rows(
                head,
                potentials,
                representations,
                data.doubles,
                validation_rows,
                row_batch_size=row_batch_size,
                device=device,
            )
            score = float(
                np.mean(
                    np.abs(
                        total - data.doubles.target[validation_rows]
                    )
                )
            )
            record["validation_mae"] = score
            record["validation_additive_mae"] = float(
                np.mean(
                    np.abs(
                        additive
                        - data.doubles.target[validation_rows]
                    )
                )
            )
            if score < best_score - 1e-5:
                best_score = score
                best_epoch = epoch
                best_state = {
                    name: value.detach().cpu().clone()
                    for name, value in head.state_dict().items()
                }
                stale = 0
            elif epoch >= minimum_epochs:
                stale += 1
            print(
                f"epistasis epoch={epoch:03d} "
                f"train={record['train_loss']:.4f} "
                f"val_mae={score:.4f} "
                f"additive_mae={record['validation_additive_mae']:.4f}",
                flush=True,
            )
        else:
            best_epoch = epoch
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in head.state_dict().items()
            }
            print(
                f"epistasis final epoch={epoch:03d} "
                f"train={record['train_loss']:.4f}",
                flush=True,
            )
        history.append(record)
        if (
            validation_rows is not None
            and epoch >= minimum_epochs
            and stale >= patience
        ):
            break
    if best_state is None:
        raise RuntimeError("epistasis training produced no checkpoint")
    return best_state, {
        "seed": seed,
        "epochs_completed": len(history),
        "best_epoch": best_epoch,
        "best_validation_mae": (
            best_score if validation_rows is not None else None
        ),
        "known_epistasis_rows": int(np.sum(known_mask[train_rows])),
        "elapsed_seconds": time.monotonic() - started,
        "history": history,
    }


def _candidate_metrics(
    data: FullStructureArrays,
    rows: np.ndarray,
    prediction: np.ndarray,
    *,
    threshold: float,
    seed: int,
) -> dict[str, object]:
    return _evaluate_predictions(
        data.singles.target[rows],
        prediction,
        data.protein_id[data.singles.protein_index[rows]],
        threshold=threshold,
        bootstrap_seed=seed,
    )


def _save_torch(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".partial")
    partial.unlink(missing_ok=True)
    torch.save(value, partial)
    partial.replace(path)


def train_full_structure_architecture(
    data_path: Path,
    proteinmpnn_repository: Path,
    output_dir: Path,
    *,
    seed: int = 20260723,
    cv_epochs: int = 30,
    cv_minimum_epochs: int = 15,
    final_epochs: int = 70,
    final_minimum_epochs: int = 50,
    epistasis_epochs: int = 50,
    epistasis_minimum_epochs: int = 25,
    patience: int = 8,
    protein_batch_size: int = 8,
    row_batch_size: int = 4096,
    learning_rate: float = 3.0e-4,
    structure_learning_rate: float = 1.0e-4,
    epistasis_learning_rate: float = 7.5e-4,
    weight_decay: float = 1.0e-4,
    device: str = "cuda",
    objective: FullStructureObjective = FullStructureObjective(),
) -> dict[str, object]:
    """Run family-clustered structure ablation, then train the final model."""

    data_path = Path(data_path).resolve()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    outer_report = output_dir / "outer_evaluation.json"
    if outer_report.exists():
        raise FileExistsError(
            "sealed outer evaluation already exists; use a new output directory "
            "for another training run"
        )
    data = load_full_structure_arrays(data_path)
    torch_device = torch.device(
        device if device != "cuda" or torch.cuda.is_available() else "cpu"
    )
    _strict_fp32(torch_device)
    folds = sorted(set(data.split[data.split >= 0].tolist()))
    expected_folds = int(
        data.provenance["split_policy"]["development_folds"]
    )
    if folds != list(range(expected_folds)):
        raise RuntimeError(
            f"development folds are incomplete: found {folds}"
        )
    single_count = np.bincount(
        data.singles.protein_index, minlength=len(data.protein_id)
    )
    candidate_states: dict[str, dict[int, dict[str, torch.Tensor]]] = {
        "sequence_only": {},
        "full_structure": {},
    }
    candidate_reports: dict[str, object] = {}
    candidate_oof: dict[str, np.ndarray] = {}
    started = time.monotonic()

    for candidate, use_structure in (
        ("sequence_only", False),
        ("full_structure", True),
    ):
        prediction = np.full(
            len(data.singles.target), np.nan, dtype=np.float32
        )
        fold_reports: list[dict[str, object]] = []
        for fold in folds:
            train_proteins = np.flatnonzero(
                (data.split >= 0)
                & (data.split != fold)
                & (single_count > 0)
            )
            validation_proteins = np.flatnonzero(
                (data.split == fold) & (single_count > 0)
            )
            if not len(train_proteins) or not len(validation_proteins):
                raise RuntimeError(
                    f"fold {fold} has an empty single-protein partition"
                )
            print(
                f"starting {candidate} fold={fold}: "
                f"train_proteins={len(train_proteins)} "
                f"validation_proteins={len(validation_proteins)}",
                flush=True,
            )
            state, training = _train_single_model(
                data,
                proteinmpnn_repository,
                train_proteins,
                validation_proteins,
                use_structure=use_structure,
                seed=seed + 1000 * int(use_structure) + fold,
                epochs=cv_epochs,
                minimum_epochs=cv_minimum_epochs,
                patience=patience,
                protein_batch_size=protein_batch_size,
                learning_rate=(
                    structure_learning_rate
                    if use_structure
                    else learning_rate
                ),
                weight_decay=weight_decay,
                objective=objective,
                device=torch_device,
                initial_state=(
                    candidate_states["sequence_only"][fold]
                    if use_structure
                    else None
                ),
            )
            candidate_states[candidate][fold] = state
            model = _model_from_state(
                data,
                proteinmpnn_repository,
                use_structure=use_structure,
                state=state,
                device=torch_device,
            )
            validation_rows = np.flatnonzero(
                np.isin(
                    data.singles.protein_index, validation_proteins
                )
            )
            prediction[validation_rows] = _predict_single_rows(
                model,
                data,
                validation_rows,
                protein_batch_size=protein_batch_size,
                device=torch_device,
            )
            fold_metrics = _candidate_metrics(
                data,
                validation_rows,
                prediction[validation_rows],
                threshold=objective.stabilizer_threshold,
                seed=seed + fold,
            )
            fold_reports.append(
                {
                    "fold": fold,
                    "train_proteins": len(train_proteins),
                    "validation_proteins": len(validation_proteins),
                    "training": training,
                    "metrics": fold_metrics,
                }
            )
            if use_structure:
                _save_torch(
                    output_dir / "cv" / candidate / f"fold_{fold}.pt",
                    {
                        "schema": FULL_STRUCTURE_CHECKPOINT_SCHEMA,
                        "data_sha256": data.provenance["sha256"],
                        "use_structure": use_structure,
                        "fold": fold,
                        "model_state_dict": state,
                    },
                )
            del model
            if torch_device.type == "cuda":
                torch.cuda.empty_cache()
        development_rows = np.flatnonzero(
            data.split[data.singles.protein_index] >= 0
        )
        if np.any(~np.isfinite(prediction[development_rows])):
            raise RuntimeError(f"{candidate} OOF predictions are incomplete")
        metrics = _candidate_metrics(
            data,
            development_rows,
            prediction[development_rows],
            threshold=objective.stabilizer_threshold,
            seed=seed,
        )
        candidate_reports[candidate] = {
            "use_structure": use_structure,
            "folds": fold_reports,
            "oof": metrics,
        }
        candidate_oof[candidate] = prediction

    sequence_metrics = candidate_reports["sequence_only"]["oof"]
    structure_metrics = candidate_reports["full_structure"]["oof"]
    structure_gate = {
        "mae_improvement_at_least_0_02": (
            float(sequence_metrics["regression"]["mae"])
            - float(structure_metrics["regression"]["mae"])
            >= 0.02
        ),
        "spearman_not_lower": (
            float(structure_metrics["regression"]["spearman"])
            >= float(sequence_metrics["regression"]["spearman"])
        ),
        "average_precision_not_lower": (
            float(structure_metrics["retrieval"]["average_precision"])
            >= float(sequence_metrics["retrieval"]["average_precision"])
        ),
    }
    structure_gate["passed"] = all(structure_gate.values())
    selected = (
        "full_structure" if structure_gate["passed"] else "sequence_only"
    )
    selected_structure = selected == "full_structure"
    development_proteins = np.flatnonzero(
        (data.split >= 0) & (single_count > 0)
    )
    print(
        f"OOF architecture selection: {selected}; gate={structure_gate}",
        flush=True,
    )
    final_sequence_state, final_sequence_training = _train_single_model(
        data,
        proteinmpnn_repository,
        development_proteins,
        None,
        use_structure=False,
        seed=seed + 10000,
        epochs=final_epochs,
        minimum_epochs=final_minimum_epochs,
        patience=patience,
        protein_batch_size=protein_batch_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        objective=objective,
        device=torch_device,
    )
    if selected_structure:
        final_state, final_structure_training = _train_single_model(
            data,
            proteinmpnn_repository,
            development_proteins,
            None,
            use_structure=True,
            seed=seed + 11000,
            epochs=final_epochs,
            minimum_epochs=final_minimum_epochs,
            patience=patience,
            protein_batch_size=protein_batch_size,
            learning_rate=structure_learning_rate,
            weight_decay=weight_decay,
            objective=objective,
            device=torch_device,
            initial_state=final_sequence_state,
        )
    else:
        final_state = final_sequence_state
        final_structure_training = None
    final_model = _model_from_state(
        data,
        proteinmpnn_repository,
        use_structure=selected_structure,
        state=final_state,
        device=torch_device,
    )
    final_potentials, final_representations = _all_state_outputs(
        final_model,
        data,
        protein_batch_size=protein_batch_size,
        device=torch_device,
    )

    # Double-mutant OOF evaluation uses the fold-specific selected single
    # model, so no held-out family contributes to either component.
    double_oof_total = np.full(
        len(data.doubles.target), np.nan, dtype=np.float32
    )
    double_oof_additive = np.full_like(double_oof_total, np.nan)
    double_oof_epistasis = np.full_like(double_oof_total, np.nan)
    double_fold_reports: list[dict[str, object]] = []
    for fold in folds:
        fold_model = _model_from_state(
            data,
            proteinmpnn_repository,
            use_structure=selected_structure,
            state=candidate_states[selected][fold],
            device=torch_device,
        )
        fold_potentials, fold_representations = _all_state_outputs(
            fold_model,
            data,
            protein_batch_size=protein_batch_size,
            device=torch_device,
        )
        train_rows = np.flatnonzero(
            (data.split[data.doubles.protein_index] >= 0)
            & (data.split[data.doubles.protein_index] != fold)
        )
        validation_rows = np.flatnonzero(
            data.split[data.doubles.protein_index] == fold
        )
        if not len(train_rows) or not len(validation_rows):
            raise RuntimeError(
                f"fold {fold} has an empty double-mutant partition"
            )
        head_state, head_training = _train_epistasis_head(
            data,
            fold_potentials,
            fold_representations,
            train_rows,
            validation_rows,
            seed=seed + 20000 + fold,
            epochs=epistasis_epochs,
            minimum_epochs=epistasis_minimum_epochs,
            patience=patience,
            row_batch_size=row_batch_size,
            learning_rate=epistasis_learning_rate,
            weight_decay=weight_decay,
            device=torch_device,
        )
        head = FullStructureEpistasisHead(
            FullStructureEpistasisConfig(
                representation_dim=fold_representations.shape[-1]
            )
        ).to(torch_device)
        head.load_state_dict(head_state)
        total, additive, epistasis = _predict_double_rows(
            head,
            fold_potentials,
            fold_representations,
            data.doubles,
            validation_rows,
            row_batch_size=row_batch_size,
            device=torch_device,
        )
        double_oof_total[validation_rows] = total
        double_oof_additive[validation_rows] = additive
        double_oof_epistasis[validation_rows] = epistasis
        double_fold_reports.append(
            {
                "fold": fold,
                "training": head_training,
                "total": _evaluate_predictions(
                    data.doubles.target[validation_rows],
                    total,
                    data.protein_id[
                        data.doubles.protein_index[validation_rows]
                    ],
                    threshold=objective.stabilizer_threshold,
                    bootstrap_samples=200,
                    bootstrap_seed=seed + 20000 + fold,
                ),
                "additive": regression_metrics(
                    data.doubles.target[validation_rows], additive
                ),
            }
        )
        del fold_model, head
        if torch_device.type == "cuda":
            torch.cuda.empty_cache()

    development_double_rows = np.flatnonzero(
        data.split[data.doubles.protein_index] >= 0
    )
    if np.any(~np.isfinite(double_oof_total[development_double_rows])):
        raise RuntimeError("double-mutant OOF predictions are incomplete")
    double_oof_metrics = {
        "total": _evaluate_predictions(
            data.doubles.target[development_double_rows],
            double_oof_total[development_double_rows],
            data.protein_id[
                data.doubles.protein_index[development_double_rows]
            ],
            threshold=objective.stabilizer_threshold,
            bootstrap_seed=seed + 20000,
        ),
        "additive": regression_metrics(
            data.doubles.target[development_double_rows],
            double_oof_additive[development_double_rows],
        ),
        "epistasis_summary": {
            "mean": float(
                np.mean(double_oof_epistasis[development_double_rows])
            ),
            "standard_deviation": float(
                np.std(double_oof_epistasis[development_double_rows])
            ),
        },
    }
    double_gate = {
        "mae_improves": (
            float(double_oof_metrics["total"]["regression"]["mae"])
            < float(double_oof_metrics["additive"]["mae"])
        ),
        "spearman_not_lower": (
            float(
                double_oof_metrics["total"]["regression"]["spearman"]
            )
            >= float(double_oof_metrics["additive"]["spearman"])
        ),
    }
    double_gate["passed"] = all(double_gate.values())
    selected_double_decoder = (
        "learned_epistasis" if double_gate["passed"] else "additive"
    )
    final_double_rows = np.flatnonzero(
        data.split[data.doubles.protein_index] >= 0
    )
    final_epistasis_state, final_epistasis_training = (
        _train_epistasis_head(
            data,
            final_potentials,
            final_representations,
            final_double_rows,
            None,
            seed=seed + 30000,
            epochs=epistasis_epochs,
            minimum_epochs=epistasis_minimum_epochs,
            patience=patience,
            row_batch_size=row_batch_size,
            learning_rate=epistasis_learning_rate,
            weight_decay=weight_decay,
            device=torch_device,
        )
    )
    checkpoint = {
        "schema": FULL_STRUCTURE_CHECKPOINT_SCHEMA,
        "model_schema": FULL_STRUCTURE_MODEL_SCHEMA,
        "created_from_data": data.provenance,
        "selected_candidate": selected,
        "selected_double_decoder": selected_double_decoder,
        "selection_uses_outer": False,
        "state_config": FullStructureConfig(
            esm_dim=data.esm_residue.shape[-1],
            use_structure=selected_structure,
        ),
        "state_model_state_dict": final_state,
        "epistasis_config": FullStructureEpistasisConfig(
            representation_dim=final_representations.shape[-1]
        ),
        "epistasis_state_dict": final_epistasis_state,
        "objective": objective,
        "seed": seed,
    }
    checkpoint_path = output_dir / "full_structure_selected.pt"
    _save_torch(checkpoint_path, checkpoint)
    report = {
        "schema": FULL_STRUCTURE_TRAINING_SCHEMA,
        "data": data.provenance,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": file_sha256(checkpoint_path),
        "device": str(torch_device),
        "numeric_policy": (
            "native float32; TF32 disabled; no autocast"
        ),
        "objective": asdict(objective),
        "configuration": {
            "seed": seed,
            "cv_epochs": cv_epochs,
            "cv_minimum_epochs": cv_minimum_epochs,
            "final_epochs": final_epochs,
            "final_minimum_epochs": final_minimum_epochs,
            "epistasis_epochs": epistasis_epochs,
            "epistasis_minimum_epochs": epistasis_minimum_epochs,
            "protein_batch_size": protein_batch_size,
            "row_batch_size": row_batch_size,
            "learning_rate": learning_rate,
            "structure_learning_rate": structure_learning_rate,
            "epistasis_learning_rate": epistasis_learning_rate,
            "weight_decay": weight_decay,
        },
        "single_architecture_ablation": candidate_reports,
        "structure_promotion_gate": structure_gate,
        "selected_candidate": selected,
        "final_single_training": {
            "sequence_warm_start": final_sequence_training,
            "structure_refinement": final_structure_training,
        },
        "double_mutant_oof": {
            "folds": double_fold_reports,
            "metrics": double_oof_metrics,
            "permutation_invariant": True,
            "reported_components": ["additive", "epistasis", "total"],
            "promotion_gate": double_gate,
            "selected_decoder": selected_double_decoder,
        },
        "final_epistasis_training": final_epistasis_training,
        "outer_partition_consumed": False,
        "elapsed_seconds": time.monotonic() - started,
    }
    report_path = output_dir / "training_report.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return {
        "schema": FULL_STRUCTURE_TRAINING_SCHEMA,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": report["checkpoint_sha256"],
        "report": str(report_path),
        "selected_candidate": selected,
        "selected_double_decoder": selected_double_decoder,
        "structure_promotion_gate": structure_gate,
        "double_promotion_gate": double_gate,
        "single_oof": candidate_reports[selected]["oof"],
        "double_oof": double_oof_metrics,
        "outer_partition_consumed": False,
    }


def _load_checkpoint_models(
    checkpoint_path: Path,
    data: FullStructureArrays,
    proteinmpnn_repository: Path,
    device: torch.device,
) -> tuple[
    FullStructureStateModel,
    FullStructureEpistasisHead,
    dict[str, object],
]:
    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False
    )
    if checkpoint.get("schema") != FULL_STRUCTURE_CHECKPOINT_SCHEMA:
        raise RuntimeError("full-structure checkpoint schema mismatch")
    if (
        checkpoint["created_from_data"]["sha256"]
        != data.provenance["sha256"]
    ):
        raise RuntimeError("checkpoint was trained from different feature data")
    config = checkpoint["state_config"]
    if not isinstance(config, FullStructureConfig):
        config = FullStructureConfig(**config)
    model, _ = _create_state_model(
        proteinmpnn_repository,
        esm_dimension=config.esm_dim,
        use_structure=config.use_structure,
        device=device,
    )
    model.load_state_dict(checkpoint["state_model_state_dict"])
    epistasis_config = checkpoint["epistasis_config"]
    if not isinstance(epistasis_config, FullStructureEpistasisConfig):
        epistasis_config = FullStructureEpistasisConfig(
            **epistasis_config
        )
    head = FullStructureEpistasisHead(epistasis_config).to(device)
    head.load_state_dict(checkpoint["epistasis_state_dict"])
    return model, head, checkpoint


def _algebra_checks(potentials: np.ndarray) -> dict[str, float]:
    values = torch.from_numpy(potentials)
    phi = values[:, :, :3]
    self_difference = phi[..., 0] - phi[..., 0]
    reverse = (
        (phi[..., 1] - phi[..., 0])
        + (phi[..., 0] - phi[..., 1])
    )
    cycle = (
        (phi[..., 1] - phi[..., 0])
        + (phi[..., 2] - phi[..., 1])
        + (phi[..., 0] - phi[..., 2])
    )
    return {
        "maximum_absolute_self_ddg": float(
            torch.max(torch.abs(self_difference))
        ),
        "maximum_absolute_reverse_sum": float(
            torch.max(torch.abs(reverse))
        ),
        "maximum_absolute_three_state_cycle": float(
            torch.max(torch.abs(cycle))
        ),
    }


def _write_predictions(
    path: Path,
    data: FullStructureArrays,
    mutations: MutationArrays,
    rows: np.ndarray,
    prediction: np.ndarray,
    *,
    additive: np.ndarray | None = None,
    epistasis: np.ndarray | None = None,
    learned_total: np.ndarray | None = None,
) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        header = [
            "protein_id",
            "mutation",
            "experimental_ddg_kcal_mol",
            "predicted_ddg_kcal_mol",
            "error_kcal_mol",
        ]
        if additive is not None:
            header.extend(
                [
                    "additive_ddg_kcal_mol",
                    "learned_epistasis_ddg_kcal_mol",
                    "learned_total_ddg_kcal_mol",
                ]
            )
        writer.writerow(header)
        for output_index, row in enumerate(rows):
            mutations_text = ";".join(
                f"{int(mutations.wt_amino_acid[row, mutation])}:"
                f"{int(mutations.position[row, mutation]) + 1}:"
                f"{int(mutations.mutant_amino_acid[row, mutation])}"
                for mutation in range(mutations.position.shape[1])
            )
            target = float(mutations.target[row])
            predicted = float(prediction[output_index])
            values: list[object] = [
                data.protein_id[mutations.protein_index[row]],
                mutations_text,
                target,
                predicted,
                predicted - target,
            ]
            if (
                additive is not None
                and epistasis is not None
                and learned_total is not None
            ):
                values.extend(
                    [
                        float(additive[output_index]),
                        float(epistasis[output_index]),
                        float(learned_total[output_index]),
                    ]
                )
            writer.writerow(values)


def _write_scatter(
    path: Path,
    target: np.ndarray,
    prediction: np.ndarray,
    *,
    title: str,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    lower = float(min(np.min(target), np.min(prediction)))
    upper = float(max(np.max(target), np.max(prediction)))
    margin = max(0.1, 0.03 * (upper - lower))
    lower -= margin
    upper += margin
    figure, axis = plt.subplots(figsize=(7.2, 7.2), dpi=160)
    axis.scatter(
        target,
        prediction,
        s=9,
        alpha=0.28,
        edgecolors="none",
        rasterized=True,
    )
    axis.plot([lower, upper], [lower, upper], color="black", linewidth=1)
    axis.axhline(0.0, color="grey", linewidth=0.6)
    axis.axvline(0.0, color="grey", linewidth=0.6)
    axis.set_xlim(lower, upper)
    axis.set_ylim(lower, upper)
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlabel("Experimental ΔΔG (kcal/mol)")
    axis.set_ylabel("Predicted ΔΔG (kcal/mol)")
    axis.set_title(title)
    figure.tight_layout()
    figure.savefig(path)
    plt.close(figure)


def evaluate_full_structure_outer(
    data_path: Path,
    checkpoint_path: Path,
    proteinmpnn_repository: Path,
    output_dir: Path,
    *,
    protein_batch_size: int = 8,
    row_batch_size: int = 4096,
    device: str = "cuda",
) -> dict[str, object]:
    """Consume the sealed family-cluster outer partition exactly once."""

    data = load_full_structure_arrays(data_path)
    checkpoint_path = Path(checkpoint_path).resolve()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "outer_evaluation.json"
    if report_path.exists():
        raise FileExistsError(
            f"sealed outer evaluation already exists: {report_path}"
        )
    torch_device = torch.device(
        device if device != "cuda" or torch.cuda.is_available() else "cpu"
    )
    _strict_fp32(torch_device)
    model, head, checkpoint = _load_checkpoint_models(
        checkpoint_path,
        data,
        proteinmpnn_repository,
        torch_device,
    )
    single_rows = np.flatnonzero(
        data.split[data.singles.protein_index] == -1
    )
    double_rows = np.flatnonzero(
        data.split[data.doubles.protein_index] == -1
    )
    if not len(single_rows) or not len(double_rows):
        raise RuntimeError("outer partition has no single or double rows")
    single_prediction = _predict_single_rows(
        model,
        data,
        single_rows,
        protein_batch_size=protein_batch_size,
        device=torch_device,
    )
    potentials, representations = _all_state_outputs(
        model,
        data,
        protein_batch_size=protein_batch_size,
        device=torch_device,
    )
    learned_total, additive, epistasis = _predict_double_rows(
        head,
        potentials,
        representations,
        data.doubles,
        double_rows,
        row_batch_size=row_batch_size,
        device=torch_device,
    )
    selected_double_decoder = checkpoint.get(
        "selected_double_decoder", "learned_epistasis"
    )
    if selected_double_decoder not in {
        "learned_epistasis",
        "additive",
    }:
        raise RuntimeError("checkpoint has an unknown double decoder")
    double_prediction = (
        learned_total
        if selected_double_decoder == "learned_epistasis"
        else additive
    )
    objective = checkpoint["objective"]
    if not isinstance(objective, FullStructureObjective):
        objective = FullStructureObjective(**objective)
    single_metrics = _evaluate_predictions(
        data.singles.target[single_rows],
        single_prediction,
        data.protein_id[data.singles.protein_index[single_rows]],
        threshold=objective.stabilizer_threshold,
    )
    double_metrics = {
        "selected_decoder": selected_double_decoder,
        "selected": _evaluate_predictions(
            data.doubles.target[double_rows],
            double_prediction,
            data.protein_id[
                data.doubles.protein_index[double_rows]
            ],
            threshold=objective.stabilizer_threshold,
        ),
        "learned_total": _evaluate_predictions(
            data.doubles.target[double_rows],
            learned_total,
            data.protein_id[
                data.doubles.protein_index[double_rows]
            ],
            threshold=objective.stabilizer_threshold,
        ),
        "additive": regression_metrics(
            data.doubles.target[double_rows], additive
        ),
        "epistasis_summary": {
            "mean": float(np.mean(epistasis)),
            "standard_deviation": float(np.std(epistasis)),
        },
    }
    single_csv = output_dir / "outer_single_predictions.csv"
    double_csv = output_dir / "outer_double_predictions.csv"
    _write_predictions(
        single_csv,
        data,
        data.singles,
        single_rows,
        single_prediction,
    )
    _write_predictions(
        double_csv,
        data,
        data.doubles,
        double_rows,
        double_prediction,
        additive=additive,
        epistasis=epistasis,
        learned_total=learned_total,
    )
    single_scatter = output_dir / "outer_single_pred_vs_exp.png"
    double_scatter = output_dir / "outer_double_pred_vs_exp.png"
    _write_scatter(
        single_scatter,
        data.singles.target[single_rows],
        single_prediction,
        title="Outer-family single mutants",
    )
    _write_scatter(
        double_scatter,
        data.doubles.target[double_rows],
        double_prediction,
        title="Outer-family double mutants",
    )
    report = {
        "schema": FULL_STRUCTURE_OUTER_EVALUATION_SCHEMA,
        "data": data.provenance,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": file_sha256(checkpoint_path),
        "selected_candidate": checkpoint["selected_candidate"],
        "selection_uses_outer": False,
        "numeric_policy": (
            "native float32; TF32 disabled; no autocast"
        ),
        "sign_convention": "negative ddG means stabilizing",
        "units": "kcal/mol",
        "single": single_metrics,
        "double": double_metrics,
        "algebra": _algebra_checks(potentials),
        "artifacts": {
            "single_predictions": str(single_csv),
            "double_predictions": str(double_csv),
            "single_scatter": str(single_scatter),
            "double_scatter": str(double_scatter),
        },
    }
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return {
        "schema": FULL_STRUCTURE_OUTER_EVALUATION_SCHEMA,
        "report": str(report_path),
        "checkpoint_sha256": report["checkpoint_sha256"],
        "single": single_metrics,
        "double": double_metrics,
        "algebra": report["algebra"],
        "artifacts": report["artifacts"],
    }
