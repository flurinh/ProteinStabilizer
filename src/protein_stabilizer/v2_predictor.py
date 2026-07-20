"""Application inference for promoted hierarchical single and multi models."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch

from .data import (
    AMINO_ACIDS,
    EmbeddingRequest,
    Mutation,
    apply_mutations,
    normalize_sequence,
)
from .embeddings import ESMCEmbedder, HierarchicalEmbedding, file_sha256, token_batches
from .predictor import _percentile_ranks, _thermostability_consensus
from .state_potential import load_state_potential_ensemble
from .structure import ProteinMPNNBackboneEmbedder
from .training import load_auxiliary_checkpoint
from .v2_features import membrane_topology_features
from .v2_multi import load_hierarchical_multi_checkpoint
from .v2_training import load_hierarchical_ensemble


def _device(name: str) -> torch.device:
    return torch.device(
        name if name.startswith("cuda") and torch.cuda.is_available() else "cpu"
    )


def _embed_requests(
    embedder: ESMCEmbedder,
    requests: Sequence[EmbeddingRequest],
    *,
    window_radius: int,
    max_tokens: int,
    max_batch_size: int,
) -> list[HierarchicalEmbedding]:
    by_request: dict[
        tuple[str, tuple[int, ...]], HierarchicalEmbedding
    ] = {}
    for batch in token_batches(
        requests,
        max_tokens=max_tokens,
        max_batch_size=max_batch_size,
    ):
        embedded = embedder.encode_hierarchy(
            batch, window_radius=window_radius
        )
        for request, value in zip(batch, embedded, strict=True):
            by_request[(request.sequence_hash, request.positions)] = value
    results = [
        by_request[(request.sequence_hash, request.positions)]
        for request in requests
    ]
    if len(results) != len(requests):
        raise RuntimeError("hierarchy inference returned the wrong request count")
    return results


def _annotation_rows(
    sequence: str,
    mutations: Sequence[Mutation],
    *,
    topology: str | None,
    generic_numbering: Mapping[int, str] | None,
    pdb_path: Path | None,
    proteinmpnn_repository: Path | None,
    device: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, object] | None]:
    membrane = np.stack(
        [
            membrane_topology_features(
                topology,
                generic_numbering=(
                    None
                    if generic_numbering is None
                    else generic_numbering.get(mutation.position)
                ),
                is_gpcr=(topology or "").strip().lower()
                == "alpha_helical_gpcr",
            )
            for mutation in mutations
        ]
    ).astype(np.float32)
    structure = np.zeros((len(mutations), 128), dtype=np.float32)
    structure_mask = np.zeros(len(mutations), dtype=bool)
    provenance: dict[str, object] | None = None
    if pdb_path is not None:
        if proteinmpnn_repository is None:
            raise ValueError(
                "ProteinMPNN repository is required when a PDB is supplied"
            )
        structure_embedder = ProteinMPNNBackboneEmbedder(
            proteinmpnn_repository,
            device=device,
        )
        residue = structure_embedder.encode(Path(pdb_path), sequence)
        structure = np.stack(
            [residue[mutation.position - 1] for mutation in mutations]
        ).astype(np.float32)
        structure_mask[:] = True
        provenance = {
            "pdb_path": str(Path(pdb_path).resolve()),
            "pdb_sha256": file_sha256(Path(pdb_path)),
            "proteinmpnn": json.loads(
                structure_embedder.provenance.canonical_json()
            ),
        }
    return structure, structure_mask, membrane, provenance


def _state_arrays(
    wt: HierarchicalEmbedding,
    mutants: Sequence[HierarchicalEmbedding],
    *,
    positions: Sequence[int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    index_by_position = {
        position: index for index, position in enumerate(positions)
    }
    wt_windows = np.stack(
        [wt.windows[index_by_position[position]] for position in positions]
    )
    wt_masks = np.stack(
        [wt.window_mask[index_by_position[position]] for position in positions]
    )
    mutant_windows = np.concatenate(
        [embedding.windows for embedding in mutants], axis=0
    )
    if mutant_windows.shape != wt_windows.shape:
        raise RuntimeError("WT and mutant hierarchy windows do not align")
    return (
        wt_windows,
        mutant_windows,
        wt_masks,
        np.repeat(wt.global_mean[None], len(positions), axis=0),
        np.stack([embedding.global_mean for embedding in mutants]),
    )


def _torch_state(
    arrays: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    *,
    structure: np.ndarray,
    structure_mask: np.ndarray,
    membrane: np.ndarray,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    wt_window, mutant_window, window_mask, wt_global, mutant_global = arrays
    return {
        "wt_window": torch.from_numpy(wt_window.astype(np.float32)).to(device),
        "mutant_window": torch.from_numpy(
            mutant_window.astype(np.float32)
        ).to(device),
        "window_mask": torch.from_numpy(window_mask).to(device),
        "wt_global": torch.from_numpy(wt_global.astype(np.float32)).to(device),
        "mutant_global": torch.from_numpy(
            mutant_global.astype(np.float32)
        ).to(device),
        "structure": torch.from_numpy(structure.astype(np.float32)).to(device),
        "structure_mask": torch.from_numpy(structure_mask).to(device),
        "membrane": torch.from_numpy(membrane.astype(np.float32)).to(device),
    }


def _load_promoted_state_potential(
    checkpoint_path: Path,
    baseline_path: Path,
    device: torch.device,
) -> tuple[torch.nn.Module, dict[str, object], float]:
    checkpoint_path = Path(checkpoint_path)
    baseline_path = Path(baseline_path)
    model, payload = load_state_potential_ensemble(checkpoint_path, device)
    if not bool(payload.get("promotion", {}).get("production_eligible")):
        raise RuntimeError(
            "state-potential checkpoint did not pass its frozen promotion gates"
        )
    expected_baseline = str(payload.get("baseline_checkpoint_sha256", ""))
    actual_baseline = file_sha256(baseline_path)
    if expected_baseline != actual_baseline:
        raise RuntimeError(
            "state-potential checkpoint was selected against a different "
            "hierarchical baseline"
        )
    weight = float(payload["state_potential_weight"])
    if not 0.0 <= weight <= 1.0:
        raise RuntimeError("state-potential blend weight is invalid")
    return model, payload, weight


def _amino_acid_indices(
    mutations: Sequence[Mutation],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    index = {
        amino_acid: value
        for value, amino_acid in enumerate(AMINO_ACIDS)
    }
    return (
        torch.tensor(
            [index[mutation.wt] for mutation in mutations],
            dtype=torch.long,
            device=device,
        ),
        torch.tensor(
            [index[mutation.mutant] for mutation in mutations],
            dtype=torch.long,
            device=device,
        ),
    )


def predict_hierarchical_mutations(
    sequence: str,
    mutations: Sequence[str | Mutation],
    checkpoint_dir: Path,
    *,
    model_name: str = "esmc_600m",
    device: str = "cuda",
    topology: str | None = None,
    generic_numbering: Mapping[int, str] | None = None,
    pdb_path: Path | None = None,
    proteinmpnn_repository: Path | None = None,
    max_tokens: int = 8192,
    max_batch_size: int = 128,
    embedder: ESMCEmbedder | None = None,
    state_potential_checkpoint: Path | None = None,
) -> dict[str, object]:
    """Predict one unordered mutation set with additive/epistasis reporting."""

    wt_sequence = normalize_sequence(sequence)
    parsed = sorted(
        [
            mutation
            if isinstance(mutation, Mutation)
            else Mutation.parse(mutation)
            for mutation in mutations
        ],
        key=lambda mutation: mutation.position,
    )
    if not parsed:
        raise ValueError("at least one mutation is required")
    positions = tuple(mutation.position for mutation in parsed)
    if len(positions) != len(set(positions)):
        raise ValueError("a mutation set may contain only one substitution per site")
    single_sequences = [
        apply_mutations(wt_sequence, [mutation]) for mutation in parsed
    ]
    joint_sequence = apply_mutations(wt_sequence, parsed)
    requests = [EmbeddingRequest(wt_sequence, positions)]
    requests.extend(
        EmbeddingRequest(single, (mutation.position,))
        for single, mutation in zip(single_sequences, parsed, strict=True)
    )
    if len(parsed) > 1:
        requests.append(EmbeddingRequest(joint_sequence, positions))
    active_embedder = embedder or ESMCEmbedder(
        model_name=model_name, device=device
    )
    window_radius = 4
    embedded = _embed_requests(
        active_embedder,
        requests,
        window_radius=window_radius,
        max_tokens=max_tokens,
        max_batch_size=max_batch_size,
    )
    wt_embedding = embedded[0]
    single_embeddings = embedded[1 : 1 + len(parsed)]
    joint_embedding = (
        embedded[-1] if len(parsed) > 1 else single_embeddings[0]
    )
    structure, structure_mask, membrane, structure_provenance = (
        _annotation_rows(
            wt_sequence,
            parsed,
            topology=topology,
            generic_numbering=generic_numbering,
            pdb_path=pdb_path,
            proteinmpnn_repository=proteinmpnn_repository,
            device=device,
        )
    )
    torch_device = _device(device)
    base_path = Path(checkpoint_dir) / "hierarchy_selected_ensemble.pt"
    base, payload = load_hierarchical_ensemble(base_path, torch_device)
    single_arrays = _state_arrays(
        wt_embedding,
        single_embeddings,
        positions=positions,
    )
    joint_arrays = (
        single_arrays
        if len(parsed) == 1
        else (
            single_arrays[0],
            joint_embedding.windows,
            single_arrays[2],
            single_arrays[3],
            np.repeat(
                joint_embedding.global_mean[None], len(parsed), axis=0
            ),
        )
    )
    single_tensors = _torch_state(
        single_arrays,
        structure=structure,
        structure_mask=structure_mask,
        membrane=membrane,
        device=torch_device,
    )
    joint_tensors = _torch_state(
        joint_arrays,
        structure=structure,
        structure_mask=structure_mask,
        membrane=membrane,
        device=torch_device,
    )
    with torch.inference_mode():
        single_heads = base.predict_heads(**single_tensors)
        single_latent = base.latent(**single_tensors)
        joint_latent = base.latent(**joint_tensors)
    baseline_single_ddg = single_heads["ddg"].float()
    single_ddg = baseline_single_ddg
    state_single_ddg: torch.Tensor | None = None
    state_single_latent: torch.Tensor | None = None
    state_model: torch.nn.Module | None = None
    state_payload: dict[str, object] | None = None
    state_weight: float | None = None
    state_checkpoint_sha256: str | None = None
    if state_potential_checkpoint is not None:
        state_model, state_payload, state_weight = (
            _load_promoted_state_potential(
                state_potential_checkpoint,
                base_path,
                torch_device,
            )
        )
        wt_amino_acid, mutant_amino_acid = _amino_acid_indices(
            parsed, torch_device
        )
        with torch.inference_mode():
            state_single_ddg = state_model(
                single_tensors["wt_window"],
                single_tensors["window_mask"],
                single_tensors["wt_global"],
                wt_amino_acid,
                mutant_amino_acid,
                structure=single_tensors["structure"],
                structure_mask=single_tensors["structure_mask"],
                membrane=single_tensors["membrane"],
            ).float()
            state_single_latent = state_model.latent(
                single_tensors["wt_window"],
                single_tensors["window_mask"],
                single_tensors["wt_global"],
                wt_amino_acid,
                mutant_amino_acid,
                structure=single_tensors["structure"],
                structure_mask=single_tensors["structure_mask"],
                membrane=single_tensors["membrane"],
            ).float()
        single_ddg = (
            (1.0 - state_weight) * baseline_single_ddg
            + state_weight * state_single_ddg
        )
        state_checkpoint_sha256 = file_sha256(
            Path(state_potential_checkpoint)
        )
    additive = float(single_ddg.sum().cpu())
    total = additive
    epistasis = 0.0
    multi_path = Path(checkpoint_dir) / "hierarchy_multi_head.pt"
    multi_checkpoint_sha256: str | None = None
    multi_state_weight: float | None = None
    if len(parsed) > 1 and multi_path.is_file():
        if state_potential_checkpoint is not None:
            candidate_multi = (
                Path(state_potential_checkpoint).parent
                / "hierarchy_multi_head.pt"
            )
            if candidate_multi.is_file():
                _, candidate_payload = load_hierarchical_multi_checkpoint(
                    candidate_multi, torch_device
                )
                if bool(
                    candidate_payload.get("promotion", {}).get(
                        "production_eligible"
                    )
                ):
                    if (
                        str(
                            candidate_payload.get(
                                "state_potential_checkpoint_sha256", ""
                            )
                        )
                        != state_checkpoint_sha256
                    ):
                        raise RuntimeError(
                            "multi-mutant head uses a different state-potential "
                            "checkpoint"
                        )
                    multi_path = candidate_multi
        multi, multi_payload = load_hierarchical_multi_checkpoint(
            multi_path, torch_device
        )
        multi_single_ddg = baseline_single_ddg
        multi_single_latent = single_latent
        multi_joint_latent = joint_latent
        if (
            multi_payload.get("base_candidate")
            == "hierarchy_state_potential_blend"
        ):
            if state_single_ddg is None or state_single_latent is None:
                raise RuntimeError(
                    "promoted multi-mutant head requires state-potential features"
                )
            multi_state_weight = float(
                multi_payload["state_potential_weight"]
            )
            multi_single_ddg = (
                (1.0 - multi_state_weight) * baseline_single_ddg
                + multi_state_weight * state_single_ddg
            )
            multi_single_latent = torch.cat(
                [single_latent, state_single_latent], dim=-1
            )
            multi_joint_latent = torch.cat(
                [joint_latent, state_single_latent], dim=-1
            )
        with torch.inference_mode():
            values = multi(
                multi_single_ddg.unsqueeze(0),
                multi_single_latent.unsqueeze(0),
                multi_joint_latent.unsqueeze(0),
            )
        single_ddg = multi_single_ddg
        additive = float(values[1][0].cpu())
        epistasis = float(values[2][0].cpu())
        total = float(values[0][0].cpu())
        multi_checkpoint_sha256 = file_sha256(multi_path)
    mutation_rows = []
    for index, mutation in enumerate(parsed):
        row = {
            "mutation": str(mutation),
            "ddg": float(single_ddg[index].cpu()),
            "retrieval_score": float(-single_ddg[index].cpu()),
        }
        if state_single_ddg is not None:
            row["hierarchy_ddg"] = float(
                baseline_single_ddg[index].cpu()
            )
            row["state_potential_ddg"] = float(
                state_single_ddg[index].cpu()
            )
        else:
            row["retrieval_score"] = float(
                single_heads["retrieval"][index].float().cpu()
            )
        mutation_rows.append(row)
    return {
        "mutations": mutation_rows,
        "total_ddg": total,
        "additive_ddg": additive,
        "epistasis_ddg": epistasis,
        "ddg_units": "kcal/mol",
        "sign_convention": "negative is stabilizing",
        "mutation_set_policy": (
            (
                "validation-selected single-mutant fusion plus retained "
                "permutation-invariant learned double-mutant epistasis"
                if state_payload is not None
                else "permutation-invariant learned double-mutant epistasis"
            )
            if len(parsed) == 2 and multi_checkpoint_sha256 is not None
            else (
                (
                    "validation-selected single-mutant fusion plus retained "
                    "permutation-invariant epistasis extrapolation beyond "
                    "double-mutant training"
                    if state_payload is not None
                    else (
                        "permutation-invariant extrapolation beyond "
                        "double-mutant training"
                    )
                )
                if len(parsed) > 2 and multi_checkpoint_sha256 is not None
                else (
                    "validation-selected fused additive constituent prediction"
                    if state_payload is not None
                    else "additive constituent prediction"
                )
            )
        ),
        "topology": topology,
        "structure_provenance": structure_provenance,
        "model": {
            "name": model_name,
            "candidate": payload["candidate"],
            "checkpoint": str(base_path.resolve()),
            "checkpoint_sha256": file_sha256(base_path),
            "multi_checkpoint_sha256": multi_checkpoint_sha256,
            "multi_state_potential_weight": multi_state_weight,
            "state_potential_checkpoint": (
                None
                if state_potential_checkpoint is None
                else str(Path(state_potential_checkpoint).resolve())
            ),
            "state_potential_checkpoint_sha256": state_checkpoint_sha256,
            "state_potential_weight": state_weight,
            "prediction": (
                "validation-selected hierarchy/state-potential blend"
                if state_payload is not None
                else "hierarchical baseline"
            ),
        },
    }


def screen_hierarchical_single_mutants(
    sequence: str,
    checkpoint_dir: Path,
    output_path: Path,
    *,
    positions: Sequence[int] | None = None,
    model_name: str = "esmc_600m",
    device: str = "cuda",
    topology: str | None = None,
    generic_numbering: Mapping[int, str] | None = None,
    pdb_path: Path | None = None,
    proteinmpnn_repository: Path | None = None,
    legacy_checkpoint_dir: Path | None = None,
    max_tokens: int = 8192,
    max_batch_size: int = 128,
    top: int = 50,
    embedder: ESMCEmbedder | None = None,
    state_potential_checkpoint: Path | None = None,
    scan_mode: str = "exact",
) -> dict[str, object]:
    """Score every non-WT amino acid independently at selected positions."""

    wt_sequence = normalize_sequence(sequence)
    selected_positions = tuple(
        range(1, len(wt_sequence) + 1)
        if positions is None
        else sorted(set(int(position) for position in positions))
    )
    if not selected_positions or any(
        position < 1 or position > len(wt_sequence)
        for position in selected_positions
    ):
        raise ValueError("screen positions must be within the sequence")
    if top < 1:
        raise ValueError("top must be positive")
    if scan_mode not in {"exact", "state-only"}:
        raise ValueError("scan mode must be exact or state-only")
    if scan_mode == "state-only" and state_potential_checkpoint is None:
        raise ValueError(
            "state-only scanning requires a state-potential checkpoint"
        )
    candidates = [
        Mutation(wt_sequence[position - 1], position, amino_acid)
        for position in selected_positions
        for amino_acid in AMINO_ACIDS
        if amino_acid != wt_sequence[position - 1]
    ]
    requests = [EmbeddingRequest(wt_sequence, selected_positions)]
    if scan_mode == "exact":
        requests.extend(
            EmbeddingRequest(
                apply_mutations(wt_sequence, [mutation]),
                (mutation.position,),
            )
            for mutation in candidates
        )
    active_embedder = embedder or ESMCEmbedder(
        model_name=model_name, device=device
    )
    embedded = _embed_requests(
        active_embedder,
        requests,
        window_radius=4,
        max_tokens=max_tokens,
        max_batch_size=max_batch_size,
    )
    wt = embedded[0]
    index_by_position = {
        position: index for index, position in enumerate(selected_positions)
    }
    repeated_wt = HierarchicalEmbedding(
        global_mean=wt.global_mean,
        windows=np.stack(
            [wt.windows[index_by_position[item.position]] for item in candidates]
        ),
        window_mask=np.stack(
            [
                wt.window_mask[index_by_position[item.position]]
                for item in candidates
            ]
        ),
    )
    structure, structure_mask, membrane, structure_provenance = (
        _annotation_rows(
            wt_sequence,
            candidates,
            topology=topology,
            generic_numbering=generic_numbering,
            pdb_path=pdb_path,
            proteinmpnn_repository=proteinmpnn_repository,
            device=device,
        )
    )
    torch_device = _device(device)
    base_path = Path(checkpoint_dir) / "hierarchy_selected_ensemble.pt"
    base: torch.nn.Module | None = None
    payload: dict[str, object] | None = None
    arrays: tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
    ] | None = None
    baseline_ddg: np.ndarray | None = None
    baseline_retrieval: np.ndarray | None = None
    if scan_mode == "exact":
        arrays = (
            repeated_wt.windows,
            np.concatenate([value.windows for value in embedded[1:]], axis=0),
            repeated_wt.window_mask,
            np.repeat(wt.global_mean[None], len(candidates), axis=0),
            np.stack([value.global_mean for value in embedded[1:]]),
        )
        base, payload = load_hierarchical_ensemble(base_path, torch_device)
        with torch.inference_mode():
            heads = base.predict_heads(
                **_torch_state(
                    arrays,
                    structure=structure,
                    structure_mask=structure_mask,
                    membrane=membrane,
                    device=torch_device,
                )
            )
        baseline_ddg = heads["ddg"].float().cpu().numpy()
        baseline_retrieval = heads["retrieval"].float().cpu().numpy()
    state_ddg: np.ndarray | None = None
    state_payload: dict[str, object] | None = None
    state_weight: float | None = None
    state_checkpoint_sha256: str | None = None
    if state_potential_checkpoint is not None:
        state_model, state_payload, state_weight = (
            _load_promoted_state_potential(
                state_potential_checkpoint,
                base_path,
                torch_device,
            )
        )
        first_candidate_by_position = {
            position: next(
                index
                for index, mutation in enumerate(candidates)
                if mutation.position == position
            )
            for position in selected_positions
        }
        site_indices = np.asarray(
            [
                first_candidate_by_position[position]
                for position in selected_positions
            ],
            dtype=np.int64,
        )
        site_structure = torch.from_numpy(
            structure[site_indices].astype(np.float32)
        ).to(torch_device)
        site_structure_mask = torch.from_numpy(
            structure_mask[site_indices]
        ).to(torch_device)
        site_membrane = torch.from_numpy(
            membrane[site_indices].astype(np.float32)
        ).to(torch_device)
        with torch.inference_mode():
            potentials = state_model.all_potentials(
                torch.from_numpy(wt.windows.astype(np.float32)).to(
                    torch_device
                ),
                torch.from_numpy(wt.window_mask).to(torch_device),
                torch.from_numpy(
                    np.repeat(
                        wt.global_mean[None],
                        len(selected_positions),
                        axis=0,
                    ).astype(np.float32)
                ).to(torch_device),
                structure=site_structure,
                structure_mask=site_structure_mask,
                membrane=site_membrane,
            ).float()
        amino_acid_index = {
            amino_acid: index
            for index, amino_acid in enumerate(AMINO_ACIDS)
        }
        position_index = {
            position: index
            for index, position in enumerate(selected_positions)
        }
        state_ddg = np.asarray(
            [
                float(
                    (
                        potentials[
                            position_index[mutation.position],
                            amino_acid_index[mutation.mutant],
                        ]
                        - potentials[
                            position_index[mutation.position],
                            amino_acid_index[mutation.wt],
                        ]
                    ).cpu()
                )
                for mutation in candidates
            ],
            dtype=np.float32,
        )
        state_checkpoint_sha256 = file_sha256(
            Path(state_potential_checkpoint)
        )
    if scan_mode == "state-only":
        assert state_ddg is not None
        ddg = state_ddg
        retrieval = -state_ddg
    elif state_ddg is not None:
        assert baseline_ddg is not None and state_weight is not None
        ddg = (1.0 - state_weight) * baseline_ddg + state_weight * state_ddg
        retrieval = -ddg
    else:
        assert baseline_ddg is not None and baseline_retrieval is not None
        ddg = baseline_ddg
        retrieval = baseline_retrieval
    ddg_order = np.argsort(ddg, kind="stable")
    ddg_rank = np.empty(len(ddg_order), dtype=np.int64)
    ddg_rank[ddg_order] = np.arange(1, len(ddg_order) + 1)
    ranking_score = -ddg
    ranking_policy = "ascending signed v2 ddG; negative is stabilizing"
    mptherm: np.ndarray | None = None
    masked_marginal: np.ndarray | None = None
    if legacy_checkpoint_dir is not None:
        if arrays is None or base is None:
            raise ValueError(
                "the retained GPCR consensus requires exact scan mode"
            )
        mptherm_path = Path(legacy_checkpoint_dir) / "mptherm_dtm_head.pt"
        if not mptherm_path.is_file():
            raise FileNotFoundError(mptherm_path)
        center = base.config.window_size // 2
        delta = arrays[1][:, center].astype(np.float32) - arrays[0][
            :, center
        ].astype(np.float32)
        mptherm_model = load_auxiliary_checkpoint(
            mptherm_path, torch_device
        )
        with torch.inference_mode():
            mptherm = (
                mptherm_model(torch.from_numpy(delta).to(torch_device))
                .float()
                .cpu()
                .numpy()
            )
        masked = active_embedder.masked_marginal_log_probabilities(
            wt_sequence,
            selected_positions,
            max_tokens=max_tokens,
            max_batch_size=max_batch_size,
        )
        amino_acid_index = {
            amino_acid: index
            for index, amino_acid in enumerate(AMINO_ACIDS)
        }
        masked_by_position = {
            position: masked[index]
            for index, position in enumerate(selected_positions)
        }
        masked_marginal = np.asarray(
            [
                masked_by_position[mutation.position][
                    amino_acid_index[mutation.mutant]
                ]
                - masked_by_position[mutation.position][
                    amino_acid_index[mutation.wt]
                ]
                for mutation in candidates
            ],
            dtype=np.float32,
        )
        ranking_score = _thermostability_consensus(
            mptherm, masked_marginal
        )
        ranking_policy = (
            "0.80 retained MPTherm delta-Tm percentile + "
            "0.20 ESM-C masked-marginal percentile; v2 signed ddG is "
            "reported as orthogonal generic-stability support"
        )
    order = np.argsort(-ranking_score, kind="stable")
    rank = np.empty(len(order), dtype=np.int64)
    rank[order] = np.arange(1, len(order) + 1)
    rows = [
        {
            "mutation": str(mutation),
            "position": mutation.position,
            "wt": mutation.wt,
            "mutant": mutation.mutant,
            "ddg": float(ddg[index]),
            "retrieval_score": float(retrieval[index]),
            "ddg_rank": int(ddg_rank[index]),
            "stabilizer_rank": int(rank[index]),
            "selection_rank_score": float(ranking_score[index]),
        }
        for index, mutation in enumerate(candidates)
    ]
    if state_ddg is not None:
        for index, row in enumerate(rows):
            row["state_potential_ddg"] = float(state_ddg[index])
            if baseline_ddg is not None:
                row["hierarchy_ddg"] = float(baseline_ddg[index])
    if mptherm is not None and masked_marginal is not None:
        mptherm_percentile = _percentile_ranks(mptherm)
        masked_percentile = _percentile_ranks(masked_marginal)
        for index, row in enumerate(rows):
            row["gpcr_consensus_rank_score"] = float(
                ranking_score[index]
            )
            row["retained_mptherm_delta_tm"] = float(mptherm[index])
            row["retained_mptherm_percentile"] = float(
                mptherm_percentile[index]
            )
            row["masked_marginal_log_odds"] = float(
                masked_marginal[index]
            )
            row["masked_marginal_percentile"] = float(
                masked_percentile[index]
            )
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    best = [rows[index] for index in order[: min(top, len(order))]]
    return {
        "output": str(output_path.resolve()),
        "rows": len(rows),
        "positions": len(selected_positions),
        "top": best,
        "ranking_policy": ranking_policy,
        "scan_mode": scan_mode,
        "topology": topology,
        "structure_provenance": structure_provenance,
        "model": {
            "name": model_name,
            "candidate": (
                state_payload["candidate"]
                if scan_mode == "state-only" and state_payload is not None
                else (
                    payload["candidate"]
                    if state_payload is None and payload is not None
                    else "hierarchy_state_potential_blend"
                )
            ),
            "checkpoint": str(base_path.resolve()),
            "checkpoint_sha256": file_sha256(base_path),
            "state_potential_checkpoint": (
                None
                if state_potential_checkpoint is None
                else str(Path(state_potential_checkpoint).resolve())
            ),
            "state_potential_checkpoint_sha256": state_checkpoint_sha256,
            "state_potential_weight": (
                1.0 if scan_mode == "state-only" else state_weight
            ),
            "legacy_checkpoint_dir": (
                None
                if legacy_checkpoint_dir is None
                else str(Path(legacy_checkpoint_dir).resolve())
            ),
        },
    }
