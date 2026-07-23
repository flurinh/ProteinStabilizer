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
from .embeddings import (
    ESMCEmbedder,
    HierarchicalEmbedding,
    HierarchyEmbeddingReader,
    HierarchyEmbeddingWriter,
    file_sha256,
    token_batches,
)
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


def _embed_requests_cached(
    embedder: ESMCEmbedder,
    requests: Sequence[EmbeddingRequest],
    *,
    window_radius: int,
    max_tokens: int,
    max_batch_size: int,
    cache_path: Path | None,
) -> tuple[list[HierarchicalEmbedding], dict[str, int]]:
    """Embed requests, optionally reusing a provenance-locked HDF5 cache."""

    if cache_path is None:
        return (
            _embed_requests(
                embedder,
                requests,
                window_radius=window_radius,
                max_tokens=max_tokens,
                max_batch_size=max_batch_size,
            ),
            {
                "requested": len(requests),
                "computed": len(requests),
                "cache_hits": 0,
            },
        )
    provenance = getattr(embedder, "provenance", None)
    if provenance is None:
        raise ValueError("embedding cache requires embedder model provenance")
    cache_path = Path(cache_path)
    with HierarchyEmbeddingWriter(
        cache_path,
        provenance,
        window_radius=window_radius,
    ) as writer:
        missing = writer.missing_requests(requests)
        for batch in token_batches(
            missing,
            max_tokens=max_tokens,
            max_batch_size=max_batch_size,
        ):
            writer.append(
                batch,
                embedder.encode_hierarchy(
                    batch,
                    window_radius=window_radius,
                ),
            )
    results: list[HierarchicalEmbedding] = []
    with HierarchyEmbeddingReader(cache_path) as reader:
        for request in requests:
            features = reader.features(
                [
                    (request.sequence_hash, position)
                    for position in request.positions
                ]
            )
            results.append(
                HierarchicalEmbedding(
                    global_mean=features["global_mean"][0],
                    windows=features["window"],
                    window_mask=features["window_mask"],
                )
            )
    return results, {
        "requested": len(requests),
        "computed": len(missing),
        "cache_hits": len(requests) - len(missing),
    }


def _bounded_candidate_indices(
    order: Sequence[int],
    mutations: Sequence[Mutation],
    limit: int,
    *,
    max_per_site: int | None,
) -> list[int]:
    """Take a score-ordered, site-diverse experimental shortlist."""

    if limit < 1:
        raise ValueError("candidate limit must be positive")
    if max_per_site is not None and max_per_site < 1:
        raise ValueError("max substitutions per site must be positive")
    selected: list[int] = []
    per_site: dict[int, int] = {}
    for raw_index in order:
        index = int(raw_index)
        position = mutations[index].position
        if (
            max_per_site is not None
            and per_site.get(position, 0) >= max_per_site
        ):
            continue
        selected.append(index)
        per_site[position] = per_site.get(position, 0) + 1
        if len(selected) == limit:
            break
    return selected


def _compact_position_ranges(positions: Sequence[int]) -> list[str]:
    """Render sorted one-based positions without expanding long mask ranges."""

    ordered = sorted(set(int(position) for position in positions))
    if not ordered:
        return []
    result: list[str] = []
    start = previous = ordered[0]
    for position in ordered[1:]:
        if position == previous + 1:
            previous = position
            continue
        result.append(str(start) if start == previous else f"{start}-{previous}")
        start = previous = position
    result.append(str(start) if start == previous else f"{start}-{previous}")
    return result


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
    rerank_top: int = 128,
    max_per_site: int | None = None,
    protected_positions: Sequence[int] | None = None,
    protected_reasons: Mapping[int, str] | None = None,
    embedding_cache: Path | None = None,
) -> dict[str, object]:
    """Score allowed substitutions with exact, state-only, or staged inference."""

    wt_sequence = normalize_sequence(sequence)
    requested_positions = tuple(
        range(1, len(wt_sequence) + 1)
        if positions is None
        else sorted(set(int(position) for position in positions))
    )
    if not requested_positions or any(
        position < 1 or position > len(wt_sequence)
        for position in requested_positions
    ):
        raise ValueError("screen positions must be within the sequence")
    protected = tuple(
        sorted(set(int(position) for position in (protected_positions or ())))
    )
    if any(position < 1 or position > len(wt_sequence) for position in protected):
        raise ValueError("protected positions must be within the sequence")
    protected_set = set(protected)
    selected_positions = tuple(
        position
        for position in requested_positions
        if position not in protected_set
    )
    if not selected_positions:
        raise ValueError("the protected mask excludes every requested position")
    normalized_reasons = {
        int(position): str(reason)
        for position, reason in (protected_reasons or {}).items()
    }
    if not set(normalized_reasons).issubset(protected_set):
        raise ValueError("protected reasons contain an unmasked position")
    if top < 1:
        raise ValueError("top must be positive")
    if rerank_top < 1:
        raise ValueError("rerank top must be positive")
    if max_per_site is not None and max_per_site < 1:
        raise ValueError("max substitutions per site must be positive")
    if scan_mode not in {"exact", "state-only", "two-stage"}:
        raise ValueError("scan mode must be exact, state-only, or two-stage")
    if scan_mode in {"state-only", "two-stage"} and state_potential_checkpoint is None:
        raise ValueError(
            f"{scan_mode} scanning requires a state-potential checkpoint"
        )
    if scan_mode == "two-stage" and legacy_checkpoint_dir is not None:
        raise ValueError(
            "two-stage scanning cannot run the retained GPCR consensus; "
            "use the exact mode or omit legacy checkpoints"
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
    embedded, initial_embedding_stats = _embed_requests_cached(
        active_embedder,
        requests,
        window_radius=4,
        max_tokens=max_tokens,
        max_batch_size=max_batch_size,
        cache_path=embedding_cache,
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
    exact_indices: list[int] = []
    rerank_embedding_stats = {
        "requested": 0,
        "computed": 0,
        "cache_hits": 0,
    }
    exact_embeddings: Sequence[HierarchicalEmbedding] = embedded[1:]
    if scan_mode == "exact":
        exact_indices = list(range(len(candidates)))
    elif scan_mode == "two-stage":
        assert state_ddg is not None
        state_order = np.argsort(state_ddg, kind="stable")
        exact_indices = _bounded_candidate_indices(
            state_order,
            candidates,
            min(rerank_top, len(candidates)),
            max_per_site=max_per_site,
        )
        rerank_requests = [
            EmbeddingRequest(
                apply_mutations(wt_sequence, [candidates[index]]),
                (candidates[index].position,),
            )
            for index in exact_indices
        ]
        exact_embeddings, rerank_embedding_stats = _embed_requests_cached(
            active_embedder,
            rerank_requests,
            window_radius=4,
            max_tokens=max_tokens,
            max_batch_size=max_batch_size,
            cache_path=embedding_cache,
        )
    if exact_indices:
        exact_index_array = np.asarray(exact_indices, dtype=np.int64)
        arrays = (
            repeated_wt.windows[exact_index_array],
            np.concatenate(
                [value.windows for value in exact_embeddings], axis=0
            ),
            repeated_wt.window_mask[exact_index_array],
            np.repeat(wt.global_mean[None], len(exact_indices), axis=0),
            np.stack([value.global_mean for value in exact_embeddings]),
        )
        base, payload = load_hierarchical_ensemble(base_path, torch_device)
        with torch.inference_mode():
            heads = base.predict_heads(
                **_torch_state(
                    arrays,
                    structure=structure[exact_index_array],
                    structure_mask=structure_mask[exact_index_array],
                    membrane=membrane[exact_index_array],
                    device=torch_device,
                )
            )
        baseline_ddg = np.full(len(candidates), np.nan, dtype=np.float32)
        baseline_retrieval = np.full(
            len(candidates), np.nan, dtype=np.float32
        )
        baseline_ddg[exact_index_array] = (
            heads["ddg"].float().cpu().numpy()
        )
        baseline_retrieval[exact_index_array] = (
            heads["retrieval"].float().cpu().numpy()
        )
    if scan_mode == "state-only":
        assert state_ddg is not None
        ddg = state_ddg.copy()
        retrieval = -state_ddg
    elif scan_mode == "two-stage":
        assert (
            state_ddg is not None
            and baseline_ddg is not None
            and state_weight is not None
        )
        ddg = state_ddg.copy()
        retrieval = -state_ddg.copy()
        exact_index_array = np.asarray(exact_indices, dtype=np.int64)
        ddg[exact_index_array] = (
            (1.0 - state_weight) * baseline_ddg[exact_index_array]
            + state_weight * state_ddg[exact_index_array]
        )
        retrieval[exact_index_array] = -ddg[exact_index_array]
    elif state_ddg is not None:
        assert baseline_ddg is not None and state_weight is not None
        ddg = (1.0 - state_weight) * baseline_ddg + state_weight * state_ddg
        retrieval = -ddg
    else:
        assert baseline_ddg is not None and baseline_retrieval is not None
        ddg = baseline_ddg
        retrieval = baseline_retrieval
    eligible = np.ones(len(candidates), dtype=bool)
    if scan_mode == "two-stage":
        eligible[:] = False
        eligible[np.asarray(exact_indices, dtype=np.int64)] = True
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
    if scan_mode == "two-stage":
        ranking_policy = (
            f"WT-only state-potential pre-screen followed by exact promoted "
            f"hierarchy/state fusion for {len(exact_indices)} candidates; "
            "only exact-reranked candidates are suggestion-eligible"
        )
    ordered_all = np.argsort(-ranking_score, kind="stable")
    order = np.asarray(
        [index for index in ordered_all if eligible[index]],
        dtype=np.int64,
    )
    rank = np.zeros(len(candidates), dtype=np.int64)
    rank[order] = np.arange(1, len(order) + 1)
    ddg_order = np.asarray(
        [index for index in np.argsort(ddg, kind="stable") if eligible[index]],
        dtype=np.int64,
    )
    ddg_rank = np.zeros(len(candidates), dtype=np.int64)
    ddg_rank[ddg_order] = np.arange(1, len(ddg_order) + 1)
    state_prescreen_rank: np.ndarray | None = None
    if state_ddg is not None:
        state_order = np.argsort(state_ddg, kind="stable")
        state_prescreen_rank = np.empty(len(candidates), dtype=np.int64)
        state_prescreen_rank[state_order] = np.arange(1, len(candidates) + 1)
    rows = [
        {
            "mutation": str(mutation),
            "position": mutation.position,
            "wt": mutation.wt,
            "mutant": mutation.mutant,
            "ddg": float(ddg[index]),
            "retrieval_score": float(retrieval[index]),
            "ddg_rank": int(ddg_rank[index]) if eligible[index] else "",
            "stabilizer_rank": int(rank[index]) if eligible[index] else "",
            "selection_rank_score": float(ranking_score[index]),
            "score_stage": (
                "exact-reranked"
                if scan_mode == "two-stage" and eligible[index]
                else (
                    "state-prescreen"
                    if scan_mode in {"state-only", "two-stage"}
                    else "exact"
                )
            ),
            "suggestion_eligible": bool(eligible[index]),
        }
        for index, mutation in enumerate(candidates)
    ]
    if state_ddg is not None:
        for index, row in enumerate(rows):
            row["state_potential_ddg"] = float(state_ddg[index])
            assert state_prescreen_rank is not None
            row["state_prescreen_rank"] = int(
                state_prescreen_rank[index]
            )
            if baseline_ddg is not None:
                row["hierarchy_ddg"] = (
                    float(baseline_ddg[index])
                    if np.isfinite(baseline_ddg[index])
                    else ""
                )
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
    best_indices = _bounded_candidate_indices(
        order,
        candidates,
        min(top, len(order)),
        max_per_site=max_per_site,
    )
    best = [rows[index] for index in best_indices]
    shortlist_path = output_path.with_name(
        f"{output_path.stem}.shortlist{output_path.suffix or '.csv'}"
    )
    with shortlist_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(best)
    embedding_stats = {
        "cache": (
            None
            if embedding_cache is None
            else str(Path(embedding_cache).resolve())
        ),
        "sequence_embedding_requests": int(
            initial_embedding_stats["requested"]
            + rerank_embedding_stats["requested"]
        ),
        "sequence_embeddings_computed": int(
            initial_embedding_stats["computed"]
            + rerank_embedding_stats["computed"]
        ),
        "cache_hits": int(
            initial_embedding_stats["cache_hits"]
            + rerank_embedding_stats["cache_hits"]
        ),
        "full_exact_mutant_embeddings": len(candidates),
        "mutant_embeddings_requested": len(exact_indices),
        "mutant_embeddings_avoided": len(candidates) - len(exact_indices),
        "masked_site_passes": (
            len(selected_positions) if legacy_checkpoint_dir is not None else 0
        ),
    }
    positions_by_reason: dict[str, list[int]] = {}
    for position, reason in normalized_reasons.items():
        positions_by_reason.setdefault(reason, []).append(position)
    return {
        "output": str(output_path.resolve()),
        "shortlist_output": str(shortlist_path.resolve()),
        "rows": len(rows),
        "positions": len(selected_positions),
        "top": best,
        "max_per_site": max_per_site,
        "ranking_policy": ranking_policy,
        "scan_mode": scan_mode,
        "rerank_top": rerank_top if scan_mode == "two-stage" else None,
        "embedding_cost": embedding_stats,
        "protected_mask": {
            "policy": "hard exclusion before candidate generation and embedding",
            "position_ranges": _compact_position_ranges(protected),
            "count": len(protected),
            "in_requested_scope": len(
                protected_set.intersection(requested_positions)
            ),
            "reasons": [
                {
                    "reason": reason,
                    "position_ranges": _compact_position_ranges(positions),
                }
                for reason, positions in positions_by_reason.items()
            ],
        },
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


def _single_screen_candidates(
    path: Path,
    sequence: str,
    *,
    protected_positions: set[int],
    limit: int,
    ddg_ceiling: float | None,
    require_component_agreement: bool,
) -> tuple[list[Mutation], np.ndarray, dict[str, int]]:
    """Load exact, suggestion-eligible singles for combination design."""

    path = Path(path)
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {
            "mutation",
            "ddg",
            "score_stage",
            "suggestion_eligible",
        }
        if require_component_agreement:
            required.update({"hierarchy_ddg", "state_potential_ddg"})
        missing = required.difference(reader.fieldnames or ())
        if missing:
            raise ValueError(
                "single-screen CSV is missing required columns: "
                + ", ".join(sorted(missing))
            )
        raw_rows = list(reader)
    accepted: dict[Mutation, float] = {}
    excluded_non_exact = 0
    excluded_ineligible = 0
    excluded_protected = 0
    excluded_ceiling = 0
    excluded_component_disagreement = 0
    for row in raw_rows:
        if str(row["score_stage"]).strip() not in {"exact", "exact-reranked"}:
            excluded_non_exact += 1
            continue
        if str(row["suggestion_eligible"]).strip().lower() not in {
            "1",
            "true",
            "yes",
        }:
            excluded_ineligible += 1
            continue
        mutation = Mutation.parse(str(row["mutation"]))
        if (
            mutation.position < 1
            or mutation.position > len(sequence)
            or sequence[mutation.position - 1] != mutation.wt
        ):
            raise ValueError(
                f"single-screen mutation {mutation} does not match the target sequence"
            )
        if mutation.position in protected_positions:
            excluded_protected += 1
            continue
        ddg = float(row["ddg"])
        if not np.isfinite(ddg):
            raise ValueError(f"single-screen mutation {mutation} has non-finite ddG")
        if ddg_ceiling is not None and ddg > ddg_ceiling:
            excluded_ceiling += 1
            continue
        if require_component_agreement:
            hierarchy_ddg = float(row["hierarchy_ddg"])
            state_potential_ddg = float(row["state_potential_ddg"])
            if not np.isfinite(hierarchy_ddg) or not np.isfinite(
                state_potential_ddg
            ):
                raise ValueError(
                    f"single-screen mutation {mutation} has non-finite "
                    "component ddG"
                )
            if hierarchy_ddg > 0.0 or state_potential_ddg > 0.0:
                excluded_component_disagreement += 1
                continue
        previous = accepted.get(mutation)
        if previous is None or ddg < previous:
            accepted[mutation] = ddg
    ordered = sorted(
        accepted.items(),
        key=lambda item: (
            item[1],
            item[0].position,
            item[0].mutant,
        ),
    )[:limit]
    mutations = [item[0] for item in ordered]
    if len({mutation.position for mutation in mutations}) < 2:
        raise ValueError(
            "combination screening requires eligible singles at two distinct sites"
        )
    return (
        mutations,
        np.asarray([item[1] for item in ordered], dtype=np.float32),
        {
            "rows": len(raw_rows),
            "accepted_before_limit": len(accepted),
            "selected": len(mutations),
            "excluded_non_exact": excluded_non_exact,
            "excluded_ineligible": excluded_ineligible,
            "excluded_protected": excluded_protected,
            "excluded_ddg_ceiling": excluded_ceiling,
            "excluded_component_disagreement": (
                excluded_component_disagreement
            ),
        },
    )


def _bounded_pair_indices(
    order: Sequence[int],
    pairs: Sequence[tuple[Mutation, Mutation]],
    limit: int,
    *,
    max_pairs_per_site: int | None,
) -> list[int]:
    """Take a score-ordered pair shortlist with bounded site reuse."""

    if limit < 1:
        raise ValueError("pair candidate limit must be positive")
    if max_pairs_per_site is not None and max_pairs_per_site < 1:
        raise ValueError("maximum pairs per site must be positive")
    selected: list[int] = []
    per_site: dict[int, int] = {}
    for raw_index in order:
        index = int(raw_index)
        positions = tuple(mutation.position for mutation in pairs[index])
        if (
            max_pairs_per_site is not None
            and any(
                per_site.get(position, 0) >= max_pairs_per_site
                for position in positions
            )
        ):
            continue
        selected.append(index)
        for position in positions:
            per_site[position] = per_site.get(position, 0) + 1
        if len(selected) == limit:
            break
    return selected


def screen_hierarchical_double_mutants(
    sequence: str,
    single_screen_path: Path,
    checkpoint_dir: Path,
    output_path: Path,
    *,
    model_name: str = "esmc_600m",
    device: str = "cuda",
    topology: str | None = None,
    generic_numbering: Mapping[int, str] | None = None,
    pdb_path: Path | None = None,
    proteinmpnn_repository: Path | None = None,
    max_tokens: int = 8192,
    max_batch_size: int = 128,
    top: int = 20,
    single_limit: int = 20,
    single_ddg_ceiling: float | None = 0.0,
    require_component_agreement: bool = True,
    pair_rerank_top: int = 64,
    max_pairs_per_site: int | None = 4,
    min_position_separation: int = 1,
    protected_positions: Sequence[int] | None = None,
    protected_reasons: Mapping[int, str] | None = None,
    embedding_cache: Path | None = None,
    embedder: ESMCEmbedder | None = None,
    state_potential_checkpoint: Path | None = None,
) -> dict[str, object]:
    """Design and exactly rerank bounded double-mutant combinations.

    The input must be an exact single-mutant screen. Candidate pairs are
    prescreened by the input additive ddG. Only ``pair_rerank_top`` joint
    sequences receive new contextual embeddings; WT and constituent singles
    are embedded once and reuse the persistent application cache.
    """

    wt_sequence = normalize_sequence(sequence)
    if top < 1:
        raise ValueError("top must be positive")
    if single_limit < 2:
        raise ValueError("single limit must be at least two")
    if pair_rerank_top < 1:
        raise ValueError("pair rerank top must be positive")
    if min_position_separation < 1:
        raise ValueError("minimum position separation must be positive")
    protected = tuple(
        sorted(set(int(position) for position in (protected_positions or ())))
    )
    if any(position < 1 or position > len(wt_sequence) for position in protected):
        raise ValueError("protected positions must be within the sequence")
    protected_set = set(protected)
    normalized_reasons = {
        int(position): str(reason)
        for position, reason in (protected_reasons or {}).items()
    }
    if not set(normalized_reasons).issubset(protected_set):
        raise ValueError("protected reasons contain an unmasked position")

    singles, input_single_ddg, input_stats = _single_screen_candidates(
        single_screen_path,
        wt_sequence,
        protected_positions=protected_set,
        limit=single_limit,
        ddg_ceiling=single_ddg_ceiling,
        require_component_agreement=require_component_agreement,
    )
    pair_candidates: list[tuple[Mutation, Mutation]] = []
    pair_input_additive: list[float] = []
    for first_index, first in enumerate(singles):
        for second_index in range(first_index + 1, len(singles)):
            second = singles[second_index]
            if first.position == second.position:
                continue
            if (
                abs(first.position - second.position)
                < min_position_separation
            ):
                continue
            pair = tuple(
                sorted((first, second), key=lambda mutation: mutation.position)
            )
            pair_candidates.append(pair)
            pair_input_additive.append(
                float(input_single_ddg[first_index] + input_single_ddg[second_index])
            )
    if not pair_candidates:
        raise ValueError("the selected singles produce no valid mutation pairs")
    pair_input_additive_array = np.asarray(
        pair_input_additive, dtype=np.float32
    )
    pair_order = sorted(
        range(len(pair_candidates)),
        key=lambda index: (
            float(pair_input_additive_array[index]),
            str(pair_candidates[index][0]),
            str(pair_candidates[index][1]),
        ),
    )
    exact_indices = pair_order[: min(pair_rerank_top, len(pair_order))]
    exact_pairs = [pair_candidates[index] for index in exact_indices]
    unique_mutations = sorted(
        {mutation for pair in exact_pairs for mutation in pair},
        key=lambda mutation: (mutation.position, mutation.mutant),
    )
    mutation_index = {
        mutation: index for index, mutation in enumerate(unique_mutations)
    }
    positions = tuple(
        sorted({mutation.position for mutation in unique_mutations})
    )

    requests = [EmbeddingRequest(wt_sequence, positions)]
    requests.extend(
        EmbeddingRequest(
            apply_mutations(wt_sequence, [mutation]),
            (mutation.position,),
        )
        for mutation in unique_mutations
    )
    requests.extend(
        EmbeddingRequest(
            apply_mutations(wt_sequence, list(pair)),
            tuple(mutation.position for mutation in pair),
        )
        for pair in exact_pairs
    )
    active_embedder = embedder or ESMCEmbedder(
        model_name=model_name, device=device
    )
    embedded, embedding_stats = _embed_requests_cached(
        active_embedder,
        requests,
        window_radius=4,
        max_tokens=max_tokens,
        max_batch_size=max_batch_size,
        cache_path=embedding_cache,
    )
    wt = embedded[0]
    single_embeddings = embedded[1 : 1 + len(unique_mutations)]
    joint_embeddings = embedded[1 + len(unique_mutations) :]
    wt_position_index = {
        position: index for index, position in enumerate(positions)
    }
    single_arrays = (
        np.stack(
            [
                wt.windows[wt_position_index[mutation.position]]
                for mutation in unique_mutations
            ]
        ),
        np.concatenate(
            [value.windows for value in single_embeddings], axis=0
        ),
        np.stack(
            [
                wt.window_mask[wt_position_index[mutation.position]]
                for mutation in unique_mutations
            ]
        ),
        np.repeat(wt.global_mean[None], len(unique_mutations), axis=0),
        np.stack([value.global_mean for value in single_embeddings]),
    )
    structure, structure_mask, membrane, structure_provenance = (
        _annotation_rows(
            wt_sequence,
            unique_mutations,
            topology=topology,
            generic_numbering=generic_numbering,
            pdb_path=pdb_path,
            proteinmpnn_repository=proteinmpnn_repository,
            device=device,
        )
    )
    torch_device = _device(device)
    base_path = Path(checkpoint_dir) / "hierarchy_selected_ensemble.pt"
    base, base_payload = load_hierarchical_ensemble(base_path, torch_device)
    single_tensors = _torch_state(
        single_arrays,
        structure=structure,
        structure_mask=structure_mask,
        membrane=membrane,
        device=torch_device,
    )
    with torch.inference_mode():
        single_heads = base.predict_heads(**single_tensors)
        baseline_single_latent = base.latent(**single_tensors).float()
    baseline_single_ddg = single_heads["ddg"].float()

    state_single_ddg: torch.Tensor | None = None
    state_single_latent: torch.Tensor | None = None
    state_weight: float | None = None
    state_checkpoint_sha256: str | None = None
    state_payload: dict[str, object] | None = None
    if state_potential_checkpoint is not None:
        state_model, state_payload, state_weight = (
            _load_promoted_state_potential(
                state_potential_checkpoint,
                base_path,
                torch_device,
            )
        )
        wt_amino_acid, mutant_amino_acid = _amino_acid_indices(
            unique_mutations, torch_device
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
        state_checkpoint_sha256 = file_sha256(
            Path(state_potential_checkpoint)
        )

    pair_mutation_indices = np.asarray(
        [
            [mutation_index[mutation] for mutation in pair]
            for pair in exact_pairs
        ],
        dtype=np.int64,
    )
    flattened_mutation_indices = pair_mutation_indices.reshape(-1)
    joint_arrays = (
        np.stack(
            [
                wt.windows[wt_position_index[mutation.position]]
                for pair in exact_pairs
                for mutation in pair
            ]
        ),
        np.concatenate(
            [value.windows for value in joint_embeddings], axis=0
        ),
        np.stack(
            [
                wt.window_mask[wt_position_index[mutation.position]]
                for pair in exact_pairs
                for mutation in pair
            ]
        ),
        np.repeat(wt.global_mean[None], 2 * len(exact_pairs), axis=0),
        np.repeat(
            np.stack([value.global_mean for value in joint_embeddings]),
            2,
            axis=0,
        ),
    )
    joint_tensors = _torch_state(
        joint_arrays,
        structure=structure[flattened_mutation_indices],
        structure_mask=structure_mask[flattened_mutation_indices],
        membrane=membrane[flattened_mutation_indices],
        device=torch_device,
    )
    with torch.inference_mode():
        baseline_joint_latent = base.latent(**joint_tensors).float().reshape(
            len(exact_pairs), 2, -1
        )

    multi_path = Path(checkpoint_dir) / "hierarchy_multi_head.pt"
    if not multi_path.is_file():
        raise FileNotFoundError(multi_path)
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
    pair_index_tensor = torch.from_numpy(pair_mutation_indices).to(torch_device)
    multi_single_ddg = baseline_single_ddg
    multi_single_latent = baseline_single_latent
    multi_joint_latent = baseline_joint_latent
    multi_state_weight: float | None = None
    if multi_payload.get("base_candidate") == "hierarchy_state_potential_blend":
        if state_single_ddg is None or state_single_latent is None:
            raise RuntimeError(
                "promoted multi-mutant head requires state-potential features"
            )
        multi_state_weight = float(multi_payload["state_potential_weight"])
        multi_single_ddg = (
            (1.0 - multi_state_weight) * baseline_single_ddg
            + multi_state_weight * state_single_ddg
        )
        multi_single_latent = torch.cat(
            [baseline_single_latent, state_single_latent], dim=-1
        )
        pair_state_latent = state_single_latent[pair_index_tensor]
        multi_joint_latent = torch.cat(
            [baseline_joint_latent, pair_state_latent], dim=-1
        )
    with torch.inference_mode():
        total_values, additive_values, epistasis_values = multi(
            multi_single_ddg[pair_index_tensor],
            multi_single_latent[pair_index_tensor],
            multi_joint_latent,
        )
    total_exact = total_values.float().cpu().numpy()
    additive_exact = additive_values.float().cpu().numpy()
    epistasis_exact = epistasis_values.float().cpu().numpy()
    constituent_exact = (
        multi_single_ddg[pair_index_tensor].float().cpu().numpy()
    )
    baseline_constituent = (
        baseline_single_ddg[pair_index_tensor].float().cpu().numpy()
    )
    state_constituent = (
        None
        if state_single_ddg is None
        else state_single_ddg[pair_index_tensor].float().cpu().numpy()
    )
    exact_lookup = {
        candidate_index: exact_index
        for exact_index, candidate_index in enumerate(exact_indices)
    }
    exact_total_order = sorted(
        exact_indices,
        key=lambda candidate_index: (
            float(total_exact[exact_lookup[candidate_index]]),
            str(pair_candidates[candidate_index][0]),
            str(pair_candidates[candidate_index][1]),
        ),
    )
    exact_rank = {
        candidate_index: rank
        for rank, candidate_index in enumerate(exact_total_order, start=1)
    }
    prescreen_rank = {
        candidate_index: rank
        for rank, candidate_index in enumerate(pair_order, start=1)
    }
    input_ddg_by_mutation = {
        mutation: float(ddg)
        for mutation, ddg in zip(singles, input_single_ddg, strict=True)
    }
    rows: list[dict[str, object]] = []
    for candidate_index in pair_order:
        first, second = pair_candidates[candidate_index]
        exact_index = exact_lookup.get(candidate_index)
        row: dict[str, object] = {
            "mutation_set": f"{first},{second}",
            "mutation_1": str(first),
            "mutation_2": str(second),
            "position_1": first.position,
            "position_2": second.position,
            "input_single_1_ddg": input_ddg_by_mutation[first],
            "input_single_2_ddg": input_ddg_by_mutation[second],
            "pair_prescreen_additive_ddg": float(
                pair_input_additive_array[candidate_index]
            ),
            "pair_prescreen_rank": prescreen_rank[candidate_index],
            "constituent_1_ddg": "",
            "constituent_2_ddg": "",
            "additive_ddg": "",
            "epistasis_ddg": "",
            "total_ddg": "",
            "retrieval_score": "",
            "total_ddg_rank": "",
            "score_stage": "additive-prescreen",
            "suggestion_eligible": False,
            "constituent_1_hierarchy_ddg": "",
            "constituent_2_hierarchy_ddg": "",
            "constituent_1_state_potential_ddg": "",
            "constituent_2_state_potential_ddg": "",
        }
        if exact_index is not None:
            row.update(
                {
                    "constituent_1_ddg": float(
                        constituent_exact[exact_index, 0]
                    ),
                    "constituent_2_ddg": float(
                        constituent_exact[exact_index, 1]
                    ),
                    "additive_ddg": float(additive_exact[exact_index]),
                    "epistasis_ddg": float(epistasis_exact[exact_index]),
                    "total_ddg": float(total_exact[exact_index]),
                    "retrieval_score": float(-total_exact[exact_index]),
                    "total_ddg_rank": exact_rank[candidate_index],
                    "score_stage": "exact-pair-reranked",
                    "suggestion_eligible": True,
                    "constituent_1_hierarchy_ddg": float(
                        baseline_constituent[exact_index, 0]
                    ),
                    "constituent_2_hierarchy_ddg": float(
                        baseline_constituent[exact_index, 1]
                    ),
                }
            )
            if state_constituent is not None:
                row["constituent_1_state_potential_ddg"] = float(
                    state_constituent[exact_index, 0]
                )
                row["constituent_2_state_potential_ddg"] = float(
                    state_constituent[exact_index, 1]
                )
        rows.append(row)

    row_index_by_candidate = {
        candidate_index: row_index
        for row_index, candidate_index in enumerate(pair_order)
    }
    shortlist_candidate_indices = _bounded_pair_indices(
        exact_total_order,
        pair_candidates,
        min(top, len(exact_total_order)),
        max_pairs_per_site=max_pairs_per_site,
    )
    shortlist = [
        rows[row_index_by_candidate[candidate_index]]
        for candidate_index in shortlist_candidate_indices
    ]
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    shortlist_path = output_path.with_name(
        f"{output_path.stem}.shortlist{output_path.suffix or '.csv'}"
    )
    with shortlist_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(shortlist)

    positions_by_reason: dict[str, list[int]] = {}
    for position, reason in normalized_reasons.items():
        positions_by_reason.setdefault(reason, []).append(position)
    return {
        "output": str(output_path.resolve()),
        "shortlist_output": str(shortlist_path.resolve()),
        "rows": len(rows),
        "exact_pairs": len(exact_pairs),
        "top": shortlist,
        "ranking_policy": (
            "input exact-single additive prescreen followed by promoted "
            "permutation-invariant double-mutant epistasis; ascending total "
            "ddG, negative is stabilizing"
        ),
        "ddg_units": "kcal/mol",
        "sign_convention": "negative is stabilizing",
        "single_screen": {
            "path": str(Path(single_screen_path).resolve()),
            "sha256": file_sha256(Path(single_screen_path)),
            "limit": single_limit,
            "ddg_ceiling": single_ddg_ceiling,
            "require_component_agreement": require_component_agreement,
            **input_stats,
        },
        "pair_search": {
            "candidate_pairs": len(pair_candidates),
            "pair_rerank_top": pair_rerank_top,
            "min_position_separation": min_position_separation,
            "max_pairs_per_site": max_pairs_per_site,
        },
        "embedding_cost": {
            "cache": (
                None
                if embedding_cache is None
                else str(Path(embedding_cache).resolve())
            ),
            "sequence_embedding_requests": embedding_stats["requested"],
            "sequence_embeddings_computed": embedding_stats["computed"],
            "cache_hits": embedding_stats["cache_hits"],
            "unique_single_mutant_embeddings": len(unique_mutations),
            "joint_pair_embeddings_requested": len(exact_pairs),
            "joint_pair_embeddings_avoided": (
                len(pair_candidates) - len(exact_pairs)
            ),
            "constituent_embeddings_reused": (
                2 * len(exact_pairs) - len(unique_mutations)
            ),
            "naive_exact_pair_mutant_embeddings": 3 * len(exact_pairs),
            "optimized_exact_pair_mutant_embeddings": (
                len(unique_mutations) + len(exact_pairs)
            ),
        },
        "protected_mask": {
            "policy": "hard exclusion before pair generation and embedding",
            "position_ranges": _compact_position_ranges(protected),
            "count": len(protected),
            "reasons": [
                {
                    "reason": reason,
                    "position_ranges": _compact_position_ranges(positions),
                }
                for reason, positions in positions_by_reason.items()
            ],
        },
        "topology": topology,
        "structure_provenance": structure_provenance,
        "model": {
            "name": model_name,
            "candidate": base_payload["candidate"],
            "checkpoint": str(base_path.resolve()),
            "checkpoint_sha256": file_sha256(base_path),
            "multi_checkpoint": str(multi_path.resolve()),
            "multi_checkpoint_sha256": file_sha256(multi_path),
            "multi_state_potential_weight": multi_state_weight,
            "state_potential_checkpoint": (
                None
                if state_potential_checkpoint is None
                else str(Path(state_potential_checkpoint).resolve())
            ),
            "state_potential_checkpoint_sha256": state_checkpoint_sha256,
            "state_potential_weight": state_weight,
            "multi_training_scope": "double mutants",
        },
    }
