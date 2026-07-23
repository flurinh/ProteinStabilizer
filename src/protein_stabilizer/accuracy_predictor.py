"""Low-cost application screening with the promoted accuracy ensemble."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch

from .accuracy import (
    PortablePriorConfig,
    backbone_geometry_features,
    blend_state_and_prior,
    load_portable_prior,
    portable_prior_features,
    predict_portable_prior,
)
from .accuracy_training import ACCURACY_ENSEMBLE_SCHEMA
from .data import (
    AMINO_ACIDS,
    EmbeddingRequest,
    Mutation,
    normalize_sequence,
)
from .embeddings import ESMCEmbedder, file_sha256
from .full_structure import (
    FullStructureConfig,
    FullStructureStateModel,
    MaskedProteinMPNNEncoder,
    load_trainable_proteinmpnn,
)
from .structure import ProteinMPNNBackboneEmbedder
from .v2_predictor import _bounded_candidate_indices, _compact_position_ranges


ACCURACY_SCREEN_SCHEMA = "protein-stabilizer.accuracy-screen.v1"


def _device(name: str) -> torch.device:
    return torch.device(
        name if name.startswith("cuda") and torch.cuda.is_available() else "cpu"
    )


def _load_accuracy_checkpoint(
    checkpoint_path: Path,
    proteinmpnn_repository: Path,
    device: torch.device,
) -> tuple[
    FullStructureStateModel,
    object,
    PortablePriorConfig,
    dict[str, object],
    dict[str, object],
]:
    path = Path(checkpoint_path).resolve()
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("schema") != ACCURACY_ENSEMBLE_SCHEMA:
        raise RuntimeError("accuracy ensemble checkpoint schema mismatch")
    config = payload["config"]
    if not isinstance(config, PortablePriorConfig):
        config = PortablePriorConfig(**config)
    state_config = payload["state_config"]
    if not isinstance(state_config, FullStructureConfig):
        state_config = FullStructureConfig(**state_config)
    if not state_config.use_structure:
        raise RuntimeError("accuracy ensemble requires its structure state")

    proteinmpnn, module, dynamic_provenance = load_trainable_proteinmpnn(
        proteinmpnn_repository
    )
    state_model = FullStructureStateModel(
        MaskedProteinMPNNEncoder(proteinmpnn, module),
        state_config,
    ).to(device)
    state_model.load_state_dict(payload["state_model_state_dict"])
    state_model.eval()

    prior_record = payload["portable_prior"]
    recorded_prior = Path(prior_record["model"])
    local_prior = path.parent / recorded_prior.name
    prior_path = local_prior if local_prior.is_file() else recorded_prior
    prior, prior_manifest = load_portable_prior(prior_path)
    if prior_manifest["model_sha256"] != prior_record["model_sha256"]:
        raise RuntimeError("accuracy checkpoint portable-prior hash mismatch")
    return (
        state_model,
        prior,
        config,
        payload,
        asdict(dynamic_provenance),
    )


def _validate_runtime_provenance(
    checkpoint: Mapping[str, object],
    embedder: ESMCEmbedder,
    structure_embedder: ProteinMPNNBackboneEmbedder,
    dynamic_provenance: Mapping[str, object],
) -> None:
    feature_provenance = checkpoint["feature_provenance"]
    masked_model = feature_provenance["masked_marginal"][
        "model_provenance"
    ]
    if embedder.provenance.checkpoint_sha256 != masked_model[
        "checkpoint_sha256"
    ]:
        raise RuntimeError("runtime ESM-C checkpoint differs from training")
    row_provenance = feature_provenance["hierarchy_rows"][0][
        "structure_provenance"
    ]["proteinmpnn"]
    static = json.loads(structure_embedder.provenance.canonical_json())
    if static["checkpoint_sha256"] != row_provenance[
        "checkpoint_sha256"
    ]:
        raise RuntimeError(
            "runtime static ProteinMPNN checkpoint differs from training"
        )
    trained_dynamic = checkpoint["training"]["structure"][
        "proteinmpnn_provenance"
    ]
    if dynamic_provenance["checkpoint_sha256"] != trained_dynamic[
        "checkpoint_sha256"
    ]:
        raise RuntimeError(
            "runtime dynamic ProteinMPNN checkpoint differs from training"
        )


def screen_accuracy_single_mutants(
    sequence: str,
    checkpoint_path: Path,
    output_path: Path,
    *,
    pdb_path: Path,
    proteinmpnn_repository: Path,
    positions: Sequence[int] | None = None,
    protected_positions: Sequence[int] | None = None,
    protected_reasons: Mapping[int, str] | None = None,
    structure_residue_mask: np.ndarray | None = None,
    structure_source_provenance: Mapping[str, object] | None = None,
    model_name: str = "esmc_600m",
    device: str = "cuda",
    max_tokens: int = 8192,
    max_batch_size: int = 128,
    top: int = 50,
    max_per_site: int | None = None,
    require_component_agreement: bool = True,
    embedder: ESMCEmbedder | None = None,
) -> dict[str, object]:
    """Screen all substitutions with separate ddG and retrieval outputs."""

    wt_sequence = normalize_sequence(sequence)
    requested = tuple(
        range(1, len(wt_sequence) + 1)
        if positions is None
        else sorted(set(int(position) for position in positions))
    )
    if not requested or any(
        position < 1 or position > len(wt_sequence)
        for position in requested
    ):
        raise ValueError("screen positions must be within the sequence")
    protected = tuple(
        sorted(set(int(value) for value in (protected_positions or ())))
    )
    if any(value < 1 or value > len(wt_sequence) for value in protected):
        raise ValueError("protected positions must be within the sequence")
    allowed = tuple(
        position for position in requested if position not in set(protected)
    )
    if not allowed:
        raise ValueError("the protected mask excludes every requested site")
    reasons = {
        int(position): str(reason)
        for position, reason in (protected_reasons or {}).items()
    }
    if not set(reasons).issubset(protected):
        raise ValueError("protected reasons contain an unmasked position")
    if top < 1:
        raise ValueError("top must be positive")
    if max_per_site is not None and max_per_site < 1:
        raise ValueError("max substitutions per site must be positive")

    candidates = [
        Mutation(wt_sequence[position - 1], position, amino_acid)
        for position in allowed
        for amino_acid in AMINO_ACIDS
        if amino_acid != wt_sequence[position - 1]
    ]
    torch_device = _device(device)
    if torch_device.type == "cuda":
        torch.set_float32_matmul_precision("highest")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    active_embedder = embedder or ESMCEmbedder(
        model_name=model_name,
        device=device,
        storage_dtype="float32",
    )
    state_model, prior, config, checkpoint, dynamic_provenance = (
        _load_accuracy_checkpoint(
            checkpoint_path,
            proteinmpnn_repository,
            torch_device,
        )
    )
    structure_embedder = ProteinMPNNBackboneEmbedder(
        proteinmpnn_repository,
        device=device,
        storage_dtype="float32",
    )
    _validate_runtime_provenance(
        checkpoint,
        active_embedder,
        structure_embedder,
        dynamic_provenance,
    )

    full_request = EmbeddingRequest(
        wt_sequence, tuple(range(1, len(wt_sequence) + 1))
    )
    esm_residue = active_embedder.encode([full_request])[0].astype(
        np.float32
    )
    masked_by_site = active_embedder.masked_marginal_log_probabilities(
        wt_sequence,
        allowed,
        max_tokens=max_tokens,
        max_batch_size=max_batch_size,
    )
    coordinates, coordinate_mask, chain = (
        structure_embedder.backbone_coordinates(
            Path(pdb_path), wt_sequence
        )
    )
    active_structure_mask = coordinate_mask.copy()
    if structure_residue_mask is not None:
        source_mask = np.asarray(structure_residue_mask, dtype=bool)
        if source_mask.shape != (len(wt_sequence),):
            raise ValueError(
                "structure residue mask must have one value per residue"
            )
        active_structure_mask &= source_mask
    static_proteinmpnn = structure_embedder.encode(
        Path(pdb_path),
        wt_sequence,
        residue_mask=active_structure_mask,
    )

    amino_acid_index = {
        amino_acid: index
        for index, amino_acid in enumerate(AMINO_ACIDS)
    }
    sequence_tokens = np.asarray(
        [amino_acid_index[value] for value in wt_sequence],
        dtype=np.int64,
    )
    with torch.inference_mode():
        potential, _ = state_model.all_potentials(
            torch.from_numpy(esm_residue[None]).to(torch_device),
            torch.from_numpy(esm_residue.mean(axis=0, keepdims=True)).to(
                torch_device
            ),
            torch.from_numpy(coordinates[None]).to(torch_device),
            torch.from_numpy(sequence_tokens[None]).to(torch_device),
            torch.ones(
                (1, len(wt_sequence)),
                dtype=torch.bool,
                device=torch_device,
            ),
            structure_mask=torch.from_numpy(
                active_structure_mask[None]
            ).to(torch_device),
        )
    state_values = potential[0].float().cpu().numpy()
    state_ddg = np.asarray(
        [
            state_values[
                mutation.position - 1,
                amino_acid_index[mutation.mutant],
            ]
            - state_values[
                mutation.position - 1,
                amino_acid_index[mutation.wt],
            ]
            for mutation in candidates
        ],
        dtype=np.float32,
    )

    site_index = {position: index for index, position in enumerate(allowed)}
    masked_rows = np.stack(
        [masked_by_site[site_index[value.position]] for value in candidates]
    )
    position = np.asarray(
        [value.position - 1 for value in candidates], dtype=np.int64
    )
    wt = np.asarray(
        [amino_acid_index[value.wt] for value in candidates],
        dtype=np.int64,
    )
    mutant = np.asarray(
        [amino_acid_index[value.mutant] for value in candidates],
        dtype=np.int64,
    )
    geometry = backbone_geometry_features(
        coordinates[None],
        active_structure_mask[None],
        np.asarray([len(wt_sequence)]),
    )
    features = portable_prior_features(
        masked_rows,
        wt,
        mutant,
        position,
        np.zeros(len(candidates), dtype=np.int64),
        np.asarray([len(wt_sequence)]),
        geometry,
        static_proteinmpnn[None],
    )
    prior_ddg = predict_portable_prior(prior, features, wt, mutant)
    expected_ddg = blend_state_and_prior(
        state_ddg, prior_ddg, config=config
    )

    stabilizer_order = np.argsort(state_ddg, kind="stable")
    ddg_order = np.argsort(expected_ddg, kind="stable")
    stabilizer_rank = np.empty(len(candidates), dtype=np.int64)
    stabilizer_rank[stabilizer_order] = np.arange(1, len(candidates) + 1)
    ddg_rank = np.empty(len(candidates), dtype=np.int64)
    ddg_rank[ddg_order] = np.arange(1, len(candidates) + 1)
    agreement = (state_ddg < 0.0) & (expected_ddg < 0.0)
    eligible = (
        agreement
        if require_component_agreement
        else np.ones(len(candidates), dtype=bool)
    )
    rows = []
    for index, mutation in enumerate(candidates):
        rows.append({
            "mutation": str(mutation),
            "position": mutation.position,
            "wt": mutation.wt,
            "mutant": mutation.mutant,
            "expected_general_ddg": float(expected_ddg[index]),
            "ddg": float(expected_ddg[index]),
            "state_ddg": float(state_ddg[index]),
            "portable_prior_ddg": float(prior_ddg[index]),
            "stabilizer_rank_score": float(-state_ddg[index]),
            "stabilizer_rank": int(stabilizer_rank[index]),
            "expected_ddg_rank": int(ddg_rank[index]),
            "component_agreement": bool(agreement[index]),
            "suggestion_eligible": bool(eligible[index]),
        })
    eligible_order = [
        int(index) for index in stabilizer_order if eligible[index]
    ]
    shortlist_index = _bounded_candidate_indices(
        eligible_order,
        candidates,
        min(top, len(eligible_order)),
        max_per_site=max_per_site,
    ) if eligible_order else []
    shortlist = [rows[index] for index in shortlist_index]
    output = Path(output_path).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    shortlist_path = output.with_name(
        f"{output.stem}.shortlist{output.suffix or '.csv'}"
    )
    with shortlist_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(shortlist)

    positions_by_reason: dict[str, list[int]] = {}
    for position_value, reason in reasons.items():
        positions_by_reason.setdefault(reason, []).append(position_value)
    report = {
        "schema": ACCURACY_SCREEN_SCHEMA,
        "output": str(output),
        "shortlist_output": str(shortlist_path),
        "rows": len(rows),
        "positions": len(allowed),
        "suggestion_eligible_rows": int(eligible.sum()),
        "top": shortlist,
        "max_per_site": max_per_site,
        "require_component_agreement": require_component_agreement,
        "sign_convention": "negative ddG means stabilizing",
        "routing": {
            "expected_general_ddg_kcal_mol": (
                "0.60 exact state ddG + 0.40 portable prior ddG"
            ),
            "stabilizer_ranking": (
                "exact state score; selected because it retained higher "
                "family-held-out average precision"
            ),
            "suggestion_policy": (
                "require state ranking component and general ddG ensemble "
                "to both predict stabilization"
                if require_component_agreement
                else "rank by state score without component agreement"
            ),
        },
        "applicability": {
            "training_domain": checkpoint.get(
                "training_domain",
                "MegaScale proteins; family-held-out regression evaluation",
            ),
            "input_sequence_length": len(wt_sequence),
            "outside_training_length_range": (
                len(wt_sequence)
                > int(
                    checkpoint.get("training_domain", {})
                    .get("sequence_length", {})
                    .get("maximum", 72)
                )
            ),
            "gpcr_warning": (
                "GPCR expected general ddG is a domain-shift extrapolation, "
                "not a GPCR-calibrated kcal/mol measurement; treat the "
                "state score as ranking evidence and test shortlisted "
                "mutations experimentally"
            ),
        },
        "embedding_cost": {
            "full_wt_forward_batches": 1,
            "masked_site_contexts": len(allowed),
            "masked_forward_batches": int(
                np.ceil(
                    len(allowed)
                    / min(
                        max_batch_size,
                        max(1, max_tokens // (len(wt_sequence) + 2)),
                    )
                )
            ),
            "mutant_sequence_passes": 0,
            "substitutions_scored_per_site": 19,
        },
        "protected_mask": {
            "policy": "hard exclusion before embedding and candidate generation",
            "position_ranges": _compact_position_ranges(protected),
            "count": len(protected),
            "reasons": [
                {
                    "reason": reason,
                    "position_ranges": _compact_position_ranges(values),
                }
                for reason, values in positions_by_reason.items()
            ],
        },
        "structure": {
            "pdb_path": str(Path(pdb_path).resolve()),
            "pdb_sha256": file_sha256(Path(pdb_path)),
            "chain": chain,
            "usable_residues": int(active_structure_mask.sum()),
            "source": (
                None
                if structure_source_provenance is None
                else dict(structure_source_provenance)
            ),
        },
        "model": {
            "name": model_name,
            "checkpoint": str(Path(checkpoint_path).resolve()),
            "checkpoint_sha256": file_sha256(
                Path(checkpoint_path).resolve()
            ),
            "configuration": asdict(config),
            "feature_schema": checkpoint["feature_provenance"]["schema"],
        },
    }
    return report
