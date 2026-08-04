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
    AFFINE_DDG_CALIBRATION_SCHEMA,
    MULTISCALE_AFFINE_DDG_CALIBRATION_SCHEMA,
    PROTEINMPNN_MULTISCALE_DDG_CALIBRATION_SCHEMA,
    PortablePriorConfig,
    apply_affine_ddg_calibration,
    apply_multiscale_affine_ddg_calibration,
    apply_proteinmpnn_multiscale_ddg_calibration,
    backbone_geometry_features,
    blend_state_and_prior,
    load_portable_prior,
    portable_prior_features,
    predict_portable_prior,
)
from .accuracy_data import (
    load_target_masked_marginals,
    load_target_state_embeddings,
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
from .v2_predictor import (
    _bounded_candidate_indices,
    _compact_position_ranges,
    _embed_requests_cached,
)


ACCURACY_SCREEN_SCHEMA = "protein-stabilizer.accuracy-screen.v1"
SECONDARY_ACCURACY_STATE_SCHEMA = (
    "protein-stabilizer.secondary-accuracy-state.v1"
)


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
    FullStructureStateModel | None,
    object,
    PortablePriorConfig,
    dict[str, object],
    dict[str, object],
    dict[str, object] | None,
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

    secondary_model = None
    secondary_dynamic_provenance = None
    secondary = payload.get("secondary_state")
    calibration = payload.get("ddg_calibration")
    if (
        isinstance(calibration, dict)
        and calibration.get("schema")
        in {
            MULTISCALE_AFFINE_DDG_CALIBRATION_SCHEMA,
            PROTEINMPNN_MULTISCALE_DDG_CALIBRATION_SCHEMA,
        }
    ):
        if (
            not isinstance(secondary, dict)
            or secondary.get("schema") != SECONDARY_ACCURACY_STATE_SCHEMA
        ):
            raise RuntimeError(
                "multiscale calibration lacks its secondary state model"
            )
        secondary_config = secondary.get("state_config")
        if not isinstance(secondary_config, FullStructureConfig):
            secondary_config = FullStructureConfig(**secondary_config)
        if not secondary_config.use_structure:
            raise RuntimeError(
                "secondary accuracy state requires its structure encoder"
            )
        secondary_proteinmpnn, secondary_module, secondary_provenance = (
            load_trainable_proteinmpnn(proteinmpnn_repository)
        )
        secondary_model = FullStructureStateModel(
            MaskedProteinMPNNEncoder(
                secondary_proteinmpnn, secondary_module
            ),
            secondary_config,
        ).to(device)
        secondary_model.load_state_dict(secondary["state_model_state_dict"])
        secondary_model.eval()
        secondary_dynamic_provenance = asdict(secondary_provenance)

    prior_record = payload["portable_prior"]
    recorded_prior = Path(prior_record["model"])
    local_prior = path.parent / recorded_prior.name
    prior_path = local_prior if local_prior.is_file() else recorded_prior
    prior, prior_manifest = load_portable_prior(prior_path)
    if prior_manifest["model_sha256"] != prior_record["model_sha256"]:
        raise RuntimeError("accuracy checkpoint portable-prior hash mismatch")
    return (
        state_model,
        secondary_model,
        prior,
        config,
        payload,
        asdict(dynamic_provenance),
        secondary_dynamic_provenance,
    )


def _validate_esmc_provenance(
    label: str,
    runtime: Mapping[str, object],
    trained: Mapping[str, object],
) -> None:
    numeric_fields = (
        "checkpoint_sha256",
        "embedding_dimension",
        "inference_dtype",
        "storage_dtype",
        "residue_policy",
    )
    mismatched = [
        name for name in numeric_fields if runtime.get(name) != trained.get(name)
    ]
    if mismatched:
        raise RuntimeError(
            f"runtime {label} ESM-C provenance differs from training "
            f"for {mismatched}"
        )


def _validate_runtime_provenance(
    checkpoint: Mapping[str, object],
    state_model_provenance: Mapping[str, object],
    masked_model_provenance: Mapping[str, object],
    structure_embedder: ProteinMPNNBackboneEmbedder,
    dynamic_provenance: Mapping[str, object],
) -> None:
    feature_provenance = checkpoint["feature_provenance"]
    state_model = feature_provenance["hierarchy_rows"][0][
        "hierarchy_provenance"
    ]
    masked_model = feature_provenance["masked_marginal"][
        "model_provenance"
    ]
    for label, runtime, trained in (
        ("state", state_model_provenance, state_model),
        ("masked", masked_model_provenance, masked_model),
    ):
        _validate_esmc_provenance(label, runtime, trained)
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
    embedder: object | None = None,
    masked_marginal_path: Path | None = None,
    embedding_cache: Path | None = None,
    secondary_state_embedding_cache: Path | None = None,
    state_cache_only: bool = False,
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
    if state_cache_only and embedding_cache is None:
        raise ValueError(
            "state-cache-only inference requires an embedding cache"
        )
    active_embedder = (
        None
        if state_cache_only
        else embedder
        or ESMCEmbedder(
            model_name=model_name,
            device=device,
            storage_dtype="float32",
        )
    )
    (
        state_model,
        secondary_state_model,
        prior,
        config,
        checkpoint,
        dynamic_provenance,
        secondary_dynamic_provenance,
    ) = _load_accuracy_checkpoint(
        checkpoint_path,
        proteinmpnn_repository,
        torch_device,
    )
    structure_embedder = ProteinMPNNBackboneEmbedder(
        proteinmpnn_repository,
        device=device,
        storage_dtype="float32",
    )

    if state_cache_only:
        esm_residue, state_cache_provenance = (
            load_target_state_embeddings(
                Path(embedding_cache),
                wt_sequence,
            )
        )
        state_model_provenance = state_cache_provenance[
            "model_provenance"
        ]
        state_embedding_stats = {
            "requested": 1,
            "computed": 0,
            "cache_hits": 1,
        }
    else:
        assert active_embedder is not None
        full_request = EmbeddingRequest(
            wt_sequence, tuple(range(1, len(wt_sequence) + 1))
        )
        embedded, state_embedding_stats = _embed_requests_cached(
            active_embedder,
            [full_request],
            window_radius=4,
            max_tokens=max_tokens,
            max_batch_size=max_batch_size,
            cache_path=embedding_cache,
        )
        esm_residue = np.asarray(
            embedded[0].windows[:, 4], dtype=np.float32
        )
        state_model_provenance = json.loads(
            active_embedder.provenance.canonical_json()
        )
        state_cache_provenance = (
            None
            if embedding_cache is None
            else {
                "path": str(Path(embedding_cache).resolve()),
                "sha256": file_sha256(Path(embedding_cache).resolve()),
                "model_provenance": state_model_provenance,
            }
        )
    secondary_esm_residue = None
    secondary_cache_provenance = None
    if secondary_state_model is not None:
        if secondary_state_embedding_cache is None:
            raise ValueError(
                "multiscale accuracy inference requires a secondary "
                "600M WT embedding cache"
            )
        secondary_esm_residue, secondary_cache_provenance = (
            load_target_state_embeddings(
                Path(secondary_state_embedding_cache),
                wt_sequence,
            )
        )
        secondary_record = checkpoint["secondary_state"]
        _validate_esmc_provenance(
            "secondary state",
            secondary_cache_provenance["model_provenance"],
            secondary_record["embedding_provenance"],
        )
        if (
            secondary_dynamic_provenance is None
            or secondary_dynamic_provenance["checkpoint_sha256"]
            != secondary_record["proteinmpnn_provenance"][
                "checkpoint_sha256"
            ]
        ):
            raise RuntimeError(
                "runtime secondary dynamic ProteinMPNN checkpoint differs "
                "from training"
            )
    if masked_marginal_path is None:
        if active_embedder is None:
            raise ValueError(
                "state-cache-only inference requires a masked-marginal cache"
            )
        masked_by_site = (
            active_embedder.masked_marginal_log_probabilities(
                wt_sequence,
                allowed,
                max_tokens=max_tokens,
                max_batch_size=max_batch_size,
            )
        )
        masked_model_provenance = json.loads(
            active_embedder.provenance.canonical_json()
        )
        masked_cache_provenance = None
    else:
        masked_by_site, masked_cache_provenance = (
            load_target_masked_marginals(
                masked_marginal_path,
                wt_sequence,
                allowed,
            )
        )
        masked_model_provenance = masked_cache_provenance[
            "model_provenance"
        ]
    _validate_runtime_provenance(
        checkpoint,
        state_model_provenance,
        masked_model_provenance,
        structure_embedder,
        dynamic_provenance,
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
    ddg_calibration = checkpoint.get("ddg_calibration")
    proteinmpnn_leave_one_out_ddg = None
    if (
        isinstance(ddg_calibration, dict)
        and ddg_calibration.get("schema")
        == PROTEINMPNN_MULTISCALE_DDG_CALIBRATION_SCHEMA
    ):
        # The portable prior already owns a pinned, frozen ProteinMPNN. Reuse
        # it for one simultaneous leave-one-residue-out 20-state pass. This is
        # a structure computation only: it adds no ESM-C or mutant-sequence
        # encodings.
        static_leave_one_out = MaskedProteinMPNNEncoder(
            structure_embedder.model,
            structure_embedder.module,
        ).to(torch_device)
        static_leave_one_out.eval()
        with torch.inference_mode():
            proteinmpnn_representation = static_leave_one_out(
                torch.from_numpy(coordinates[None]).to(torch_device),
                torch.from_numpy(sequence_tokens[None]).to(torch_device),
                torch.from_numpy(active_structure_mask[None]).to(
                    torch_device
                ),
            )
            proteinmpnn_log_probability = torch.log_softmax(
                structure_embedder.model.W_out(
                    proteinmpnn_representation[..., :128]
                ),
                dim=-1,
            )[..., : len(AMINO_ACIDS)]
        proteinmpnn_values = (
            proteinmpnn_log_probability[0].float().cpu().numpy()
        )
        proteinmpnn_leave_one_out_ddg = np.asarray(
            [
                proteinmpnn_values[
                    mutation.position - 1,
                    amino_acid_index[mutation.wt],
                ]
                - proteinmpnn_values[
                    mutation.position - 1,
                    amino_acid_index[mutation.mutant],
                ]
                for mutation in candidates
            ],
            dtype=np.float32,
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
    secondary_state_ddg = None
    if secondary_state_model is not None:
        assert secondary_esm_residue is not None
        with torch.inference_mode():
            secondary_potential, _ = secondary_state_model.all_potentials(
                torch.from_numpy(secondary_esm_residue[None]).to(
                    torch_device
                ),
                torch.from_numpy(
                    secondary_esm_residue.mean(axis=0, keepdims=True)
                ).to(torch_device),
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
        secondary_values = (
            secondary_potential[0].float().cpu().numpy()
        )
        secondary_state_ddg = np.asarray(
            [
                secondary_values[
                    mutation.position - 1,
                    amino_acid_index[mutation.mutant],
                ]
                - secondary_values[
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
    if ddg_calibration is None:
        expected_ddg = blend_state_and_prior(
            state_ddg, prior_ddg, config=config
        )
    elif (
        ddg_calibration.get("schema")
        == PROTEINMPNN_MULTISCALE_DDG_CALIBRATION_SCHEMA
    ):
        if (
            secondary_state_ddg is None
            or proteinmpnn_leave_one_out_ddg is None
        ):
            raise RuntimeError(
                "ProteinMPNN multiscale calibration lacks a component"
            )
        expected_ddg = apply_proteinmpnn_multiscale_ddg_calibration(
            state_ddg,
            secondary_state_ddg,
            prior_ddg,
            proteinmpnn_leave_one_out_ddg,
            ddg_calibration,
            self_mask=wt == mutant,
        )
    elif (
        ddg_calibration.get("schema")
        == MULTISCALE_AFFINE_DDG_CALIBRATION_SCHEMA
    ):
        if secondary_state_ddg is None:
            raise RuntimeError(
                "multiscale calibration has no secondary state predictions"
            )
        expected_ddg = apply_multiscale_affine_ddg_calibration(
            state_ddg,
            secondary_state_ddg,
            prior_ddg,
            ddg_calibration,
            self_mask=wt == mutant,
        )
    elif ddg_calibration.get("schema") == AFFINE_DDG_CALIBRATION_SCHEMA:
        expected_ddg = apply_affine_ddg_calibration(
            state_ddg,
            prior_ddg,
            ddg_calibration,
            self_mask=wt == mutant,
        )
    else:
        raise RuntimeError("unsupported ddG calibration schema")

    proteinmpnn_ranking = (
        isinstance(ddg_calibration, dict)
        and ddg_calibration.get("schema")
        == PROTEINMPNN_MULTISCALE_DDG_CALIBRATION_SCHEMA
    )
    stabilizer_ranking_ddg = (
        expected_ddg if proteinmpnn_ranking else state_ddg
    )
    stabilizer_order = np.argsort(stabilizer_ranking_ddg, kind="stable")
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
            "stabilizer_rank_score": float(
                -stabilizer_ranking_ddg[index]
            ),
            "stabilizer_rank": int(stabilizer_rank[index]),
            "expected_ddg_rank": int(ddg_rank[index]),
            "component_agreement": bool(agreement[index]),
            "suggestion_eligible": bool(eligible[index]),
        })
        if secondary_state_ddg is not None:
            rows[-1]["secondary_state_ddg"] = float(
                secondary_state_ddg[index]
            )
        if proteinmpnn_leave_one_out_ddg is not None:
            rows[-1]["proteinmpnn_leave_one_out_ddg"] = float(
                proteinmpnn_leave_one_out_ddg[index]
            )
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
                if ddg_calibration is None
                else str(ddg_calibration["formula"])
            ),
            "stabilizer_ranking": (
                "ProteinMPNN-augmented expected ddG; selected because it "
                "improved family-held-out early stabilizer retrieval"
                if proteinmpnn_ranking
                else "exact state score; selected because it retained higher "
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
                "ranked shortlist as prioritization evidence and test "
                "shortlisted "
                "mutations experimentally"
            ),
        },
        "embedding_cost": {
            "full_wt_forward_batches": int(
                state_embedding_stats["computed"]
            ),
            "full_wt_cache_hits": int(
                state_embedding_stats["cache_hits"]
            ),
            "masked_site_contexts": (
                len(allowed) if masked_marginal_path is None else 0
            ),
            "masked_site_cache_hits": (
                0 if masked_marginal_path is None else len(allowed)
            ),
            "masked_forward_batches": int(
                0
                if masked_marginal_path is not None
                else np.ceil(
                    len(allowed)
                    / min(
                        max_batch_size,
                        max(
                            1,
                            max_tokens // (len(wt_sequence) + 2),
                        ),
                    )
                )
            ),
            "mutant_sequence_passes": 0,
            "substitutions_scored_per_site": 19,
            "proteinmpnn_leave_one_out_structure_passes": (
                1 if proteinmpnn_leave_one_out_ddg is not None else 0
            ),
            "state_embedding_cache": state_cache_provenance,
            "secondary_state_embedding_cache": (
                secondary_cache_provenance
            ),
            "masked_marginal_cache": masked_cache_provenance,
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
            "state_model": state_model_provenance["model_name"],
            "masked_model": masked_model_provenance["model_name"],
            "secondary_state_model": (
                None
                if secondary_cache_provenance is None
                else secondary_cache_provenance["model_provenance"][
                    "model_name"
                ]
            ),
            "checkpoint": str(Path(checkpoint_path).resolve()),
            "checkpoint_sha256": file_sha256(
                Path(checkpoint_path).resolve()
            ),
            "configuration": asdict(config),
            "ddg_calibration": ddg_calibration,
            "feature_schema": checkpoint["feature_provenance"]["schema"],
        },
    }
    return report
