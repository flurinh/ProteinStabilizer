"""Inference for single and multiple substitutions."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Sequence

import joblib
import numpy as np
import torch

from .data import (
    AMINO_ACIDS,
    EmbeddingRequest,
    Mutation,
    apply_mutations,
    normalize_sequence,
)
from .embeddings import ESMCEmbedder, token_batches
from .models import SingleMutationEnsemble
from .training import (
    gpcr_dtm_adapter_features,
    load_auxiliary_checkpoint,
    load_multi_checkpoint,
    load_scoring_checkpoint,
    membrane_adapter_features,
)

THERMOSTABILITY_SCREENING_WEIGHTS = {
    "mptherm_delta_tm": 0.80,
    "masked_marginal": 0.20,
}


def _percentile_ranks(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError("percentile ranks require a one-dimensional array")
    if not np.all(np.isfinite(values)):
        raise ValueError("percentile ranks require finite values")
    if len(values) == 0:
        return np.empty(0, dtype=np.float32)
    if len(values) == 1:
        return np.ones(1, dtype=np.float32)
    order = np.argsort(values, kind="stable")
    ordered = values[order]
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and ordered[end] == ordered[start]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2
        start = end
    return (ranks / (len(values) - 1)).astype(np.float32)


def _masked_marginal_mutation_scores(
    log_probabilities: np.ndarray,
    mutations: Sequence[Mutation],
) -> np.ndarray:
    if log_probabilities.shape != (len(mutations), len(AMINO_ACIDS)):
        raise ValueError("masked-marginal probability shape mismatch")
    amino_acid_index = {
        amino_acid: index for index, amino_acid in enumerate(AMINO_ACIDS)
    }
    return np.asarray(
        [
            log_probabilities[index, amino_acid_index[mutation.mutant]]
            - log_probabilities[index, amino_acid_index[mutation.wt]]
            for index, mutation in enumerate(mutations)
        ],
        dtype=np.float32,
    )


def _thermostability_consensus(
    mptherm_delta_tm: np.ndarray,
    masked_marginal: np.ndarray,
) -> np.ndarray:
    if mptherm_delta_tm.shape != masked_marginal.shape:
        raise ValueError("thermostability consensus arrays must have equal shape")
    return (
        THERMOSTABILITY_SCREENING_WEIGHTS["mptherm_delta_tm"]
        * _percentile_ranks(mptherm_delta_tm)
        + THERMOSTABILITY_SCREENING_WEIGHTS["masked_marginal"]
        * _percentile_ranks(masked_marginal)
    )


def _single_calibration_features(
    latent: np.ndarray,
    ddg: np.ndarray,
    deltas: np.ndarray,
    checkpoint_dir: Path,
    device: torch.device,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    norm = np.linalg.norm(deltas.astype(np.float32), axis=1)
    representation = np.concatenate(
        [latent, ddg[:, None], norm[:, None]], axis=1
    ).astype(np.float32)
    named = {
        "latent_base": representation,
    }
    auxiliary: dict[str, np.ndarray] = {}
    protherm_path = checkpoint_dir / "protherm_ddg_head.pt"
    mptherm_path = checkpoint_dir / "mptherm_dtm_head.pt"
    if protherm_path.exists() and mptherm_path.exists():
        tensor = torch.from_numpy(deltas.astype(np.float32)).to(device)
        protherm_model = load_auxiliary_checkpoint(protherm_path, device)
        mptherm_model = load_auxiliary_checkpoint(mptherm_path, device)
        with torch.inference_mode():
            protherm = protherm_model(tensor).float().cpu().numpy()
            mptherm = mptherm_model(tensor).float().cpu().numpy()
        auxiliary = {
            "protherm_ddg": protherm,
            "mptherm_delta_tm": mptherm,
        }
        named.update(
            {
                "latent_protherm": np.concatenate(
                    [representation, protherm[:, None]], axis=1
                ),
                "latent_mptherm": np.concatenate(
                    [representation, mptherm[:, None]], axis=1
                ),
                "latent_both": np.concatenate(
                    [representation, protherm[:, None], mptherm[:, None]], axis=1
                ),
            }
        )
        gpcr_dtm_path = checkpoint_dir / "gpcr_dtm.joblib"
        if gpcr_dtm_path.exists():
            gpcr_dtm_payload = joblib.load(gpcr_dtm_path)
            if (
                gpcr_dtm_payload.get("schema")
                != "protein-stabilizer.gpcr-dtm-ridge.v1"
            ):
                raise RuntimeError("GPCR delta-Tm checkpoint schema mismatch")
            gpcr_features = gpcr_dtm_adapter_features(
                str(gpcr_dtm_payload["feature_kind"]),
                deltas,
                latent,
                ddg,
                mptherm,
            )
            gpcr_dtm = gpcr_dtm_payload["model"].predict(gpcr_features)
            auxiliary["gpcr_delta_tm"] = gpcr_dtm
            named.update(
                {
                    "latent_gpcr_dtm": np.concatenate(
                        [representation, gpcr_dtm[:, None]], axis=1
                    ),
                    "latent_gpcr_thermo": np.concatenate(
                        [
                            representation,
                            protherm[:, None],
                            mptherm[:, None],
                            gpcr_dtm[:, None],
                        ],
                        axis=1,
                    ),
                }
            )
    membrane_path = checkpoint_dir / "membrane_ddg.joblib"
    if membrane_path.exists():
        membrane_payload = joblib.load(membrane_path)
        if membrane_payload.get("schema") != "protein-stabilizer.membrane-ddg-ridge.v1":
            raise RuntimeError("membrane ddG checkpoint schema mismatch")
        membrane_features = membrane_adapter_features(
            str(membrane_payload["feature_kind"]), deltas, latent, ddg
        )
        membrane = membrane_payload["model"].predict(membrane_features)
        auxiliary["membrane_ddg"] = membrane
        named["latent_membrane"] = np.concatenate(
            [representation, membrane[:, None]], axis=1
        )
        if "protherm_ddg" in auxiliary and "mptherm_delta_tm" in auxiliary:
            protherm = auxiliary["protherm_ddg"]
            mptherm = auxiliary["mptherm_delta_tm"]
            all_columns = [
                representation,
                protherm[:, None],
                mptherm[:, None],
            ]
            score_columns = [ddg, norm, protherm, mptherm]
            if "gpcr_delta_tm" in auxiliary:
                all_columns.append(auxiliary["gpcr_delta_tm"][:, None])
                score_columns.append(auxiliary["gpcr_delta_tm"])
            all_columns.append(membrane[:, None])
            score_columns.append(membrane)
            named["latent_all"] = np.concatenate(
                all_columns, axis=1
            )
            named["thermodynamic_scores"] = np.stack(score_columns, axis=1)
    elif "protherm_ddg" in auxiliary and "mptherm_delta_tm" in auxiliary:
        score_columns = [
            ddg,
            norm,
            auxiliary["protherm_ddg"],
            auxiliary["mptherm_delta_tm"],
        ]
        if "gpcr_delta_tm" in auxiliary:
            score_columns.append(auxiliary["gpcr_delta_tm"])
        named["thermodynamic_scores"] = np.stack(score_columns, axis=1)
    return named, auxiliary


def predict_mutations(
    sequence: str,
    mutations: Sequence[str | Mutation],
    checkpoint_dir: Path,
    *,
    device: str = "cuda",
    model_name: str = "esmc_600m",
) -> dict[str, object]:
    wt_sequence = normalize_sequence(sequence)
    parsed = [
        mutation if isinstance(mutation, Mutation) else Mutation.parse(mutation)
        for mutation in mutations
    ]
    if not parsed:
        raise ValueError("at least one mutation is required")
    parsed = sorted(parsed, key=lambda mutation: mutation.position)
    joint_sequence = apply_mutations(wt_sequence, parsed)
    single_sequences = [apply_mutations(wt_sequence, [mutation]) for mutation in parsed]
    positions = tuple(mutation.position for mutation in parsed)
    requests = [EmbeddingRequest(wt_sequence, positions)]
    requests.extend(
        EmbeddingRequest(single, (mutation.position,))
        for single, mutation in zip(single_sequences, parsed, strict=True)
    )
    requests.append(EmbeddingRequest(joint_sequence, positions))
    embedder = ESMCEmbedder(model_name=model_name, device=device)
    vectors = embedder.encode(requests)
    masked_log_probabilities = embedder.masked_marginal_log_probabilities(
        wt_sequence, positions
    )
    masked_marginal = _masked_marginal_mutation_scores(
        masked_log_probabilities, parsed
    )
    joint_masked_log_probabilities = (
        embedder.masked_marginal_log_probabilities(joint_sequence, positions)
        if len(parsed) > 1
        else masked_log_probabilities
    )
    joint_masked_marginal = _masked_marginal_mutation_scores(
        joint_masked_log_probabilities, parsed
    )
    wt_vectors = vectors[0].astype(np.float32)
    single_vectors = np.concatenate(vectors[1:-1], axis=0).astype(np.float32)
    joint_vectors = vectors[-1].astype(np.float32)
    single_delta = single_vectors - wt_vectors
    joint_delta = joint_vectors - wt_vectors

    torch_device = torch.device(device if torch.cuda.is_available() else "cpu")
    checkpoint_dir = Path(checkpoint_dir)
    single_head = load_scoring_checkpoint(checkpoint_dir, torch_device)
    single_tensor = torch.from_numpy(single_delta).to(torch_device)
    with torch.inference_mode():
        constituent = single_head(single_tensor).float().cpu().numpy()
        latent = single_head.latent(single_tensor).float().cpu().numpy()
        member_std = (
            single_head.member_predictions(single_tensor)
            .std(dim=0, unbiased=False)
            .float()
            .cpu()
            .numpy()
            if isinstance(single_head, SingleMutationEnsemble)
            else None
        )
    named_features, auxiliary = _single_calibration_features(
        latent, constituent, single_delta, checkpoint_dir, torch_device
    )
    result: dict[str, object] = {
        "mutations": [str(mutation) for mutation in parsed],
        "mutation_count": len(parsed),
        "sign_convention": "negative predicted_ddg is stabilizing",
        "constituent_single_ddg": {
            str(mutation): float(value)
            for mutation, value in zip(parsed, constituent, strict=True)
        },
        "additive_ddg": float(constituent.sum()),
        "constituent_masked_marginal_log_odds": {
            str(mutation): float(value)
            for mutation, value in zip(parsed, masked_marginal, strict=True)
        },
        "additive_masked_marginal_log_odds": float(masked_marginal.sum()),
        "constituent_joint_context_masked_marginal_log_odds": {
            str(mutation): float(value)
            for mutation, value in zip(parsed, joint_masked_marginal, strict=True)
        },
        "joint_context_masked_pseudologlikelihood_log_odds": float(
            joint_masked_marginal.sum()
        ),
        "masked_context_epistasis_log_odds": float(
            joint_masked_marginal.sum() - masked_marginal.sum()
        ),
        "model_provenance": embedder.provenance.canonical_json(),
    }
    if member_std is not None:
        result["ensemble_size"] = len(single_head.members)
        result["constituent_single_ddg_std"] = {
            str(mutation): float(value)
            for mutation, value in zip(parsed, member_std, strict=True)
        }
    if "membrane_ddg" in auxiliary:
        result["membrane_constituent_single_ddg"] = {
            str(mutation): float(value)
            for mutation, value in zip(
                parsed, auxiliary["membrane_ddg"], strict=True
            )
        }
        result["membrane_additive_ddg"] = float(auxiliary["membrane_ddg"].sum())
    if len(parsed) == 1:
        predicted_ddg = float(auxiliary.get("membrane_ddg", constituent)[0])
        result["predicted_ddg"] = predicted_ddg
        result["pretrained_ddg"] = float(constituent[0])
        result["masked_marginal_log_odds"] = float(masked_marginal[0])
        if member_std is not None:
            result["pretrained_ddg_std"] = float(member_std[0])
        if auxiliary:
            if "protherm_ddg" in auxiliary:
                result["protherm_calibrated_ddg"] = float(
                    auxiliary["protherm_ddg"][0]
                )
            if "mptherm_delta_tm" in auxiliary:
                result["mptherm_predicted_delta_tm"] = float(
                    auxiliary["mptherm_delta_tm"][0]
                )
            if "gpcr_delta_tm" in auxiliary:
                result["gpcr_predicted_delta_tm"] = float(
                    auxiliary["gpcr_delta_tm"][0]
                )
            if "membrane_ddg" in auxiliary:
                result["membrane_calibrated_ddg"] = float(
                    auxiliary["membrane_ddg"][0]
                )
        calibration_path = checkpoint_dir / "gpcr_calibration.joblib"
        if calibration_path.exists():
            calibration = joblib.load(calibration_path)
            schema = calibration.get("schema")
            if schema not in {
                "protein-stabilizer.gpcr-ridge.v1",
                "protein-stabilizer.gpcr-ridge.v2",
                "protein-stabilizer.gpcr-ridge.v3",
                "protein-stabilizer.gpcr-ridge.v4",
                "protein-stabilizer.gpcr-ridge.v5",
            }:
                raise RuntimeError("GPCR calibration schema mismatch")
            feature_set = calibration.get("feature_set", "latent_base")
            if feature_set not in named_features:
                raise RuntimeError(f"missing inference features for {feature_set}")
            result["gpcr_stability_rank_score"] = float(
                calibration["pipeline"].predict(named_features[feature_set])[0]
            )
        return result

    multi_head = load_multi_checkpoint(checkpoint_dir / "multi_head.pt", torch_device)
    single_batch = torch.from_numpy(single_delta[None]).to(torch_device)
    joint_batch = torch.from_numpy(joint_delta[None]).to(torch_device)
    with torch.inference_mode():
        _, training_additive, epistasis = multi_head(single_batch, joint_batch)
    ensemble_additive = float(constituent.sum())
    corrected = ensemble_additive + float(epistasis.item())
    result.update(
        {
            "predicted_ddg": corrected,
            "model_additive_ddg": ensemble_additive,
            "epistasis_training_additive_ddg": float(training_additive.item()),
            "epistasis_ddg": float(epistasis.item()),
            "extrapolation_warning": (
                "epistasis head was trained on double mutants"
                if len(parsed) > 2
                else None
            ),
        }
    )
    return result


def screen_single_mutants(
    sequence: str,
    checkpoint_dir: Path,
    *,
    positions: Sequence[int] | None = None,
    output_csv: Path | None = None,
    top: int = 50,
    device: str = "cuda",
    model_name: str = "esmc_600m",
    max_tokens: int = 8192,
    max_batch_size: int = 128,
) -> dict[str, object]:
    """Enumerate and rank all 19 substitutions at selected positions."""

    wt_sequence = normalize_sequence(sequence)
    selected_positions = (
        sorted(set(int(position) for position in positions))
        if positions is not None
        else list(range(1, len(wt_sequence) + 1))
    )
    if not selected_positions or selected_positions[0] < 1 or selected_positions[-1] > len(
        wt_sequence
    ):
        raise ValueError("screen positions must lie inside the sequence")
    mutations: list[Mutation] = []
    requests: list[EmbeddingRequest] = []
    for position in selected_positions:
        wt = wt_sequence[position - 1]
        for mutant in AMINO_ACIDS:
            if mutant == wt:
                continue
            mutation = Mutation(wt=wt, position=position, mutant=mutant)
            mutations.append(mutation)
            requests.append(
                EmbeddingRequest(
                    apply_mutations(wt_sequence, [mutation]), (position,)
                )
            )

    embedder = ESMCEmbedder(model_name=model_name, device=device)
    masked_log_probabilities = embedder.masked_marginal_log_probabilities(
        wt_sequence,
        selected_positions,
        max_tokens=max_tokens,
        max_batch_size=max_batch_size,
    )
    masked_by_position = {
        position: masked_log_probabilities[index]
        for index, position in enumerate(selected_positions)
    }
    wt_vectors = embedder.encode(
        [EmbeddingRequest(wt_sequence, tuple(selected_positions))]
    )[0].astype(np.float32)
    wt_by_position = {
        position: wt_vectors[index]
        for index, position in enumerate(selected_positions)
    }
    mutant_vector_by_hash: dict[str, np.ndarray] = {}
    for batch in token_batches(
        requests, max_tokens=max_tokens, max_batch_size=max_batch_size
    ):
        for request, vector in zip(batch, embedder.encode(batch), strict=True):
            mutant_vector_by_hash[request.sequence_hash] = vector[0].astype(np.float32)
    deltas = np.stack(
        [
            mutant_vector_by_hash[request.sequence_hash]
            - wt_by_position[mutation.position]
            for mutation, request in zip(mutations, requests, strict=True)
        ]
    )

    torch_device = torch.device(device if torch.cuda.is_available() else "cpu")
    checkpoint_dir = Path(checkpoint_dir)
    model = load_scoring_checkpoint(checkpoint_dir, torch_device)
    delta_tensor = torch.from_numpy(deltas).to(torch_device)
    with torch.inference_mode():
        ddg = model(delta_tensor).float().cpu().numpy()
        latent = model.latent(delta_tensor).float().cpu().numpy()
        member_std = (
            model.member_predictions(delta_tensor)
            .std(dim=0, unbiased=False)
            .float()
            .cpu()
            .numpy()
            if isinstance(model, SingleMutationEnsemble)
            else None
        )
    named_features, auxiliary = _single_calibration_features(
        latent, ddg, deltas, checkpoint_dir, torch_device
    )
    amino_acid_index = {
        amino_acid: index for index, amino_acid in enumerate(AMINO_ACIDS)
    }
    masked_marginal = np.asarray(
        [
            masked_by_position[mutation.position][amino_acid_index[mutation.mutant]]
            - masked_by_position[mutation.position][amino_acid_index[mutation.wt]]
            for mutation in mutations
        ],
        dtype=np.float32,
    )
    calibration_path = checkpoint_dir / "gpcr_calibration.joblib"
    gpcr_score: np.ndarray | None = None
    if calibration_path.exists():
        calibration = joblib.load(calibration_path)
        schema = calibration.get("schema")
        if schema not in {
            "protein-stabilizer.gpcr-ridge.v1",
            "protein-stabilizer.gpcr-ridge.v2",
            "protein-stabilizer.gpcr-ridge.v3",
            "protein-stabilizer.gpcr-ridge.v4",
            "protein-stabilizer.gpcr-ridge.v5",
        }:
            raise RuntimeError("GPCR calibration schema mismatch")
        feature_set = calibration.get("feature_set", "latent_base")
        if feature_set not in named_features:
            raise RuntimeError(f"missing inference features for {feature_set}")
        gpcr_score = calibration["pipeline"].predict(named_features[feature_set])

    final_ddg = auxiliary.get("membrane_ddg", ddg)
    general_percentile = _percentile_ranks(-final_ddg)
    masked_percentile = _percentile_ranks(masked_marginal)
    gpcr_percentile = (
        _percentile_ranks(gpcr_score) if gpcr_score is not None else None
    )
    if "mptherm_delta_tm" in auxiliary:
        mptherm_percentile = _percentile_ranks(auxiliary["mptherm_delta_tm"])
        consensus = _thermostability_consensus(
            auxiliary["mptherm_delta_tm"], masked_marginal
        )
        ranking_policy = (
            "0.80 MPTherm delta-Tm percentile + "
            "0.20 ESM-C masked-marginal percentile; "
            "the assay-specific GPCR score is reported separately"
        )
    else:
        mptherm_percentile = None
        consensus = general_percentile
        ranking_policy = "general-stability percentile"

    rows: list[dict[str, object]] = []
    for index, mutation in enumerate(mutations):
        row: dict[str, object] = {
            "mutation": str(mutation),
            "position": mutation.position,
            "wt_aa": mutation.wt,
            "mutant_aa": mutation.mutant,
            "predicted_ddg": float(final_ddg[index]),
            "pretrained_ddg": float(ddg[index]),
            "masked_marginal_log_odds": float(masked_marginal[index]),
            "masked_marginal_percentile": float(masked_percentile[index]),
            "predicted_stabilizing": bool(final_ddg[index] < 0),
            "general_stability_percentile": float(general_percentile[index]),
            "consensus_rank_score": float(consensus[index]),
        }
        if member_std is not None:
            row["pretrained_ddg_std"] = float(member_std[index])
        if gpcr_score is not None:
            row["gpcr_stability_rank_score"] = float(gpcr_score[index])
            row["gpcr_stability_percentile"] = float(gpcr_percentile[index])
        if mptherm_percentile is not None:
            row["mptherm_delta_tm_percentile"] = float(
                mptherm_percentile[index]
            )
        if auxiliary:
            if "protherm_ddg" in auxiliary:
                row["protherm_calibrated_ddg"] = float(
                    auxiliary["protherm_ddg"][index]
                )
            if "mptherm_delta_tm" in auxiliary:
                row["mptherm_predicted_delta_tm"] = float(
                    auxiliary["mptherm_delta_tm"][index]
                )
            if "gpcr_delta_tm" in auxiliary:
                row["gpcr_predicted_delta_tm"] = float(
                    auxiliary["gpcr_delta_tm"][index]
                )
            if "membrane_ddg" in auxiliary:
                row["membrane_calibrated_ddg"] = float(
                    auxiliary["membrane_ddg"][index]
                )
        rows.append(row)
    sort_key = "consensus_rank_score"
    rows.sort(key=lambda row: float(row[sort_key]), reverse=True)
    for rank, row in enumerate(rows, start=1):
        row["rank"] = rank
    if output_csv is not None:
        output_csv = Path(output_csv)
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        with output_csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    return {
        "sequence_length": len(wt_sequence),
        "screened_positions": len(selected_positions),
        "candidate_count": len(rows),
        "ranking_key": sort_key,
        "ranking_policy": ranking_policy,
        "sign_convention": "negative predicted_ddg is stabilizing; higher ranking scores are better",
        "output_csv": str(output_csv) if output_csv is not None else None,
        "top_candidates": rows[: max(1, top)],
        "model_provenance": embedder.provenance.canonical_json(),
        "scoring_ensemble_size": (
            len(model.members) if isinstance(model, SingleMutationEnsemble) else 1
        ),
    }
