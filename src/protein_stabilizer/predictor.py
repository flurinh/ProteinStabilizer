"""Inference for single and multiple substitutions."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import csv
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
from .training import (
    load_auxiliary_checkpoint,
    load_multi_checkpoint,
    load_single_checkpoint,
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
                "thermodynamic_scores": np.stack(
                    [ddg, norm, protherm, mptherm], axis=1
                ),
            }
        )
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
    wt_vectors = vectors[0].astype(np.float32)
    single_vectors = np.concatenate(vectors[1:-1], axis=0).astype(np.float32)
    joint_vectors = vectors[-1].astype(np.float32)
    single_delta = single_vectors - wt_vectors
    joint_delta = joint_vectors - wt_vectors

    torch_device = torch.device(device if torch.cuda.is_available() else "cpu")
    checkpoint_dir = Path(checkpoint_dir)
    single_head = load_single_checkpoint(
        checkpoint_dir / "single_head.pt", torch_device
    )
    single_tensor = torch.from_numpy(single_delta).to(torch_device)
    with torch.inference_mode():
        constituent = single_head(single_tensor).float().cpu().numpy()
    result: dict[str, object] = {
        "mutations": [str(mutation) for mutation in parsed],
        "mutation_count": len(parsed),
        "sign_convention": "negative predicted_ddg is stabilizing",
        "constituent_single_ddg": {
            str(mutation): float(value)
            for mutation, value in zip(parsed, constituent, strict=True)
        },
        "additive_ddg": float(constituent.sum()),
        "model_provenance": embedder.provenance.canonical_json(),
    }
    if len(parsed) == 1:
        predicted_ddg = float(constituent[0])
        result["predicted_ddg"] = predicted_ddg
        with torch.inference_mode():
            latent = single_head.latent(single_tensor).float().cpu().numpy()
        named_features, auxiliary = _single_calibration_features(
            latent, constituent, single_delta, checkpoint_dir, torch_device
        )
        if auxiliary:
            result["protherm_calibrated_ddg"] = float(auxiliary["protherm_ddg"][0])
            result["mptherm_predicted_delta_tm"] = float(
                auxiliary["mptherm_delta_tm"][0]
            )
        calibration_path = checkpoint_dir / "gpcr_calibration.joblib"
        if calibration_path.exists():
            calibration = joblib.load(calibration_path)
            schema = calibration.get("schema")
            if schema not in {
                "protein-stabilizer.gpcr-ridge.v1",
                "protein-stabilizer.gpcr-ridge.v2",
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
        total, additive, epistasis = multi_head(single_batch, joint_batch)
    result.update(
        {
            "predicted_ddg": float(total.item()),
            "model_additive_ddg": float(additive.item()),
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
    model = load_single_checkpoint(checkpoint_dir / "single_head.pt", torch_device)
    delta_tensor = torch.from_numpy(deltas).to(torch_device)
    with torch.inference_mode():
        ddg = model(delta_tensor).float().cpu().numpy()
        latent = model.latent(delta_tensor).float().cpu().numpy()
    named_features, auxiliary = _single_calibration_features(
        latent, ddg, deltas, checkpoint_dir, torch_device
    )
    calibration_path = checkpoint_dir / "gpcr_calibration.joblib"
    gpcr_score: np.ndarray | None = None
    if calibration_path.exists():
        calibration = joblib.load(calibration_path)
        schema = calibration.get("schema")
        if schema not in {
            "protein-stabilizer.gpcr-ridge.v1",
            "protein-stabilizer.gpcr-ridge.v2",
        }:
            raise RuntimeError("GPCR calibration schema mismatch")
        feature_set = calibration.get("feature_set", "latent_base")
        if feature_set not in named_features:
            raise RuntimeError(f"missing inference features for {feature_set}")
        gpcr_score = calibration["pipeline"].predict(named_features[feature_set])

    def percentile(values: np.ndarray) -> np.ndarray:
        if len(values) == 1:
            return np.ones(1, dtype=np.float32)
        order = np.argsort(values, kind="stable")
        ranks = np.empty(len(values), dtype=np.float32)
        ranks[order] = np.arange(len(values), dtype=np.float32)
        return ranks / (len(values) - 1)

    general_percentile = percentile(-ddg)
    gpcr_percentile = percentile(gpcr_score) if gpcr_score is not None else None
    if gpcr_percentile is not None:
        weights = calibration.get(
            "screening_weights",
            {"general_stability": 0.75, "gpcr_calibration": 0.25},
        )
        general_weight = float(weights["general_stability"])
        gpcr_weight = float(weights["gpcr_calibration"])
        consensus = general_weight * general_percentile + gpcr_weight * gpcr_percentile
    else:
        general_weight = 1.0
        gpcr_weight = 0.0
        consensus = general_percentile

    rows: list[dict[str, object]] = []
    for index, mutation in enumerate(mutations):
        row: dict[str, object] = {
            "mutation": str(mutation),
            "position": mutation.position,
            "wt_aa": mutation.wt,
            "mutant_aa": mutation.mutant,
            "predicted_ddg": float(ddg[index]),
            "predicted_stabilizing": bool(ddg[index] < 0),
            "general_stability_percentile": float(general_percentile[index]),
            "consensus_rank_score": float(consensus[index]),
        }
        if gpcr_score is not None:
            row["gpcr_stability_rank_score"] = float(gpcr_score[index])
            row["gpcr_stability_percentile"] = float(gpcr_percentile[index])
        if auxiliary:
            row["protherm_calibrated_ddg"] = float(
                auxiliary["protherm_ddg"][index]
            )
            row["mptherm_predicted_delta_tm"] = float(
                auxiliary["mptherm_delta_tm"][index]
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
        "ranking_policy": (
            f"{general_weight:.2f} general-stability percentile + "
            f"{gpcr_weight:.2f} GPCR fine-tune percentile"
            if gpcr_score is not None
            else "general-stability percentile"
        ),
        "sign_convention": "negative predicted_ddg is stabilizing; higher ranking scores are better",
        "output_csv": str(output_csv) if output_csv is not None else None,
        "top_candidates": rows[: max(1, top)],
        "model_provenance": embedder.provenance.canonical_json(),
    }
