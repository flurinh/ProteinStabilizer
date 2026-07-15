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
from .training import load_multi_checkpoint, load_single_checkpoint


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
        calibration_path = checkpoint_dir / "gpcr_calibration.joblib"
        if calibration_path.exists():
            calibration = joblib.load(calibration_path)
            if calibration.get("schema") != "protein-stabilizer.gpcr-ridge.v1":
                raise RuntimeError("GPCR calibration schema mismatch")
            with torch.inference_mode():
                latent = single_head.latent(single_tensor).float().cpu().numpy()
            features = np.concatenate(
                [
                    latent,
                    np.asarray([[predicted_ddg]], dtype=np.float32),
                    np.linalg.norm(single_delta, axis=1, keepdims=True),
                ],
                axis=1,
            )
            result["gpcr_stability_rank_score"] = float(
                calibration["pipeline"].predict(features)[0]
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
    calibration_path = checkpoint_dir / "gpcr_calibration.joblib"
    gpcr_score: np.ndarray | None = None
    if calibration_path.exists():
        calibration = joblib.load(calibration_path)
        features = np.concatenate(
            [
                latent,
                ddg[:, None],
                np.linalg.norm(deltas, axis=1, keepdims=True),
            ],
            axis=1,
        )
        gpcr_score = calibration["pipeline"].predict(features)

    def percentile(values: np.ndarray) -> np.ndarray:
        if len(values) == 1:
            return np.ones(1, dtype=np.float32)
        order = np.argsort(values, kind="stable")
        ranks = np.empty(len(values), dtype=np.float32)
        ranks[order] = np.arange(len(values), dtype=np.float32)
        return ranks / (len(values) - 1)

    general_percentile = percentile(-ddg)
    gpcr_percentile = percentile(gpcr_score) if gpcr_score is not None else None
    consensus = (
        0.75 * general_percentile + 0.25 * gpcr_percentile
        if gpcr_percentile is not None
        else general_percentile
    )

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
            "0.75 general-stability percentile + 0.25 GPCR fine-tune percentile"
            if gpcr_score is not None
            else "general-stability percentile"
        ),
        "sign_convention": "negative predicted_ddg is stabilizing; higher ranking scores are better",
        "output_csv": str(output_csv) if output_csv is not None else None,
        "top_candidates": rows[: max(1, top)],
        "model_provenance": embedder.provenance.canonical_json(),
    }
