"""Optional ESM-C 6B inference and GPCR screen reranking."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import torch

from .data import (
    AMINO_ACIDS,
    EmbeddingRequest,
    Mutation,
    apply_mutations,
    normalize_sequence,
)
from .embeddings import ESMCProvenance, file_sha256, token_batches
from .models import SingleMutationEnsemble
from .predictor import (
    DUAL_BACKBONE_SCREENING_WEIGHTS,
    _masked_marginal_mutation_scores,
    dual_backbone_thermostability_consensus,
)
from .training import load_auxiliary_checkpoint, load_scoring_checkpoint


DEFAULT_ESMC6B_MODEL = "biohub/ESMC-6B"


def _resolved_checkpoint_file(model_name_or_path: str) -> Path:
    local = Path(model_name_or_path).expanduser()
    filenames = (
        "model.safetensors.index.json",
        "model.safetensors",
        "pytorch_model.bin.index.json",
        "pytorch_model.bin",
    )
    if local.is_dir():
        for filename in filenames:
            candidate = local / filename
            if candidate.is_file():
                return candidate.absolute()
        raise FileNotFoundError(
            f"no supported model weight or index file found under {local}"
        )
    try:
        from transformers.utils.hub import cached_file
    except ImportError as exc:
        raise RuntimeError(
            "ESM-C 6B requires transformers==4.57.6; "
            "install the project with the esmc6b extra in a separate environment"
        ) from exc
    for filename in filenames:
        try:
            resolved = cached_file(model_name_or_path, filename)
        except Exception:
            resolved = None
        if resolved:
            return Path(resolved).resolve()
    raise FileNotFoundError(
        f"could not resolve ESM-C 6B weights for {model_name_or_path!r}"
    )


class ESMC6BEmbedder:
    """Hugging Face ESM-C adapter returning final-layer residue vectors."""

    def __init__(
        self,
        model_name_or_path: str = DEFAULT_ESMC6B_MODEL,
        device: str = "cuda",
    ) -> None:
        try:
            import transformers
            from transformers import AutoModelForMaskedLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "ESM-C 6B dependencies are missing; create a separate environment "
                "and install with `pip install -e '.[esmc6b]'`"
            ) from exc
        version = tuple(
            int(part)
            for part in transformers.__version__.split(".")[:2]
            if part.isdigit()
        )
        if version < (4, 57):
            raise RuntimeError(
                "ESM-C 6B requires transformers>=4.57; use the separate 6B environment"
            )
        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")
        self.device = torch.device(device)
        self.model_name = model_name_or_path
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            trust_remote_code=True,
        )
        inference_dtype = torch.bfloat16 if self.device.type == "cuda" else torch.float32
        self.model = AutoModelForMaskedLM.from_pretrained(
            model_name_or_path,
            trust_remote_code=True,
            dtype=inference_dtype,
            attn_implementation="sdpa",
        ).to(self.device).eval()
        dimension = getattr(
            self.model.config,
            "hidden_size",
            getattr(self.model.config, "d_model", None),
        )
        if dimension is None:
            raise RuntimeError("ESM-C 6B configuration has no embedding dimension")
        self.dimension = int(dimension)
        checkpoint = _resolved_checkpoint_file(model_name_or_path)
        self.provenance = ESMCProvenance(
            model_name=model_name_or_path,
            embedding_dimension=self.dimension,
            package_version=(
                f"transformers={transformers.__version__};"
                f"torch={torch.__version__}"
            ),
            checkpoint_path=str(checkpoint.parent),
            checkpoint_sha256=file_sha256(checkpoint),
            inference_dtype=(
                "bfloat16" if inference_dtype == torch.bfloat16 else "float32"
            ),
        )

    def _tokenize(self, sequences: Sequence[str]) -> dict[str, torch.Tensor]:
        normalized = [normalize_sequence(sequence) for sequence in sequences]
        encoded = self.tokenizer(
            normalized,
            add_special_tokens=True,
            padding=True,
            truncation=False,
            return_tensors="pt",
        )
        expected_lengths = torch.tensor(
            [len(sequence) + 2 for sequence in normalized]
        )
        if not torch.equal(encoded["attention_mask"].sum(dim=1), expected_lengths):
            raise RuntimeError("ESM-C 6B tokenizer truncated or misaligned a sequence")
        return {
            name: value.to(self.device)
            for name, value in encoded.items()
            if isinstance(value, torch.Tensor)
        }

    def encode(self, requests: Sequence[EmbeddingRequest]) -> list[np.ndarray]:
        if not requests:
            return []
        encoded = self._tokenize([request.sequence for request in requests])
        with torch.inference_mode():
            output = self.model(**encoded)
        hidden = output.last_hidden_state
        results: list[np.ndarray] = []
        for row, request in enumerate(requests):
            selected = hidden[row, list(request.positions)]
            if selected.shape != (len(request.positions), self.dimension):
                raise RuntimeError("unexpected ESM-C 6B residue embedding shape")
            results.append(selected.float().cpu().numpy().astype(np.float16))
        return results

    def masked_marginal_log_probabilities(
        self,
        sequence: str,
        positions: Sequence[int],
        *,
        max_tokens: int = 4096,
        max_batch_size: int = 16,
    ) -> np.ndarray:
        normalized = normalize_sequence(sequence)
        selected = tuple(int(position) for position in positions)
        if len(selected) != len(set(selected)):
            raise ValueError("masked-marginal positions must be unique")
        if any(position < 1 or position > len(normalized) for position in selected):
            raise ValueError("masked-marginal position is outside the sequence")
        if max_tokens < len(normalized) + 2:
            raise ValueError("max_tokens is smaller than one tokenized sequence")
        if max_batch_size < 1:
            raise ValueError("max_batch_size must be positive")
        if not selected:
            return np.empty((0, len(AMINO_ACIDS)), dtype=np.float32)

        amino_acid_ids = [
            int(self.tokenizer.convert_tokens_to_ids(amino_acid))
            for amino_acid in AMINO_ACIDS
        ]
        batch_size = min(
            max_batch_size,
            max(1, max_tokens // (len(normalized) + 2)),
        )
        results: list[np.ndarray] = []
        for start in range(0, len(selected), batch_size):
            batch_positions = selected[start : start + batch_size]
            encoded = self._tokenize([normalized] * len(batch_positions))
            rows = torch.arange(len(batch_positions), device=self.device)
            token_positions = torch.tensor(batch_positions, device=self.device)
            encoded["input_ids"][rows, token_positions] = self.tokenizer.mask_token_id
            with torch.inference_mode():
                output = self.model(**encoded)
                site_logits = output.logits[rows, token_positions]
                log_probabilities = torch.log_softmax(
                    site_logits.float(), dim=-1
                )[:, amino_acid_ids]
            results.append(log_probabilities.cpu().numpy().astype(np.float32))
        return np.concatenate(results, axis=0)


def _torch_prediction(
    model: torch.nn.Module,
    delta: np.ndarray,
    *,
    device: torch.device,
    batch_size: int = 512,
) -> np.ndarray:
    outputs: list[np.ndarray] = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(delta), batch_size):
            tensor = torch.from_numpy(
                delta[start : start + batch_size].astype(np.float32)
            ).to(device)
            outputs.append(model(tensor).float().cpu().numpy())
    return np.concatenate(outputs)


def _ensemble_member_std(
    model: SingleMutationEnsemble,
    delta: np.ndarray,
    *,
    device: torch.device,
    batch_size: int = 512,
) -> np.ndarray:
    outputs: list[np.ndarray] = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(delta), batch_size):
            tensor = torch.from_numpy(
                delta[start : start + batch_size].astype(np.float32)
            ).to(device)
            outputs.append(
                model.member_predictions(tensor)
                .std(dim=0, unbiased=False)
                .float()
                .cpu()
                .numpy()
            )
    return np.concatenate(outputs)


def rerank_esmc6b_screen(
    sequence: str,
    input_csv: Path,
    output_csv: Path,
    checkpoint_dir: Path,
    *,
    model_name_or_path: str = DEFAULT_ESMC6B_MODEL,
    device: str = "cuda",
    max_tokens: int = 4096,
    max_batch_size: int = 16,
    top: int = 50,
) -> dict[str, object]:
    """Add full ESM-C 6B predictions to a completed 600M single-mutant scan."""

    wt_sequence = normalize_sequence(sequence)
    frame = pd.read_csv(input_csv)
    required = {
        "mutation",
        "mptherm_predicted_delta_tm",
        "masked_marginal_log_odds",
        "consensus_rank_score",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"600M screen is missing required columns: {missing}")
    if frame.empty:
        raise ValueError("600M screen contains no mutations")
    mutations = [Mutation.parse(str(value)) for value in frame["mutation"]]
    if len(set(mutations)) != len(mutations):
        raise ValueError("600M screen contains duplicate mutations")
    for mutation in mutations:
        if mutation.position > len(wt_sequence):
            raise ValueError(f"mutation {mutation} lies outside the sequence")
        if wt_sequence[mutation.position - 1] != mutation.wt:
            raise ValueError(f"mutation {mutation} does not match the WT sequence")

    embedder = ESMC6BEmbedder(model_name_or_path, device)
    positions = sorted({mutation.position for mutation in mutations})
    wt_vectors = embedder.encode(
        [EmbeddingRequest(wt_sequence, tuple(positions))]
    )[0].astype(np.float32)
    wt_by_position = {
        position: wt_vectors[index] for index, position in enumerate(positions)
    }
    requests = [
        EmbeddingRequest(
            apply_mutations(wt_sequence, [mutation]),
            (mutation.position,),
        )
        for mutation in mutations
    ]
    mutant_by_hash: dict[str, np.ndarray] = {}
    for batch_number, batch in enumerate(
        token_batches(
            requests,
            max_tokens=max_tokens,
            max_batch_size=max_batch_size,
        ),
        start=1,
    ):
        for request, vector in zip(batch, embedder.encode(batch), strict=True):
            mutant_by_hash[request.sequence_hash] = vector[0].astype(np.float32)
        if batch_number == 1 or batch_number % 50 == 0:
            print(
                f"ESM-C 6B rerank embedded {len(mutant_by_hash):,}/"
                f"{len(requests):,} mutant sequences",
                flush=True,
            )
    print(
        f"ESM-C 6B rerank embedded {len(mutant_by_hash):,}/"
        f"{len(requests):,} mutant sequences",
        flush=True,
    )
    delta = np.stack(
        [
            mutant_by_hash[request.sequence_hash] - wt_by_position[mutation.position]
            for request, mutation in zip(requests, mutations, strict=True)
        ]
    ).astype(np.float32)
    masked_probabilities = embedder.masked_marginal_log_probabilities(
        wt_sequence,
        positions,
        max_tokens=max_tokens,
        max_batch_size=max_batch_size,
    )
    masked_by_position = {
        position: masked_probabilities[index]
        for index, position in enumerate(positions)
    }
    esmc6b_masked = _masked_marginal_mutation_scores(
        np.stack([masked_by_position[mutation.position] for mutation in mutations]),
        mutations,
    )

    torch_device = torch.device(device if torch.cuda.is_available() else "cpu")
    checkpoint_dir = Path(checkpoint_dir)
    base_model = load_scoring_checkpoint(checkpoint_dir, torch_device)
    mptherm_model = load_auxiliary_checkpoint(
        checkpoint_dir / "mptherm_dtm_head.pt",
        torch_device,
    )
    protherm_model = load_auxiliary_checkpoint(
        checkpoint_dir / "protherm_ddg_head.pt",
        torch_device,
    )
    if base_model.config.embedding_dim != embedder.dimension:
        raise RuntimeError("ESM-C 6B checkpoint embedding dimension mismatch")
    if mptherm_model.config.embedding_dim != embedder.dimension:
        raise RuntimeError("ESM-C 6B MPTherm checkpoint dimension mismatch")
    if protherm_model.config.embedding_dim != embedder.dimension:
        raise RuntimeError("ESM-C 6B ProTherm checkpoint dimension mismatch")
    base_ddg = _torch_prediction(base_model, delta, device=torch_device)
    mptherm = _torch_prediction(mptherm_model, delta, device=torch_device)
    protherm_ddg = _torch_prediction(
        protherm_model,
        delta,
        device=torch_device,
    )
    member_std = (
        None
        if not isinstance(base_model, SingleMutationEnsemble)
        else _ensemble_member_std(
            base_model,
            delta,
            device=torch_device,
        )
    )

    frame = frame.copy()
    frame["esmc600m_consensus_rank_score"] = frame["consensus_rank_score"]
    if "rank" in frame:
        frame["esmc600m_rank"] = frame["rank"]
    frame["esmc6b_pretrained_ddg"] = base_ddg
    frame["esmc6b_protherm_calibrated_ddg"] = protherm_ddg
    if member_std is not None:
        frame["esmc6b_pretrained_ddg_std"] = member_std
    frame["esmc6b_mptherm_predicted_delta_tm"] = mptherm
    frame["esmc6b_masked_marginal_log_odds"] = esmc6b_masked
    frame["consensus_rank_score"] = dual_backbone_thermostability_consensus(
        frame["mptherm_predicted_delta_tm"].to_numpy(dtype=np.float32),
        frame["masked_marginal_log_odds"].to_numpy(dtype=np.float32),
        mptherm,
    )
    frame = frame.sort_values(
        ["consensus_rank_score", "mutation"],
        ascending=[False, True],
    ).reset_index(drop=True)
    frame["rank"] = np.arange(1, len(frame) + 1)
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_csv, index=False, quoting=csv.QUOTE_MINIMAL)
    top_rows = frame.head(max(1, top)).to_dict(orient="records")
    return {
        "input_csv": str(input_csv),
        "output_csv": str(output_csv),
        "candidate_count": int(len(frame)),
        "screened_positions": int(len(positions)),
        "ranking_key": "consensus_rank_score",
        "ranking_policy": (
            "optional 6B rerank: 0.60 ESM-C 600M MPTherm percentile + "
            "0.15 ESM-C 600M masked-marginal percentile + "
            "0.25 full ESM-C 6B MPTherm percentile"
        ),
        "ranking_weights": DUAL_BACKBONE_SCREENING_WEIGHTS,
        "sign_convention": (
            "negative ESM-C 6B pretrained and ProTherm ddG are stabilizing; "
            "positive MPTherm delta-Tm and higher rank scores are favorable"
        ),
        "model_provenance": json.loads(embedder.provenance.canonical_json()),
        "scoring_ensemble_size": (
            len(base_model.members)
            if isinstance(base_model, SingleMutationEnsemble)
            else 1
        ),
        "top_candidates": top_rows,
    }
