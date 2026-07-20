"""Optional ESM-C 6B inference and GPCR screen reranking."""

from __future__ import annotations

import csv
import json
import time
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
from .embeddings import (
    ESMCProvenance,
    HierarchicalEmbedding,
    HierarchyEmbeddingWriter,
    file_sha256,
    request_manifest_sha256,
    token_batches,
)
from .models import SingleMutationEnsemble
from .predictor import (
    DUAL_BACKBONE_SCREENING_WEIGHTS,
    _masked_marginal_mutation_scores,
    dual_backbone_thermostability_consensus,
)
from .training import (
    load_auxiliary_checkpoint,
    load_multi_checkpoint,
    load_scoring_checkpoint,
)


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
        *,
        inference_dtype: str = "float32",
        storage_dtype: str = "float32",
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
        dtype_by_name = {
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        storage_by_name = {
            "float16": np.dtype(np.float16),
            "float32": np.dtype(np.float32),
        }
        if inference_dtype not in dtype_by_name:
            raise ValueError("6B inference dtype must be bfloat16 or float32")
        if storage_dtype not in storage_by_name:
            raise ValueError("6B storage dtype must be float16 or float32")
        self.inference_dtype = dtype_by_name[inference_dtype]
        self.storage_dtype = storage_by_name[storage_dtype]
        self.strict_fp32 = self.inference_dtype == torch.float32
        if self.device.type == "cuda":
            torch.set_float32_matmul_precision("highest")
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
        self.model_name = model_name_or_path
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            trust_remote_code=True,
        )
        self.model = AutoModelForMaskedLM.from_pretrained(
            model_name_or_path,
            trust_remote_code=True,
            dtype=self.inference_dtype,
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
                f"torch={torch.__version__};"
                f"attention={'math' if self.strict_fp32 else 'automatic'};"
                "tf32=false"
            ),
            checkpoint_path=str(checkpoint.parent),
            checkpoint_sha256=file_sha256(checkpoint),
            inference_dtype=inference_dtype,
            storage_dtype=storage_dtype,
        )

    def _forward(self, encoded: dict[str, torch.Tensor]) -> object:
        """Run strict native-FP32 attention for the accuracy-first 6B path."""

        if self.device.type == "cuda" and self.strict_fp32:
            from torch.nn.attention import SDPBackend, sdpa_kernel

            with sdpa_kernel(SDPBackend.MATH):
                return self.model(**encoded)
        return self.model(**encoded)

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
            output = self._forward(encoded)
        hidden = output.last_hidden_state
        results: list[np.ndarray] = []
        for row, request in enumerate(requests):
            selected = hidden[row, list(request.positions)]
            if selected.shape != (len(request.positions), self.dimension):
                raise RuntimeError("unexpected ESM-C 6B residue embedding shape")
            results.append(
                selected.float().cpu().numpy().astype(self.storage_dtype)
            )
        return results

    def encode_hierarchy(
        self,
        requests: Sequence[EmbeddingRequest],
        *,
        window_radius: int = 4,
    ) -> list[HierarchicalEmbedding]:
        """Return 6B whole-protein means and ordered site windows."""

        if not requests:
            return []
        if window_radius < 0:
            raise ValueError("window radius must be non-negative")
        sequences = [normalize_sequence(request.sequence) for request in requests]
        encoded = self._tokenize(sequences)
        with torch.inference_mode():
            output = self._forward(encoded)
        hidden = output.last_hidden_state
        window_size = 2 * window_radius + 1
        results: list[HierarchicalEmbedding] = []
        for row, (request, sequence) in enumerate(
            zip(requests, sequences, strict=True)
        ):
            residues = hidden[row, 1 : len(sequence) + 1].float()
            windows = torch.zeros(
                (len(request.positions), window_size, self.dimension),
                dtype=residues.dtype,
                device=residues.device,
            )
            mask = torch.zeros(
                (len(request.positions), window_size),
                dtype=torch.bool,
                device=residues.device,
            )
            for site_index, position in enumerate(request.positions):
                source_start = max(1, position - window_radius)
                source_stop = min(len(sequence), position + window_radius)
                destination_start = source_start - (position - window_radius)
                count = source_stop - source_start + 1
                windows[
                    site_index,
                    destination_start : destination_start + count,
                ] = residues[source_start - 1 : source_stop]
                mask[
                    site_index,
                    destination_start : destination_start + count,
                ] = True
            results.append(
                HierarchicalEmbedding(
                    global_mean=residues.mean(dim=0)
                    .cpu()
                    .numpy()
                    .astype(self.storage_dtype),
                    windows=windows.cpu().numpy().astype(self.storage_dtype),
                    window_mask=mask.cpu().numpy(),
                )
            )
        del output, hidden
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
                output = self._forward(encoded)
                site_logits = output.logits[rows, token_positions]
                log_probabilities = torch.log_softmax(
                    site_logits.float(), dim=-1
                )[:, amino_acid_ids]
            results.append(log_probabilities.cpu().numpy().astype(np.float32))
        return np.concatenate(results, axis=0)


def build_esmc6b_hierarchy_cache(
    path: Path,
    requests: Sequence[EmbeddingRequest],
    *,
    model_name_or_path: str = DEFAULT_ESMC6B_MODEL,
    device: str = "cuda",
    window_radius: int = 4,
    max_tokens: int = 4096,
    max_batch_size: int = 8,
    inference_dtype: str = "float32",
    storage_dtype: str = "float32",
) -> dict[str, object]:
    """Build the promoted 6B cache with the same schema as 600M discovery."""

    embedder = ESMC6BEmbedder(
        model_name_or_path,
        device,
        inference_dtype=inference_dtype,
        storage_dtype=storage_dtype,
    )
    manifest_hash = request_manifest_sha256(requests)
    started = time.monotonic()
    with HierarchyEmbeddingWriter(
        path, embedder.provenance, window_radius=window_radius
    ) as writer:
        missing = writer.missing_requests(requests)
        initial_sites = len(writer.site_index)
        batches = list(
            token_batches(
                missing,
                max_tokens=max_tokens,
                max_batch_size=max_batch_size,
            )
        )
        for batch_number, batch in enumerate(batches, start=1):
            writer.append(
                batch,
                embedder.encode_hierarchy(batch, window_radius=window_radius),
            )
            if (
                batch_number == 1
                or batch_number % 50 == 0
                or batch_number == len(batches)
            ):
                print(
                    f"6B hierarchy batch {batch_number}/{len(batches)}; "
                    f"cached sites={len(writer.site_index):,}",
                    flush=True,
                )
        result = {
            "schema": str(writer.handle.attrs["schema"]),
            "path": str(path),
            "model_provenance": json.loads(
                embedder.provenance.canonical_json()
            ),
            "request_manifest_sha256": manifest_hash,
            "window_radius": window_radius,
            "requested_sequences": len(requests),
            "requested_sites": sum(
                len(request.positions) for request in requests
            ),
            "initial_sites": initial_sites,
            "embedded_sites": len(writer.site_index) - initial_sites,
            "cached_sites": len(writer.site_index),
            "elapsed_seconds": time.monotonic() - started,
        }
        writer.handle.attrs["request_manifest_sha256"] = manifest_hash
        writer.handle.attrs["last_build_manifest"] = json.dumps(
            result, sort_keys=True, separators=(",", ":")
        )
        writer.handle.flush()
        return result


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


def predict_esmc6b_mutations(
    sequence: str,
    mutations: Sequence[str | Mutation],
    checkpoint_dir: Path,
    *,
    model_name_or_path: str = DEFAULT_ESMC6B_MODEL,
    device: str = "cuda",
    max_tokens: int = 4096,
    max_batch_size: int = 16,
) -> dict[str, object]:
    """Predict one or more substitutions with the full ESM-C 6B heads."""

    wt_sequence = normalize_sequence(sequence)
    parsed = [
        mutation if isinstance(mutation, Mutation) else Mutation.parse(mutation)
        for mutation in mutations
    ]
    if not parsed:
        raise ValueError("at least one mutation is required")
    parsed = sorted(parsed, key=lambda mutation: mutation.position)
    positions = tuple(mutation.position for mutation in parsed)
    joint_sequence = apply_mutations(wt_sequence, parsed)
    single_sequences = [
        apply_mutations(wt_sequence, [mutation]) for mutation in parsed
    ]
    requests = [EmbeddingRequest(wt_sequence, positions)]
    requests.extend(
        EmbeddingRequest(single, (mutation.position,))
        for single, mutation in zip(single_sequences, parsed, strict=True)
    )
    requests.append(EmbeddingRequest(joint_sequence, positions))

    embedder = ESMC6BEmbedder(model_name_or_path, device)
    vector_by_hash: dict[str, np.ndarray] = {}
    for batch in token_batches(
        requests,
        max_tokens=max_tokens,
        max_batch_size=max_batch_size,
    ):
        for request, vector in zip(batch, embedder.encode(batch), strict=True):
            vector_by_hash[request.sequence_hash] = vector.astype(np.float32)
    wt_vectors = vector_by_hash[requests[0].sequence_hash]
    single_vectors = np.concatenate(
        [
            vector_by_hash[request.sequence_hash]
            for request in requests[1:-1]
        ],
        axis=0,
    )
    joint_vectors = vector_by_hash[requests[-1].sequence_hash]
    single_delta = single_vectors - wt_vectors
    joint_delta = joint_vectors - wt_vectors

    masked_log_probabilities = embedder.masked_marginal_log_probabilities(
        wt_sequence,
        positions,
        max_tokens=max_tokens,
        max_batch_size=max_batch_size,
    )
    masked_marginal = _masked_marginal_mutation_scores(
        masked_log_probabilities,
        parsed,
    )
    joint_masked_log_probabilities = (
        embedder.masked_marginal_log_probabilities(
            joint_sequence,
            positions,
            max_tokens=max_tokens,
            max_batch_size=max_batch_size,
        )
        if len(parsed) > 1
        else masked_log_probabilities
    )
    joint_masked_marginal = _masked_marginal_mutation_scores(
        joint_masked_log_probabilities,
        parsed,
    )

    torch_device = torch.device(device if torch.cuda.is_available() else "cpu")
    checkpoint_dir = Path(checkpoint_dir)
    base_model = load_scoring_checkpoint(checkpoint_dir, torch_device)
    protherm_model = load_auxiliary_checkpoint(
        checkpoint_dir / "protherm_ddg_head.pt",
        torch_device,
    )
    mptherm_model = load_auxiliary_checkpoint(
        checkpoint_dir / "mptherm_dtm_head.pt",
        torch_device,
    )
    for name, model in (
        ("base", base_model),
        ("ProTherm", protherm_model),
        ("MPTherm", mptherm_model),
    ):
        if model.config.embedding_dim != embedder.dimension:
            raise RuntimeError(f"ESM-C 6B {name} checkpoint dimension mismatch")
    constituent = _torch_prediction(
        base_model,
        single_delta,
        device=torch_device,
    )
    protherm_ddg = _torch_prediction(
        protherm_model,
        single_delta,
        device=torch_device,
    )
    mptherm_delta_tm = _torch_prediction(
        mptherm_model,
        single_delta,
        device=torch_device,
    )
    member_std = (
        _ensemble_member_std(
            base_model,
            single_delta,
            device=torch_device,
        )
        if isinstance(base_model, SingleMutationEnsemble)
        else None
    )

    result: dict[str, object] = {
        "mutations": [str(mutation) for mutation in parsed],
        "mutation_count": len(parsed),
        "sign_convention": (
            "negative predicted and ProTherm ddG are stabilizing; "
            "positive MPTherm delta-Tm is favorable"
        ),
        "constituent_single_ddg": {
            str(mutation): float(value)
            for mutation, value in zip(parsed, constituent, strict=True)
        },
        "additive_ddg": float(constituent.sum()),
        "constituent_protherm_calibrated_ddg": {
            str(mutation): float(value)
            for mutation, value in zip(parsed, protherm_ddg, strict=True)
        },
        "constituent_mptherm_predicted_delta_tm": {
            str(mutation): float(value)
            for mutation, value in zip(parsed, mptherm_delta_tm, strict=True)
        },
        "constituent_masked_marginal_log_odds": {
            str(mutation): float(value)
            for mutation, value in zip(parsed, masked_marginal, strict=True)
        },
        "additive_masked_marginal_log_odds": float(masked_marginal.sum()),
        "constituent_joint_context_masked_marginal_log_odds": {
            str(mutation): float(value)
            for mutation, value in zip(
                parsed,
                joint_masked_marginal,
                strict=True,
            )
        },
        "joint_context_masked_pseudologlikelihood_log_odds": float(
            joint_masked_marginal.sum()
        ),
        "masked_context_epistasis_log_odds": float(
            joint_masked_marginal.sum() - masked_marginal.sum()
        ),
        "model_provenance": json.loads(embedder.provenance.canonical_json()),
    }
    if member_std is not None:
        result["ensemble_size"] = len(base_model.members)
        result["constituent_single_ddg_std"] = {
            str(mutation): float(value)
            for mutation, value in zip(parsed, member_std, strict=True)
        }
    if len(parsed) == 1:
        result.update(
            {
                "predicted_ddg": float(constituent[0]),
                "pretrained_ddg": float(constituent[0]),
                "protherm_calibrated_ddg": float(protherm_ddg[0]),
                "mptherm_predicted_delta_tm": float(mptherm_delta_tm[0]),
                "masked_marginal_log_odds": float(masked_marginal[0]),
            }
        )
        if member_std is not None:
            result["pretrained_ddg_std"] = float(member_std[0])
        return result

    multi_model = load_multi_checkpoint(
        checkpoint_dir / "multi_head.pt",
        torch_device,
    )
    if multi_model.config.embedding_dim != embedder.dimension:
        raise RuntimeError("ESM-C 6B epistasis checkpoint dimension mismatch")
    single_batch = torch.from_numpy(single_delta[None].astype(np.float32)).to(
        torch_device
    )
    joint_batch = torch.from_numpy(joint_delta[None].astype(np.float32)).to(
        torch_device
    )
    with torch.inference_mode():
        training_total, training_additive, epistasis = multi_model(
            single_batch,
            joint_batch,
        )
    ensemble_additive = float(constituent.sum())
    correction = float(epistasis.item())
    result.update(
        {
            "predicted_ddg": ensemble_additive + correction,
            "model_additive_ddg": ensemble_additive,
            "epistasis_ddg": correction,
            "epistasis_training_additive_ddg": float(training_additive.item()),
            "epistasis_training_total_ddg": float(training_total.item()),
            "extrapolation_warning": (
                "epistasis head was trained on double mutants"
                if len(parsed) > 2
                else None
            ),
        }
    )
    return result


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
