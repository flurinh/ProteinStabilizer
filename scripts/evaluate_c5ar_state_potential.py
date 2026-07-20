"""Evaluate the strict-FP32 state-potential fusion on the C5aR scan."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
from Bio.PDB import PDBParser
from Bio.SeqUtils import seq1
from sklearn.metrics import average_precision_score, roc_auc_score

from protein_stabilizer.data import (
    AMINO_ACIDS,
    EmbeddingRequest,
    Mutation,
    apply_mutations,
    merge_embedding_requests,
    sequence_hash,
)
from protein_stabilizer.embeddings import (
    HierarchyEmbeddingReader,
    file_sha256,
)
from protein_stabilizer.esmc6b import build_esmc6b_hierarchy_cache
from protein_stabilizer.state_potential import (
    load_state_potential_ensemble,
)
from protein_stabilizer.v2_features import membrane_topology_features
from protein_stabilizer.v2_training import load_hierarchical_ensemble


ROOT = Path(__file__).resolve().parents[1]


def _pdb_sequence(path: Path) -> str:
    structure = PDBParser(QUIET=True).get_structure("c5ar", path)
    chains = list(structure[0])
    if len(chains) != 1:
        raise RuntimeError("C5aR AlphaFold input must contain exactly one chain")
    residues = [
        residue for residue in chains[0] if residue.id[0] == " "
    ]
    sequence = "".join(seq1(residue.resname) for residue in residues)
    numbering = [int(residue.id[1]) for residue in residues]
    if numbering != list(range(1, len(sequence) + 1)):
        raise RuntimeError("C5aR AlphaFold residue numbering is not canonical")
    return sequence


def _metrics(target: np.ndarray, score: np.ndarray) -> dict[str, object]:
    order = np.argsort(-score)
    return {
        "rows": len(target),
        "positives": int(target.sum()),
        "roc_auc": float(roc_auc_score(target, score)),
        "average_precision": float(
            average_precision_score(target, score)
        ),
        "positives_in_top20": int(target[order[:20]].sum()),
        "positives_in_top50": int(target[order[:50]].sum()),
    }


def _paired_bootstrap(
    target: np.ndarray,
    candidate: np.ndarray,
    baseline: np.ndarray,
    *,
    seed: int,
    samples: int,
) -> dict[str, object]:
    rng = np.random.default_rng(seed)
    auc: list[float] = []
    average_precision: list[float] = []
    for _ in range(samples):
        indices = rng.integers(0, len(target), size=len(target))
        selected_target = target[indices]
        if len(np.unique(selected_target)) != 2:
            continue
        auc.append(
            float(
                roc_auc_score(selected_target, candidate[indices])
                - roc_auc_score(selected_target, baseline[indices])
            )
        )
        average_precision.append(
            float(
                average_precision_score(
                    selected_target, candidate[indices]
                )
                - average_precision_score(
                    selected_target, baseline[indices]
                )
            )
        )

    def interval(values: list[float]) -> list[float]:
        return [
            float(value)
            for value in np.quantile(values, [0.025, 0.5, 0.975])
        ]

    return {
        "seed": seed,
        "requested_samples": samples,
        "valid_samples": len(auc),
        "candidate_minus_retained_rank_auc_95ci": interval(auc),
        "candidate_minus_retained_rank_ap_95ci": interval(
            average_precision
        ),
    }


def evaluate(args: argparse.Namespace) -> dict[str, object]:
    started = time.monotonic()
    sequence = _pdb_sequence(args.pdb)
    with h5py.File(args.labels, "r") as handle:
        mutations = [
            Mutation.parse(value)
            for value in handle["mutation"].asstr()[:]
        ]
        target = np.asarray(handle["target"], dtype=np.float32)
        source_sha256 = str(handle.attrs["source_sha256"])
    if any(
        sequence[mutation.position - 1] != mutation.wt
        for mutation in mutations
    ):
        raise RuntimeError("C5aR labels do not match the AlphaFold sequence")
    positions = tuple(mutation.position for mutation in mutations)
    wt_request = EmbeddingRequest(sequence, tuple(sorted(set(positions))))
    requests = [wt_request]
    requests.extend(
        EmbeddingRequest(
            apply_mutations(sequence, [mutation]),
            (mutation.position,),
        )
        for mutation in mutations
    )
    requests = merge_embedding_requests(requests)
    cache_report = build_esmc6b_hierarchy_cache(
        args.cache,
        requests,
        device=args.device,
        max_tokens=args.max_tokens,
        max_batch_size=args.max_batch_size,
        inference_dtype="float32",
        storage_dtype="float32",
    )
    wt_keys = [
        (sequence_hash(sequence), mutation.position)
        for mutation in mutations
    ]
    mutant_sequences = [
        apply_mutations(sequence, [mutation]) for mutation in mutations
    ]
    mutant_keys = [
        (sequence_hash(mutant), mutation.position)
        for mutant, mutation in zip(
            mutant_sequences, mutations, strict=True
        )
    ]
    with HierarchyEmbeddingReader(args.cache) as cache:
        wt = cache.features(wt_keys)
        mutant = cache.features(mutant_keys)
        cache_provenance = cache.provenance
    if not np.array_equal(wt["window_mask"], mutant["window_mask"]):
        raise RuntimeError("C5aR WT and mutant windows do not align")
    device = torch.device(
        args.device
        if args.device.startswith("cuda") and torch.cuda.is_available()
        else "cpu"
    )
    if device.type == "cuda":
        torch.set_float32_matmul_precision("highest")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    state, state_payload = load_state_potential_ensemble(
        args.state_checkpoint, device
    )
    baseline_path = Path(state_payload["baseline_checkpoint"])
    if file_sha256(baseline_path) != str(
        state_payload["baseline_checkpoint_sha256"]
    ):
        raise RuntimeError("C5aR baseline checkpoint hash mismatch")
    baseline, baseline_payload = load_hierarchical_ensemble(
        baseline_path, device
    )
    if not bool(state_payload["use_structure"]) or not bool(
        baseline_payload["use_structure"]
    ):
        raise RuntimeError("C5aR evaluation expected the promoted structure branch")
    count = len(mutations)
    structure = np.zeros((count, 128), dtype=np.float32)
    structure_mask = np.zeros(count, dtype=bool)
    membrane = np.repeat(
        membrane_topology_features(
            "alpha_helical_gpcr", is_gpcr=True
        )[None],
        count,
        axis=0,
    )
    amino_acid_index = {
        amino_acid: index
        for index, amino_acid in enumerate(AMINO_ACIDS)
    }
    baseline_tensors = {
        "wt_window": torch.from_numpy(
            wt["window"].astype(np.float32)
        ).to(device),
        "mutant_window": torch.from_numpy(
            mutant["window"].astype(np.float32)
        ).to(device),
        "window_mask": torch.from_numpy(wt["window_mask"]).to(device),
        "wt_global": torch.from_numpy(
            wt["global_mean"].astype(np.float32)
        ).to(device),
        "mutant_global": torch.from_numpy(
            mutant["global_mean"].astype(np.float32)
        ).to(device),
        "structure": torch.from_numpy(structure).to(device),
        "structure_mask": torch.from_numpy(structure_mask).to(device),
        "membrane": torch.from_numpy(membrane).to(device),
    }
    state_tensors = {
        key: value
        for key, value in baseline_tensors.items()
        if key
        in {
            "wt_window",
            "window_mask",
            "wt_global",
            "structure",
            "structure_mask",
            "membrane",
        }
    }
    state_tensors["wt_amino_acid"] = torch.tensor(
        [amino_acid_index[mutation.wt] for mutation in mutations],
        dtype=torch.long,
        device=device,
    )
    state_tensors["mutant_amino_acid"] = torch.tensor(
        [
            amino_acid_index[mutation.mutant]
            for mutation in mutations
        ],
        dtype=torch.long,
        device=device,
    )
    baseline.eval()
    state.eval()
    with torch.inference_mode():
        baseline_ddg = baseline(**baseline_tensors)
        state_ddg = state(**state_tensors)
    state_weight = float(state_payload["state_potential_weight"])
    candidate_score = -(
        (1.0 - state_weight) * baseline_ddg
        + state_weight * state_ddg
    ).float().cpu().numpy()
    comparators = pd.read_csv(args.comparators)
    expected = [str(mutation) for mutation in mutations]
    if comparators["mutation"].astype(str).tolist() != expected:
        raise RuntimeError("C5aR comparator rows do not align with labels")
    retained_score = comparators["rank6b"].to_numpy(dtype=np.float32)
    fast_score = comparators["mptherm_delta_tm"].to_numpy(
        dtype=np.float32
    )
    result = {
        "schema": "protein-stabilizer.c5ar-state-potential-audit.v1",
        "candidate": (
            "strict-FP32 6B hierarchy + WT-conditioned state potential"
        ),
        "evaluation": {
            "candidate": _metrics(target, candidate_score),
            "retained_gpcr_reranker": _metrics(target, retained_score),
            "retained_fast_mptherm": _metrics(target, fast_score),
        },
        "paired_bootstrap": _paired_bootstrap(
            target,
            candidate_score,
            retained_score,
            seed=args.seed,
            samples=args.bootstrap_samples,
        ),
        "runtime": {
            "elapsed_seconds": time.monotonic() - started,
            "embedding_cache_initial_sites": cache_report[
                "initial_sites"
            ],
            "embedding_cache_new_sites": cache_report["embedded_sites"],
        },
        "precision": {
            "inference_dtype": cache_provenance["inference_dtype"],
            "storage_dtype": cache_provenance["storage_dtype"],
            "tf32": False,
        },
        "artifacts": {
            "labels": str(args.labels.resolve()),
            "labels_sha256": file_sha256(args.labels),
            "labels_source_sha256": source_sha256,
            "pdb": str(args.pdb.resolve()),
            "pdb_sha256": file_sha256(args.pdb),
            "comparators": str(args.comparators.resolve()),
            "comparators_sha256": file_sha256(args.comparators),
            "cache": str(args.cache.resolve()),
            "cache_sha256": file_sha256(args.cache),
            "state_checkpoint": str(args.state_checkpoint.resolve()),
            "state_checkpoint_sha256": file_sha256(
                args.state_checkpoint
            ),
            "baseline_checkpoint": str(baseline_path.resolve()),
            "baseline_checkpoint_sha256": file_sha256(baseline_path),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--labels",
        type=Path,
        default=ROOT / "artifacts/features_esmc_6b/c5ar.h5",
    )
    parser.add_argument(
        "--pdb",
        type=Path,
        default=Path(
            "/data/fast/tmp/protein-stabilizer/proteinmpnn-gpcr/"
            "input/c5ar/AF-P21730-F1-model_v6.pdb"
        ),
    )
    parser.add_argument(
        "--comparators",
        type=Path,
        default=Path(
            "/data/fast/tmp/protein-stabilizer/proteinmpnn-gpcr/"
            "esmc6b_proteinmpnn_c5ar_components.csv"
        ),
    )
    parser.add_argument(
        "--state-checkpoint",
        type=Path,
        default=(
            ROOT
            / "checkpoints/esmc_6b_state_potential_fp32/"
            "state_potential_ensemble.pt"
        ),
    )
    parser.add_argument(
        "--cache",
        type=Path,
        default=(
            ROOT
            / "embeddings/esmc_6b/"
            "c5ar_state_potential_cache_fp32.h5"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "docs/esmc6b_state_potential_c5ar_audit.json",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--max-batch-size", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260718)
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    return parser


if __name__ == "__main__":
    print(
        json.dumps(
            evaluate(build_parser().parse_args()),
            indent=2,
            sort_keys=True,
        )
    )
