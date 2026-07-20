"""Command-line interface for data preparation, training, and inference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from .data import DatasetPaths, all_embedding_requests, single_embedding_requests
from .embeddings import build_embedding_cache, build_hierarchy_embedding_cache
from .esmc6b import (
    DEFAULT_ESMC6B_MODEL,
    ESMC6BEmbedder,
    build_esmc6b_hierarchy_cache,
    predict_esmc6b_mutations,
    rerank_esmc6b_screen,
)
from .features import build_feature_files
from .gpcr_ranking import train_zero_shot_membrane_ranker
from .predictor import predict_mutations, screen_single_mutants
from .structure import (
    build_aligned_structure_features,
    build_megascale_structure_features,
)
from .state_potential import (
    build_state_potential_multi_representations,
    build_state_potential_transfer_representations,
    train_state_potential_candidate,
)
from .training import (
    evaluate_directional_ablation,
    train_directional_single_head,
    train_epistasis_head,
    train_gpcr_calibration,
    train_single_head,
    train_transfer_heads,
)
from .v2_features import (
    build_hierarchy_multi_row_features,
    build_hierarchy_row_features,
)
from .v2_training import (
    train_hierarchical_single_ablation,
    train_promoted_hierarchical_single,
)
from .v2_multi import (
    build_multi_representations,
    promote_hierarchical_multi_candidate,
    train_hierarchical_multi_head,
)
from .v2_transfer import (
    build_transfer_representations,
    train_transfer_adapters,
)
from .v2_predictor import (
    predict_hierarchical_mutations,
    screen_hierarchical_single_mutants,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CACHE = ROOT / "embeddings/esmc_600m/residue_cache.h5"
DEFAULT_HIERARCHY_CACHE = ROOT / "embeddings/esmc_600m/hierarchy_cache.h5"
DEFAULT_FEATURES = ROOT / "artifacts/features"
DEFAULT_CHECKPOINTS = ROOT / "checkpoints/esmc_600m"
DEFAULT_V2_CHECKPOINTS = ROOT / "checkpoints/esmc_600m_v2"
DEFAULT_V2_FEATURES = ROOT / "artifacts/features_v2"
DEFAULT_V2_STRUCTURE = ROOT / "artifacts/features_v2_structure"
DEFAULT_V2_MULTI = ROOT / "artifacts/features_v2_multi"
DEFAULT_V2_TRANSFER = ROOT / "artifacts/features_v2_transfer"
DEFAULT_MEGASCALE_ARCHIVE = ROOT / "data/raw/thermompnn_d/Megascale.tar.gz"
DEFAULT_PROTEINMPNN_REPOSITORY = Path(
    "/data/fast/tmp/protein-stabilizer/upstreams/ProteinMPNN"
)
DEFAULT_ESMC6B_CHECKPOINTS = ROOT / "checkpoints/esmc_6b"
DEFAULT_ESMC6B_V2_CHECKPOINTS = ROOT / "checkpoints/esmc_6b_v2"
DEFAULT_ESMC6B_HIERARCHY_CACHE = (
    ROOT / "embeddings/esmc_6b/hierarchy_cache_fp32.h5"
)
DEFAULT_FP32_HIERARCHY_CACHE = (
    ROOT / "embeddings/esmc_600m/hierarchy_cache_fp32.h5"
)
DEFAULT_FP32_V2_FEATURES = ROOT / "artifacts/features_v2_600m_fp32"
DEFAULT_FP32_V2_CHECKPOINTS = ROOT / "checkpoints/esmc_600m_v2_fp32"
DEFAULT_STATE_POTENTIAL_CHECKPOINTS = (
    ROOT / "checkpoints/esmc_600m_state_potential_fp32"
)
DEFAULT_STATE_POTENTIAL_TRANSFER = (
    ROOT / "artifacts/features_v2_transfer_600m_state_potential_fp32"
)
DEFAULT_ESMC6B_STATE_TRANSFER = (
    ROOT / "artifacts/features_v2_transfer_6b_state_potential_fp32"
)
DEFAULT_ESMC6B_STATE_MULTI = (
    ROOT / "artifacts/features_v2_multi_6b_state_potential_fp32"
)
DEFAULT_ESMC6B_STATE_CHECKPOINTS = (
    ROOT / "checkpoints/esmc_6b_state_potential_fp32"
)


def _json(value: object) -> None:
    print(json.dumps(value, indent=2, sort_keys=True))


def _read_fasta(path: Path) -> str:
    lines = [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith(">")
    ]
    if not lines:
        raise ValueError(f"no sequence found in {path}")
    return "".join(lines)


def _require_data(root: Path) -> None:
    paths = DatasetPaths(root)
    required = [
        paths.single("train"),
        paths.single("test"),
        *(paths.double(split) for split in ("train", "val", "test")),
        paths.gpcr,
        paths.protherm,
        paths.mptherm,
        paths.mcsm_membrane,
        paths.gpcr_tm,
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "required prepared datasets are missing; run scripts/download_data.py and "
            "scripts/prepare_gpcr_benchmark.py and scripts/download_transfer_data.py "
            f"first: {missing}"
        )


def command_embed(args: argparse.Namespace) -> dict[str, object]:
    root = args.root.resolve()
    _require_data(root)
    requests = all_embedding_requests(DatasetPaths(root))
    print(
        f"embedding plan: {len(requests):,} unique sequences, "
        f"{sum(len(request.positions) for request in requests):,} residue sites",
        flush=True,
    )
    return build_embedding_cache(
        args.cache,
        requests,
        model_name=args.model,
        device=args.device,
        max_tokens=args.max_tokens,
        max_batch_size=args.max_batch_size,
    )


def command_features(args: argparse.Namespace) -> dict[str, object]:
    _require_data(args.root.resolve())
    return build_feature_files(
        args.root.resolve(),
        args.cache,
        args.features,
        seed=args.seed,
        chunk_size=args.chunk_size,
    )


def command_embed_v2_context(args: argparse.Namespace) -> dict[str, object]:
    paths = DatasetPaths(args.root.resolve())
    for split in ("train", "test"):
        if not paths.single(split).is_file():
            raise FileNotFoundError(paths.single(split))
    requests = (
        all_embedding_requests(paths)
        if args.scope == "all"
        else single_embedding_requests(paths)
    )
    print(
        f"hierarchical embedding plan: {len(requests):,} unique sequences, "
        f"{sum(len(request.positions) for request in requests):,} local windows "
        f"(scope={args.scope})",
        flush=True,
    )
    return build_hierarchy_embedding_cache(
        args.cache,
        requests,
        model_name=args.model,
        device=args.device,
        window_radius=args.window_radius,
        max_tokens=args.max_tokens,
        max_batch_size=args.max_batch_size,
        storage_dtype=args.storage_dtype,
    )


def command_structure_v2(args: argparse.Namespace) -> dict[str, object]:
    return build_megascale_structure_features(
        args.base_features,
        args.structure_archive,
        args.output,
        args.proteinmpnn_repository,
        temp_root=args.temp_root,
        device=args.device,
        storage_dtype=args.storage_dtype,
    )


def command_structure_v2_aligned(args: argparse.Namespace) -> dict[str, object]:
    return build_aligned_structure_features(
        args.rows,
        args.structure_map,
        args.output,
        args.proteinmpnn_repository,
        device=args.device,
        storage_dtype=args.storage_dtype,
    )


def command_embed_v2_context_6b(args: argparse.Namespace) -> dict[str, object]:
    promotion = json.loads(args.promotion_report.read_text(encoding="utf-8"))
    if not bool(
        promotion.get("promotion", {}).get("promoted_to_transfer_and_6b")
    ):
        raise RuntimeError(
            "600M hierarchy candidate did not pass the frozen promotion gate"
        )
    paths = DatasetPaths(args.root.resolve())
    requests = all_embedding_requests(paths)
    print(
        f"promoted 6B hierarchy plan: {len(requests):,} unique sequences, "
        f"{sum(len(request.positions) for request in requests):,} local windows",
        flush=True,
    )
    return build_esmc6b_hierarchy_cache(
        args.cache,
        requests,
        model_name_or_path=args.model,
        device=args.device,
        window_radius=args.window_radius,
        max_tokens=args.max_tokens,
        max_batch_size=args.max_batch_size,
        inference_dtype=args.inference_dtype,
        storage_dtype=args.storage_dtype,
    )


def command_features_v2(args: argparse.Namespace) -> dict[str, object]:
    return build_hierarchy_row_features(
        args.root.resolve(),
        args.cache,
        args.output,
        structure_dir=args.structure,
        seed=args.seed,
    )


def command_features_v2_multi(args: argparse.Namespace) -> dict[str, object]:
    return build_hierarchy_multi_row_features(
        args.root.resolve(),
        args.cache,
        args.output,
    )


def command_train(args: argparse.Namespace) -> dict[str, object]:
    single = train_single_head(
        args.features,
        args.checkpoints,
        seed=args.seed,
        epochs=args.single_epochs,
        ensemble_size=args.ensemble_size,
        device=args.device,
    )
    multi = train_epistasis_head(
        args.features,
        args.checkpoints,
        seed=args.seed,
        epochs=args.multi_epochs,
        device=args.device,
    )
    transfer = train_transfer_heads(
        args.features,
        args.checkpoints,
        seed=args.seed,
        device=args.device,
    )
    gpcr = train_gpcr_calibration(
        args.features, args.checkpoints, device=args.device
    )
    return {"single": single, "multi": multi, "transfer": transfer, "gpcr": gpcr}


def command_train_v2(args: argparse.Namespace) -> dict[str, object]:
    return train_directional_single_head(
        args.features,
        args.checkpoints,
        seed=args.seed,
        epochs=args.single_epochs,
        patience=args.patience,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        ensemble_size=args.ensemble_size,
        device=args.device,
    )


def command_train_v2_hierarchy(args: argparse.Namespace) -> dict[str, object]:
    return train_hierarchical_single_ablation(
        args.features,
        args.cache,
        args.baseline_checkpoints,
        args.checkpoints,
        seed=args.seed,
        discovery_epochs=args.discovery_epochs,
        discovery_patience=args.discovery_patience,
        main_epochs=args.main_epochs,
        minimum_main_epochs=args.minimum_main_epochs,
        main_patience=args.main_patience,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        ensemble_size=args.ensemble_size,
        device=args.device,
    )


def command_represent_v2_multi(args: argparse.Namespace) -> dict[str, object]:
    return build_multi_representations(
        args.features,
        args.cache,
        args.base_checkpoint,
        args.output,
        batch_size=args.batch_size,
        device=args.device,
    )


def command_train_v2_hierarchy_6b(args: argparse.Namespace) -> dict[str, object]:
    return train_promoted_hierarchical_single(
        args.features,
        args.cache,
        args.baseline_checkpoints,
        args.discovery_report,
        args.checkpoints,
        seed=args.seed,
        epochs=args.epochs,
        minimum_epochs=args.minimum_epochs,
        patience=args.patience,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        ensemble_size=args.ensemble_size,
        device=args.device,
    )


def command_train_state_potential(args: argparse.Namespace) -> dict[str, object]:
    return train_state_potential_candidate(
        args.features,
        args.cache,
        args.baseline_checkpoint,
        args.checkpoints,
        seed=args.seed,
        epochs=args.epochs,
        minimum_epochs=args.minimum_epochs,
        patience=args.patience,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        ensemble_size=args.ensemble_size,
        device=args.device,
    )


def command_represent_state_potential_transfer(
    args: argparse.Namespace,
) -> dict[str, object]:
    return build_state_potential_transfer_representations(
        args.features,
        args.cache,
        args.state_checkpoint,
        args.output,
        batch_size=args.batch_size,
        device=args.device,
    )


def command_represent_state_potential_multi(
    args: argparse.Namespace,
) -> dict[str, object]:
    return build_state_potential_multi_representations(
        args.features,
        args.cache,
        args.baseline_representations,
        args.state_checkpoint,
        args.output,
        batch_size=args.batch_size,
        device=args.device,
    )


def command_train_zero_shot_gpcr(
    args: argparse.Namespace,
) -> dict[str, object]:
    return train_zero_shot_membrane_ranker(
        args.representations,
        args.gpcr_tm_source,
        args.checkpoints,
        seed=args.seed,
        discovery_epochs=args.discovery_epochs,
        discovery_minimum_epochs=args.discovery_minimum_epochs,
        main_epochs=args.main_epochs,
        minimum_main_epochs=args.minimum_main_epochs,
        patience=args.patience,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        ensemble_size=args.ensemble_size,
        device=args.device,
    )


def command_train_v2_multi(args: argparse.Namespace) -> dict[str, object]:
    return train_hierarchical_multi_head(
        args.representations,
        args.checkpoints,
        seed=args.seed,
        epochs=args.epochs,
        minimum_epochs=args.minimum_epochs,
        patience=args.patience,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        device=args.device,
        constituent_state_weight=args.constituent_state_weight,
    )


def command_promote_v2_multi(args: argparse.Namespace) -> dict[str, object]:
    return promote_hierarchical_multi_candidate(
        args.candidate_checkpoints,
        args.baseline_checkpoints,
    )


def command_represent_v2_transfer(args: argparse.Namespace) -> dict[str, object]:
    return build_transfer_representations(
        args.features,
        args.cache,
        args.base_checkpoint,
        args.output,
        batch_size=args.batch_size,
        device=args.device,
    )


def command_train_v2_transfer(args: argparse.Namespace) -> dict[str, object]:
    return train_transfer_adapters(
        args.representations,
        args.checkpoints,
        seed=args.seed,
        epochs=args.epochs,
        minimum_epochs=args.minimum_epochs,
        patience=args.patience,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        device=args.device,
    )


def command_evaluate_v2(args: argparse.Namespace) -> dict[str, object]:
    return evaluate_directional_ablation(
        args.features,
        args.baseline_checkpoints,
        args.checkpoints,
        output_path=args.output,
        device=args.device,
        batch_size=args.batch_size,
    )


def command_run(args: argparse.Namespace) -> dict[str, object]:
    embedding = command_embed(args)
    features = command_features(args)
    training = command_train(args)
    return {"embedding": embedding, "features": features, "training": training}


def command_predict(args: argparse.Namespace) -> dict[str, object]:
    sequence = args.sequence if args.sequence is not None else _read_fasta(args.fasta)
    mutations = [value.strip() for value in args.mutations.split(",") if value.strip()]
    return predict_mutations(
        sequence,
        mutations,
        args.checkpoints,
        device=args.device,
        model_name=args.model,
    )


def _parse_generic_numbering(value: str | None) -> dict[int, str] | None:
    if value is None:
        return None
    result: dict[int, str] = {}
    for item in value.split(","):
        position, separator, generic_number = item.strip().partition("=")
        if not separator or not position or not generic_number:
            raise ValueError(
                "generic numbering must use position=number entries"
            )
        parsed_position = int(position)
        if parsed_position in result:
            raise ValueError(
                f"duplicate generic numbering position {parsed_position}"
            )
        result[parsed_position] = generic_number
    return result


def command_predict_v2(args: argparse.Namespace) -> dict[str, object]:
    sequence = args.sequence if args.sequence is not None else _read_fasta(args.fasta)
    mutations = [
        value.strip() for value in args.mutations.split(",") if value.strip()
    ]
    embedder = ESMCEmbedder(
        model_name=args.model,
        device=args.device,
        storage_dtype="float32",
    )
    return predict_hierarchical_mutations(
        sequence,
        mutations,
        args.checkpoints,
        model_name=args.model,
        device=args.device,
        topology=args.topology,
        generic_numbering=_parse_generic_numbering(args.generic_numbering),
        pdb_path=args.pdb,
        proteinmpnn_repository=args.proteinmpnn_repository,
        max_tokens=args.max_tokens,
        max_batch_size=args.max_batch_size,
        embedder=embedder,
        state_potential_checkpoint=args.state_potential_checkpoint,
    )


def command_predict_v2_6b(args: argparse.Namespace) -> dict[str, object]:
    """Run the promoted hierarchical head with native-FP32 ESM-C 6B."""

    sequence = args.sequence if args.sequence is not None else _read_fasta(args.fasta)
    mutations = [
        value.strip() for value in args.mutations.split(",") if value.strip()
    ]
    embedder = ESMC6BEmbedder(
        args.model,
        args.device,
        inference_dtype="float32",
        storage_dtype="float32",
    )
    return predict_hierarchical_mutations(
        sequence,
        mutations,
        args.checkpoints,
        model_name=args.model,
        device=args.device,
        topology=args.topology,
        generic_numbering=_parse_generic_numbering(args.generic_numbering),
        pdb_path=args.pdb,
        proteinmpnn_repository=args.proteinmpnn_repository,
        max_tokens=args.max_tokens,
        max_batch_size=args.max_batch_size,
        embedder=embedder,
        state_potential_checkpoint=args.state_potential_checkpoint,
    )


def _parse_positions(value: str | None) -> list[int] | None:
    if value is None:
        return None
    positions: set[int] = set()
    for part in value.split(","):
        bounds = part.strip().split("-")
        if len(bounds) == 1:
            positions.add(int(bounds[0]))
        elif len(bounds) == 2:
            start, stop = (int(bound) for bound in bounds)
            if stop < start:
                raise ValueError(f"invalid position range {part!r}")
            positions.update(range(start, stop + 1))
        else:
            raise ValueError(f"invalid position expression {part!r}")
    return sorted(positions)


def command_screen(args: argparse.Namespace) -> dict[str, object]:
    sequence = args.sequence if args.sequence is not None else _read_fasta(args.fasta)
    return screen_single_mutants(
        sequence,
        args.checkpoints,
        positions=_parse_positions(args.positions),
        output_csv=args.output,
        top=args.top,
        device=args.device,
        model_name=args.model,
        max_tokens=args.max_tokens,
        max_batch_size=args.max_batch_size,
    )


def command_screen_v2(args: argparse.Namespace) -> dict[str, object]:
    sequence = args.sequence if args.sequence is not None else _read_fasta(args.fasta)
    embedder = ESMCEmbedder(
        model_name=args.model,
        device=args.device,
        storage_dtype="float32",
    )
    return screen_hierarchical_single_mutants(
        sequence,
        args.checkpoints,
        args.output,
        positions=_parse_positions(args.positions),
        model_name=args.model,
        device=args.device,
        topology=args.topology,
        generic_numbering=_parse_generic_numbering(args.generic_numbering),
        pdb_path=args.pdb,
        proteinmpnn_repository=args.proteinmpnn_repository,
        legacy_checkpoint_dir=args.legacy_checkpoints,
        max_tokens=args.max_tokens,
        max_batch_size=args.max_batch_size,
        top=args.top,
        embedder=embedder,
        state_potential_checkpoint=args.state_potential_checkpoint,
        scan_mode=args.scan_mode,
    )


def command_screen_v2_6b(args: argparse.Namespace) -> dict[str, object]:
    """Screen selected sites with the promoted native-FP32 ESM-C 6B head."""

    sequence = args.sequence if args.sequence is not None else _read_fasta(args.fasta)
    embedder = ESMC6BEmbedder(
        args.model,
        args.device,
        inference_dtype="float32",
        storage_dtype="float32",
    )
    return screen_hierarchical_single_mutants(
        sequence,
        args.checkpoints,
        args.output,
        positions=_parse_positions(args.positions),
        model_name=args.model,
        device=args.device,
        topology=args.topology,
        generic_numbering=_parse_generic_numbering(args.generic_numbering),
        pdb_path=args.pdb,
        proteinmpnn_repository=args.proteinmpnn_repository,
        legacy_checkpoint_dir=None,
        max_tokens=args.max_tokens,
        max_batch_size=args.max_batch_size,
        top=args.top,
        embedder=embedder,
        state_potential_checkpoint=args.state_potential_checkpoint,
        scan_mode=args.scan_mode,
    )


def command_rerank_esmc6b(args: argparse.Namespace) -> dict[str, object]:
    sequence = args.sequence if args.sequence is not None else _read_fasta(args.fasta)
    return rerank_esmc6b_screen(
        sequence,
        args.input,
        args.output,
        args.checkpoints,
        model_name_or_path=args.model,
        device=args.device,
        max_tokens=args.max_tokens,
        max_batch_size=args.max_batch_size,
        top=args.top,
    )


def command_predict_esmc6b(args: argparse.Namespace) -> dict[str, object]:
    sequence = args.sequence if args.sequence is not None else _read_fasta(args.fasta)
    mutations = [value.strip() for value in args.mutations.split(",") if value.strip()]
    return predict_esmc6b_mutations(
        sequence,
        mutations,
        args.checkpoints,
        model_name_or_path=args.model,
        device=args.device,
        max_tokens=args.max_tokens,
        max_batch_size=args.max_batch_size,
    )


def _common_pipeline_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--features", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--checkpoints", type=Path, default=DEFAULT_CHECKPOINTS)
    parser.add_argument("--model", default="esmc_600m", choices=["esmc_600m"])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=20260715)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--max-batch-size", type=int, default=128)
    parser.add_argument("--chunk-size", type=int, default=4096)
    parser.add_argument("--single-epochs", type=int, default=30)
    parser.add_argument("--ensemble-size", type=int, default=5)
    parser.add_argument("--multi-epochs", type=int, default=25)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="protein-stabilizer")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("embed", "features", "train", "run"):
        child = subparsers.add_parser(name)
        _common_pipeline_arguments(child)
    embed_v2 = subparsers.add_parser("embed-v2-context")
    embed_v2.add_argument("--root", type=Path, default=ROOT)
    embed_v2.add_argument("--cache", type=Path, default=DEFAULT_HIERARCHY_CACHE)
    embed_v2.add_argument("--model", default="esmc_600m", choices=["esmc_600m"])
    embed_v2.add_argument("--device", default="cuda")
    embed_v2.add_argument("--window-radius", type=int, default=4)
    embed_v2.add_argument(
        "--scope",
        choices=["generic-single", "all"],
        default="all",
        help="cache only MegaScale singles or every staged v2 dataset",
    )
    embed_v2.add_argument("--max-tokens", type=int, default=8192)
    embed_v2.add_argument("--max-batch-size", type=int, default=128)
    embed_v2.add_argument(
        "--storage-dtype",
        choices=["float32", "float16"],
        default="float32",
    )
    structure_v2 = subparsers.add_parser("structure-v2")
    structure_v2.add_argument("--base-features", type=Path, default=DEFAULT_FEATURES)
    structure_v2.add_argument(
        "--structure-archive",
        type=Path,
        default=DEFAULT_MEGASCALE_ARCHIVE,
    )
    structure_v2.add_argument("--output", type=Path, default=DEFAULT_V2_STRUCTURE)
    structure_v2.add_argument(
        "--proteinmpnn-repository",
        type=Path,
        default=DEFAULT_PROTEINMPNN_REPOSITORY,
    )
    structure_v2.add_argument(
        "--temp-root",
        type=Path,
        default=Path("/data/fast/tmp/protein-stabilizer/v2"),
    )
    structure_v2.add_argument("--device", default="cuda")
    structure_v2.add_argument(
        "--storage-dtype",
        choices=["float32", "float16"],
        default="float32",
    )
    structure_v2_aligned = subparsers.add_parser("structure-v2-aligned")
    structure_v2_aligned.add_argument("--rows", type=Path, required=True)
    structure_v2_aligned.add_argument(
        "--structure-map", type=Path, required=True
    )
    structure_v2_aligned.add_argument("--output", type=Path, required=True)
    structure_v2_aligned.add_argument(
        "--proteinmpnn-repository",
        type=Path,
        default=DEFAULT_PROTEINMPNN_REPOSITORY,
    )
    structure_v2_aligned.add_argument("--device", default="cuda")
    structure_v2_aligned.add_argument(
        "--storage-dtype",
        choices=["float32", "float16"],
        default="float32",
    )
    embed_v2_6b = subparsers.add_parser("embed-v2-context-6b")
    embed_v2_6b.add_argument("--root", type=Path, default=ROOT)
    embed_v2_6b.add_argument(
        "--cache", type=Path, default=DEFAULT_ESMC6B_HIERARCHY_CACHE
    )
    embed_v2_6b.add_argument("--model", default=DEFAULT_ESMC6B_MODEL)
    embed_v2_6b.add_argument("--device", default="cuda")
    embed_v2_6b.add_argument("--window-radius", type=int, default=4)
    embed_v2_6b.add_argument("--max-tokens", type=int, default=4096)
    embed_v2_6b.add_argument("--max-batch-size", type=int, default=8)
    embed_v2_6b.add_argument(
        "--inference-dtype",
        choices=["float32", "bfloat16"],
        default="float32",
    )
    embed_v2_6b.add_argument(
        "--storage-dtype",
        choices=["float32", "float16"],
        default="float32",
    )
    embed_v2_6b.add_argument(
        "--promotion-report",
        type=Path,
        default=DEFAULT_V2_CHECKPOINTS / "hierarchy_ablation.json",
    )
    features_v2 = subparsers.add_parser("features-v2")
    features_v2.add_argument("--root", type=Path, default=ROOT)
    features_v2.add_argument("--cache", type=Path, default=DEFAULT_HIERARCHY_CACHE)
    features_v2.add_argument("--output", type=Path, default=DEFAULT_V2_FEATURES)
    features_v2.add_argument("--structure", type=Path, default=DEFAULT_V2_STRUCTURE)
    features_v2.add_argument("--seed", type=int, default=20260715)
    features_v2_multi = subparsers.add_parser("features-v2-multi")
    features_v2_multi.add_argument("--root", type=Path, default=ROOT)
    features_v2_multi.add_argument(
        "--cache", type=Path, default=DEFAULT_HIERARCHY_CACHE
    )
    features_v2_multi.add_argument("--output", type=Path, default=DEFAULT_V2_FEATURES)
    train_v2 = subparsers.add_parser("train-v2")
    train_v2.add_argument("--features", type=Path, default=DEFAULT_FEATURES)
    train_v2.add_argument("--checkpoints", type=Path, default=DEFAULT_V2_CHECKPOINTS)
    train_v2.add_argument("--device", default="cuda")
    train_v2.add_argument("--seed", type=int, default=20260715)
    train_v2.add_argument("--single-epochs", type=int, default=30)
    train_v2.add_argument("--patience", type=int, default=6)
    train_v2.add_argument("--batch-size", type=int, default=1024)
    train_v2.add_argument("--learning-rate", type=float, default=1e-3)
    train_v2.add_argument("--weight-decay", type=float, default=1e-4)
    train_v2.add_argument("--ensemble-size", type=int, default=5)
    train_hierarchy = subparsers.add_parser("train-v2-hierarchy")
    train_hierarchy.add_argument("--features", type=Path, default=DEFAULT_V2_FEATURES)
    train_hierarchy.add_argument("--cache", type=Path, default=DEFAULT_HIERARCHY_CACHE)
    train_hierarchy.add_argument(
        "--baseline-checkpoints",
        type=Path,
        default=DEFAULT_CHECKPOINTS,
    )
    train_hierarchy.add_argument(
        "--checkpoints",
        type=Path,
        default=DEFAULT_V2_CHECKPOINTS,
    )
    train_hierarchy.add_argument("--device", default="cuda")
    train_hierarchy.add_argument("--seed", type=int, default=20260715)
    train_hierarchy.add_argument("--discovery-epochs", type=int, default=20)
    train_hierarchy.add_argument("--discovery-patience", type=int, default=4)
    train_hierarchy.add_argument("--main-epochs", type=int, default=70)
    train_hierarchy.add_argument("--minimum-main-epochs", type=int, default=50)
    train_hierarchy.add_argument("--main-patience", type=int, default=8)
    train_hierarchy.add_argument("--batch-size", type=int, default=256)
    train_hierarchy.add_argument("--learning-rate", type=float, default=7.5e-4)
    train_hierarchy.add_argument("--weight-decay", type=float, default=1e-4)
    train_hierarchy.add_argument("--ensemble-size", type=int, default=3)
    train_hierarchy_6b = subparsers.add_parser("train-v2-hierarchy-6b")
    train_hierarchy_6b.add_argument(
        "--features", type=Path, default=ROOT / "artifacts/features_v2_6b"
    )
    train_hierarchy_6b.add_argument(
        "--cache", type=Path, default=DEFAULT_ESMC6B_HIERARCHY_CACHE
    )
    train_hierarchy_6b.add_argument(
        "--baseline-checkpoints",
        type=Path,
        default=DEFAULT_ESMC6B_CHECKPOINTS,
    )
    train_hierarchy_6b.add_argument(
        "--discovery-report",
        type=Path,
        default=DEFAULT_V2_CHECKPOINTS / "hierarchy_ablation.json",
    )
    train_hierarchy_6b.add_argument(
        "--checkpoints", type=Path, default=DEFAULT_ESMC6B_V2_CHECKPOINTS
    )
    train_hierarchy_6b.add_argument("--seed", type=int, default=20260715)
    train_hierarchy_6b.add_argument("--epochs", type=int, default=70)
    train_hierarchy_6b.add_argument("--minimum-epochs", type=int, default=50)
    train_hierarchy_6b.add_argument("--patience", type=int, default=8)
    train_hierarchy_6b.add_argument("--batch-size", type=int, default=256)
    train_hierarchy_6b.add_argument("--learning-rate", type=float, default=7.5e-4)
    train_hierarchy_6b.add_argument("--weight-decay", type=float, default=1e-4)
    train_hierarchy_6b.add_argument("--ensemble-size", type=int, default=5)
    train_hierarchy_6b.add_argument("--device", default="cuda")
    train_state_potential = subparsers.add_parser("train-state-potential")
    train_state_potential.add_argument(
        "--features",
        type=Path,
        default=DEFAULT_FP32_V2_FEATURES,
    )
    train_state_potential.add_argument(
        "--cache",
        type=Path,
        default=DEFAULT_FP32_HIERARCHY_CACHE,
    )
    train_state_potential.add_argument(
        "--baseline-checkpoint",
        type=Path,
        default=(
            DEFAULT_FP32_V2_CHECKPOINTS / "hierarchy_selected_ensemble.pt"
        ),
    )
    train_state_potential.add_argument(
        "--checkpoints",
        type=Path,
        default=DEFAULT_STATE_POTENTIAL_CHECKPOINTS,
    )
    train_state_potential.add_argument("--seed", type=int, default=20260718)
    train_state_potential.add_argument("--epochs", type=int, default=70)
    train_state_potential.add_argument("--minimum-epochs", type=int, default=50)
    train_state_potential.add_argument("--patience", type=int, default=8)
    train_state_potential.add_argument("--batch-size", type=int, default=256)
    train_state_potential.add_argument(
        "--learning-rate", type=float, default=7.5e-4
    )
    train_state_potential.add_argument(
        "--weight-decay", type=float, default=1e-4
    )
    train_state_potential.add_argument("--ensemble-size", type=int, default=3)
    train_state_potential.add_argument("--device", default="cuda")
    represent_state_transfer = subparsers.add_parser(
        "represent-state-potential-transfer"
    )
    represent_state_transfer.add_argument(
        "--features",
        type=Path,
        default=DEFAULT_FP32_V2_FEATURES,
    )
    represent_state_transfer.add_argument(
        "--cache",
        type=Path,
        default=DEFAULT_FP32_HIERARCHY_CACHE,
    )
    represent_state_transfer.add_argument(
        "--state-checkpoint",
        type=Path,
        default=(
            DEFAULT_STATE_POTENTIAL_CHECKPOINTS
            / "state_potential_ensemble.pt"
        ),
    )
    represent_state_transfer.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_STATE_POTENTIAL_TRANSFER,
    )
    represent_state_transfer.add_argument(
        "--batch-size", type=int, default=256
    )
    represent_state_transfer.add_argument("--device", default="cuda")
    represent_state_multi = subparsers.add_parser(
        "represent-state-potential-multi"
    )
    represent_state_multi.add_argument(
        "--features",
        type=Path,
        default=ROOT / "artifacts/features_v2_6b_fp32",
    )
    represent_state_multi.add_argument(
        "--cache",
        type=Path,
        default=DEFAULT_ESMC6B_HIERARCHY_CACHE,
    )
    represent_state_multi.add_argument(
        "--baseline-representations",
        type=Path,
        default=ROOT / "artifacts/features_v2_multi_6b_fp32",
    )
    represent_state_multi.add_argument(
        "--state-checkpoint",
        type=Path,
        default=(
            DEFAULT_ESMC6B_STATE_CHECKPOINTS
            / "state_potential_ensemble.pt"
        ),
    )
    represent_state_multi.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_ESMC6B_STATE_MULTI,
    )
    represent_state_multi.add_argument(
        "--batch-size", type=int, default=256
    )
    represent_state_multi.add_argument("--device", default="cuda")
    train_zero_shot_gpcr = subparsers.add_parser(
        "train-zero-shot-gpcr"
    )
    train_zero_shot_gpcr.add_argument(
        "--representations",
        type=Path,
        default=DEFAULT_ESMC6B_STATE_TRANSFER,
    )
    train_zero_shot_gpcr.add_argument(
        "--gpcr-tm-source",
        type=Path,
        default=ROOT / "data/curated/gpcr_tm_dtm.csv",
    )
    train_zero_shot_gpcr.add_argument(
        "--checkpoints",
        type=Path,
        default=DEFAULT_ESMC6B_STATE_CHECKPOINTS,
    )
    train_zero_shot_gpcr.add_argument(
        "--seed", type=int, default=20260718
    )
    train_zero_shot_gpcr.add_argument(
        "--discovery-epochs", type=int, default=40
    )
    train_zero_shot_gpcr.add_argument(
        "--discovery-minimum-epochs", type=int, default=20
    )
    train_zero_shot_gpcr.add_argument(
        "--main-epochs", type=int, default=120
    )
    train_zero_shot_gpcr.add_argument(
        "--minimum-main-epochs", type=int, default=50
    )
    train_zero_shot_gpcr.add_argument(
        "--patience", type=int, default=12
    )
    train_zero_shot_gpcr.add_argument(
        "--batch-size", type=int, default=256
    )
    train_zero_shot_gpcr.add_argument(
        "--learning-rate", type=float, default=5e-4
    )
    train_zero_shot_gpcr.add_argument(
        "--weight-decay", type=float, default=1e-4
    )
    train_zero_shot_gpcr.add_argument(
        "--ensemble-size", type=int, default=5
    )
    train_zero_shot_gpcr.add_argument("--device", default="cuda")
    represent_multi = subparsers.add_parser("represent-v2-multi")
    represent_multi.add_argument("--features", type=Path, default=DEFAULT_V2_FEATURES)
    represent_multi.add_argument("--cache", type=Path, default=DEFAULT_HIERARCHY_CACHE)
    represent_multi.add_argument(
        "--base-checkpoint",
        type=Path,
        default=DEFAULT_V2_CHECKPOINTS / "hierarchy_selected_ensemble.pt",
    )
    represent_multi.add_argument("--output", type=Path, default=DEFAULT_V2_MULTI)
    represent_multi.add_argument("--batch-size", type=int, default=256)
    represent_multi.add_argument("--device", default="cuda")
    train_multi = subparsers.add_parser("train-v2-multi")
    train_multi.add_argument(
        "--representations", type=Path, default=DEFAULT_V2_MULTI
    )
    train_multi.add_argument(
        "--checkpoints", type=Path, default=DEFAULT_V2_CHECKPOINTS
    )
    train_multi.add_argument("--seed", type=int, default=20260715)
    train_multi.add_argument("--epochs", type=int, default=70)
    train_multi.add_argument("--minimum-epochs", type=int, default=50)
    train_multi.add_argument("--patience", type=int, default=8)
    train_multi.add_argument("--batch-size", type=int, default=256)
    train_multi.add_argument("--learning-rate", type=float, default=7.5e-4)
    train_multi.add_argument("--weight-decay", type=float, default=1e-4)
    train_multi.add_argument("--device", default="cuda")
    train_multi.add_argument(
        "--constituent-state-weight",
        type=float,
        help=(
            "override the state-potential weight for multi-mutant "
            "constituent ddG; select on protein-held-out validation only"
        ),
    )
    promote_multi = subparsers.add_parser("promote-v2-multi")
    promote_multi.add_argument(
        "--candidate-checkpoints",
        type=Path,
        default=DEFAULT_ESMC6B_STATE_CHECKPOINTS,
    )
    promote_multi.add_argument(
        "--baseline-checkpoints",
        type=Path,
        default=DEFAULT_ESMC6B_V2_CHECKPOINTS,
    )
    represent_transfer = subparsers.add_parser("represent-v2-transfer")
    represent_transfer.add_argument(
        "--features", type=Path, default=DEFAULT_V2_FEATURES
    )
    represent_transfer.add_argument(
        "--cache", type=Path, default=DEFAULT_HIERARCHY_CACHE
    )
    represent_transfer.add_argument(
        "--base-checkpoint",
        type=Path,
        default=DEFAULT_V2_CHECKPOINTS / "hierarchy_selected_ensemble.pt",
    )
    represent_transfer.add_argument(
        "--output", type=Path, default=DEFAULT_V2_TRANSFER
    )
    represent_transfer.add_argument("--batch-size", type=int, default=256)
    represent_transfer.add_argument("--device", default="cuda")
    train_transfer = subparsers.add_parser("train-v2-transfer")
    train_transfer.add_argument(
        "--representations", type=Path, default=DEFAULT_V2_TRANSFER
    )
    train_transfer.add_argument(
        "--checkpoints", type=Path, default=DEFAULT_V2_CHECKPOINTS
    )
    train_transfer.add_argument("--seed", type=int, default=20260715)
    train_transfer.add_argument("--epochs", type=int, default=120)
    train_transfer.add_argument("--minimum-epochs", type=int, default=50)
    train_transfer.add_argument("--patience", type=int, default=12)
    train_transfer.add_argument("--batch-size", type=int, default=256)
    train_transfer.add_argument("--learning-rate", type=float, default=5e-4)
    train_transfer.add_argument("--weight-decay", type=float, default=1e-4)
    train_transfer.add_argument("--device", default="cuda")
    evaluate_v2 = subparsers.add_parser("evaluate-v2")
    evaluate_v2.add_argument("--features", type=Path, default=DEFAULT_FEATURES)
    evaluate_v2.add_argument(
        "--baseline-checkpoints", type=Path, default=DEFAULT_CHECKPOINTS
    )
    evaluate_v2.add_argument(
        "--checkpoints", type=Path, default=DEFAULT_V2_CHECKPOINTS
    )
    evaluate_v2.add_argument("--output", type=Path)
    evaluate_v2.add_argument("--device", default="cuda")
    evaluate_v2.add_argument("--batch-size", type=int, default=2048)
    predict = subparsers.add_parser("predict")
    predict.add_argument("--sequence")
    predict.add_argument("--fasta", type=Path)
    predict.add_argument("--mutations", required=True)
    predict.add_argument("--checkpoints", type=Path, default=DEFAULT_CHECKPOINTS)
    predict.add_argument("--model", default="esmc_600m", choices=["esmc_600m"])
    predict.add_argument("--device", default="cuda")
    predict_v2 = subparsers.add_parser("predict-v2")
    predict_v2.add_argument("--sequence")
    predict_v2.add_argument("--fasta", type=Path)
    predict_v2.add_argument("--mutations", required=True)
    predict_v2.add_argument(
        "--checkpoints", type=Path, default=DEFAULT_FP32_V2_CHECKPOINTS
    )
    predict_v2.add_argument(
        "--state-potential-checkpoint",
        type=Path,
        default=(
            DEFAULT_STATE_POTENTIAL_CHECKPOINTS
            / "state_potential_ensemble.pt"
        ),
    )
    predict_v2.add_argument(
        "--model", default="esmc_600m", choices=["esmc_600m"]
    )
    predict_v2.add_argument("--device", default="cuda")
    predict_v2.add_argument(
        "--topology",
        choices=[
            "membrane",
            "alpha_helical",
            "beta_barrel",
            "alpha_helical_gpcr",
            "unknown",
        ],
    )
    predict_v2.add_argument(
        "--generic-numbering",
        help="comma-separated position=generic-number entries",
    )
    predict_v2.add_argument("--pdb", type=Path)
    predict_v2.add_argument(
        "--proteinmpnn-repository",
        type=Path,
        default=DEFAULT_PROTEINMPNN_REPOSITORY,
    )
    predict_v2.add_argument("--max-tokens", type=int, default=8192)
    predict_v2.add_argument("--max-batch-size", type=int, default=128)
    predict_v2_6b = subparsers.add_parser("predict-v2-6b")
    predict_v2_6b.add_argument("--sequence")
    predict_v2_6b.add_argument("--fasta", type=Path)
    predict_v2_6b.add_argument("--mutations", required=True)
    predict_v2_6b.add_argument(
        "--checkpoints", type=Path, default=DEFAULT_ESMC6B_V2_CHECKPOINTS
    )
    predict_v2_6b.add_argument(
        "--state-potential-checkpoint",
        type=Path,
        default=(
            DEFAULT_ESMC6B_STATE_CHECKPOINTS
            / "state_potential_ensemble.pt"
        ),
    )
    predict_v2_6b.add_argument("--model", default=DEFAULT_ESMC6B_MODEL)
    predict_v2_6b.add_argument("--device", default="cuda")
    predict_v2_6b.add_argument(
        "--topology",
        choices=[
            "membrane",
            "alpha_helical",
            "beta_barrel",
            "alpha_helical_gpcr",
            "unknown",
        ],
    )
    predict_v2_6b.add_argument(
        "--generic-numbering",
        help="comma-separated position=generic-number entries",
    )
    predict_v2_6b.add_argument("--pdb", type=Path)
    predict_v2_6b.add_argument(
        "--proteinmpnn-repository",
        type=Path,
        default=DEFAULT_PROTEINMPNN_REPOSITORY,
    )
    predict_v2_6b.add_argument("--max-tokens", type=int, default=4096)
    predict_v2_6b.add_argument("--max-batch-size", type=int, default=2)
    screen = subparsers.add_parser("screen")
    screen.add_argument("--sequence")
    screen.add_argument("--fasta", type=Path)
    screen.add_argument("--positions", help="comma-separated positions/ranges; default all")
    screen.add_argument("--output", type=Path, default=ROOT / "artifacts/screen.csv")
    screen.add_argument("--top", type=int, default=50)
    screen.add_argument("--checkpoints", type=Path, default=DEFAULT_CHECKPOINTS)
    screen.add_argument("--model", default="esmc_600m", choices=["esmc_600m"])
    screen.add_argument("--device", default="cuda")
    screen.add_argument("--max-tokens", type=int, default=8192)
    screen.add_argument("--max-batch-size", type=int, default=128)
    screen_v2 = subparsers.add_parser("screen-v2")
    screen_v2.add_argument("--sequence")
    screen_v2.add_argument("--fasta", type=Path)
    screen_v2.add_argument(
        "--positions", help="comma-separated positions/ranges; default all"
    )
    screen_v2.add_argument(
        "--output", type=Path, default=ROOT / "artifacts/screen_v2.csv"
    )
    screen_v2.add_argument("--top", type=int, default=50)
    screen_v2.add_argument(
        "--checkpoints", type=Path, default=DEFAULT_FP32_V2_CHECKPOINTS
    )
    screen_v2.add_argument(
        "--state-potential-checkpoint",
        type=Path,
        default=(
            DEFAULT_STATE_POTENTIAL_CHECKPOINTS
            / "state_potential_ensemble.pt"
        ),
    )
    screen_v2.add_argument(
        "--legacy-checkpoints", type=Path, default=DEFAULT_CHECKPOINTS
    )
    screen_v2.add_argument(
        "--model", default="esmc_600m", choices=["esmc_600m"]
    )
    screen_v2.add_argument("--device", default="cuda")
    screen_v2.add_argument(
        "--topology",
        choices=[
            "membrane",
            "alpha_helical",
            "beta_barrel",
            "alpha_helical_gpcr",
            "unknown",
        ],
    )
    screen_v2.add_argument(
        "--generic-numbering",
        help="comma-separated position=generic-number entries",
    )
    screen_v2.add_argument("--pdb", type=Path)
    screen_v2.add_argument(
        "--proteinmpnn-repository",
        type=Path,
        default=DEFAULT_PROTEINMPNN_REPOSITORY,
    )
    screen_v2.add_argument("--max-tokens", type=int, default=8192)
    screen_v2.add_argument("--max-batch-size", type=int, default=128)
    screen_v2.add_argument(
        "--scan-mode",
        choices=["exact", "state-only"],
        default="exact",
        help=(
            "exact runs the promoted hierarchy/state blend; state-only "
            "scores all amino acids from one WT embedding as a fast pre-screen"
        ),
    )
    screen_v2_6b = subparsers.add_parser("screen-v2-6b")
    screen_v2_6b.add_argument("--sequence")
    screen_v2_6b.add_argument("--fasta", type=Path)
    screen_v2_6b.add_argument(
        "--positions", help="comma-separated positions/ranges; default all"
    )
    screen_v2_6b.add_argument(
        "--output", type=Path, default=ROOT / "artifacts/screen_v2_6b.csv"
    )
    screen_v2_6b.add_argument("--top", type=int, default=50)
    screen_v2_6b.add_argument(
        "--checkpoints", type=Path, default=DEFAULT_ESMC6B_V2_CHECKPOINTS
    )
    screen_v2_6b.add_argument(
        "--state-potential-checkpoint",
        type=Path,
        default=(
            DEFAULT_ESMC6B_STATE_CHECKPOINTS
            / "state_potential_ensemble.pt"
        ),
    )
    screen_v2_6b.add_argument("--model", default=DEFAULT_ESMC6B_MODEL)
    screen_v2_6b.add_argument("--device", default="cuda")
    screen_v2_6b.add_argument(
        "--topology",
        choices=[
            "membrane",
            "alpha_helical",
            "beta_barrel",
            "alpha_helical_gpcr",
            "unknown",
        ],
    )
    screen_v2_6b.add_argument(
        "--generic-numbering",
        help="comma-separated position=generic-number entries",
    )
    screen_v2_6b.add_argument("--pdb", type=Path)
    screen_v2_6b.add_argument(
        "--proteinmpnn-repository",
        type=Path,
        default=DEFAULT_PROTEINMPNN_REPOSITORY,
    )
    screen_v2_6b.add_argument("--max-tokens", type=int, default=4096)
    screen_v2_6b.add_argument("--max-batch-size", type=int, default=2)
    screen_v2_6b.add_argument(
        "--scan-mode",
        choices=["exact", "state-only"],
        default="exact",
        help=(
            "exact runs the promoted hierarchy/state blend; state-only "
            "scores all amino acids from one WT embedding as a fast pre-screen"
        ),
    )
    rerank = subparsers.add_parser("rerank-6b")
    rerank.add_argument("--sequence")
    rerank.add_argument("--fasta", type=Path)
    rerank.add_argument("--input", type=Path, required=True)
    rerank.add_argument(
        "--output",
        type=Path,
        default=ROOT / "artifacts/screen_esmc6b.csv",
    )
    rerank.add_argument("--top", type=int, default=50)
    rerank.add_argument(
        "--checkpoints",
        type=Path,
        default=DEFAULT_ESMC6B_CHECKPOINTS,
    )
    rerank.add_argument("--model", default=DEFAULT_ESMC6B_MODEL)
    rerank.add_argument("--device", default="cuda")
    rerank.add_argument("--max-tokens", type=int, default=4096)
    rerank.add_argument("--max-batch-size", type=int, default=16)
    predict_6b = subparsers.add_parser("predict-6b")
    predict_6b.add_argument("--sequence")
    predict_6b.add_argument("--fasta", type=Path)
    predict_6b.add_argument("--mutations", required=True)
    predict_6b.add_argument(
        "--checkpoints",
        type=Path,
        default=DEFAULT_ESMC6B_CHECKPOINTS,
    )
    predict_6b.add_argument("--model", default=DEFAULT_ESMC6B_MODEL)
    predict_6b.add_argument("--device", default="cuda")
    predict_6b.add_argument("--max-tokens", type=int, default=4096)
    predict_6b.add_argument("--max-batch-size", type=int, default=16)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command in {
        "predict",
        "predict-v2",
        "predict-v2-6b",
        "screen",
        "screen-v2",
        "screen-v2-6b",
        "rerank-6b",
        "predict-6b",
    } and (
        args.sequence is None
    ) == (
        args.fasta is None
    ):
        parser.error(f"{args.command} requires exactly one of --sequence or --fasta")
    commands = {
        "embed": command_embed,
        "embed-v2-context": command_embed_v2_context,
        "embed-v2-context-6b": command_embed_v2_context_6b,
        "structure-v2": command_structure_v2,
        "structure-v2-aligned": command_structure_v2_aligned,
        "features-v2": command_features_v2,
        "features-v2-multi": command_features_v2_multi,
        "features": command_features,
        "train": command_train,
        "train-v2": command_train_v2,
        "train-v2-hierarchy": command_train_v2_hierarchy,
        "train-v2-hierarchy-6b": command_train_v2_hierarchy_6b,
        "train-state-potential": command_train_state_potential,
        "represent-state-potential-transfer": (
            command_represent_state_potential_transfer
        ),
        "represent-state-potential-multi": (
            command_represent_state_potential_multi
        ),
        "train-zero-shot-gpcr": command_train_zero_shot_gpcr,
        "represent-v2-multi": command_represent_v2_multi,
        "train-v2-multi": command_train_v2_multi,
        "promote-v2-multi": command_promote_v2_multi,
        "represent-v2-transfer": command_represent_v2_transfer,
        "train-v2-transfer": command_train_v2_transfer,
        "evaluate-v2": command_evaluate_v2,
        "run": command_run,
        "predict": command_predict,
        "predict-v2": command_predict_v2,
        "predict-v2-6b": command_predict_v2_6b,
        "screen": command_screen,
        "screen-v2": command_screen_v2,
        "screen-v2-6b": command_screen_v2_6b,
        "rerank-6b": command_rerank_esmc6b,
        "predict-6b": command_predict_esmc6b,
    }
    _json(commands[args.command](args))


if __name__ == "__main__":
    main()
