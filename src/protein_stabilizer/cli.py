"""Command-line interface for data preparation, training, and inference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from .data import DatasetPaths, all_embedding_requests
from .embeddings import build_embedding_cache
from .features import build_feature_files
from .predictor import predict_mutations, screen_single_mutants
from .training import (
    train_epistasis_head,
    train_gpcr_calibration,
    train_single_head,
    train_transfer_heads,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CACHE = ROOT / "embeddings/esmc_600m/residue_cache.h5"
DEFAULT_FEATURES = ROOT / "artifacts/features"
DEFAULT_CHECKPOINTS = ROOT / "checkpoints/esmc_600m"


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


def command_train(args: argparse.Namespace) -> dict[str, object]:
    single = train_single_head(
        args.features,
        args.checkpoints,
        seed=args.seed,
        epochs=args.single_epochs,
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
    parser.add_argument("--multi-epochs", type=int, default=25)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="protein-stabilizer")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("embed", "features", "train", "run"):
        child = subparsers.add_parser(name)
        _common_pipeline_arguments(child)
    predict = subparsers.add_parser("predict")
    predict.add_argument("--sequence")
    predict.add_argument("--fasta", type=Path)
    predict.add_argument("--mutations", required=True)
    predict.add_argument("--checkpoints", type=Path, default=DEFAULT_CHECKPOINTS)
    predict.add_argument("--model", default="esmc_600m", choices=["esmc_600m"])
    predict.add_argument("--device", default="cuda")
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
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command in {"predict", "screen"} and (args.sequence is None) == (
        args.fasta is None
    ):
        parser.error(f"{args.command} requires exactly one of --sequence or --fasta")
    commands = {
        "embed": command_embed,
        "features": command_features,
        "train": command_train,
        "run": command_run,
        "predict": command_predict,
        "screen": command_screen,
    }
    _json(commands[args.command](args))


if __name__ == "__main__":
    main()
