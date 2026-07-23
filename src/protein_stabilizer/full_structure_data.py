"""Leakage-controlled full-protein tensors for structure-aware training."""

from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Iterable, Sequence

import h5py
import numpy as np

from .data import (
    AMINO_ACIDS,
    DatasetPaths,
    EmbeddingRequest,
    Mutation,
    double_rows,
    normalize_sequence,
    sequence_hash,
    single_rows,
)
from .embeddings import ESMCEmbedder, file_sha256, token_batches
from .structure import _extract_pdb_archive


FULL_SEQUENCE_EMBEDDING_SCHEMA = (
    "protein-stabilizer.full-sequence-embeddings.v1"
)
FULL_STRUCTURE_DATA_SCHEMA = "protein-stabilizer.full-structure-data.v1"
DEFAULT_SPLIT_SEED = 20260723


def canonical_protein_id(value: str) -> str:
    """Normalize the two identifier conventions used by MegaScale S and D."""

    return Path(str(value)).stem.lower()


@dataclass(frozen=True)
class ProteinRecord:
    protein_id: str
    sequence: str


@dataclass(frozen=True)
class MutationRow:
    protein_id: str
    positions: tuple[int, ...]
    wt_amino_acids: tuple[int, ...]
    mutant_amino_acids: tuple[int, ...]
    target: float
    source_partition: str


@dataclass(frozen=True)
class MegaScaleRecords:
    proteins: tuple[ProteinRecord, ...]
    singles: tuple[MutationRow, ...]
    doubles: tuple[MutationRow, ...]
    source_paths: tuple[Path, ...]


def _amino_acid_index(value: str) -> int:
    try:
        return AMINO_ACIDS.index(value)
    except ValueError as exc:
        raise ValueError(f"non-canonical amino acid {value!r}") from exc


def collect_megascale_records(root: Path) -> MegaScaleRecords:
    """Read every staged MegaScale single/double row with one WT per protein."""

    paths = DatasetPaths(Path(root).resolve())
    single_sources = (
        ("historical_train", paths.single("train")),
        ("historical_test", paths.single("test")),
    )
    double_sources = tuple(
        (f"historical_{split}", paths.double(split))
        for split in ("train", "val", "test")
    )
    sequences: dict[str, str] = {}
    singles: list[MutationRow] = []
    doubles: list[MutationRow] = []

    def register(protein: str, sequence: str) -> str:
        identifier = canonical_protein_id(protein)
        normalized = normalize_sequence(sequence)
        previous = sequences.setdefault(identifier, normalized)
        if previous != normalized:
            raise RuntimeError(
                f"inconsistent WT sequences for MegaScale protein {identifier}"
            )
        return identifier

    for source_partition, path in single_sources:
        if not path.is_file():
            raise FileNotFoundError(path)
        for row in single_rows(path):
            protein = register(
                str(row["protein_id"]), str(row["wt_sequence"])
            )
            mutation = Mutation.parse(str(row["mutation"]))
            singles.append(
                MutationRow(
                    protein_id=protein,
                    positions=(mutation.position - 1,),
                    wt_amino_acids=(_amino_acid_index(mutation.wt),),
                    mutant_amino_acids=(
                        _amino_acid_index(mutation.mutant),
                    ),
                    target=float(row["target"]),
                    source_partition=source_partition,
                )
            )
    for source_partition, path in double_sources:
        if not path.is_file():
            raise FileNotFoundError(path)
        for row in double_rows(path):
            protein = register(
                str(row["protein_id"]), str(row["wt_sequence"])
            )
            mutations = tuple(
                Mutation.parse(value) for value in row["mutations"]
            )
            doubles.append(
                MutationRow(
                    protein_id=protein,
                    positions=tuple(
                        mutation.position - 1 for mutation in mutations
                    ),
                    wt_amino_acids=tuple(
                        _amino_acid_index(mutation.wt)
                        for mutation in mutations
                    ),
                    mutant_amino_acids=tuple(
                        _amino_acid_index(mutation.mutant)
                        for mutation in mutations
                    ),
                    target=float(row["target"]),
                    source_partition=source_partition,
                )
            )
    proteins = tuple(
        ProteinRecord(protein_id=protein, sequence=sequence)
        for protein, sequence in sorted(sequences.items())
    )
    return MegaScaleRecords(
        proteins=proteins,
        singles=tuple(singles),
        doubles=tuple(doubles),
        source_paths=tuple(
            path for _, path in (*single_sources, *double_sources)
        ),
    )


def _string_dataset(
    group: h5py.Group,
    name: str,
    values: Sequence[str],
) -> None:
    group.create_dataset(
        name,
        data=np.asarray(values, dtype=object),
        dtype=h5py.string_dtype(encoding="utf-8"),
    )


def build_full_sequence_embeddings(
    root: Path,
    output: Path,
    *,
    model_name: str = "esmc_600m",
    device: str = "cuda",
    inference_dtype: str = "float32",
    storage_dtype: str = "float32",
    max_tokens: int = 8192,
    max_batch_size: int = 64,
) -> dict[str, object]:
    """Embed each unique WT once and persist every final-layer residue."""

    records = collect_megascale_records(root)
    output = Path(output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(
            f"full-sequence embedding bank already exists: {output}"
        )
    if model_name == "esmc_600m":
        embedder = ESMCEmbedder(
            model_name=model_name,
            device=device,
            storage_dtype=storage_dtype,
        )
    else:
        from .esmc6b import ESMC6BEmbedder

        embedder = ESMC6BEmbedder(
            model_name,
            device,
            inference_dtype=inference_dtype,
            storage_dtype=storage_dtype,
        )
    requests = [
        EmbeddingRequest(
            sequence=record.sequence,
            positions=tuple(range(1, len(record.sequence) + 1)),
        )
        for record in records.proteins
    ]
    maximum_length = max(len(record.sequence) for record in records.proteins)
    dimension = int(embedder.dimension)
    partial = output.with_suffix(output.suffix + ".partial")
    partial.unlink(missing_ok=True)
    with h5py.File(partial, "w", libver="latest") as handle:
        handle.attrs["schema"] = FULL_SEQUENCE_EMBEDDING_SCHEMA
        handle.attrs["model_provenance"] = (
            embedder.provenance.canonical_json()
        )
        handle.attrs["storage_dtype"] = storage_dtype
        _string_dataset(
            handle,
            "protein_id",
            [record.protein_id for record in records.proteins],
        )
        _string_dataset(
            handle,
            "sequence",
            [record.sequence for record in records.proteins],
        )
        handle.create_dataset(
            "sequence_hash",
            data=np.asarray(
                [
                    sequence_hash(record.sequence).encode("ascii")
                    for record in records.proteins
                ],
                dtype="S64",
            ),
        )
        handle.create_dataset(
            "length",
            data=np.asarray(
                [len(record.sequence) for record in records.proteins],
                dtype=np.int32,
            ),
        )
        residue = handle.create_dataset(
            "residue",
            shape=(len(records.proteins), maximum_length, dimension),
            dtype=np.dtype(storage_dtype),
            chunks=(1, maximum_length, dimension),
            compression="lzf",
            fillvalue=0,
        )
        global_mean = handle.create_dataset(
            "global_mean",
            shape=(len(records.proteins), dimension),
            dtype=np.dtype(storage_dtype),
            chunks=(1, dimension),
            compression="lzf",
        )
        request_index = {
            request.sequence_hash: index
            for index, request in enumerate(requests)
        }
        completed = 0
        for batch in token_batches(
            requests,
            max_tokens=max_tokens,
            max_batch_size=max_batch_size,
        ):
            values = embedder.encode(batch)
            for request, embedding in zip(batch, values, strict=True):
                index = request_index[request.sequence_hash]
                length = len(request.sequence)
                if embedding.shape != (length, dimension):
                    raise RuntimeError(
                        "full-sequence ESM-C embedding shape mismatch"
                    )
                residue[index, :length] = embedding
                global_mean[index] = embedding.astype(
                    np.float32, copy=False
                ).mean(axis=0).astype(np.dtype(storage_dtype))
                completed += 1
            print(
                f"embedded full WT proteins: {completed}/{len(requests)}",
                flush=True,
            )
        handle.attrs["request_manifest_sha256"] = hashlib.sha256(
            "\n".join(
                f"{request.sequence_hash}:{len(request.sequence)}"
                for request in requests
            ).encode("ascii")
        ).hexdigest()
        handle.flush()
    partial.replace(output)
    return {
        "schema": FULL_SEQUENCE_EMBEDDING_SCHEMA,
        "path": str(output),
        "sha256": file_sha256(output),
        "proteins": len(records.proteins),
        "maximum_length": maximum_length,
        "embedding_dimension": dimension,
        "model_provenance": asdict(embedder.provenance),
    }


def _mmseqs_clusters(
    proteins: Sequence[ProteinRecord],
    temp_root: Path,
    *,
    minimum_identity: float,
    coverage: float,
) -> tuple[dict[str, str], dict[str, object]]:
    temp_root = Path(temp_root).resolve()
    if not temp_root.is_relative_to(Path("/data/fast")):
        raise ValueError("bulk split temporary root must be below /data/fast")
    temp_root.mkdir(parents=True, exist_ok=True)
    version = subprocess.run(
        ["mmseqs", "version"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    with TemporaryDirectory(dir=temp_root) as temporary:
        directory = Path(temporary)
        fasta = directory / "proteins.fasta"
        fasta.write_text(
            "".join(
                f">p{index}\n{record.sequence}\n"
                for index, record in enumerate(proteins)
            ),
            encoding="ascii",
        )
        prefix = directory / "family"
        subprocess.run(
            [
                "mmseqs",
                "easy-cluster",
                str(fasta),
                str(prefix),
                str(directory / "work"),
                "--min-seq-id",
                str(minimum_identity),
                "-c",
                str(coverage),
                "--cov-mode",
                "0",
                "--cluster-mode",
                "2",
                "--threads",
                "8",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        cluster_file = Path(f"{prefix}_cluster.tsv")
        assignments: dict[int, int] = {}
        for line in cluster_file.read_text(encoding="utf-8").splitlines():
            representative, member = line.split("\t")
            assignments[int(member.removeprefix("p"))] = int(
                representative.removeprefix("p")
            )
    if len(assignments) != len(proteins):
        missing = sorted(set(range(len(proteins))) - set(assignments))
        raise RuntimeError(f"MMseqs omitted protein indices {missing}")
    return (
        {
            record.protein_id: f"family_{assignments[index]:04d}"
            for index, record in enumerate(proteins)
        },
        {
            "program": "mmseqs",
            "version": version,
            "minimum_sequence_identity": minimum_identity,
            "coverage": coverage,
            "coverage_mode": 0,
            "cluster_mode": 2,
        },
    )


def _cluster_split_assignments(
    proteins: Sequence[ProteinRecord],
    families: dict[str, str],
    singles: Sequence[MutationRow],
    doubles: Sequence[MutationRow],
    *,
    seed: int,
    outer_fraction: float,
    folds: int,
) -> dict[str, int]:
    if folds < 2:
        raise ValueError("at least two development folds are required")
    if not 0.05 <= outer_fraction <= 0.5:
        raise ValueError("outer fraction must be between 0.05 and 0.5")
    single_count = {record.protein_id: 0 for record in proteins}
    double_count = {record.protein_id: 0 for record in proteins}
    for row in singles:
        single_count[row.protein_id] += 1
    for row in doubles:
        double_count[row.protein_id] += 1
    family_members: dict[str, list[str]] = {}
    for protein, family in families.items():
        family_members.setdefault(family, []).append(protein)

    weights: dict[str, tuple[int, int]] = {}
    for family, members in family_members.items():
        weights[family] = (
            sum(single_count[protein] for protein in members),
            sum(double_count[protein] for protein in members),
        )
    total_single = sum(value[0] for value in weights.values())
    total_double = sum(value[1] for value in weights.values())
    target_outer = (
        outer_fraction * total_single,
        outer_fraction * total_double,
    )

    def tie_breaker(family: str) -> str:
        return hashlib.sha256(f"{seed}:{family}".encode("ascii")).hexdigest()

    remaining = set(family_members)
    outer: set[str] = set()
    current = (0, 0)

    def distance(value: tuple[int, int]) -> float:
        single_scale = max(target_outer[0], 1.0)
        double_scale = max(target_outer[1], 1.0)
        return (
            abs(value[0] - target_outer[0]) / single_scale
            + abs(value[1] - target_outer[1]) / double_scale
        )

    while remaining:
        candidates = []
        for family in remaining:
            weight = weights[family]
            candidate = (
                current[0] + weight[0],
                current[1] + weight[1],
            )
            candidates.append(
                (distance(candidate), tie_breaker(family), family, candidate)
            )
        best = min(candidates)
        if outer and best[0] >= distance(current):
            break
        _, _, family, current = best
        outer.add(family)
        remaining.remove(family)
    if not outer:
        family = min(remaining, key=tie_breaker)
        outer.add(family)
        remaining.remove(family)

    fold_weights = [[0, 0] for _ in range(folds)]
    fold_by_family: dict[str, int] = {}
    ordered = sorted(
        remaining,
        key=lambda family: (
            -(weights[family][0] + weights[family][1]),
            tie_breaker(family),
        ),
    )
    for family in ordered:
        weight = weights[family]

        def fold_score(fold: int) -> tuple[float, int]:
            totals = fold_weights[fold]
            return (
                totals[0] / max(total_single, 1)
                + totals[1] / max(total_double, 1),
                fold,
            )

        selected = min(range(folds), key=fold_score)
        fold_by_family[family] = selected
        fold_weights[selected][0] += weight[0]
        fold_weights[selected][1] += weight[1]
    return {
        protein.protein_id: (
            -1
            if families[protein.protein_id] in outer
            else fold_by_family[families[protein.protein_id]]
        )
        for protein in proteins
    }


def _extract_backbone(
    module: object,
    pdb_path: Path,
    target_sequence: str,
) -> tuple[np.ndarray, np.ndarray, str]:
    records = module.parse_PDB(str(pdb_path), ca_only=False)
    if len(records) != 1:
        raise ValueError(f"expected one parsed structure in {pdb_path}")
    record = records[0]
    chains = {
        key.removeprefix("seq_chain_"): str(value)
        for key, value in record.items()
        if key.startswith("seq_chain_")
    }
    matching = [
        chain for chain, sequence in chains.items() if sequence == target_sequence
    ]
    if len(matching) != 1:
        raise ValueError(
            f"{pdb_path.name} has {len(matching)} exact chains for target sequence"
        )
    chain = matching[0]
    coordinate_record = record[f"coords_chain_{chain}"]
    coordinates = np.stack(
        [
            np.asarray(
                coordinate_record[f"{atom}_chain_{chain}"],
                dtype=np.float32,
            )
            for atom in ("N", "CA", "C", "O")
        ],
        axis=1,
    )
    if coordinates.shape != (len(target_sequence), 4, 3):
        raise RuntimeError(f"coordinate shape mismatch for {pdb_path.name}")
    structure_mask = np.isfinite(coordinates).all(axis=(1, 2))
    coordinates = np.nan_to_num(
        coordinates, nan=0.0, posinf=0.0, neginf=0.0
    )
    return coordinates, structure_mask, chain


def _write_rows(
    group: h5py.Group,
    rows: Sequence[MutationRow],
    protein_index: dict[str, int],
) -> None:
    if not rows:
        raise ValueError("mutation table cannot be empty")
    mutation_count = len(rows[0].positions)
    if any(len(row.positions) != mutation_count for row in rows):
        raise ValueError("mutation table mixes mutation counts")
    group.create_dataset(
        "protein_index",
        data=np.asarray(
            [protein_index[row.protein_id] for row in rows],
            dtype=np.int32,
        ),
    )
    group.create_dataset(
        "position",
        data=np.asarray([row.positions for row in rows], dtype=np.int16),
    )
    group.create_dataset(
        "wt_amino_acid",
        data=np.asarray(
            [row.wt_amino_acids for row in rows], dtype=np.int8
        ),
    )
    group.create_dataset(
        "mutant_amino_acid",
        data=np.asarray(
            [row.mutant_amino_acids for row in rows], dtype=np.int8
        ),
    )
    group.create_dataset(
        "target",
        data=np.asarray([row.target for row in rows], dtype=np.float32),
    )
    _string_dataset(
        group,
        "source_partition",
        [row.source_partition for row in rows],
    )


def build_full_structure_dataset(
    root: Path,
    embedding_path: Path,
    structure_archive: Path,
    proteinmpnn_repository: Path,
    output: Path,
    *,
    temp_root: Path = Path(
        "/data/fast/tmp/protein-stabilizer/full-structure"
    ),
    seed: int = DEFAULT_SPLIT_SEED,
    minimum_identity: float = 0.25,
    coverage: float = 0.80,
    outer_fraction: float = 0.20,
    folds: int = 5,
) -> dict[str, object]:
    """Join WT embeddings, raw backbones, mutation rows, and family splits."""

    from .full_structure import load_trainable_proteinmpnn

    records = collect_megascale_records(root)
    embedding_path = Path(embedding_path).resolve()
    structure_archive = Path(structure_archive).resolve()
    output = Path(output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(
            f"full-structure data bank already exists: {output}"
        )
    _, proteinmpnn_module, proteinmpnn_provenance = (
        load_trainable_proteinmpnn(proteinmpnn_repository)
    )
    families, clustering = _mmseqs_clusters(
        records.proteins,
        temp_root,
        minimum_identity=minimum_identity,
        coverage=coverage,
    )
    splits = _cluster_split_assignments(
        records.proteins,
        families,
        records.singles,
        records.doubles,
        seed=seed,
        outer_fraction=outer_fraction,
        folds=folds,
    )
    protein_index = {
        record.protein_id: index
        for index, record in enumerate(records.proteins)
    }

    with h5py.File(embedding_path, "r") as embeddings:
        if (
            embeddings.attrs.get("schema")
            != FULL_SEQUENCE_EMBEDDING_SCHEMA
        ):
            raise RuntimeError("full-sequence embedding schema mismatch")
        embedded_ids = list(embeddings["protein_id"].asstr()[:])
        embedded_sequences = list(embeddings["sequence"].asstr()[:])
        expected_ids = [record.protein_id for record in records.proteins]
        expected_sequences = [record.sequence for record in records.proteins]
        if embedded_ids != expected_ids or embedded_sequences != expected_sequences:
            raise RuntimeError(
                "full-sequence embeddings do not match MegaScale proteins"
            )
        residue_embeddings = np.asarray(embeddings["residue"])
        global_embeddings = np.asarray(embeddings["global_mean"])
        embedding_provenance = str(
            embeddings.attrs["model_provenance"]
        )
        storage_dtype = residue_embeddings.dtype
    maximum_length = residue_embeddings.shape[1]
    lengths = np.asarray(
        [len(record.sequence) for record in records.proteins],
        dtype=np.int32,
    )
    sequence_mask = (
        np.arange(maximum_length)[None, :] < lengths[:, None]
    )
    sequence_tokens = np.zeros(
        (len(records.proteins), maximum_length), dtype=np.int8
    )
    for index, record in enumerate(records.proteins):
        sequence_tokens[index, : len(record.sequence)] = [
            _amino_acid_index(amino_acid) for amino_acid in record.sequence
        ]

    coordinates = np.zeros(
        (len(records.proteins), maximum_length, 4, 3), dtype=np.float32
    )
    structure_mask = np.zeros(
        (len(records.proteins), maximum_length), dtype=bool
    )
    structure_chain: list[str] = [""] * len(records.proteins)
    with TemporaryDirectory(dir=Path(temp_root).resolve()) as temporary:
        pdb_files = _extract_pdb_archive(
            structure_archive, Path(temporary) / "pdb"
        )
        for index, record in enumerate(records.proteins):
            pdb_path = pdb_files.get(record.protein_id)
            if pdb_path is None:
                raise FileNotFoundError(
                    f"structure archive is missing {record.protein_id}.pdb"
                )
            backbone, usable, chain = _extract_backbone(
                proteinmpnn_module, pdb_path, record.sequence
            )
            length = len(record.sequence)
            coordinates[index, :length] = backbone
            structure_mask[index, :length] = usable
            structure_chain[index] = chain

    source_hashes = {
        str(path.resolve()): file_sha256(path.resolve())
        for path in records.source_paths
    }
    partial = output.with_suffix(output.suffix + ".partial")
    partial.unlink(missing_ok=True)
    with h5py.File(partial, "w", libver="latest") as handle:
        handle.attrs["schema"] = FULL_STRUCTURE_DATA_SCHEMA
        handle.attrs["amino_acid_order"] = AMINO_ACIDS
        handle.attrs["negative_target_means_stabilizing"] = True
        handle.attrs["embedding_source"] = str(embedding_path)
        handle.attrs["embedding_source_sha256"] = file_sha256(embedding_path)
        handle.attrs["embedding_provenance"] = embedding_provenance
        handle.attrs["structure_archive"] = str(structure_archive)
        handle.attrs["structure_archive_sha256"] = file_sha256(
            structure_archive
        )
        handle.attrs["proteinmpnn_provenance"] = (
            proteinmpnn_provenance.canonical_json()
        )
        handle.attrs["source_sha256"] = json.dumps(
            source_hashes, sort_keys=True, separators=(",", ":")
        )
        handle.attrs["split_policy"] = json.dumps(
            {
                "seed": seed,
                "outer_value": -1,
                "outer_fraction": outer_fraction,
                "development_folds": folds,
                "assignment_uses_targets": False,
                "clustering": clustering,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        proteins = handle.create_group("proteins")
        _string_dataset(
            proteins,
            "protein_id",
            [record.protein_id for record in records.proteins],
        )
        _string_dataset(
            proteins,
            "sequence",
            [record.sequence for record in records.proteins],
        )
        _string_dataset(
            proteins,
            "family_cluster",
            [families[record.protein_id] for record in records.proteins],
        )
        _string_dataset(proteins, "structure_chain", structure_chain)
        proteins.create_dataset("length", data=lengths)
        proteins.create_dataset(
            "split",
            data=np.asarray(
                [splits[record.protein_id] for record in records.proteins],
                dtype=np.int8,
            ),
        )
        proteins.create_dataset(
            "esm_residue",
            data=residue_embeddings,
            chunks=(1, maximum_length, residue_embeddings.shape[-1]),
            compression="lzf",
        )
        proteins.create_dataset(
            "esm_global",
            data=global_embeddings,
            chunks=(1, global_embeddings.shape[-1]),
            compression="lzf",
        )
        proteins.create_dataset(
            "coordinates",
            data=coordinates,
            chunks=(1, maximum_length, 4, 3),
            compression="lzf",
        )
        proteins.create_dataset(
            "sequence_tokens", data=sequence_tokens, compression="lzf"
        )
        proteins.create_dataset(
            "sequence_mask", data=sequence_mask, compression="lzf"
        )
        proteins.create_dataset(
            "structure_mask", data=structure_mask, compression="lzf"
        )
        _write_rows(
            handle.create_group("singles"),
            records.singles,
            protein_index,
        )
        _write_rows(
            handle.create_group("doubles"),
            records.doubles,
            protein_index,
        )
        handle.flush()
    partial.replace(output)
    split_values = np.asarray(
        [splits[record.protein_id] for record in records.proteins]
    )
    single_proteins = {
        row.protein_id for row in records.singles
    }
    return {
        "schema": FULL_STRUCTURE_DATA_SCHEMA,
        "path": str(output),
        "sha256": file_sha256(output),
        "proteins": len(records.proteins),
        "single_proteins": len(single_proteins),
        "single_rows": len(records.singles),
        "double_rows": len(records.doubles),
        "families": len(set(families.values())),
        "outer_proteins": int(np.sum(split_values == -1)),
        "development_proteins": int(np.sum(split_values >= 0)),
        "embedding_dimension": int(residue_embeddings.shape[-1]),
        "storage_dtype": str(storage_dtype),
        "clustering": clustering,
    }
