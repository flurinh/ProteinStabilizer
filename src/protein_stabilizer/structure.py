"""Pinned ProteinMPNN structure encodings for row-aligned stability features."""

from __future__ import annotations

import importlib
import json
import subprocess
import sys
import tarfile
from dataclasses import asdict, dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Sequence

import h5py
import numpy as np
import pandas as pd
import torch
from Bio.Align import PairwiseAligner

from .embeddings import file_sha256


PROTEINMPNN_COMMIT = "8907e6671bfbfc92303b5f79c4b5e6ce47cdef57"
PROTEINMPNN_MODEL_NAME = "v_48_020"
STRUCTURE_FEATURE_SCHEMA = "protein-stabilizer.proteinmpnn-residue-features.v1"


@dataclass(frozen=True)
class ProteinMPNNProvenance:
    repository: str
    commit: str
    model_name: str
    checkpoint_path: str
    checkpoint_sha256: str
    source_sha256: str
    hidden_dimension: int
    residue_policy: str
    storage_dtype: str

    def canonical_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))


def _aligned_structure_indices(
    target_sequence: str,
    structure_sequence: str,
) -> tuple[np.ndarray, dict[str, object]]:
    """Map a possibly gapped/truncated structure sequence onto a canonical one."""

    target = str(target_sequence).upper()
    gapped = str(structure_sequence).upper()
    observed_positions = np.flatnonzero(
        np.asarray(list(gapped), dtype="U1") != "-"
    )
    observed = "".join(gapped[index] for index in observed_positions)
    if not target or not observed:
        raise ValueError("structure alignment requires non-empty sequences")
    aligner = PairwiseAligner()
    aligner.mode = "global"
    aligner.match_score = 2.0
    aligner.mismatch_score = -4.0
    aligner.open_gap_score = -8.0
    aligner.extend_gap_score = -0.5
    alignment = aligner.align(target, observed)[0]
    record_to_target = np.full(len(gapped), -1, dtype=np.int32)
    matches = 0
    mapped = 0
    for target_block, observed_block in zip(
        alignment.aligned[0], alignment.aligned[1], strict=True
    ):
        target_start, target_stop = (int(value) for value in target_block)
        observed_start, observed_stop = (
            int(value) for value in observed_block
        )
        if target_stop - target_start != observed_stop - observed_start:
            raise RuntimeError("aligned structure blocks have unequal lengths")
        for target_index, observed_index in zip(
            range(target_start, target_stop),
            range(observed_start, observed_stop),
            strict=True,
        ):
            record_index = int(observed_positions[observed_index])
            record_to_target[record_index] = target_index
            mapped += 1
            matches += int(target[target_index] == observed[observed_index])
    identity = matches / mapped if mapped else 0.0
    observed_coverage = mapped / len(observed)
    if identity < 0.90 or observed_coverage < 0.90:
        raise ValueError(
            "structure-to-canonical alignment is not reliable: "
            f"identity={identity:.3f}, observed_coverage={observed_coverage:.3f}"
        )
    return record_to_target, {
        "target_length": len(target),
        "structure_span_length": len(gapped),
        "observed_residues": len(observed),
        "mapped_residues": mapped,
        "identity": identity,
        "observed_coverage": observed_coverage,
        "target_coverage": mapped / len(target),
    }


class ProteinMPNNBackboneEmbedder:
    """Extract frozen final encoder residue vectors from official ProteinMPNN."""

    def __init__(
        self,
        repository: Path,
        *,
        model_name: str = PROTEINMPNN_MODEL_NAME,
        device: str = "cuda",
        storage_dtype: str = "float32",
    ) -> None:
        self.repository = Path(repository).resolve()
        source = self.repository / "protein_mpnn_utils.py"
        checkpoint_path = (
            self.repository / "vanilla_model_weights" / f"{model_name}.pt"
        )
        if not source.is_file() or not checkpoint_path.is_file():
            raise FileNotFoundError(
                "ProteinMPNN repository must contain protein_mpnn_utils.py and "
                f"vanilla_model_weights/{model_name}.pt"
            )
        commit = subprocess.run(
            ["git", "-C", str(self.repository), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        if commit != PROTEINMPNN_COMMIT:
            raise RuntimeError(
                f"ProteinMPNN commit mismatch: expected {PROTEINMPNN_COMMIT}, got {commit}"
            )
        repository_string = str(self.repository)
        if repository_string not in sys.path:
            sys.path.insert(0, repository_string)
        self.module = importlib.import_module("protein_mpnn_utils")
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        if self.device.type == "cuda":
            torch.set_float32_matmul_precision("highest")
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
        if storage_dtype not in {"float16", "float32"}:
            raise ValueError("ProteinMPNN storage dtype must be float16 or float32")
        self.storage_dtype = np.dtype(storage_dtype)
        checkpoint = torch.load(
            checkpoint_path, map_location=self.device, weights_only=False
        )
        hidden_dimension = 128
        model = self.module.ProteinMPNN(
            ca_only=False,
            num_letters=21,
            node_features=hidden_dimension,
            edge_features=hidden_dimension,
            hidden_dim=hidden_dimension,
            num_encoder_layers=3,
            num_decoder_layers=3,
            augment_eps=0.0,
            k_neighbors=checkpoint["num_edges"],
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        self.model = model.to(self.device).eval()
        self.provenance = ProteinMPNNProvenance(
            repository="https://github.com/dauparas/ProteinMPNN",
            commit=commit,
            model_name=model_name,
            checkpoint_path=str(checkpoint_path),
            checkpoint_sha256=file_sha256(checkpoint_path),
            source_sha256=file_sha256(source),
            hidden_dimension=hidden_dimension,
            residue_policy=(
                "final-unmasked-encoder-residue-vector-target-chain-first-v1"
            ),
            storage_dtype=storage_dtype,
        )

    def encode(
        self,
        pdb_path: Path,
        target_sequence: str,
        *,
        residue_mask: np.ndarray | None = None,
    ) -> np.ndarray:
        """Return one structure-only vector for every target-sequence residue.

        ``residue_mask`` removes unreliable target residues from the
        ProteinMPNN encoder graph and zeros their returned vectors.
        """

        records = self.module.parse_PDB(str(pdb_path), ca_only=False)
        if len(records) != 1:
            raise ValueError(f"expected one parsed structure in {pdb_path}")
        record = records[0]
        chains = {
            key.removeprefix("seq_chain_"): str(value)
            for key, value in record.items()
            if key.startswith("seq_chain_")
        }
        matching = [chain for chain, sequence in chains.items() if sequence == target_sequence]
        if len(matching) != 1:
            raise ValueError(
                f"{pdb_path.name} has {len(matching)} exact chains for target sequence"
            )
        target_chain = matching[0]
        visible = sorted(chain for chain in chains if chain != target_chain)
        chain_dict = {record["name"]: ([target_chain], visible)}
        values = self.module.tied_featurize(
            [record],
            self.device,
            chain_dict,
            None,
            None,
            None,
            None,
            None,
            ca_only=False,
        )
        (
            coordinates,
            _,
            mask,
            _,
            _,
            chain_encoding,
            _,
            _,
            _,
            _,
            _,
            _,
            residue_index,
            _,
            _,
            _,
            _,
            _,
            _,
            _,
        ) = values
        length = len(target_sequence)
        if not torch.all(mask[0, :length] > 0):
            missing = torch.nonzero(mask[0, :length] <= 0).flatten().tolist()
            raise ValueError(
                f"{pdb_path.name} has missing target coordinates at {missing}"
            )
        active_mask: np.ndarray | None = None
        if residue_mask is not None:
            active_mask = np.asarray(residue_mask, dtype=bool)
            if active_mask.shape != (length,):
                raise ValueError(
                    "ProteinMPNN residue mask must have one value per "
                    "target-sequence residue"
                )
            mask = mask.clone()
            mask[0, :length] *= torch.from_numpy(active_mask).to(
                device=mask.device,
                dtype=mask.dtype,
            )
        with torch.inference_mode():
            edges, edge_index = self.model.features(
                coordinates, mask, residue_index, chain_encoding
            )
            residue = torch.zeros(
                (edges.shape[0], edges.shape[1], edges.shape[-1]),
                device=edges.device,
            )
            edge = self.model.W_e(edges)
            attend = self.module.gather_nodes(
                mask.unsqueeze(-1), edge_index
            ).squeeze(-1)
            attend = mask.unsqueeze(-1) * attend
            for layer in self.model.encoder_layers:
                residue, edge = layer(
                    residue, edge, edge_index, mask, attend
                )
        result = residue[0, :length].float().cpu().numpy()
        if active_mask is not None:
            result[~active_mask] = 0.0
        return result

    def backbone_coordinates(
        self,
        pdb_path: Path,
        target_sequence: str,
    ) -> tuple[np.ndarray, np.ndarray, str]:
        """Return exact-chain N/CA/C/O coordinates in sequence order."""

        records = self.module.parse_PDB(str(pdb_path), ca_only=False)
        if len(records) != 1:
            raise ValueError(f"expected one parsed structure in {pdb_path}")
        record = records[0]
        chains = {
            key.removeprefix("seq_chain_"): str(value)
            for key, value in record.items()
            if key.startswith("seq_chain_")
        }
        matching = [
            chain
            for chain, sequence in chains.items()
            if sequence == target_sequence
        ]
        if len(matching) != 1:
            raise ValueError(
                f"{Path(pdb_path).name} has {len(matching)} exact chains "
                "for target sequence"
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
        expected = (len(target_sequence), 4, 3)
        if coordinates.shape != expected:
            raise RuntimeError(
                f"{Path(pdb_path).name} backbone shape is "
                f"{coordinates.shape}, expected {expected}"
            )
        mask = np.isfinite(coordinates).all(axis=(1, 2))
        return (
            np.nan_to_num(
                coordinates,
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            ),
            mask,
            chain,
        )

    def encode_aligned(
        self,
        pdb_path: Path,
        target_sequence: str,
    ) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
        """Map a truncated or internally unresolved receptor chain safely.

        The returned vector array follows canonical sequence numbering. Only
        residues with coordinates and a high-confidence sequence alignment are
        marked in the boolean mask.
        """

        records = self.module.parse_PDB(str(pdb_path), ca_only=False)
        if len(records) != 1:
            raise ValueError(f"expected one parsed structure in {pdb_path}")
        record = records[0]
        chains = {
            key.removeprefix("seq_chain_"): str(value)
            for key, value in record.items()
            if key.startswith("seq_chain_")
        }
        candidates: list[
            tuple[float, int, str, np.ndarray, dict[str, object]]
        ] = []
        for chain, sequence in chains.items():
            try:
                mapping, diagnostics = _aligned_structure_indices(
                    target_sequence, sequence
                )
            except ValueError:
                continue
            candidates.append(
                (
                    float(diagnostics["identity"]),
                    int(diagnostics["mapped_residues"]),
                    chain,
                    mapping,
                    diagnostics,
                )
            )
        if not candidates:
            raise ValueError(
                f"{pdb_path.name} has no reliably aligned target chain"
            )
        candidates.sort(key=lambda value: (value[0], value[1]), reverse=True)
        _, _, target_chain, mapping, diagnostics = candidates[0]
        visible = sorted(chain for chain in chains if chain != target_chain)
        chain_dict = {record["name"]: ([target_chain], visible)}
        values = self.module.tied_featurize(
            [record],
            self.device,
            chain_dict,
            None,
            None,
            None,
            None,
            None,
            ca_only=False,
        )
        (
            coordinates,
            _,
            mask,
            _,
            _,
            chain_encoding,
            _,
            _,
            _,
            _,
            _,
            _,
            residue_index,
            _,
            _,
            _,
            _,
            _,
            _,
            _,
        ) = values
        with torch.inference_mode():
            edges, edge_index = self.model.features(
                coordinates, mask, residue_index, chain_encoding
            )
            residue = torch.zeros(
                (edges.shape[0], edges.shape[1], edges.shape[-1]),
                device=edges.device,
            )
            edge = self.model.W_e(edges)
            attend = self.module.gather_nodes(
                mask.unsqueeze(-1), edge_index
            ).squeeze(-1)
            attend = mask.unsqueeze(-1) * attend
            for layer in self.model.encoder_layers:
                residue, edge = layer(
                    residue, edge, edge_index, mask, attend
                )
        canonical = np.zeros(
            (len(target_sequence), self.provenance.hidden_dimension),
            dtype=np.float32,
        )
        canonical_mask = np.zeros(len(target_sequence), dtype=bool)
        chain_length = len(chains[target_chain])
        encoded = residue[0, :chain_length].float().cpu().numpy()
        coordinate_mask = mask[0, :chain_length].bool().cpu().numpy()
        for record_index, target_index in enumerate(mapping):
            if target_index < 0 or not coordinate_mask[record_index]:
                continue
            if canonical_mask[target_index]:
                raise RuntimeError(
                    f"{pdb_path.name} maps two residues to canonical "
                    f"position {target_index + 1}"
                )
            canonical[target_index] = encoded[record_index]
            canonical_mask[target_index] = True
        diagnostics = {
            **diagnostics,
            "chain": target_chain,
            "coordinate_residues": int(canonical_mask.sum()),
            "pdb_path": str(Path(pdb_path).resolve()),
            "pdb_sha256": file_sha256(Path(pdb_path)),
        }
        return (
            canonical.astype(self.storage_dtype),
            canonical_mask,
            diagnostics,
        )


def _string_dataset(handle: h5py.File, name: str, values: Sequence[str]) -> None:
    handle.create_dataset(
        name,
        data=np.asarray(values, dtype=object),
        dtype=h5py.string_dtype(encoding="utf-8"),
    )


def _extract_pdb_archive(archive_path: Path, destination: Path) -> dict[str, Path]:
    destination.mkdir(parents=True, exist_ok=True)
    extracted: dict[str, Path] = {}
    with tarfile.open(archive_path, mode="r:gz") as archive:
        for member in archive.getmembers():
            member_path = Path(member.name)
            if (
                not member.isfile()
                or member_path.suffix.lower() != ".pdb"
                or member_path.is_absolute()
                or ".." in member_path.parts
            ):
                continue
            output = destination / member_path.name
            source = archive.extractfile(member)
            if source is None:
                raise RuntimeError(f"could not read {member.name} from structure archive")
            output.write_bytes(source.read())
            key = output.stem.lower()
            if key in extracted:
                raise RuntimeError(f"duplicate case-insensitive PDB identifier {key}")
            extracted[key] = output
    return extracted


def build_megascale_structure_features(
    feature_dir: Path,
    structure_archive: Path,
    output_dir: Path,
    proteinmpnn_repository: Path,
    *,
    temp_root: Path = Path("/data/fast/tmp/protein-stabilizer/v2"),
    device: str = "cuda",
    storage_dtype: str = "float32",
) -> dict[str, object]:
    """Create row-aligned frozen ProteinMPNN features for MegaScale singles."""

    feature_dir = Path(feature_dir)
    structure_archive = Path(structure_archive)
    output_dir = Path(output_dir)
    temp_root = Path(temp_root).resolve()
    if not temp_root.is_relative_to(Path("/data/fast")):
        raise ValueError("bulk structure temporary root must be below /data/fast")
    temp_root.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    embedder = ProteinMPNNBackboneEmbedder(
        proteinmpnn_repository,
        device=device,
        storage_dtype=storage_dtype,
    )
    archive_sha256 = file_sha256(structure_archive)
    manifest: dict[str, object] = {
        "schema": STRUCTURE_FEATURE_SCHEMA,
        "structure_archive": {
            "path": str(structure_archive),
            "sha256": archive_sha256,
        },
        "proteinmpnn": asdict(embedder.provenance),
        "outputs": {},
    }
    with TemporaryDirectory(dir=temp_root) as temporary:
        pdb_files = _extract_pdb_archive(
            structure_archive, Path(temporary) / "pdb"
        )
        encoded: dict[str, np.ndarray] = {}
        for split_name, filename in (
            ("train", "single_train.h5"),
            ("test", "single_test.h5"),
        ):
            feature_path = feature_dir / filename
            with h5py.File(feature_path, "r") as handle:
                protein_id = np.asarray(handle["protein_id"].asstr()[:])
                position = np.asarray(handle["position"], dtype=np.int32)
                mutation = np.asarray(handle["mutation"].asstr()[:])
                source_path = Path(str(handle.attrs["source"]))
                source_sha256 = str(handle.attrs["source_sha256"])
                embedding_provenance = str(handle.attrs["embedding_provenance"])
            source = pd.read_csv(source_path, usecols=["pdb_id", "wt_seq"])
            sequence_by_protein = {
                str(row.pdb_id): str(row.wt_seq)
                for row in source.drop_duplicates(["pdb_id", "wt_seq"]).itertuples(
                    index=False
                )
            }
            if len(sequence_by_protein) != source["pdb_id"].nunique():
                raise RuntimeError(f"{source_path} has inconsistent WT sequences")
            for protein in sorted(set(protein_id.tolist())):
                if protein not in encoded:
                    pdb_path = pdb_files.get(protein.lower())
                    if pdb_path is None:
                        raise FileNotFoundError(
                            f"structure archive is missing {protein}.pdb"
                        )
                    encoded[protein] = embedder.encode(
                        pdb_path, sequence_by_protein[protein]
                    )
            row_vectors = np.stack(
                [
                    encoded[str(protein)][int(site) - 1]
                    for protein, site in zip(protein_id, position, strict=True)
                ]
            ).astype(embedder.storage_dtype)
            output_path = output_dir / f"structure_single_{split_name}.h5"
            partial = output_path.with_suffix(".h5.partial")
            partial.unlink(missing_ok=True)
            with h5py.File(partial, "w", libver="latest") as handle:
                handle.attrs["schema"] = STRUCTURE_FEATURE_SCHEMA
                handle.attrs["row_feature_source"] = str(feature_path)
                handle.attrs["row_feature_source_sha256"] = file_sha256(feature_path)
                handle.attrs["source_sha256"] = source_sha256
                handle.attrs["embedding_provenance"] = embedding_provenance
                handle.attrs["structure_archive_sha256"] = archive_sha256
                handle.attrs["proteinmpnn_provenance"] = (
                    embedder.provenance.canonical_json()
                )
                handle.create_dataset(
                    "proteinmpnn",
                    data=row_vectors,
                    chunks=(min(4096, len(row_vectors)), row_vectors.shape[1]),
                )
                handle.create_dataset("position", data=position)
                _string_dataset(handle, "protein_id", protein_id.tolist())
                _string_dataset(handle, "mutation", mutation.tolist())
            partial.replace(output_path)
            manifest["outputs"][split_name] = {
                "path": str(output_path),
                "sha256": file_sha256(output_path),
                "rows": int(len(row_vectors)),
                "proteins": int(len(set(protein_id.tolist()))),
                "dimension": int(row_vectors.shape[1]),
                "source_sha256": source_sha256,
            }
    manifest_path = output_dir / "structure_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    return manifest


def build_aligned_structure_features(
    row_feature_path: Path,
    structure_map_path: Path,
    output_path: Path,
    proteinmpnn_repository: Path,
    *,
    device: str = "cuda",
    storage_dtype: str = "float32",
) -> dict[str, object]:
    """Build row-aligned features from truncated or unresolved structures.

    ``structure_map_path`` is a JSON object mapping row ``protein_id`` values
    to PDB files. Proteins omitted from the map remain explicitly missing.
    """

    row_feature_path = Path(row_feature_path).resolve()
    structure_map_path = Path(structure_map_path).resolve()
    output_path = Path(output_path).resolve()
    raw_map = json.loads(structure_map_path.read_text(encoding="utf-8"))
    if not isinstance(raw_map, dict) or not all(
        isinstance(key, str) and isinstance(value, str)
        for key, value in raw_map.items()
    ):
        raise ValueError("structure map must be a JSON object of protein=PDB")
    structure_map = {
        protein: (
            Path(path).expanduser().resolve()
            if Path(path).expanduser().is_absolute()
            else (structure_map_path.parent / path).resolve()
        )
        for protein, path in raw_map.items()
    }
    missing_files = [
        str(path) for path in structure_map.values() if not path.is_file()
    ]
    if missing_files:
        raise FileNotFoundError(
            f"structure map references missing files: {missing_files}"
        )
    with h5py.File(row_feature_path, "r") as handle:
        protein_id = np.asarray(handle["protein_id"].asstr()[:])
        mutation = np.asarray(handle["mutation"].asstr()[:])
        position = np.asarray(handle["position"], dtype=np.int32)
        source_path = Path(str(handle.attrs["source"])).resolve()
        source_sha256 = str(handle.attrs["source_sha256"])
        hierarchy_provenance = str(handle.attrs["hierarchy_provenance"])
    source = pd.read_csv(source_path)
    sequence_column = next(
        (
            column
            for column in ("wt_sequence", "sequence", "wt_seq")
            if column in source.columns
        ),
        None,
    )
    if sequence_column is None or "protein_id" not in source.columns:
        raise ValueError(
            f"{source_path} lacks protein_id and a supported sequence column"
        )
    sequences: dict[str, str] = {}
    for protein, values in source.groupby("protein_id", sort=False):
        unique = {
            str(value)
            for value in values[sequence_column].dropna().tolist()
        }
        if len(unique) != 1:
            raise RuntimeError(
                f"{source_path} has inconsistent sequences for {protein}"
            )
        sequences[str(protein)] = unique.pop()
    embedder = ProteinMPNNBackboneEmbedder(
        proteinmpnn_repository,
        device=device,
        storage_dtype=storage_dtype,
    )
    encoded: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    alignments: dict[str, object] = {}
    for protein in sorted(set(protein_id.tolist()) & set(structure_map)):
        if protein not in sequences:
            raise KeyError(f"no canonical sequence found for {protein}")
        vectors, mask, diagnostics = embedder.encode_aligned(
            structure_map[protein], sequences[protein]
        )
        encoded[protein] = (vectors, mask)
        alignments[protein] = diagnostics
    row_vectors = np.zeros(
        (len(protein_id), embedder.provenance.hidden_dimension),
        dtype=embedder.storage_dtype,
    )
    row_mask = np.zeros(len(protein_id), dtype=bool)
    for index, (protein, site) in enumerate(
        zip(protein_id, position, strict=True)
    ):
        value = encoded.get(str(protein))
        if value is None:
            continue
        vectors, mask = value
        canonical_index = int(site) - 1
        if canonical_index < 0 or canonical_index >= len(vectors):
            raise ValueError(
                f"{protein} mutation position {site} is outside its sequence"
            )
        if not mask[canonical_index]:
            continue
        row_vectors[index] = vectors[canonical_index]
        row_mask[index] = True
    output_path.parent.mkdir(parents=True, exist_ok=True)
    partial = output_path.with_suffix(output_path.suffix + ".partial")
    partial.unlink(missing_ok=True)
    with h5py.File(partial, "w", libver="latest") as handle:
        handle.attrs["schema"] = STRUCTURE_FEATURE_SCHEMA
        handle.attrs["row_feature_source"] = str(row_feature_path)
        handle.attrs["row_feature_source_sha256"] = file_sha256(
            row_feature_path
        )
        handle.attrs["source_sha256"] = source_sha256
        handle.attrs["embedding_provenance"] = hierarchy_provenance
        handle.attrs["structure_map"] = str(structure_map_path)
        handle.attrs["structure_map_sha256"] = file_sha256(
            structure_map_path
        )
        handle.attrs["structure_archive_sha256"] = file_sha256(
            structure_map_path
        )
        handle.attrs["proteinmpnn_provenance"] = (
            embedder.provenance.canonical_json()
        )
        handle.attrs["alignment_provenance"] = json.dumps(
            alignments, sort_keys=True, separators=(",", ":")
        )
        handle.create_dataset("proteinmpnn", data=row_vectors)
        handle.create_dataset("structure_mask", data=row_mask)
        handle.create_dataset("position", data=position)
        _string_dataset(handle, "protein_id", protein_id.tolist())
        _string_dataset(handle, "mutation", mutation.tolist())
    partial.replace(output_path)
    result = {
        "schema": STRUCTURE_FEATURE_SCHEMA,
        "path": str(output_path),
        "sha256": file_sha256(output_path),
        "rows": int(len(row_vectors)),
        "structure_rows": int(row_mask.sum()),
        "proteins": int(len(encoded)),
        "proteinmpnn": asdict(embedder.provenance),
        "structure_map": {
            "path": str(structure_map_path),
            "sha256": file_sha256(structure_map_path),
            "structures": {
                protein: {
                    "path": str(path),
                    "sha256": file_sha256(path),
                }
                for protein, path in sorted(structure_map.items())
            },
        },
        "alignments": alignments,
    }
    report_path = output_path.with_suffix(".json")
    report_path.write_text(
        json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
    )
    result["report_path"] = str(report_path)
    return result
