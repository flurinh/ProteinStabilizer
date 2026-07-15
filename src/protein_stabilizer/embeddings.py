"""Resumable ESM-C residue embedding cache."""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Iterator, Sequence

import h5py
import numpy as np
import torch

from .data import EmbeddingRequest, normalize_sequence, sequence_hash


CACHE_SCHEMA = "protein-stabilizer.esmc-residue-cache.v1"


def file_sha256(path: Path, chunk_size: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class ESMCProvenance:
    model_name: str
    embedding_dimension: int
    package_version: str
    checkpoint_path: str
    checkpoint_sha256: str
    inference_dtype: str = "bfloat16"
    storage_dtype: str = "float16"
    residue_policy: str = "final-layer-contextual-residue-bos-offset-v1"

    def canonical_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))


class ESMCEmbedder:
    """Thin local ESM-C adapter that returns final-layer residue vectors."""

    def __init__(self, model_name: str = "esmc_600m", device: str = "cuda") -> None:
        try:
            import esm
            from esm.models.esmc import ESMC
            from esm.pretrained import data_root
        except ImportError as exc:
            raise RuntimeError(
                "ESM-C dependencies are missing; install with `pip install -e '.[esmc]'`"
            ) from exc
        if model_name != "esmc_600m":
            raise ValueError("the validated V1 backend is esmc_600m")
        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")
        self.device = torch.device(device)
        checkpoint = (
            Path(data_root("esmc-600"))
            / "data/weights/esmc_600m_2024_12_v0.pth"
        ).resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        self.model = ESMC.from_pretrained(model_name, device=self.device).eval()
        self.model_name = model_name
        self.dimension = int(self.model.embed.embedding_dim)
        self.provenance = ESMCProvenance(
            model_name=model_name,
            embedding_dimension=self.dimension,
            package_version=str(getattr(esm, "__version__", "unknown")),
            checkpoint_path=str(checkpoint),
            checkpoint_sha256=file_sha256(checkpoint),
        )

    def encode(self, requests: Sequence[EmbeddingRequest]) -> list[np.ndarray]:
        sequences = [normalize_sequence(request.sequence) for request in requests]
        encoded = self.model.tokenizer(
            sequences,
            add_special_tokens=True,
            padding=True,
            truncation=False,
            return_tensors="pt",
        )
        tokens = encoded["input_ids"].to(self.device)
        attention = encoded["attention_mask"]
        expected_lengths = torch.tensor([len(sequence) + 2 for sequence in sequences])
        if not torch.equal(attention.sum(dim=1).cpu(), expected_lengths):
            raise RuntimeError("ESM-C tokenizer truncated or misaligned a sequence")
        with torch.inference_mode():
            output = self.model(sequence_tokens=tokens)
        embeddings = output.embeddings
        results: list[np.ndarray] = []
        for row, request in enumerate(requests):
            selected = embeddings[row, list(request.positions)]
            if selected.shape != (len(request.positions), self.dimension):
                raise RuntimeError("unexpected ESM-C residue embedding shape")
            results.append(selected.float().cpu().numpy().astype(np.float16))
        del output, embeddings, tokens
        return results


def request_manifest_sha256(requests: Sequence[EmbeddingRequest]) -> str:
    digest = hashlib.sha256()
    for request in sorted(requests, key=lambda value: value.sequence_hash):
        digest.update(request.sequence_hash.encode("ascii"))
        digest.update(b":")
        digest.update(",".join(str(position) for position in request.positions).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def token_batches(
    requests: Sequence[EmbeddingRequest],
    *,
    max_tokens: int,
    max_batch_size: int,
) -> Iterator[list[EmbeddingRequest]]:
    """Length-bucket requests while bounding padded residues per forward pass."""

    ordered = sorted(requests, key=lambda value: (len(value.sequence), value.sequence_hash))
    batch: list[EmbeddingRequest] = []
    longest = 0
    for request in ordered:
        candidate_longest = max(longest, len(request.sequence) + 2)
        candidate_size = len(batch) + 1
        if batch and (
            candidate_size > max_batch_size
            or candidate_longest * candidate_size > max_tokens
        ):
            yield batch
            batch = []
            longest = 0
        batch.append(request)
        longest = max(longest, len(request.sequence) + 2)
    if batch:
        yield batch


class ResidueEmbeddingWriter:
    """Append-only HDF5 cache with rollback markers for interrupted batches."""

    def __init__(self, path: Path, provenance: ESMCProvenance) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = h5py.File(self.path, "a", libver="latest")
        self.provenance = provenance
        self._initialize()
        self._rollback_pending()
        self._load_indices()

    def _initialize(self) -> None:
        h = self.handle
        if "schema" in h.attrs:
            if h.attrs["schema"] != CACHE_SCHEMA:
                raise RuntimeError("embedding cache schema mismatch")
            if h.attrs["model_provenance"] != self.provenance.canonical_json():
                raise RuntimeError("embedding cache model provenance mismatch")
            return
        h.attrs["schema"] = CACHE_SCHEMA
        h.attrs["model_provenance"] = self.provenance.canonical_json()
        h.attrs["created_unix"] = time.time()
        h.create_dataset("sequence_hashes", shape=(0,), maxshape=(None,), dtype="S64")
        h.create_dataset(
            "sequences",
            shape=(0,),
            maxshape=(None,),
            dtype=h5py.string_dtype(encoding="ascii"),
        )
        h.create_dataset("site_sequence_index", shape=(0,), maxshape=(None,), dtype="i8")
        h.create_dataset("site_position", shape=(0,), maxshape=(None,), dtype="i4")
        h.create_dataset(
            "site_embeddings",
            shape=(0, self.provenance.embedding_dimension),
            maxshape=(None, self.provenance.embedding_dimension),
            chunks=(min(1024, max(1, 1024)), self.provenance.embedding_dimension),
            dtype="f2",
        )
        h.flush()

    def _rollback_pending(self) -> None:
        h = self.handle
        if "pending_sequence_start" not in h.attrs:
            return
        sequence_start = int(h.attrs["pending_sequence_start"])
        site_start = int(h.attrs["pending_site_start"])
        h["sequence_hashes"].resize((sequence_start,))
        h["sequences"].resize((sequence_start,))
        h["site_sequence_index"].resize((site_start,))
        h["site_position"].resize((site_start,))
        h["site_embeddings"].resize((site_start, self.provenance.embedding_dimension))
        del h.attrs["pending_sequence_start"]
        del h.attrs["pending_site_start"]
        h.flush()

    def _load_indices(self) -> None:
        hashes = [value.decode("ascii") for value in self.handle["sequence_hashes"][:]]
        if len(hashes) != len(set(hashes)):
            raise RuntimeError("embedding cache contains duplicate sequence hashes")
        self.sequence_index = {digest: index for index, digest in enumerate(hashes)}
        sequence_indices = self.handle["site_sequence_index"][:]
        positions = self.handle["site_position"][:]
        self.site_index: dict[tuple[str, int], int] = {}
        for site_index, (sequence_index, position) in enumerate(
            zip(sequence_indices, positions, strict=True)
        ):
            digest = hashes[int(sequence_index)]
            key = (digest, int(position))
            if key in self.site_index:
                raise RuntimeError(f"embedding cache contains duplicate site {key}")
            self.site_index[key] = site_index

    def missing_requests(self, requests: Iterable[EmbeddingRequest]) -> list[EmbeddingRequest]:
        missing: list[EmbeddingRequest] = []
        for request in requests:
            digest = request.sequence_hash
            positions = tuple(
                position
                for position in request.positions
                if (digest, position) not in self.site_index
            )
            if positions:
                missing.append(EmbeddingRequest(request.sequence, positions))
        return missing

    def append(
        self,
        requests: Sequence[EmbeddingRequest],
        embeddings: Sequence[np.ndarray],
    ) -> None:
        if len(requests) != len(embeddings):
            raise ValueError("request and embedding counts differ")
        h = self.handle
        old_sequence_count = len(h["sequence_hashes"])
        old_site_count = len(h["site_position"])
        h.attrs["pending_sequence_start"] = old_sequence_count
        h.attrs["pending_site_start"] = old_site_count
        h.flush()

        new_requests = [
            request for request in requests if request.sequence_hash not in self.sequence_index
        ]
        new_sequence_count = old_sequence_count + len(new_requests)
        h["sequence_hashes"].resize((new_sequence_count,))
        h["sequences"].resize((new_sequence_count,))
        for offset, request in enumerate(new_requests):
            index = old_sequence_count + offset
            digest = request.sequence_hash
            h["sequence_hashes"][index] = digest.encode("ascii")
            h["sequences"][index] = request.sequence
            self.sequence_index[digest] = index

        site_count = sum(len(request.positions) for request in requests)
        new_site_count = old_site_count + site_count
        h["site_sequence_index"].resize((new_site_count,))
        h["site_position"].resize((new_site_count,))
        h["site_embeddings"].resize(
            (new_site_count, self.provenance.embedding_dimension)
        )
        cursor = old_site_count
        for request, matrix in zip(requests, embeddings, strict=True):
            expected = (len(request.positions), self.provenance.embedding_dimension)
            if matrix.shape != expected:
                raise ValueError(f"embedding shape {matrix.shape} does not match {expected}")
            sequence_index = self.sequence_index[request.sequence_hash]
            stop = cursor + len(request.positions)
            h["site_sequence_index"][cursor:stop] = sequence_index
            h["site_position"][cursor:stop] = request.positions
            h["site_embeddings"][cursor:stop] = matrix
            for site_index, position in enumerate(request.positions, start=cursor):
                self.site_index[(request.sequence_hash, position)] = site_index
            cursor = stop
        del h.attrs["pending_sequence_start"]
        del h.attrs["pending_site_start"]
        h.attrs["updated_unix"] = time.time()
        h.flush()

    def close(self) -> None:
        self.handle.close()

    def __enter__(self) -> "ResidueEmbeddingWriter":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


class ResidueEmbeddingReader:
    """Read a validated residue cache and vectorize site lookups."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.handle = h5py.File(self.path, "r")
        if self.handle.attrs.get("schema") != CACHE_SCHEMA:
            raise RuntimeError("embedding cache schema mismatch")
        provenance = json.loads(self.handle.attrs["model_provenance"])
        self.dimension = int(provenance["embedding_dimension"])
        self.provenance = provenance
        hashes = [value.decode("ascii") for value in self.handle["sequence_hashes"][:]]
        sequence_indices = self.handle["site_sequence_index"][:]
        positions = self.handle["site_position"][:]
        self.site_index = {
            (hashes[int(sequence_index)], int(position)): site_index
            for site_index, (sequence_index, position) in enumerate(
                zip(sequence_indices, positions, strict=True)
            )
        }

    def vectors(self, keys: Sequence[tuple[str, int]]) -> np.ndarray:
        try:
            indices = np.asarray([self.site_index[key] for key in keys], dtype=np.int64)
        except KeyError as exc:
            raise KeyError(f"residue embedding is missing for {exc.args[0]}") from exc
        if len(indices) == 0:
            return np.empty((0, self.dimension), dtype=np.float16)
        order = np.argsort(indices)
        sorted_indices = indices[order]
        if len(np.unique(sorted_indices)) == len(sorted_indices):
            sorted_values = self.handle["site_embeddings"][sorted_indices]
        else:
            unique, inverse = np.unique(sorted_indices, return_inverse=True)
            sorted_values = self.handle["site_embeddings"][unique][inverse]
        inverse_order = np.empty_like(order)
        inverse_order[order] = np.arange(len(order))
        return np.asarray(sorted_values[inverse_order], dtype=np.float16)

    def close(self) -> None:
        self.handle.close()

    def __enter__(self) -> "ResidueEmbeddingReader":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


def build_embedding_cache(
    path: Path,
    requests: Sequence[EmbeddingRequest],
    *,
    model_name: str = "esmc_600m",
    device: str = "cuda",
    max_tokens: int = 8192,
    max_batch_size: int = 128,
) -> dict[str, object]:
    embedder = ESMCEmbedder(model_name=model_name, device=device)
    manifest_hash = request_manifest_sha256(requests)
    start = time.monotonic()
    with ResidueEmbeddingWriter(path, embedder.provenance) as writer:
        missing = writer.missing_requests(requests)
        initial_sites = len(writer.site_index)
        batches = list(
            token_batches(
                missing, max_tokens=max_tokens, max_batch_size=max_batch_size
            )
        )
        for batch_number, batch in enumerate(batches, start=1):
            writer.append(batch, embedder.encode(batch))
            if batch_number == 1 or batch_number % 50 == 0 or batch_number == len(batches):
                print(
                    f"embedded batch {batch_number}/{len(batches)}; "
                    f"cached sites={len(writer.site_index):,}",
                    flush=True,
                )
        writer.handle.attrs["request_manifest_sha256"] = manifest_hash
        writer.handle.attrs["request_count"] = len(requests)
        writer.handle.flush()
        result = {
            "cache": str(path),
            "model": model_name,
            "dimension": embedder.dimension,
            "request_manifest_sha256": manifest_hash,
            "requested_sequences": len(requests),
            "new_sites": len(writer.site_index) - initial_sites,
            "total_sites": len(writer.site_index),
            "elapsed_seconds": time.monotonic() - start,
            "provenance": asdict(embedder.provenance),
        }
    return result
