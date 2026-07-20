"""Resumable ESM-C residue embedding cache."""

from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Iterator, Sequence

import h5py
import numpy as np
import torch

from .data import AMINO_ACIDS, EmbeddingRequest, normalize_sequence, sequence_hash


CACHE_SCHEMA = "protein-stabilizer.esmc-residue-cache.v1"
HIERARCHY_CACHE_SCHEMA = "protein-stabilizer.esmc-hierarchy-cache.v1"


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


@dataclass(frozen=True)
class HierarchicalEmbedding:
    """One sequence's global state and ordered local windows."""

    global_mean: np.ndarray
    windows: np.ndarray
    window_mask: np.ndarray


class ESMCEmbedder:
    """Thin local ESM-C adapter that returns final-layer residue vectors."""

    def __init__(
        self,
        model_name: str = "esmc_600m",
        device: str = "cuda",
        *,
        storage_dtype: str = "float16",
    ) -> None:
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
        if storage_dtype not in {"float16", "float32"}:
            raise ValueError("ESM-C storage dtype must be float16 or float32")
        self.storage_dtype = np.dtype(storage_dtype)
        if self.device.type == "cuda":
            torch.set_float32_matmul_precision("highest")
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
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
            inference_dtype="float32",
            storage_dtype=storage_dtype,
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
            results.append(
                selected.float().cpu().numpy().astype(self.storage_dtype)
            )
        del output, embeddings, tokens
        return results

    def encode_hierarchy(
        self,
        requests: Sequence[EmbeddingRequest],
        *,
        window_radius: int = 4,
    ) -> list[HierarchicalEmbedding]:
        """Return global means and ordered local windows from one forward pass."""

        if window_radius < 0:
            raise ValueError("window radius must be non-negative")
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
        window_size = 2 * window_radius + 1
        results: list[HierarchicalEmbedding] = []
        for row, (request, sequence) in enumerate(zip(requests, sequences, strict=True)):
            residues = embeddings[row, 1 : len(sequence) + 1].float()
            global_mean = (
                residues.mean(dim=0).cpu().numpy().astype(self.storage_dtype)
            )
            windows = torch.zeros(
                (len(request.positions), window_size, self.dimension),
                dtype=residues.dtype,
                device=residues.device,
            )
            window_mask = torch.zeros(
                (len(request.positions), window_size),
                dtype=torch.bool,
                device=residues.device,
            )
            for site_index, position in enumerate(request.positions):
                if position < 1 or position > len(sequence):
                    raise ValueError("hierarchy position is outside the sequence")
                source_start = max(1, position - window_radius)
                source_stop = min(len(sequence), position + window_radius)
                destination_start = source_start - (position - window_radius)
                count = source_stop - source_start + 1
                windows[
                    site_index,
                    destination_start : destination_start + count,
                ] = residues[source_start - 1 : source_stop]
                window_mask[
                    site_index,
                    destination_start : destination_start + count,
                ] = True
            results.append(
                HierarchicalEmbedding(
                    global_mean=global_mean,
                    windows=windows.cpu().numpy().astype(self.storage_dtype),
                    window_mask=window_mask.cpu().numpy(),
                )
            )
        del output, embeddings, tokens
        return results

    def masked_marginal_log_probabilities(
        self,
        sequence: str,
        positions: Sequence[int],
        *,
        max_tokens: int = 8192,
        max_batch_size: int = 128,
    ) -> np.ndarray:
        """Score every canonical amino acid after masking each requested site.

        Rows follow ``positions`` and columns follow ``data.AMINO_ACIDS``.
        One WT-context forward pass is required per unique position, so all
        nineteen substitutions at a site share the same probability vector.
        """

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

        tokenizer = self.model.tokenizer
        amino_acid_ids = [
            int(tokenizer.convert_tokens_to_ids(amino_acid))
            for amino_acid in AMINO_ACIDS
        ]
        batch_size = min(
            max_batch_size,
            max(1, max_tokens // (len(normalized) + 2)),
        )
        results: list[np.ndarray] = []
        for start in range(0, len(selected), batch_size):
            batch_positions = selected[start : start + batch_size]
            encoded = tokenizer(
                [normalized] * len(batch_positions),
                add_special_tokens=True,
                padding=True,
                truncation=False,
                return_tensors="pt",
            )
            tokens = encoded["input_ids"].to(self.device)
            attention = encoded["attention_mask"]
            expected_length = len(normalized) + 2
            if not torch.all(attention.sum(dim=1) == expected_length):
                raise RuntimeError("ESM-C tokenizer truncated or misaligned a sequence")
            row_indices = torch.arange(len(batch_positions), device=self.device)
            token_positions = torch.tensor(batch_positions, device=self.device)
            tokens[row_indices, token_positions] = tokenizer.mask_token_id
            with torch.inference_mode():
                output = self.model(sequence_tokens=tokens)
                site_logits = output.sequence_logits[row_indices, token_positions]
                log_probabilities = torch.log_softmax(
                    site_logits.float(), dim=-1
                )[:, amino_acid_ids]
            results.append(log_probabilities.cpu().numpy().astype(np.float32))
            del output, site_logits, log_probabilities, tokens
        return np.concatenate(results, axis=0)


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
        # The separate 6B environment currently links HDF5 2.x while the
        # training environment links HDF5 1.14.  Pin the on-disk format so
        # caches remain readable across both validated runtimes.
        self.handle = h5py.File(self.path, "a", libver="v110")
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


class HierarchyEmbeddingWriter:
    """Resumable sequence/global and ordered-window embedding cache."""

    def __init__(
        self,
        path: Path,
        provenance: ESMCProvenance,
        *,
        window_radius: int = 4,
    ) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # Keep 6B HDF5 2.x output consumable by the HDF5 1.14 trainer.
        self.handle = h5py.File(self.path, "a", libver="v110")
        self.provenance = provenance
        self.window_radius = int(window_radius)
        self.window_size = 2 * self.window_radius + 1
        self._initialize()
        self._rollback_pending()
        self._load_indices()

    def _initialize(self) -> None:
        h = self.handle
        if "schema" in h.attrs:
            if h.attrs["schema"] != HIERARCHY_CACHE_SCHEMA:
                raise RuntimeError("hierarchy embedding cache schema mismatch")
            if h.attrs["model_provenance"] != self.provenance.canonical_json():
                raise RuntimeError("hierarchy cache model provenance mismatch")
            if int(h.attrs["window_radius"]) != self.window_radius:
                raise RuntimeError("hierarchy cache window-radius mismatch")
            return
        h.attrs["schema"] = HIERARCHY_CACHE_SCHEMA
        h.attrs["model_provenance"] = self.provenance.canonical_json()
        h.attrs["window_radius"] = self.window_radius
        h.attrs["created_unix"] = time.time()
        storage_dtype = np.dtype(self.provenance.storage_dtype)
        if storage_dtype not in {np.dtype(np.float16), np.dtype(np.float32)}:
            raise ValueError(
                "hierarchy embedding storage dtype must be float16 or float32"
            )
        h.create_dataset("sequence_hashes", shape=(0,), maxshape=(None,), dtype="S64")
        h.create_dataset(
            "sequences",
            shape=(0,),
            maxshape=(None,),
            dtype=h5py.string_dtype(encoding="ascii"),
        )
        h.create_dataset("sequence_length", shape=(0,), maxshape=(None,), dtype="i4")
        h.create_dataset(
            "global_mean",
            shape=(0, self.provenance.embedding_dimension),
            maxshape=(None, self.provenance.embedding_dimension),
            chunks=(256, self.provenance.embedding_dimension),
            dtype=storage_dtype,
        )
        h.create_dataset("site_sequence_index", shape=(0,), maxshape=(None,), dtype="i8")
        h.create_dataset("site_position", shape=(0,), maxshape=(None,), dtype="i4")
        h.create_dataset(
            "window_embeddings",
            shape=(0, self.window_size, self.provenance.embedding_dimension),
            maxshape=(None, self.window_size, self.provenance.embedding_dimension),
            chunks=(16, self.window_size, self.provenance.embedding_dimension),
            dtype=storage_dtype,
        )
        h.create_dataset(
            "window_mask",
            shape=(0, self.window_size),
            maxshape=(None, self.window_size),
            chunks=(1024, self.window_size),
            dtype="?",
        )
        h.flush()

    def _rollback_pending(self) -> None:
        h = self.handle
        if "pending_sequence_start" not in h.attrs:
            return
        sequence_start = int(h.attrs["pending_sequence_start"])
        site_start = int(h.attrs["pending_site_start"])
        for name in ("sequence_hashes", "sequences", "sequence_length"):
            h[name].resize((sequence_start,))
        h["global_mean"].resize(
            (sequence_start, self.provenance.embedding_dimension)
        )
        for name in ("site_sequence_index", "site_position"):
            h[name].resize((site_start,))
        h["window_embeddings"].resize(
            (site_start, self.window_size, self.provenance.embedding_dimension)
        )
        h["window_mask"].resize((site_start, self.window_size))
        del h.attrs["pending_sequence_start"]
        del h.attrs["pending_site_start"]
        h.flush()

    def _load_indices(self) -> None:
        hashes = [value.decode("ascii") for value in self.handle["sequence_hashes"][:]]
        if len(hashes) != len(set(hashes)):
            raise RuntimeError("hierarchy cache contains duplicate sequence hashes")
        self.sequence_index = {digest: index for index, digest in enumerate(hashes)}
        sequence_indices = self.handle["site_sequence_index"][:]
        positions = self.handle["site_position"][:]
        self.site_index: dict[tuple[str, int], int] = {}
        for site_index, (sequence_index, position) in enumerate(
            zip(sequence_indices, positions, strict=True)
        ):
            key = (hashes[int(sequence_index)], int(position))
            if key in self.site_index:
                raise RuntimeError(f"hierarchy cache contains duplicate site {key}")
            self.site_index[key] = site_index

    def missing_requests(
        self, requests: Iterable[EmbeddingRequest]
    ) -> list[EmbeddingRequest]:
        missing: list[EmbeddingRequest] = []
        for request in requests:
            positions = tuple(
                position
                for position in request.positions
                if (request.sequence_hash, position) not in self.site_index
            )
            if positions:
                missing.append(EmbeddingRequest(request.sequence, positions))
        return missing

    def append(
        self,
        requests: Sequence[EmbeddingRequest],
        embeddings: Sequence[HierarchicalEmbedding],
    ) -> None:
        if len(requests) != len(embeddings):
            raise ValueError("request and hierarchy embedding counts differ")
        h = self.handle
        old_sequence_count = len(h["sequence_hashes"])
        old_site_count = len(h["site_position"])
        h.attrs["pending_sequence_start"] = old_sequence_count
        h.attrs["pending_site_start"] = old_site_count
        h.flush()

        new = [
            (request, embedding)
            for request, embedding in zip(requests, embeddings, strict=True)
            if request.sequence_hash not in self.sequence_index
        ]
        new_sequence_count = old_sequence_count + len(new)
        for name in ("sequence_hashes", "sequences", "sequence_length"):
            h[name].resize((new_sequence_count,))
        h["global_mean"].resize(
            (new_sequence_count, self.provenance.embedding_dimension)
        )
        for offset, (request, embedding) in enumerate(new):
            index = old_sequence_count + offset
            if embedding.global_mean.shape != (
                self.provenance.embedding_dimension,
            ):
                raise ValueError("hierarchy global embedding dimension mismatch")
            h["sequence_hashes"][index] = request.sequence_hash.encode("ascii")
            h["sequences"][index] = request.sequence
            h["sequence_length"][index] = len(request.sequence)
            h["global_mean"][index] = embedding.global_mean
            self.sequence_index[request.sequence_hash] = index

        site_count = sum(len(request.positions) for request in requests)
        new_site_count = old_site_count + site_count
        for name in ("site_sequence_index", "site_position"):
            h[name].resize((new_site_count,))
        h["window_embeddings"].resize(
            (new_site_count, self.window_size, self.provenance.embedding_dimension)
        )
        h["window_mask"].resize((new_site_count, self.window_size))
        cursor = old_site_count
        for request, embedding in zip(requests, embeddings, strict=True):
            expected = (
                len(request.positions),
                self.window_size,
                self.provenance.embedding_dimension,
            )
            if embedding.windows.shape != expected:
                raise ValueError(
                    f"hierarchy window shape {embedding.windows.shape} != {expected}"
                )
            if embedding.window_mask.shape != expected[:2]:
                raise ValueError("hierarchy window mask shape mismatch")
            stop = cursor + len(request.positions)
            sequence_index = self.sequence_index[request.sequence_hash]
            h["site_sequence_index"][cursor:stop] = sequence_index
            h["site_position"][cursor:stop] = request.positions
            h["window_embeddings"][cursor:stop] = embedding.windows
            h["window_mask"][cursor:stop] = embedding.window_mask
            for site_index, position in enumerate(request.positions, start=cursor):
                self.site_index[(request.sequence_hash, position)] = site_index
            cursor = stop
        del h.attrs["pending_sequence_start"]
        del h.attrs["pending_site_start"]
        h.attrs["updated_unix"] = time.time()
        h.flush()

    def close(self) -> None:
        self.handle.close()

    def __enter__(self) -> "HierarchyEmbeddingWriter":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


class HierarchyEmbeddingReader:
    """Vectorized hierarchical feature lookup by sequence hash and site."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.handle = h5py.File(self.path, "r")
        if self.handle.attrs.get("schema") != HIERARCHY_CACHE_SCHEMA:
            raise RuntimeError("hierarchy embedding cache schema mismatch")
        self.provenance = json.loads(self.handle.attrs["model_provenance"])
        self.dimension = int(self.provenance["embedding_dimension"])
        self.storage_dtype = np.dtype(
            str(self.provenance.get("storage_dtype", "float16"))
        )
        if self.storage_dtype not in {
            np.dtype(np.float16),
            np.dtype(np.float32),
        }:
            raise RuntimeError("unsupported hierarchy embedding storage dtype")
        self.window_radius = int(self.handle.attrs["window_radius"])
        self.window_size = 2 * self.window_radius + 1
        hashes = [value.decode("ascii") for value in self.handle["sequence_hashes"][:]]
        sequence_indices = self.handle["site_sequence_index"][:]
        positions = self.handle["site_position"][:]
        self.site_index = {
            (hashes[int(sequence_index)], int(position)): (
                site_index,
                int(sequence_index),
            )
            for site_index, (sequence_index, position) in enumerate(
                zip(sequence_indices, positions, strict=True)
            )
        }

    def locations(
        self,
        keys: Sequence[tuple[str, int]],
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return cache site and sequence indices for stable row manifests."""

        try:
            values = [self.site_index[key] for key in keys]
        except KeyError as exc:
            raise KeyError(f"hierarchy embedding is missing for {exc.args[0]}") from exc
        return (
            np.asarray([value[0] for value in values], dtype=np.int64),
            np.asarray([value[1] for value in values], dtype=np.int64),
        )

    @staticmethod
    def _rows(dataset: h5py.Dataset, indices: np.ndarray) -> np.ndarray:
        if not len(indices):
            return np.empty((0, *dataset.shape[1:]), dtype=dataset.dtype)
        unique = np.unique(indices)
        first_chunk = (
            int(dataset.chunks[0])
            if dataset.chunks is not None
            else max(1, len(dataset))
        )
        covered_chunks = len(np.unique(unique // first_chunk))
        total_chunks = max(1, math.ceil(len(dataset) / first_chunk))
        if len(unique) >= 4096 and covered_chunks / total_chunks >= 0.25:
            row_elements = int(np.prod(dataset.shape[1:], dtype=np.int64))
            row_bytes = max(1, row_elements * dataset.dtype.itemsize)
            block_rows = max(
                first_chunk,
                (64 * 1024 * 1024 // row_bytes // first_chunk)
                * first_chunk,
            )
            output = np.empty(
                (len(indices), *dataset.shape[1:]), dtype=dataset.dtype
            )
            for start in range(0, len(dataset), block_rows):
                stop = min(len(dataset), start + block_rows)
                selected = np.flatnonzero(
                    (indices >= start) & (indices < stop)
                )
                if not len(selected):
                    continue
                block = np.asarray(dataset[start:stop])
                output[selected] = block[indices[selected] - start]
            return output
        order = np.argsort(indices)
        sorted_indices = indices[order]
        unique, inverse = np.unique(sorted_indices, return_inverse=True)
        sorted_values = dataset[unique][inverse]
        inverse_order = np.empty_like(order)
        inverse_order[order] = np.arange(len(order))
        return np.asarray(sorted_values[inverse_order])

    def features(
        self, keys: Sequence[tuple[str, int]]
    ) -> dict[str, np.ndarray]:
        site_indices, sequence_indices = self.locations(keys)
        return self.indexed_features(site_indices, sequence_indices)

    def indexed_features(
        self,
        site_indices: np.ndarray,
        sequence_indices: np.ndarray,
    ) -> dict[str, np.ndarray]:
        """Read hierarchy tensors using indices recorded in a row manifest."""

        site_indices = np.asarray(site_indices, dtype=np.int64)
        sequence_indices = np.asarray(sequence_indices, dtype=np.int64)
        if (
            site_indices.ndim != 1
            or sequence_indices.ndim != 1
            or len(site_indices) != len(sequence_indices)
        ):
            raise ValueError("hierarchy cache indices must be equal one-dimensional arrays")
        return {
            "window": self._rows(
                self.handle["window_embeddings"], site_indices
            ).astype(self.storage_dtype, copy=False),
            "window_mask": self._rows(
                self.handle["window_mask"], site_indices
            ).astype(bool),
            "global_mean": self._rows(
                self.handle["global_mean"], sequence_indices
            ).astype(self.storage_dtype, copy=False),
            "sequence_length": self._rows(
                self.handle["sequence_length"], sequence_indices
            ).astype(np.int32),
        }

    def close(self) -> None:
        self.handle.close()

    def __enter__(self) -> "HierarchyEmbeddingReader":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


def build_hierarchy_embedding_cache(
    path: Path,
    requests: Sequence[EmbeddingRequest],
    *,
    model_name: str = "esmc_600m",
    device: str = "cuda",
    window_radius: int = 4,
    max_tokens: int = 8192,
    max_batch_size: int = 128,
    storage_dtype: str = "float32",
) -> dict[str, object]:
    embedder = ESMCEmbedder(
        model_name=model_name,
        device=device,
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
                missing, max_tokens=max_tokens, max_batch_size=max_batch_size
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
                    f"hierarchy batch {batch_number}/{len(batches)}; "
                    f"cached sites={len(writer.site_index):,}",
                    flush=True,
                )
        result = {
            "schema": HIERARCHY_CACHE_SCHEMA,
            "path": str(path),
            "model_provenance": json.loads(embedder.provenance.canonical_json()),
            "request_manifest_sha256": manifest_hash,
            "window_radius": window_radius,
            "requested_sequences": len(requests),
            "requested_sites": sum(len(request.positions) for request in requests),
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
