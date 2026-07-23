"""Validated AlphaFold DB structures for application-time ProteinMPNN context."""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Callable
from urllib.parse import quote, urlparse
from urllib.request import Request, urlopen

import numpy as np
from Bio.Data.PDBData import protein_letters_3to1_extended
from Bio.PDB import PDBParser

from .data import normalize_sequence, sequence_hash
from .embeddings import file_sha256


ALPHAFOLD_API = "https://alphafold.ebi.ac.uk/api/prediction"
ALPHAFOLD_STRUCTURE_SCHEMA = "protein-stabilizer.alphafold-structure.v1"
_ACCESSION_RE = re.compile(r"^[A-Z0-9]{6,10}(?:-\d+)?$")
_MAX_API_BYTES = 2 * 1024 * 1024
_MAX_PDB_BYTES = 100 * 1024 * 1024


@dataclass(frozen=True)
class AlphaFoldStructure:
    """One exact-sequence AlphaFold DB PDB and its usable-residue mask."""

    pdb_path: Path
    residue_mask: np.ndarray
    provenance: dict[str, object]


def _request_bytes(
    url: str,
    *,
    timeout: float,
    limit: int,
    opener: Callable[..., object],
) -> bytes:
    request = Request(
        url,
        headers={"User-Agent": "ProteinStabilizer/0.1 AlphaFoldDB client"},
    )
    with opener(request, timeout=timeout) as response:
        payload = response.read(limit + 1)
    if len(payload) > limit:
        raise ValueError(f"response from {url} exceeds {limit} bytes")
    return payload


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with NamedTemporaryFile(
            dir=path.parent,
            prefix=f".{path.name}.",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _pdb_sequence_and_plddt(
    pdb_path: Path,
) -> tuple[str, np.ndarray, str]:
    """Return the sole protein chain sequence and residue CA B factors."""

    structure = PDBParser(QUIET=True).get_structure("alphafold", str(pdb_path))
    models = list(structure.get_models())
    if len(models) != 1:
        raise ValueError(f"{pdb_path.name} must contain exactly one model")
    protein_chains: list[tuple[str, str, np.ndarray]] = []
    for chain in models[0]:
        letters: list[str] = []
        confidence: list[float] = []
        for residue in chain:
            if residue.id[0] != " ":
                continue
            amino_acid = protein_letters_3to1_extended.get(
                residue.resname.upper()
            )
            if amino_acid is None or len(amino_acid) != 1:
                raise ValueError(
                    f"{pdb_path.name} contains unsupported residue "
                    f"{residue.resname!r}"
                )
            if "CA" not in residue:
                raise ValueError(
                    f"{pdb_path.name} is missing CA coordinates at "
                    f"chain {chain.id} residue {residue.id[1]}"
                )
            letters.append(amino_acid.upper())
            confidence.append(float(residue["CA"].bfactor))
        if letters:
            protein_chains.append(
                (
                    str(chain.id),
                    "".join(letters),
                    np.asarray(confidence, dtype=np.float32),
                )
            )
    if len(protein_chains) != 1:
        raise ValueError(
            f"{pdb_path.name} must contain exactly one protein chain; "
            f"found {len(protein_chains)}"
        )
    chain, sequence, confidence = protein_chains[0]
    return sequence, confidence, chain


def _cached_structure(
    metadata_path: Path,
    *,
    target_sha256: str,
    min_plddt: float,
) -> AlphaFoldStructure | None:
    if not metadata_path.is_file():
        return None
    try:
        provenance = json.loads(metadata_path.read_text(encoding="utf-8"))
        if (
            provenance.get("schema") != ALPHAFOLD_STRUCTURE_SCHEMA
            or provenance.get("target_sequence_sha256") != target_sha256
        ):
            return None
        pdb_path = metadata_path.parent / str(provenance["pdb_filename"])
        if (
            not pdb_path.is_file()
            or file_sha256(pdb_path) != provenance.get("pdb_sha256")
        ):
            return None
        sequence, confidence, chain = _pdb_sequence_and_plddt(pdb_path)
        if sequence_hash(sequence) != target_sha256:
            return None
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
        return None
    mask = confidence >= min_plddt
    runtime_provenance = {
        **provenance,
        "cache_status": "offline-cache",
        "confidence": _confidence_provenance(
            confidence, mask, min_plddt, chain
        ),
    }
    return AlphaFoldStructure(pdb_path, mask, runtime_provenance)


def _confidence_provenance(
    confidence: np.ndarray,
    mask: np.ndarray,
    min_plddt: float,
    chain: str,
) -> dict[str, object]:
    return {
        "metric": "AlphaFold pLDDT stored in PDB CA B-factor",
        "chain": chain,
        "minimum_plddt": float(min_plddt),
        "residue_count": int(confidence.size),
        "usable_residue_count": int(mask.sum()),
        "masked_residue_count": int((~mask).sum()),
        "minimum": float(confidence.min()),
        "mean": float(confidence.mean()),
        "maximum": float(confidence.max()),
    }


def _entry_sequence(entry: dict[str, object]) -> str | None:
    raw_sequence = entry.get("sequence") or entry.get("uniprotSequence")
    if not raw_sequence:
        return None
    try:
        return normalize_sequence(str(raw_sequence))
    except ValueError:
        return None


def fetch_alphafold_structure(
    accession: str,
    target_sequence: str,
    cache_dir: Path,
    *,
    min_plddt: float = 70.0,
    timeout: float = 60.0,
    opener: Callable[..., object] = urlopen,
) -> AlphaFoldStructure:
    """Fetch the current AFDB PDB only when it exactly matches the target.

    The API-provided PDB URL is used rather than constructing a model-version
    URL. A verified cached structure is retained as an offline fallback.
    """

    normalized_accession = str(accession).strip().upper()
    if _ACCESSION_RE.fullmatch(normalized_accession) is None:
        raise ValueError(f"invalid UniProt accession: {accession!r}")
    sequence = normalize_sequence(target_sequence)
    target_sha256 = sequence_hash(sequence)
    if not 0.0 <= min_plddt <= 100.0:
        raise ValueError("minimum AlphaFold pLDDT must be between 0 and 100")
    accession_dir = Path(cache_dir).resolve() / normalized_accession
    metadata_path = accession_dir / "provenance.json"
    api_url = f"{ALPHAFOLD_API}/{quote(normalized_accession)}"
    try:
        api_bytes = _request_bytes(
            api_url,
            timeout=timeout,
            limit=_MAX_API_BYTES,
            opener=opener,
        )
        decoded = json.loads(api_bytes)
    except Exception as error:
        cached = _cached_structure(
            metadata_path,
            target_sha256=target_sha256,
            min_plddt=min_plddt,
        )
        if cached is not None:
            return cached
        raise RuntimeError(
            f"could not retrieve AlphaFold DB metadata for "
            f"{normalized_accession}"
        ) from error
    entries = decoded if isinstance(decoded, list) else [decoded]
    exact = [
        entry
        for entry in entries
        if isinstance(entry, dict)
        and str(entry.get("uniprotAccession", "")).upper()
        == normalized_accession
        and _entry_sequence(entry) == sequence
    ]
    if not exact:
        raise ValueError(
            f"AlphaFold DB has no exact sequence match for "
            f"{normalized_accession}"
        )
    exact.sort(
        key=lambda entry: int(entry.get("latestVersion") or 0),
        reverse=True,
    )
    entry = exact[0]
    pdb_url = str(entry.get("pdbUrl") or "")
    if urlparse(pdb_url).scheme != "https":
        raise ValueError("AlphaFold DB response has no secure PDB URL")
    filename = Path(urlparse(pdb_url).path).name
    if not filename.lower().endswith(".pdb") or "/" in filename:
        raise ValueError("AlphaFold DB response has an invalid PDB filename")
    pdb_path = accession_dir / filename

    cache_status = "downloaded"
    cached_metadata: dict[str, object] = {}
    if metadata_path.is_file():
        try:
            cached_metadata = json.loads(
                metadata_path.read_text(encoding="utf-8")
            )
        except (OSError, json.JSONDecodeError):
            cached_metadata = {}
    expected_hash = cached_metadata.get("pdb_sha256")
    cache_valid = (
        cached_metadata.get("pdb_url") == pdb_url
        and cached_metadata.get("target_sequence_sha256") == target_sha256
        and isinstance(expected_hash, str)
        and pdb_path.is_file()
        and file_sha256(pdb_path) == expected_hash
    )
    if cache_valid:
        cache_status = "validated-cache"
    else:
        pdb_bytes = _request_bytes(
            pdb_url,
            timeout=timeout,
            limit=_MAX_PDB_BYTES,
            opener=opener,
        )
        if not pdb_bytes.startswith((b"HEADER", b"ATOM", b"REMARK")):
            raise ValueError("AlphaFold DB response is not a PDB file")
        _atomic_write(pdb_path, pdb_bytes)

    pdb_sequence, confidence, chain = _pdb_sequence_and_plddt(pdb_path)
    if pdb_sequence != sequence:
        raise ValueError(
            f"{pdb_path.name} sequence does not exactly match "
            f"{normalized_accession}"
        )
    mask = confidence >= min_plddt
    provenance: dict[str, object] = {
        "schema": ALPHAFOLD_STRUCTURE_SCHEMA,
        "source": "AlphaFold Protein Structure Database",
        "api_url": api_url,
        "accession": normalized_accession,
        "entry_id": str(entry.get("entryId") or ""),
        "latest_version": int(entry.get("latestVersion") or 0),
        "model_created_date": entry.get("modelCreatedDate"),
        "pdb_url": pdb_url,
        "pdb_filename": filename,
        "pdb_sha256": file_sha256(pdb_path),
        "target_sequence_sha256": target_sha256,
        "api_sequence_checksum": entry.get("sequenceChecksum"),
        "global_metric_value": entry.get("globalMetricValue"),
        "retrieved_at": datetime.now(timezone.utc).isoformat(),
        "confidence": _confidence_provenance(
            confidence, mask, min_plddt, chain
        ),
    }
    _atomic_write(
        metadata_path,
        json.dumps(
            provenance, indent=2, sort_keys=True
        ).encode("utf-8")
        + b"\n",
    )
    return AlphaFoldStructure(
        pdb_path,
        mask,
        {**provenance, "cache_status": cache_status},
    )
