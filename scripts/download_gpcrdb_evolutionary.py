#!/usr/bin/env python3
"""Pin one target and its parent-family alignment from GPCRdb."""

from __future__ import annotations

import argparse
import hashlib
import json
import urllib.request
from pathlib import Path
from urllib.parse import quote


ROOT = Path(__file__).resolve().parents[1]
BASE_URL = "https://gpcrdb.org/services"


def _download_json(url: str) -> tuple[dict[str, object], bytes]:
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "ProteinStabilizer/0.1"},
    )
    with urllib.request.urlopen(request, timeout=180) as response:
        raw = response.read()
    payload = json.loads(raw.decode("utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"GPCRdb returned a non-object payload for {url}")
    return payload, raw


def _pin(
    output_dir: Path,
    name: str,
    url: str,
    raw: bytes,
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{name}.json"
    if path.exists() and path.read_bytes() != raw:
        raise RuntimeError(
            f"{path} already pins different content; use a fresh target cache "
            "directory to preserve provenance"
        )
    if not path.exists():
        partial = path.with_suffix(".json.part")
        partial.write_bytes(raw)
        partial.replace(path)
    return {
        "url": url,
        "path": str(path.resolve()),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "bytes": len(raw),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--entry-name",
        required=True,
        help="GPCRdb identifier such as c5ar1_human",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="fresh target-specific cache directory",
    )
    args = parser.parse_args()

    entry_name = str(args.entry_name).strip().lower()
    protein_url = f"{BASE_URL}/protein/{quote(entry_name, safe='')}/"
    metadata, metadata_raw = _download_json(protein_url)
    if str(metadata.get("entry_name", "")).lower() != entry_name:
        raise RuntimeError("GPCRdb protein response does not match entry name")
    family = str(metadata.get("family", ""))
    if family.count("_") < 1:
        raise RuntimeError("GPCRdb protein response lacks a receptor family")
    family_scope = family.rsplit("_", 1)[0]
    alignment_url = (
        f"{BASE_URL}/alignment/family/{quote(family_scope, safe='_')}/"
    )
    alignment, alignment_raw = _download_json(alignment_url)
    if entry_name not in alignment or "CONSENSUS" not in alignment:
        raise RuntimeError(
            "GPCRdb parent-family alignment lacks target or consensus"
        )

    report: dict[str, object] = {
        "schema": "protein-stabilizer.gpcrdb-download.v1",
        "entry_name": entry_name,
        "accession": str(metadata.get("accession", "")),
        "family": family,
        "alignment_family_scope": family_scope,
        "metadata": _pin(
            args.output,
            "protein",
            protein_url,
            metadata_raw,
        ),
        "alignment": _pin(
            args.output,
            "family_alignment",
            alignment_url,
            alignment_raw,
        ),
    }
    manifest_path = args.output / "download_manifest.json"
    manifest_bytes = (
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    if (
        manifest_path.exists()
        and manifest_path.read_bytes() != manifest_bytes
    ):
        raise RuntimeError(
            f"{manifest_path} already records different sources; use a "
            "fresh target cache directory"
        )
    manifest_path.write_bytes(manifest_bytes)
    report["manifest"] = {
        "path": str(manifest_path.resolve()),
        "sha256": hashlib.sha256(manifest_bytes).hexdigest(),
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
