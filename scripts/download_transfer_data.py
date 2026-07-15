#!/usr/bin/env python3
"""Download and normalize focused transfer data for GPCR stability."""

from __future__ import annotations

import hashlib
import json
import shutil
import urllib.request
from pathlib import Path

from protein_stabilizer.transfer_data import prepare_mptherm, prepare_protherm


ROOT = Path(__file__).resolve().parents[1]

FILES = {
    ROOT / "data/raw/mpthermpred/Tm_dataset.tab": (
        "https://web.iitm.ac.in/bioinfo2/mpthermpred/Tm_dataset.tab",
        "9419ce4b0c848384b634c6481b1d7f140b3ed43d942e855a2be270e1727ae4cb",
    ),
    ROOT / "data/raw/fireprotdb2/mutation_ddg_train.parquet": (
        "https://huggingface.co/datasets/drake463/FireProtDB2/resolve/main/data/subsets/mutation_ddg/train.parquet?download=true",
        "90553e511bfe8a8853a1cea8f06e1f6e73661aad1019cea6e63a215c075ecc1c",
    ),
    ROOT / "data/raw/fireprotdb2/mutation_ddg_validation.parquet": (
        "https://huggingface.co/datasets/drake463/FireProtDB2/resolve/main/data/subsets/mutation_ddg/validation.parquet?download=true",
        "845481ce7651595931964ba118bf03766da0daa6ed6571345002d2ebed0e3d47",
    ),
    ROOT / "data/raw/fireprotdb2/mutation_ddg_test.parquet": (
        "https://huggingface.co/datasets/drake463/FireProtDB2/resolve/main/data/subsets/mutation_ddg/test.parquet?download=true",
        "d776b34f375bdd9e31f83c6954853f31e32a453f8eed2462b3d19d34b25570fb",
    ),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(path: Path, url: str, expected_sha256: str) -> None:
    if path.exists() and sha256(path) == expected_sha256:
        print(f"verified {path.relative_to(ROOT)}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".partial")
    request = urllib.request.Request(url, headers={"User-Agent": "ProteinStabilizer/0.1"})
    print(f"downloading {url}")
    with urllib.request.urlopen(request, timeout=180) as response, partial.open("wb") as out:
        shutil.copyfileobj(response, out)
    partial.replace(path)
    actual = sha256(path)
    if actual != expected_sha256:
        path.unlink(missing_ok=True)
        raise RuntimeError(f"checksum mismatch for {path.name}: {actual}")


def main() -> None:
    for path, (url, expected) in FILES.items():
        download(path, url, expected)
    curated = ROOT / "data/curated"
    protherm = prepare_protherm(
        sorted((ROOT / "data/raw/fireprotdb2").glob("mutation_ddg_*.parquet")),
        curated / "protherm_ddg.csv",
        curated / "protherm_ddg.metadata.json",
    )
    mptherm = prepare_mptherm(
        ROOT / "data/raw/mpthermpred/Tm_dataset.tab",
        curated / "gpcr_finetune.csv",
        ROOT / "data/raw/uniprot",
        curated / "mptherm_dtm.csv",
        curated / "mptherm_dtm.metadata.json",
    )
    print(json.dumps({"protherm": protherm, "mptherm": mptherm}, indent=2))


if __name__ == "__main__":
    main()
