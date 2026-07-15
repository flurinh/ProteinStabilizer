#!/usr/bin/env python3
"""Download the compact curated datasets used by ThermoMPNN-D.

Only CSV tables and dataset notes are extracted. The 6.6 GB Rosetta sweep
archive is intentionally excluded from this application-focused project.
"""

from __future__ import annotations

import hashlib
import shutil
import tarfile
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw" / "thermompnn_d"
EXTRACT_DIR = ROOT / "data" / "processed" / "thermompnn_d"
BASE_URL = "https://zenodo.org/records/13345274/files"

FILES = {
    "Megascale.tar.gz": "935d9c165f80431d3f5fdacbe01b95da",
    "DMS.tar.gz": "1b118c5a33cab42f9149c3a53dd097d9",
    "PTMUL-D.tar.gz": "4fe760d0a29b1e107a17243efa6a348a",
}


def md5(path: Path) -> str:
    digest = hashlib.md5()  # noqa: S324 - checksum supplied by the data record.
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(name: str, expected_md5: str) -> Path:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    target = RAW_DIR / name
    if target.exists() and md5(target) == expected_md5:
        print(f"verified {target.relative_to(ROOT)}")
        return target

    partial = target.with_suffix(target.suffix + ".partial")
    url = f"{BASE_URL}/{name}?download=1"
    print(f"downloading {url}")
    with urllib.request.urlopen(url) as response, partial.open("wb") as output:
        shutil.copyfileobj(response, output)
    partial.replace(target)

    actual = md5(target)
    if actual != expected_md5:
        target.unlink(missing_ok=True)
        raise RuntimeError(f"checksum mismatch for {name}: {actual}")
    return target


def safe_member(member: tarfile.TarInfo) -> bool:
    path = Path(member.name)
    if path.is_absolute() or ".." in path.parts:
        raise RuntimeError(f"unsafe archive path: {member.name}")
    return member.name.endswith("/NOTE.txt") or "/csv/" in member.name


def extract_tables(archive: Path) -> None:
    EXTRACT_DIR.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "r:gz") as bundle:
        members = [member for member in bundle.getmembers() if safe_member(member)]
        bundle.extractall(EXTRACT_DIR, members=members, filter="data")
    print(f"extracted tables from {archive.name}")


def main() -> None:
    for name, expected_md5 in FILES.items():
        extract_tables(download(name, expected_md5))


if __name__ == "__main__":
    main()
