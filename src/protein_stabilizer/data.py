"""Dataset normalization and leakage-safe split utilities."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Sequence

import numpy as np
import pandas as pd


AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
MUTATION_RE = re.compile(r"^([A-Z])(\d+)([A-Z])$")


@dataclass(frozen=True)
class Mutation:
    """One one-based amino-acid substitution."""

    wt: str
    position: int
    mutant: str

    @classmethod
    def parse(cls, value: str) -> "Mutation":
        match = MUTATION_RE.fullmatch(str(value).strip().upper())
        if match is None:
            raise ValueError(f"invalid mutation notation: {value!r}")
        wt, position, mutant = match.groups()
        if wt not in AMINO_ACIDS or mutant not in AMINO_ACIDS:
            raise ValueError(f"non-canonical mutation: {value!r}")
        return cls(wt=wt, position=int(position), mutant=mutant)

    def __str__(self) -> str:
        return f"{self.wt}{self.position}{self.mutant}"


def normalize_sequence(sequence: str) -> str:
    value = "".join(str(sequence).split()).upper()
    if not value or any(aa not in AMINO_ACIDS for aa in value):
        raise ValueError("sequence must contain only the 20 canonical amino acids")
    return value


def sequence_hash(sequence: str) -> str:
    return hashlib.sha256(normalize_sequence(sequence).encode("ascii")).hexdigest()


def apply_mutations(sequence: str, mutations: Sequence[Mutation]) -> str:
    chars = list(normalize_sequence(sequence))
    positions: set[int] = set()
    for mutation in mutations:
        if mutation.position in positions:
            raise ValueError(f"multiple mutations at position {mutation.position}")
        positions.add(mutation.position)
        index = mutation.position - 1
        if not 0 <= index < len(chars):
            raise ValueError(f"mutation position {mutation.position} is outside sequence")
        if chars[index] != mutation.wt:
            raise ValueError(
                f"mutation {mutation} expects {mutation.wt} at {mutation.position}, "
                f"found {chars[index]}"
            )
        chars[index] = mutation.mutant
    return "".join(chars)


def reconstruct_double(joint_sequence: str, mut1: str, mut2: str) -> tuple[str, str, str]:
    """Return WT, first-single, and second-single sequences from a joint mutant."""

    joint = normalize_sequence(joint_sequence)
    mutations = (Mutation.parse(mut1), Mutation.parse(mut2))
    if mutations[0].position == mutations[1].position:
        raise ValueError("double mutant substitutions must use distinct positions")
    wt_chars = list(joint)
    for mutation in mutations:
        index = mutation.position - 1
        if not 0 <= index < len(wt_chars):
            raise ValueError(f"mutation position {mutation.position} is outside sequence")
        if wt_chars[index] != mutation.mutant:
            raise ValueError(
                f"joint sequence does not contain {mutation.mutant} for {mutation}"
            )
        wt_chars[index] = mutation.wt
    wt = "".join(wt_chars)
    return wt, apply_mutations(wt, [mutations[0]]), apply_mutations(wt, [mutations[1]])


@dataclass(frozen=True)
class EmbeddingRequest:
    sequence: str
    positions: tuple[int, ...]

    @property
    def sequence_hash(self) -> str:
        return sequence_hash(self.sequence)


def merge_embedding_requests(requests: Iterable[EmbeddingRequest]) -> list[EmbeddingRequest]:
    by_hash: dict[str, tuple[str, set[int]]] = {}
    for request in requests:
        sequence = normalize_sequence(request.sequence)
        digest = sequence_hash(sequence)
        current = by_hash.get(digest)
        if current is None:
            current = (sequence, set())
            by_hash[digest] = current
        elif current[0] != sequence:
            raise RuntimeError("SHA-256 sequence collision")
        for position in request.positions:
            if not 1 <= position <= len(sequence):
                raise ValueError(f"requested position {position} is outside sequence")
            current[1].add(int(position))
    return [
        EmbeddingRequest(sequence=sequence, positions=tuple(sorted(positions)))
        for _, (sequence, positions) in sorted(by_hash.items())
    ]


@dataclass(frozen=True)
class DatasetPaths:
    root: Path

    @property
    def megascale(self) -> Path:
        return self.root / "data/processed/thermompnn_d/Megascale/csv"

    @property
    def gpcr(self) -> Path:
        return self.root / "data/curated/gpcr_finetune.csv"

    @property
    def protherm(self) -> Path:
        return self.root / "data/curated/protherm_ddg.csv"

    @property
    def mptherm(self) -> Path:
        return self.root / "data/curated/mptherm_dtm.csv"

    def single(self, split: str) -> Path:
        names = {"train": "cdna1_train.csv", "test": "cdna1_test.csv"}
        return self.megascale / names[split]

    def double(self, split: str) -> Path:
        return self.megascale / f"Megascale-D-{split}.csv"


def single_rows(path: Path) -> Iterator[dict[str, object]]:
    frame = pd.read_csv(path)
    required = {"pdb_id", "ddg", "mut_info", "mut_seq", "wt_seq"}
    if not required.issubset(frame.columns):
        raise ValueError(f"{path} is missing columns {sorted(required - set(frame.columns))}")
    for row in frame.itertuples(index=False):
        mutation = Mutation.parse(row.mut_info)
        wt = normalize_sequence(row.wt_seq)
        mutant = normalize_sequence(row.mut_seq)
        if apply_mutations(wt, [mutation]) != mutant:
            raise ValueError(f"single-mutant sequence mismatch for {row.pdb_id} {mutation}")
        yield {
            "protein_id": str(row.pdb_id),
            "mutation": str(mutation),
            "wt_sequence": wt,
            "mutant_sequence": mutant,
            "position": mutation.position,
            "target": float(row.ddg),
        }


def double_rows(path: Path) -> Iterator[dict[str, object]]:
    frame = pd.read_csv(path)
    required = {"WT_name", "ddG_ML", "mut1", "mut2", "aa_seq"}
    if not required.issubset(frame.columns):
        raise ValueError(f"{path} is missing columns {sorted(required - set(frame.columns))}")
    for row in frame.itertuples(index=False):
        mutations = (Mutation.parse(row.mut1), Mutation.parse(row.mut2))
        joint = normalize_sequence(row.aa_seq)
        wt, single1, single2 = reconstruct_double(joint, row.mut1, row.mut2)
        yield {
            "protein_id": str(row.WT_name),
            "description": str(row.description),
            "mutations": (str(mutations[0]), str(mutations[1])),
            "positions": (mutations[0].position, mutations[1].position),
            "wt_sequence": wt,
            "single_sequences": (single1, single2),
            "joint_sequence": joint,
            "target": float(row.ddG_ML),
        }


def gpcr_rows(path: Path, *, include_wt: bool = False) -> Iterator[dict[str, object]]:
    frame = pd.read_csv(path)
    for row in frame.itertuples(index=False):
        if int(row.is_wt):
            if include_wt:
                yield {
                    "protein_id": str(row.protein_id),
                    "assay_id": str(row.assay_id),
                    "mutation": "WT",
                    "wt_sequence": normalize_sequence(row.sequence),
                    "mutant_sequence": normalize_sequence(row.sequence),
                    "position": 0,
                    "target": float(row.stability_delta_percent),
                }
            continue
        mutation = Mutation.parse(row.mutation)
        wt = normalize_sequence(row.sequence)
        yield {
            "protein_id": str(row.protein_id),
            "assay_id": str(row.assay_id),
            "mutation": str(mutation),
            "wt_sequence": wt,
            "mutant_sequence": apply_mutations(wt, [mutation]),
            "position": mutation.position,
            "target": float(row.stability_delta_percent),
        }


def transfer_rows(path: Path) -> Iterator[dict[str, object]]:
    """Read normalized single-mutant transfer data without mixing target types."""

    frame = pd.read_csv(path)
    required = {
        "protein_id",
        "mutation",
        "wt_sequence",
        "mutant_sequence",
        "position",
        "target",
        "target_kind",
        "split",
    }
    if not required.issubset(frame.columns):
        raise ValueError(f"{path} is missing columns {sorted(required - set(frame.columns))}")
    target_kinds = set(frame["target_kind"].astype(str))
    if len(target_kinds) != 1:
        raise ValueError(f"{path} mixes target types: {sorted(target_kinds)}")
    for row in frame.to_dict("records"):
        source_mutation = Mutation.parse(str(row["mutation"]))
        mutation = Mutation.parse(str(row.get("embedding_mutation", row["mutation"])))
        wt = normalize_sequence(str(row["wt_sequence"]))
        mutant = normalize_sequence(str(row["mutant_sequence"]))
        if apply_mutations(wt, [mutation]) != mutant:
            raise ValueError(f"transfer sequence mismatch for {row['protein_id']} {mutation}")
        value = dict(row)
        value.update(
            {
                "protein_id": str(row["protein_id"]),
                "mutation": str(source_mutation),
                "embedding_mutation": str(mutation),
                "wt_sequence": wt,
                "mutant_sequence": mutant,
                "position": mutation.position,
                "source_position": source_mutation.position,
                "target": float(row["target"]),
                "split": str(row["split"]),
                "target_kind": str(row["target_kind"]),
            }
        )
        yield value


def all_embedding_requests(paths: DatasetPaths) -> list[EmbeddingRequest]:
    requests: list[EmbeddingRequest] = []
    for split in ("train", "test"):
        for row in single_rows(paths.single(split)):
            position = int(row["position"])
            requests.extend(
                [
                    EmbeddingRequest(str(row["wt_sequence"]), (position,)),
                    EmbeddingRequest(str(row["mutant_sequence"]), (position,)),
                ]
            )
    for split in ("train", "val", "test"):
        for row in double_rows(paths.double(split)):
            positions = tuple(int(value) for value in row["positions"])
            requests.append(EmbeddingRequest(str(row["wt_sequence"]), positions))
            for position, sequence in zip(positions, row["single_sequences"], strict=True):
                requests.append(EmbeddingRequest(str(sequence), (position,)))
            requests.append(EmbeddingRequest(str(row["joint_sequence"]), positions))
    for row in gpcr_rows(paths.gpcr):
        position = int(row["position"])
        requests.extend(
            [
                EmbeddingRequest(str(row["wt_sequence"]), (position,)),
                EmbeddingRequest(str(row["mutant_sequence"]), (position,)),
            ]
        )
    for source in (paths.protherm, paths.mptherm):
        if not source.is_file():
            continue
        for row in transfer_rows(source):
            position = int(row["position"])
            requests.extend(
                [
                    EmbeddingRequest(str(row["wt_sequence"]), (position,)),
                    EmbeddingRequest(str(row["mutant_sequence"]), (position,)),
                ]
            )
    return merge_embedding_requests(requests)


def gpcr_site_splits(path: Path, seed: int = 20260715) -> pd.DataFrame:
    """Assign whole mutation sites to train/validation/test within each protein."""

    frame = pd.DataFrame(gpcr_rows(path))
    frame["site_id"] = frame["protein_id"] + ":" + frame["position"].astype(str)
    split_by_site: dict[str, str] = {}
    rng = np.random.default_rng(seed)
    for _, protein_rows in frame.groupby("protein_id", sort=True):
        sites = np.array(sorted(protein_rows["site_id"].unique()))
        n_test = max(1, round(0.20 * len(sites)))
        n_val = max(1, round(0.20 * len(sites)))
        if n_test + n_val >= len(sites):
            raise ValueError("not enough mutation sites for a three-way split")

        full_assay_counts = protein_rows["assay_id"].value_counts().to_dict()
        target_counts = {
            str(assay): 0.20 * int(count)
            for assay, count in full_assay_counts.items()
        }

        def choose_balanced(
            candidates: np.ndarray, count: int, excluded: set[str]
        ) -> list[str]:
            available = np.asarray(
                [site for site in candidates if str(site) not in excluded]
            )
            best: tuple[float, tuple[str, ...]] | None = None
            trials = max(1024, min(8192, 256 * len(available)))
            for _ in range(trials):
                chosen = tuple(sorted(str(value) for value in rng.choice(available, count, replace=False)))
                subset = protein_rows[protein_rows["site_id"].isin(chosen)]
                observed = subset["assay_id"].value_counts().to_dict()
                penalty = 0.0
                for assay, target in target_counts.items():
                    value = int(observed.get(assay, 0))
                    penalty += ((value - target) ** 2) / max(1.0, target)
                    if full_assay_counts[assay] >= 10 and value < 2:
                        penalty += 25.0
                candidate = (penalty, chosen)
                if best is None or candidate < best:
                    best = candidate
            if best is None:
                raise RuntimeError("could not construct a balanced grouped split")
            return list(best[1])

        test_sites = choose_balanced(sites, n_test, set())
        val_sites = choose_balanced(sites, n_val, set(test_sites))
        assigned = set(test_sites) | set(val_sites)
        train_sites = [str(site) for site in sites if str(site) not in assigned]
        for site in test_sites:
            split_by_site[str(site)] = "test"
        for site in val_sites:
            split_by_site[str(site)] = "val"
        for site in train_sites:
            split_by_site[str(site)] = "train"
    frame["split"] = frame["site_id"].map(split_by_site)
    if frame.groupby("site_id")["split"].nunique().max() != 1:
        raise RuntimeError("mutation-site leakage detected")
    return frame
