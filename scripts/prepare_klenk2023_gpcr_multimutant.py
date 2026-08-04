#!/usr/bin/env python3
"""Prepare a frozen quantitative GPCR multi-mutant evaluation set.

The source study reports replicate delta-Tm measurements for evolved NTR1 and
PTH1R variants.  Only variants whose complete amino-acid substitution sets can
be recovered from Supplementary Figures 3 and 5 are retained.  External files
remain generated artifacts; this script records and verifies every source hash.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd


PAPER_DOI = "10.1038/s41467-023-37191-8"
PAPER_URL = f"https://doi.org/{PAPER_DOI}"
SOURCE_DATA_URL = (
    "https://static-content.springer.com/esm/"
    "art%3A10.1038%2Fs41467-023-37191-8/MediaObjects/"
    "41467_2023_37191_MOESM4_ESM.xlsx"
)
SUPPLEMENT_URL = (
    "https://static-content.springer.com/esm/"
    "art%3A10.1038%2Fs41467-023-37191-8/MediaObjects/"
    "41467_2023_37191_MOESM1_ESM.pdf"
)
EXPECTED_SOURCE_HASHES = {
    "source_data.xlsx": (
        "44e855deab3562bd89a33f2a52d3b4ff8159959e1d39be5a3b4200cc50815399"
    ),
    "supplement.pdf": (
        "41c057af88a5e7f1d4c259269d84eb36faeec9f8bd615e4d011f72a76e8945da"
    ),
    "P20789.fasta": (
        "7528aa48b711481a21514cbb37f4c0acfffaf8a89515196e83ba5af698a38f04"
    ),
    "Q03431.fasta": (
        "2cc9db8bfaa2a37c0499ddcc0f959cff6ae6c974bf502cc69d3710b4a9ae7a39"
    ),
}
SOURCE_URLS = {
    "source_data.xlsx": SOURCE_DATA_URL,
    "supplement.pdf": SUPPLEMENT_URL,
    "P20789.fasta": "https://rest.uniprot.org/uniprotkb/P20789.fasta",
    "Q03431.fasta": "https://rest.uniprot.org/uniprotkb/Q03431.fasta",
}


@dataclass(frozen=True)
class VariantDefinition:
    accession: str
    receptor: str
    receptor_class: str
    variant: str
    mutations: tuple[str, ...]
    source_sheet: str
    source_figure: str
    benchmark_role: str
    assay_context: str
    functional_context: str


NTR1_CONTEXT = (
    "full-length NTR1 in membrane fractions; ligand-binding heat challenge; "
    "all variants include signaling-disrupting R167L"
)
PTH1R_CONTEXT = (
    "PTH1R transmembrane-domain construct with residues 1-170 removed; "
    "membrane fractions without mini-Gs; ligand-binding heat challenge"
)

VARIANTS: tuple[VariantDefinition, ...] = (
    VariantDefinition(
        "P20789",
        "NTR1_RAT",
        "A",
        "N8",
        ("G107V", "R167L", "A170S", "Q239T", "N365K"),
        "Fig. 2D",
        "Supplementary Figure 3",
        "prior_receptor_continuity",
        NTR1_CONTEXT,
        "inactive/signaling-disrupted by the protected DRY-site mutation R167L",
    ),
    VariantDefinition(
        "P20789",
        "NTR1_RAT",
        "A",
        "N12",
        ("R167L", "Q239T", "N365K"),
        "Fig. 2D",
        "Supplementary Figure 3",
        "prior_receptor_continuity",
        NTR1_CONTEXT,
        "inactive/signaling-disrupted by the protected DRY-site mutation R167L",
    ),
    VariantDefinition(
        "P20789",
        "NTR1_RAT",
        "A",
        "N13",
        ("A155T", "R167L", "Q239T", "N365K"),
        "Fig. 2D",
        "Supplementary Figure 3",
        "prior_receptor_continuity",
        NTR1_CONTEXT,
        "inactive/signaling-disrupted by the protected DRY-site mutation R167L",
    ),
    VariantDefinition(
        "P20789",
        "NTR1_RAT",
        "A",
        "N21",
        ("A155T", "R167L", "M293T", "Q239T", "F358L", "N365K"),
        "Fig. 2D",
        "Supplementary Figure 3",
        "prior_receptor_continuity",
        NTR1_CONTEXT,
        "inactive/signaling-disrupted by the protected DRY-site mutation R167L",
    ),
    VariantDefinition(
        "P20789",
        "NTR1_RAT",
        "A",
        "N23",
        ("A155T", "R167L", "Q239T", "N365K", "R392C"),
        "Fig. 2D",
        "Supplementary Figure 3",
        "prior_receptor_continuity",
        NTR1_CONTEXT,
        "inactive/signaling-disrupted by the protected DRY-site mutation R167L",
    ),
    VariantDefinition(
        "Q03431",
        "PTH1R_HUMAN",
        "B1",
        "P34_05",
        ("P271S", "V283M", "M312Q", "V412C"),
        "Supplementary Fig. 7A",
        "Supplementary Figure 5",
        "new_receptor_lockbox",
        PTH1R_CONTEXT,
        "signaling-active evolved receptor; stability measured without G protein",
    ),
    VariantDefinition(
        "Q03431",
        "PTH1R_HUMAN",
        "B1",
        "P34_13",
        ("T294A", "T325S", "S341I", "A369C", "L373I"),
        "Supplementary Fig. 7A",
        "Supplementary Figure 5",
        "new_receptor_lockbox",
        PTH1R_CONTEXT,
        "signaling-active evolved receptor; stability measured without G protein",
    ),
    VariantDefinition(
        "Q03431",
        "PTH1R_HUMAN",
        "B1",
        "P14_12",
        (
            "F184W",
            "G188S",
            "L368M",
            "A369I",
            "L373I",
            "L407V",
            "T427A",
            "F461Y",
        ),
        "Supplementary Fig. 7A",
        "Supplementary Figure 5",
        "new_receptor_lockbox",
        PTH1R_CONTEXT,
        "signaling-active evolved receptor; stability measured without G protein",
    ),
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sequence_sha256(sequence: str) -> str:
    return hashlib.sha256(sequence.encode("ascii")).hexdigest()


def _download(url: str, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "ProteinStabilizer/1 benchmark-preparation"},
    )
    temporary = output.with_suffix(output.suffix + ".partial")
    with urllib.request.urlopen(request, timeout=120) as response:
        temporary.write_bytes(response.read())
    temporary.replace(output)


def ensure_sources(output_dir: Path) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    for name, url in SOURCE_URLS.items():
        path = output_dir / "raw" / name
        if not path.is_file():
            _download(url, path)
        actual = file_sha256(path)
        expected = EXPECTED_SOURCE_HASHES[name]
        if actual != expected:
            raise RuntimeError(
                f"source hash mismatch for {name}: expected {expected}, got {actual}"
            )
        paths[name] = path
    return paths


def read_fasta(path: Path) -> tuple[str, str]:
    lines = path.read_text(encoding="utf-8").splitlines()
    headers = [line for line in lines if line.startswith(">")]
    if len(headers) != 1:
        raise ValueError(f"expected one FASTA record in {path}")
    sequence = "".join(line.strip() for line in lines if not line.startswith(">"))
    if not sequence or any(
        amino_acid not in "ACDEFGHIKLMNPQRSTVWY" for amino_acid in sequence
    ):
        raise ValueError(f"non-canonical sequence in {path}")
    return headers[0][1:], sequence


def _parse_mutation(mutation: str) -> tuple[str, int, str]:
    if len(mutation) < 3:
        raise ValueError(f"invalid mutation {mutation!r}")
    wt = mutation[0]
    mutant = mutation[-1]
    try:
        position = int(mutation[1:-1])
    except ValueError as error:
        raise ValueError(f"invalid mutation {mutation!r}") from error
    return wt, position, mutant


def canonical_mutation_set(
    mutations: Sequence[str],
    sequence: str,
) -> tuple[str, ...]:
    parsed = sorted(
        (_parse_mutation(value) for value in mutations),
        key=lambda row: row[1],
    )
    positions = [position for _, position, _ in parsed]
    if len(positions) != len(set(positions)):
        raise ValueError("mutation set contains two substitutions at one position")
    result: list[str] = []
    for wt, position, mutant in parsed:
        if position < 1 or position > len(sequence):
            raise ValueError(f"mutation {wt}{position}{mutant} lies outside sequence")
        if sequence[position - 1] != wt:
            raise ValueError(
                f"mutation {wt}{position}{mutant} disagrees with canonical "
                f"sequence residue {sequence[position - 1]}"
            )
        if mutant == wt:
            raise ValueError(f"mutation {wt}{position}{mutant} is synonymous")
        result.append(f"{wt}{position}{mutant}")
    return tuple(result)


def _numeric_replicates(row: Iterable[object]) -> list[float]:
    values: list[float] = []
    for value in row:
        if pd.isna(value):
            continue
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(number):
            values.append(number)
    if len(values) < 2:
        raise ValueError("expected at least two finite delta-Tm replicates")
    return values


def _sheet_replicates(
    workbook: Path,
    sheet: str,
    variants: set[str],
) -> dict[str, list[float]]:
    frame = pd.read_excel(workbook, sheet_name=sheet, header=None)
    section_rows = [
        index
        for index, value in frame.iloc[:, 0].items()
        if str(value).strip().lower().startswith("delta-tm values from wt")
    ]
    if len(section_rows) != 1:
        raise ValueError(
            f"{sheet} has {len(section_rows)} delta-Tm sections; expected one"
        )
    delta_tm_frame = frame.loc[section_rows[0] + 1 :]
    result: dict[str, list[float]] = {}
    for _, row in delta_tm_frame.iterrows():
        name = str(row.iloc[0]).strip()
        if name not in variants:
            continue
        if name in result:
            raise ValueError(f"duplicate source-data row for {name}")
        result[name] = _numeric_replicates(row.iloc[1:].tolist())
    missing = sorted(variants - set(result))
    if missing:
        raise ValueError(f"{sheet} is missing variants: {missing}")
    return result


def prepare(output_dir: Path) -> dict[str, object]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    sources = ensure_sources(output_dir)
    sequences: dict[str, str] = {}
    fasta_headers: dict[str, str] = {}
    for accession in ("P20789", "Q03431"):
        header, sequence = read_fasta(sources[f"{accession}.fasta"])
        fasta_headers[accession] = header
        sequences[accession] = sequence

    by_sheet: dict[str, set[str]] = {}
    for definition in VARIANTS:
        by_sheet.setdefault(definition.source_sheet, set()).add(definition.variant)
    replicate_by_variant: dict[str, list[float]] = {}
    for sheet, variants in by_sheet.items():
        replicate_by_variant.update(
            _sheet_replicates(sources["source_data.xlsx"], sheet, variants)
        )

    rows: list[dict[str, object]] = []
    for definition in VARIANTS:
        sequence = sequences[definition.accession]
        mutations = canonical_mutation_set(definition.mutations, sequence)
        replicates = replicate_by_variant[definition.variant]
        standard_deviation = statistics.stdev(replicates)
        rows.append(
            {
                "protein_id": definition.receptor,
                "uniprot_id": definition.accession,
                "receptor_class": definition.receptor_class,
                "variant": definition.variant,
                "mutation_set": ",".join(mutations),
                "mutation_count": len(mutations),
                "wt_sequence": sequence,
                "wt_sequence_sha256": sequence_sha256(sequence),
                "experimental_delta_tm_mean_c": statistics.fmean(replicates),
                "experimental_delta_tm_sd_c": standard_deviation,
                "experimental_delta_tm_sem_c": (
                    standard_deviation / math.sqrt(len(replicates))
                ),
                "experimental_delta_tm_replicates_c": "|".join(
                    f"{value:.12g}" for value in replicates
                ),
                "replicate_count": len(replicates),
                "target_kind": "delta_tm",
                "target_units": "degrees Celsius",
                "target_direction": "positive is stabilizing",
                "benchmark_role": definition.benchmark_role,
                "assay_context": definition.assay_context,
                "functional_context": definition.functional_context,
                "source_sheet": definition.source_sheet,
                "source_figure": definition.source_figure,
                "source_doi": PAPER_DOI,
                "source_url": PAPER_URL,
            }
        )

    output_csv = output_dir / "klenk2023_gpcr_multimutant.csv"
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    metadata = {
        "schema": "protein-stabilizer.klenk2023-gpcr-multimutant.v1",
        "frozen_on": "2026-07-24",
        "model_predictions_consulted_during_definition": False,
        "source": {
            "doi": PAPER_DOI,
            "url": PAPER_URL,
            "files": {
                name: {
                    "url": SOURCE_URLS[name],
                    "path": str(path.resolve()),
                    "sha256": file_sha256(path),
                }
                for name, path in sources.items()
            },
            "fasta_headers": fasta_headers,
        },
        "dataset": {
            "path": str(output_csv.resolve()),
            "sha256": file_sha256(output_csv),
            "rows": len(rows),
            "receptors": sorted({str(row["protein_id"]) for row in rows}),
            "new_receptor_lockbox_rows": sum(
                row["benchmark_role"] == "new_receptor_lockbox" for row in rows
            ),
            "prior_receptor_continuity_rows": sum(
                row["benchmark_role"] == "prior_receptor_continuity"
                for row in rows
            ),
            "mutation_count_range": [
                min(int(row["mutation_count"]) for row in rows),
                max(int(row["mutation_count"]) for row in rows),
            ],
        },
        "selection_policy": {
            "included": (
                "quantitative delta-Tm variants with at least two source-data "
                "replicates and a complete pure-substitution set recoverable "
                "from Supplementary Figures 3 or 5"
            ),
            "excluded": {
                "N14,N15": "frameshift variants, not pure substitution sets",
                "P34_06,P34_07": (
                    "complete mutation count is not recoverable from "
                    "Supplementary Figure 5"
                ),
                "PTy03": "control sequence is not defined in this source package",
            },
            "primary_pth1r_endpoint": (
                "delta-Tm without mini-Gs from Supplementary Figure 7A"
            ),
            "no_cross_unit_mae": (
                "model kcal/mol predictions may be compared only by favorable "
                "direction/rank with experimental degrees Celsius"
            ),
            "no_fitting": (
                "do not fit, calibrate, select a blend, or tune a threshold on "
                "these rows"
            ),
            "overlap_policy": {
                "PTH1R_HUMAN": (
                    "new receptor lockbox relative to the project's GPCR-tm, "
                    "GPCR workbook, Muk scan, C5aR, and MPTherm receptors"
                ),
                "NTR1_RAT": (
                    "same-receptor continuity diagnostic only; NTR1 appears in "
                    "prior GPCR and MPTherm development data"
                ),
            },
        },
    }
    metadata_path = output_dir / "klenk2023_gpcr_multimutant.metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return metadata


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "/data/fast/tmp/protein-stabilizer/prospective-gpcr/klenk2023"
        ),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    print(json.dumps(prepare(args.output_dir), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
