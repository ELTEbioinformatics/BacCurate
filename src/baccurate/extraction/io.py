"""IO helpers specific to the extraction stage."""

import csv
from collections.abc import Collection, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from baccurate.adapters.compressed_io import open_text
from baccurate.pathogen_registry.species_label_matching import NA

InclusionRoute = Literal["biosample_taxonomy", "allthebacteria"]

SEQUENCE_ACCESSION_COLUMNS = (
    "sra_run_accessions",
    "genbank_assembly_accessions",
    "refseq_assembly_accessions",
)


@dataclass(frozen=True, slots=True)
class TargetPathogenAssignment:
    """A target pathogen key, the route that included it, and its linked sequence accessions."""

    pathogen_key: str
    inclusion_route: InclusionRoute
    sequence_accessions: tuple[str, ...] = ("",) * len(SEQUENCE_ACCESSION_COLUMNS)


def resolve_pathogen_assignment(
    row: Mapping[str, str | None],
    registered_pathogen_keys: Collection[str],
) -> TargetPathogenAssignment | None:
    """Return the registered target pathogen from a prepared-index row, preferring BioSample
    taxonomy."""
    sequence_accessions = tuple(
        "" if (cell := (row.get(column) or "").strip()) in ("", NA) else cell.replace(",", "||")
        for column in SEQUENCE_ACCESSION_COLUMNS
    )
    biosample_pathogen_key = (row.get("pathogen_biosample") or "").strip()
    if biosample_pathogen_key in registered_pathogen_keys:
        return TargetPathogenAssignment(
            biosample_pathogen_key, "biosample_taxonomy", sequence_accessions
        )
    atb_pathogen_key = (row.get("pathogen_ATB") or "").strip()
    if atb_pathogen_key in registered_pathogen_keys:
        return TargetPathogenAssignment(atb_pathogen_key, "allthebacteria", sequence_accessions)
    return None


def load_pathogen_map(
    index_path: Path,
    registered_pathogen_keys: Collection[str],
    names: list[str] | None = None,
) -> dict[str, TargetPathogenAssignment]:
    """Load target pathogen assignments for BioSample accessions in a prepared index."""
    selected = set(names) if names else None
    assignment_by_biosample_accession: dict[str, TargetPathogenAssignment] = {}
    with open_text(index_path, newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            accession = (row.get("accession") or "").strip()
            assignment = resolve_pathogen_assignment(row, registered_pathogen_keys)
            if not accession or assignment is None:
                continue
            if selected is not None and assignment.pathogen_key not in selected:
                continue
            assignment_by_biosample_accession[accession] = assignment
    return assignment_by_biosample_accession
