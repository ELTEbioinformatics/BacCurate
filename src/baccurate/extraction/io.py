"""IO helpers specific to the extraction stage."""

import csv
from collections.abc import Collection, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from baccurate.adapters.compressed_io import open_text
from baccurate.taxon_registry.registry import TaxonRegistry
from baccurate.taxon_registry.species_label_matching import (
    NA,
    build_taxon_key_maps,
    sylph_to_taxon_key,
)

InclusionRoute = Literal["biosample_taxonomy", "allthebacteria"]

SEQUENCE_ACCESSION_COLUMNS = (
    "sra_run_accessions",
    "genbank_assembly_accessions",
    "refseq_assembly_accessions",
)


@dataclass(frozen=True, slots=True)
class TaxonAssignment:
    """
    A taxon key, the route that included it, and the classification each route gave.
    """

    taxon_key: str
    inclusion_route: InclusionRoute
    sequence_accessions: tuple[str, ...] = ("",) * len(SEQUENCE_ACCESSION_COLUMNS)
    ncbi_organism: str = NA
    sylph_species: str = NA


def resolve_taxon_assignment(
    row: Mapping[str, str | None],
    registered_taxon_keys: Collection[str],
    genus_map: dict[str, str],
    species_map: dict[tuple[str, str], str],
) -> TaxonAssignment | None:
    """
    Return the taxon assignment for a prepared-index row.

    Prefers the sylph_species classification over BioSample NCBI taxonomy (ncbi_organism).
    Falls back to biosample_taxonomy when sylph_species does not name a registered taxon.
    """
    sequence_accessions = tuple(
        "" if (cell := (row.get(column) or "").strip()) in ("", NA) else cell.replace(",", "||")
        for column in SEQUENCE_ACCESSION_COLUMNS
    )
    ncbi_organism = (row.get("organism_value") or "").strip() or NA
    sylph_species = (row.get("sylph_species") or "").strip() or NA
    atb_taxon_key = sylph_to_taxon_key(sylph_species, genus_map, species_map)
    biosample_taxon_key = (row.get("taxon_biosample") or "").strip()
    taxon_key = atb_taxon_key if atb_taxon_key in registered_taxon_keys else biosample_taxon_key
    if taxon_key not in registered_taxon_keys:
        return None
    route: InclusionRoute = (
        "biosample_taxonomy" if taxon_key == biosample_taxon_key else "allthebacteria"
    )
    return TaxonAssignment(taxon_key, route, sequence_accessions, ncbi_organism, sylph_species)


def load_taxon_map(
    index_path: Path,
    taxon_registry: TaxonRegistry,
    names: list[str] | None = None,
) -> dict[str, TaxonAssignment]:
    """Load taxon assignments for BioSample accessions in a prepared index."""
    selected = set(names) if names else None
    registered_taxon_keys = frozenset(taxon_registry.taxon_keys)
    genus_map, species_map = build_taxon_key_maps(taxon_registry)
    assignment_by_biosample_accession: dict[str, TaxonAssignment] = {}
    with open_text(index_path, newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            accession = (row.get("accession") or "").strip()
            assignment = resolve_taxon_assignment(
                row, registered_taxon_keys, genus_map, species_map
            )
            if not accession or assignment is None:
                continue
            if selected is not None and assignment.taxon_key not in selected:
                continue
            assignment_by_biosample_accession[accession] = assignment
    return assignment_by_biosample_accession
