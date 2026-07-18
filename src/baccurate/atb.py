"""
Maps a ``sylph_species`` label (GTDB-style) back to the short pathogen key
used across the project. GTDB splits polyphyletic taxa with an uppercase suffix
(``Enterococcus_B``, ``kobei_A``) that is absent from NCBI names, so those suffixes
are stripped before matching against the pathogen registry.
"""

from __future__ import annotations

import csv
import re
from collections import defaultdict
from pathlib import Path

from baccurate.pathogens import load_pathogens
from baccurate.utils.compressed_io import open_text

NA = "NA"

_GTDB_SUFFIX = re.compile(r"_[A-Z]+$")


def _norm(token: str) -> str:
    return _GTDB_SUFFIX.sub("", token).lower()


def build_keyword_maps() -> tuple[dict[str, str], dict[tuple[str, str], str]]:
    """Build (genus -> keyword) and ((genus, species) -> keyword) lookups from the registry."""
    genus_map: dict[str, str] = {}
    species_map: dict[tuple[str, str], str] = {}
    for p in load_pathogens().values():
        tokens = p.scientific_name.lower().split()
        genus = tokens[0]
        if p.rank == "genus":
            genus_map[genus] = p.key
        else:
            species_map[(genus, tokens[1])] = p.key
    return genus_map, species_map


def sylph_to_keyword(
    sylph: str,
    genus_map: dict[str, str],
    species_map: dict[tuple[str, str], str],
) -> str:
    """Resolve a sylph_species label to a pathogen key, or ``NA`` if it maps to no target."""
    parts = sylph.split()
    if not parts:
        return NA
    genus = _norm(parts[0])
    species = _norm(parts[1]) if len(parts) > 1 else None
    if species is not None and (genus, species) in species_map:
        return species_map[(genus, species)]
    return genus_map.get(genus, NA)


def load_atb_accessions_by_pathogen(index_path: Path) -> dict[str, set[str]]:
    by_pathogen: dict[str, set[str]] = defaultdict(set)
    with open_text(index_path, newline="") as stream:
        for row in csv.DictReader(stream, delimiter="\t"):
            if (row.get("in_ATB") or "").strip() != "True":
                continue
            accession = (row.get("accession") or "").strip()
            pathogen = (row.get("pathogen_ATB") or "").strip()
            if accession and pathogen:
                by_pathogen[pathogen].add(accession)
    return dict(by_pathogen)
