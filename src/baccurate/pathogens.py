"""Registry of target pathogen groups."""

import logging
from dataclasses import dataclass
from functools import cache
from pathlib import Path

import yaml

from baccurate.paths import PATHOGENS_YAML

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Pathogen:
    """One target pathogen group."""

    key: str
    scientific_name: str
    ncbi_taxid: int
    rank: str


@cache
def load_pathogens(path: Path = PATHOGENS_YAML) -> dict[str, Pathogen]:
    """Load the pathogen registry keyed by short keyword, preserving file order."""
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    return {
        key: Pathogen(
            key=key,
            scientific_name=entry["scientific_name"],
            ncbi_taxid=int(entry["ncbi_taxid"]),
            rank=entry["rank"],
        )
        for key, entry in data.items()
    }


def pathogen_keys(path: Path = PATHOGENS_YAML) -> list[str]:
    """Return the valid pathogen keywords, in registry order."""
    return list(load_pathogens(path))


def scientific_name(key: str, path: Path = PATHOGENS_YAML) -> str:
    """Return the scientific name for a pathogen key, or '' if unknown."""
    pathogen = load_pathogens(path).get(key)
    return pathogen.scientific_name if pathogen else ""
