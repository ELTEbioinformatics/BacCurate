"""Registry of target pathogen groups."""

import logging
import sys
from dataclasses import dataclass, field
from functools import cache
from pathlib import Path

import yaml

from baccurate.paths import PATHOGENS_YAML

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Pathogen:
    """One target pathogen (a leaf entry in the registry)."""

    key: str
    scientific_name: str
    ncbi_taxid: int
    rank: str
    group: str | None = None
    also_taxids: tuple[int, ...] = field(default_factory=tuple)

    @property
    def taxids(self) -> tuple[int, ...]:
        """Primary taxid plus any ``also_taxids``."""
        return (self.ncbi_taxid, *self.also_taxids)


def _is_leaf(entry: dict) -> bool:
    return "scientific_name" in entry


def _make(key: str, entry: dict, group: str | None) -> Pathogen:
    also = entry.get("also_taxids") or []
    return Pathogen(
        key=key,
        scientific_name=entry["scientific_name"],
        ncbi_taxid=int(entry["ncbi_taxid"]),
        rank=entry["rank"],
        group=group,
        also_taxids=tuple(int(t) for t in also),
    )


@cache
def load_pathogens(path: Path = PATHOGENS_YAML) -> dict[str, Pathogen]:
    """Load the flattened leaf registry keyed by pathogen key, in file order."""
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    out: dict[str, Pathogen] = {}
    for key, entry in data.items():
        if _is_leaf(entry):
            out[key] = _make(key, entry, group=None)
        else:  # container: register its children individually
            for pathogen_key, pathogen_entry in entry.items():
                out[pathogen_key] = _make(pathogen_key, pathogen_entry, group=key)
    return out


@cache
def load_groups(path: Path = PATHOGENS_YAML) -> dict[str, list[str]]:
    """Map each container keyword to its pathogen keys, in file order."""
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    return {key: list(entry) for key, entry in data.items() if not _is_leaf(entry)}


def pathogen_keys(path: Path = PATHOGENS_YAML) -> list[str]:
    """Return the valid pathogen keys, in registry order."""
    return list(load_pathogens(path))


def group_keys(path: Path = PATHOGENS_YAML) -> list[str]:
    """Return the container keywords, in registry order."""
    return list(load_groups(path))


def all_keywords(path: Path = PATHOGENS_YAML) -> list[str]:
    """Every keyword accepted on the CLI."""
    return pathogen_keys(path) + group_keys(path)


def expand_keys(keys: list[str], path: Path = PATHOGENS_YAML) -> list[str]:
    """Expand container keywords to pathogen keys."""
    groups = load_groups(path)
    out: list[str] = []
    for key in keys:
        for pathogen_key in groups.get(key, [key]):
            if pathogen_key not in out:
                out.append(pathogen_key)
    return out


def scientific_name(key: str, path: Path = PATHOGENS_YAML) -> str:
    """Return the scientific name for a pathogen key, or '' if unknown."""
    pathogen = load_pathogens(path).get(key)
    return pathogen.scientific_name if pathogen else ""


def pathogen_key_table(path: Path = PATHOGENS_YAML) -> str:
    """TSV of ``pathogen_key / taxids / rank / group``."""
    lines = ["pathogen_key\ttaxids\trank\tgroup"]
    for p in load_pathogens(path).values():
        taxids = " ".join(str(t) for t in p.taxids)
        lines.append(f"{p.key}\t{taxids}\t{p.rank}\t{p.group or ''}")
    return "\n".join(lines)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Emit the pathogen registry as a flat pathogen-key table.")
    ap.add_argument("--pathogen-keys", action="store_true", help="emit pathogen_key/taxids/rank/group TSV")
    ap.parse_args()
    sys.stdout.reconfigure(newline="\n")
    sys.stdout.write(pathogen_key_table() + "\n")
