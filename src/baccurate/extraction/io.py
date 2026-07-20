"""IO helpers specific to the extraction stage."""

import csv
from pathlib import Path

from baccurate.utils.compressed_io import open_text


def load_pathogen_map(index_path: Path, names: list[str] | None = None) -> dict[str, str]:
    selected = set(names) if names else None
    mapping: dict[str, str] = {}
    with open_text(index_path, newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            accession = (row.get("accession") or "").strip()
            pathogen = (row.get("pathogen_biosample") or "").strip()
            if not accession or not pathogen:
                continue
            if selected is not None and pathogen not in selected:
                continue
            mapping[accession] = pathogen
    return mapping
