"""IO helpers specific to the extraction stage: input resolution + output writers."""

import csv
import logging
from pathlib import Path

from baccurate.extraction.tables import COLUMNS, RecordTable
from baccurate.utils.compressed_io import open_text

logger = logging.getLogger(__name__)


def resolve_input_files(input_path: Path, uncompressed: bool = False) -> list[Path]:
    """Resolve one XML input or the selected XML representation in a directory."""
    if input_path.is_file():
        return [input_path]
    if input_path.is_dir():
        pattern = "*.xml" if uncompressed else "*.xml.gz"
        return sorted(input_path.glob(pattern))
    logger.error("Invalid input path: %s", input_path)
    return []


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

def write_tsv(records: RecordTable, path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(COLUMNS)
        writer.writerows(records.rows())
