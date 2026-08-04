"""Find the accession for each linked BioProject ID."""

from __future__ import annotations

import csv
from collections.abc import Iterable, Mapping
from pathlib import Path

from lxml import etree

from baccurate.adapters.compressed_io import open_binary


def resolve_bioproject_accessions(
    input_file: str | Path, linked_ids: Iterable[str]
) -> dict[str, str]:
    """Find accessions for the requested BioProject IDs in an XML file."""
    wanted = frozenset(linked_ids)
    if not wanted:
        return {}

    accession_by_id: dict[str, str] = {}
    with open_binary(input_file) as stream:
        context = etree.iterparse(
            stream,
            tag="Package",
            resolve_entities=False,
            collect_ids=False,
        )
        for _event, bioproject_record in context:
            try:
                archive_id = bioproject_record.find(".//ArchiveID")
                project_id = (
                    (archive_id.get("id") or "").strip() if archive_id is not None else ""
                )
                if project_id not in wanted:
                    continue
                accession = (archive_id.get("accession") or "").strip()
                if not accession:
                    continue
                accession_by_id[project_id] = accession
            finally:
                _clear_bioproject_record(bioproject_record)
    return accession_by_id


def write_unresolved_bioproject_links(
    linked_samples: Mapping[str, set[str]],
    resolved_ids: Iterable[str],
    destination: Path | str,
) -> Path | None:
    """Write missing linked IDs with counts and representative BioSamples.

    Returns the written path, or None when every link resolved and no
    artifact was produced.
    """
    unresolved_ids = linked_samples.keys() - set(resolved_ids)
    if not unresolved_ids:
        return None
    path = Path(destination)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, delimiter="\t", lineterminator="\n")
        writer.writerow(("bioproject_id", "count", "representative_biosample_accessions"))
        for project_id in sorted(unresolved_ids):
            accessions = sorted(linked_samples[project_id])
            representatives = "||".join(accessions[:3])
            writer.writerow((project_id, len(accessions), representatives))
    return path


def _clear_bioproject_record(bioproject_record: etree._Element) -> None:
    bioproject_record.clear()
    parent = bioproject_record.getparent()
    if parent is not None:
        while bioproject_record.getprevious() is not None:
            del parent[0]
