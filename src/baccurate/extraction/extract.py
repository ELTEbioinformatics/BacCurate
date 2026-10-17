"""Extraction of BacCurate-specific BioSample metadata from the NCBI snapshots."""

import csv
import logging
import os
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

from baccurate import paths
from baccurate.extraction.bioproject import (
    resolve_bioproject_accessions,
    write_unresolved_bioproject_links,
)
from baccurate.extraction.io import InclusionRoute, load_taxon_map
from baccurate.extraction.manual_review import ReviewWorklists
from baccurate.extraction.selection import SelectionPolicy
from baccurate.extraction.tables import COLUMNS, INTERMEDIATE_COLUMNS, extracted_metadata_row
from baccurate.extraction.xml import SelectionCounters, process_biosample_xml
from baccurate.provenance.extracted_metadata import (
    ExtractedMetadataProvenance,
    provenance_path_for,
)
from baccurate.provenance.snapshot_manifest import validate_snapshot_pair
from baccurate.taxon_registry.registry import TaxonRegistry, load_taxon_registry

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class ExtractionReport:
    """Provenance and selection counts for one extraction."""

    prepared_input_paths: tuple[Path, ...]
    extracted_metadata_path: Path
    extracted_record_count: int
    counters: SelectionCounters
    automatic_rejection_counts: dict[str, dict[str, int]]
    unreviewed_count: int
    uncertain_count: int
    review_worklist_paths: dict[str, Path]
    biosample_snapshot_id: str
    bioproject_snapshot_id: str
    metadata_reference_date: date
    extracted_metadata_provenance_path: Path
    inclusion_route_counts: dict[InclusionRoute, int] = field(default_factory=dict)


def run_extraction(
    output_path: Path,
    index_path: Path = paths.DEFAULT_INDEX_TSV,
    names: list[str] | None = None,
    *,
    selection_policy: SelectionPolicy,
    taxon_registry: TaxonRegistry | None = None,
) -> ExtractionReport:
    snapshot_pair = validate_snapshot_pair(
        biosample_snapshot_path=paths.DEFAULT_BIOSAMPLE_SNAPSHOT_PATH,
        bioproject_snapshot_path=paths.DEFAULT_BIOPROJECT_SNAPSHOT_PATH,
        biosample_manifest_path=paths.DEFAULT_BIOSAMPLE_MANIFEST_PATH,
        bioproject_manifest_path=paths.DEFAULT_BIOPROJECT_MANIFEST_PATH,
    )

    counters = SelectionCounters()
    review_worklists = ReviewWorklists()

    registry = taxon_registry or load_taxon_registry()
    taxon_assignment_by_accession = load_taxon_map(index_path, registry, names)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    provenance_path = provenance_path_for(output_path)
    extracted_record_count = 0
    inclusion_route_counts: Counter[InclusionRoute] = Counter()
    linked_project_samples: dict[str, set[str]] = defaultdict(set)

    with tempfile.TemporaryDirectory(
        prefix=f".{output_path.stem}-", dir=output_path.parent
    ) as temporary_dir:
        temporary_dir_path = Path(temporary_dir)
        spool_path = temporary_dir_path / "biosample_rows.tsv"
        temporary_output_path = temporary_dir_path / output_path.name
        temporary_provenance_path = temporary_dir_path / provenance_path.name

        with spool_path.open("w", newline="", encoding="utf-8") as spool_stream:
            writer = csv.writer(spool_stream, delimiter="\t")
            writer.writerow(INTERMEDIATE_COLUMNS)
            logger.info("Parsing %s...", paths.DEFAULT_BIOSAMPLE_SNAPSHOT_PATH)
            records = process_biosample_xml(
                str(paths.DEFAULT_BIOSAMPLE_SNAPSHOT_PATH), selection_policy.evaluate, counters
            )
            for accession, decisions, bioproject_ids, ncbi_organism in records:
                for decision in decisions:
                    review_worklists.observe(decision, accession=accession)
                assignment = taxon_assignment_by_accession.get(accession)
                if assignment is None:
                    continue
                extracted_metadata_values = extracted_metadata_row(
                    accession=accession,
                    assignment=assignment,
                    ncbi_organism=ncbi_organism,
                    bioproject_id="||".join(bioproject_ids),
                    bioproject_accession="",
                    decisions=decisions,
                )
                if extracted_metadata_values is not None:
                    writer.writerow(extracted_metadata_values)
                    extracted_record_count += 1
                    inclusion_route_counts[assignment.inclusion_route] += 1
                    for project_id in bioproject_ids:
                        linked_project_samples[project_id].add(accession)

        accession_by_bioproject_id = resolve_bioproject_accessions(
            paths.DEFAULT_BIOPROJECT_SNAPSHOT_PATH,
            linked_project_samples,
        )
        _write_resolved_rows(spool_path, temporary_output_path, accession_by_bioproject_id)

        provenance = ExtractedMetadataProvenance.create(
            biosample_manifest_path=paths.DEFAULT_BIOSAMPLE_MANIFEST_PATH,
            bioproject_manifest_path=paths.DEFAULT_BIOPROJECT_MANIFEST_PATH,
            extracted_metadata_path=temporary_output_path,
        )
        provenance.write(temporary_provenance_path)

        review_worklist_paths = review_worklists.write(output_path.parent)
        unresolved_path = write_unresolved_bioproject_links(
            linked_project_samples,
            accession_by_bioproject_id.keys(),
            output_path.parent / "unresolved_bioproject_links.tsv",
        )
        if unresolved_path is not None:
            review_worklist_paths["unresolved_bioproject_links"] = unresolved_path
        if review_worklists.has_unreviewed:
            logger.warning(
                "Unreviewed metadata attributes were excluded. See unreviewed_attributes.tsv"
            )
        review_worklists.log_automatic_rejections(logger)
        logger.info("Selection summary: %s", counters.summary())

        os.replace(temporary_output_path, output_path)
        os.replace(temporary_provenance_path, provenance_path)
    return ExtractionReport(
        prepared_input_paths=(
            paths.DEFAULT_BIOSAMPLE_SNAPSHOT_PATH,
            paths.DEFAULT_BIOPROJECT_SNAPSHOT_PATH,
        ),
        extracted_metadata_path=output_path,
        extracted_record_count=extracted_record_count,
        counters=counters,
        automatic_rejection_counts=review_worklists.automatic_rejection_counts,
        unreviewed_count=review_worklists.unreviewed_count,
        uncertain_count=review_worklists.uncertain_count,
        review_worklist_paths=review_worklist_paths,
        biosample_snapshot_id=snapshot_pair.biosample.snapshot_id,
        bioproject_snapshot_id=snapshot_pair.bioproject.snapshot_id,
        metadata_reference_date=snapshot_pair.metadata_reference_date,
        extracted_metadata_provenance_path=provenance_path,
        inclusion_route_counts={
            route: inclusion_route_counts[route]
            for route in ("biosample_taxonomy", "allthebacteria")
        },
    )


def _write_resolved_rows(
    spool_path: Path,
    output_path: Path,
    accession_by_bioproject_id: dict[str, str],
) -> None:
    with (
        spool_path.open(newline="", encoding="utf-8") as input_stream,
        output_path.open("w", newline="", encoding="utf-8") as output_stream,
    ):
        reader = csv.DictReader(input_stream, delimiter="\t")
        writer = csv.DictWriter(
            output_stream,
            fieldnames=COLUMNS,
            delimiter="\t",
            lineterminator="\n",
            extrasaction="ignore",
        )
        writer.writeheader()
        for row in reader:
            project_ids = row["bioproject_id"].split("||") if row["bioproject_id"] else ()
            row["bioproject_accession"] = "||".join(
                sorted(
                    accession_by_bioproject_id[project_id]
                    for project_id in project_ids
                    if project_id in accession_by_bioproject_id
                )
            )
            writer.writerow(row)
