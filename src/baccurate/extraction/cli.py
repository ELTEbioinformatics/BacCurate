"""CLI entry point for metadata extraction stage."""

import argparse
import csv
import logging
import os
import tempfile
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from baccurate import paths
from baccurate.extraction._review_reports import ReviewReports
from baccurate.extraction.bioproject import (
    BioProjectContext,
    resolve_bioproject_contexts,
    write_bioproject_catalog,
    write_unresolved_bioproject_links,
)
from baccurate.extraction.curation import CurationSchema
from baccurate.extraction.io import load_pathogen_map
from baccurate.extraction.tables import COLUMNS, record_row
from baccurate.extraction.xml import CandidateCounters, process_biosample_xml
from baccurate.source_snapshot import (
    DerivedBundleProvenance,
    bioproject_catalog_path_for,
    provenance_path_for,
    validate_paired_source_contract,
)
from baccurate.utils.progress import make_progress_bar

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class ExtractionReport:
    """Data provenance and a curation summary of extraction."""

    source_xml_paths: tuple[Path, ...]
    extracted_metadata_path: Path
    extracted_record_count: int
    counters: CandidateCounters
    automatic_rejection_counts: dict[str, dict[str, int]]
    unreviewed_count: int
    uncertain_count: int
    review_artifact_paths: dict[str, Path]
    biosample_snapshot_id: str
    bioproject_snapshot_id: str
    metadata_reference_date: date
    bundle_provenance_path: Path
    bioproject_catalog_path: Path


def run_extraction(
    output_path: Path,
    index_path: Path = paths.DEFAULT_INDEX_TSV,
    names: list[str] | None = None,
    log_level: str = "INFO",
    config_dir: Path = paths.CONFIG_DIR,
    disable_progress: bool = False,
    curation_schema: CurationSchema | None = None,
) -> ExtractionReport:
    logging.basicConfig(
        level=log_level.upper(),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    biosample_paths = (paths.DEFAULT_BIOSAMPLE_XML_INPUT,)
    source_contract = validate_paired_source_contract(
        biosample_path=biosample_paths,
        bioproject_path=paths.DEFAULT_BIOPROJECT_XML_INPUT,
        biosample_manifest_path=paths.DEFAULT_BIOSAMPLE_SNAPSHOT_MANIFEST,
        bioproject_manifest_path=paths.DEFAULT_BIOPROJECT_SNAPSHOT_MANIFEST,
    )

    if curation_schema is None:
        curation_schema = CurationSchema.load(config_dir / "curation_schema.yaml")
    counters = CandidateCounters()
    review_reports = ReviewReports()

    pathogen_by_accession = load_pathogen_map(index_path, names)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    catalog_path = bioproject_catalog_path_for(output_path)
    bundle_provenance_path = provenance_path_for(output_path)
    extracted_record_count = 0
    linked_project_samples: dict[str, set[str]] = defaultdict(set)

    with tempfile.TemporaryDirectory(
        prefix=f".{output_path.stem}-", dir=output_path.parent
    ) as temporary_dir:
        temporary_dir_path = Path(temporary_dir)
        spool_path = temporary_dir_path / "biosample_rows.tsv"
        temporary_output_path = temporary_dir_path / output_path.name
        temporary_catalog_path = temporary_dir_path / catalog_path.name
        temporary_provenance_path = temporary_dir_path / bundle_provenance_path.name

        with spool_path.open("w", newline="", encoding="utf-8") as spool_stream:
            writer = csv.writer(spool_stream, delimiter="\t")
            writer.writerow(COLUMNS)
            with make_progress_bar(
                len(biosample_paths), "extracting BioSample XML", disable=disable_progress
            ) as bar:
                for xml_file in biosample_paths:
                    logger.info("Parsing %s...", xml_file)
                    for accession, candidates, bioproject_ids in process_biosample_xml(
                        str(xml_file), curation_schema.evaluate, counters
                    ):
                        for decision in candidates:
                            review_reports.observe(decision, accession=accession)
                        pathogen = pathogen_by_accession.get(accession)
                        if pathogen is None:
                            continue
                        row = record_row(
                            accession=accession,
                            pathogen=pathogen,
                            bioproject_id="||".join(bioproject_ids),
                            bioproject_accession="",
                            candidates=candidates,
                        )
                        if row is not None:
                            writer.writerow(row)
                            extracted_record_count += 1
                            for project_id in bioproject_ids:
                                linked_project_samples[project_id].add(accession)
                    bar.update(1)

        resolved_projects = resolve_bioproject_contexts(
            paths.DEFAULT_BIOPROJECT_XML_INPUT,
            linked_project_samples,
        )
        _write_resolved_rows(spool_path, temporary_output_path, resolved_projects)
        write_bioproject_catalog(resolved_projects.values(), temporary_catalog_path)

        provenance = DerivedBundleProvenance.create(
            source_contract=source_contract,
            biosample_manifest_path=paths.DEFAULT_BIOSAMPLE_SNAPSHOT_MANIFEST,
            bioproject_manifest_path=paths.DEFAULT_BIOPROJECT_SNAPSHOT_MANIFEST,
            extracted_metadata_path=temporary_output_path,
            bioproject_context_path=temporary_catalog_path,
        )
        provenance.write(temporary_provenance_path)

        review_artifact_paths = review_reports.write(output_path.parent)
        unresolved_path = write_unresolved_bioproject_links(
            linked_project_samples,
            resolved_projects.keys(),
            output_path.parent / "unresolved_bioproject_links.tsv",
        )
        if unresolved_path is not None:
            review_artifact_paths["unresolved_bioproject_links"] = unresolved_path
        if review_reports.has_unreviewed:
            logger.warning(
                "Unreviewed metadata attributes were excluded. See unreviewed_attributes.tsv"
            )
        review_reports.log_automatic_rejections(logger)
        logger.info("Curation summary: %s", counters.summary())

        _publish_bundle(
            temporary_output_path=temporary_output_path,
            output_path=output_path,
            temporary_catalog_path=temporary_catalog_path,
            catalog_path=catalog_path,
            temporary_provenance_path=temporary_provenance_path,
            provenance_path=bundle_provenance_path,
        )
    return ExtractionReport(
        source_xml_paths=(*biosample_paths, paths.DEFAULT_BIOPROJECT_XML_INPUT),
        extracted_metadata_path=output_path,
        extracted_record_count=extracted_record_count,
        counters=counters,
        automatic_rejection_counts=review_reports.automatic_rejection_counts,
        unreviewed_count=review_reports.unreviewed_count,
        uncertain_count=review_reports.uncertain_count,
        review_artifact_paths=review_artifact_paths,
        biosample_snapshot_id=source_contract.biosample.snapshot_id,
        bioproject_snapshot_id=source_contract.bioproject.snapshot_id,
        metadata_reference_date=source_contract.metadata_reference_date,
        bundle_provenance_path=bundle_provenance_path,
        bioproject_catalog_path=catalog_path,
    )


def _write_resolved_rows(
    spool_path: Path,
    output_path: Path,
    resolved_projects: dict[str, BioProjectContext],
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
        )
        writer.writeheader()
        for row in reader:
            project_ids = row["bioproject_id"].split("||") if row["bioproject_id"] else ()
            row["bioproject_accession"] = "||".join(
                sorted(
                    resolved_projects[project_id].accession
                    for project_id in project_ids
                    if project_id in resolved_projects
                )
            )
            writer.writerow(row)


def _publish_bundle(
    *,
    temporary_output_path: Path,
    output_path: Path,
    temporary_catalog_path: Path,
    catalog_path: Path,
    temporary_provenance_path: Path,
    provenance_path: Path,
) -> None:
    provenance_path.unlink(missing_ok=True)
    try:
        os.replace(temporary_output_path, output_path)
        os.replace(temporary_catalog_path, catalog_path)
        os.replace(temporary_provenance_path, provenance_path)
    except Exception:
        provenance_path.unlink(missing_ok=True)
        raise


def cli(argv: Sequence[str] | None = None) -> None:
    """Run the extraction command-line interface."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--index",
        default=str(paths.DEFAULT_INDEX_TSV),
        help="TSV mapping accession to pathogen.",
    )
    parser.add_argument("--names", nargs="*")
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument("--quiet", action="store_true", help="Disable progress bars.")
    args = parser.parse_args(argv)

    run_extraction(
        output_path=Path(args.output),
        index_path=Path(args.index),
        names=args.names,
        log_level=args.log_level,
        disable_progress=args.quiet,
    )


if __name__ == "__main__":
    cli()
