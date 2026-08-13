"""CLI entry point for metadata extraction stage."""

import argparse
import csv
import logging
import tempfile
from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

from baccurate import paths
from baccurate.adapters.progress import make_progress_bar
from baccurate.extraction.bioproject import (
    resolve_bioproject_accessions,
    write_unresolved_bioproject_links,
)
from baccurate.extraction.curation import CurationSchema
from baccurate.extraction.io import InclusionRoute, load_pathogen_map
from baccurate.extraction.manual_review import ReviewWorklists
from baccurate.extraction.tables import COLUMNS, extracted_metadata_row
from baccurate.extraction.xml import CurationCounters, process_biosample_xml
from baccurate.pathogen_registry.registry import PathogenRegistry, load_pathogen_registry
from baccurate.provenance.source_snapshot import (
    DerivedBundleProvenance,
    _publish_bundle,
    provenance_path_for,
    validate_paired_source_contract,
)
from baccurate.standardization_target.policy_slot import POLICY_FILENAMES, PolicySlot

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class ExtractionReport:
    """Data provenance and a curation summary of extraction."""

    prepared_input_paths: tuple[Path, ...]
    extracted_metadata_path: Path
    extracted_record_count: int
    counters: CurationCounters
    automatic_rejection_counts: dict[str, dict[str, int]]
    unreviewed_count: int
    uncertain_count: int
    review_worklist_paths: dict[str, Path]
    biosample_snapshot_id: str
    bioproject_snapshot_id: str
    metadata_reference_date: date
    bundle_provenance_path: Path
    inclusion_route_counts: dict[InclusionRoute, int] = field(default_factory=dict)


def run_extraction(
    output_path: Path,
    index_path: Path = paths.DEFAULT_INDEX_TSV,
    names: list[str] | None = None,
    log_level: str = "INFO",
    disable_progress: bool = False,
    *,
    curation_schema: CurationSchema,
    pathogen_registry: PathogenRegistry | None = None,
) -> ExtractionReport:
    logging.basicConfig(
        level=log_level.upper(),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    biosample_paths = (paths.DEFAULT_BIOSAMPLE_XML_INPUT,)
    source_contract = validate_paired_source_contract(
        biosample_path=paths.DEFAULT_BIOSAMPLE_XML_INPUT,
        bioproject_path=paths.DEFAULT_BIOPROJECT_XML_INPUT,
        biosample_manifest_path=paths.DEFAULT_BIOSAMPLE_SNAPSHOT_MANIFEST,
        bioproject_manifest_path=paths.DEFAULT_BIOPROJECT_SNAPSHOT_MANIFEST,
    )

    counters = CurationCounters()
    review_worklists = ReviewWorklists()

    registry = pathogen_registry or load_pathogen_registry()
    target_pathogen_assignment_by_accession = load_pathogen_map(
        index_path, registry.pathogen_keys, names
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    bundle_provenance_path = provenance_path_for(output_path)
    extracted_record_count = 0
    inclusion_route_counts: Counter[InclusionRoute] = Counter()
    linked_project_samples: dict[str, set[str]] = defaultdict(set)

    with tempfile.TemporaryDirectory(
        prefix=f".{output_path.stem}-", dir=output_path.parent
    ) as temporary_dir:
        temporary_dir_path = Path(temporary_dir)
        spool_path = temporary_dir_path / "biosample_rows.tsv"
        temporary_output_path = temporary_dir_path / output_path.name
        temporary_provenance_path = temporary_dir_path / bundle_provenance_path.name

        with spool_path.open("w", newline="", encoding="utf-8") as spool_stream:
            writer = csv.writer(spool_stream, delimiter="\t")
            writer.writerow(COLUMNS)
            with make_progress_bar(
                len(biosample_paths), "extracting BioSample XML", disable=disable_progress
            ) as bar:
                for xml_file in biosample_paths:
                    logger.info("Parsing %s...", xml_file)
                    for accession, decisions, bioproject_ids in process_biosample_xml(
                        str(xml_file), curation_schema.evaluate, counters
                    ):
                        for decision in decisions:
                            review_worklists.observe(decision, accession=accession)
                        assignment = target_pathogen_assignment_by_accession.get(accession)
                        if assignment is None:
                            continue
                        extracted_metadata_values = extracted_metadata_row(
                            accession=accession,
                            pathogen=assignment.pathogen_key,
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
                    bar.update(1)

        accession_by_bioproject_id = resolve_bioproject_accessions(
            paths.DEFAULT_BIOPROJECT_XML_INPUT,
            linked_project_samples,
        )
        _write_resolved_rows(spool_path, temporary_output_path, accession_by_bioproject_id)

        provenance = DerivedBundleProvenance.create(
            source_contract=source_contract,
            biosample_manifest_path=paths.DEFAULT_BIOSAMPLE_SNAPSHOT_MANIFEST,
            bioproject_manifest_path=paths.DEFAULT_BIOPROJECT_SNAPSHOT_MANIFEST,
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
        logger.info("Curation summary: %s", counters.summary())

        _publish_bundle(
            temporary_output_path=temporary_output_path,
            output_path=output_path,
            temporary_provenance_path=temporary_provenance_path,
            provenance_path=bundle_provenance_path,
        )
    return ExtractionReport(
        prepared_input_paths=(*biosample_paths, paths.DEFAULT_BIOPROJECT_XML_INPUT),
        extracted_metadata_path=output_path,
        extracted_record_count=extracted_record_count,
        counters=counters,
        automatic_rejection_counts=review_worklists.automatic_rejection_counts,
        unreviewed_count=review_worklists.unreviewed_count,
        uncertain_count=review_worklists.uncertain_count,
        review_worklist_paths=review_worklist_paths,
        biosample_snapshot_id=source_contract.biosample.snapshot_id,
        bioproject_snapshot_id=source_contract.bioproject.snapshot_id,
        metadata_reference_date=source_contract.metadata_reference_date,
        bundle_provenance_path=bundle_provenance_path,
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
    curation_schema = CurationSchema.load(
        paths.CONFIG_DIR / POLICY_FILENAMES[PolicySlot.CURATION_SCHEMA]
    )

    run_extraction(
        output_path=Path(args.output),
        curation_schema=curation_schema,
        index_path=Path(args.index),
        names=args.names,
        log_level=args.log_level,
        disable_progress=args.quiet,
    )


if __name__ == "__main__":
    cli()
