"""CLI entry point for metadata extraction stage."""

import argparse
import csv
import logging
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from baccurate.extraction._review_reports import ReviewReports
from baccurate.extraction.curation import CurationSchema
from baccurate.extraction.io import load_pathogen_map, resolve_input_files
from baccurate.extraction.tables import COLUMNS, record_row
from baccurate.extraction.xml import CandidateCounters, process_biosample_xml
from baccurate.paths import (
    CONFIG_DIR,
    DEFAULT_BIOPROJECT_SNAPSHOT_MANIFEST,
    DEFAULT_BIOPROJECT_XML_INPUT,
    DEFAULT_BIOSAMPLE_SNAPSHOT_MANIFEST,
    DEFAULT_INDEX_TSV,
)
from baccurate.source_snapshot import (
    DerivedSourceRecord,
    provenance_path_for,
    validate_paired_source_contract,
)
from baccurate.utils.progress import make_progress_bar

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class ExtractionReport:
    """What one extraction produced: data provenance and a curation summary.

    Holds the raw inputs and derived-source path, how many records were
    written, the candidate curation counts, and the review artifacts.
    """

    source_xml_paths: tuple[Path, ...]
    extracted_metadata_path: Path
    extracted_record_count: int
    counters: CandidateCounters
    automatic_rejection_counts: dict[str, dict[str, int]]
    unreviewed_count: int
    uncertain_count: int
    review_artifact_paths: dict[str, Path]
    source_snapshot_id: str
    bioproject_snapshot_id: str
    metadata_reference_date: date
    source_record_path: Path

    @property
    def biosample_snapshot_id(self) -> str:
        return self.source_snapshot_id


def main(
    input_path: Path,
    output_path: Path,
    index_path: Path = DEFAULT_INDEX_TSV,
    names: list[str] | None = None,
    log_level: str = "INFO",
    config_dir: Path = CONFIG_DIR,
    disable_progress: bool = False,
    uncompressed: bool = False,
    curation_schema: CurationSchema | None = None,
    source_manifest_path: Path = DEFAULT_BIOSAMPLE_SNAPSHOT_MANIFEST,
    bioproject_input_path: Path = DEFAULT_BIOPROJECT_XML_INPUT,
    bioproject_manifest_path: Path = DEFAULT_BIOPROJECT_SNAPSHOT_MANIFEST,
) -> ExtractionReport:
    logging.basicConfig(
        level=log_level.upper(),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    files = resolve_input_files(input_path, uncompressed=uncompressed)
    source_contract = validate_paired_source_contract(
        biosample_path=files,
        bioproject_path=bioproject_input_path,
        biosample_manifest_path=source_manifest_path,
        bioproject_manifest_path=bioproject_manifest_path,
    )

    if curation_schema is None:
        curation_schema = CurationSchema.load(config_dir / "curation_schema.yaml")
    counters = CandidateCounters()
    review_reports = ReviewReports()

    pathogen_by_accession = load_pathogen_map(index_path, names)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    provenance_path_for(output_path).unlink(missing_ok=True)
    extracted_record_count = 0
    with output_path.open("w", newline="", encoding="utf-8") as output_stream:
        writer = csv.writer(output_stream, delimiter="\t")
        writer.writerow(COLUMNS)
        with make_progress_bar(
            len(files), "extracting BioSample XML", disable=disable_progress
        ) as bar:
            for xml_file in files:
                logger.info("Parsing %s...", xml_file)
                for accession, candidates, bioproject in process_biosample_xml(
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
                        bioproject=bioproject,
                        candidates=candidates,
                    )
                    if row is not None:
                        writer.writerow(row)
                        extracted_record_count += 1
                bar.update(1)

    review_artifact_paths = review_reports.write(output_path.parent)
    if review_reports.has_unreviewed:
        logger.warning(
            "Unreviewed metadata attributes were excluded. See unreviewed_attributes.tsv"
        )
    review_reports.log_automatic_rejections(logger)
    logger.info("Curation summary: %s", counters.summary())
    DerivedSourceRecord.from_manifest(source_contract.biosample, source_manifest_path).write_for(
        output_path
    )
    return ExtractionReport(
        source_xml_paths=(*files, bioproject_input_path),
        extracted_metadata_path=output_path,
        extracted_record_count=extracted_record_count,
        counters=counters,
        automatic_rejection_counts=review_reports.automatic_rejection_counts,
        unreviewed_count=review_reports.unreviewed_count,
        uncertain_count=review_reports.uncertain_count,
        review_artifact_paths=review_artifact_paths,
        source_snapshot_id=source_contract.biosample.snapshot_id,
        bioproject_snapshot_id=source_contract.bioproject.snapshot_id,
        metadata_reference_date=source_contract.metadata_reference_date,
        source_record_path=provenance_path_for(output_path),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="BioSample XML file.")
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--index", default=str(DEFAULT_INDEX_TSV), help="TSV mapping accession -> pathogen)."
    )
    parser.add_argument("--names", nargs="*")
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument(
        "--source-manifest",
        type=Path,
        default=DEFAULT_BIOSAMPLE_SNAPSHOT_MANIFEST,
        help="Manifest identifying the raw BioSample snapshot.",
    )
    parser.add_argument("--quiet", action="store_true", help="Disable progress bars.")
    parser.add_argument(
        "--uncompressed",
        action="store_true",
        help="Discover uncompressed .xml files when --input is a directory.",
    )
    args = parser.parse_args()

    main(
        Path(args.input),
        Path(args.output),
        index_path=Path(args.index),
        names=args.names,
        log_level=args.log_level,
        disable_progress=args.quiet,
        uncompressed=args.uncompressed,
        source_manifest_path=args.source_manifest,
    )
