"""CLI entry point for metadata extraction stage."""

import argparse
import csv
import logging
from pathlib import Path

from baccurate.extraction._review_reports import ReviewReports
from baccurate.extraction.io import load_pathogen_map, resolve_input_files
from baccurate.extraction.policy import ExtractionPolicy
from baccurate.extraction.tables import COLUMNS, record_row
from baccurate.paths import CONFIG_DIR, DEFAULT_INDEX_TSV
from baccurate.utils.progress import make_inner_bar
from baccurate.utils.xml import CandidateCounters, process_biosample_xml

logger = logging.getLogger(__name__)


def main(
    input_path: Path,
    output_path: Path,
    index_path: Path = DEFAULT_INDEX_TSV,
    names: list[str] | None = None,
    log_level: str = "INFO",
    config_dir: Path = CONFIG_DIR,
    disable_progress: bool = False,
    uncompressed: bool = False,
    policy: ExtractionPolicy | None = None,
) -> CandidateCounters:
    logging.basicConfig(
        level=log_level.upper(),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    if policy is None:
        policy = ExtractionPolicy.load(config_dir / "metadata_matching.yaml")
    counters = CandidateCounters()
    review_reports = ReviewReports()

    pathogen_by_accession = load_pathogen_map(index_path, names)

    files = resolve_input_files(input_path, uncompressed=uncompressed)
    with output_path.open("w", newline="", encoding="utf-8") as output_stream:
        writer = csv.writer(output_stream, delimiter="\t")
        writer.writerow(COLUMNS)
        with make_inner_bar(
            len(files), "extracting BioSample XML", disable=disable_progress
        ) as bar:
            for xml_file in files:
                logger.info("Parsing %s...", xml_file)
                try:
                    for accession, candidates, package, bioproject in process_biosample_xml(
                        str(xml_file), policy.evaluate, counters
                    ):
                        for decision in candidates:
                            review_reports.observe(decision, accession=accession)
                        pathogen = pathogen_by_accession.get(accession)
                        if pathogen is None:
                            continue
                        row = record_row(
                            accession=accession,
                            pathogen=pathogen,
                            package=package,
                            bioproject=bioproject,
                            candidates=candidates,
                        )
                        if row is not None:
                            writer.writerow(row)
                except Exception:
                    logger.exception("Failed to parse %s - skipping", xml_file)
                bar.update(1)

    review_reports.write(output_path.parent)
    if review_reports.has_unreviewed:
        logger.warning(
            "Unreviewed metadata attributes were excluded. See unreviewed_attributes.tsv"
        )
    review_reports.log_automatic_rejections(logger)
    logger.info("Candidate policy summary: %s", counters.summary())
    return counters

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="BioSample XML file.")
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--index", default=str(DEFAULT_INDEX_TSV), help="TSV mapping accession -> pathogen)."
    )
    parser.add_argument("--names", nargs="*")
    parser.add_argument("--log-level", default="INFO")
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
    )
