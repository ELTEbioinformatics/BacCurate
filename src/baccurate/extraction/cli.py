"""CLI entry point for the BioSample metadata extraction stage.

Scans BioSample XML files and classifies each attribute as host, isolation
source, location, or date, emitting a per-accession TSV and per-pathogen HTML
reports.
"""

import argparse
import logging
from pathlib import Path

from baccurate.extraction.classifiers import build_metadata_classifier
from baccurate.extraction.io import load_pathogen_map, resolve_input_files, write_tsv
from baccurate.extraction.tables import DistributionTable, RecordTable
from baccurate.paths import CONFIG_DIR, DEFAULT_INDEX_TSV
from baccurate.utils.progress import make_inner_bar
from baccurate.utils.xml import process_biosample_xml

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
) -> None:
    logging.basicConfig(
        level=log_level.upper(),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    classifier = build_metadata_classifier(config_dir)
    records = RecordTable()
    distribution = DistributionTable()

    pathogen_by_accession = load_pathogen_map(index_path, names)

    files = resolve_input_files(input_path, uncompressed=uncompressed)
    with make_inner_bar(len(files), "extracting BioSample XML", disable=disable_progress) as bar:
        for xml_file in files:
            logger.info("Parsing %s...", xml_file)
            try:
                for accession, attrs, package, bioproject in process_biosample_xml(
                    str(xml_file), lambda v, a: bool(classifier.classify(v, a))
                ):
                    pathogen = pathogen_by_accession.get(accession)
                    if pathogen is None:
                        continue
                    for attr in attrs:
                        matches = classifier.classify(attr["value"], attr["attribute"])
                        if not matches:
                            continue
                        records.add(accession, pathogen, package, bioproject, attr, matches)
                        distribution.add(pathogen, attr, matches)
            except Exception:
                logger.exception("Failed to parse %s - skipping", xml_file)
            bar.update(1)

    write_tsv(records, output_path)

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
