"""
Main pipeline entry point: extracts BioSample metadata, runs the per-attribute
standardizers per pathogen, then merges the results and adds host lineage columns.
"""

import argparse
import csv
import logging
import sys
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from baccurate.extraction import main as run_extraction
from baccurate.pathogens import all_keywords, expand_keys, pathogen_keys
from baccurate.paths import (
    CONFIG_DIR,
    DATE_OUTPUT,
    DEFAULT_EXTRACTED_TSV,
    DEFAULT_NAMES_DMP,
    DEFAULT_NODES_DMP,
    DEFAULT_PATHOGEN_OUTPUT_DIR,
    DEFAULT_XML_INPUT,
    HOST_OUTPUT,
    HOST_OVERFLOW,
    ISO_OUTPUT,
    LOC_OUTPUT,
    MERGED_OUTPUT,
    raw_input_paths,
)
from baccurate.postprocess.merge_results import merge_results
from baccurate.standardizers import date as date_std
from baccurate.standardizers import host as host_std
from baccurate.standardizers import isolation as iso_std
from baccurate.standardizers import location as loc_std
from baccurate.standardizers.host import HostStandardizer
from baccurate.taxonomy.lineage import add_lineage_columns
from baccurate.utils.compressed_io import open_text
from baccurate.utils.progress import make_outer_bar, progress_context

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class StandardizerSpec:
    name: str
    config: str
    runner: Callable
    output: str


# The order here determines processing order
PIPELINES: tuple[StandardizerSpec, ...] = (
    StandardizerSpec("host", "host.yaml", host_std.main, HOST_OUTPUT),
    StandardizerSpec("date", "date.yaml", date_std.main, DATE_OUTPUT),
    StandardizerSpec("loc", "location.yaml", loc_std.main, LOC_OUTPUT),
    StandardizerSpec("iso", "isolation_source.yaml", iso_std.main, ISO_OUTPUT),
)


HOST_RETRY_TRIGGERS: tuple[str, ...] = (
    "host-associated",
    "environmental:anthropogenic environment:food:animal product",
    "environmental:anthropogenic environment:food:plant food product",
)


def _iter_iso_for_host_retry(
    iso_output_path: Path,
) -> Iterator[tuple[str, str, str]]:
    """Yield (accession, attribute, value) for iso rows that need a host retry.

    A row qualifies when the host context column is empty,and the iso term path names
    a resolvable source organism (see HOST_RETRY_TRIGGERS).
    """
    with iso_output_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            term_path = row.get("iso_terms") or ""
            host = (row.get("iso_host") or "").strip()
            if host:
                continue
            if not any(trigger in term_path for trigger in HOST_RETRY_TRIGGERS):
                continue
            yield (
                row.get("accession", ""),
                row.get("iso_attr_orig", "") or "",
                row.get("iso_val_orig", "") or "",
            )


def _append_host_matches(
    host_standardizer: HostStandardizer,
    rows: Iterator[tuple[str, str, str]],
    host_output_path: Path,
) -> int:
    """
    Run host standardizer on `rows` and append matches to host_output_path.

    Returns the number of rows appended.
    """
    appended = 0
    with host_output_path.open("a", encoding="utf-8", newline="") as out:
        writer = csv.writer(out, delimiter="\t", quoting=csv.QUOTE_NONE, escapechar="\\")
        for accession, match in host_standardizer.classify_values(rows, skip_iso_keywords=True):
            writer.writerow(
                [
                    accession,
                    match.info.taxid,
                    match.info.scientific_name,
                    match.score,
                    match.low_confidence,
                    match.attribute,
                    match.value,
                ]
            )
            appended += 1
    return appended


def _run_host_iso_combined(
    name: str,
    pathogen_output_dir: Path,
    config_dir: Path,
    log_level: str,
    host_spec: StandardizerSpec,
    iso_spec: StandardizerSpec,
    extracted_metadata_path: Path,
    outer_bar=None,
    disable_progress: bool = False,
) -> None:
    """host standardizer -> isolation standardizer (reading host's overflow) -> host retry."""
    host_output = pathogen_output_dir / host_spec.output
    iso_output = pathogen_output_dir / iso_spec.output
    overflow_path = pathogen_output_dir / HOST_OVERFLOW

    # Stage 1: host classifies extracted_metadata; failures land in overflow.
    logger.info("Stage 1/3 (host) pathogen=%s -> %s", name, host_output)
    host_spec.runner(
        input_path=extracted_metadata_path,
        output_path=host_output,
        config_path=config_dir / host_spec.config,
        log_level=log_level,
        pathogen=name,
        overflow_path=overflow_path,
        disable_progress=disable_progress,
    )
    if outer_bar is not None:
        outer_bar.update(1)

    # Stage 2: iso classifies extracted_metadata AND host's overflow.
    logger.info("Stage 2/3 (iso) pathogen=%s overflow=%s -> %s", name, overflow_path, iso_output)
    iso_spec.runner(
        input_path=extracted_metadata_path,
        output_path=iso_output,
        config_path=config_dir / iso_spec.config,
        log_level=log_level,
        pathogen=name,
        additional_input=overflow_path,
        host_standardized_path=host_output,
        disable_progress=disable_progress,
    )

    # Stage 3: retry host on iso rows that look host-associated but lack a host.
    logger.info("Stage 3/3 (host retry on iso output) pathogen=%s", name)
    retry_standardizer = HostStandardizer(config_dir / host_spec.config)
    appended = _append_host_matches(
        retry_standardizer,
        _iter_iso_for_host_retry(iso_output),
        host_output,
    )
    logger.info("Stage 3/3 appended %d host rows from iso retry", appended)
    if outer_bar is not None:
        outer_bar.update(1)


def process_pathogen(
    name: str,
    output_dir: Path,
    config_dir: Path,
    log_level: str,
    pipelines: tuple[StandardizerSpec, ...],
    extracted_metadata_path: Path,
    outer_bar=None,
    disable_progress: bool = False,
) -> None:
    """Run every requested pipeline for a single pathogen."""
    pathogen_output_dir = output_dir / name
    pathogen_output_dir.mkdir(parents=True, exist_ok=True)

    by_name = {p.name: p for p in pipelines}
    remaining = list(pipelines)

    # host and iso cooperate via files: host writes an overflow file that
    # iso also reads, and we retry host on iso rows flagged host-associated.
    if "host" in by_name and "iso" in by_name:
        _run_host_iso_combined(
            name,
            pathogen_output_dir,
            config_dir,
            log_level,
            by_name["host"],
            by_name["iso"],
            extracted_metadata_path,
            outer_bar=outer_bar,
            disable_progress=disable_progress,
        )
        remaining = [p for p in remaining if p.name not in ("host", "iso")]

    for spec in remaining:
        output_path = pathogen_output_dir / spec.output
        config_path = config_dir / spec.config
        logger.info(
            "Executing: %s pathogen=%s input=%s output=%s config=%s log_level=%s",
            spec.name,
            name,
            extracted_metadata_path,
            output_path,
            config_path,
            log_level,
        )
        spec.runner(
            input_path=extracted_metadata_path,
            output_path=output_path,
            config_path=config_path,
            log_level=log_level,
            pathogen=name,
            disable_progress=disable_progress,
        )
        if outer_bar is not None:
            outer_bar.update(1)


def _check_output_collisions(
    names: list[str],
    run_base: Path,
    active_pipelines: tuple[StandardizerSpec, ...],
    merged_path: Path,
    force: bool,
) -> None:
    """Abort the run if any standardizer or merged output files already exists, unless `force`."""
    if force:
        return
    existing = [
        path
        for name in names
        for spec in active_pipelines
        if (path := run_base / name / spec.output).exists()
    ]
    if merged_path.exists():
        existing.append(merged_path)
    if existing:
        listing = "\n".join(f"  {p}" for p in existing)
        logger.error(
            "Refusing to overwrite %d existing output(s):\n%s\n"
            "Pass --force to overwrite, or --run-name NAME to write to a different location.",
            len(existing),
            listing,
        )
        sys.exit(1)


def _discover_pathogens(index_path: Path) -> list[str]:
    keys = set(pathogen_keys())
    found = set()
    with open_text(index_path, newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            pathogen = (row.get("pathogen_biosample") or "").strip()
            if pathogen in keys:
                found.add(pathogen)
    return sorted(found)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "names",
        nargs="*",
        choices=all_keywords(),
        metavar="PATHOGEN",
        help="One or more target pathogens to process (see config/pathogens.yaml). "
        "Pathogen keys: " + ", ".join(pathogen_keys())
        + ". Groups (expand to their pathogen keys): "
        + ", ".join(k for k in all_keywords() if k not in pathogen_keys())
        + ". If omitted, every pathogen is processed.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help=f"BioSample XML to process (default: {DEFAULT_XML_INPUT}).",
    )
    parser.add_argument(
        "--uncompressed",
        action="store_true",
        help="Use uncompressed .xml and .tsv inputs instead of .gz files.",
    )
    parser.add_argument(
        "--config-dir",
        type=Path,
        default=CONFIG_DIR,
        help="Directory containing YAML configs.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_PATHOGEN_OUTPUT_DIR,
        help="Base output directory.",
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        default=None,
        help="Filepath for the merged TSV (default: data/processed/<run-name>_merged.tsv).",
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help=(
            "The current run's per-pathogen and merged output file's names."
            "(default: current timestamp, e.g. 2026-06-18_14-30)."
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing outputs.",
    )
    parser.add_argument(
        "--attribute",
        nargs="*",
        choices=[p.name for p in PIPELINES],
        help="Specific pipelines to run. If omitted, all pipelines are run.",
    )
    parser.add_argument(
        "--extracted-metadata",
        type=Path,
        default=None,
        help=f"Path for the extracted metadata TSV (default: {DEFAULT_EXTRACTED_TSV}). ",
    )
    parser.add_argument(
        "--names-dmp",
        type=Path,
        default=DEFAULT_NAMES_DMP,
        help="NCBI names.dmp path for host lineage data generation.",
    )
    parser.add_argument(
        "--nodes-dmp",
        type=Path,
        default=DEFAULT_NODES_DMP,
        help="NCBI nodes.dmp path for host lineage data generation.",
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug logging.")
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Disable progress bars (auto-disabled on non-TTY anyway).",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = "DEBUG" if args.debug else "INFO"

    raw_inputs = raw_input_paths(uncompressed=args.uncompressed)
    input_path = args.input if args.input is not None else raw_inputs.xml
    index_path = raw_inputs.index

    names = expand_keys(args.names) if args.names else _discover_pathogens(index_path)

    if args.attribute:
        active_pipelines = tuple(p for p in PIPELINES if p.name in args.attribute)
    else:
        active_pipelines = PIPELINES

    run_name = args.run_name or datetime.now().strftime("%Y-%m-%d_%H-%M")
    run_base = args.output_dir / run_name
    merged_path = args.output_file if args.output_file is not None else run_base / MERGED_OUTPUT
    _check_output_collisions(names, run_base, active_pipelines, merged_path, force=args.force)

    # Default location -> extract all pathogens; explicit path -> extract only the named ones.
    if args.extracted_metadata is None:
        extracted_metadata_path = DEFAULT_EXTRACTED_TSV
        extraction_names = None
    else:
        extracted_metadata_path = args.extracted_metadata
        extraction_names = names

    disable_progress = args.quiet

    with progress_context(disable=disable_progress):
        if not extracted_metadata_path.exists():
            logger.info(
                "Executing extraction: input=%s output=%s",
                input_path,
                extracted_metadata_path,
            )
            run_extraction(
                input_path=input_path,
                output_path=extracted_metadata_path,
                index_path=index_path,
                names=extraction_names,
                log_level=log_level,
                disable_progress=disable_progress,
                uncompressed=args.uncompressed,
            )

        total_slots = len(names) * len(active_pipelines)
        with make_outer_bar(total_slots, disable=disable_progress) as outer_bar:
            for name in names:
                process_pathogen(
                    name,
                    run_base,
                    args.config_dir,
                    log_level,
                    active_pipelines,
                    extracted_metadata_path,
                    outer_bar=outer_bar,
                    disable_progress=disable_progress,
                )

    merge_results(
        names,
        run_base,
        index_path,
        merged_path,
        [{"name": p.name, "output": p.output} for p in active_pipelines],
        extracted_metadata_path=extracted_metadata_path,
    )

    if any(p.name == "host" for p in active_pipelines):
        add_lineage_columns(
            merged_path,
            args.names_dmp,
            args.nodes_dmp,
        )


if __name__ == "__main__":
    main()
