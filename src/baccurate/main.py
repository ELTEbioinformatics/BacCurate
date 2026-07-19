"""Main command-line entry point."""

import argparse
import csv
import logging
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from time import monotonic

from baccurate.dataset_builder import (
    DatasetBuilder,
    DatasetBuildProgress,
    DatasetBuildRequest,
    StandardizationAttribute,
)
from baccurate.extraction import run_extraction
from baccurate.llm.client import LLMSettings, load_llm_settings
from baccurate.pathogens import all_keywords, expand_keys, pathogen_keys
from baccurate.paths import (
    CONFIG_DIR,
    DEFAULT_BIOPROJECT_SNAPSHOT_MANIFEST,
    DEFAULT_BIOPROJECT_XML_INPUT,
    DEFAULT_BIOSAMPLE_SNAPSHOT_MANIFEST,
    DEFAULT_BIOSAMPLE_XML_INPUT,
    DEFAULT_EXTRACTED_TSV,
    DEFAULT_INDEX_TSV,
    DEFAULT_NAMES_DMP,
    DEFAULT_NODES_DMP,
    OUTPUT_DIR,
    PATHOGENS_YAML,
)
from baccurate.run_outputs import (
    RunContext,
    RunDiagnostics,
    RunOutputs,
    RunPhase,
    RunStatus,
    processed_rows,
)
from baccurate.utils.compressed_io import open_text
from baccurate.utils.logging import configure_run_logging
from baccurate.utils.progress import progress_context

logger = logging.getLogger(__name__)
pipeline_logger = logging.getLogger("baccurate.pipeline")


STANDARDIZATION_ATTRIBUTES: tuple[StandardizationAttribute, ...] = (
    StandardizationAttribute.HOST,
    StandardizationAttribute.DATE,
    StandardizationAttribute.LOCATION,
    StandardizationAttribute.ISOLATION_SOURCE,
)


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


def _model_identifiers(
    attributes: tuple[StandardizationAttribute, ...],
    settings: LLMSettings,
) -> dict[str, str | None]:
    identifiers = {}
    if StandardizationAttribute.LOCATION in attributes:
        identifiers["location"] = settings.model
    if StandardizationAttribute.ISOLATION_SOURCE in attributes:
        identifiers["isolation"] = settings.model
    return identifiers


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "names",
        nargs="*",
        choices=all_keywords(),
        metavar="PATHOGEN",
        help="One or more target pathogens to process (see config/pathogens.yaml). "
        "Pathogen keys: "
        + ", ".join(pathogen_keys())
        + ". Groups (expand to their pathogen keys): "
        + ", ".join(k for k in all_keywords() if k not in pathogen_keys())
        + ". If omitted, every pathogen is processed.",
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
        default=OUTPUT_DIR,
        help="Base output directory.",
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        default=None,
        help="Final dataset TSV (default: <output-dir>/<run-name>/<run-name>.tsv).",
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help=("Directory name used for the output path. (default: current timestamp)."),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing outputs.",
    )
    parser.add_argument(
        "--skip-llm",
        action="store_true",
        help="Disable LLM client initialization and API calls.",
    )
    parser.add_argument(
        "--attribute",
        nargs="*",
        choices=[attribute.value for attribute in STANDARDIZATION_ATTRIBUTES],
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

    args = parser.parse_args(argv)

    log_level = "DEBUG" if args.debug else "INFO"
    active_attributes = (
        tuple(
            attribute
            for attribute in STANDARDIZATION_ATTRIBUTES
            if attribute.value in args.attribute
        )
        if args.attribute
        else STANDARDIZATION_ATTRIBUTES
    )

    run_name = args.run_name or datetime.now().strftime("%Y-%m-%d_%H-%M")
    try:
        outputs = RunOutputs.plan(
            output_dir=args.output_dir,
            run_name=run_name,
            output_file=args.output_file,
            include_isolation=StandardizationAttribute.ISOLATION_SOURCE in active_attributes,
        )
    except ValueError as exc:
        parser.error(str(exc))
    collision = outputs.collision()
    if collision is not None and not args.force:
        parser.error(f"output already exists: {collision}. Pass --force to replace it.")

    biosample_input_path = DEFAULT_BIOSAMPLE_XML_INPUT
    index_path = DEFAULT_INDEX_TSV

    names = expand_keys(args.names) if args.names else _discover_pathogens(index_path)

    # Default location -> extract all pathogens; explicit path -> extract only the named ones.
    if args.extracted_metadata is None:
        extracted_metadata_path = DEFAULT_EXTRACTED_TSV
        extraction_names = None
    else:
        extracted_metadata_path = args.extracted_metadata
        extraction_names = names

    disable_progress = args.quiet
    extraction_required = not extracted_metadata_path.exists()
    configuration_paths = [
        PATHOGENS_YAML,
        DEFAULT_BIOSAMPLE_SNAPSHOT_MANIFEST,
        DEFAULT_BIOPROJECT_SNAPSHOT_MANIFEST,
    ]
    if extraction_required:
        configuration_paths.append(args.config_dir / "curation_schema.yaml")
    if StandardizationAttribute.HOST in active_attributes:
        configuration_paths.append(args.config_dir / "host.yaml")
    if StandardizationAttribute.LOCATION in active_attributes:
        configuration_paths.append(args.config_dir / "location.yaml")
    if StandardizationAttribute.ISOLATION_SOURCE in active_attributes:
        configuration_paths.append(args.config_dir / "isolation_source.yaml")
    normalized_options = {
        "pathogens": list(names),
        "attributes": [attribute.value for attribute in active_attributes],
        "config_dir": str(args.config_dir),
        "output_dir": str(args.output_dir),
        "output_file": str(args.output_file) if args.output_file is not None else None,
        "run_name": run_name,
        "force": args.force,
        "skip_llm": args.skip_llm,
        "extracted_metadata": str(extracted_metadata_path),
        "names_dmp": str(args.names_dmp),
        "nodes_dmp": str(args.nodes_dmp),
        "debug": args.debug,
        "quiet": args.quiet,
    }
    llm_settings = load_llm_settings()
    outputs.initialize()
    logging_state = configure_run_logging(outputs.log, console_debug=args.debug)
    diagnostics = RunDiagnostics(
        outputs,
        RunContext(
            requested_pathogens=tuple(names),
            requested_attributes=tuple(attribute.value for attribute in active_attributes),
            extracted_metadata=extracted_metadata_path,
            options=normalized_options,
            configuration_paths=tuple(configuration_paths),
            skip_llm=args.skip_llm,
            model_identifiers=_model_identifiers(active_attributes, llm_settings),
            trace_llm_calls=args.debug,
        ),
    )
    started = monotonic()
    logger.info(
        "Run started: pathogens=%s attributes=%s",
        ",".join(names),
        ",".join(attribute.value for attribute in active_attributes),
    )
    build_progress = DatasetBuildProgress()

    try:
        with progress_context(disable=disable_progress):
            if extraction_required:
                diagnostics.transition(RunPhase.EXTRACTION)
                diagnostics.begin_performed_extraction(
                    source_xml_path=biosample_input_path,
                    bioproject_source_xml_path=DEFAULT_BIOPROJECT_XML_INPUT,
                    extracted_metadata_path=extracted_metadata_path,
                    source_manifest_path=DEFAULT_BIOSAMPLE_SNAPSHOT_MANIFEST,
                    bioproject_manifest_path=DEFAULT_BIOPROJECT_SNAPSHOT_MANIFEST,
                )
                extraction_started = monotonic()
                logger.info(
                    "Extraction started: input=%s output=%s",
                    biosample_input_path,
                    extracted_metadata_path,
                )
                extraction_report = run_extraction(
                    output_path=extracted_metadata_path,
                    index_path=index_path,
                    names=extraction_names,
                    log_level=log_level,
                    config_dir=args.config_dir,
                    disable_progress=disable_progress,
                )
                extraction_elapsed = monotonic() - extraction_started
                diagnostics.record_performed_extraction(
                    extraction_report,
                    elapsed_seconds=extraction_elapsed,
                )
                diagnostics.transition(RunPhase.DATASET_STREAMING)
                logger.info("Extraction finished: elapsed=%.2fs", extraction_elapsed)
            else:
                diagnostics.record_reused_extraction(
                    extracted_metadata_path=extracted_metadata_path,
                    source_manifest_path=DEFAULT_BIOSAMPLE_SNAPSHOT_MANIFEST,
                    bioproject_manifest_path=DEFAULT_BIOPROJECT_SNAPSHOT_MANIFEST,
                )
                diagnostics.transition(RunPhase.DATASET_STREAMING)
                logger.info("Using extracted metadata from: %s", extracted_metadata_path)

            logger.info("Streaming started")
            request = DatasetBuildRequest(
                extracted_metadata=extracted_metadata_path,
                source_snapshot_manifest=DEFAULT_BIOSAMPLE_SNAPSHOT_MANIFEST,
                requested_pathogens=tuple(names),
                requested_attributes=active_attributes,
                final_destination=outputs.dataset,
                atb_index=index_path,
                location_config=args.config_dir / "location.yaml",
                host_config=args.config_dir / "host.yaml",
                isolation_config=args.config_dir / "isolation_source.yaml",
                isolation_reasoning_destination=outputs.isolation_reasoning,
                names_dmp=args.names_dmp,
                nodes_dmp=args.nodes_dmp,
                overwrite=True,
                skip_llm=args.skip_llm,
                llm_settings=llm_settings,
                logger=pipeline_logger,
                disable_progress=disable_progress,
                progress=build_progress,
            )
            report = DatasetBuilder().build(request)

        processed = processed_rows(report)
        logger.info(
            "Streaming finished: processed=%d accepted=%d",
            processed,
            report.rows_written,
        )
        diagnostics.finish(
            RunStatus.SUCCEEDED,
            progress=build_progress,
            report=report,
        )
        logger.info("Diagnostics finalized: %s", outputs.diagnostics)
        logger.info(
            "Run completed: elapsed=%.2fs rows=%d outputs=%s",
            monotonic() - started,
            report.rows_written,
            ",".join(str(path) for path in outputs.paths()),
        )
    except KeyboardInterrupt as exc:
        diagnostics.finish(
            RunStatus.INTERRUPTED,
            error=exc,
            progress=build_progress,
            report=build_progress.report,
        )
        logger.exception("Run interrupted")
        raise SystemExit(130) from None
    except Exception as exc:
        diagnostics.finish(
            RunStatus.FAILED,
            error=exc,
            progress=build_progress,
            report=build_progress.report,
        )
        logger.exception("Run failed")
        raise SystemExit(1) from None
    finally:
        logging_state.close()


if __name__ == "__main__":
    main()
