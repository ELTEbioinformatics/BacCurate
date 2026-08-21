"""Main command-line entry point."""

import logging
from collections.abc import Sequence
from time import monotonic

from baccurate.adapters.progress import progress_context
from baccurate.extraction import run_extraction
from baccurate.paths import DEFAULT_BIOPROJECT_XML_INPUT
from baccurate.run.dataset_builder import DatasetBuilder, DatasetBuildRequest
from baccurate.run.invocation import resolve_invocation
from baccurate.run.logging import LIFECYCLE_LOGGER_NAME, configure_run_logging
from baccurate.run.prompt_snapshot import write_prompt_snapshot
from baccurate.run.report import RunContext, RunPhase, RunReport, RunStatus
from baccurate.run.statistics import DatasetBuildProgress, processed_rows
from baccurate.taxon_registry.registry import TaxonRegistry

logger = logging.getLogger(LIFECYCLE_LOGGER_NAME)
pipeline_logger = logging.getLogger("baccurate.pipeline")


def main(
    argv: Sequence[str] | None = None,
    *,
    taxon_registry: TaxonRegistry | None = None,
) -> None:
    invocation = resolve_invocation(argv, taxon_registry=taxon_registry)
    args = invocation
    registry = invocation.registry
    log_level = invocation.log_level
    active_targets = invocation.active_targets
    extracted_metadata_path = invocation.extracted_metadata_path
    effective_policy = invocation.effective_policy
    curation_schema = invocation.curation_schema
    outputs = invocation.outputs
    biosample_input_path = invocation.biosample_input_path
    index_path = invocation.index_path
    names = invocation.taxon_keys
    extraction_names = invocation.extraction_taxon_keys
    disable_progress = invocation.disable_progress
    configuration_paths = invocation.configuration_paths
    normalized_options = invocation.normalized_options
    llm_settings = invocation.llm_settings
    model_identifiers = invocation.model_identifiers
    DEFAULT_BIOSAMPLE_SNAPSHOT_MANIFEST = invocation.biosample_snapshot_manifest
    DEFAULT_BIOPROJECT_SNAPSHOT_MANIFEST = invocation.bioproject_snapshot_manifest

    outputs.initialize()
    if outputs.prompt_snapshot is not None:
        write_prompt_snapshot(
            outputs.prompt_snapshot,
            model_identifiers=model_identifiers,
            isolation_source_prompt_policy=effective_policy.isolation_source_prompt_policy,
        )
    logging_state = configure_run_logging(outputs.log, console_debug=args.debug)
    run_report = RunReport(
        outputs,
        RunContext(
            requested_taxa=tuple(names),
            requested_standardization_targets=tuple(target.value for target in active_targets),
            extracted_metadata=extracted_metadata_path,
            options=normalized_options,
            configuration_paths=tuple(configuration_paths),
            skip_llm=args.skip_llm,
            model_identifiers=model_identifiers,
            trace_llm_calls=args.debug,
            isolation_source_provenance=(
                effective_policy.isolation_source_prompt_policy.provenance
                if effective_policy.isolation_source_prompt_policy is not None
                else None
            ),
        ),
    )
    started = monotonic()
    logger.info(
        "Run started: taxa=%s standardization_targets=%s",
        ",".join(names),
        ",".join(target.value for target in active_targets),
    )
    build_progress = DatasetBuildProgress()

    try:
        with progress_context(disable=disable_progress):
            if curation_schema is not None:
                run_report.transition(RunPhase.EXTRACTION)
                run_report.begin_performed_extraction(
                    biosample_input_path=biosample_input_path,
                    bioproject_input_path=DEFAULT_BIOPROJECT_XML_INPUT,
                    extracted_metadata_path=extracted_metadata_path,
                    biosample_manifest_path=DEFAULT_BIOSAMPLE_SNAPSHOT_MANIFEST,
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
                    curation_schema=curation_schema,
                    index_path=index_path,
                    names=extraction_names,
                    taxon_registry=registry,
                    log_level=log_level,
                    disable_progress=disable_progress,
                )
                extraction_elapsed = monotonic() - extraction_started
                run_report.record_performed_extraction(
                    extraction_report,
                    elapsed_seconds=extraction_elapsed,
                )
                run_report.transition(RunPhase.DATASET_STREAMING)
                logger.info("Extraction finished: elapsed=%.2fs", extraction_elapsed)
            else:
                run_report.record_reused_extraction(
                    extracted_metadata_path=extracted_metadata_path,
                    biosample_manifest_path=DEFAULT_BIOSAMPLE_SNAPSHOT_MANIFEST,
                    bioproject_manifest_path=DEFAULT_BIOPROJECT_SNAPSHOT_MANIFEST,
                )
                run_report.transition(RunPhase.DATASET_STREAMING)
                logger.info("Using extracted metadata from: %s", extracted_metadata_path)

            logger.info("Streaming started")
            request = DatasetBuildRequest(
                extracted_metadata=extracted_metadata_path,
                biosample_snapshot_manifest=DEFAULT_BIOSAMPLE_SNAPSHOT_MANIFEST,
                bioproject_snapshot_manifest=DEFAULT_BIOPROJECT_SNAPSHOT_MANIFEST,
                requested_taxa=tuple(names),
                requested_targets=active_targets,
                final_destination=outputs.dataset,
                taxon_registry=registry,
                host_policy=effective_policy.host_policy,
                location_policy=effective_policy.location_policy,
                isolation_source_prompt_policy=effective_policy.isolation_source_prompt_policy,
                isolation_source_reasoning_destination=outputs.isolation_source_reasoning,
                names_dmp=args.names_dmp,
                nodes_dmp=args.nodes_dmp,
                overwrite=True,
                skip_llm=args.skip_llm,
                llm_settings=llm_settings,
                logger=pipeline_logger,
                disable_progress=disable_progress,
                progress=build_progress,
            )
            statistics = DatasetBuilder().build(request)

        processed = processed_rows(statistics)
        logger.info(
            "Streaming finished: processed=%d accepted=%d",
            processed,
            statistics.rows_written,
        )
        run_report.finish(
            RunStatus.SUCCEEDED,
            progress=build_progress,
            statistics=statistics,
        )
        logger.info("Run report finalized: %s", outputs.run_report)
        logger.info(
            "Run completed: elapsed=%.2fs rows=%d outputs=%s",
            monotonic() - started,
            statistics.rows_written,
            ",".join(str(path) for path in outputs.paths()),
        )
    except KeyboardInterrupt as exc:
        run_report.finish(
            RunStatus.INTERRUPTED,
            error=exc,
            progress=build_progress,
            statistics=build_progress.statistics,
        )
        logger.exception("Run interrupted")
        raise SystemExit(130) from None
    except Exception as exc:
        run_report.finish(
            RunStatus.FAILED,
            error=exc,
            progress=build_progress,
            statistics=build_progress.statistics,
        )
        logger.exception("Run failed")
        raise SystemExit(1) from None
    finally:
        logging_state.close()


if __name__ == "__main__":
    main()
