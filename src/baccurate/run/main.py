"""Main command-line entry point."""

import logging
from collections.abc import Sequence
from time import monotonic

from baccurate.adapters.progress import progress_context
from baccurate.extraction import run_extraction
from baccurate.paths import DEFAULT_BIOPROJECT_SNAPSHOT_PATH
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
    prompt_config = invocation.config.isolation_source_config

    invocation.outputs.initialize()
    if invocation.outputs.prompt_snapshot is not None:
        write_prompt_snapshot(
            invocation.outputs.prompt_snapshot,
            model_identifiers=invocation.model_identifiers,
            isolation_source_config=prompt_config,
        )
    logging_state = configure_run_logging(invocation.outputs.log, console_debug=invocation.debug)
    run_report = RunReport(
        invocation.outputs,
        RunContext(
            requested_taxa=tuple(invocation.taxon_keys),
            requested_standardization_targets=tuple(
                target.value for target in invocation.active_targets
            ),
            extracted_metadata=invocation.extracted_metadata_path,
            options=invocation.normalized_options,
            configuration_paths=tuple(invocation.configuration_paths),
            skip_llm=invocation.skip_llm,
            model_identifiers=invocation.model_identifiers,
            trace_llm_calls=invocation.debug,
            isolation_source_provenance=(
                prompt_config.provenance if prompt_config is not None else None
            ),
        ),
    )
    started = monotonic()
    logger.info(
        "Run started: taxa=%s standardization_targets=%s",
        ",".join(invocation.taxon_keys),
        ",".join(target.value for target in invocation.active_targets),
    )
    build_progress = DatasetBuildProgress()

    try:
        with progress_context(disable=invocation.disable_progress):
            if invocation.selection_config is not None:
                run_report.transition(RunPhase.EXTRACTION)
                run_report.begin_performed_extraction(
                    biosample_input_path=invocation.biosample_input_path,
                    bioproject_input_path=DEFAULT_BIOPROJECT_SNAPSHOT_PATH,
                    extracted_metadata_path=invocation.extracted_metadata_path,
                    biosample_manifest_path=invocation.biosample_manifest_path,
                    bioproject_manifest_path=invocation.bioproject_manifest_path,
                )
                extraction_started = monotonic()
                logger.info(
                    "Extraction started: input=%s output=%s",
                    invocation.biosample_input_path,
                    invocation.extracted_metadata_path,
                )
                extraction_report = run_extraction(
                    output_path=invocation.extracted_metadata_path,
                    selection_config=invocation.selection_config,
                    index_path=invocation.index_path,
                    names=invocation.extraction_taxon_keys,
                    taxon_registry=invocation.registry,
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
                    extracted_metadata_path=invocation.extracted_metadata_path,
                    biosample_manifest_path=invocation.biosample_manifest_path,
                    bioproject_manifest_path=invocation.bioproject_manifest_path,
                )
                run_report.transition(RunPhase.DATASET_STREAMING)
                logger.info("Using extracted metadata from: %s", invocation.extracted_metadata_path)

            logger.info("Streaming started")
            request = DatasetBuildRequest(
                extracted_metadata=invocation.extracted_metadata_path,
                biosample_manifest_path=invocation.biosample_manifest_path,
                bioproject_manifest_path=invocation.bioproject_manifest_path,
                requested_taxa=tuple(invocation.taxon_keys),
                requested_targets=invocation.active_targets,
                final_destination=invocation.outputs.dataset,
                taxon_registry=invocation.registry,
                host_config=invocation.config.host_config,
                location_config=invocation.config.location_config,
                isolation_source_config=prompt_config,
                isolation_source_reasoning_destination=invocation.outputs.isolation_source_reasoning,
                location_review_worklist_destination=invocation.outputs.location_review_worklist,
                names_dmp=invocation.names_dmp,
                nodes_dmp=invocation.nodes_dmp,
                overwrite=True,
                skip_llm=invocation.skip_llm,
                llm_settings=invocation.llm_settings,
                logger=pipeline_logger,
                disable_progress=invocation.disable_progress,
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
        logger.info("Run report finalized: %s", invocation.outputs.run_report)
        logger.info(
            "Run completed: elapsed=%.2fs rows=%d outputs=%s",
            monotonic() - started,
            statistics.rows_written,
            ",".join(str(path) for path in invocation.outputs.paths()),
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
