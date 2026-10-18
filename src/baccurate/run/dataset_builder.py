"""Build one final standardized dataset from extracted metadata."""

import csv
import json
import logging
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from contextlib import ExitStack
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Protocol, TextIO

from baccurate.adapters.llm.client import LLMSettings, load_llm_settings
from baccurate.adapters.progress import make_progress_bar
from baccurate.extraction import SEQUENCE_ACCESSION_COLUMNS
from baccurate.paths import DEFAULT_NAMES_DMP, DEFAULT_NODES_DMP
from baccurate.provenance.extracted_metadata import validate_extracted_metadata
from baccurate.run.location_review_worklist import (
    LocationReviewWorklist,
    LocationReviewWorklistSummary,
)
from baccurate.run.statistics import (
    DatasetBuildProgress,
    DatasetBuildStatistics,
    DateBuildStatistics,
    DateStatistics,
    HostBuildStatistics,
    HostStatistics,
    IsolationSourceBuildStatistics,
    IsolationSourceStatistics,
    LocationBuildStatistics,
    LocationStatistics,
    invented_label_inventory,
)
from baccurate.standardization.collection_date import RecordDates, RecordDateStandardizer
from baccurate.standardization.host import HostConfig, HostOutcome, HostStandardizer
from baccurate.standardization.host_isolation_source import (
    HostInitialRouting,
    HostIsolationSourceStandardizer,
    HostRecoveryRouting,
)
from baccurate.standardization.host_lineage import HostLineage, HostLineageEnricher
from baccurate.standardization.isolation_source import (
    IsolationSourceOntologyGapDiagnostic,
    IsolationSourceOutcome,
    IsolationSourceRejection,
    IsolationSourceStandardizer,
)
from baccurate.standardization.isolation_source_config import (
    FacetConfiguration,
    IsolationSourceConfig,
)
from baccurate.standardization.location import (
    LocationConfig,
    LocationDiagnostic,
    LocationOutcome,
    LocationRejection,
    LocationStandardizer,
)
from baccurate.targets import (
    DATASET_COLUMN_ORDER,
    TARGET_SPECS,
    ConfigSlot,
    StandardizationTarget,
    required_config_slots,
)
from baccurate.taxon_registry.registry import TaxonRegistry
from baccurate.taxon_registry.species_label_matching import NA

logger = logging.getLogger(__name__)


def _aggregate[Statistics](by_taxon: Mapping[str, Statistics], total: Statistics) -> Statistics:
    """Add the integer and ``Counter`` fields of each per-taxon statistics into ``total``."""
    for statistics in by_taxon.values():
        for count_field in fields(total):
            count = getattr(total, count_field.name)
            if isinstance(count, Counter):
                count.update(getattr(statistics, count_field.name))
            elif isinstance(count, int):
                setattr(total, count_field.name, count + getattr(statistics, count_field.name))
    return total


@dataclass(frozen=True, slots=True)
class DatasetBuildRequest:
    """Inputs, selection, destination, and runtime settings for one build."""

    extracted_metadata: Path
    biosample_manifest_path: Path
    bioproject_manifest_path: Path
    requested_taxa: tuple[str, ...]
    requested_targets: tuple[StandardizationTarget, ...]
    final_destination: Path
    taxon_registry: TaxonRegistry
    host_config: HostConfig | None = None
    location_config: LocationConfig | None = None
    isolation_source_config: IsolationSourceConfig | None = None
    isolation_source_reasoning_destination: Path | None = None
    location_review_worklist_destination: Path | None = None
    names_dmp: Path = DEFAULT_NAMES_DMP
    nodes_dmp: Path = DEFAULT_NODES_DMP
    overwrite: bool = False
    skip_llm: bool = False
    llm_settings: LLMSettings | None = None
    logger: logging.Logger = field(default_factory=lambda: logger)
    disable_progress: bool = False
    progress: DatasetBuildProgress = field(default_factory=DatasetBuildProgress)


@dataclass(frozen=True, slots=True)
class _FinalRow:
    accession: str
    taxon: str
    ncbi_organism: str
    sylph_species: str
    bioproject: str
    sequence_accessions: tuple[str, ...]
    date: RecordDates | None
    location: LocationOutcome | LocationRejection | None
    isolation_source: IsolationSourceOutcome | None
    host: HostOutcome | None
    host_lineage: HostLineage | None


class _RowWriter(Protocol):
    def writerow(self, final_row_values: Iterable[object], /) -> object:
        """Write one projected dataset row."""


class _FinalRowAssembler:
    """Turn an extracted metadata record and standardized outcomes into a final dataset row."""

    base_columns = (
        "accession",
        "bioproject",
        *SEQUENCE_ACCESSION_COLUMNS,
        "taxon",
        "ncbi_organism",
        "sylph_species",
    )

    def __init__(
        self,
        targets: Sequence[StandardizationTarget],
        taxon_registry: TaxonRegistry,
        isolation_source_facets: Sequence[FacetConfiguration] = (),
        exact_match_ids: Mapping[str, str] | None = None,
    ):
        self._taxon_registry = taxon_registry
        self._selected_targets = tuple(targets)
        self._isolation_source_facets = tuple(isolation_source_facets)
        self._exact_match_ids = exact_match_ids or {}
        selected = set(targets)
        self.columns = self.base_columns
        for target in DATASET_COLUMN_ORDER:
            if target in selected:
                if target is StandardizationTarget.ISOLATION_SOURCE:
                    self.columns += (
                        "iso_attr_orig",
                        "iso_val_orig",
                        *(facet.output_column for facet in self._isolation_source_facets),
                        "iso_term_ids",
                        "iso_external_ids",
                    )
                else:
                    self.columns += TARGET_SPECS[target].output_columns

    def assemble(
        self,
        extracted_record: Mapping[str, str],
        date: RecordDates | None,
        location: LocationOutcome | LocationRejection | None,
        isolation_source: IsolationSourceOutcome | None,
        host: HostOutcome | None,
        host_lineage: HostLineage | None,
    ) -> _FinalRow:
        """Assemble a final dataset row.

        Inputs are one extracted metadata record and its standardized outcomes.
        """
        taxon = extracted_record["taxon_key"]
        accession = extracted_record["accession"]
        return _FinalRow(
            accession=accession,
            taxon=self._taxon_registry.scientific_name(taxon),
            ncbi_organism=extracted_record.get("ncbi_organism", NA),
            sylph_species=extracted_record.get("sylph_species", NA),
            bioproject=extracted_record.get("bioproject_accession", ""),
            sequence_accessions=tuple(
                extracted_record.get(column, "") for column in SEQUENCE_ACCESSION_COLUMNS
            ),
            date=date,
            location=location,
            isolation_source=isolation_source,
            host=host,
            host_lineage=host_lineage,
        )

    def project(self, final_row: _FinalRow) -> tuple[object, ...]:
        values: tuple[object, ...] = (
            final_row.accession,
            final_row.bioproject,
            *final_row.sequence_accessions,
            final_row.taxon,
            final_row.ncbi_organism,
            final_row.sylph_species,
        )
        if StandardizationTarget.DATE in self._selected_targets:
            if final_row.date is None:
                values += ("",) * len(TARGET_SPECS[StandardizationTarget.DATE].output_columns)
            else:
                collection = final_row.date.collection
                record = final_row.date.record
                pairs = tuple(
                    pair
                    for outcome in (collection, record)
                    if outcome is not None
                    for pair in outcome.supporting_pairs
                )
                values += (
                    "||".join(pair.attribute for pair in pairs),
                    "||".join(pair.value for pair in pairs),
                    collection.bounds.start.isoformat() if collection else "",
                    collection.bounds.end.isoformat() if collection else "",
                    record.bounds.start.isoformat() if record else "",
                    collection.precision if collection else "",
                    collection.structure if collection else "",
                    "||".join(collection.diagnostics) if collection else "",
                )
        if StandardizationTarget.LOCATION in self._selected_targets:
            location = final_row.location or LocationRejection()
            if location.coordinate is None:
                latitude, longitude = "", ""
            else:
                latitude, longitude = (
                    f"{degrees:.5f}".rstrip("0").rstrip(".") for degrees in location.coordinate
                )
            if isinstance(location, LocationRejection):
                country, un_region, sublocation, route = "", "", "", ""
            else:
                country, un_region, sublocation, route = (
                    location.country,
                    location.un_region,
                    location.sublocation or "NA",
                    location.route,
                )
            values += (
                "||".join(pair.attribute for pair in location.supporting_pairs),
                "||".join(pair.value for pair in location.supporting_pairs),
                country,
                un_region,
                sublocation,
                latitude,
                longitude,
                route,
                "||".join(location.diagnostics),
            )
        if StandardizationTarget.ISOLATION_SOURCE in self._selected_targets:
            if final_row.isolation_source is None:
                values += ("",) * (len(self._isolation_source_facets) + 4)
            else:
                labels_by_facet: dict[str, list[str]] = {
                    facet.key: [] for facet in self._isolation_source_facets
                }
                for term in final_row.isolation_source.selected_terms:
                    labels_by_facet[term.facet].append(term.label)
                values += (
                    "||".join(
                        pair.attribute for pair in final_row.isolation_source.supporting_pairs
                    ),
                    "||".join(pair.value for pair in final_row.isolation_source.supporting_pairs),
                    *(
                        "||".join(labels_by_facet[facet.key]) or "NA"
                        for facet in self._isolation_source_facets
                    ),
                    "||".join(term.term_id for term in final_row.isolation_source.selected_terms),
                    "||".join(
                        self._exact_match_ids[term.term_id]
                        for term in final_row.isolation_source.selected_terms
                        if term.term_id in self._exact_match_ids
                    )
                    or "NA",
                )
        if StandardizationTarget.HOST in self._selected_targets:
            if final_row.host is None or final_row.host.standardized is None:
                values += ("",) * len(TARGET_SPECS[StandardizationTarget.HOST].output_columns)
            else:
                pairs = final_row.host.supporting_pairs
                lineage = final_row.host_lineage
                if lineage is None:
                    raise ValueError(f"Missing host lineage enrichment for {final_row.accession}")
                values += (
                    "||".join(pair.attribute for pair in pairs),
                    "||".join(pair.value for pair in pairs),
                    final_row.host.standardized.taxid,
                    final_row.host.standardized.scientific_name,
                    lineage.common_names,
                    lineage.lineage_names,
                    lineage.lineage_taxids,
                    final_row.host.route,
                    "||".join(final_row.host.record_diagnostics),
                )
        return values


class DatasetBuilder:
    """
    Stream one final dataset for the requested records and standardization targets.

    The four optional factories are test seams for the external dependencies:
    the LLM client (isolation source), the INSDC location list and reviewed config
    (location), the NCBI Taxonomy table (host), and ``names.dmp``/``nodes.dmp`` (lineage).
    """

    def __init__(
        self,
        location_standardizer_factory: (
            Callable[[LocationConfig, logging.Logger], LocationStandardizer] | None
        ) = None,
        host_standardizer_factory: (
            Callable[[HostConfig, logging.Logger], HostStandardizer] | None
        ) = None,
        host_lineage_factory: Callable[[Path, Path], HostLineageEnricher] | None = None,
        isolation_source_standardizer_factory: (
            Callable[[IsolationSourceConfig, logging.Logger], IsolationSourceStandardizer] | None
        ) = None,
    ) -> None:
        self._location_standardizer_factory = location_standardizer_factory
        self._host_standardizer_factory = host_standardizer_factory or (
            lambda config, result_logger: HostStandardizer(
                config,
                result_logger=result_logger,
            )
        )
        self._host_lineage_factory = host_lineage_factory or HostLineageEnricher
        self._isolation_source_standardizer_factory = isolation_source_standardizer_factory

    def build(self, request: DatasetBuildRequest) -> DatasetBuildStatistics:
        taxa, targets = self._validate_request(request)
        llm_settings = request.llm_settings or load_llm_settings()
        request.progress.processed_rows = 0
        request.progress.rows_written = 0
        destination = Path(request.final_destination)
        reasoning_destination = (
            Path(request.isolation_source_reasoning_destination)
            if StandardizationTarget.ISOLATION_SOURCE in targets
            and request.isolation_source_reasoning_destination is not None
            else None
        )
        destinations = tuple(
            path for path in (destination, reasoning_destination) if path is not None
        )
        isolation_source_facets = (
            request.isolation_source_config.facets
            if StandardizationTarget.ISOLATION_SOURCE in targets
            and request.isolation_source_config is not None
            else ()
        )
        if not request.overwrite:
            collision = next((path for path in destinations if path.exists()), None)
            if collision is not None:
                raise FileExistsError(f"Build output already exists: {collision}")
        for path in destinations:
            path.parent.mkdir(parents=True, exist_ok=True)
        mode = "w" if request.overwrite else "x"

        with ExitStack() as resources:
            destination_stream = resources.enter_context(
                destination.open(mode, encoding="utf-8", newline="")
            )
            reasoning_stream = (
                resources.enter_context(
                    reasoning_destination.open(mode, encoding="utf-8", newline="")
                )
                if reasoning_destination is not None
                else None
            )
            assembler = _FinalRowAssembler(
                targets,
                request.taxon_registry,
                isolation_source_facets,
                request.isolation_source_config.ontology.exact_match_ids
                if request.isolation_source_config is not None
                else None,
            )
            writer = csv.writer(destination_stream, delimiter="\t", lineterminator="\n")
            writer.writerow(assembler.columns)

            snapshot_pair = validate_extracted_metadata(
                request.extracted_metadata,
                request.biosample_manifest_path,
                request.bioproject_manifest_path,
            )
            extracted_row_counts = self._count_selected_extracted_rows(
                request.extracted_metadata, taxa
            )
            if StandardizationTarget.DATE in targets:
                date_standardizers = {
                    taxon: RecordDateStandardizer(snapshot_pair.metadata_reference_date)
                    for taxon in taxa
                }
            else:
                date_standardizers = {}
            location_standardizer = None
            location_review_worklist = None
            if StandardizationTarget.LOCATION in targets:
                location_standardizer = (
                    self._location_standardizer_factory(request.location_config, request.logger)
                    if self._location_standardizer_factory is not None
                    else LocationStandardizer(
                        request.location_config,
                        result_logger=request.logger,
                    )
                )
                if request.location_review_worklist_destination is not None:
                    location_review_worklist = LocationReviewWorklist()
            host_standardizer = (
                self._host_standardizer_factory(request.host_config, request.logger)
                if StandardizationTarget.HOST in targets
                or StandardizationTarget.ISOLATION_SOURCE in targets
                else None
            )
            isolation_source_standardizer = None
            if StandardizationTarget.ISOLATION_SOURCE in targets:
                if self._isolation_source_standardizer_factory is not None:
                    isolation_source_standardizer = self._isolation_source_standardizer_factory(
                        request.isolation_source_config,
                        request.logger,
                    )
                else:
                    isolation_source_options = {
                        "result_logger": request.logger,
                        "llm_settings": llm_settings,
                    }
                    if request.skip_llm:
                        isolation_source_options["client"] = None
                    isolation_source_standardizer = IsolationSourceStandardizer(
                        request.isolation_source_config,
                        **isolation_source_options,
                    )
            if isolation_source_standardizer is not None:
                resources.callback(isolation_source_standardizer.close)
            host_lineage = (
                self._host_lineage_factory(request.names_dmp, request.nodes_dmp)
                if StandardizationTarget.HOST in targets
                or StandardizationTarget.ISOLATION_SOURCE in targets
                else None
            )
            host_isolation_source_standardizer = (
                HostIsolationSourceStandardizer(
                    host_standardizer,
                    isolation_source_standardizer,
                    host_lineage,
                )
                if host_standardizer is not None
                and isolation_source_standardizer is not None
                and host_lineage is not None
                else None
            )
            date_stats = {
                taxon: DateStatistics(
                    diagnostics=standardizer.diagnostic_counts,
                    parsed_date_rejections=standardizer.rejection_counts,
                    notices=standardizer.notice_counts,
                )
                for taxon, standardizer in date_standardizers.items()
            }
            location_stats = (
                {taxon: LocationStatistics() for taxon in taxa}
                if location_standardizer is not None
                else {}
            )
            host_stats = (
                {taxon: HostStatistics() for taxon in taxa}
                if StandardizationTarget.HOST in targets
                else {}
            )
            isolation_source_stats = (
                {taxon: IsolationSourceStatistics() for taxon in taxa}
                if isolation_source_standardizer is not None
                else {}
            )
            ontology_gaps: dict[str, Counter[IsolationSourceOntologyGapDiagnostic]] = {
                taxon: Counter() for taxon in isolation_source_stats
            }
            rows_written = 0
            try:
                rows_written = self._process_extracted_records(
                    request,
                    taxa,
                    targets,
                    extracted_row_counts,
                    date_standardizers,
                    location_standardizer,
                    host_standardizer,
                    host_isolation_source_standardizer,
                    host_lineage,
                    date_stats,
                    location_stats,
                    location_review_worklist,
                    host_stats,
                    isolation_source_stats,
                    ontology_gaps,
                    assembler,
                    writer,
                    reasoning_stream,
                )
            finally:
                request.progress.statistics = self._make_build_statistics(
                    destination,
                    date_stats,
                    location_stats,
                    (
                        location_review_worklist.write(request.location_review_worklist_destination)
                        if location_review_worklist is not None
                        and request.location_review_worklist_destination is not None
                        else None
                    ),
                    host_stats,
                    isolation_source_stats,
                    ontology_gaps,
                    request.progress.rows_written,
                )
        statistics = request.progress.statistics
        self._warn_recoverable_conditions(request.logger, statistics)
        request.logger.info("Built final dataset with %d rows at %s", rows_written, destination)
        return statistics

    @staticmethod
    def _warn_recoverable_conditions(
        result_logger: logging.Logger,
        statistics: DatasetBuildStatistics,
    ) -> None:
        if statistics.location is None:
            return
        count = statistics.location.aggregate.diagnostics.get(
            LocationDiagnostic.RECOVERABLE_COORDINATE_FAILURE, 0
        )
        if count:
            result_logger.warning("Recoverable coordinate resolution failure: %d record(s)", count)

    @staticmethod
    def _validate_request(
        request: DatasetBuildRequest,
    ) -> tuple[tuple[str, ...], tuple[StandardizationTarget, ...]]:
        taxa = tuple(request.requested_taxa)
        if not taxa:
            raise ValueError("Need at least one taxon")
        if len(set(taxa)) != len(taxa):
            raise ValueError("Taxa must be unique")
        targets = tuple(request.requested_targets)
        if not targets:
            raise ValueError("Need at least one standardization attribute")
        if len(set(targets)) != len(targets):
            raise ValueError("Standardization attributes must be unique")
        required_configs = required_config_slots(targets)
        if ConfigSlot.HOST in required_configs and request.host_config is None:
            raise ValueError("Host or isolation-source standardization requires a host config")
        if ConfigSlot.LOCATION in required_configs and request.location_config is None:
            raise ValueError("Geographic-location standardization requires a location config")
        if (
            ConfigSlot.ISOLATION_SOURCE in required_configs
            and request.isolation_source_config is None
        ):
            raise ValueError("Isolation-source standardization requires an isolation-source config")
        return taxa, targets

    @staticmethod
    def _count_selected_extracted_rows(
        input_path: Path,
        taxa: Sequence[str],
    ) -> dict[str, int]:
        counts = dict.fromkeys(taxa, 0)
        with Path(input_path).open("r", encoding="utf-8", newline="") as stream:
            for record in csv.DictReader(stream, delimiter="\t"):
                taxon = (record.get("taxon_key") or "").strip()
                if taxon in counts:
                    counts[taxon] += 1
        return counts

    @staticmethod
    def _process_extracted_records(
        request: DatasetBuildRequest,
        taxa: Sequence[str],
        targets: Sequence[StandardizationTarget],
        extracted_row_counts: Mapping[str, int],
        date_standardizers: Mapping[str, RecordDateStandardizer],
        location_standardizer: LocationStandardizer | None,
        host_standardizer: HostStandardizer | None,
        host_isolation_source_standardizer: HostIsolationSourceStandardizer | None,
        host_lineage: HostLineageEnricher | None,
        date_stats: Mapping[str, DateStatistics],
        location_stats: Mapping[str, LocationStatistics],
        location_review_worklist: LocationReviewWorklist | None,
        host_stats: Mapping[str, HostStatistics],
        isolation_source_stats: Mapping[str, IsolationSourceStatistics],
        ontology_gaps: Mapping[str, Counter[IsolationSourceOntologyGapDiagnostic]],
        assembler: _FinalRowAssembler,
        writer: _RowWriter,
        reasoning_stream: TextIO | None,
    ) -> int:
        selected = set(taxa)
        rows_written = 0
        total = sum(extracted_row_counts.values())
        with (
            make_progress_bar(
                total, "dataset build", disable=request.disable_progress
            ) as progress_bar,
            Path(request.extracted_metadata).open(
                "r",
                encoding="utf-8",
                newline="",
            ) as stream,
        ):
            reader = csv.DictReader(stream, delimiter="\t")
            _require_columns(reader.fieldnames, targets)
            for row_number, extracted_record in enumerate(reader, start=1):
                accession = (extracted_record.get("accession") or "").strip()
                taxon = (extracted_record.get("taxon_key") or "").strip()
                if not accession:
                    raise ValueError(f"Record {row_number} has no accession")
                if not taxon:
                    raise ValueError(f"Record {accession} has no taxon")
                if taxon not in selected:
                    continue
                date_outcome = None
                if taxon in date_standardizers:
                    stats = date_stats[taxon]
                    stats.processed += 1
                    date_outcome = date_standardizers[taxon].standardize(extracted_record)
                    if date_outcome is None:
                        stats.rejected += 1
                    else:
                        stats.standardized += 1
                        collection = date_outcome.collection
                        if collection is not None:
                            stats.structures[collection.structure] += 1
                            stats.precisions[collection.precision] += 1
                            stats.date_diagnostics.update(collection.diagnostics)

                location_outcome = None
                location_result: LocationOutcome | LocationRejection | None = None
                if location_standardizer is not None:
                    stats = location_stats[taxon]
                    stats.processed += 1
                    location_result = location_standardizer.standardize(extracted_record)
                    stats.coordinate_decodes += location_result.coordinate_decodes
                    stats.insdc_term_matches += location_result.insdc_term_matches
                    stats.country_conversion_matches += location_result.country_conversion_matches
                    stats.reviewed_mapping_matches += location_result.reviewed_mapping_matches
                    stats.diagnostics.update(location_result.diagnostics)
                    if location_review_worklist is not None:
                        location_review_worklist.observe(
                            location_result.unresolved_inputs,
                            accession=accession,
                            taxon_key=taxon,
                        )
                    if isinstance(location_result, LocationRejection):
                        stats.rejected += 1
                    else:
                        stats.standardized += 1
                        stats.resolution_routes[location_result.route] += 1
                        location_outcome = location_result

                host_outcome = None
                isolation_source_outcome = None
                lineage_outcome = None
                if host_isolation_source_standardizer is not None:
                    host_isolation_source_result = host_isolation_source_standardizer.standardize(
                        extracted_record
                    )
                    host_outcome = host_isolation_source_result.host
                    isolation_source_result = host_isolation_source_result.isolation_source

                    if host_stats:
                        host_taxon_stats = host_stats[taxon]
                        host_taxon_stats.processed += 1
                        host_taxon_stats.diagnostics.update(
                            host_isolation_source_result.diagnostics.host
                        )
                        if host_outcome.standardized is not None:
                            host_taxon_stats.standardized += 1
                            host_taxon_stats.needs_review += int(host_outcome.needs_review)
                        elif (
                            host_isolation_source_result.routing.host_initial
                            is HostInitialRouting.OVERFLOW
                        ):
                            host_taxon_stats.overflow += 1
                        else:
                            host_taxon_stats.rejected += 1
                        host_taxon_stats.host_recovery_passes += int(
                            host_isolation_source_result.routing.host_recovery
                            is not HostRecoveryRouting.NOT_ELIGIBLE
                        )
                    if host_lineage is not None and host_outcome.standardized is not None:
                        lineage_outcome = host_lineage.enrich(host_outcome.standardized.taxid)

                    isolation_source_taxon_stats = isolation_source_stats[taxon]
                    isolation_source_taxon_stats.processed += 1
                    isolation_source_taxon_stats.diagnostics.update(
                        host_isolation_source_result.diagnostics.isolation_source
                    )
                    ontology_gaps[taxon].update(isolation_source_result.ontology_gap_diagnostics)
                    isolation_source_taxon_stats.host_recovery_passes += int(
                        host_isolation_source_result.routing.host_recovery
                        is not HostRecoveryRouting.NOT_ELIGIBLE
                    )
                    if isinstance(isolation_source_result, IsolationSourceRejection):
                        isolation_source_taxon_stats.rejected += 1
                    else:
                        isolation_source_outcome = isolation_source_result
                        isolation_source_taxon_stats.standardized += 1
                        isolation_source_taxon_stats.exact_matches += (
                            isolation_source_result.exact_matches
                        )
                        isolation_source_taxon_stats.cache_hits += (
                            isolation_source_result.cache_hits
                        )
                        isolation_source_taxon_stats.llm_calls += isolation_source_result.llm_calls
                        isolation_source_taxon_stats.evidence_levels[
                            isolation_source_result.evidence_level
                        ] += 1
                elif host_standardizer is not None:
                    stats = host_stats[taxon]
                    stats.processed += 1
                    host_result = host_standardizer.standardize(extracted_record)
                    host_outcome = host_result
                    stats.diagnostics.update(host_result.diagnostics)
                    if host_result.standardized is not None:
                        stats.standardized += 1
                        stats.needs_review += int(host_result.needs_review)
                        lineage_outcome = host_lineage.enrich(host_result.standardized.taxid)
                    elif host_result.overflow is not None:
                        stats.overflow += 1
                    else:
                        stats.rejected += 1
                    stats.host_recovery_passes += int(host_result.from_recovery_pass)

                if (
                    date_outcome is not None
                    or location_outcome is not None
                    or isolation_source_outcome is not None
                    or (
                        StandardizationTarget.HOST in targets
                        and host_outcome is not None
                        and host_outcome.standardized is not None
                    )
                ):
                    reasoning_json = (
                        _isolation_source_reasoning_json(accession, taxon, isolation_source_outcome)
                        if isolation_source_outcome is not None and reasoning_stream is not None
                        else None
                    )
                    final_row = assembler.assemble(
                        extracted_record,
                        date_outcome,
                        location_result,
                        isolation_source_outcome,
                        host_outcome,
                        lineage_outcome,
                    )
                    writer.writerow(assembler.project(final_row))
                    if reasoning_json is not None and reasoning_stream is not None:
                        reasoning_stream.write(reasoning_json + "\n")
                    rows_written += 1
                    request.progress.rows_written = rows_written
                request.progress.processed_rows += 1
                progress_bar.update(1)
        return rows_written

    @staticmethod
    def _make_build_statistics(
        destination: Path,
        date_stats: Mapping[str, DateStatistics],
        location_stats: Mapping[str, LocationStatistics],
        location_review_worklist: LocationReviewWorklistSummary | None,
        host_stats: Mapping[str, HostStatistics],
        isolation_source_stats: Mapping[str, IsolationSourceStatistics],
        ontology_gaps: Mapping[str, Counter[IsolationSourceOntologyGapDiagnostic]],
        rows_written: int,
    ) -> DatasetBuildStatistics:
        isolation_source_statistics = None
        if isolation_source_stats:
            for taxon, stats in isolation_source_stats.items():
                stats.invented_labels = invented_label_inventory(ontology_gaps[taxon])
            aggregate = _aggregate(isolation_source_stats, IsolationSourceStatistics())
            aggregate.invented_labels = invented_label_inventory(
                sum(ontology_gaps.values(), Counter())
            )
            isolation_source_statistics = IsolationSourceBuildStatistics(
                aggregate, isolation_source_stats
            )
        return DatasetBuildStatistics(
            final_destination=destination,
            rows_written=rows_written,
            date=(
                DateBuildStatistics(_aggregate(date_stats, DateStatistics()), date_stats)
                if date_stats
                else None
            ),
            location=(
                LocationBuildStatistics(
                    _aggregate(location_stats, LocationStatistics()),
                    location_stats,
                    location_review_worklist,
                )
                if location_stats
                else None
            ),
            host=(
                HostBuildStatistics(_aggregate(host_stats, HostStatistics()), host_stats)
                if host_stats
                else None
            ),
            isolation_source=isolation_source_statistics,
        )


def _require_columns(
    fieldnames: Sequence[str] | None,
    targets: Sequence[StandardizationTarget],
) -> None:
    required = {
        "accession",
        "taxon_key",
        "bioproject_accession",
        *SEQUENCE_ACCESSION_COLUMNS,
    }
    for target in targets:
        required.update(TARGET_SPECS[target].input_columns)
    available = set(fieldnames or ())
    missing = sorted(required - available)
    if missing:
        raise ValueError(f"Extracted metadata is missing required columns: {', '.join(missing)}")


def _isolation_source_reasoning_json(
    accession: str,
    taxon: str,
    outcome: IsolationSourceOutcome,
) -> str:
    """Render an extracted metadata record's isolation-source reasoning as a JSON line."""
    selected_terms: dict[str, list[dict[str, str]]] = {}
    for term in outcome.selected_terms:
        selected_terms.setdefault(term.facet, []).append(
            {"term_id": term.term_id, "label": term.label}
        )
    reasoning_record = {
        "accession": accession,
        "taxon_key": taxon,
        "origins": [
            {"attribute": pair.attribute, "value": pair.value} for pair in outcome.supporting_pairs
        ],
        "selected_terms": selected_terms,
        "evidence_level": outcome.evidence_level.value,
        "diagnostics": [diagnostic.value for diagnostic in outcome.diagnostics],
        "reasoning": [step.as_dict() for step in outcome.reasoning],
    }
    return json.dumps(reasoning_record, ensure_ascii=False)
