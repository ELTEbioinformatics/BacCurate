"""Build one final standardized dataset from extracted metadata."""

import csv
import json
import logging
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from contextlib import ExitStack
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, TextIO

from baccurate.adapters.llm.client import LLMSettings, load_llm_settings
from baccurate.adapters.progress import make_progress_bar
from baccurate.extraction import SEQUENCE_ACCESSION_COLUMNS
from baccurate.paths import DEFAULT_NAMES_DMP, DEFAULT_NODES_DMP
from baccurate.provenance.source_snapshot import validate_extracted_metadata_bundle
from baccurate.run.location_review_worklist import (
    LOCATION_REVIEW_WORKLIST_FILENAME,
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
from baccurate.standardization.collection_date import (
    DateCategory,
    DateDiagnostic,
    DateOutcome,
    DatePrecision,
    DateStructure,
    RecordDateStandardizer,
)
from baccurate.standardization.host import (
    HostDiagnostic,
    HostOutcome,
    HostPolicy,
    HostStandardizer,
)
from baccurate.standardization.host_isolation_source import (
    HostInitialRouting,
    HostIsolationSourceStandardizer,
    HostRecoveryRouting,
    host_isolation_source_standardizer_from_components,
)
from baccurate.standardization.host_lineage import HostLineage, HostLineageEnricher
from baccurate.standardization.isolation_source import (
    IsolationSourceDiagnostic,
    IsolationSourceEvidenceLevel,
    IsolationSourceOntologyGapDiagnostic,
    IsolationSourceOutcome,
    IsolationSourcePromptPolicy,
    IsolationSourceRejection,
    IsolationSourceStandardizer,
)
from baccurate.standardization.isolation_source_ontology import IsolationSourceFacet
from baccurate.standardization.location import (
    LocationDiagnostic,
    LocationOutcome,
    LocationPolicy,
    LocationRejection,
    LocationResolutionRoute,
    LocationStandardizer,
)
from baccurate.standardization_target import specifications as target_specifications
from baccurate.standardization_target.policy_slot import PolicySlot
from baccurate.standardization_target.specifications import (
    DATASET_COLUMN_ORDER,
    StandardizationTarget,
    required_policy_slots,
)
from baccurate.taxon_registry.registry import TaxonRegistry
from baccurate.taxon_registry.species_label_matching import NA

logger = logging.getLogger(__name__)


_LOCATION_ANSWER_COLUMN_COUNT = (
    len(target_specifications.TARGET_SPECS[StandardizationTarget.LOCATION].output_columns) - 3
)


def _sum_counts[Key: str](counts: Iterable[Counter[Key]]) -> dict[Key, int]:
    return dict(sorted(sum(counts, Counter()).items()))


@dataclass(frozen=True, slots=True)
class DatasetBuildRequest:
    """Inputs, selection, destination, and runtime settings for one build."""

    extracted_metadata: Path
    biosample_snapshot_manifest: Path
    bioproject_snapshot_manifest: Path
    requested_taxa: tuple[str, ...]
    requested_targets: tuple[StandardizationTarget, ...]
    final_destination: Path
    taxon_registry: TaxonRegistry
    host_policy: HostPolicy | None = None
    location_policy: LocationPolicy | None = None
    isolation_source_prompt_policy: IsolationSourcePromptPolicy | None = None
    isolation_source_reasoning_destination: Path | None = None
    names_dmp: Path = DEFAULT_NAMES_DMP
    nodes_dmp: Path = DEFAULT_NODES_DMP
    overwrite: bool = False
    skip_llm: bool = False
    llm_settings: LLMSettings | None = None
    logger: logging.Logger = field(default_factory=lambda: logger)
    disable_progress: bool = False
    progress: DatasetBuildProgress = field(default_factory=DatasetBuildProgress)


@dataclass(slots=True)
class _MutableDateStatistics:
    processed: int = 0
    standardized: int = 0
    rejected: int = 0
    categories: Counter[DateCategory] = field(default_factory=Counter)
    structures: Counter[DateStructure] = field(default_factory=Counter)
    precisions: Counter[DatePrecision] = field(default_factory=Counter)
    derivations: Counter[str] = field(default_factory=Counter)


@dataclass(slots=True)
class _MutableLocationStatistics:
    processed: int = 0
    standardized: int = 0
    rejected: int = 0
    coordinate_decodes: int = 0
    insdc_term_matches: int = 0
    country_conversion_matches: int = 0
    reviewed_mapping_matches: int = 0
    resolution_routes: Counter[LocationResolutionRoute] = field(default_factory=Counter)
    diagnostics: Counter[LocationDiagnostic] = field(default_factory=Counter)


@dataclass(slots=True)
class _MutableHostStatistics:
    processed: int = 0
    standardized: int = 0
    rejected: int = 0
    overflow: int = 0
    needs_review: int = 0
    host_recovery_passes: int = 0
    diagnostics: Counter[HostDiagnostic] = field(default_factory=Counter)


@dataclass(slots=True)
class _MutableIsolationSourceStatistics:
    processed: int = 0
    standardized: int = 0
    rejected: int = 0
    exact_matches: int = 0
    cache_hits: int = 0
    llm_calls: int = 0
    host_recovery_passes: int = 0
    evidence_levels: Counter[IsolationSourceEvidenceLevel] = field(default_factory=Counter)
    diagnostics: Counter[IsolationSourceDiagnostic] = field(default_factory=Counter)
    ontology_gap_diagnostics: Counter[IsolationSourceOntologyGapDiagnostic] = field(
        default_factory=Counter
    )


@dataclass(frozen=True, slots=True)
class _FinalRow:
    accession: str
    taxon: str
    ncbi_organism: str
    sylph_species: str
    bioproject: str
    sequence_accessions: tuple[str, ...]
    date: DateOutcome | None
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
        "taxon",
        "ncbi_organism",
        "sylph_species",
        "bioproject",
        *SEQUENCE_ACCESSION_COLUMNS,
    )

    def __init__(
        self,
        targets: Sequence[StandardizationTarget],
        taxon_registry: TaxonRegistry,
        isolation_source_facets: Sequence[IsolationSourceFacet] = (),
    ):
        self._taxon_registry = taxon_registry
        self._selected_targets = tuple(targets)
        self._isolation_source_facets = tuple(isolation_source_facets)
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
                    )
                else:
                    self.columns += target_specifications.TARGET_SPECS[target].output_columns

    def assemble(
        self,
        extracted_record: Mapping[str, str],
        date: DateOutcome | None,
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
            final_row.taxon,
            final_row.ncbi_organism,
            final_row.sylph_species,
            final_row.bioproject,
            *final_row.sequence_accessions,
        )
        if StandardizationTarget.DATE in self._selected_targets:
            if final_row.date is None:
                values += ("",) * len(
                    target_specifications.TARGET_SPECS[StandardizationTarget.DATE].output_columns
                )
            else:
                attributes = "||".join(pair.attribute for pair in final_row.date.supporting_pairs)
                date_values = "||".join(pair.value for pair in final_row.date.supporting_pairs)
                values += (
                    final_row.date.category,
                    final_row.date.structure,
                    final_row.date.precision,
                    final_row.date.bounds.start.isoformat(),
                    final_row.date.bounds.end.isoformat(),
                    "||".join(final_row.date.derivations),
                    attributes,
                    date_values,
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
                answer_columns: tuple[object, ...] = ("",) * (_LOCATION_ANSWER_COLUMN_COUNT - 2)
            else:
                answer_columns = (
                    "||".join(map(str, location.selected_pair_positions)),
                    location.route,
                    location.country,
                    location.un_region,
                    location.sublocation or "NA",
                )
            values += (
                "||".join(pair.attribute for pair in location.supporting_pairs),
                "||".join(pair.value for pair in location.supporting_pairs),
                *answer_columns,
                latitude,
                longitude,
                "||".join(location.diagnostics),
            )
        if StandardizationTarget.ISOLATION_SOURCE in self._selected_targets:
            if final_row.isolation_source is None:
                values += ("",) * (len(self._isolation_source_facets) + 3)
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
                )
        if StandardizationTarget.HOST in self._selected_targets:
            if final_row.host is None or final_row.host.standardized is None:
                values += ("", "", "", "", "", "", "", "", "")
            else:
                pair = final_row.host.supporting_pairs[0]
                lineage = final_row.host_lineage
                if lineage is None:
                    raise ValueError(f"Missing host lineage enrichment for {final_row.accession}")
                values += (
                    pair.attribute,
                    pair.value,
                    final_row.host.standardized.taxid,
                    final_row.host.standardized.scientific_name,
                    lineage.common_names,
                    lineage.lineage_names,
                    lineage.lineage_taxids,
                    final_row.host.match_quality_score,
                    final_row.host.needs_review,
                )
        return values


class DatasetBuilder:
    """
    Stream one final dataset for the requested records and standardization targets.

    The four optional factories are test seams for the external dependencies:
    the LLM client (isolation source), the INSDC location list and reviewed policy
    (location), the NCBI Taxonomy table (host), and ``names.dmp``/``nodes.dmp`` (lineage).
    """

    def __init__(
        self,
        location_standardizer_factory: (
            Callable[[LocationPolicy, logging.Logger], LocationStandardizer] | None
        ) = None,
        host_standardizer_factory: (
            Callable[[HostPolicy, logging.Logger], HostStandardizer] | None
        ) = None,
        host_lineage_factory: Callable[[Path, Path], HostLineageEnricher] | None = None,
        isolation_source_standardizer_factory: (
            Callable[[IsolationSourcePromptPolicy, logging.Logger], IsolationSourceStandardizer]
            | None
        ) = None,
    ) -> None:
        self._location_standardizer_factory = location_standardizer_factory
        self._host_standardizer_factory = host_standardizer_factory or (
            lambda policy, result_logger: HostStandardizer(
                policy,
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
            tuple(
                sorted(
                    request.isolation_source_prompt_policy.ontology.facets.values(),
                    key=lambda facet: facet.render_order,
                )
            )
            if StandardizationTarget.ISOLATION_SOURCE in targets
            and request.isolation_source_prompt_policy is not None
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
            writer = csv.writer(destination_stream, delimiter="\t", lineterminator="\n")
            writer.writerow(
                _FinalRowAssembler(
                    targets,
                    request.taxon_registry,
                    isolation_source_facets,
                ).columns
            )

            source_contract = validate_extracted_metadata_bundle(
                request.extracted_metadata,
                request.biosample_snapshot_manifest,
                request.bioproject_snapshot_manifest,
            )
            extracted_row_counts = self._count_selected_extracted_rows(
                request.extracted_metadata, taxa
            )
            assembler = _FinalRowAssembler(
                targets,
                request.taxon_registry,
                isolation_source_facets,
            )
            if StandardizationTarget.DATE in targets:
                date_standardizers = {
                    taxon: RecordDateStandardizer(source_contract.metadata_reference_date)
                    for taxon in taxa
                }
            else:
                date_standardizers = {}
            location_standardizer = None
            location_review_worklist = None
            if StandardizationTarget.LOCATION in targets:
                location_standardizer = (
                    self._location_standardizer_factory(request.location_policy, request.logger)
                    if self._location_standardizer_factory is not None
                    else LocationStandardizer(
                        request.location_policy,
                        result_logger=request.logger,
                    )
                )
                location_review_worklist = LocationReviewWorklist()
            host_standardizer = (
                self._host_standardizer_factory(request.host_policy, request.logger)
                if StandardizationTarget.HOST in targets
                or StandardizationTarget.ISOLATION_SOURCE in targets
                else None
            )
            isolation_source_standardizer = None
            if StandardizationTarget.ISOLATION_SOURCE in targets:
                if self._isolation_source_standardizer_factory is not None:
                    isolation_source_standardizer = self._isolation_source_standardizer_factory(
                        request.isolation_source_prompt_policy,
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
                        request.isolation_source_prompt_policy,
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
                host_isolation_source_standardizer_from_components(
                    host_standardizer,
                    isolation_source_standardizer,
                    host_lineage,
                )
                if host_standardizer is not None
                and isolation_source_standardizer is not None
                and host_lineage is not None
                else None
            )
            date_stats = (
                {taxon: _MutableDateStatistics() for taxon in taxa} if date_standardizers else {}
            )
            location_stats = (
                {taxon: _MutableLocationStatistics() for taxon in taxa}
                if location_standardizer is not None
                else {}
            )
            host_stats = (
                {taxon: _MutableHostStatistics() for taxon in taxa}
                if StandardizationTarget.HOST in targets
                else {}
            )
            isolation_source_stats = (
                {taxon: _MutableIsolationSourceStatistics() for taxon in taxa}
                if isolation_source_standardizer is not None
                else {}
            )
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
                    isolation_source_standardizer,
                    host_isolation_source_standardizer,
                    host_lineage,
                    date_stats,
                    location_stats,
                    location_review_worklist,
                    host_stats,
                    isolation_source_stats,
                    assembler,
                    writer,
                    reasoning_stream,
                )
            finally:
                request.progress.statistics = self._make_build_statistics(
                    destination,
                    date_stats,
                    date_standardizers,
                    location_stats,
                    (
                        location_review_worklist.write(
                            destination.parent / LOCATION_REVIEW_WORKLIST_FILENAME
                        )
                        if location_review_worklist is not None
                        else None
                    ),
                    host_stats,
                    isolation_source_stats,
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
        summaries = {
            LocationDiagnostic.RECOVERABLE_COORDINATE_FAILURE: (
                "recoverable coordinate resolution failure"
            ),
        }
        for diagnostic, description in summaries.items():
            count = statistics.location.aggregate.diagnostics.get(diagnostic, 0)
            if count:
                result_logger.warning("%s: %d record(s)", description.capitalize(), count)

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
        required_policies = required_policy_slots(targets)
        if PolicySlot.HOST in required_policies and request.host_policy is None:
            raise ValueError("Host or isolation-source standardization requires a host policy")
        if PolicySlot.LOCATION in required_policies and request.location_policy is None:
            raise ValueError("Geographic-location standardization requires a location policy")
        if (
            PolicySlot.ISOLATION_SOURCE in required_policies
            and request.isolation_source_prompt_policy is None
        ):
            raise ValueError(
                "Isolation-source standardization requires an isolation-source prompt policy"
            )
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
        isolation_source_standardizer: IsolationSourceStandardizer | None,
        host_isolation_source_standardizer: HostIsolationSourceStandardizer | None,
        host_lineage: HostLineageEnricher | None,
        date_stats: Mapping[str, _MutableDateStatistics],
        location_stats: Mapping[str, _MutableLocationStatistics],
        location_review_worklist: LocationReviewWorklist | None,
        host_stats: Mapping[str, _MutableHostStatistics],
        isolation_source_stats: Mapping[str, _MutableIsolationSourceStatistics],
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
                        stats.categories[date_outcome.category] += 1
                        stats.structures[date_outcome.structure] += 1
                        stats.precisions[date_outcome.precision] += 1
                        stats.derivations.update(date_outcome.derivations)

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
                        if (
                            host_isolation_source_result.routing.host_initial
                            is HostInitialRouting.MATCHED
                        ):
                            host_taxon_stats.standardized += 1
                            host_taxon_stats.needs_review += int(host_outcome.needs_review)
                        elif (
                            host_isolation_source_result.routing.host_initial
                            is HostInitialRouting.OVERFLOW
                        ):
                            host_taxon_stats.overflow += 1
                        else:
                            host_taxon_stats.rejected += 1
                        if (
                            host_isolation_source_result.routing.host_recovery
                            is not HostRecoveryRouting.NOT_ELIGIBLE
                        ):
                            host_taxon_stats.host_recovery_passes += 1
                            if host_outcome.standardized is not None:
                                host_taxon_stats.standardized += 1
                                host_taxon_stats.needs_review += int(host_outcome.needs_review)
                    if host_lineage is not None and host_outcome.standardized is not None:
                        lineage_outcome = host_lineage.enrich(host_outcome.standardized.taxid)

                    isolation_source_taxon_stats = isolation_source_stats[taxon]
                    isolation_source_taxon_stats.processed += 1
                    isolation_source_taxon_stats.diagnostics.update(
                        host_isolation_source_result.diagnostics.isolation_source
                    )
                    isolation_source_taxon_stats.ontology_gap_diagnostics.update(
                        isolation_source_result.ontology_gap_diagnostics
                    )
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
                    host_isolation_source_standardizer is None
                    and isolation_source_standardizer is not None
                ):
                    stats = isolation_source_stats[taxon]
                    stats.processed += 1
                    isolation_source_result = isolation_source_standardizer.standardize(
                        extracted_record,
                        overflow=(host_outcome.overflow if host_outcome is not None else None),
                    )
                    stats.diagnostics.update(isolation_source_result.diagnostics)
                    stats.ontology_gap_diagnostics.update(
                        isolation_source_result.ontology_gap_diagnostics
                    )
                    if isinstance(isolation_source_result, IsolationSourceRejection):
                        stats.rejected += 1
                    else:
                        isolation_source_outcome = isolation_source_result
                        stats.standardized += 1
                        stats.exact_matches += isolation_source_result.exact_matches
                        stats.cache_hits += isolation_source_result.cache_hits
                        stats.llm_calls += isolation_source_result.llm_calls
                        stats.evidence_levels[isolation_source_result.evidence_level] += 1

                if (
                    host_isolation_source_standardizer is None
                    and host_standardizer is not None
                    and host_outcome is not None
                    and host_outcome.standardized is None
                    and isolation_source_outcome is not None
                    and isolation_source_outcome.host_recovery_eligible
                    and isolation_source_outcome.host_recovery_pairs
                ):
                    isolation_source_stats[taxon].host_recovery_passes += 1
                    recovery_outcome = host_standardizer.recovery_pass(
                        accession,
                        "||".join(
                            pair.attribute for pair in isolation_source_outcome.host_recovery_pairs
                        ),
                        "||".join(
                            pair.value for pair in isolation_source_outcome.host_recovery_pairs
                        ),
                    )
                    host_stats[taxon].host_recovery_passes += 1
                    host_stats[taxon].diagnostics.update(recovery_outcome.diagnostics)
                    host_outcome = recovery_outcome
                    if recovery_outcome.standardized is not None:
                        host_stats[taxon].standardized += 1
                        host_stats[taxon].needs_review += int(recovery_outcome.needs_review)
                        lineage_outcome = host_lineage.enrich(recovery_outcome.standardized.taxid)

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
        date_stats: Mapping[str, _MutableDateStatistics],
        date_standardizers: Mapping[str, RecordDateStandardizer],
        location_stats: Mapping[str, _MutableLocationStatistics],
        location_review_worklist: LocationReviewWorklistSummary | None,
        host_stats: Mapping[str, _MutableHostStatistics],
        isolation_source_stats: Mapping[str, _MutableIsolationSourceStatistics],
        rows_written: int,
    ) -> DatasetBuildStatistics:
        date_statistics = DatasetBuilder._make_date_statistics(date_stats, date_standardizers)
        location_statistics = DatasetBuilder._make_location_statistics(
            location_stats,
            location_review_worklist,
        )
        host_statistics = DatasetBuilder._make_host_statistics(host_stats)
        isolation_source_statistics = DatasetBuilder._make_isolation_source_statistics(
            isolation_source_stats
        )
        return DatasetBuildStatistics(
            final_destination=destination,
            rows_written=rows_written,
            date=date_statistics,
            location=location_statistics,
            host=host_statistics,
            isolation_source=isolation_source_statistics,
        )

    @staticmethod
    def _make_date_statistics(
        mutable_stats: Mapping[str, _MutableDateStatistics],
        standardizers: Mapping[str, RecordDateStandardizer],
    ) -> DateBuildStatistics | None:
        if not mutable_stats:
            return None
        by_taxon: dict[str, DateStatistics] = {}
        aggregate_diagnostics: Counter[DateDiagnostic] = Counter()
        aggregate_rejections: Counter[str] = Counter()
        aggregate_notices: Counter[str] = Counter()
        for taxon, stats in mutable_stats.items():
            standardizer = standardizers[taxon]
            rejection_counts = dict(sorted(standardizer.rejection_counts.items()))
            notice_counts = dict(sorted(standardizer.notice_counts.items()))
            aggregate_rejections.update(rejection_counts)
            aggregate_notices.update(notice_counts)
            aggregate_diagnostics.update(getattr(standardizer, "diagnostic_counts", {}))
            by_taxon[taxon] = DateStatistics(
                processed=stats.processed,
                standardized=stats.standardized,
                rejected=stats.rejected,
                categories=dict(sorted(stats.categories.items())),
                structures=dict(sorted(stats.structures.items())),
                precisions=dict(sorted(stats.precisions.items())),
                derivations=dict(sorted(stats.derivations.items())),
                diagnostics=dict(sorted(getattr(standardizer, "diagnostic_counts", {}).items())),
                parsed_date_rejections=rejection_counts,
                notices=notice_counts,
            )
        aggregate = DateStatistics(
            processed=sum(stats.processed for stats in mutable_stats.values()),
            standardized=sum(stats.standardized for stats in mutable_stats.values()),
            rejected=sum(stats.rejected for stats in mutable_stats.values()),
            categories=_sum_counts(stats.categories for stats in mutable_stats.values()),
            structures=_sum_counts(stats.structures for stats in mutable_stats.values()),
            precisions=_sum_counts(stats.precisions for stats in mutable_stats.values()),
            derivations=_sum_counts(stats.derivations for stats in mutable_stats.values()),
            diagnostics=dict(sorted(aggregate_diagnostics.items())),
            parsed_date_rejections=dict(sorted(aggregate_rejections.items())),
            notices=dict(sorted(aggregate_notices.items())),
        )
        return DateBuildStatistics(aggregate=aggregate, by_taxon=by_taxon)

    @staticmethod
    def _make_location_statistics(
        mutable_stats: Mapping[str, _MutableLocationStatistics],
        review_worklist: LocationReviewWorklistSummary | None,
    ) -> LocationBuildStatistics | None:
        if not mutable_stats:
            return None

        def freeze(stats: _MutableLocationStatistics) -> LocationStatistics:
            return LocationStatistics(
                processed=stats.processed,
                standardized=stats.standardized,
                rejected=stats.rejected,
                coordinate_decodes=stats.coordinate_decodes,
                insdc_term_matches=stats.insdc_term_matches,
                country_conversion_matches=stats.country_conversion_matches,
                reviewed_mapping_matches=stats.reviewed_mapping_matches,
                resolution_routes=dict(sorted(stats.resolution_routes.items())),
                diagnostics=dict(sorted(stats.diagnostics.items())),
            )

        by_taxon = {taxon: freeze(stats) for taxon, stats in mutable_stats.items()}
        aggregate = LocationStatistics(
            processed=sum(stats.processed for stats in mutable_stats.values()),
            standardized=sum(stats.standardized for stats in mutable_stats.values()),
            rejected=sum(stats.rejected for stats in mutable_stats.values()),
            coordinate_decodes=sum(stats.coordinate_decodes for stats in mutable_stats.values()),
            insdc_term_matches=sum(stats.insdc_term_matches for stats in mutable_stats.values()),
            country_conversion_matches=sum(
                stats.country_conversion_matches for stats in mutable_stats.values()
            ),
            reviewed_mapping_matches=sum(
                stats.reviewed_mapping_matches for stats in mutable_stats.values()
            ),
            resolution_routes=_sum_counts(
                stats.resolution_routes for stats in mutable_stats.values()
            ),
            diagnostics=_sum_counts(stats.diagnostics for stats in mutable_stats.values()),
        )
        return LocationBuildStatistics(
            aggregate=aggregate,
            by_taxon=by_taxon,
            review_worklist=review_worklist,
        )

    @staticmethod
    def _make_host_statistics(
        mutable_stats: Mapping[str, _MutableHostStatistics],
    ) -> HostBuildStatistics | None:
        if not mutable_stats:
            return None

        def freeze(stats: _MutableHostStatistics) -> HostStatistics:
            return HostStatistics(
                processed=stats.processed,
                standardized=stats.standardized,
                rejected=stats.rejected,
                overflow=stats.overflow,
                needs_review=stats.needs_review,
                host_recovery_passes=stats.host_recovery_passes,
                diagnostics=dict(sorted(stats.diagnostics.items())),
            )

        by_taxon = {taxon: freeze(stats) for taxon, stats in mutable_stats.items()}
        aggregate_diagnostics: Counter[HostDiagnostic] = Counter()
        for stats in mutable_stats.values():
            aggregate_diagnostics.update(stats.diagnostics)
        aggregate = HostStatistics(
            processed=sum(stats.processed for stats in mutable_stats.values()),
            standardized=sum(stats.standardized for stats in mutable_stats.values()),
            rejected=sum(stats.rejected for stats in mutable_stats.values()),
            overflow=sum(stats.overflow for stats in mutable_stats.values()),
            needs_review=sum(stats.needs_review for stats in mutable_stats.values()),
            host_recovery_passes=sum(
                stats.host_recovery_passes for stats in mutable_stats.values()
            ),
            diagnostics=dict(sorted(aggregate_diagnostics.items())),
        )
        return HostBuildStatistics(aggregate=aggregate, by_taxon=by_taxon)

    @staticmethod
    def _make_isolation_source_statistics(
        mutable_stats: Mapping[str, _MutableIsolationSourceStatistics],
    ) -> IsolationSourceBuildStatistics | None:
        if not mutable_stats:
            return None

        def freeze(stats: _MutableIsolationSourceStatistics) -> IsolationSourceStatistics:
            return IsolationSourceStatistics(
                processed=stats.processed,
                standardized=stats.standardized,
                rejected=stats.rejected,
                exact_matches=stats.exact_matches,
                cache_hits=stats.cache_hits,
                llm_calls=stats.llm_calls,
                host_recovery_passes=stats.host_recovery_passes,
                evidence_levels=dict(sorted(stats.evidence_levels.items())),
                diagnostics=dict(sorted(stats.diagnostics.items())),
                invented_labels=invented_label_inventory(stats.ontology_gap_diagnostics),
            )

        by_taxon = {taxon: freeze(stats) for taxon, stats in mutable_stats.items()}
        aggregate_diagnostics: Counter[IsolationSourceDiagnostic] = Counter()
        aggregate_evidence_levels: Counter[IsolationSourceEvidenceLevel] = Counter()
        aggregate_ontology_gaps: Counter[IsolationSourceOntologyGapDiagnostic] = Counter()
        for stats in mutable_stats.values():
            aggregate_diagnostics.update(stats.diagnostics)
            aggregate_evidence_levels.update(stats.evidence_levels)
            aggregate_ontology_gaps.update(stats.ontology_gap_diagnostics)
        aggregate = IsolationSourceStatistics(
            processed=sum(stats.processed for stats in mutable_stats.values()),
            standardized=sum(stats.standardized for stats in mutable_stats.values()),
            rejected=sum(stats.rejected for stats in mutable_stats.values()),
            exact_matches=sum(stats.exact_matches for stats in mutable_stats.values()),
            cache_hits=sum(stats.cache_hits for stats in mutable_stats.values()),
            llm_calls=sum(stats.llm_calls for stats in mutable_stats.values()),
            host_recovery_passes=sum(
                stats.host_recovery_passes for stats in mutable_stats.values()
            ),
            evidence_levels=dict(sorted(aggregate_evidence_levels.items())),
            diagnostics=dict(sorted(aggregate_diagnostics.items())),
            invented_labels=invented_label_inventory(aggregate_ontology_gaps),
        )
        return IsolationSourceBuildStatistics(aggregate=aggregate, by_taxon=by_taxon)


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
        required.update(target_specifications.TARGET_SPECS[target].input_columns)
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
