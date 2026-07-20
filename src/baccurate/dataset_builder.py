"""Build one final standardized dataset from extracted metadata."""

import csv
import json
import logging
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from contextlib import ExitStack
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Protocol, TextIO

from baccurate.atb import load_atb_accessions_by_pathogen
from baccurate.llm.client import LLMSettings, load_llm_settings
from baccurate.pathogens import scientific_name
from baccurate.paths import CONFIG_DIR, DEFAULT_NAMES_DMP, DEFAULT_NODES_DMP
from baccurate.source_snapshot import validate_derived_metadata_source
from baccurate.standardizers._date_record import DateDiagnostic, DateOutcome, RecordDateStandardizer
from baccurate.standardizers.host import HostDiagnostic, HostOutcome, HostStandardizer
from baccurate.standardizers.isolation import (
    IsolationDiagnostic,
    IsolationEvidenceLevel,
    IsolationOutcome,
    IsolationRejection,
    IsoStandardizer,
)
from baccurate.standardizers.location import (
    LocationDiagnostic,
    LocationOutcome,
    LocationRejection,
    LocationStandardizer,
)
from baccurate.taxonomy.lineage import HostLineage, HostLineageEnricher
from baccurate.utils.progress import make_progress_bar

logger = logging.getLogger(__name__)


class StandardizationAttribute(StrEnum):
    """A metadata attribute this pipeline can standardize."""

    HOST = "host"
    DATE = "date"
    LOCATION = "loc"
    ISOLATION_SOURCE = "iso"


@dataclass(slots=True)
class DatasetBuildProgress:
    """Counters and latest report from a running or failed build."""

    processed_rows: int = 0
    rows_written: int = 0
    report: "DatasetBuildReport | None" = None


@dataclass(frozen=True, slots=True)
class DatasetBuildRequest:
    """Inputs, selection, destination, and runtime settings for one build."""

    extracted_metadata: Path
    biosample_snapshot_manifest: Path
    bioproject_snapshot_manifest: Path
    requested_pathogens: tuple[str, ...]
    requested_attributes: tuple[StandardizationAttribute, ...]
    final_destination: Path
    atb_index: Path
    location_config: Path = CONFIG_DIR / "location.yaml"
    host_config: Path = CONFIG_DIR / "host.yaml"
    isolation_config: Path = CONFIG_DIR / "isolation_source.yaml"
    isolation_reasoning_destination: Path | None = None
    names_dmp: Path = DEFAULT_NAMES_DMP
    nodes_dmp: Path = DEFAULT_NODES_DMP
    overwrite: bool = False
    skip_llm: bool = False
    llm_settings: LLMSettings | None = None
    logger: logging.Logger = field(default_factory=lambda: logger)
    disable_progress: bool = False
    progress: DatasetBuildProgress = field(default_factory=DatasetBuildProgress)


@dataclass(frozen=True, slots=True)
class DateBuildStatistics:
    """Collection-date counts, either overall or for a single pathogen."""

    processed: int
    standardized: int
    rejected: int
    diagnostics: Mapping[DateDiagnostic, int]
    candidate_rejections: Mapping[str, int]
    notices: Mapping[str, int]


@dataclass(frozen=True, slots=True)
class DateBuildReport:
    """Aggregate and per-pathogen date statistics."""

    aggregate: DateBuildStatistics
    by_pathogen: Mapping[str, DateBuildStatistics]


@dataclass(frozen=True, slots=True)
class DatasetBuildReport:
    """Structured result of a successful dataset build."""

    final_destination: Path
    rows_written: int
    date: DateBuildReport | None
    location: "LocationBuildReport | None"
    host: "HostBuildReport | None"
    isolation: "IsolationBuildReport | None"


@dataclass(frozen=True, slots=True)
class LocationBuildStatistics:
    """Location counts, either overall or for a single pathogen."""

    processed: int
    standardized: int
    rejected: int
    coordinate_decodes: int
    direct_matches: int
    cache_hits: int
    llm_calls: int
    diagnostics: Mapping[LocationDiagnostic, int]


@dataclass(frozen=True, slots=True)
class LocationBuildReport:
    """Aggregate and per-pathogen location statistics."""

    aggregate: LocationBuildStatistics
    by_pathogen: Mapping[str, LocationBuildStatistics]


@dataclass(frozen=True, slots=True)
class HostBuildStatistics:
    """Host counts, either overall or for a single pathogen."""

    processed: int
    matched: int
    rejected: int
    overflow: int
    low_confidence: int
    retry_eligible: int
    diagnostics: Mapping[HostDiagnostic, int]


@dataclass(frozen=True, slots=True)
class HostBuildReport:
    """Aggregate and per-pathogen host statistics."""

    aggregate: HostBuildStatistics
    by_pathogen: Mapping[str, HostBuildStatistics]


@dataclass(frozen=True, slots=True)
class IsolationBuildStatistics:
    """Isolation-source counts, either overall or for a single pathogen."""

    processed: int
    matched: int
    rejected: int
    exact_matches: int
    cache_hits: int
    llm_calls: int
    host_contexts: int
    host_retries: int
    evidence_levels: Mapping[IsolationEvidenceLevel, int]
    diagnostics: Mapping[IsolationDiagnostic, int]


@dataclass(frozen=True, slots=True)
class IsolationBuildReport:
    """Aggregate and per-pathogen isolation-source statistics."""

    aggregate: IsolationBuildStatistics
    by_pathogen: Mapping[str, IsolationBuildStatistics]


@dataclass(slots=True)
class _MutableDateStatistics:
    processed: int = 0
    standardized: int = 0
    rejected: int = 0


@dataclass(slots=True)
class _MutableLocationStatistics:
    processed: int = 0
    standardized: int = 0
    rejected: int = 0
    coordinate_decodes: int = 0
    direct_matches: int = 0
    cache_hits: int = 0
    llm_calls: int = 0
    diagnostics: Counter[LocationDiagnostic] = field(default_factory=Counter)


@dataclass(slots=True)
class _MutableHostStatistics:
    processed: int = 0
    matched: int = 0
    rejected: int = 0
    overflow: int = 0
    low_confidence: int = 0
    retry_eligible: int = 0
    diagnostics: Counter[HostDiagnostic] = field(default_factory=Counter)


@dataclass(slots=True)
class _MutableIsolationStatistics:
    processed: int = 0
    matched: int = 0
    rejected: int = 0
    exact_matches: int = 0
    cache_hits: int = 0
    llm_calls: int = 0
    host_contexts: int = 0
    host_retries: int = 0
    evidence_levels: Counter[IsolationEvidenceLevel] = field(default_factory=Counter)
    diagnostics: Counter[IsolationDiagnostic] = field(default_factory=Counter)


@dataclass(frozen=True, slots=True)
class _FinalRow:
    accession: str
    pathogen: str
    pathogen_sci_name: str
    in_atb: bool
    bioproject: str
    date: DateOutcome | None
    location: LocationOutcome | None
    isolation: IsolationOutcome | None
    host: HostOutcome | None
    host_lineage: HostLineage | None


class _RowWriter(Protocol):
    def writerow(self, row: Iterable[object], /) -> object:
        """Write one projected dataset row."""


class _RecordAssembler:
    """Turn a record and its standardized outcomes into an output row,
    keeping only the columns for the selected attributes.
    """

    base_columns = (
        "accession",
        "pathogen",
        "pathogen_sci_name",
        "in_ATB",
        "bioproject",
    )
    date_columns = (
        "date_attr_orig",
        "date_val_orig",
        "date_start",
        "date_end",
        "date_score",
    )
    location_columns = (
        "loc_attr_orig",
        "loc_val_orig",
        "loc_continent",
        "loc_UNregion",
        "loc_country",
        "loc_other",
    )
    host_columns = (
        "host_attr_orig",
        "host_val_orig",
        "host_taxid",
        "host_sci_name",
        "host_common_names",
        "host_lineage_names",
        "host_lineage_taxids",
        "host_score",
        "host_low_conf",
    )
    isolation_columns = (
        "iso_attr_orig",
        "iso_val_orig",
        "iso_host",
        "iso_terms",
        "iso_display_term",
        "iso_ontology_id",
    )

    def __init__(
        self,
        atb_by_pathogen: Mapping[str, set[str]],
        attributes: Sequence[StandardizationAttribute],
    ):
        self.atb_by_pathogen = atb_by_pathogen
        self._selected_attributes = tuple(attributes)
        selected = set(attributes)
        self.columns = self.base_columns
        if StandardizationAttribute.DATE in selected:
            self.columns += self.date_columns
        if StandardizationAttribute.LOCATION in selected:
            self.columns += self.location_columns
        if StandardizationAttribute.ISOLATION_SOURCE in selected:
            self.columns += self.isolation_columns
        if StandardizationAttribute.HOST in selected:
            self.columns += self.host_columns

    def assemble(
        self,
        record: Mapping[str, str],
        date: DateOutcome | None,
        location: LocationOutcome | None,
        isolation: IsolationOutcome | None,
        host: HostOutcome | None,
        host_lineage: HostLineage | None,
    ) -> _FinalRow:
        pathogen = record["pathogen"]
        accession = record["accession"]
        return _FinalRow(
            accession=accession,
            pathogen=pathogen,
            pathogen_sci_name=scientific_name(pathogen),
            in_atb=accession in self.atb_by_pathogen.get(pathogen, set()),
            bioproject=record.get("bioproject_accession", ""),
            date=date,
            location=location,
            isolation=isolation,
            host=host,
            host_lineage=host_lineage,
        )

    def project(self, row: _FinalRow) -> tuple[object, ...]:
        values: tuple[object, ...] = (
            row.accession,
            row.pathogen,
            row.pathogen_sci_name,
            row.in_atb,
            row.bioproject,
        )
        if StandardizationAttribute.DATE in self._selected_attributes:
            if row.date is None:
                values += ("", "", "", "", "")
            else:
                attributes = "||".join(origin.attribute for origin in row.date.origins)
                origins = "||".join(origin.value for origin in row.date.origins)
                values += (
                    attributes,
                    origins,
                    row.date.bounds.start.isoformat(),
                    row.date.bounds.end.isoformat(),
                    row.date.score,
                )
        if StandardizationAttribute.LOCATION in self._selected_attributes:
            if row.location is None:
                values += ("", "", "", "", "", "")
            else:
                values += (
                    "||".join(origin.attribute for origin in row.location.origins),
                    "||".join(origin.value for origin in row.location.origins),
                    row.location.continent,
                    row.location.un_region,
                    row.location.country,
                    row.location.sublocation or "NA",
                )
        if StandardizationAttribute.ISOLATION_SOURCE in self._selected_attributes:
            if row.isolation is None:
                values += ("", "", "", "", "", "")
            else:
                values += (
                    "||".join(origin.attribute for origin in row.isolation.origins),
                    "||".join(origin.value for origin in row.isolation.origins),
                    row.isolation.host_context,
                    row.isolation.term_paths,
                    row.isolation.display_terms,
                    row.isolation.ontology_links,
                )
        if StandardizationAttribute.HOST in self._selected_attributes:
            if row.host is None or row.host.standardized is None:
                values += ("", "", "", "", "", "", "", "", "")
            else:
                origin = row.host.origins[0]
                lineage = row.host_lineage
                if lineage is None:
                    raise ValueError(f"Missing host lineage enrichment for {row.accession}")
                values += (
                    origin.attribute,
                    origin.value,
                    row.host.standardized.taxid,
                    row.host.standardized.scientific_name,
                    lineage.common_names,
                    lineage.lineage_names,
                    lineage.lineage_taxids,
                    row.host.score,
                    row.host.low_confidence,
                )
        return values


class DatasetBuilder:
    """Stream one final dataset for the requested records and attributes."""

    def __init__(
        self,
        location_standardizer_factory: (
            Callable[[Path, logging.Logger], LocationStandardizer] | None
        ) = None,
        host_standardizer_factory: (
            Callable[[Path, logging.Logger], HostStandardizer] | None
        ) = None,
        host_lineage_factory: Callable[[Path, Path], HostLineageEnricher] | None = None,
        isolation_standardizer_factory: (
            Callable[[Path, Path, logging.Logger], IsoStandardizer] | None
        ) = None,
    ) -> None:
        self._location_standardizer_factory = location_standardizer_factory
        self._host_standardizer_factory = host_standardizer_factory or (
            lambda path, result_logger: HostStandardizer(path, result_logger=result_logger)
        )
        self._host_lineage_factory = host_lineage_factory or HostLineageEnricher
        self._isolation_standardizer_factory = isolation_standardizer_factory

    def build(self, request: DatasetBuildRequest) -> DatasetBuildReport:
        pathogens, attributes = self._validate_request(request)
        llm_settings = request.llm_settings or load_llm_settings()
        request.progress.processed_rows = 0
        request.progress.rows_written = 0
        destination = Path(request.final_destination)
        reasoning_destination = (
            Path(request.isolation_reasoning_destination)
            if StandardizationAttribute.ISOLATION_SOURCE in attributes
            and request.isolation_reasoning_destination is not None
            else None
        )
        destinations = tuple(
            path for path in (destination, reasoning_destination) if path is not None
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
            writer.writerow(_RecordAssembler({}, attributes).columns)

            source_contract = validate_derived_metadata_source(
                request.extracted_metadata,
                request.biosample_snapshot_manifest,
                request.bioproject_snapshot_manifest,
            )
            row_counts = self._count_selected_rows(request.extracted_metadata, pathogens)
            atb_by_pathogen = load_atb_accessions_by_pathogen(request.atb_index)
            assembler = _RecordAssembler(atb_by_pathogen, attributes)
            date_standardizers = (
                {
                    pathogen: RecordDateStandardizer(source_contract.metadata_reference_date)
                    for pathogen in pathogens
                }
                if StandardizationAttribute.DATE in attributes
                else {}
            )
            location_standardizer = None
            if StandardizationAttribute.LOCATION in attributes:
                if self._location_standardizer_factory is not None:
                    location_standardizer = self._location_standardizer_factory(
                        request.location_config,
                        request.logger,
                    )
                else:
                    location_options = {
                        "result_logger": request.logger,
                        "llm_settings": llm_settings,
                    }
                    if request.skip_llm:
                        location_options["client"] = None
                    location_standardizer = LocationStandardizer(
                        request.location_config, **location_options
                    )
            if location_standardizer is not None:
                resources.callback(location_standardizer.close)
            host_standardizer = (
                self._host_standardizer_factory(request.host_config, request.logger)
                if StandardizationAttribute.HOST in attributes
                else None
            )
            isolation_standardizer = None
            if StandardizationAttribute.ISOLATION_SOURCE in attributes:
                if self._isolation_standardizer_factory is not None:
                    isolation_standardizer = self._isolation_standardizer_factory(
                        request.isolation_config,
                        request.extracted_metadata,
                        request.logger,
                    )
                else:
                    isolation_options = {
                        "result_logger": request.logger,
                        "llm_settings": llm_settings,
                    }
                    if request.skip_llm:
                        isolation_options["client"] = None
                    isolation_standardizer = IsoStandardizer(
                        request.isolation_config,
                        request.extracted_metadata,
                        **isolation_options,
                    )
            if isolation_standardizer is not None:
                resources.callback(isolation_standardizer.close)
            host_lineage = (
                self._host_lineage_factory(request.names_dmp, request.nodes_dmp)
                if host_standardizer is not None
                else None
            )
            date_stats = (
                {pathogen: _MutableDateStatistics() for pathogen in pathogens}
                if date_standardizers
                else {}
            )
            location_stats = (
                {pathogen: _MutableLocationStatistics() for pathogen in pathogens}
                if location_standardizer is not None
                else {}
            )
            host_stats = (
                {pathogen: _MutableHostStatistics() for pathogen in pathogens}
                if host_standardizer is not None
                else {}
            )
            isolation_stats = (
                {pathogen: _MutableIsolationStatistics() for pathogen in pathogens}
                if isolation_standardizer is not None
                else {}
            )
            rows_written = 0
            try:
                rows_written = self._process_records(
                    request,
                    pathogens,
                    attributes,
                    row_counts,
                    date_standardizers,
                    location_standardizer,
                    host_standardizer,
                    isolation_standardizer,
                    host_lineage,
                    date_stats,
                    location_stats,
                    host_stats,
                    isolation_stats,
                    assembler,
                    writer,
                    reasoning_stream,
                )
            finally:
                request.progress.report = self._make_report(
                    destination,
                    date_stats,
                    date_standardizers,
                    location_stats,
                    host_stats,
                    isolation_stats,
                    request.progress.rows_written,
                )
        report = request.progress.report
        self._warn_recoverable_conditions(request.logger, report)
        request.logger.info("Built final dataset with %d rows at %s", rows_written, destination)
        return report

    @staticmethod
    def _warn_recoverable_conditions(
        result_logger: logging.Logger,
        report: DatasetBuildReport,
    ) -> None:
        if report.location is None:
            return
        summaries = {
            LocationDiagnostic.LLM_DISABLED: "location LLM disabled",
            LocationDiagnostic.RECOVERABLE_LLM_FAILURE: "recoverable location LLM failure",
            LocationDiagnostic.RECOVERABLE_COORDINATE_FAILURE: (
                "recoverable coordinate resolution failure"
            ),
            LocationDiagnostic.INVALID_LLM_RESPONSE: "invalid location LLM response",
        }
        for diagnostic, description in summaries.items():
            count = report.location.aggregate.diagnostics.get(diagnostic, 0)
            if count:
                result_logger.warning("%s: %d record(s)", description.capitalize(), count)

    @staticmethod
    def _validate_request(
        request: DatasetBuildRequest,
    ) -> tuple[tuple[str, ...], tuple[StandardizationAttribute, ...]]:
        pathogens = tuple(request.requested_pathogens)
        if not pathogens:
            raise ValueError("Need at least one pathogen")
        if len(set(pathogens)) != len(pathogens):
            raise ValueError("Pathogens must be unique")
        attributes = tuple(request.requested_attributes)
        if not attributes:
            raise ValueError("Need at least one standardization attribute")
        if len(set(attributes)) != len(attributes):
            raise ValueError("Standardization attributes must be unique")
        return pathogens, attributes

    @staticmethod
    def _count_selected_rows(
        input_path: Path,
        pathogens: Sequence[str],
    ) -> dict[str, int]:
        counts = dict.fromkeys(pathogens, 0)
        with Path(input_path).open("r", encoding="utf-8", newline="") as stream:
            for record in csv.DictReader(stream, delimiter="\t"):
                pathogen = (record.get("pathogen") or "").strip()
                if pathogen in counts:
                    counts[pathogen] += 1
        return counts

    @staticmethod
    def _process_records(
        request: DatasetBuildRequest,
        pathogens: Sequence[str],
        attributes: Sequence[StandardizationAttribute],
        row_counts: Mapping[str, int],
        date_standardizers: Mapping[str, RecordDateStandardizer],
        location_standardizer: LocationStandardizer | None,
        host_standardizer: HostStandardizer | None,
        isolation_standardizer: IsoStandardizer | None,
        host_lineage: HostLineageEnricher | None,
        date_stats: Mapping[str, _MutableDateStatistics],
        location_stats: Mapping[str, _MutableLocationStatistics],
        host_stats: Mapping[str, _MutableHostStatistics],
        isolation_stats: Mapping[str, _MutableIsolationStatistics],
        assembler: _RecordAssembler,
        writer: _RowWriter,
        reasoning_stream: TextIO | None,
    ) -> int:
        selected = set(pathogens)
        rows_written = 0
        total = sum(row_counts.values())
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
            _require_columns(reader.fieldnames, attributes)
            for source_position, record in enumerate(reader, start=1):
                accession = (record.get("accession") or "").strip()
                pathogen = (record.get("pathogen") or "").strip()
                if not accession:
                    raise ValueError(f"Record {source_position} has no accession")
                if not pathogen:
                    raise ValueError(f"Record {accession} has no pathogen")
                if pathogen not in selected:
                    continue
                date_outcome = None
                if pathogen in date_standardizers:
                    stats = date_stats[pathogen]
                    stats.processed += 1
                    date_outcome = date_standardizers[pathogen].standardize(record)
                    if date_outcome is None:
                        stats.rejected += 1
                    else:
                        stats.standardized += 1

                location_outcome = None
                if location_standardizer is not None:
                    stats = location_stats[pathogen]
                    stats.processed += 1
                    location_result = location_standardizer.standardize(record)
                    stats.coordinate_decodes += location_result.coordinate_decodes
                    stats.direct_matches += location_result.direct_matches
                    stats.cache_hits += location_result.cache_hits
                    stats.llm_calls += location_result.llm_calls
                    stats.diagnostics.update(location_result.diagnostics)
                    if isinstance(location_result, LocationRejection):
                        stats.rejected += 1
                    else:
                        stats.standardized += 1
                        location_outcome = location_result

                host_outcome = None
                lineage_outcome = None
                if host_standardizer is not None:
                    stats = host_stats[pathogen]
                    stats.processed += 1
                    host_result = host_standardizer.standardize(record)
                    host_outcome = host_result
                    stats.diagnostics.update(host_result.diagnostics)
                    if host_result.standardized is not None:
                        stats.matched += 1
                        stats.low_confidence += int(host_result.low_confidence)
                        lineage_outcome = host_lineage.enrich(host_result.standardized.taxid)
                    elif host_result.overflow is not None:
                        stats.overflow += 1
                    else:
                        stats.rejected += 1
                    stats.retry_eligible += int(host_result.retry_eligible)

                isolation_outcome = None
                if isolation_standardizer is not None:
                    stats = isolation_stats[pathogen]
                    stats.processed += 1
                    standardized_host = (
                        host_outcome.standardized.scientific_name
                        if host_outcome is not None and host_outcome.standardized is not None
                        else ""
                    )
                    host_context = (
                        standardized_host or str(record.get("host_val_orig", "") or "").strip()
                    )
                    isolation_result = isolation_standardizer.standardize(
                        record,
                        host_context=host_context,
                        overflow=(host_outcome.overflow if host_outcome is not None else None),
                    )
                    stats.host_contexts += int(bool(host_context))
                    stats.diagnostics.update(isolation_result.diagnostics)
                    if isinstance(isolation_result, IsolationRejection):
                        stats.rejected += 1
                    else:
                        isolation_outcome = isolation_result
                        stats.matched += 1
                        stats.exact_matches += isolation_result.exact_matches
                        stats.cache_hits += isolation_result.cache_hits
                        stats.llm_calls += isolation_result.llm_calls
                        stats.evidence_levels[isolation_result.evidence_level] += 1

                if (
                    host_standardizer is not None
                    and host_outcome is not None
                    and host_outcome.standardized is None
                    and isolation_outcome is not None
                    and isolation_outcome.host_retry_eligible
                    and isolation_outcome.sample_origins
                ):
                    isolation_stats[pathogen].host_retries += 1
                    retry_result = host_standardizer.retry(
                        accession,
                        "||".join(origin.attribute for origin in isolation_outcome.sample_origins),
                        "||".join(origin.value for origin in isolation_outcome.sample_origins),
                    )
                    host_stats[pathogen].retry_eligible += 1
                    host_stats[pathogen].diagnostics.update(retry_result.diagnostics)
                    host_outcome = retry_result
                    if retry_result.standardized is not None:
                        host_stats[pathogen].matched += 1
                        host_stats[pathogen].low_confidence += int(retry_result.low_confidence)
                        lineage_outcome = host_lineage.enrich(retry_result.standardized.taxid)

                if (
                    date_outcome is not None
                    or location_outcome is not None
                    or isolation_outcome is not None
                    or (host_outcome is not None and host_outcome.standardized is not None)
                ):
                    reasoning_json = (
                        _isolation_reasoning_json(accession, pathogen, isolation_outcome)
                        if isolation_outcome is not None and reasoning_stream is not None
                        else None
                    )
                    row = assembler.assemble(
                        record,
                        date_outcome,
                        location_outcome,
                        isolation_outcome,
                        host_outcome,
                        lineage_outcome,
                    )
                    writer.writerow(assembler.project(row))
                    if reasoning_json is not None and reasoning_stream is not None:
                        reasoning_stream.write(reasoning_json + "\n")
                    rows_written += 1
                    request.progress.rows_written = rows_written
                request.progress.processed_rows += 1
                progress_bar.update(1)
        return rows_written

    @staticmethod
    def _make_report(
        destination: Path,
        date_stats: Mapping[str, _MutableDateStatistics],
        date_standardizers: Mapping[str, RecordDateStandardizer],
        location_stats: Mapping[str, _MutableLocationStatistics],
        host_stats: Mapping[str, _MutableHostStatistics],
        isolation_stats: Mapping[str, _MutableIsolationStatistics],
        rows_written: int,
    ) -> DatasetBuildReport:
        date_report = DatasetBuilder._make_date_report(date_stats, date_standardizers)
        location_report = DatasetBuilder._make_location_report(location_stats)
        host_report = DatasetBuilder._make_host_report(host_stats)
        isolation_report = DatasetBuilder._make_isolation_report(isolation_stats)
        return DatasetBuildReport(
            final_destination=destination,
            rows_written=rows_written,
            date=date_report,
            location=location_report,
            host=host_report,
            isolation=isolation_report,
        )

    @staticmethod
    def _make_date_report(
        mutable_stats: Mapping[str, _MutableDateStatistics],
        standardizers: Mapping[str, RecordDateStandardizer],
    ) -> DateBuildReport | None:
        if not mutable_stats:
            return None
        by_pathogen: dict[str, DateBuildStatistics] = {}
        aggregate_diagnostics: Counter[DateDiagnostic] = Counter()
        aggregate_rejections: Counter[str] = Counter()
        aggregate_notices: Counter[str] = Counter()
        for pathogen, stats in mutable_stats.items():
            standardizer = standardizers[pathogen]
            rejection_counts = dict(sorted(standardizer.rejection_counts.items()))
            notice_counts = dict(sorted(standardizer.notice_counts.items()))
            aggregate_rejections.update(rejection_counts)
            aggregate_notices.update(notice_counts)
            aggregate_diagnostics.update(getattr(standardizer, "diagnostic_counts", {}))
            by_pathogen[pathogen] = DateBuildStatistics(
                processed=stats.processed,
                standardized=stats.standardized,
                rejected=stats.rejected,
                diagnostics=dict(sorted(getattr(standardizer, "diagnostic_counts", {}).items())),
                candidate_rejections=rejection_counts,
                notices=notice_counts,
            )
        aggregate = DateBuildStatistics(
            processed=sum(stats.processed for stats in mutable_stats.values()),
            standardized=sum(stats.standardized for stats in mutable_stats.values()),
            rejected=sum(stats.rejected for stats in mutable_stats.values()),
            diagnostics=dict(sorted(aggregate_diagnostics.items())),
            candidate_rejections=dict(sorted(aggregate_rejections.items())),
            notices=dict(sorted(aggregate_notices.items())),
        )
        return DateBuildReport(aggregate=aggregate, by_pathogen=by_pathogen)

    @staticmethod
    def _make_location_report(
        mutable_stats: Mapping[str, _MutableLocationStatistics],
    ) -> LocationBuildReport | None:
        if not mutable_stats:
            return None

        def freeze(stats: _MutableLocationStatistics) -> LocationBuildStatistics:
            return LocationBuildStatistics(
                processed=stats.processed,
                standardized=stats.standardized,
                rejected=stats.rejected,
                coordinate_decodes=stats.coordinate_decodes,
                direct_matches=stats.direct_matches,
                cache_hits=stats.cache_hits,
                llm_calls=stats.llm_calls,
                diagnostics=dict(sorted(stats.diagnostics.items())),
            )

        by_pathogen = {pathogen: freeze(stats) for pathogen, stats in mutable_stats.items()}
        aggregate = LocationBuildStatistics(
            processed=sum(stats.processed for stats in mutable_stats.values()),
            standardized=sum(stats.standardized for stats in mutable_stats.values()),
            rejected=sum(stats.rejected for stats in mutable_stats.values()),
            coordinate_decodes=sum(stats.coordinate_decodes for stats in mutable_stats.values()),
            direct_matches=sum(stats.direct_matches for stats in mutable_stats.values()),
            cache_hits=sum(stats.cache_hits for stats in mutable_stats.values()),
            llm_calls=sum(stats.llm_calls for stats in mutable_stats.values()),
            diagnostics=dict(
                sorted(
                    sum((stats.diagnostics for stats in mutable_stats.values()), Counter()).items()
                )
            ),
        )
        return LocationBuildReport(aggregate=aggregate, by_pathogen=by_pathogen)

    @staticmethod
    def _make_host_report(
        mutable_stats: Mapping[str, _MutableHostStatistics],
    ) -> HostBuildReport | None:
        if not mutable_stats:
            return None

        def freeze(stats: _MutableHostStatistics) -> HostBuildStatistics:
            return HostBuildStatistics(
                processed=stats.processed,
                matched=stats.matched,
                rejected=stats.rejected,
                overflow=stats.overflow,
                low_confidence=stats.low_confidence,
                retry_eligible=stats.retry_eligible,
                diagnostics=dict(sorted(stats.diagnostics.items())),
            )

        by_pathogen = {pathogen: freeze(stats) for pathogen, stats in mutable_stats.items()}
        aggregate_diagnostics: Counter[HostDiagnostic] = Counter()
        for stats in mutable_stats.values():
            aggregate_diagnostics.update(stats.diagnostics)
        aggregate = HostBuildStatistics(
            processed=sum(stats.processed for stats in mutable_stats.values()),
            matched=sum(stats.matched for stats in mutable_stats.values()),
            rejected=sum(stats.rejected for stats in mutable_stats.values()),
            overflow=sum(stats.overflow for stats in mutable_stats.values()),
            low_confidence=sum(stats.low_confidence for stats in mutable_stats.values()),
            retry_eligible=sum(stats.retry_eligible for stats in mutable_stats.values()),
            diagnostics=dict(sorted(aggregate_diagnostics.items())),
        )
        return HostBuildReport(aggregate=aggregate, by_pathogen=by_pathogen)

    @staticmethod
    def _make_isolation_report(
        mutable_stats: Mapping[str, _MutableIsolationStatistics],
    ) -> IsolationBuildReport | None:
        if not mutable_stats:
            return None

        def freeze(stats: _MutableIsolationStatistics) -> IsolationBuildStatistics:
            return IsolationBuildStatistics(
                processed=stats.processed,
                matched=stats.matched,
                rejected=stats.rejected,
                exact_matches=stats.exact_matches,
                cache_hits=stats.cache_hits,
                llm_calls=stats.llm_calls,
                host_contexts=stats.host_contexts,
                host_retries=stats.host_retries,
                evidence_levels=dict(sorted(stats.evidence_levels.items())),
                diagnostics=dict(sorted(stats.diagnostics.items())),
            )

        by_pathogen = {pathogen: freeze(stats) for pathogen, stats in mutable_stats.items()}
        aggregate_diagnostics: Counter[IsolationDiagnostic] = Counter()
        aggregate_evidence_levels: Counter[IsolationEvidenceLevel] = Counter()
        for stats in mutable_stats.values():
            aggregate_diagnostics.update(stats.diagnostics)
            aggregate_evidence_levels.update(stats.evidence_levels)
        aggregate = IsolationBuildStatistics(
            processed=sum(stats.processed for stats in mutable_stats.values()),
            matched=sum(stats.matched for stats in mutable_stats.values()),
            rejected=sum(stats.rejected for stats in mutable_stats.values()),
            exact_matches=sum(stats.exact_matches for stats in mutable_stats.values()),
            cache_hits=sum(stats.cache_hits for stats in mutable_stats.values()),
            llm_calls=sum(stats.llm_calls for stats in mutable_stats.values()),
            host_contexts=sum(stats.host_contexts for stats in mutable_stats.values()),
            host_retries=sum(stats.host_retries for stats in mutable_stats.values()),
            evidence_levels=dict(sorted(aggregate_evidence_levels.items())),
            diagnostics=dict(sorted(aggregate_diagnostics.items())),
        )
        return IsolationBuildReport(aggregate=aggregate, by_pathogen=by_pathogen)


def _require_columns(
    fieldnames: Sequence[str] | None,
    attributes: Sequence[StandardizationAttribute],
) -> None:
    required = {
        "accession",
        "pathogen",
        "bioproject_accession",
    }
    if StandardizationAttribute.DATE in attributes:
        required.update({"date_attr_orig", "date_val_orig", "date_category"})
    if StandardizationAttribute.LOCATION in attributes:
        required.update({"loc_attr_orig", "loc_val_orig"})
    if StandardizationAttribute.HOST in attributes:
        required.update({"host_attr_orig", "host_val_orig"})
    if StandardizationAttribute.ISOLATION_SOURCE in attributes:
        required.update({"iso_attr_orig", "iso_val_orig", "host_val_orig"})
    available = set(fieldnames or ())
    missing = sorted(required - available)
    if missing:
        raise ValueError(f"Extracted metadata is missing required columns: {', '.join(missing)}")


def _isolation_reasoning_json(
    accession: str,
    pathogen: str,
    outcome: IsolationOutcome,
) -> str:
    """Render a record's isolation-source reasoning as a JSON line."""
    record = {
        "accession": accession,
        "pathogen": pathogen,
        "origins": [
            {"attribute": origin.attribute, "value": origin.value} for origin in outcome.origins
        ],
        "host_context": outcome.host_context,
        "categories": outcome.categories,
        "term_paths": outcome.term_paths,
        "display_terms": outcome.display_terms,
        "ontology_links": outcome.ontology_links,
        "evidence_level": outcome.evidence_level.value,
        "diagnostics": [diagnostic.value for diagnostic in outcome.diagnostics],
        "reasoning": list(outcome.reasoning),
    }
    return json.dumps(record, ensure_ascii=False)
