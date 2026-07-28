"""Statistics vocabulary for complete, partial, and running dataset builds."""

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from baccurate.standardization.collection_date import DateDiagnostic
from baccurate.standardization.host import HostDiagnostic
from baccurate.standardization.isolation_source import (
    IsolationSourceDiagnostic,
    IsolationSourceEvidenceLevel,
)
from baccurate.standardization.location import LocationDiagnostic


@dataclass(slots=True)
class DatasetBuildProgress:
    """Counters and latest statistics from a running or failed build."""

    processed_rows: int = 0
    rows_written: int = 0
    statistics: "DatasetBuildStatistics | None" = None


@dataclass(frozen=True, slots=True)
class DateStatistics:
    """Collection-date counts, either overall or for a single pathogen."""

    processed: int
    standardized: int
    rejected: int
    diagnostics: Mapping[DateDiagnostic, int]
    parsed_date_rejections: Mapping[str, int]
    notices: Mapping[str, int]


@dataclass(frozen=True, slots=True)
class DateBuildStatistics:
    """Aggregate and per-pathogen date statistics."""

    aggregate: DateStatistics
    by_pathogen: Mapping[str, DateStatistics]


@dataclass(frozen=True, slots=True)
class DatasetBuildStatistics:
    """Statistics accumulated for a dataset build, complete or partial."""

    final_destination: Path
    rows_written: int
    date: DateBuildStatistics | None
    location: "LocationBuildStatistics | None"
    host: "HostBuildStatistics | None"
    isolation_source: "IsolationSourceBuildStatistics | None"


@dataclass(frozen=True, slots=True)
class LocationStatistics:
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
class LocationBuildStatistics:
    """Aggregate and per-pathogen location statistics."""

    aggregate: LocationStatistics
    by_pathogen: Mapping[str, LocationStatistics]


@dataclass(frozen=True, slots=True)
class HostStatistics:
    """Host counts, either overall or for a single pathogen."""

    processed: int
    standardized: int
    rejected: int
    overflow: int
    needs_review: int
    host_recovery_passes: int
    diagnostics: Mapping[HostDiagnostic, int]


@dataclass(frozen=True, slots=True)
class HostBuildStatistics:
    """Aggregate and per-pathogen host statistics."""

    aggregate: HostStatistics
    by_pathogen: Mapping[str, HostStatistics]


@dataclass(frozen=True, slots=True)
class IsolationSourceStatistics:
    """Isolation-source counts, either overall or for a single pathogen."""

    processed: int
    standardized: int
    rejected: int
    exact_matches: int
    cache_hits: int
    llm_calls: int
    host_contexts: int
    host_recovery_passes: int
    evidence_levels: Mapping[IsolationSourceEvidenceLevel, int]
    diagnostics: Mapping[IsolationSourceDiagnostic, int]


@dataclass(frozen=True, slots=True)
class IsolationSourceBuildStatistics:
    """Aggregate and per-pathogen isolation-source statistics."""

    aggregate: IsolationSourceStatistics
    by_pathogen: Mapping[str, IsolationSourceStatistics]


def processed_rows(statistics: DatasetBuildStatistics) -> int:
    """Return the processed-row count exposed by standardization-target statistics."""
    counts = [
        target_statistics.aggregate.processed
        for target_statistics in (
            statistics.date,
            statistics.location,
            statistics.host,
            statistics.isolation_source,
        )
        if target_statistics is not None
    ]
    return max(counts, default=statistics.rows_written)
