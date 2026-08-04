"""Statistics vocabulary for complete, partial, and running dataset builds."""

from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

from baccurate.standardization.collection_date import DateDiagnostic
from baccurate.standardization.host import HostDiagnostic
from baccurate.standardization.isolation_source import (
    IsolationSourceDiagnostic,
    IsolationSourceEvidenceLevel,
    IsolationSourceOntologyGapDiagnostic,
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
class InventedLabelStatistics:
    """Number of occurrences of one non-existing label and their BioSample accessions."""

    occurrences: int
    accessions: Mapping[str, int]

@dataclass(frozen=True, slots=True)
class IsolationSourceStatistics:
    """Isolation-source counts, either overall or for a single pathogen."""

    processed: int
    standardized: int
    rejected: int
    exact_matches: int
    cache_hits: int
    llm_calls: int
    host_recovery_passes: int
    evidence_levels: Mapping[IsolationSourceEvidenceLevel, int]
    diagnostics: Mapping[IsolationSourceDiagnostic, int]
    invented_labels: Mapping[str, Mapping[str, InventedLabelStatistics]] = field(
        default_factory=dict
    )


@dataclass(frozen=True, slots=True)
class IsolationSourceBuildStatistics:
    """Aggregate and per-pathogen isolation-source statistics."""

    aggregate: IsolationSourceStatistics
    by_pathogen: Mapping[str, IsolationSourceStatistics]


def invented_label_inventory(
    diagnostics: Mapping[IsolationSourceOntologyGapDiagnostic, int],
) -> dict[str, dict[str, InventedLabelStatistics]]:
    """Group ontology-gap diagnostic counts by facet, label, and BioSample accession."""
    accessions_by_facet_and_label: dict[str, dict[str, Counter[str]]] = {}
    for diagnostic, occurrences in diagnostics.items():
        accessions_by_label = accessions_by_facet_and_label.setdefault(diagnostic.facet, {})
        accessions = accessions_by_label.setdefault(diagnostic.label, Counter())
        accessions[diagnostic.accession] += occurrences

    return {
        facet: {
            label: InventedLabelStatistics(
                occurrences=sum(accessions.values()),
                accessions=dict(sorted(accessions.items())),
            )
            for label, accessions in sorted(accessions_by_label.items())
        }
        for facet, accessions_by_label in sorted(accessions_by_facet_and_label.items())
    }


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
