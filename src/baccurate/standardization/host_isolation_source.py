"""
Coordinate host and isolation-source standardization for one record.

A resolved host becomes context for the isolation-source prompt. If the classification
selects a host-implying ontology branch, the isolation-source attributes probably name
an organism. The host standardizer then makes a recovery pass over those submitted
values.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum

from baccurate.standardization.host import HostDiagnostic, HostOutcome, HostStandardizer
from baccurate.standardization.host_lineage import HostLineageEnricher
from baccurate.standardization.isolation_source import (
    HOST_SOURCE_TYPE_BY_LINEAGE_ROOT,
    IsolationSourceDiagnostic,
    IsolationSourceEvidenceLevel,
    IsolationSourceOutcome,
    IsolationSourceRejection,
    IsolationSourceStandardizer,
)

# --- Routing classification ---


class HostInitialRouting(StrEnum):
    """How the initial host pass resolved a record."""

    MATCHED = "matched"
    OVERFLOW = "overflow"
    REJECTED = "rejected"


class HostRecoveryRouting(StrEnum):
    """Whether eligible isolation-source evidence caused a host recovery pass."""

    NOT_ELIGIBLE = "not_eligible"
    RESOLVED = "resolved"
    UNRESOLVED = "unresolved"


@dataclass(frozen=True, slots=True)
class HostIsolationSourceRouting:
    """Coverage classification for the host--isolation-source collaboration."""

    host_initial: HostInitialRouting
    host_recovery: HostRecoveryRouting


# --- Result contract ---


@dataclass(frozen=True, slots=True)
class HostIsolationSourceDiagnostics:
    """All diagnostics emitted while producing the final result."""

    host: tuple[HostDiagnostic, ...]
    isolation_source: tuple[IsolationSourceDiagnostic, ...]


@dataclass(frozen=True, slots=True)
class HostIsolationSourceResult:
    """Immutable final result of coordinated host--isolation-source standardization."""

    host: HostOutcome
    isolation_source: IsolationSourceOutcome | IsolationSourceRejection
    diagnostics: HostIsolationSourceDiagnostics
    routing: HostIsolationSourceRouting


# --- Main class ---


class HostIsolationSourceStandardizer:
    """Standardize one BioSample record through the coordinated host--isolation-source policy."""

    def __init__(
        self,
        host: HostStandardizer,
        isolation_source: IsolationSourceStandardizer,
        lineage: HostLineageEnricher,
    ) -> None:
        self._host = host
        self._isolation_source = isolation_source
        self._host_lineage = lineage

    def standardize(self, extracted_record: Mapping[str, str]) -> HostIsolationSourceResult:
        """Standardize one extracted metadata record without exposing orchestration steps."""

        # 1st Host pass
        initial_host = self._host.standardize(extracted_record)
        initial_lineage_source = self._lineage_source_type(initial_host)
        host_only = not str(extracted_record.get("iso_val_orig", "") or "").strip()
        lineage_applied = initial_lineage_source is not None and host_only
        if lineage_applied:
            lineage_root_taxid, source_type_term_id = initial_lineage_source
            isolation_source = self._isolation_source.refine_source_type_from_host_lineage(
                IsolationSourceOutcome(
                    selected_terms=(),
                    evidence_level=IsolationSourceEvidenceLevel.NONE,
                    supporting_pairs=(),
                    host_recovery_pairs=(),
                    reasoning=(),
                    diagnostics=(),
                    exact_matches=0,
                    cache_hits=0,
                    llm_calls=0,
                ),
                host_taxid=initial_host.standardized.taxid,
                lineage_root_taxid=lineage_root_taxid,
                source_type_term_id=source_type_term_id,
            )
        else:
            isolation_source = self._isolation_source.standardize(
                extracted_record,
                overflow=initial_host.overflow,
                host_pairs=initial_host.supporting_pairs if initial_host.standardized else (),
            )

        # 2nd host pass
        # An eligible isolation-source result means the organism is named in the submitted
        # values. Retry only the record's own pairs, excluding text that
        # the first host pass already saw.
        final_host = initial_host
        recovery_routing = HostRecoveryRouting.NOT_ELIGIBLE
        host_diagnostics = initial_host.diagnostics
        if (
            initial_host.standardized is None
            and isinstance(isolation_source, IsolationSourceOutcome)
            and isolation_source.host_recovery_eligible
            and isolation_source.host_recovery_pairs
        ):
            final_host = self._host.recovery_pass(
                str(extracted_record.get("accession", "") or ""),
                "||".join(pair.attribute for pair in isolation_source.host_recovery_pairs),
                "||".join(pair.value for pair in isolation_source.host_recovery_pairs),
            )
            host_diagnostics += final_host.diagnostics
            recovery_routing = (
                HostRecoveryRouting.RESOLVED
                if final_host.standardized is not None
                else HostRecoveryRouting.UNRESOLVED
            )

        final_lineage_source = self._lineage_source_type(final_host)
        if (
            not lineage_applied
            and final_lineage_source is not None
            and isinstance(isolation_source, IsolationSourceOutcome)
        ):
            lineage_root_taxid, source_type_term_id = final_lineage_source
            isolation_source = self._isolation_source.refine_source_type_from_host_lineage(
                isolation_source,
                host_taxid=final_host.standardized.taxid,
                lineage_root_taxid=lineage_root_taxid,
                source_type_term_id=source_type_term_id,
            )
        isolation_source = self._isolation_source.add_laboratory_from_strain(
            isolation_source,
            str(extracted_record.get("ncbi_organism", "") or ""),
            str(extracted_record.get("strain", "") or ""),
        )

        initial_routing = (
            HostInitialRouting.MATCHED
            if initial_host.standardized is not None
            else HostInitialRouting.OVERFLOW
            if initial_host.overflow is not None
            else HostInitialRouting.REJECTED
        )
        return HostIsolationSourceResult(
            host=final_host,
            isolation_source=isolation_source,
            diagnostics=HostIsolationSourceDiagnostics(
                host=host_diagnostics,
                isolation_source=isolation_source.diagnostics,
            ),
            routing=HostIsolationSourceRouting(
                host_initial=initial_routing,
                host_recovery=recovery_routing,
            ),
        )

    def _lineage_source_type(self, host: HostOutcome) -> tuple[int, str] | None:
        if host.standardized is None:
            return None
        return next(
            (
                (lineage_root_taxid, source_type_term_id)
                for lineage_root_taxid, source_type_term_id in HOST_SOURCE_TYPE_BY_LINEAGE_ROOT
                if self._host_lineage.is_descendant_or_self(
                    host.standardized.taxid, lineage_root_taxid
                )
            ),
            None,
        )
