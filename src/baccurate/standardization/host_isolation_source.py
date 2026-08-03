"""
Coordinate host and isolation-source standardization for one record.

A resolved host becomes context for the isolation-source prompt. If the classification
selects a host-implying ontology branch, the isolation-source attributes probably name
an organism. The host standardizer then makes a recovery pass over those submitted
values.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from time import perf_counter

from baccurate.adapters.llm.client import LLMSettings, load_llm_settings
from baccurate.adapters.llm.request import canonical_json_sha256
from baccurate.paths import DEFAULT_NAMES_DMP, DEFAULT_NODES_DMP
from baccurate.standardization.host import (
    HostDiagnostic,
    HostOutcome,
    HostPolicy,
    HostStandardizer,
)
from baccurate.standardization.host_lineage import HostLineageEnricher
from baccurate.standardization.isolation_source import (
    HOST_SOURCE_TYPE_BY_LINEAGE_ROOT,
    ISOLATION_SOURCE_LLM_PARAMETERS,
    IsolationSourceDiagnostic,
    IsolationSourceEvidenceLevel,
    IsolationSourceOutcome,
    IsolationSourcePromptPolicy,
    IsolationSourceReasoningStep,
    IsolationSourceRejection,
    IsolationSourceStandardizer,
)
from baccurate.standardization.isolation_source_ontology import ontology_semantics_fingerprint

# None disables the LLM path here.
_LOAD_LLM_ADAPTER = object()

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


class IsolationSourceRouting(StrEnum):
    """The terminal route used for isolation-source standardization."""

    REJECTED = "rejected"
    DETERMINISTIC = "deterministic"
    CACHE = "cache"
    LLM = "llm"
    LLM_DISABLED = "llm_disabled"


@dataclass(frozen=True, slots=True)
class HostIsolationSourceRouting:
    """Coverage classification for the host--isolation-source collaboration."""

    host_initial: HostInitialRouting
    host_recovery: HostRecoveryRouting
    isolation_source: IsolationSourceRouting
    host_overflow_used: bool
    bioproject_context_available: bool
    crosslink_applied: bool


# --- Result contract ---


@dataclass(frozen=True, slots=True)
class HostIsolationSourceDiagnostics:
    """All diagnostics emitted while producing the final result."""

    host: tuple[HostDiagnostic, ...]
    isolation_source: tuple[IsolationSourceDiagnostic, ...]


@dataclass(frozen=True, slots=True)
class HostIsolationSourceTiming:
    """Record-level elapsed execution time."""

    elapsed_seconds: float


@dataclass(frozen=True, slots=True)
class HostIsolationSourceFingerprints:
    """Optional canonical model request identity."""

    request: str | None


@dataclass(frozen=True, slots=True)
class HostIsolationSourceResult:
    """Immutable final result of coordinated host--isolation-source standardization."""

    host: HostOutcome
    isolation_source: IsolationSourceOutcome | IsolationSourceRejection
    reasoning: tuple[IsolationSourceReasoningStep, ...]
    diagnostics: HostIsolationSourceDiagnostics
    evidence_level: IsolationSourceEvidenceLevel
    routing: HostIsolationSourceRouting
    timing: HostIsolationSourceTiming
    fingerprints: HostIsolationSourceFingerprints


# --- Main class ---


class HostIsolationSourceStandardizer:
    """Standardize one BioSample record through the coordinated host--isolation-source policy."""

    def __init__(
        self,
        *,
        host_policy: HostPolicy,
        isolation_source_prompt_policy: IsolationSourcePromptPolicy,
        extracted_metadata: Path | str,
        result_logger: logging.Logger | None = None,
        llm_adapter: object = _LOAD_LLM_ADAPTER,
        llm_settings: LLMSettings | None = None,
        read_llm_cache: bool = True,
        host_lineage: HostLineageEnricher | None = None,
        names_dmp: Path = DEFAULT_NAMES_DMP,
        nodes_dmp: Path = DEFAULT_NODES_DMP,
    ) -> None:
        effective_llm_settings = llm_settings or load_llm_settings()
        self._host = HostStandardizer(host_policy, result_logger=result_logger)
        if llm_adapter is _LOAD_LLM_ADAPTER:
            self._isolation_source = IsolationSourceStandardizer(
                isolation_source_prompt_policy,
                extracted_metadata,
                result_logger=result_logger,
                llm_settings=effective_llm_settings,
                read_llm_cache=read_llm_cache,
            )
        else:
            self._isolation_source = IsolationSourceStandardizer(
                isolation_source_prompt_policy,
                extracted_metadata,
                result_logger=result_logger,
                client=None,
                llm_settings=effective_llm_settings,
                read_llm_cache=read_llm_cache,
            )
            self._isolation_source.pipeline.client = llm_adapter
        self._owns_components = True
        self._host_lineage = host_lineage
        self._host_lineage_paths = (names_dmp, nodes_dmp)
        self._initialize_fingerprints()

    def _initialize_fingerprints(self) -> None:
        """
        Capture the component identities needed to reproduce results.

        Reading each value from the components makes both construction routes describe
        the run identically. Copying the endpoint or cache-read policy from arguments
        could instead record a configuration that the components are not using.
        """
        pipeline = getattr(self._isolation_source, "pipeline", None)
        ontology = getattr(self._isolation_source, "ontology", None)
        isolation_source_policy = getattr(self._isolation_source, "policy", None)
        isolation_source_config = (
            isolation_source_policy.as_legacy_mapping()
            if isolation_source_policy is not None
            else {}
        )
        self._model_identifier = getattr(pipeline, "model", "")
        self._llm_cache_reads_enabled = getattr(pipeline, "read_cache", True)
        self._model_endpoint_fingerprint = canonical_json_sha256(
            {"server": getattr(pipeline, "server", None)}
        )
        if ontology is None:
            # A real IsolationSourceStandardizer always carries an ontology; test doubles need
            # not. Fingerprint the empty graph so identity stays well defined.
            self._ontology_fingerprint = ontology_semantics_fingerprint(None)
            prompt_contract = {
                "isolation_configuration": isolation_source_config,
                "request_parameters": ISOLATION_SOURCE_LLM_PARAMETERS,
            }
            self._prompt_configuration_fingerprint = canonical_json_sha256(prompt_contract)
        else:
            self._ontology_fingerprint = ontology_semantics_fingerprint(ontology)
            self._prompt_configuration_fingerprint = (
                isolation_source_policy.prompt_configuration_fingerprint
            )

    @property
    def llm_cache_reads_enabled(self) -> bool:
        return self._llm_cache_reads_enabled

    @property
    def model_identifier(self) -> str:
        return self._model_identifier

    @property
    def ontology_fingerprint(self) -> str:
        return self._ontology_fingerprint

    @property
    def prompt_configuration_fingerprint(self) -> str:
        return self._prompt_configuration_fingerprint

    @property
    def model_endpoint_fingerprint(self) -> str:
        return self._model_endpoint_fingerprint

    def standardize(self, extracted_record: Mapping[str, str]) -> HostIsolationSourceResult:
        """Standardize one extracted metadata record without exposing orchestration steps."""
        started = perf_counter()

        # 1st Host pass
        initial_host = self._host.standardize(extracted_record)
        standardized_host = (
            initial_host.standardized.scientific_name
            if initial_host.standardized is not None
            else ""
        )
        host_context = (
            standardized_host or str(extracted_record.get("host_val_orig", "") or "").strip()
        )
        initial_lineage_source = self._lineage_source_type(initial_host)
        host_only = not any(
            str(extracted_record.get(key, "") or "").strip()
            for key in ("iso_val_orig", "bioproject_id", "bioproject_accession")
        )
        lineage_applied = initial_lineage_source is not None and host_only
        if lineage_applied:
            lineage_root_taxid, source_type_term_id = initial_lineage_source
            isolation_source = self._isolation_source.refine_source_type_from_host_lineage(
                IsolationSourceOutcome(
                    selected_terms=(),
                    evidence_level=IsolationSourceEvidenceLevel.NONE,
                    host_context=host_context,
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
                host_context=host_context,
                overflow=initial_host.overflow,
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

        initial_routing = (
            HostInitialRouting.MATCHED
            if initial_host.standardized is not None
            else HostInitialRouting.OVERFLOW
            if initial_host.overflow is not None
            else HostInitialRouting.REJECTED
        )
        isolation_source_routing = self._isolation_source_routing(isolation_source)
        if isinstance(isolation_source, IsolationSourceOutcome):
            reasoning = isolation_source.reasoning
            evidence_level = isolation_source.evidence_level
            request_fingerprint = isolation_source.request_fingerprint
            project_context_available = bool(isolation_source.resolved_bioproject_accessions)
        else:
            reasoning = ()
            evidence_level = IsolationSourceEvidenceLevel.NONE
            request_fingerprint = None
            project_context_available = False
        return HostIsolationSourceResult(
            host=final_host,
            isolation_source=isolation_source,
            reasoning=reasoning,
            diagnostics=HostIsolationSourceDiagnostics(
                host=host_diagnostics,
                isolation_source=isolation_source.diagnostics,
            ),
            evidence_level=evidence_level,
            routing=HostIsolationSourceRouting(
                host_initial=initial_routing,
                host_recovery=recovery_routing,
                isolation_source=isolation_source_routing,
                host_overflow_used=initial_host.overflow is not None,
                bioproject_context_available=project_context_available,
                crosslink_applied=any(step.node == "crosslink" for step in reasoning),
            ),
            timing=HostIsolationSourceTiming(elapsed_seconds=perf_counter() - started),
            fingerprints=HostIsolationSourceFingerprints(
                request=request_fingerprint,
            ),
        )

    def _lineage_source_type(self, host: HostOutcome) -> tuple[int, str] | None:
        if host.standardized is None:
            return None
        if self._host_lineage is None:
            self._host_lineage = HostLineageEnricher(*self._host_lineage_paths)
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

    @staticmethod
    def _isolation_source_routing(
        isolation_source: IsolationSourceOutcome | IsolationSourceRejection,
    ) -> IsolationSourceRouting:
        """Name the most expensive route the record traveled, cheapest last."""
        if isinstance(isolation_source, IsolationSourceRejection):
            return IsolationSourceRouting.REJECTED
        if isolation_source.llm_calls:
            return IsolationSourceRouting.LLM
        if isolation_source.cache_hits:
            return IsolationSourceRouting.CACHE
        if isolation_source.exact_matches:
            return IsolationSourceRouting.DETERMINISTIC
        if any(step.node == "host_lineage_derivation" for step in isolation_source.reasoning):
            return IsolationSourceRouting.DETERMINISTIC
        # No route left any trace, which only happens with no configured client.
        return IsolationSourceRouting.LLM_DISABLED

    def close(self) -> None:
        if self._owns_components:
            self._isolation_source.close()


# --- Builder adaptation ---


def host_isolation_source_standardizer_from_components(
    host_standardizer: HostStandardizer,
    isolation_source_standardizer: IsolationSourceStandardizer,
    host_lineage: HostLineageEnricher,
) -> HostIsolationSourceStandardizer:
    """
    Adapt builder-owned standardizers to the cross-target standardizer.

    Bypasses __init__ because the builder constructs and owns both components
    itself. Every attribute __init__ assigns must therefore be assigned here
    too, or the omission surfaces as an AttributeError mid-run.
    """
    standardizer = HostIsolationSourceStandardizer.__new__(HostIsolationSourceStandardizer)
    standardizer._host = host_standardizer
    standardizer._isolation_source = isolation_source_standardizer
    standardizer._host_lineage = host_lineage
    standardizer._host_lineage_paths = None
    standardizer._owns_components = False
    standardizer._initialize_fingerprints()
    return standardizer
