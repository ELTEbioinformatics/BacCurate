"""
Standardize host and isolation-source as one decision rather than two.

A resolved host is added to the isolation-source prompt. In return, a classification
landing on a host-implying ontology branch says the record's isolation-source
attributes probably name an organism, so the host pass runs again over those raw
values.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from time import perf_counter

from baccurate.llm.client import LLMSettings, load_llm_settings
from baccurate.llm.request import canonical_json_sha256
from baccurate.standardizers.host import HostDiagnostic, HostOutcome, HostStandardizer
from baccurate.standardizers.isolation import (
    ISOLATION_LLM_PARAMETERS,
    IsolationDiagnostic,
    IsolationEvidenceLevel,
    IsolationOutcome,
    IsolationReasoningStep,
    IsolationRejection,
    IsoStandardizer,
    effective_isolation_prompts,
    ontology_semantics_fingerprint,
)

# None disables the LLM path here.
_LOAD_LLM_ADAPTER = object()

# --- Routing classification ---


class HostInitialRouting(StrEnum):
    """How the initial host pass resolved a record."""

    MATCHED = "matched"
    OVERFLOW = "overflow"
    REJECTED = "rejected"


class HostRetryRouting(StrEnum):
    """Whether eligible isolation evidence caused a host retry."""

    NOT_ELIGIBLE = "not_eligible"
    RESOLVED = "resolved"
    UNRESOLVED = "unresolved"


class IsolationRouting(StrEnum):
    """The terminal route used for isolation standardization."""

    REJECTED = "rejected"
    DETERMINISTIC = "deterministic"
    CACHE = "cache"
    LLM = "llm"
    LLM_DISABLED = "llm_disabled"


@dataclass(frozen=True, slots=True)
class HostIsolationRouting:
    """Coverage classification for the host--isolation collaboration."""

    host_initial: HostInitialRouting
    host_retry: HostRetryRouting
    isolation: IsolationRouting
    host_overflow_used: bool
    bioproject_context_available: bool
    crosslink_applied: bool


# --- Result contract ---


@dataclass(frozen=True, slots=True)
class HostIsolationDiagnostics:
    """All diagnostics emitted while producing the final result."""

    host: tuple[HostDiagnostic, ...]
    isolation: tuple[IsolationDiagnostic, ...]


@dataclass(frozen=True, slots=True)
class HostIsolationTiming:
    """Record-level elapsed execution time."""

    elapsed_seconds: float


@dataclass(frozen=True, slots=True)
class HostIsolationFingerprints:
    """Stable configuration identity and optional canonical model request identity."""

    configuration: str
    request: str | None


@dataclass(frozen=True, slots=True)
class HostIsolationResult:
    """Immutable final result of coordinated host--isolation standardization."""

    host: HostOutcome
    isolation: IsolationOutcome | IsolationRejection
    reasoning: tuple[IsolationReasoningStep, ...]
    diagnostics: HostIsolationDiagnostics
    evidence_level: IsolationEvidenceLevel
    routing: HostIsolationRouting
    timing: HostIsolationTiming
    fingerprints: HostIsolationFingerprints


# --- Main class ---


class HostIsolationStandardizer:
    """Standardize one BioSample record through the coordinated host--isolation policy."""

    def __init__(
        self,
        *,
        host_config: Path | str,
        isolation_config: Path | str,
        extracted_metadata: Path | str,
        result_logger: logging.Logger | None = None,
        llm_adapter: object = _LOAD_LLM_ADAPTER,
        llm_settings: LLMSettings | None = None,
        read_llm_cache: bool = True,
    ) -> None:
        effective_llm_settings = llm_settings or load_llm_settings()
        self._host = HostStandardizer(host_config, result_logger=result_logger)
        if llm_adapter is _LOAD_LLM_ADAPTER:
            self._isolation = IsoStandardizer(
                isolation_config,
                extracted_metadata,
                result_logger=result_logger,
                llm_settings=effective_llm_settings,
                read_llm_cache=read_llm_cache,
            )
        else:
            self._isolation = IsoStandardizer(
                isolation_config,
                extracted_metadata,
                result_logger=result_logger,
                client=None,
                llm_settings=effective_llm_settings,
                read_llm_cache=read_llm_cache,
            )
            self._isolation.pipeline.client = llm_adapter
        self._owns_components = True
        self._llm_cache_reads_enabled = read_llm_cache
        self._model_endpoint_fingerprint = canonical_json_sha256(
            {"server": effective_llm_settings.server}
        )
        self._initialize_fingerprints()

    def _initialize_fingerprints(self) -> None:
        """Snapshot everything a rerun would have to match to reproduce results."""
        pipeline = getattr(self._isolation, "pipeline", None)
        ontology = getattr(self._isolation, "ontology", None)
        isolation_config = getattr(self._isolation, "config", {})
        self._model_identifier = getattr(pipeline, "model", "")
        if ontology is None:
            # A real IsoStandardizer always carries an ontology; test doubles need
            # not. Fingerprint the empty graph so identity stays well defined.
            self._ontology_fingerprint = ontology_semantics_fingerprint({}, {}, {})
            prompt_contract = {
                "isolation_configuration": isolation_config,
                "request_parameters": ISOLATION_LLM_PARAMETERS,
            }
        else:
            self._ontology_fingerprint = ontology_semantics_fingerprint(
                ontology.node_metadata,
                ontology.children_map,
                ontology.crosslink_map,
            )
            prompts = effective_isolation_prompts(isolation_config, ontology)
            prompt_contract = {
                "prompt_version": isolation_config.get("prompt_version"),
                "system_prompt": prompts.system,
                "user_prompt_template": prompts.user_template,
                "bioproject_system_prompt": prompts.bioproject_system,
                "bioproject_user_prompt_template": prompts.bioproject_user,
                "request_parameters": ISOLATION_LLM_PARAMETERS,
            }
        self._prompt_configuration_fingerprint = canonical_json_sha256(prompt_contract)
        self._configuration_snapshot = {
            "host": getattr(self._host, "config", {}),
            "isolation": isolation_config,
            "model_identifier": self._model_identifier,
            "model_endpoint_sha256": self._model_endpoint_fingerprint,
            "request_parameters": ISOLATION_LLM_PARAMETERS,
            "effective_prompts": prompt_contract,
        }
        self._configuration_fingerprint = self._fingerprint_configuration()

    def _fingerprint_configuration(self) -> str:
        return canonical_json_sha256(
            {
                "effective_configuration": self._configuration_snapshot,
                "ontology_fingerprint": self._ontology_fingerprint,
            }
        )

    @property
    def configuration_fingerprint(self) -> str:
        return self._configuration_fingerprint

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

    @property
    def configuration_snapshot(self) -> Mapping[str, object]:
        return self._configuration_snapshot

    def standardize(self, record: Mapping[str, str]) -> HostIsolationResult:
        """Return the final decision without exposing orchestration steps."""
        started = perf_counter()

        # 1st Host pass
        initial_host = self._host.standardize(record)
        standardized_host = (
            initial_host.standardized.scientific_name
            if initial_host.standardized is not None
            else ""
        )
        host_context = standardized_host or str(record.get("host_val_orig", "") or "").strip()
        isolation = self._isolation.standardize(
            record,
            host_context=host_context,
            overflow=initial_host.overflow,
        )

        # 2nd host pass
        # A term path under HOST_RETRY_TRIGGERS means the organism is named in the
        # isolation-source values themselves, so retry host over the record's own raw
        # pairs, excluding overflow and project text, which the first hoss pass already saw.
        final_host = initial_host
        retry_routing = HostRetryRouting.NOT_ELIGIBLE
        host_diagnostics = initial_host.diagnostics
        if (
            initial_host.standardized is None
            and isinstance(isolation, IsolationOutcome)
            and isolation.host_retry_eligible
            and isolation.sample_origins
        ):
            final_host = self._host.retry(
                str(record.get("accession", "") or ""),
                "||".join(origin.attribute for origin in isolation.sample_origins),
                "||".join(origin.value for origin in isolation.sample_origins),
            )
            host_diagnostics += final_host.diagnostics
            retry_routing = (
                HostRetryRouting.RESOLVED
                if final_host.standardized is not None
                else HostRetryRouting.UNRESOLVED
            )

        initial_routing = (
            HostInitialRouting.MATCHED
            if initial_host.standardized is not None
            else HostInitialRouting.OVERFLOW
            if initial_host.overflow is not None
            else HostInitialRouting.REJECTED
        )
        isolation_routing = self._isolation_routing(isolation)
        if isinstance(isolation, IsolationOutcome):
            reasoning = isolation.reasoning
            evidence_level = isolation.evidence_level
            request_fingerprint = isolation.request_fingerprint
            project_context_available = bool(isolation.resolved_bioproject_accessions)
        else:
            reasoning = ()
            evidence_level = IsolationEvidenceLevel.NONE
            request_fingerprint = None
            project_context_available = False
        return HostIsolationResult(
            host=final_host,
            isolation=isolation,
            reasoning=reasoning,
            diagnostics=HostIsolationDiagnostics(
                host=host_diagnostics,
                isolation=isolation.diagnostics,
            ),
            evidence_level=evidence_level,
            routing=HostIsolationRouting(
                host_initial=initial_routing,
                host_retry=retry_routing,
                isolation=isolation_routing,
                host_overflow_used=initial_host.overflow is not None,
                bioproject_context_available=project_context_available,
                crosslink_applied=any(step.node == "crosslink" for step in reasoning),
            ),
            timing=HostIsolationTiming(elapsed_seconds=perf_counter() - started),
            fingerprints=HostIsolationFingerprints(
                configuration=self._configuration_fingerprint,
                request=request_fingerprint,
            ),
        )

    @staticmethod
    def _isolation_routing(
        isolation: IsolationOutcome | IsolationRejection,
    ) -> IsolationRouting:
        """Name the most expensive route the record traveled, cheapest last."""
        if isinstance(isolation, IsolationRejection):
            return IsolationRouting.REJECTED
        if isolation.llm_calls:
            return IsolationRouting.LLM
        if isolation.cache_hits:
            return IsolationRouting.CACHE
        if isolation.exact_matches:
            return IsolationRouting.DETERMINISTIC
        # No route left any trace, which only happens with no configured client.
        return IsolationRouting.LLM_DISABLED

    def close(self) -> None:
        if self._owns_components:
            self._isolation.close()


# --- Builder adaptation ---


def _host_isolation_standardizer_from_components(
    host_standardizer: HostStandardizer,
    isolation_standardizer: IsoStandardizer,
) -> HostIsolationStandardizer:
    """
    Adapt builder-owned modules without widening the public seam.

    Bypasses __init__ because the builder constructs and owns both components
    itself. Every attribute __init__ assigns must therefore be assigned here
    too, or the omission surfaces as an AttributeError mid-run.
    """
    standardizer = HostIsolationStandardizer.__new__(HostIsolationStandardizer)
    standardizer._host = host_standardizer
    standardizer._isolation = isolation_standardizer
    standardizer._owns_components = False
    standardizer._llm_cache_reads_enabled = True
    standardizer._model_endpoint_fingerprint = canonical_json_sha256({"server": None})
    standardizer._initialize_fingerprints()
    return standardizer
