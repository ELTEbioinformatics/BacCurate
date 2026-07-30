"""Pin the cross-target host and isolation-source routing contract."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from pathlib import Path
from types import SimpleNamespace

import pytest

import baccurate.standardization.host_isolation_source as host_isolation_source_module
from baccurate.adapters.llm.client import LLMSettings
from baccurate.provenance.source_snapshot import bioproject_catalog_path_for
from baccurate.standardization.host import HostPolicy, HostStandardizer
from baccurate.standardization.host_isolation_source import (
    HostIsolationSourceStandardizer,
    host_isolation_source_standardizer_from_components,
)
from baccurate.standardization.isolation_source import IsolationSourceStandardizer

FIRST_ENDPOINT = "https://models.example/v1"
SECOND_ENDPOINT = "https://models.internal/v1"


def _extracted_bundle(tmp_path: Path) -> Path:
    extracted = tmp_path / "extracted.tsv"
    bioproject_catalog_path_for(extracted).write_text("", encoding="utf-8")
    return extracted


@pytest.fixture
def coordinators(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    standardization_fixture_resources,
    fixture_host_policy,
    fixture_isolation_source_prompt_policy,
) -> Iterator[SimpleNamespace]:
    """
    Construct coordinators over one shared configuration, through either route.

    Both routes receive the same host policy, isolation-source policy, ontology, and cache
    location, so any identity difference between them comes from construction alone.
    """

    class FixtureHostStandardizer(HostStandardizer):
        """Read the small fixture taxonomy table instead of the full NCBI release."""

        def __init__(self, policy: HostPolicy, result_logger: object = None) -> None:
            super().__init__(
                policy,
                standardization_fixture_resources.ncbi_taxonomy_reference_table,
                result_logger,  # type: ignore[arg-type]
            )

    monkeypatch.setattr(host_isolation_source_module, "HostStandardizer", FixtureHostStandardizer)
    extracted_metadata = _extracted_bundle(tmp_path)
    closers: list[Callable[[], None]] = []

    def public(server: str) -> HostIsolationSourceStandardizer:
        coordinator = HostIsolationSourceStandardizer(
            host_policy=fixture_host_policy,
            isolation_source_prompt_policy=fixture_isolation_source_prompt_policy,
            extracted_metadata=extracted_metadata,
            llm_adapter=None,
            llm_settings=LLMSettings("test-key", server, "test-model"),
        )
        closers.append(coordinator.close)
        return coordinator

    def from_components(server: str) -> HostIsolationSourceStandardizer:
        isolation_source = IsolationSourceStandardizer(
            fixture_isolation_source_prompt_policy,
            extracted_metadata,
            client=None,
            llm_settings=LLMSettings("test-key", server, "test-model"),
        )
        closers.append(isolation_source.close)
        return host_isolation_source_standardizer_from_components(
            FixtureHostStandardizer(fixture_host_policy), isolation_source
        )

    yield SimpleNamespace(public=public, from_components=from_components)
    for close in closers:
        close()


def test_runtime_settings_agree_across_construction_routes(
    coordinators: SimpleNamespace,
) -> None:
    """One runtime setup must report the same settings through either construction route."""
    public = coordinators.public(FIRST_ENDPOINT)
    adapted = coordinators.from_components(FIRST_ENDPOINT)

    assert adapted.model_endpoint_fingerprint == public.model_endpoint_fingerprint
    assert adapted.model_identifier == public.model_identifier == "test-model"
    assert adapted.llm_cache_reads_enabled == public.llm_cache_reads_enabled


def test_model_endpoint_fingerprint_distinguishes_endpoints_on_both_routes(
    coordinators: SimpleNamespace,
) -> None:
    """Two model endpoints must have different fingerprints through either route."""
    first = coordinators.from_components(FIRST_ENDPOINT)
    second = coordinators.from_components(SECOND_ENDPOINT)

    assert first.model_endpoint_fingerprint != second.model_endpoint_fingerprint
    assert (
        first.model_endpoint_fingerprint
        == coordinators.public(FIRST_ENDPOINT).model_endpoint_fingerprint
    )
    assert (
        second.model_endpoint_fingerprint
        == coordinators.public(SECOND_ENDPOINT).model_endpoint_fingerprint
    )
