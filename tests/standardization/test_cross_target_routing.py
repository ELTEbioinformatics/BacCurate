"""Pin the cross-target host and isolation-source routing contract."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from types import SimpleNamespace

import pytest

import baccurate.standardization.host_isolation_source as host_isolation_source_module
from baccurate.adapters.llm.client import LLMSettings
from baccurate.standardization.host import HostPolicy, HostStandardizer
from baccurate.standardization.host_isolation_source import (
    HostIsolationSourceStandardizer,
    host_isolation_source_standardizer_from_components,
)
from baccurate.standardization.isolation_source import IsolationSourceStandardizer

ANIMAL_HOST = "BACC:0000002"
PLANT_HOST = "BACC:0000003"
ENVIRONMENTAL = "BACC:0000004"
FOOD_OR_FEED = "BACC:0000007"


class ScriptedClient:
    def __init__(self, answer: dict[str, object]) -> None:
        self.answer = answer
        self.calls = 0
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, **kwargs: object) -> object:
        self.calls += 1
        return kwargs["response_model"].model_validate(self.answer)  # type: ignore[union-attr]


def _coordinator(
    resources,
    host_policy,
    isolation_source_policy,
    *,
    client: ScriptedClient | None = None,
) -> HostIsolationSourceStandardizer:
    isolation_source = IsolationSourceStandardizer(
        isolation_source_policy,
        client=None,
    )
    isolation_source.pipeline.client = client
    lineage_memberships = {(9606, 33208), (9031, 33208), (3702, 33090)}
    coordinator = host_isolation_source_standardizer_from_components(
        HostStandardizer(host_policy, resources.ncbi_taxonomy_reference_table),
        isolation_source,
        SimpleNamespace(
            is_descendant_or_self=lambda taxid, root: (taxid, root) in lineage_memberships
        ),
    )
    coordinator._owns_components = True
    return coordinator


def _classifier_answer(source_type: str) -> dict[str, object]:
    return {
        "reasoning": "scripted classification",
        "evidence_level": "sample",
        "source_type": source_type,
        "body_product": [],
        "body_site": [],
        "lesion": [],
        "environmental_material": [],
        "facility": [],
        "sampled_object": [],
        "food_type": [],
    }


FIRST_ENDPOINT = "https://models.example/v1"
SECOND_ENDPOINT = "https://models.internal/v1"


@pytest.fixture
def coordinators(
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
    closers: list[Callable[[], None]] = []

    def public(server: str) -> HostIsolationSourceStandardizer:
        coordinator = HostIsolationSourceStandardizer(
            host_policy=fixture_host_policy,
            isolation_source_prompt_policy=fixture_isolation_source_prompt_policy,
            llm_adapter=None,
            llm_settings=LLMSettings("test-key", server, "test-model"),
        )
        closers.append(coordinator.close)
        return coordinator

    def from_components(server: str) -> HostIsolationSourceStandardizer:
        isolation_source = IsolationSourceStandardizer(
            fixture_isolation_source_prompt_policy,
            client=None,
            llm_settings=LLMSettings("test-key", server, "test-model"),
        )
        closers.append(isolation_source.close)
        return host_isolation_source_standardizer_from_components(
            FixtureHostStandardizer(fixture_host_policy),
            isolation_source,
            SimpleNamespace(is_descendant_or_self=lambda _taxid, _root: False),
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


@pytest.mark.parametrize(
    ("host_value", "host_taxid", "lineage_root", "expected_term_id", "expected_label"),
    [
        ("human", 9606, 33208, ANIMAL_HOST, "animal host"),
        ("thale cress", 3702, 33090, PLANT_HOST, "plant host"),
    ],
)
def test_standardized_host_lineage_fills_unspecified_source_type(
    standardization_fixture_resources,
    fixture_host_policy,
    fixture_isolation_source_prompt_policy,
    host_value: str,
    host_taxid: int,
    lineage_root: int,
    expected_term_id: str,
    expected_label: str,
) -> None:
    host = HostStandardizer(
        fixture_host_policy,
        standardization_fixture_resources.ncbi_taxonomy_reference_table,
    )
    isolation_source = IsolationSourceStandardizer(
        fixture_isolation_source_prompt_policy,
        client=None,
    )
    lineage = SimpleNamespace(
        is_descendant_or_self=lambda taxid, root: taxid == host_taxid and root == lineage_root
    )
    coordinator = host_isolation_source_standardizer_from_components(
        host,
        isolation_source,
        lineage,
    )
    try:
        result = coordinator.standardize(
            {
                "accession": f"HOST_ONLY_{host_taxid}",
                "host_attr_orig": "host",
                "host_val_orig": host_value,
            }
        )
    finally:
        isolation_source.close()

    assert result.isolation_source.selected_terms[-1].term_id == expected_term_id
    assert result.isolation_source.selected_terms[-1].label == expected_label
    assert result.reasoning[-1].node == "host_lineage_derivation"
    assert str(host_taxid) in result.reasoning[-1].reasoning
    assert str(lineage_root) in result.reasoning[-1].reasoning


@pytest.mark.parametrize(
    ("record", "expected_term_ids"),
    [
        (
            {"accession": "UNRELATED", "host_attr_orig": "host", "host_val_orig": "yeast"},
            (),
        ),
        ({"accession": "UNRESOLVED", "host_attr_orig": "host", "host_val_orig": "unknown"}, ()),
        (
            {
                "accession": "ENVIRONMENTAL",
                "host_attr_orig": "host",
                "host_val_orig": "human",
                "iso_attr_orig": "isolation_source",
                "iso_val_orig": "soil",
            },
            (ENVIRONMENTAL,),
        ),
        (
            {
                "accession": "FOOD",
                "host_attr_orig": "host",
                "host_val_orig": "human",
                "iso_attr_orig": "isolation_source",
                "iso_val_orig": "meat product",
            },
            (FOOD_OR_FEED,),
        ),
    ],
)
def test_host_lineage_leaves_unresolved_unrelated_and_non_host_sources_unchanged(
    standardization_fixture_resources,
    fixture_host_policy,
    fixture_isolation_source_prompt_policy,
    record: dict[str, str],
    expected_term_ids: tuple[str, ...],
) -> None:
    coordinator = _coordinator(
        standardization_fixture_resources,
        fixture_host_policy,
        fixture_isolation_source_prompt_policy,
    )
    try:
        result = coordinator.standardize(record)
    finally:
        coordinator.close()

    source_type_ids = tuple(
        term.term_id
        for term in getattr(result.isolation_source, "selected_terms", ())
        if term.facet == "source_type" and term.term_id != "BACC:0000001"
    )
    assert source_type_ids == expected_term_ids
    assert all(step.node != "host_lineage_derivation" for step in result.reasoning)


@pytest.mark.parametrize("model_source_type", ["host-associated", "plant host"])
def test_metazoa_lineage_refines_generic_or_conflicting_host_source_type(
    standardization_fixture_resources,
    fixture_host_policy,
    fixture_isolation_source_prompt_policy,
    model_source_type: str,
) -> None:
    client = ScriptedClient(_classifier_answer(model_source_type))
    coordinator = _coordinator(
        standardization_fixture_resources,
        fixture_host_policy,
        fixture_isolation_source_prompt_policy,
        client=client,
    )
    try:
        result = coordinator.standardize(
            {
                "accession": model_source_type,
                "host_attr_orig": "host",
                "host_val_orig": "human",
                "iso_attr_orig": "isolation_source",
                "iso_val_orig": "unresolved material 481516",
            }
        )
    finally:
        coordinator.close()

    assert [term.term_id for term in result.isolation_source.selected_terms] == [
        "BACC:0000001",
        ANIMAL_HOST,
    ]
    assert result.reasoning[-1].node == "host_lineage_derivation"
    assert ("conflicting" in result.reasoning[-1].reasoning) == (model_source_type == "plant host")


def test_host_only_metazoa_lineage_is_deterministic_without_classification(
    standardization_fixture_resources,
    fixture_host_policy,
    fixture_isolation_source_prompt_policy,
) -> None:
    client = ScriptedClient(_classifier_answer("environmental"))
    coordinator = _coordinator(
        standardization_fixture_resources,
        fixture_host_policy,
        fixture_isolation_source_prompt_policy,
        client=client,
    )
    try:
        result = coordinator.standardize(
            {
                "accession": "HOST_ONLY_DETERMINISTIC",
                "host_attr_orig": "host",
                "host_val_orig": "human",
            }
        )
    finally:
        coordinator.close()

    assert result.isolation_source.selected_terms[-1].term_id == ANIMAL_HOST
    assert result.reasoning[-1].node == "host_lineage_derivation"
    assert result.evidence_level.value == "sample"
    assert result.routing.isolation_source.value == "deterministic"
    assert client.calls == 0


def test_host_lineage_refinement_applies_after_cache_hit(
    standardization_fixture_resources,
    fixture_host_policy,
    fixture_isolation_source_prompt_policy,
) -> None:
    client = ScriptedClient(_classifier_answer("host-associated"))
    coordinator = _coordinator(
        standardization_fixture_resources,
        fixture_host_policy,
        fixture_isolation_source_prompt_policy,
        client=client,
    )
    record = {
        "host_attr_orig": "host",
        "host_val_orig": "human",
        "iso_attr_orig": "isolation_source",
        "iso_val_orig": "unresolved material 12345",
    }
    try:
        first = coordinator.standardize({"accession": "CACHE_FIRST", **record})
        second = coordinator.standardize({"accession": "CACHE_SECOND", **record})
    finally:
        coordinator.close()

    assert client.calls == 1
    assert second.routing.isolation_source.value == "cache"
    assert second.isolation_source.selected_terms[-1].term_id == ANIMAL_HOST
    assert first.reasoning[-1].node == second.reasoning[-1].node == "host_lineage_derivation"


def test_recovered_host_lineage_refines_host_source_but_not_food_source(
    standardization_fixture_resources,
    fixture_host_policy,
    fixture_isolation_source_prompt_policy,
) -> None:
    host_coordinator = _coordinator(
        standardization_fixture_resources,
        fixture_host_policy,
        fixture_isolation_source_prompt_policy,
    )
    try:
        host_result = host_coordinator.standardize(
            {
                "accession": "RECOVERED_HOST",
                "iso_attr_orig": "isolation_source",
                "iso_val_orig": "human",
            }
        )
    finally:
        host_coordinator.close()

    food_client = ScriptedClient(
        {**_classifier_answer("food or feed"), "food_type": ["meat product"]}
    )
    food_coordinator = _coordinator(
        standardization_fixture_resources,
        fixture_host_policy,
        fixture_isolation_source_prompt_policy,
        client=food_client,
    )
    try:
        food_result = food_coordinator.standardize(
            {
                "accession": "RECOVERED_FOOD_HOST",
                "iso_attr_orig": "isolation_source",
                "iso_val_orig": "human meat 24680",
            }
        )
    finally:
        food_coordinator.close()

    assert host_result.routing.host_recovery.value == "resolved"
    assert host_result.host.standardized.taxid == 9606
    assert host_result.isolation_source.selected_terms[-1].term_id == ANIMAL_HOST
    assert host_result.reasoning[-1].node == "host_lineage_derivation"
    assert food_result.routing.host_recovery.value == "resolved"
    assert food_result.host.standardized.taxid == 9606
    assert FOOD_OR_FEED in {term.term_id for term in food_result.isolation_source.selected_terms}
    assert all(step.node != "host_lineage_derivation" for step in food_result.reasoning)
