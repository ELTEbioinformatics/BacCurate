"""Pin the cross-target host and isolation-source routing contract."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from types import SimpleNamespace

import pytest

import baccurate.standardization.host_isolation_source as host_isolation_source_module
from baccurate.adapters.llm.client import LLMSettings
from baccurate.paths import CONFIG_DIR
from baccurate.standardization.host import HostPolicy, HostStandardizer
from baccurate.standardization.host_isolation_source import (
    HostIsolationSourceStandardizer,
    host_isolation_source_standardizer_from_components,
)
from baccurate.standardization.isolation_source import (
    IsolationSourcePromptPolicy,
    IsolationSourceStandardizer,
)

HOST_ASSOCIATED = "BACC:0000001"
ANIMAL_HOST = "BACC:0000002"
HUMAN_HOST = "BACC:0000122"
PLANT_HOST = "BACC:0000003"
ENVIRONMENTAL = "BACC:0000004"
FOOD_OR_FEED = "BACC:0000007"
LABORATORY = "BACC:0000008"
HEALTHCARE_FACILITY = "BACC:0000079"


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
    lineage_memberships = {(9606, 9606), (9606, 33208), (9031, 33208), (3702, 33090)}
    coordinator = host_isolation_source_standardizer_from_components(
        HostStandardizer(host_policy, resources.ncbi_taxonomy_reference_table),
        isolation_source,
        SimpleNamespace(
            is_descendant_or_self=lambda taxid, root: (taxid, root) in lineage_memberships
        ),
    )
    coordinator._owns_components = True
    return coordinator


def _classifier_answer(source_type: str | None) -> dict[str, object]:
    return {
        "reasoning": "scripted classification",
        "evidence_level": "sample",
        "source_type": [source_type] if source_type is not None else [],
        "body_product": [],
        "body_site": [],
        "lesion": [],
        "environmental_material": [],
        "facility": [],
        "sampled_object": [],
        "food_type": [],
        "host_disease": [],
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


@pytest.mark.parametrize("model_source_type", ["host-associated", "plant host"])
@pytest.mark.parametrize(
    ("host", "host_source_type"), [("human", HUMAN_HOST), ("chicken", ANIMAL_HOST)]
)
def test_metazoa_lineage_refines_generic_or_conflicting_host_source_type(
    standardization_fixture_resources,
    fixture_host_policy,
    fixture_isolation_source_prompt_policy,
    model_source_type: str,
    host: str,
    host_source_type: str,
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
                "host_val_orig": host,
                "iso_attr_orig": "isolation_source",
                "iso_val_orig": "unresolved material 481516",
            }
        )
    finally:
        coordinator.close()

    assert [term.term_id for term in result.isolation_source.selected_terms] == [
        "BACC:0000001",
        host_source_type,
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

    assert result.isolation_source.selected_terms[-1].term_id == HUMAN_HOST
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
    assert second.isolation_source.selected_terms[-1].term_id == HUMAN_HOST
    assert first.reasoning[-1].node == second.reasoning[-1].node == "host_lineage_derivation"


def test_recovered_host_lineage_preserves_food_source(
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
        {**_classifier_answer("food or feed"), "source_type": [], "food_type": ["meat product"]}
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
    assert host_result.isolation_source.selected_terms[-1].term_id == HUMAN_HOST
    assert host_result.reasoning[-1].node == "host_lineage_derivation"
    assert food_result.routing.host_recovery.value == "resolved"
    assert food_result.host.standardized.taxid == 9606
    assert FOOD_OR_FEED in {term.term_id for term in food_result.isolation_source.selected_terms}
    assert HUMAN_HOST not in {term.term_id for term in food_result.isolation_source.selected_terms}


def test_host_lineage_adds_the_host_source_type_to_a_non_source_value(
    standardization_fixture_resources,
    fixture_host_policy,
    fixture_isolation_source_prompt_policy,
) -> None:
    client = ScriptedClient(_classifier_answer("laboratory strain"))
    coordinator = _coordinator(
        standardization_fixture_resources,
        fixture_host_policy,
        fixture_isolation_source_prompt_policy,
        client=client,
    )
    try:
        result = coordinator.standardize(
            {
                "accession": "NON_SOURCE_WITH_HOST",
                "host_attr_orig": "host",
                "host_val_orig": "chicken",
                "iso_attr_orig": "isolation_source",
                "iso_val_orig": "plasmid DNA",
            }
        )
    finally:
        coordinator.close()

    assert client.calls == 0
    assert {term.term_id for term in result.isolation_source.selected_terms} == {
        HOST_ASSOCIATED,
        ANIMAL_HOST,
    }


def test_host_lineage_adds_no_host_source_type_to_food(
    standardization_fixture_resources,
    fixture_host_policy,
    fixture_isolation_source_prompt_policy,
) -> None:
    client = ScriptedClient({**_classifier_answer(None), "food_type": ["meat product"]})
    coordinator = _coordinator(
        standardization_fixture_resources,
        fixture_host_policy,
        fixture_isolation_source_prompt_policy,
        client=client,
    )
    try:
        result = coordinator.standardize(
            {
                "accession": "FOOD_WITH_HOST",
                "host_attr_orig": "host",
                "host_val_orig": "chicken",
                "iso_attr_orig": "isolation_source",
                "iso_val_orig": "chicken breast 97531",
            }
        )
    finally:
        coordinator.close()

    selected_ids = {term.term_id for term in result.isolation_source.selected_terms}
    assert result.host.standardized.taxid == 9031
    assert FOOD_OR_FEED in selected_ids
    assert ANIMAL_HOST not in selected_ids


def test_laboratory_strain_in_organism_name_adds_laboratory(
    standardization_fixture_resources,
    fixture_host_policy,
    fixture_isolation_source_prompt_policy,
) -> None:
    client = ScriptedClient(_classifier_answer(None))
    coordinator = _coordinator(
        standardization_fixture_resources,
        fixture_host_policy,
        fixture_isolation_source_prompt_policy,
        client=client,
    )
    try:
        result = coordinator.standardize(
            {
                "accession": "SAMN48918554",
                "ncbi_organism": "Escherichia coli str. K-12 substr. MG1655",
                "iso_attr_orig": "isolation_source",
                "iso_val_orig": "unresolved material 31415",
            }
        )
    finally:
        coordinator.close()

    assert [term.term_id for term in result.isolation_source.selected_terms] == [LABORATORY]
    assert result.reasoning[-1].node == "laboratory_strain"
    assert "unspecified" not in result.diagnostics.isolation_source


@pytest.mark.parametrize(
    ("host", "expected"), [("", {LABORATORY}), ("human", {LABORATORY, HOST_ASSOCIATED, HUMAN_HOST})]
)
def test_laboratory_strain_needs_no_isolation_source_value(
    standardization_fixture_resources,
    fixture_host_policy,
    fixture_isolation_source_prompt_policy,
    host: str,
    expected: set[str],
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
                "accession": "K12_WITHOUT_VALUE",
                "ncbi_organism": "Escherichia coli K-12",
                "host_attr_orig": "host" if host else "",
                "host_val_orig": host,
            }
        )
    finally:
        coordinator.close()

    assert {term.term_id for term in result.isolation_source.selected_terms} == expected
    assert result.routing.isolation_source.value == "deterministic"
    assert result.evidence_level.value == "sample"
    assert client.calls == 0


def test_other_strain_gets_no_laboratory(
    standardization_fixture_resources,
    fixture_host_policy,
    fixture_isolation_source_prompt_policy,
) -> None:
    coordinator = _coordinator(
        standardization_fixture_resources,
        fixture_host_policy,
        fixture_isolation_source_prompt_policy,
    )
    try:
        result = coordinator.standardize(
            {"accession": "ATCC_STRAIN", "ncbi_organism": "Escherichia coli ATCC 25922"}
        )
    finally:
        coordinator.close()

    assert result.routing.isolation_source.value == "rejected"


@pytest.mark.parametrize(
    ("organism", "strain", "expected"),
    [
        ("Escherichia coli", "K-12 MG1655", {LABORATORY}),
        ("Escherichia coli", "DH5-alpha", {LABORATORY}),
        ("Escherichia coli", "DH5\N{GREEK SMALL LETTER ALPHA}", {LABORATORY}),
        ("Escherichia coli", "EcK12-3", None),
        ("Klebsiella pneumoniae", "K12", None),
    ],
)
def test_laboratory_strain_in_strain_attribute(
    standardization_fixture_resources,
    fixture_host_policy,
    fixture_isolation_source_prompt_policy,
    organism: str,
    strain: str,
    expected: set[str] | None,
) -> None:
    coordinator = _coordinator(
        standardization_fixture_resources,
        fixture_host_policy,
        fixture_isolation_source_prompt_policy,
    )
    try:
        result = coordinator.standardize(
            {"accession": "STRAIN", "ncbi_organism": organism, "strain": strain}
        )
    finally:
        coordinator.close()

    if expected is None:
        assert result.routing.isolation_source.value == "rejected"
    else:
        assert {term.term_id for term in result.isolation_source.selected_terms} == expected


def test_configured_laboratory_strain_names_match_spelling_variants() -> None:
    policy = IsolationSourcePromptPolicy.load(CONFIG_DIR / "isolation_source.yaml")
    patterns = dict(policy.laboratory_strain_patterns)
    laboratory_strains = {
        "Escherichia coli": ["K12 MG1655", "OneShot Top10", "B (REL606)", "dh10b", "DH5alpha"],
        "Pseudomonas aeruginosa": ["UCBPP-PA14", "mPAO1"],
        "Staphylococcus aureus": ["SH1000", "NCTC 8325"],
    }
    other_strains = {
        "Escherichia coli": ["K12a", "EcK12-3", "UCBPP-PA14"],
        "Pseudomonas aeruginosa": ["PA149"],
        "Staphylococcus aureus": ["USA300_FPR3757 (CA-MRSA)-JE2", "Newman"],
    }

    for species, strains in laboratory_strains.items():
        assert all(patterns[species].search(strain.casefold()) for strain in strains)
    for species, strains in other_strains.items():
        assert not any(patterns[species].search(strain.casefold()) for strain in strains)


def test_configured_laboratory_strain_names_match_organism_names_by_species() -> None:
    policy = IsolationSourcePromptPolicy.load(CONFIG_DIR / "isolation_source.yaml")

    def names_laboratory_strain(organism: str) -> bool:
        return any(
            organism.startswith(species) and pattern.search(organism[len(species) :].casefold())
            for species, pattern in policy.laboratory_strain_patterns
        )

    laboratory_strains = [
        "Escherichia coli str. K-12 substr. MG1655",
        "Escherichia coli BL21(DE3)",
        "Escherichia coli 'BL21-Gold(DE3)pLysS AG'",
        "Escherichia coli DH5[alpha]",
        "Pseudomonas aeruginosa MPAO1/P1",
        "Pseudomonas aeruginosa UCBPP-PA14",
        "Staphylococcus aureus subsp. aureus RN4220",
        "Escherichia coli B str. REL606",
        "Enterococcus faecalis OG1RF",
        "Pseudomonas aeruginosa PAO1LM07",
    ]
    other_strains = [
        "Escherichia coli ATCC 25922",
        "Escherichia coli PA14",
        "Pseudomonas aeruginosa BL21",
        "Escherichia coli O17 str. K12a",
        "Escherichia coli B088",
        "Escherichia coli B7A",
    ]

    assert all(names_laboratory_strain(organism) for organism in laboratory_strains)
    assert not any(names_laboratory_strain(organism) for organism in other_strains)
