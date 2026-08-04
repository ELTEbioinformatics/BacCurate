"""Pin the isolation-source standardization contract."""

from __future__ import annotations

import shutil
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import yaml
from instructor import Mode
from instructor.core import InstructorRetryException
from pydantic import BaseModel, ValidationError

import baccurate.standardization.isolation_source as isolation_source_module
from baccurate.adapters.llm.client import LLMSettings
from baccurate.adapters.llm.diagnostics import LLMObservability
from baccurate.adapters.policy_yaml import PolicyConfigurationError
from baccurate.standardization.host import HostOverflowContext
from baccurate.standardization.isolation_source import (
    IsolationSourceClassifierAnswer,
    IsolationSourceDiagnostic,
    IsolationSourceEvidenceLevel,
    IsolationSourceOntologyGapDiagnostic,
    IsolationSourceOutcome,
    IsolationSourcePromptPolicy,
    IsolationSourceReasoningStep,
    IsolationSourceRejection,
    IsolationSourceStandardizer,
    SelectedTerm,
    SQLiteCache,
)

ROOT = Path(__file__).parents[2]
CONFIG_PATH = ROOT / "config" / "isolation_source.yaml"
FIXTURE_ROOT = ROOT / "tests" / "fixtures" / "standardization"


# Stable term identifiers used across the tests.
FECES = "BACC:0000025"
RECTUM = "BACC:0000041"
BLOOD = "BACC:0000010"
BODY_FLUID = "BACC:0000009"
DIGESTIVE_TRACT = "BACC:0000035"
INTESTINE = "BACC:0000039"
WOUND = "BACC:0000061"
SKIN = "BACC:0000060"
RHIZOSPHERE = "BACC:0000065"
SOIL = "BACC:0000064"
HOST_ASSOCIATED = "BACC:0000001"
ANIMAL_HOST = "BACC:0000002"
PLANT_HOST = "BACC:0000003"
ENVIRONMENTAL = "BACC:0000004"
FOOD_OR_FEED = "BACC:0000007"
LABORATORY = "BACC:0000008"
PUS = "BACC:0000017"
ABSCESS = "BACC:0000063"
ANIMAL_FOOD_PRODUCT = "BACC:0000095"
MEAT_PRODUCT = "BACC:0000097"

FACET_BY_LABEL = {
    "host-associated": "source_type",
    "animal host": "source_type",
    "environmental": "source_type",
    "blood": "body_product",
    "feces": "body_product",
    "wound": "lesion",
    "soil": "environmental_material",
    "meat product": "food_type",
}

# =============================================================================
# Fake LLM client
# =============================================================================


def _classifier_answer(
    *,
    reasoning: str = "because",
    evidence_level: str = "sample",
    source_type: str | None = None,
    body_product: list[str] | None = None,
    body_site: list[str] | None = None,
    lesion: list[str] | None = None,
    environmental_material: list[str] | None = None,
    facility: list[str] | None = None,
    sampled_object: list[str] | None = None,
    food_type: list[str] | None = None,
) -> dict[str, object]:
    return {
        "reasoning": reasoning,
        "evidence_level": evidence_level,
        "source_type": source_type,
        "body_product": body_product or [],
        "body_site": body_site or [],
        "lesion": lesion or [],
        "environmental_material": environmental_material or [],
        "facility": facility or [],
        "sampled_object": sampled_object or [],
        "food_type": food_type or [],
    }


class _FakeCompletions:
    def __init__(self, parent: FakeClient) -> None:
        self._parent = parent

    def create(self, **kwargs):
        self._parent.calls.append(kwargs)
        behavior = self._parent.behavior
        if behavior is None:
            raise AssertionError("LLM create() was called unexpectedly")
        if isinstance(behavior, Exception):
            raise behavior
        if isinstance(behavior, dict):
            return kwargs["response_model"].model_validate(behavior)
        if callable(behavior):
            return behavior(kwargs["response_model"])
        return behavior


class FakeClient:
    def __init__(self) -> None:
        self.calls: list[dict] = []
        self.behavior: object | None = None
        self.chat = SimpleNamespace(completions=_FakeCompletions(self))

    def respond_with(
        self,
        terms,
        reasoning: str = "because",
        evidence_level: str = "sample",
    ) -> None:
        terms = list(terms)
        facets: dict[str, object] = {
            "source_type": None,
            "body_product": [],
            "body_site": [],
            "lesion": [],
            "environmental_material": [],
            "facility": [],
            "sampled_object": [],
            "food_type": [],
        }
        for term in terms:
            facet = FACET_BY_LABEL[term]
            if facet == "source_type":
                facets[facet] = term
            else:
                facets[facet].append(term)
        self.respond_with_facets(
            reasoning=reasoning,
            evidence_level=evidence_level,
            **facets,
        )

    def fail_with(self, exc: Exception) -> None:
        self.behavior = exc

    def respond_with_facets(
        self,
        *,
        reasoning: str = "because",
        evidence_level: str = "sample",
        source_type: str | None = None,
        body_product: list[str] | None = None,
        body_site: list[str] | None = None,
        lesion: list[str] | None = None,
        environmental_material: list[str] | None = None,
        facility: list[str] | None = None,
        sampled_object: list[str] | None = None,
        food_type: list[str] | None = None,
    ) -> None:
        self.behavior = _classifier_answer(
            reasoning=reasoning,
            evidence_level=evidence_level,
            source_type=source_type,
            body_product=body_product,
            body_site=body_site,
            lesion=lesion,
            environmental_material=environmental_material,
            facility=facility,
            sampled_object=sampled_object,
            food_type=food_type,
        )

    def close(self) -> None:
        pass


def test_classifier_requests_endpoint_enforced_json_schema_mode(
    fixture_isolation_source_prompt_policy: IsolationSourcePromptPolicy,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_client = FakeClient()
    from_openai = Mock(return_value=fake_client)
    monkeypatch.setattr(isolation_source_module.instructor, "from_openai", from_openai)
    standardizer = IsolationSourceStandardizer(
        fixture_isolation_source_prompt_policy,
        client=fake_client,
        llm_settings=LLMSettings(None, "https://model.example/v1", "test-model"),
    )

    standardizer.close()

    from_openai.assert_called_once_with(fake_client, mode=Mode.JSON_SCHEMA)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def cache(tmp_path) -> SQLiteCache:
    c = SQLiteCache(tmp_path / "iso_cache.db")
    yield c
    c.close()


# =============================================================================
# Record-level outcomes
# =============================================================================


def test_typed_record_outcome_preserves_supporting_pairs_and_diagnostics(
    tmp_path,
    monkeypatch,
):
    config_path = _isolation_source_config(tmp_path)
    monkeypatch.setattr(
        "baccurate.standardization.isolation_source.load_llm_client",
        lambda *_args: (None, None),
    )
    standardizer = IsolationSourceStandardizer(
        IsolationSourcePromptPolicy.load(config_path),
    )

    try:
        result = standardizer.standardize(
            {
                "accession": "SAME_ACCESSION",
                "iso_attr_orig": "isolation_source",
                "iso_val_orig": "stool",
            },
            overflow=HostOverflowContext(
                attribute="host sample",
                value="blood",
            ),
        )
        with pytest.raises(
            ValueError,
            match=(
                "Malformed isolation-source selected attribute-value pairs for accession "
                "MALFORMED: 2 attributes for 1 values"
            ),
        ):
            standardizer.standardize(
                {
                    "accession": "MALFORMED",
                    "iso_attr_orig": "isolation_source||tissue",
                    "iso_val_orig": "blood",
                },
            )
        fake_client = FakeClient()
        fake_client.fail_with(RuntimeError("boom"))
        standardizer.pipeline.client = fake_client
        with pytest.raises(
            RuntimeError,
            match="Isolation-source LLM failed for accession TYPED_FAILURE",
        ):
            standardizer.standardize(
                {
                    "accession": "TYPED_FAILURE",
                    "iso_attr_orig": "isolation_source",
                    "iso_val_orig": "venous draw",
                },
            )
    finally:
        standardizer.close()

    assert isinstance(result, IsolationSourceOutcome)
    assert [(pair.attribute, pair.value) for pair in result.supporting_pairs] == [
        ("isolation_source", "stool"),
        ("host sample", "blood"),
    ]
    assert [(pair.attribute, pair.value) for pair in result.host_recovery_pairs] == [
        ("isolation_source", "stool")
    ]
    assert [(term.term_id, term.facet, term.label) for term in result.selected_terms] == [
        (HOST_ASSOCIATED, "source_type", "host-associated"),
        (ANIMAL_HOST, "source_type", "animal host"),
        (BODY_FLUID, "body_product", "body fluid"),
        (BLOOD, "body_product", "blood"),
        (FECES, "body_product", "feces"),
    ]
    assert result.reasoning == (
        IsolationSourceReasoningStep(
            node="direct_match",
            reasoning="All values resolved manually.",
            selected_terms={"body_product": ("blood", "feces")},
        ),
        IsolationSourceReasoningStep(
            node="source_type_derivation",
            reasoning="Filled facets determined the broad source kind.",
            selected_terms={"source_type": ("animal host",)},
        ),
        IsolationSourceReasoningStep(
            node="ancestor_expansion",
            reasoning="Selected terms were expanded to their facet ancestors.",
            selected_terms={
                "source_type": ("host-associated",),
                "body_product": ("body fluid",),
            },
        ),
    )
    assert result.exact_matches == 2
    assert result.cache_hits == 0
    assert result.llm_calls == 0
    assert result.evidence_level is IsolationSourceEvidenceLevel.SAMPLE
    assert result.diagnostics == (IsolationSourceDiagnostic.EXACT_MATCH,)


def _standardize_fixture_record(
    policy: IsolationSourcePromptPolicy,
    tmp_path: Path,
    *,
    values: str,
    attributes: str = "isolation_source",
    fake: FakeClient | None = None,
    monkeypatch: pytest.MonkeyPatch | None = None,
    accession: str = "PUBLIC_CLASSIFICATION",
) -> IsolationSourceOutcome | IsolationSourceRejection:
    if fake is not None:
        assert monkeypatch is not None
        monkeypatch.setattr(
            isolation_source_module.instructor,
            "from_openai",
            lambda client, **_kwargs: client,
        )
    standardizer = IsolationSourceStandardizer(
        policy,
        client=fake,
        llm_settings=LLMSettings(None, None, "test-model"),
    )
    try:
        result = standardizer.standardize(
            {
                "accession": accession,
                "iso_attr_orig": attributes,
                "iso_val_orig": values,
            },
        )
    finally:
        standardizer.close()
    return result


def _classify_fixture_record(
    policy: IsolationSourcePromptPolicy,
    tmp_path: Path,
    **kwargs: object,
) -> IsolationSourceOutcome:
    result = _standardize_fixture_record(policy, tmp_path, **kwargs)
    assert isinstance(result, IsolationSourceOutcome)
    return result


@pytest.mark.parametrize(
    ("submitted_value", "expected_terms"),
    [
        (
            "feces",
            (
                SelectedTerm(HOST_ASSOCIATED, "source_type", "host-associated"),
                SelectedTerm(ANIMAL_HOST, "source_type", "animal host"),
                SelectedTerm(FECES, "body_product", "feces"),
            ),
        ),
        (
            "rectal swab",
            (
                SelectedTerm(HOST_ASSOCIATED, "source_type", "host-associated"),
                SelectedTerm(ANIMAL_HOST, "source_type", "animal host"),
                SelectedTerm(DIGESTIVE_TRACT, "body_site", "digestive tract"),
                SelectedTerm(INTESTINE, "body_site", "intestine"),
                SelectedTerm(RECTUM, "body_site", "rectum"),
            ),
        ),
        (
            "ENVO:02000020",
            (
                SelectedTerm(HOST_ASSOCIATED, "source_type", "host-associated"),
                SelectedTerm(ANIMAL_HOST, "source_type", "animal host"),
                SelectedTerm(BODY_FLUID, "body_product", "body fluid"),
                SelectedTerm(BLOOD, "body_product", "blood"),
            ),
        ),
        (
            "rhizosphere",
            (
                SelectedTerm(HOST_ASSOCIATED, "source_type", "host-associated"),
                SelectedTerm(PLANT_HOST, "source_type", "plant host"),
                SelectedTerm(SOIL, "environmental_material", "soil"),
                SelectedTerm(RHIZOSPHERE, "environmental_material", "rhizosphere soil"),
            ),
        ),
    ],
)
def test_public_classification_resolves_ontology_terms(
    fixture_isolation_source_prompt_policy: IsolationSourcePromptPolicy,
    tmp_path: Path,
    submitted_value: str,
    expected_terms: tuple[SelectedTerm, ...],
) -> None:
    result = _classify_fixture_record(
        fixture_isolation_source_prompt_policy,
        tmp_path,
        values=submitted_value,
    )

    assert result.selected_terms == expected_terms


@pytest.mark.parametrize(
    ("submitted_value", "expected_eligible"),
    [
        pytest.param("host-associated", True, id="broad-host-source-kind"),
        pytest.param("meat product", True, id="meat-product"),
        pytest.param("animal feed", False, id="animal-feed"),
    ],
)
def test_assigned_term_flag_decides_host_recovery_eligibility_without_classifier(
    tmp_path: Path,
    submitted_value: str,
    expected_eligible: bool,
) -> None:
    policy = IsolationSourcePromptPolicy.load(
        _isolation_source_config(
            tmp_path,
            overrides={"ontology_directory": (ROOT / "data" / "reference" / "ontology").as_posix()},
        )
    )

    result = _classify_fixture_record(policy, tmp_path, values=submitted_value)

    assert result.host_recovery_eligible is expected_eligible
    assert result.llm_calls == 0


def test_host_recovery_eligibility_follows_the_flag_instead_of_term_identity(
    tmp_path: Path,
) -> None:
    ontology_directory = tmp_path / "ontology"
    shutil.copytree(FIXTURE_ROOT / "ontology", ontology_directory)
    terms_path = ontology_directory / "terms.tsv"
    terms_text = terms_path.read_text(encoding="utf-8")
    terms_text = terms_text.replace(
        "BACC:0000001\thost-associated\tsource_type\t\t"
        "human-associated;clinical sample;clinical material;clinical specimen;clinical isolate\t\t"
        "Host evidence\ttrue",
        "BACC:0000001\thost-associated\tsource_type\t\t"
        "human-associated;clinical sample;clinical material;clinical specimen;clinical isolate\t\t"
        "Host evidence\tfalse",
    )
    terms_text = terms_text.replace(
        "BACC:0000004\tenvironmental\tsource_type\t\tenvironment\t\tEnvironmental evidence\tfalse",
        "BACC:0000004\tenvironmental\tsource_type\t\tenvironment\t\tEnvironmental evidence\ttrue",
    )
    terms_path.write_text(terms_text, encoding="utf-8")
    policy = IsolationSourcePromptPolicy.load(
        _isolation_source_config(
            tmp_path,
            overrides={"ontology_directory": ontology_directory.as_posix()},
        )
    )

    host_associated = _classify_fixture_record(policy, tmp_path, values="clinical sample")
    environmental = _classify_fixture_record(policy, tmp_path, values="environment")

    assert host_associated.host_recovery_eligible is False
    assert environmental.host_recovery_eligible is True


def test_required_plant_host_term_must_descend_from_host_associated(
    tmp_path: Path,
) -> None:
    ontology_directory = tmp_path / "ontology"
    shutil.copytree(FIXTURE_ROOT / "ontology", ontology_directory)
    terms_path = ontology_directory / "terms.tsv"
    terms_path.write_text(
        terms_path.read_text(encoding="utf-8").replace(
            "BACC:0000003\tplant host\tsource_type\tBACC:0000001",
            "BACC:0000003\tplant host\tsource_type\tBACC:0000004",
        ),
        encoding="utf-8",
    )
    with pytest.raises(
        PolicyConfigurationError,
        match=r"BACC:0000003.*must descend from.*BACC:0000001",
    ):
        IsolationSourcePromptPolicy.load(
            _isolation_source_config(
                tmp_path,
                overrides={"ontology_directory": ontology_directory.as_posix()},
            )
        )


@pytest.mark.parametrize(
    "submitted_value",
    [
        "sample drawn, ENVO:02000020 present",
        "soil HMH:KPN:1777",
        "soil [HMH:1777]",
        "cattle [NCBITaxon:9913]",
        "blood [ENVO:02000020]; feces [ENVO:00002003]",
    ],
)
def test_identifier_outside_a_recognized_whole_value_shape_reaches_classifier_unchanged(
    fixture_isolation_source_prompt_policy: IsolationSourcePromptPolicy,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    submitted_value: str,
) -> None:
    fake = FakeClient()
    fake.respond_with(["blood"], reasoning="The whole submitted value was classified.")

    result = _classify_fixture_record(
        fixture_isolation_source_prompt_policy,
        tmp_path,
        values=submitted_value,
        fake=fake,
        monkeypatch=monkeypatch,
    )

    assert result.exact_matches == 0
    assert result.llm_calls == 1
    assert fake.calls[0]["messages"][1]["content"].count(submitted_value) == 1


@pytest.mark.parametrize(
    "submitted_value",
    [
        "[ENVO:02000020]",
        "blood [ENVO:02000020]",
        "blood (ENVO:02000020)",
        "blood ENVO:02000020",
        "ENVO:02000020 blood",
    ],
)
def test_identifier_whole_value_shapes_resolve_without_classifier_call(
    fixture_isolation_source_prompt_policy: IsolationSourcePromptPolicy,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    submitted_value: str,
) -> None:
    fake = FakeClient()

    result = _classify_fixture_record(
        fixture_isolation_source_prompt_policy,
        tmp_path,
        values=submitted_value,
        fake=fake,
        monkeypatch=monkeypatch,
    )

    assert SelectedTerm(BLOOD, "body_product", "blood") in result.selected_terms
    assert result.exact_matches == 1
    assert result.llm_calls == 0
    assert fake.calls == []


def test_disagreeing_paired_label_and_identifier_reports_diagnostic(
    fixture_isolation_source_prompt_policy: IsolationSourcePromptPolicy,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = FakeClient()
    fake.respond_with(["blood"], reasoning="The conflicting annotation was classified as blood.")

    result = _classify_fixture_record(
        fixture_isolation_source_prompt_policy,
        tmp_path,
        values="blood [ENVO:00002003]",
        fake=fake,
        monkeypatch=monkeypatch,
    )

    assert result.exact_matches == 0
    assert result.llm_calls == 1
    assert IsolationSourceDiagnostic.IDENTIFIER_DISAGREEMENT in result.diagnostics
    assert "blood [ENVO:00002003]" in fake.calls[0]["messages"][1]["content"]


def test_disagreeing_pair_reports_diagnostic_when_classifier_validation_fails(
    fixture_isolation_source_prompt_policy: IsolationSourcePromptPolicy,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = FakeClient()
    fake.fail_with(
        InstructorRetryException(
            "invalid response",
            n_attempts=3,
            total_usage=0,
        )
    )
    monkeypatch.setattr(
        isolation_source_module.instructor,
        "from_openai",
        lambda client, **_kwargs: client,
    )
    standardizer = IsolationSourceStandardizer(
        fixture_isolation_source_prompt_policy,
        client=fake,
        llm_settings=LLMSettings(None, None, "test-model"),
    )
    try:
        result = standardizer.standardize(
            {
                "accession": "DISAGREEING_FAILURE",
                "iso_attr_orig": "isolation_source",
                "iso_val_orig": "blood [ENVO:00002003]",
            },
        )
    finally:
        standardizer.close()

    assert isinstance(result, IsolationSourceRejection)
    assert result.diagnostics == (
        IsolationSourceDiagnostic.CLASSIFICATION_FAILURE,
        IsolationSourceDiagnostic.IDENTIFIER_DISAGREEMENT,
    )


def test_repeated_identifier_prefix_is_collapsed(
    fixture_isolation_source_prompt_policy: IsolationSourcePromptPolicy,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = FakeClient()

    result = _classify_fixture_record(
        fixture_isolation_source_prompt_policy,
        tmp_path,
        values="blood [ENVO:ENVO:02000020]",
        fake=fake,
        monkeypatch=monkeypatch,
    )

    assert result.selected_terms[-1] == SelectedTerm(BLOOD, "body_product", "blood")
    assert result.exact_matches == 1
    assert result.llm_calls == 0
    assert fake.calls == []


@pytest.mark.parametrize(
    ("submitted_value", "resolves"),
    [
        ("meat product [FOODON:00001006]", True),
        ("wound [MONDO:0021178]", False),
        ("bone [UBERON:0001474]", False),
    ],
)
def test_mapping_predicate_controls_identifier_resolution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    submitted_value: str,
    resolves: bool,
) -> None:
    policy = replace(
        IsolationSourcePromptPolicy.load(CONFIG_PATH),
        cache_db_path=tmp_path / "isolation-source-cache.db",
    )
    fake = FakeClient()
    if not resolves:
        fake.respond_with(["blood"], reasoning="The non-resolving mapping was classified.")

    result = _classify_fixture_record(
        policy,
        tmp_path,
        values=submitted_value,
        fake=fake,
        monkeypatch=monkeypatch,
    )

    assert result.exact_matches == int(resolves)
    assert result.llm_calls == int(not resolves)


def test_repository_ontology_directory_publishes_opaque_term_identifiers(tmp_path: Path) -> None:
    policy = replace(
        IsolationSourcePromptPolicy.load(CONFIG_PATH),
        cache_db_path=tmp_path / "isolation-source-cache.db",
    )

    result = _classify_fixture_record(policy, tmp_path, values="feces")

    assert result.selected_terms == (
        SelectedTerm(HOST_ASSOCIATED, "source_type", "host-associated"),
        SelectedTerm(ANIMAL_HOST, "source_type", "animal host"),
        SelectedTerm(FECES, "body_product", "feces"),
    )
    assert result.evidence_level is IsolationSourceEvidenceLevel.SAMPLE
    assert result.exact_matches == 1
    assert result.llm_calls == 0
    assert result.diagnostics == (IsolationSourceDiagnostic.EXACT_MATCH,)


def test_explicit_non_source_value_does_not_call_model(
    fixture_isolation_source_prompt_policy: IsolationSourcePromptPolicy,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = FakeClient()

    result = _classify_fixture_record(
        fixture_isolation_source_prompt_policy,
        tmp_path,
        values="no host",
        fake=fake,
        monkeypatch=monkeypatch,
    )

    assert result.selected_terms == ()
    assert result.evidence_level is IsolationSourceEvidenceLevel.NONE
    assert result.llm_calls == 0
    assert result.diagnostics == (IsolationSourceDiagnostic.UNSPECIFIED,)
    assert fake.calls == []


def test_public_classification_unions_direct_and_model_terms(
    fixture_isolation_source_prompt_policy: IsolationSourcePromptPolicy,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = FakeClient()
    fake.respond_with(["blood"], reasoning="The unresolved value describes blood.")

    result = _classify_fixture_record(
        fixture_isolation_source_prompt_policy,
        tmp_path,
        attributes="isolation_source||tissue",
        values="stool||venous draw",
        fake=fake,
        monkeypatch=monkeypatch,
    )

    assert result.selected_terms == (
        SelectedTerm(HOST_ASSOCIATED, "source_type", "host-associated"),
        SelectedTerm(ANIMAL_HOST, "source_type", "animal host"),
        SelectedTerm(BODY_FLUID, "body_product", "body fluid"),
        SelectedTerm(BLOOD, "body_product", "blood"),
        SelectedTerm(FECES, "body_product", "feces"),
    )
    assert result.exact_matches == 1
    assert result.llm_calls == 1
    assert result.evidence_level is IsolationSourceEvidenceLevel.SAMPLE
    classifier_prompt = fake.calls[0]["messages"][1]["content"]
    assert "isolation_source = stool" in classifier_prompt
    assert "tissue = venous draw" in classifier_prompt
    assert result.reasoning == (
        IsolationSourceReasoningStep(
            node="classifier",
            reasoning="The unresolved value describes blood.",
            selected_terms={"body_product": ("blood",)},
        ),
        IsolationSourceReasoningStep(
            node="source_type_derivation",
            reasoning="Filled facets determined the broad source kind.",
            selected_terms={"source_type": ("animal host",)},
        ),
        IsolationSourceReasoningStep(
            node="ancestor_expansion",
            reasoning="Selected terms were expanded to their facet ancestors.",
            selected_terms={
                "source_type": ("host-associated",),
                "body_product": ("body fluid",),
            },
        ),
    )


def test_public_classification_assigns_one_value_to_independent_facets(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy = replace(
        IsolationSourcePromptPolicy.load(CONFIG_PATH),
        cache_db_path=tmp_path / "faceted-isolation-source-cache.db",
    )
    fake = FakeClient()
    fake.respond_with_facets(
        reasoning="The value names an anatomical site, lesion, and body product.",
        source_type="animal host",
        body_product=["pus"],
        body_site=["liver"],
        lesion=["abscess"],
    )

    result = _classify_fixture_record(
        policy,
        tmp_path,
        values="liver abscess pus 739105",
        fake=fake,
        monkeypatch=monkeypatch,
    )

    assert [(term.term_id, term.facet, term.label) for term in result.selected_terms] == [
        ("BACC:0000001", "source_type", "host-associated"),
        ("BACC:0000002", "source_type", "animal host"),
        ("BACC:0000009", "body_product", "body fluid"),
        ("BACC:0000017", "body_product", "pus"),
        ("BACC:0000057", "body_site", "liver"),
        ("BACC:0000063", "lesion", "abscess"),
    ]


def test_classifier_response_schema_enforces_the_faceted_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy = replace(
        IsolationSourcePromptPolicy.load(CONFIG_PATH),
        cache_db_path=tmp_path / "faceted-response-schema-cache.db",
    )
    fake = FakeClient()
    fake.respond_with_facets(source_type="animal host")
    _classify_fixture_record(
        policy,
        tmp_path,
        values="clinical isolate 739105",
        fake=fake,
        monkeypatch=monkeypatch,
    )
    response_model = fake.calls[0]["response_model"]
    empty_answer = {
        "reasoning": "No origin is supported.",
        "evidence_level": "none",
        "source_type": None,
        "body_product": [],
        "body_site": [],
        "lesion": [],
        "environmental_material": [],
        "facility": [],
        "sampled_object": [],
        "food_type": [],
    }

    assert tuple(response_model.model_fields) == (
        "reasoning",
        "evidence_level",
        "source_type",
        "body_product",
        "body_site",
        "lesion",
        "environmental_material",
        "facility",
        "sampled_object",
        "food_type",
    )
    response_schema = response_model.model_json_schema()
    facet_labels = {
        facet.key: {
            term.label for term in policy.ontology.terms.values() if term.facet == facet.key
        }
        for facet in policy.ontology.facets.values()
    }
    source_type_schema = response_schema["properties"]["source_type"]
    source_type_enum = next(
        choice["enum"] for choice in source_type_schema["anyOf"] if "enum" in choice
    )
    assert set(source_type_enum) == facet_labels["source_type"]
    assert {choice.get("type") for choice in source_type_schema["anyOf"]} == {
        "string",
        "null",
    }
    for facet_key in facet_labels.keys() - {"source_type"}:
        assert (
            set(response_schema["properties"][facet_key]["items"]["enum"])
            == facet_labels[facet_key]
        )
    response_model.model_validate(empty_answer)

    invalid_answers = (
        {**empty_answer, "source_type": "animal host"},
        {
            **empty_answer,
            "evidence_level": "sample",
            "source_type": ["animal host", "environmental"],
        },
        {**empty_answer, "evidence_level": "sample", "body_site": ["blood"]},
    )
    for invalid_answer in invalid_answers:
        with pytest.raises(ValidationError):
            response_model.model_validate(invalid_answer)
    with pytest.raises(
        ValidationError,
        match="Every facet must be empty if and only if evidence_level is 'none'",
    ):
        response_model.model_validate({**empty_answer, "evidence_level": "sample"})
    with pytest.raises(
        ValidationError,
        match="Unknown source_type labels: 'animal-host'",
    ):
        response_model.model_validate(
            {**empty_answer, "evidence_level": "sample", "source_type": "animal-host"}
        )
    with pytest.raises(
        ValidationError,
        match=("'rectum' cannot be returned with its ancestor 'digestive tract' in body_site"),
    ):
        response_model.model_validate(
            {
                **empty_answer,
                "evidence_level": "sample",
                "body_site": ["digestive tract", "rectum"],
            }
        )

    assert fake.calls[0]["max_retries"] == 3


def test_model_selected_feces_does_not_infer_anatomical_origin(
    fixture_isolation_source_prompt_policy: IsolationSourcePromptPolicy,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = FakeClient()
    fake.respond_with(["feces"], reasoning="The evidence describes fecal material.")

    result = _classify_fixture_record(
        fixture_isolation_source_prompt_policy,
        tmp_path,
        values="gut contents",
        fake=fake,
        monkeypatch=monkeypatch,
    )

    assert result.selected_terms == (
        SelectedTerm(HOST_ASSOCIATED, "source_type", "host-associated"),
        SelectedTerm(ANIMAL_HOST, "source_type", "animal host"),
        SelectedTerm(FECES, "body_product", "feces"),
    )
    assert RECTUM not in {term.term_id for term in result.selected_terms}
    assert result.reasoning == (
        IsolationSourceReasoningStep(
            node="classifier",
            reasoning="The evidence describes fecal material.",
            selected_terms={"body_product": ("feces",)},
        ),
        IsolationSourceReasoningStep(
            node="source_type_derivation",
            reasoning="Filled facets determined the broad source kind.",
            selected_terms={"source_type": ("animal host",)},
        ),
        IsolationSourceReasoningStep(
            node="ancestor_expansion",
            reasoning="Selected terms were expanded to their facet ancestors.",
            selected_terms={"source_type": ("host-associated",)},
        ),
    )
    assert result.diagnostics == (IsolationSourceDiagnostic.LLM_CALL,)


def test_empty_model_result_remains_a_typed_outcome(
    fixture_isolation_source_prompt_policy: IsolationSourcePromptPolicy,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = FakeClient()
    fake.respond_with([], reasoning="No specific source is supported.", evidence_level="none")

    result = _classify_fixture_record(
        fixture_isolation_source_prompt_policy,
        tmp_path,
        values="GENOMIC",
        fake=fake,
        monkeypatch=monkeypatch,
    )

    assert result.selected_terms == ()
    assert result.evidence_level is IsolationSourceEvidenceLevel.NONE
    assert result.diagnostics == (
        IsolationSourceDiagnostic.LLM_CALL,
        IsolationSourceDiagnostic.UNSPECIFIED,
    )


def test_partial_deterministic_result_uses_sample_evidence_when_llm_is_disabled(
    tmp_path: Path,
) -> None:
    config_path = _isolation_source_config(tmp_path)
    standardizer = IsolationSourceStandardizer(
        IsolationSourcePromptPolicy.load(config_path),
        client=None,
        llm_settings=LLMSettings(None, None, "test-model"),
    )

    try:
        result = standardizer.standardize(
            {
                "accession": "PARTIAL_DETERMINISTIC",
                "iso_attr_orig": "isolation_source||environment",
                "iso_val_orig": "stool||unresolved material 12345",
            },
        )
    finally:
        standardizer.close()

    assert isinstance(result, IsolationSourceOutcome)
    assert result.selected_terms == (
        SelectedTerm(HOST_ASSOCIATED, "source_type", "host-associated"),
        SelectedTerm(ANIMAL_HOST, "source_type", "animal host"),
        SelectedTerm(FECES, "body_product", "feces"),
    )
    assert result.evidence_level is IsolationSourceEvidenceLevel.SAMPLE


def test_typed_record_outcome_preserves_model_reasoning_on_exact_cache_hit(tmp_path, monkeypatch):
    config_path = _isolation_source_config(tmp_path)
    monkeypatch.setenv("LLM_MODEL", "test-model")
    policy = IsolationSourcePromptPolicy.load(config_path)
    first_standardizer = IsolationSourceStandardizer(policy, client=None)
    fake_client = FakeClient()
    fake_client.respond_with(["wound"], reasoning="clinical wound")
    first_standardizer.pipeline.client = fake_client

    try:
        modelled = first_standardizer.standardize(
            {
                "accession": "MODELLED",
                "iso_attr_orig": "isolation_source",
                "iso_val_orig": "wound patient 1",
            },
        )
    finally:
        first_standardizer.close()

    cached_standardizer = IsolationSourceStandardizer(policy, client=None)
    try:
        cached = cached_standardizer.standardize(
            {
                "accession": "CACHED",
                "iso_attr_orig": "isolation_source",
                "iso_val_orig": "wound patient 1",
            },
        )
    finally:
        cached_standardizer.close()

    assert isinstance(modelled, IsolationSourceOutcome)
    assert modelled.evidence_level is IsolationSourceEvidenceLevel.SAMPLE
    assert modelled.diagnostics == (IsolationSourceDiagnostic.LLM_CALL,)
    assert modelled.reasoning == (
        IsolationSourceReasoningStep(
            node="classifier",
            reasoning="clinical wound",
            selected_terms={"lesion": ("wound",)},
        ),
        IsolationSourceReasoningStep(
            node="source_type_derivation",
            reasoning="Filled facets determined the broad source kind.",
            selected_terms={"source_type": ("animal host",)},
        ),
        IsolationSourceReasoningStep(
            node="ancestor_expansion",
            reasoning="Selected terms were expanded to their facet ancestors.",
            selected_terms={"source_type": ("host-associated",)},
        ),
    )
    assert isinstance(cached, IsolationSourceOutcome)
    assert cached.selected_terms == modelled.selected_terms
    assert cached.evidence_level is IsolationSourceEvidenceLevel.SAMPLE
    assert cached.reasoning == modelled.reasoning
    assert cached.diagnostics == (IsolationSourceDiagnostic.CACHE_HIT,)
    assert len(fake_client.calls) == 1


def test_warm_cache_uses_current_crosslinks(tmp_path: Path) -> None:
    ontology_directory = tmp_path / "ontology"
    shutil.copytree(FIXTURE_ROOT / "ontology", ontology_directory)
    config_path = _isolation_source_config(
        tmp_path,
        overrides={"ontology_directory": ontology_directory.as_posix()},
    )
    fake = FakeClient()
    fake.respond_with_facets(
        source_type="animal host",
        lesion=["abscess"],
        reasoning="The submitted material is from an abscess.",
    )

    first = _standardize_model_isolation_source(config_path, fake)
    terms_path = ontology_directory / "terms.tsv"
    terms_path.write_text(
        terms_path.read_text(encoding="utf-8").replace(
            "BACC:0000063\tabscess\tlesion\t\t\tBACC:0000017",
            "BACC:0000063\tabscess\tlesion\t\t\tBACC:0000010",
        ),
        encoding="utf-8",
    )
    second = _standardize_model_isolation_source(config_path, fake)

    assert SelectedTerm(PUS, "body_product", "pus") in first.selected_terms
    assert SelectedTerm(BLOOD, "body_product", "blood") in second.selected_terms
    assert SelectedTerm(PUS, "body_product", "pus") not in second.selected_terms
    assert second.diagnostics == (IsolationSourceDiagnostic.CACHE_HIT,)


def test_response_that_failed_validation_is_not_cached(tmp_path: Path) -> None:
    fake = FakeClient()
    fake.fail_with(
        InstructorRetryException(
            "invalid response",
            n_attempts=3,
            total_usage=0,
        )
    )
    standardizer = IsolationSourceStandardizer(
        IsolationSourcePromptPolicy.load(_isolation_source_config(tmp_path)),
        client=None,
        llm_settings=LLMSettings(None, None, "test-model"),
    )
    standardizer.pipeline.client = fake
    record = {
        "accession": "INVALID_THEN_VALID",
        "iso_attr_orig": "isolation_source",
        "iso_val_orig": "ambiguous material 6248",
    }
    try:
        failed = standardizer.standardize(record)
        fake.respond_with_facets(
            source_type="animal host",
            lesion=["wound"],
        )
        retried = standardizer.standardize(record)
    finally:
        standardizer.close()

    assert isinstance(failed, IsolationSourceRejection)
    assert failed.diagnostics == (IsolationSourceDiagnostic.CLASSIFICATION_FAILURE,)
    assert isinstance(retried, IsolationSourceOutcome)
    assert retried.diagnostics == (IsolationSourceDiagnostic.LLM_CALL,)


def test_invented_label_is_preserved_when_validation_retry_recovers(
    fixture_isolation_source_prompt_policy: IsolationSourcePromptPolicy,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    invalid_answer = _classifier_answer(
        reasoning="The sample names an unsupported anatomical site.",
        source_type="animal host",
        body_site=["nasal cavity", "nasal cavity"],
    )
    valid_answer = {**invalid_answer, "body_site": ["liver"]}

    def respond(response_model: type[BaseModel]) -> object:
        with pytest.raises(ValidationError):
            response_model.model_validate(invalid_answer)
        return response_model.model_validate(valid_answer)

    fake = FakeClient()
    fake.behavior = respond
    result = _standardize_fixture_record(
        fixture_isolation_source_prompt_policy,
        tmp_path,
        values="nasal aspirate 78143",
        fake=fake,
        monkeypatch=monkeypatch,
        accession="INVENTED_THEN_VALID",
    )

    assert isinstance(result, IsolationSourceOutcome)
    assert result.ontology_gap_diagnostics == (
        IsolationSourceOntologyGapDiagnostic(
            accession="INVENTED_THEN_VALID",
            facet="body_site",
            label="nasal cavity",
        ),
    )


def test_each_invented_label_is_preserved_when_validation_retries_are_exhausted(
    fixture_isolation_source_prompt_policy: IsolationSourcePromptPolicy,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    invalid_answer = _classifier_answer(
        reasoning="The sample names an unsupported anatomical site.",
        source_type="animal host",
        body_site=["nasal cavity"],
    )

    def respond(response_model: type[BaseModel]) -> object:
        for _ in range(3):
            with pytest.raises(ValidationError):
                response_model.model_validate(invalid_answer)
        raise InstructorRetryException("invalid response", n_attempts=3, total_usage=0)

    fake = FakeClient()
    fake.behavior = respond
    result = _standardize_fixture_record(
        fixture_isolation_source_prompt_policy,
        tmp_path,
        values="nasal aspirate 61512",
        fake=fake,
        monkeypatch=monkeypatch,
        accession="INVENTED_FAILURE",
    )

    expected_diagnostic = IsolationSourceOntologyGapDiagnostic(
        accession="INVENTED_FAILURE",
        facet="body_site",
        label="nasal cavity",
    )
    assert isinstance(result, IsolationSourceRejection)
    assert result.diagnostics == (IsolationSourceDiagnostic.CLASSIFICATION_FAILURE,)
    assert result.ontology_gap_diagnostics == (expected_diagnostic,) * 3


def test_ancestor_exclusion_does_not_create_an_ontology_gap_diagnostic(
    fixture_isolation_source_prompt_policy: IsolationSourcePromptPolicy,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    invalid_answer = _classifier_answer(
        reasoning="The response selects an ancestor and descendant.",
        source_type="animal host",
        body_site=["digestive tract", "rectum"],
    )
    valid_answer = {**invalid_answer, "body_site": ["rectum"]}

    def respond(response_model: type[BaseModel]) -> object:
        with pytest.raises(ValidationError):
            response_model.model_validate(invalid_answer)
        return response_model.model_validate(valid_answer)

    fake = FakeClient()
    fake.behavior = respond
    result = _standardize_fixture_record(
        fixture_isolation_source_prompt_policy,
        tmp_path,
        values="rectal sample 49115",
        fake=fake,
        monkeypatch=monkeypatch,
        accession="ANCESTOR_THEN_VALID",
    )

    assert isinstance(result, IsolationSourceOutcome)
    assert result.ontology_gap_diagnostics == ()


def test_direct_matches_receive_derived_source_kind_ancestors_and_preorder(
    tmp_path: Path,
) -> None:
    standardizer = IsolationSourceStandardizer(
        IsolationSourcePromptPolicy.load(_isolation_source_config(tmp_path)),
        client=None,
    )
    try:
        outcome = standardizer.standardize(
            {
                "accession": "DIRECT_ENRICHMENT",
                "iso_attr_orig": "specimen||isolation_source||anatomical_site||lesion",
                "iso_val_orig": "blood||stool||rectal swab||abscess",
            },
        )
    finally:
        standardizer.close()

    assert isinstance(outcome, IsolationSourceOutcome)
    assert outcome.selected_terms == (
        SelectedTerm(HOST_ASSOCIATED, "source_type", "host-associated"),
        SelectedTerm(ANIMAL_HOST, "source_type", "animal host"),
        SelectedTerm(BODY_FLUID, "body_product", "body fluid"),
        SelectedTerm(BLOOD, "body_product", "blood"),
        SelectedTerm(PUS, "body_product", "pus"),
        SelectedTerm(FECES, "body_product", "feces"),
        SelectedTerm(DIGESTIVE_TRACT, "body_site", "digestive tract"),
        SelectedTerm(INTESTINE, "body_site", "intestine"),
        SelectedTerm(RECTUM, "body_site", "rectum"),
        SelectedTerm(ABSCESS, "lesion", "abscess"),
    )


def test_crosslink_overwrites_classifier_source_kind_and_reports_disagreement(
    tmp_path: Path,
) -> None:
    fake = FakeClient()
    fake.respond_with_facets(
        source_type="environmental",
        environmental_material=["rhizosphere soil"],
    )
    standardizer = IsolationSourceStandardizer(
        IsolationSourcePromptPolicy.load(_isolation_source_config(tmp_path)),
        client=None,
        llm_settings=LLMSettings(None, None, "test-model"),
    )
    standardizer.pipeline.client = fake
    try:
        outcome = standardizer.standardize(
            {
                "accession": "CROSSLINK_OVERRIDE",
                "iso_attr_orig": "isolation_source",
                "iso_val_orig": "root-zone material 8472",
            },
        )
    finally:
        standardizer.close()

    assert isinstance(outcome, IsolationSourceOutcome)
    assert outcome.selected_terms == (
        SelectedTerm(HOST_ASSOCIATED, "source_type", "host-associated"),
        SelectedTerm(PLANT_HOST, "source_type", "plant host"),
        SelectedTerm(SOIL, "environmental_material", "soil"),
        SelectedTerm(RHIZOSPHERE, "environmental_material", "rhizosphere soil"),
    )
    assert outcome.diagnostics == (
        IsolationSourceDiagnostic.LLM_CALL,
        IsolationSourceDiagnostic.CROSSLINK_DISAGREEMENT,
    )


def test_direct_crosslink_conflict_does_not_report_classifier_disagreement(
    tmp_path: Path,
) -> None:
    standardizer = IsolationSourceStandardizer(
        IsolationSourcePromptPolicy.load(_isolation_source_config(tmp_path)),
        client=None,
    )
    try:
        outcome = standardizer.standardize(
            {
                "accession": "DIRECT_CROSSLINK_CONFLICT",
                "iso_attr_orig": "environment||isolation_source",
                "iso_val_orig": "environment||rhizosphere",
            },
        )
    finally:
        standardizer.close()

    assert isinstance(outcome, IsolationSourceOutcome)
    assert outcome.diagnostics == (IsolationSourceDiagnostic.EXACT_MATCH,)


def test_derived_source_kind_uses_host_environment_food_precedence(tmp_path: Path) -> None:
    fake = FakeClient()
    fake.respond_with_facets(
        source_type="food or feed",
        body_product=["blood"],
        environmental_material=["soil"],
        food_type=["meat product"],
    )
    standardizer = IsolationSourceStandardizer(
        IsolationSourcePromptPolicy.load(_isolation_source_config(tmp_path)),
        client=None,
        llm_settings=LLMSettings(None, None, "test-model"),
    )
    standardizer.pipeline.client = fake
    try:
        outcome = standardizer.standardize(
            {
                "accession": "SOURCE_KIND_PRECEDENCE",
                "iso_attr_orig": "isolation_source",
                "iso_val_orig": "mixed material 1937",
            },
        )
    finally:
        standardizer.close()

    assert isinstance(outcome, IsolationSourceOutcome)
    assert outcome.selected_terms == (
        SelectedTerm(HOST_ASSOCIATED, "source_type", "host-associated"),
        SelectedTerm(ANIMAL_HOST, "source_type", "animal host"),
        SelectedTerm(BODY_FLUID, "body_product", "body fluid"),
        SelectedTerm(BLOOD, "body_product", "blood"),
        SelectedTerm(SOIL, "environmental_material", "soil"),
        SelectedTerm(ANIMAL_FOOD_PRODUCT, "food_type", "animal food product"),
        SelectedTerm(MEAT_PRODUCT, "food_type", "meat product"),
    )
    assert IsolationSourceDiagnostic.CROSSLINK_DISAGREEMENT in outcome.diagnostics


@pytest.mark.parametrize(
    ("record", "answer", "expected_terms"),
    [
        (
            {
                "accession": "LABORATORY_ONLY",
                "iso_attr_orig": "isolation_source",
                "iso_val_orig": "laboratory strain",
            },
            None,
            (SelectedTerm(LABORATORY, "source_type", "laboratory"),),
        ),
    ],
)
def test_source_kind_stands_when_no_other_facet_implies_one(
    tmp_path: Path,
    record: dict[str, str],
    answer: str | None,
    expected_terms: tuple[SelectedTerm, ...],
) -> None:
    fake = FakeClient()
    if answer is not None:
        fake.respond_with([answer])
    standardizer = IsolationSourceStandardizer(
        IsolationSourcePromptPolicy.load(_isolation_source_config(tmp_path)),
        client=None,
        llm_settings=LLMSettings(None, None, "test-model"),
    )
    standardizer.pipeline.client = fake
    try:
        outcome = standardizer.standardize(record)
    finally:
        standardizer.close()

    assert isinstance(outcome, IsolationSourceOutcome)
    assert outcome.selected_terms == expected_terms


def _isolation_source_config(
    tmp_path: Path,
    *,
    overrides: dict[str, object] | None = None,
) -> Path:
    config_path = tmp_path / f"isolation-{len(list(tmp_path.glob('isolation-*.yaml')))}.yaml"
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    config.update(overrides or {})
    if not overrides or "ontology_directory" not in overrides:
        config["ontology_directory"] = (FIXTURE_ROOT / "ontology").as_posix()
    config["cache_db_path"] = (tmp_path / "isolation-cache.db").as_posix()
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return config_path


def test_prompt_policy_rejects_enrichment_term_in_the_wrong_facet(tmp_path: Path) -> None:
    ontology_directory = tmp_path / "ontology"
    shutil.copytree(FIXTURE_ROOT / "ontology", ontology_directory)
    terms_path = ontology_directory / "terms.tsv"
    terms_path.write_text(
        terms_path.read_text(encoding="utf-8").replace(
            "BACC:0000004\tenvironmental\tsource_type",
            "BACC:0000004\tenvironmental\tfood_type",
        ),
        encoding="utf-8",
    )

    with pytest.raises(
        PolicyConfigurationError,
        match=r"BACC:0000004.*source_type",
    ):
        IsolationSourcePromptPolicy.load(
            _isolation_source_config(
                tmp_path,
                overrides={"ontology_directory": ontology_directory.as_posix()},
            )
        )


def _standardize_model_isolation_source(
    config_path: Path,
    fake: FakeClient,
    *,
    model: str = "test-model",
) -> IsolationSourceOutcome:
    standardizer = IsolationSourceStandardizer(
        IsolationSourcePromptPolicy.load(config_path),
        client=None,
        llm_settings=LLMSettings(None, None, model),
    )
    standardizer.pipeline.client = fake
    try:
        result = standardizer.standardize(
            {
                "accession": "REQUEST_FINGERPRINT",
                "iso_attr_orig": "isolation_source",
                "iso_val_orig": "wound patient 739105",
            },
        )
    finally:
        standardizer.close()
    assert isinstance(result, IsolationSourceOutcome)
    return result


def test_isolation_source_llm_observability_uses_the_published_target_key(
    tmp_path: Path,
) -> None:
    fake = FakeClient()
    fake.respond_with(["wound"])
    observability = LLMObservability({"isolation_source": "test-model"})
    observability.start()
    try:
        _standardize_model_isolation_source(_isolation_source_config(tmp_path), fake)
        snapshot = observability.snapshot()
    finally:
        observability.close()

    assert [entry["target"] for entry in snapshot["by_target_and_model"]] == ["isolation_source"]


def test_isolation_source_standardizer_does_not_reopen_loaded_policy_source(tmp_path: Path) -> None:
    config_path = _isolation_source_config(tmp_path)
    policy = IsolationSourcePromptPolicy.load(config_path)
    config_path.write_text("broken: [\n", encoding="utf-8")

    standardizer = IsolationSourceStandardizer(
        policy,
        client=None,
    )
    try:
        outcome = standardizer.standardize(
            {
                "accession": "NO_REOPEN",
                "iso_attr_orig": "isolation_source",
                "iso_val_orig": "stool",
            },
        )
    finally:
        standardizer.close()

    assert isinstance(outcome, IsolationSourceOutcome)
    assert outcome.selected_terms == (
        SelectedTerm(HOST_ASSOCIATED, "source_type", "host-associated"),
        SelectedTerm(ANIMAL_HOST, "source_type", "animal host"),
        SelectedTerm(FECES, "body_product", "feces"),
    )


def test_isolation_source_cache_reuses_identical_request_when_only_prompt_metadata_changes(
    tmp_path,
):
    fake = FakeClient()
    fake.respond_with(["wound"], reasoning="clinical wound")
    first_config = _isolation_source_config(tmp_path, overrides={"prompt_version": "first"})
    second_config = _isolation_source_config(tmp_path, overrides={"prompt_version": "second"})

    first = _standardize_model_isolation_source(first_config, fake)
    second = _standardize_model_isolation_source(second_config, fake)

    assert first.diagnostics == (IsolationSourceDiagnostic.LLM_CALL,)
    assert second.diagnostics == (IsolationSourceDiagnostic.CACHE_HIT,)
    assert second.reasoning == first.reasoning
    assert len(fake.calls) == 1


def test_isolation_source_request_uses_rendered_synthetic_prompts(tmp_path: Path) -> None:
    fake = FakeClient()
    fake.respond_with(["wound"], reasoning="clinical wound")
    config_path = _isolation_source_config(
        tmp_path,
        overrides={
            "prompt_version": "synthetic-v1",
            "system_prompt": "Synthetic ontology:\n{ontology_tree}",
            "user_prompt": "Synthetic sample:\n{metadata}",
            "ontology_directory": (
                ROOT / "tests" / "fixtures" / "standardization" / "ontology"
            ).as_posix(),
        },
    )
    standardizer = IsolationSourceStandardizer(
        IsolationSourcePromptPolicy.load(config_path),
        client=None,
        llm_settings=LLMSettings(None, None, "test-model"),
    )
    standardizer.pipeline.client = fake
    try:
        direct = standardizer.standardize(
            {
                "accession": "DIRECT",
                "iso_attr_orig": "isolation_source",
                "iso_val_orig": "stool",
            },
        )
        assert fake.calls == []
        outcome = standardizer.standardize(
            {
                "accession": "REQUEST_FINGERPRINT",
                "iso_attr_orig": "isolation_source",
                "iso_val_orig": "wound patient 739105",
            },
        )
    finally:
        standardizer.close()

    assert isinstance(direct, IsolationSourceOutcome)
    assert direct.diagnostics == (IsolationSourceDiagnostic.EXACT_MATCH,)
    assert fake.calls[0]["messages"][0]["role"] == "system"
    system_message = fake.calls[0]["messages"][0]["content"]
    user_message = fake.calls[0]["messages"][1]
    assert "Synthetic ontology:" in system_message
    assert "wound" in system_message
    assert user_message["role"] == "user"
    assert "isolation_source = wound patient 739105" in user_message["content"]
    assert "Homo sapiens" not in str(fake.calls[0]["messages"])
    assert fake.calls[0]["model"] == "test-model"
    assert fake.calls[0]["temperature"] == 0
    assert fake.calls[0]["seed"] == 100
    assert fake.calls[0]["response_model"].__name__ == "IsolationSourceClassification"
    assert outcome.request_fingerprint


@pytest.mark.parametrize(
    "changed_component",
    ["message", "model", "parameter", "schema_id", "structured_output_mode"],
)
def test_isolation_source_cache_misses_when_canonical_request_changes(
    tmp_path, monkeypatch, changed_component
):
    fake = FakeClient()
    fake.respond_with(["wound"], reasoning="clinical wound")
    first_config = _isolation_source_config(tmp_path)
    _standardize_model_isolation_source(first_config, fake)

    second_config = _isolation_source_config(tmp_path)
    second_model = "test-model"
    if changed_component == "message":
        second_config = _isolation_source_config(
            tmp_path,
            overrides={"system_prompt": "{ontology_tree}\nA changed fully rendered system prompt."},
        )
    elif changed_component == "model":
        second_model = "changed-model"
    elif changed_component == "parameter":
        monkeypatch.setattr(
            isolation_source_module,
            "ISOLATION_SOURCE_LLM_PARAMETERS",
            {"temperature": 0, "seed": 101},
        )
    elif changed_component == "schema_id":
        monkeypatch.setattr(
            isolation_source_module,
            "ISOLATION_SOURCE_RESPONSE_SCHEMA_ID",
            "baccurate.isolation.classification.changed",
        )
    elif changed_component == "structured_output_mode":
        monkeypatch.setattr(
            isolation_source_module,
            "ISOLATION_SOURCE_STRUCTURED_OUTPUT_MODE",
            Mode.TOOLS,
        )
    second = _standardize_model_isolation_source(
        second_config,
        fake,
        model=second_model,
    )

    assert second.diagnostics == (IsolationSourceDiagnostic.LLM_CALL,)
    assert len(fake.calls) == 2


# --- Cache persistence adapter ---


def test_cache_round_trip_preserves_only_the_classifier_answer(cache):
    answer = IsolationSourceClassifierAnswer(
        facet_values={
            "source_type": "animal host",
            "body_product": ("feces",),
            "body_site": (),
            "lesion": (),
            "environmental_material": (),
            "facility": (),
            "sampled_object": (),
            "food_type": (),
        },
        reasoning="The sample is fecal material from an animal host.",
        evidence_level=IsolationSourceEvidenceLevel.SAMPLE,
    )
    cache.set("fingerprint", answer)

    got = cache.get("fingerprint")

    assert got == answer
