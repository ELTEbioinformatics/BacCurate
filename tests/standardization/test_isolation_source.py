"""Pin the isolation-source standardization contract."""

from __future__ import annotations

import re
import shutil
from dataclasses import replace
from pathlib import Path

import pytest
import yaml
from instructor import Mode
from instructor.core import InstructorRetryException
from pydantic import BaseModel, ValidationError

import baccurate.standardization.isolation_source as isolation_source_module
from baccurate.adapters.config_yaml import ConfigError
from baccurate.adapters.llm.client import LLMSettings
from baccurate.standardization.host import HostOverflowContext
from baccurate.standardization.isolation_source import (
    IsolationSourceClassifierAnswer,
    IsolationSourceDiagnostic,
    IsolationSourceEvidenceLevel,
    IsolationSourceOntologyGapDiagnostic,
    IsolationSourceOutcome,
    IsolationSourceRejection,
    IsolationSourceStandardizer,
    SelectedTerm,
    SQLiteCache,
)
from baccurate.standardization.isolation_source_config import (
    IsolationSourceConfig,
)
from baccurate.standardization.supporting_attribute_value_pair import SupportingAttributeValuePair

ROOT = Path(__file__).parents[2]
CONFIG_PATH = ROOT / "config" / "isolation_source.yaml"
FIXTURE_ROOT = ROOT / "tests" / "fixtures" / "standardization"
PRODUCTION_VOCABULARY = {
    "ontology_directory": (ROOT / "data" / "reference" / "ontology").as_posix(),
    **{
        key: yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))[key]
        for key in (
            "host_recovery_term_ids",
            "direct_match_values",
            "disease_body_terms",
        )
    },
}


# Stable term identifiers used across the tests.
FECES = "BACC:0000025"
RECTUM = "BACC:0000041"
BLOOD = "BACC:0000010"
BODY_FLUID = "BACC:0000009"
DIGESTIVE_TRACT = "BACC:0000035"
INTESTINE = "BACC:0000039"
LIVER = "BACC:0000057"
RHIZOSPHERE = "BACC:0000065"
SOIL = "BACC:0000064"
HOST_ASSOCIATED = "BACC:0000001"
ANIMAL_HOST = "BACC:0000002"
HUMAN_HOST = "BACC:0000122"
PLANT_HOST = "BACC:0000003"
ENVIRONMENTAL = "BACC:0000004"
FOOD_OR_FEED = "BACC:0000007"
LABORATORY = "BACC:0000008"
PUS = "BACC:0000017"
ABSCESS = "BACC:0000063"
HEALTHCARE_FACILITY = "BACC:0000079"
INANIMATE_OBJECT = "BACC:0000087"
SINK = "BACC:0000088"
ANIMAL_FOOD_PRODUCT = "BACC:0000095"
MEAT_PRODUCT = "BACC:0000097"
HUMAN_MILK = "BACC:0000105"
DISEASE = "BACC:0000128"
CYSTIC_FIBROSIS = "BACC:0000149"
URINARY_TRACT_INFECTION = "BACC:0000134"


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


def _standardize_fixture_record(
    config: IsolationSourceConfig,
    tmp_path: Path,
    *,
    values: str,
    attributes: str = "isolation_source",
    fake=None,
    monkeypatch: pytest.MonkeyPatch | None = None,
    accession: str = "PUBLIC_CLASSIFICATION",
    overflow: HostOverflowContext | None = None,
) -> IsolationSourceOutcome | IsolationSourceRejection:
    if fake is not None:
        assert monkeypatch is not None
        monkeypatch.setattr(
            isolation_source_module.instructor,
            "from_openai",
            lambda client, **_kwargs: client,
        )
    standardizer = IsolationSourceStandardizer(
        config,
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
            overflow=overflow,
        )
    finally:
        standardizer.close()
    return result


def _classify_fixture_record(
    config: IsolationSourceConfig,
    tmp_path: Path,
    **kwargs: object,
) -> IsolationSourceOutcome:
    result = _standardize_fixture_record(config, tmp_path, **kwargs)
    assert isinstance(result, IsolationSourceOutcome)
    return result


@pytest.mark.parametrize(
    ("submitted_value", "expected_terms"),
    [
        (
            "feces",
            (
                SelectedTerm(HOST_ASSOCIATED, "source_type", "host-associated"),
                SelectedTerm(FECES, "body_product", "feces"),
            ),
        ),
        (
            "rectal swab",
            (
                SelectedTerm(HOST_ASSOCIATED, "source_type", "host-associated"),
                SelectedTerm(DIGESTIVE_TRACT, "body_site", "digestive tract"),
                SelectedTerm(INTESTINE, "body_site", "intestine"),
                SelectedTerm(RECTUM, "body_site", "rectum"),
            ),
        ),
        (
            "inpatient",
            (
                SelectedTerm(HOST_ASSOCIATED, "source_type", "host-associated"),
                SelectedTerm(HUMAN_HOST, "source_type", "human host"),
                SelectedTerm(HEALTHCARE_FACILITY, "facility", "healthcare facility"),
            ),
        ),
        (
            "ENVO:02000020",
            (
                SelectedTerm(HOST_ASSOCIATED, "source_type", "host-associated"),
                SelectedTerm(BODY_FLUID, "body_product", "body fluid"),
                SelectedTerm(BLOOD, "body_product", "blood"),
            ),
        ),
        (
            "rhizosphere",
            (
                SelectedTerm(HOST_ASSOCIATED, "source_type", "host-associated"),
                SelectedTerm(PLANT_HOST, "source_type", "plant host"),
                SelectedTerm(ENVIRONMENTAL, "source_type", "environmental"),
                SelectedTerm(SOIL, "environmental_material", "soil"),
                SelectedTerm(RHIZOSPHERE, "environmental_material", "rhizosphere soil"),
            ),
        ),
    ],
)
def test_public_classification_resolves_ontology_terms(
    fixture_isolation_source_config: IsolationSourceConfig,
    tmp_path: Path,
    submitted_value: str,
    expected_terms: tuple[SelectedTerm, ...],
) -> None:
    result = _classify_fixture_record(
        fixture_isolation_source_config,
        tmp_path,
        values=submitted_value,
    )

    assert result.selected_terms == expected_terms


@pytest.mark.parametrize(
    ("submitted_value", "expected_eligible", "expected_term_id"),
    [
        pytest.param("host-associated", True, HOST_ASSOCIATED, id="broad-host-source-kind"),
        pytest.param("meat product", True, MEAT_PRODUCT, id="meat-product"),
        pytest.param("animal feed", False, "BACC:0000100", id="animal-feed"),
        pytest.param("breast milk", True, HUMAN_MILK, id="human-milk"),
    ],
)
def test_assigned_term_flag_decides_host_recovery_eligibility_without_classifier(
    tmp_path: Path,
    submitted_value: str,
    expected_eligible: bool,
    expected_term_id: str,
) -> None:
    config = IsolationSourceConfig.load(
        _isolation_source_config(
            tmp_path,
            overrides=PRODUCTION_VOCABULARY,
        )
    )

    result = _classify_fixture_record(config, tmp_path, values=submitted_value)

    assert result.host_recovery_eligible is expected_eligible
    assert result.selected_terms[-1].term_id == expected_term_id
    assert result.llm_calls == 0


PHARYNX_BODY_SITES = {"respiratory system", "upper respiratory tract", "pharynx"}


@pytest.mark.parametrize(
    ("submitted_value", "expected_body_sites"),
    [
        ("leg", {"limb", "lower limb"}),
        ("Knee", {"limb", "lower limb"}),
        ("Left-Popliteal-Fossa", {"limb", "lower limb"}),
        ("foot", {"limb", "lower limb"}),
        ("arm", {"limb", "upper limb"}),
        ("hand swab", {"limb", "upper limb"}),
        ("Groin", {"abdomen", "inguinal region"}),
        ("groin swab", {"abdomen", "inguinal region"}),
        ("Axilla", {"axilla"}),
        ("perineal swab", {"perineum"}),
        ("abdominal", {"abdomen"}),
        ("spleen", {"spleen"}),
        ("gall bladder", {"digestive system", "gallbladder"}),
        ("UBERON:0008337", {"abdomen", "inguinal region"}),
        ("stool", {"digestive system", "digestive tract"}),
        ("throat", {*PHARYNX_BODY_SITES, "digestive system", "digestive tract"}),
        ("pharynx", {*PHARYNX_BODY_SITES, "digestive system", "digestive tract"}),
        ("nasopharyngeal swab", {*PHARYNX_BODY_SITES, "nasopharynx"}),
        (
            "oropharyngeal swab",
            {*PHARYNX_BODY_SITES, "oropharynx", "digestive system", "digestive tract"},
        ),
        ("skin", {"skin and soft tissue", "skin"}),
        ("Soft tissue", {"skin and soft tissue", "soft tissue"}),
        ("skeletal muscle", {"skin and soft tissue", "soft tissue", "muscle tissue"}),
        ("SSTI", {"skin and soft tissue"}),
        ("skin and soft tissue infection", {"skin and soft tissue"}),
        ("Skin/Soft Tissue Infection", {"skin and soft tissue"}),
        ("Skin or soft tissue", {"skin and soft tissue"}),
    ],
)
def test_anatomical_input_resolves_without_classifier(
    tmp_path: Path,
    submitted_value: str,
    expected_body_sites: set[str],
) -> None:
    config = IsolationSourceConfig.load(
        _isolation_source_config(
            tmp_path,
            overrides=PRODUCTION_VOCABULARY,
        )
    )

    result = _classify_fixture_record(config, tmp_path, values=submitted_value)

    assert {
        term.label for term in result.selected_terms if term.facet == "body_site"
    } == expected_body_sites
    assert result.llm_calls == 0


@pytest.mark.parametrize(
    ("label", "implies_environmental"),
    [("catheter", False), ("medical device", False), ("equipment", True), ("sink", True)],
)
def test_only_a_sampled_object_outside_a_patient_implies_environmental(
    tmp_path: Path, label: str, implies_environmental: bool
) -> None:
    ontology = _repository_config(tmp_path).ontology
    term_id = next(term_id for term_id, term in ontology.terms.items() if term.label == label)

    assert (ENVIRONMENTAL in ontology.derive({term_id})) == implies_environmental


@pytest.mark.parametrize(
    ("submitted_value", "expected_labels"),
    [
        ("catheter tip", {"inanimate object", "medical device", "catheter"}),
        (
            "urine from catheter",
            {"host-associated", "body fluid", "urine", "genitourinary system", "urinary system"},
        ),
    ],
)
def test_catheter_specimen_resolves_without_a_model_call(
    tmp_path: Path, submitted_value: str, expected_labels: set[str]
) -> None:
    result = _classify_fixture_record(
        _repository_config(tmp_path), tmp_path, values=submitted_value
    )

    assert {term.label for term in result.selected_terms} == expected_labels
    assert result.llm_calls == 0


def test_derive_of_every_term_holds_its_ancestors_and_relation_objects_as_production_does(
    tmp_path: Path,
) -> None:
    config = _repository_config(tmp_path)
    ontology = config.ontology

    for term_id, term in ontology.terms.items():
        if term.facet == "host_disease":
            continue
        ancestors = set()
        parent_id = term.parent_id
        while parent_id is not None:
            ancestors.add(parent_id)
            parent_id = ontology.terms[parent_id].parent_id
        relation_objects = {
            relation.object_id for relation in ontology.relations if relation.subject_id == term_id
        }
        derived = ontology.derive({term_id})
        result = _classify_fixture_record(config, tmp_path, values=term.label)

        assert {term_id, *ancestors, *relation_objects} <= derived, term.label
        assert {selected.term_id for selected in result.selected_terms} == derived, term.label


@pytest.mark.parametrize(
    ("label", "expected_labels", "expected_source_types"),
    [
        (
            "pharynx",
            {
                "respiratory system",
                "upper respiratory tract",
                "pharynx",
                "digestive system",
                "digestive tract",
            },
            set(),
        ),
        (
            "nasopharynx",
            {"respiratory system", "upper respiratory tract", "pharynx", "nasopharynx"},
            set(),
        ),
        (
            "sputum",
            {"respiratory secretion", "sputum", "respiratory system", "lower respiratory tract"},
            set(),
        ),
        (
            "bronchoalveolar lavage fluid",
            {
                "bronchoalveolar lavage fluid",
                "respiratory secretion",
                "respiratory system",
                "lower respiratory tract",
                "lung",
            },
            set(),
        ),
        ("abscess", {"abscess", "body fluid", "pus"}, set()),
        ("rhizosphere soil", {"soil", "rhizosphere soil"}, {"plant host", "host-associated"}),
    ],
)
def test_term_derivation_gives_the_accepted_final_terms(
    tmp_path: Path,
    label: str,
    expected_labels: set[str],
    expected_source_types: set[str],
) -> None:
    result = _classify_fixture_record(_repository_config(tmp_path), tmp_path, values=label)

    assert {
        term.label for term in result.selected_terms if term.facet != "source_type"
    } == expected_labels
    assert expected_source_types <= {
        term.label for term in result.selected_terms if term.facet == "source_type"
    }
    assert result.llm_calls == 0


def _repository_config(tmp_path: Path) -> IsolationSourceConfig:
    return IsolationSourceConfig.load(
        _isolation_source_config(
            tmp_path,
            overrides=PRODUCTION_VOCABULARY,
        )
    )


DIGESTIVE_TRACT_SITES = {"digestive system", "digestive tract"}
URINARY_SITES = {"genitourinary system", "urinary system"}


@pytest.mark.parametrize(
    ("submitted_value", "expected_labels"),
    [
        ("rectal swab", {*DIGESTIVE_TRACT_SITES, "intestine", "rectum"}),
        ("gut contents", {"digestive tract content", *DIGESTIVE_TRACT_SITES}),
        *[
            (value, {"digestive tract content", *DIGESTIVE_TRACT_SITES, "intestine", "caecum"})
            for value in ("caecal contents", "cecal content")
        ],
        (
            "intestinal contents",
            {"digestive tract content", *DIGESTIVE_TRACT_SITES, "intestine"},
        ),
        ("animal-swine-sow (cecal)", {*DIGESTIVE_TRACT_SITES, "intestine", "caecum"}),
        ("perianal swab", {"perineum"}),
        ("fecal swab", {"feces", *DIGESTIVE_TRACT_SITES}),
        *[
            (
                value,
                {
                    "respiratory secretion",
                    "tracheal aspirate",
                    "respiratory system",
                    "lower respiratory tract",
                    "trachea",
                },
            )
            for value in ("tracheal aspirate", "endotracheal aspirate")
        ],
        ("drainage", {"body fluid", "drainage fluid"}),
        ("drainage fluid", {"body fluid", "drainage fluid"}),
        ("Cyst", {"cyst"}),
        ("urinary", URINARY_SITES),
        ("urinary tract", URINARY_SITES),
        *[
            (value, {"body fluid", "urine", *URINARY_SITES})
            for value in (
                "URINE_CLEAN CATCH/VOIDED",
                "Urine, Clean Catch",
                "void urine",
                "midstream urine",
            )
        ],
    ],
)
def test_product_and_lesion_input_resolves_without_classifier(
    tmp_path: Path,
    submitted_value: str,
    expected_labels: set[str],
) -> None:
    result = _classify_fixture_record(
        _repository_config(tmp_path), tmp_path, values=submitted_value
    )

    assert {
        term.label for term in result.selected_terms if term.facet != "source_type"
    } == expected_labels
    assert result.llm_calls == 0


@pytest.mark.parametrize(
    "submitted_value",
    ["drain", "clean catch"],
)
def test_input_that_needs_a_site_or_context_reaches_classifier(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    submitted_value: str,
    fake_llm_client,
) -> None:
    fake_llm_client.respond_with_facets()

    result = _classify_fixture_record(
        _repository_config(tmp_path),
        tmp_path,
        values=submitted_value,
        fake=fake_llm_client,
        monkeypatch=monkeypatch,
    )

    assert result.llm_calls == 1


@pytest.mark.parametrize(
    ("submitted_value", "model_labels", "expected_labels"),
    [
        (
            "caecal contents",
            ["digestive tract content", "caecum"],
            {"digestive tract content", *DIGESTIVE_TRACT_SITES, "intestine", "caecum"},
        ),
        (
            "intestinal contents",
            ["digestive tract content", "intestine"],
            {"digestive tract content", *DIGESTIVE_TRACT_SITES, "intestine"},
        ),
        (
            "rectum contents",
            ["digestive tract content", "rectum"],
            {"digestive tract content", *DIGESTIVE_TRACT_SITES, "intestine", "rectum"},
        ),
        (
            "wound drainage",
            ["drainage fluid", "wound"],
            {"body fluid", "drainage fluid", "wound"},
        ),
        (
            "liver abscess drainage fluid",
            ["liver", "abscess"],
            {"body fluid", "pus", "digestive system", "liver", "abscess"},
        ),
        ("cyst fluid", ["cyst", "body fluid"], {"body fluid", "cyst"}),
        (
            "cystocentesis urine sample",
            ["urine", "lower urinary tract"],
            {"body fluid", "urine", *URINARY_SITES, "lower urinary tract"},
        ),
    ],
)
def test_model_answer_labels_derive_the_same_terms_as_derive(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    submitted_value: str,
    model_labels: list[str],
    expected_labels: set[str],
    fake_llm_client,
) -> None:
    config = _repository_config(tmp_path)
    terms_by_label = {term.label: term for term in config.ontology.terms.values()}
    answer: dict[str, list[str]] = {}
    for label in model_labels:
        answer.setdefault(terms_by_label[label].facet, []).append(label)
    fake_llm_client.respond_with_facets(**answer)

    result = _classify_fixture_record(
        config, tmp_path, values=submitted_value, fake=fake_llm_client, monkeypatch=monkeypatch
    )

    selected = [term for term in result.selected_terms if term.facet != "source_type"]
    explicit_ids = {terms_by_label[label].term_id for label in model_labels}
    assert {term.label for term in selected} == expected_labels
    assert {term.term_id for term in result.selected_terms} == config.ontology.derive(explicit_ids)


def test_configured_host_recovery_term_is_eligible_after_a_direct_match(tmp_path: Path) -> None:
    config = _repository_config(tmp_path)

    for term_id in sorted(config.host_recovery_term_ids):
        result = _classify_fixture_record(
            config, tmp_path, values=config.ontology.terms[term_id].label
        )

        assert result.llm_calls == 0, term_id
        assert result.host_recovery_eligible is True, term_id


def test_each_label_synonym_and_direct_match_value_resolves_to_its_term(tmp_path: Path) -> None:
    config = _repository_config(tmp_path)
    names = [
        *(
            (name, term.term_id)
            for term in config.ontology.terms.values()
            for name in (
                term.label,
                *(synonym.text for synonym in term.synonyms),
            )
        ),
        *(
            (value, term_id)
            for value, term_ids in config.direct_match_values.items()
            for term_id in term_ids
        ),
    ]
    standardizer = IsolationSourceStandardizer(config, client=None)
    try:
        for name, term_id in names:
            result = standardizer.standardize(
                {"accession": "DIRECT", "iso_attr_orig": "isolation_source", "iso_val_orig": name}
            )

            assert isinstance(result, IsolationSourceOutcome), name
            assert result.llm_calls == 0, name
            assert term_id in {term.term_id for term in result.selected_terms}, name
    finally:
        standardizer.close()


@pytest.mark.parametrize("value", ["cell culture", "Nickels Lab"])
def test_cell_culture_and_a_laboratory_name_go_to_the_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, value: str, fake_llm_client
) -> None:
    fake_llm_client.respond_with_facets()

    result = _classify_fixture_record(
        _repository_config(tmp_path),
        tmp_path,
        values=value,
        fake=fake_llm_client,
        monkeypatch=monkeypatch,
    )

    assert result.llm_calls == 1


def test_host_recovery_eligibility_follows_the_configuration_instead_of_term_identity(
    tmp_path: Path,
) -> None:
    config = IsolationSourceConfig.load(
        _isolation_source_config(
            tmp_path,
            overrides={"host_recovery_term_ids": [ENVIRONMENTAL]},
        )
    )

    host_associated = _classify_fixture_record(config, tmp_path, values="clinical sample")
    environmental = _classify_fixture_record(config, tmp_path, values="environment")

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
        ConfigError,
        match=r"BACC:0000003.*to descend from.*BACC:0000001",
    ):
        IsolationSourceConfig.load(
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
    fixture_isolation_source_config: IsolationSourceConfig,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    submitted_value: str,
    fake_llm_client,
) -> None:
    fake_llm_client.respond_with(["blood"], reasoning="The whole submitted value was classified.")

    result = _classify_fixture_record(
        fixture_isolation_source_config,
        tmp_path,
        values=submitted_value,
        fake=fake_llm_client,
        monkeypatch=monkeypatch,
    )

    assert result.exact_matches == 0
    assert result.llm_calls == 1
    assert fake_llm_client.calls[0]["messages"][1]["content"].count(submitted_value) == 1


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
    fixture_isolation_source_config: IsolationSourceConfig,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    submitted_value: str,
    fake_llm_client,
) -> None:

    result = _classify_fixture_record(
        fixture_isolation_source_config,
        tmp_path,
        values=submitted_value,
        fake=fake_llm_client,
        monkeypatch=monkeypatch,
    )

    assert SelectedTerm(BLOOD, "body_product", "blood") in result.selected_terms
    assert result.exact_matches == 1
    assert result.llm_calls == 0
    assert fake_llm_client.calls == []


def test_disagreeing_paired_label_and_identifier_reports_diagnostic(
    fixture_isolation_source_config: IsolationSourceConfig,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fake_llm_client,
) -> None:
    fake_llm_client.respond_with(
        ["blood"], reasoning="The conflicting annotation was classified as blood."
    )

    result = _classify_fixture_record(
        fixture_isolation_source_config,
        tmp_path,
        values="blood [ENVO:00002003]",
        fake=fake_llm_client,
        monkeypatch=monkeypatch,
    )

    assert result.exact_matches == 0
    assert result.llm_calls == 1
    assert IsolationSourceDiagnostic.IDENTIFIER_DISAGREEMENT in result.diagnostics
    assert "blood [ENVO:00002003]" in fake_llm_client.calls[0]["messages"][1]["content"]


def test_disagreeing_pair_reports_diagnostic_when_classifier_validation_fails(
    fixture_isolation_source_config: IsolationSourceConfig,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fake_llm_client,
) -> None:
    fake_llm_client.behavior = InstructorRetryException(
        "invalid response",
        n_attempts=3,
        total_usage=0,
    )
    monkeypatch.setattr(
        isolation_source_module.instructor,
        "from_openai",
        lambda client, **_kwargs: client,
    )
    standardizer = IsolationSourceStandardizer(
        fixture_isolation_source_config,
        client=fake_llm_client,
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
    fixture_isolation_source_config: IsolationSourceConfig,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fake_llm_client,
) -> None:

    result = _classify_fixture_record(
        fixture_isolation_source_config,
        tmp_path,
        values="blood [ENVO:ENVO:02000020]",
        fake=fake_llm_client,
        monkeypatch=monkeypatch,
    )

    assert result.selected_terms[-1] == SelectedTerm(BLOOD, "body_product", "blood")
    assert result.exact_matches == 1
    assert result.llm_calls == 0
    assert fake_llm_client.calls == []


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
    fake_llm_client,
) -> None:
    config = replace(
        IsolationSourceConfig.load(CONFIG_PATH),
        cache_db_path=tmp_path / "isolation-source-cache.db",
    )
    if not resolves:
        fake_llm_client.respond_with(
            ["blood"], reasoning="The non-resolving mapping was classified."
        )

    result = _classify_fixture_record(
        config,
        tmp_path,
        values=submitted_value,
        fake=fake_llm_client,
        monkeypatch=monkeypatch,
    )

    assert result.exact_matches == int(resolves)
    assert result.llm_calls == int(not resolves)


def test_obsolete_identifier_resolves_like_its_replacements(tmp_path: Path) -> None:
    config = replace(
        IsolationSourceConfig.load(CONFIG_PATH),
        cache_db_path=tmp_path / "obsolete-cache.db",
    )

    for obsolete in config.ontology.obsolete_terms.values():
        if not obsolete.replaced_by:
            continue
        replacements = _classify_fixture_record(
            config,
            tmp_path,
            values="||".join(obsolete.replaced_by),
            attributes="||".join(["isolation_source"] * len(obsolete.replaced_by)),
        )
        resolved = _classify_fixture_record(config, tmp_path, values=obsolete.term_id)

        assert resolved.llm_calls == 0, obsolete.term_id
        assert resolved.selected_terms == replacements.selected_terms, obsolete.term_id


def test_explicit_non_source_value_does_not_call_model(
    fixture_isolation_source_config: IsolationSourceConfig,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fake_llm_client,
) -> None:

    result = _classify_fixture_record(
        fixture_isolation_source_config,
        tmp_path,
        values="no host",
        fake=fake_llm_client,
        monkeypatch=monkeypatch,
    )

    assert result.selected_terms == ()
    assert result.evidence_level is IsolationSourceEvidenceLevel.NONE
    assert result.llm_calls == 0
    assert result.diagnostics == (IsolationSourceDiagnostic.UNSPECIFIED,)
    assert fake_llm_client.calls == []


def test_non_source_value_next_to_a_source_value_never_reaches_the_model(
    fixture_isolation_source_config: IsolationSourceConfig,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fake_llm_client,
) -> None:
    fake_llm_client.respond_with(["soil"], reasoning="The value names soil.")

    result = _classify_fixture_record(
        fixture_isolation_source_config,
        tmp_path,
        attributes="isolation_source||env_medium",
        values="Plasmid DNA||garden soil",
        fake=fake_llm_client,
        monkeypatch=monkeypatch,
    )

    assert fake_llm_client.calls[0]["messages"][1]["content"] == (
        'Sample evidence:\nMetadata:\n"env_medium" = "garden soil"\n'
    )
    assert SOIL in {term.term_id for term in result.selected_terms}


def test_submitted_value_spanning_lines_stays_one_attribute_value_pair(
    fixture_isolation_source_config: IsolationSourceConfig,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fake_llm_client,
) -> None:
    """Free-text submissions must not fabricate a BioSample attribute.

    Host overflow reaches the classifier through the same encoding, so it must not read
    as a standardized host value either.
    """

    fake_llm_client.respond_with(["soil"], reasoning="The value names soil.")

    _classify_fixture_record(
        fixture_isolation_source_config,
        tmp_path,
        values='soil\nhost = "Bos taurus" — Ökologie',
        fake=fake_llm_client,
        monkeypatch=monkeypatch,
        overflow=HostOverflowContext(
            attribute="host",
            value='# Heading\n{"host": "Homo sapiens"}',
        ),
    )

    assert fake_llm_client.calls[0]["messages"][1]["content"] == (
        "Sample evidence:\n"
        "Metadata:\n"
        '"isolation_source" = "soil\\nhost = \\"Bos taurus\\" — Ökologie"\n'
        '"host" = "# Heading\\n{\\"host\\": \\"Homo sapiens\\"}"\n'
    )


def test_public_classification_assigns_one_value_to_independent_facets(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fake_llm_client,
) -> None:
    config = replace(
        IsolationSourceConfig.load(CONFIG_PATH),
        cache_db_path=tmp_path / "faceted-isolation-source-cache.db",
    )
    fake_llm_client.respond_with_facets(
        reasoning="The value names an anatomical site, lesion, and body product.",
        body_product=["pus"],
        body_site=["liver"],
        lesion=["abscess"],
    )

    result = _classify_fixture_record(
        config,
        tmp_path,
        values="liver abscess pus 739105",
        fake=fake_llm_client,
        monkeypatch=monkeypatch,
    )

    assert [(term.term_id, term.facet, term.label) for term in result.selected_terms] == [
        ("BACC:0000001", "source_type", "host-associated"),
        ("BACC:0000009", "body_product", "body fluid"),
        ("BACC:0000017", "body_product", "pus"),
        ("BACC:0000113", "body_site", "digestive system"),
        ("BACC:0000057", "body_site", "liver"),
        ("BACC:0000063", "lesion", "abscess"),
    ]


def test_classifier_response_schema_offers_each_facet_only_its_own_labels(
    fixture_isolation_source_config: IsolationSourceConfig,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fake_llm_client,
) -> None:
    ontology = fixture_isolation_source_config.ontology
    fake_llm_client.respond_with_facets(source_type="animal host")
    _classify_fixture_record(
        fixture_isolation_source_config,
        tmp_path,
        values="clinical isolate 739105",
        fake=fake_llm_client,
        monkeypatch=monkeypatch,
    )
    schema = fake_llm_client.calls[0]["response_model"].model_json_schema()

    for facet_key in ontology.facets:
        items = schema["properties"][facet_key]["items"]
        assert set(items.get("enum") or [items["const"]]) == {
            term.label for term in ontology.terms.values() if term.facet == facet_key
        }


@pytest.mark.parametrize(
    ("sampled_object", "expected_terms"),
    [
        (
            [],
            (SelectedTerm(HEALTHCARE_FACILITY, "facility", "healthcare facility"),),
        ),
        (
            ["sink"],
            (
                SelectedTerm(ENVIRONMENTAL, "source_type", "environmental"),
                SelectedTerm(HEALTHCARE_FACILITY, "facility", "healthcare facility"),
                SelectedTerm(INANIMATE_OBJECT, "sampled_object", "inanimate object"),
                SelectedTerm(SINK, "sampled_object", "sink"),
            ),
        ),
    ],
    ids=["facility alone", "facility with a sampled object"],
)
def test_only_the_sampled_entity_determines_the_broad_source_kind(
    fixture_isolation_source_config: IsolationSourceConfig,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    sampled_object: list[str],
    expected_terms: tuple[SelectedTerm, ...],
    fake_llm_client,
) -> None:
    """The facility is the collection setting, so it implies no broad source kind.

    A hospital holds both patients and built surfaces. Only the named material,
    object, food, or host tells us which one was sampled.
    """
    fake_llm_client.respond_with_facets(
        facility=["healthcare facility"],
        sampled_object=sampled_object,
        reasoning="The value names the hospital.",
    )

    result = _classify_fixture_record(
        fixture_isolation_source_config,
        tmp_path,
        values="Bundang Hospital ward 4821",
        fake=fake_llm_client,
        monkeypatch=monkeypatch,
    )

    assert result.selected_terms == expected_terms


def test_empty_model_result_remains_a_typed_outcome(
    fixture_isolation_source_config: IsolationSourceConfig,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fake_llm_client,
) -> None:
    fake_llm_client.respond_with([], reasoning="No specific source is supported.")

    result = _classify_fixture_record(
        fixture_isolation_source_config,
        tmp_path,
        values="GENOMIC",
        fake=fake_llm_client,
        monkeypatch=monkeypatch,
    )

    assert result.selected_terms == ()
    assert result.evidence_level is IsolationSourceEvidenceLevel.NONE
    assert result.diagnostics == (
        IsolationSourceDiagnostic.LLM_CALL,
        IsolationSourceDiagnostic.UNSPECIFIED,
    )

    cached = _classify_fixture_record(
        fixture_isolation_source_config,
        tmp_path,
        values="GENOMIC",
        fake=fake_llm_client,
        monkeypatch=monkeypatch,
    )

    assert cached.selected_terms == ()
    assert cached.evidence_level is IsolationSourceEvidenceLevel.NONE
    assert cached.diagnostics == (
        IsolationSourceDiagnostic.CACHE_HIT,
        IsolationSourceDiagnostic.UNSPECIFIED,
    )
    assert len(fake_llm_client.calls) == 1


def test_partial_deterministic_result_uses_sample_evidence_when_llm_is_disabled(
    tmp_path: Path,
) -> None:
    config_path = _isolation_source_config(tmp_path)
    standardizer = IsolationSourceStandardizer(
        IsolationSourceConfig.load(config_path),
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
        SelectedTerm(FECES, "body_product", "feces"),
    )
    assert result.evidence_level is IsolationSourceEvidenceLevel.SAMPLE


def test_relation_change_misses_the_cache(tmp_path: Path, fake_llm_client) -> None:
    ontology_directory = tmp_path / "ontology"
    shutil.copytree(FIXTURE_ROOT / "ontology", ontology_directory)
    config_path = _isolation_source_config(
        tmp_path,
        overrides={"ontology_directory": ontology_directory.as_posix()},
    )
    fake_llm_client.respond_with_facets(
        lesion=["abscess"],
        reasoning="The submitted material is from an abscess.",
    )

    first = _standardize_model_isolation_source(config_path, fake_llm_client)
    relations_path = ontology_directory / "relations.tsv"
    relations_path.write_text(
        relations_path.read_text(encoding="utf-8").replace(
            "BACC:0000063\thas_part\tBACC:0000017",
            "BACC:0000063\thas_part\tBACC:0000010",
        ),
        encoding="utf-8",
    )
    second = _standardize_model_isolation_source(config_path, fake_llm_client)

    assert SelectedTerm(PUS, "body_product", "pus") in first.selected_terms
    assert SelectedTerm(BLOOD, "body_product", "blood") in second.selected_terms
    assert SelectedTerm(PUS, "body_product", "pus") not in second.selected_terms
    assert second.diagnostics == (IsolationSourceDiagnostic.LLM_CALL,)


def test_response_that_failed_validation_is_not_cached(tmp_path: Path, fake_llm_client) -> None:
    fake_llm_client.behavior = InstructorRetryException(
        "invalid response",
        n_attempts=3,
        total_usage=0,
    )
    standardizer = IsolationSourceStandardizer(
        IsolationSourceConfig.load(_isolation_source_config(tmp_path)),
        client=None,
        llm_settings=LLMSettings(None, None, "test-model"),
    )
    standardizer.pipeline.client = fake_llm_client
    record = {
        "accession": "INVALID_THEN_VALID",
        "iso_attr_orig": "isolation_source",
        "iso_val_orig": "ambiguous material 6248",
    }
    try:
        failed = standardizer.standardize(record)
        fake_llm_client.respond_with_facets(
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
    fixture_isolation_source_config: IsolationSourceConfig,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fake_llm_client,
) -> None:
    invalid_answer = fake_llm_client.answer(
        reasoning="The sample names an unsupported anatomical site.",
        body_site=["nasal cavity", "nasal cavity"],
    )
    valid_answer = {**invalid_answer, "body_site": ["liver"]}

    def respond(response_model: type[BaseModel]) -> object:
        with pytest.raises(ValidationError):
            response_model.model_validate(invalid_answer)
        return response_model.model_validate(valid_answer)

    fake_llm_client.behavior = respond
    result = _standardize_fixture_record(
        fixture_isolation_source_config,
        tmp_path,
        values="nasal aspirate 78143",
        fake=fake_llm_client,
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
    fixture_isolation_source_config: IsolationSourceConfig,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fake_llm_client,
) -> None:
    invalid_answer = fake_llm_client.answer(
        reasoning="The sample names an unsupported anatomical site.",
        body_site=["nasal cavity"],
    )

    def respond(response_model: type[BaseModel]) -> object:
        for _ in range(3):
            with pytest.raises(ValidationError):
                response_model.model_validate(invalid_answer)
        raise InstructorRetryException("invalid response", n_attempts=3, total_usage=0)

    fake_llm_client.behavior = respond
    result = _standardize_fixture_record(
        fixture_isolation_source_config,
        tmp_path,
        values="nasal aspirate 61512",
        fake=fake_llm_client,
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


def test_an_ancestor_beside_its_descendant_costs_no_extra_model_call(
    fixture_isolation_source_config: IsolationSourceConfig,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fake_llm_client,
) -> None:
    fake_llm_client.respond_with_facets(
        reasoning="The response selects an ancestor and descendant.",
        body_site=["digestive tract", "rectum"],
    )
    result = _standardize_fixture_record(
        fixture_isolation_source_config,
        tmp_path,
        values="rectal sample 49115",
        fake=fake_llm_client,
        monkeypatch=monkeypatch,
        accession="ANCESTOR_WITH_DESCENDANT",
    )

    assert isinstance(result, IsolationSourceOutcome)
    assert len(fake_llm_client.calls) == 1
    assert result.ontology_gap_diagnostics == ()
    labels = {term.label for term in result.selected_terms}
    assert {"digestive tract", "rectum"} <= labels


def test_direct_matches_receive_derived_source_kind_ancestors_and_preorder(
    tmp_path: Path,
) -> None:
    standardizer = IsolationSourceStandardizer(
        IsolationSourceConfig.load(_isolation_source_config(tmp_path)),
        client=None,
    )
    try:
        outcome = standardizer.standardize(
            {
                "accession": "DIRECT_TERM_DERIVATION",
                "iso_attr_orig": "specimen||isolation_source||anatomical_site||lesion",
                "iso_val_orig": "blood||stool||rectal swab||abscess",
            },
        )
    finally:
        standardizer.close()

    assert isinstance(outcome, IsolationSourceOutcome)
    assert outcome.selected_terms == (
        SelectedTerm(HOST_ASSOCIATED, "source_type", "host-associated"),
        SelectedTerm(BODY_FLUID, "body_product", "body fluid"),
        SelectedTerm(BLOOD, "body_product", "blood"),
        SelectedTerm(PUS, "body_product", "pus"),
        SelectedTerm(FECES, "body_product", "feces"),
        SelectedTerm(DIGESTIVE_TRACT, "body_site", "digestive tract"),
        SelectedTerm(INTESTINE, "body_site", "intestine"),
        SelectedTerm(RECTUM, "body_site", "rectum"),
        SelectedTerm(ABSCESS, "lesion", "abscess"),
    )


def test_relation_derives_source_kind_without_classifier_disagreement(
    tmp_path: Path,
    fake_llm_client,
) -> None:
    fake_llm_client.respond_with_facets(
        environmental_material=["rhizosphere soil"],
    )
    standardizer = IsolationSourceStandardizer(
        IsolationSourceConfig.load(_isolation_source_config(tmp_path)),
        client=None,
        llm_settings=LLMSettings(None, None, "test-model"),
    )
    standardizer.pipeline.client = fake_llm_client
    try:
        outcome = standardizer.standardize(
            {
                "accession": "RELATION_OVERRIDE",
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
        SelectedTerm(ENVIRONMENTAL, "source_type", "environmental"),
        SelectedTerm(SOIL, "environmental_material", "soil"),
        SelectedTerm(RHIZOSPHERE, "environmental_material", "rhizosphere soil"),
    )
    assert outcome.diagnostics == (IsolationSourceDiagnostic.LLM_CALL,)


def test_direct_relation_conflict_does_not_report_classifier_disagreement(
    tmp_path: Path,
) -> None:
    standardizer = IsolationSourceStandardizer(
        IsolationSourceConfig.load(_isolation_source_config(tmp_path)),
        client=None,
    )
    try:
        outcome = standardizer.standardize(
            {
                "accession": "DIRECT_RELATION_CONFLICT",
                "iso_attr_orig": "environment||isolation_source",
                "iso_val_orig": "environment||rhizosphere",
            },
        )
    finally:
        standardizer.close()

    assert isinstance(outcome, IsolationSourceOutcome)
    assert outcome.diagnostics == (IsolationSourceDiagnostic.EXACT_MATCH,)


def test_derived_source_kinds_preserve_supported_contexts(tmp_path: Path, fake_llm_client) -> None:
    fake_llm_client.respond_with_facets(
        body_product=["blood"],
        environmental_material=["soil"],
    )
    standardizer = IsolationSourceStandardizer(
        IsolationSourceConfig.load(_isolation_source_config(tmp_path)),
        client=None,
        llm_settings=LLMSettings(None, None, "test-model"),
    )
    standardizer.pipeline.client = fake_llm_client
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
        SelectedTerm(ENVIRONMENTAL, "source_type", "environmental"),
        SelectedTerm(BODY_FLUID, "body_product", "body fluid"),
        SelectedTerm(BLOOD, "body_product", "blood"),
        SelectedTerm(SOIL, "environmental_material", "soil"),
    )
    assert outcome.diagnostics == (IsolationSourceDiagnostic.LLM_CALL,)


@pytest.mark.parametrize(
    ("source_type", "expected_terms"),
    [
        (
            "host-associated",
            (SelectedTerm(HOST_ASSOCIATED, "source_type", "host-associated"),),
        ),
        (
            "laboratory strain",
            (SelectedTerm(LABORATORY, "source_type", "laboratory strain"),),
        ),
        (
            "environmental",
            (SelectedTerm(ENVIRONMENTAL, "source_type", "environmental"),),
        ),
    ],
)
def test_classifier_source_type_stands_when_no_narrower_facet_applies(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    source_type: str,
    expected_terms: tuple[SelectedTerm, ...],
    fake_llm_client,
) -> None:
    config = replace(
        IsolationSourceConfig.load(CONFIG_PATH),
        cache_db_path=tmp_path / "source-type-only-cache.db",
    )
    fake_llm_client.respond_with_facets(source_type=source_type)

    outcome = _classify_fixture_record(
        config,
        tmp_path,
        values=f"source type 7291 {source_type}",
        fake=fake_llm_client,
        monkeypatch=monkeypatch,
    )

    assert outcome.selected_terms == expected_terms
    assert outcome.diagnostics == (IsolationSourceDiagnostic.LLM_CALL,)


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
            (SelectedTerm(LABORATORY, "source_type", "laboratory strain"),),
        ),
    ],
)
def test_source_kind_stands_when_no_other_facet_implies_one(
    tmp_path: Path,
    record: dict[str, str],
    answer: str | None,
    expected_terms: tuple[SelectedTerm, ...],
    fake_llm_client,
) -> None:
    if answer is not None:
        fake_llm_client.respond_with([answer])
    standardizer = IsolationSourceStandardizer(
        IsolationSourceConfig.load(_isolation_source_config(tmp_path)),
        client=None,
        llm_settings=LLMSettings(None, None, "test-model"),
    )
    standardizer.pipeline.client = fake_llm_client
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
    fixture_config = yaml.safe_load(
        (FIXTURE_ROOT / "isolation_source.yaml").read_text(encoding="utf-8")
    )
    config["ontology_directory"] = (FIXTURE_ROOT / "ontology").as_posix()
    config["host_recovery_term_ids"] = fixture_config["host_recovery_term_ids"]
    config["direct_match_values"] = fixture_config["direct_match_values"]
    config.pop("disease_body_terms")
    config.update(overrides or {})
    config["cache_db_path"] = (tmp_path / "isolation-cache.db").as_posix()
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return config_path


def _standardize_model_isolation_source(
    config_path: Path,
    fake,
    *,
    model: str = "test-model",
    value: str = "wound patient 739105",
) -> IsolationSourceOutcome:
    standardizer = IsolationSourceStandardizer(
        IsolationSourceConfig.load(config_path),
        client=None,
        llm_settings=LLMSettings(None, None, model),
    )
    standardizer.pipeline.client = fake
    try:
        result = standardizer.standardize(
            {
                "accession": "REQUEST_FINGERPRINT",
                "iso_attr_orig": "isolation_source",
                "iso_val_orig": value,
            },
        )
    finally:
        standardizer.close()
    assert isinstance(result, IsolationSourceOutcome)
    return result


def test_isolation_source_cache_reuses_identical_request_when_only_prompt_metadata_changes(
    tmp_path,
    fake_llm_client,
):
    fake_llm_client.respond_with(["wound"], reasoning="clinical wound")
    first_config = _isolation_source_config(tmp_path, overrides={"prompt_version": "first"})
    second_config = _isolation_source_config(tmp_path, overrides={"prompt_version": "second"})

    first = _standardize_model_isolation_source(first_config, fake_llm_client)
    second = _standardize_model_isolation_source(second_config, fake_llm_client)

    assert first.diagnostics == (IsolationSourceDiagnostic.LLM_CALL,)
    assert second.diagnostics == (IsolationSourceDiagnostic.CACHE_HIT,)
    assert second.reasoning == first.reasoning
    assert len(fake_llm_client.calls) == 1


def test_isolation_source_request_uses_rendered_synthetic_prompts(
    tmp_path: Path, fake_llm_client
) -> None:
    fake_llm_client.respond_with(["wound"], reasoning="clinical wound")
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
        IsolationSourceConfig.load(config_path),
        client=None,
        llm_settings=LLMSettings(None, None, "test-model"),
    )
    standardizer.pipeline.client = fake_llm_client
    try:
        direct = standardizer.standardize(
            {
                "accession": "DIRECT",
                "iso_attr_orig": "isolation_source",
                "iso_val_orig": "stool",
            },
        )
        assert fake_llm_client.calls == []
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
    assert fake_llm_client.calls[0]["messages"][0]["role"] == "system"
    system_message = fake_llm_client.calls[0]["messages"][0]["content"]
    user_message = fake_llm_client.calls[0]["messages"][1]
    assert "Synthetic ontology:" in system_message
    assert "wound" in system_message
    assert user_message["role"] == "user"
    assert '"isolation_source" = "wound patient 739105"' in user_message["content"]
    assert "Homo sapiens" not in str(fake_llm_client.calls[0]["messages"])
    assert fake_llm_client.calls[0]["model"] == "test-model"
    assert fake_llm_client.calls[0]["temperature"] == 0
    assert fake_llm_client.calls[0]["seed"] == 100
    assert fake_llm_client.calls[0]["response_model"].__name__ == "IsolationSourceClassification"
    assert outcome.request_fingerprint


@pytest.mark.parametrize(
    "changed_component",
    [
        "message",
        "model",
        "parameter",
        "schema_id",
        "structured_output_mode",
    ],
)
def test_isolation_source_cache_misses_when_canonical_request_changes(
    tmp_path, monkeypatch, changed_component, fake_llm_client
):
    fake_llm_client.respond_with(["wound"], reasoning="clinical wound")
    first_config = _isolation_source_config(tmp_path)
    _standardize_model_isolation_source(first_config, fake_llm_client)

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
        fake_llm_client,
        model=second_model,
    )

    assert second.diagnostics == (IsolationSourceDiagnostic.LLM_CALL,)
    assert len(fake_llm_client.calls) == 2


def test_ontology_tree_shows_the_facet_guidance_under_its_heading(tmp_path):
    config = _isolation_source_config(tmp_path)
    system_prompt = IsolationSourceConfig.load(config).effective_prompts.system

    guidance = "Use the sampled food or feed, including feed given to a host."
    assert re.search(rf"## food_type\n[^\n]+ {re.escape(guidance)}\n- ", system_prompt)


def test_underscores_become_spaces_only_in_the_model_request(tmp_path, fake_llm_client):
    fake_llm_client.respond_with(["wound"])
    config = _isolation_source_config(tmp_path)
    outcome = _standardize_model_isolation_source(
        config, fake_llm_client, value="wound_swab_739105"
    )

    assert (
        '"isolation_source" = "wound swab 739105"'
        in fake_llm_client.calls[0]["messages"][1]["content"]
    )
    assert outcome.supporting_pairs[0].value == "wound_swab_739105"


def test_model_request_shows_the_host_as_context_only(tmp_path, fake_llm_client):
    fake_llm_client.respond_with(["wound"])
    standardizer = IsolationSourceStandardizer(
        IsolationSourceConfig.load(_isolation_source_config(tmp_path)),
        client=None,
        llm_settings=LLMSettings(None, None, "test-model"),
    )
    standardizer.pipeline.client = fake_llm_client
    try:
        outcome = standardizer.standardize(
            {"accession": "A", "iso_attr_orig": "isolation_source", "iso_val_orig": "carcass swab"},
            host_pairs=(SupportingAttributeValuePair("host", "feces"),),
        )
    finally:
        standardizer.close()

    assert '"host" = "feces"' in fake_llm_client.calls[0]["messages"][1]["content"]
    assert FECES not in {term.term_id for term in outcome.selected_terms}
    assert [pair.value for pair in outcome.supporting_pairs] == ["carcass swab"]


@pytest.mark.parametrize(
    ("value", "disease_id"),
    [
        ("rectal swab||urinary tract infection", URINARY_TRACT_INFECTION),
        ("rectal swab||cystic fibrosis", CYSTIC_FIBROSIS),
    ],
)
def test_a_diagnosis_resolves_next_to_a_direct_match(tmp_path, value, disease_id):
    standardizer = IsolationSourceStandardizer(
        IsolationSourceConfig.load(_isolation_source_config(tmp_path)), client=None
    )
    try:
        outcome = standardizer.standardize(
            {
                "accession": "A",
                "iso_attr_orig": "isolation_source||host_disease",
                "iso_val_orig": value,
            }
        )
    finally:
        standardizer.close()

    assert outcome.llm_calls == 0
    assert {RECTUM, disease_id, DISEASE, HOST_ASSOCIATED} <= {
        term.term_id for term in outcome.selected_terms
    }


def test_food_excludes_host_terms(tmp_path, fake_llm_client):
    fake_llm_client.respond_with_facets(
        source_type=["animal host"],
        body_product=["blood"],
        body_site=["rectum"],
        food_type=["meat product"],
        host_disease=["cystic fibrosis"],
    )
    outcome = _standardize_model_isolation_source(
        _isolation_source_config(tmp_path), fake_llm_client
    )

    selected_ids = {term.term_id for term in outcome.selected_terms}
    assert selected_ids == {FOOD_OR_FEED, ANIMAL_FOOD_PRODUCT, MEAT_PRODUCT}
    assert "food_host_removal" in {step.node for step in outcome.reasoning}


def _body_labels(outcome: IsolationSourceOutcome) -> set[str]:
    return {
        term.label for term in outcome.selected_terms if term.facet in {"body_product", "body_site"}
    }


@pytest.mark.parametrize(
    ("host_disease", "expected_body_labels"),
    [
        ("mastitis", {"mammary gland"}),
        ("bacteremia", {"body fluid", "blood", "circulatory system"}),
        ("bone and joint infection", {"skeletal system"}),
        ("colibacillosis", set()),
    ],
)
def test_a_diagnosis_alone_gets_its_body_term(
    tmp_path, host_disease, expected_body_labels, fake_llm_client
):
    fake_llm_client.respond_with_facets(source_type=["animal host"], host_disease=[host_disease])
    config = _isolation_source_config(tmp_path, overrides=PRODUCTION_VOCABULARY)
    outcome = _standardize_model_isolation_source(config, fake_llm_client)

    assert _body_labels(outcome) == expected_body_labels
    assert ("disease_body_term" in {step.node for step in outcome.reasoning}) is bool(
        expected_body_labels
    )


@pytest.mark.parametrize(
    ("value", "expected_body_labels"),
    [
        ("bacteremia", {"body fluid", "blood", "circulatory system"}),
        ("cystitis", {"genitourinary system", "urinary system"}),
        ("wound swab||SSTI", set()),
    ],
)
def test_a_direct_diagnosis_gets_its_body_term_without_the_model(
    tmp_path, value, expected_body_labels
):
    standardizer = IsolationSourceStandardizer(
        IsolationSourceConfig.load(
            _isolation_source_config(tmp_path, overrides=PRODUCTION_VOCABULARY)
        ),
        client=None,
    )
    try:
        outcome = standardizer.standardize(
            {
                "accession": "A",
                "iso_attr_orig": "||".join(["isolation_source"] * (value.count("||") + 1)),
                "iso_val_orig": value,
            }
        )
    finally:
        standardizer.close()

    assert outcome.llm_calls == 0
    assert _body_labels(outcome) == expected_body_labels


@pytest.mark.parametrize(
    "sampled",
    [{"lesion": ["abscess"]}, {"sampled_object": ["sink"]}, {"body_site": ["skin"]}],
)
def test_a_diagnosis_next_to_a_sampled_part_gets_no_body_term(tmp_path, sampled, fake_llm_client):
    fake_llm_client.respond_with_facets(host_disease=["pneumonia"], **sampled)
    config = _isolation_source_config(tmp_path, overrides=PRODUCTION_VOCABULARY)
    outcome = _standardize_model_isolation_source(config, fake_llm_client)

    assert "lung" not in _body_labels(outcome)
    assert "disease_body_term" not in {step.node for step in outcome.reasoning}


@pytest.mark.parametrize(
    ("body_product", "body_sites", "host_disease", "removed", "kept"),
    [
        ("sputum", ["lung", "nasal cavity"], "pneumonia", "lung", "nasal cavity"),
        (
            "pus",
            ["lower respiratory tract", "skin"],
            "pneumonia",
            "lower respiratory tract",
            "skin",
        ),
        ("urine", ["kidney", "rectum"], "urinary tract infection", "kidney", "rectum"),
    ],
)
def test_a_diagnosis_next_to_a_body_product_loses_the_model_body_sites_on_its_lineage(
    tmp_path, body_product, body_sites, host_disease, removed, kept, fake_llm_client
):
    fake_llm_client.respond_with_facets(
        body_product=[body_product], body_site=body_sites, host_disease=[host_disease]
    )
    config = _isolation_source_config(tmp_path, overrides=PRODUCTION_VOCABULARY)
    outcome = _standardize_model_isolation_source(config, fake_llm_client)

    assert removed not in _body_labels(outcome)
    assert kept in _body_labels(outcome)
    assert "disease_body_site_removal" in {step.node for step in outcome.reasoning}


def test_a_diagnosis_next_to_a_body_product_keeps_a_body_site_from_a_direct_match(
    tmp_path, fake_llm_client
):
    fake_llm_client.respond_with_facets(
        body_product=["sputum"], body_site=["lung"], host_disease=["pneumonia"]
    )
    standardizer = IsolationSourceStandardizer(
        IsolationSourceConfig.load(
            _isolation_source_config(tmp_path, overrides=PRODUCTION_VOCABULARY)
        ),
        client=None,
        llm_settings=LLMSettings(None, None, "test-model"),
    )
    standardizer.pipeline.client = fake_llm_client
    try:
        outcome = standardizer.standardize(
            {
                "accession": "A",
                "iso_attr_orig": "isolation_source||host_disease",
                "iso_val_orig": "lung||sputum of a pneumonia patient",
            }
        )
    finally:
        standardizer.close()

    assert outcome.llm_calls == 1
    assert "lung" in _body_labels(outcome)


@pytest.mark.parametrize(
    "disease_body_terms",
    [{RECTUM: BLOOD}, {URINARY_TRACT_INFECTION: DISEASE}, {URINARY_TRACT_INFECTION: [BLOOD]}],
)
def test_prompt_config_rejects_a_wrong_disease_body_term(tmp_path, disease_body_terms):
    with pytest.raises(ConfigError, match=r"disease_body_terms"):
        IsolationSourceConfig.load(
            _isolation_source_config(tmp_path, overrides={"disease_body_terms": disease_body_terms})
        )


@pytest.mark.parametrize(
    ("value", "replaced"),
    [
        ("blood or pus from rectum or liver", True),
        ("blood/pus from rectum/liver", True),
        ("pooled blood and pus from rectum and liver", False),
        ("blood and pus, Veterinary Facility or Diagnostic Laboratory", False),
    ],
)
def test_named_alternatives_get_their_common_ancestor(tmp_path, value, replaced, fake_llm_client):
    fake_llm_client.respond_with_facets(
        source_type=["animal host", "human host"],
        body_product=["blood", "pus"],
        body_site=["rectum", "liver"],
    )
    config = _isolation_source_config(tmp_path)
    outcome = _standardize_model_isolation_source(config, fake_llm_client, value=value)

    selected_ids = {term.term_id for term in outcome.selected_terms}
    common_ids = {HOST_ASSOCIATED, BODY_FLUID, DIGESTIVE_TRACT, INTESTINE, RECTUM, LIVER}
    if replaced:
        assert selected_ids == common_ids
        assert "alternatives_common_ancestor" in {step.node for step in outcome.reasoning}
    else:
        assert selected_ids == common_ids | {ANIMAL_HOST, HUMAN_HOST, BLOOD, PUS}


# --- Cache persistence adapter ---


def test_cache_round_trip_preserves_only_the_classifier_answer(cache):
    answer = IsolationSourceClassifierAnswer(
        facet_values={
            "source_type": (),
            "body_product": ("feces",),
            "body_site": (),
            "lesion": (),
            "environmental_material": (),
            "facility": (),
            "sampled_object": (),
            "food_type": (),
            "host_disease": (),
        },
        reasoning="The sample is fecal material from an animal host.",
    )
    cache.set("fingerprint", answer)

    got = cache.get("fingerprint")

    assert got == answer


@pytest.mark.parametrize(
    ("first_value", "second_value"),
    [("patient 1", "patient 2"), ("Patient 1", "PATIENT   1")],
)
def test_records_differing_only_in_digits_case_or_spaces_share_one_model_call(
    tmp_path: Path, first_value: str, second_value: str, fake_llm_client
) -> None:
    fake_llm_client.respond_with(["wound"], reasoning="clinical wound")
    standardizer = IsolationSourceStandardizer(
        IsolationSourceConfig.load(_isolation_source_config(tmp_path)),
        client=None,
        llm_settings=LLMSettings(None, None, "test-model"),
    )
    standardizer.pipeline.client = fake_llm_client
    try:
        first = standardizer.standardize(
            {
                "accession": "SAMN00000001",
                "iso_attr_orig": "isolation_source",
                "iso_val_orig": first_value,
            },
        )
        second = standardizer.standardize(
            {
                "accession": "SAMN00000002",
                "iso_attr_orig": "isolation_source",
                "iso_val_orig": second_value,
            },
        )
    finally:
        standardizer.close()

    assert isinstance(first, IsolationSourceOutcome)
    assert isinstance(second, IsolationSourceOutcome)
    assert first.diagnostics == (IsolationSourceDiagnostic.LLM_CALL,)
    assert second.diagnostics == (IsolationSourceDiagnostic.CACHE_HIT,)
    assert first.request_fingerprint == second.request_fingerprint
    assert len(fake_llm_client.calls) == 1
    assert first_value in fake_llm_client.calls[0]["messages"][1]["content"]


def test_strip_keeps_the_digits_of_an_ontology_identifier() -> None:
    fold = isolation_source_module._strip_digits
    assert fold("Wound,  patient 12") == "wound, patient <NUM>"
    assert fold("wound [ENVO:00002007]") == "wound [envo:00002007]"
