"""Public contract tests for selective effective-policy loading."""

import shutil
from pathlib import Path

import pytest
import yaml

from baccurate.adapters.policy_yaml import PolicyConfigurationError
from baccurate.pathogen_registry.registry import load_pathogen_registry
from baccurate.paths import (
    DEFAULT_GEO_LOC_LIST,
    DEFAULT_ISOLATION_SOURCE_CACHE_DB,
    DEFAULT_LOC_CACHE_DB,
    DEFAULT_ONTOLOGY_TSV,
)
from baccurate.run.effective_policy import load_effective_policy
from baccurate.standardization.isolation_source import IsolationSourcePromptPolicy
from baccurate.standardization.location import LocationPolicy, LocationStandardizer

ROOT = Path(__file__).parents[2]
CONFIG_DIR = ROOT / "config"
PATHOGEN_REGISTRY_PATH = CONFIG_DIR / "pathogens.yaml"
STANDARDIZATION_TARGETS = ("host", "date", "loc", "iso")


def test_complete_production_effective_policy_has_required_semantics() -> None:
    registry = load_pathogen_registry(PATHOGEN_REGISTRY_PATH)

    policy = load_effective_policy(
        pathogen_registry=registry,
        configuration_root=CONFIG_DIR,
        requested_standardization_targets=STANDARDIZATION_TARGETS,
        extraction_required=True,
    )

    assert policy.pathogen_registry.schema_version == 1
    assert policy.curation_schema is not None
    assert policy.curation_schema.schema_version == 3
    assert tuple(
        match.target
        for match in policy.curation_schema.evaluate(attribute="host", value="human").matches
    ) == ("host",)
    assert tuple(
        (match.target, match.category)
        for match in policy.curation_schema.evaluate(
            attribute="collection_date", value="2020"
        ).matches
    ) == (("date", "c"),)
    missing_host = policy.curation_schema.evaluate(attribute="host", value="unknown")
    assert missing_host.matches == ()
    assert [(event.kind, event.target, event.family) for event in missing_host.events] == [
        ("rejected_value", "host", "universal_missing")
    ]
    uncertain_host = policy.curation_schema.evaluate(attribute="host", value="unkmowm")
    assert tuple(match.target for match in uncertain_host.matches) == ("host",)
    assert [(event.kind, event.target, event.family) for event in uncertain_host.events] == [
        ("uncertain_rejection", "host", "universal_missing")
    ]
    unreviewed = policy.curation_schema.evaluate(attribute="host_environment", value="human")
    assert unreviewed.matches == ()
    assert {(event.kind, event.target, event.family) for event in unreviewed.events} == {
        ("unreviewed_attribute", "host", "host_fields"),
        ("unreviewed_attribute", "iso", "environmental_origin"),
    }
    assert tuple(
        match.target
        for match in policy.curation_schema.evaluate(
            attribute="country submitted", value="Germany: Berlin"
        ).matches
    ) == ("loc",)
    assert tuple(
        (match.target, match.category)
        for match in policy.curation_schema.evaluate(
            attribute="upload_date", value="2021-02-03"
        ).matches
    ) == (("date", "f"),)
    rejected_isolation_source = policy.curation_schema.evaluate(
        attribute="isolation_source", value="GENOMIC"
    )
    assert rejected_isolation_source.matches == ()
    assert [
        (event.kind, event.target, event.family) for event in rejected_isolation_source.events
    ] == [("rejected_value", "iso", "non_discriminative_process")]
    assert policy.host_policy is not None
    assert policy.host_policy.schema_version == 3
    assert "colon" in policy.host_policy.value_rejections
    assert "Escherichia coli" in policy.host_policy.value_rejections
    assert "Klebsiella variicola" not in policy.host_policy.value_rejections
    assert "food" in policy.host_policy.isolation_source_keywords
    assert policy.location_policy is not None
    assert policy.location_policy.schema_version == 1
    assert policy.location_policy.prompt_version == "1"
    assert "lat_lon" in policy.location_policy.coordinate_attributes
    assert "{attr_val_pairs}" in policy.location_policy.prompts.user_template
    assert policy.location_policy.insdc_country_map["United States"] == "USA"
    assert policy.isolation_source_prompt_policy is not None
    assert policy.isolation_source_prompt_policy.schema_version == 1
    assert policy.isolation_source_prompt_policy.prompt_version == "1"
    assert "{ontology_tree}" in policy.isolation_source_prompt_policy.prompts.sample_system_template
    assert "{metadata}" in policy.isolation_source_prompt_policy.prompts.sample_user_template
    assert "unspecified" in policy.isolation_source_prompt_policy.effective_prompts.system
    assert policy.location_policy.geo_loc_list_path == DEFAULT_GEO_LOC_LIST
    assert policy.location_policy.geo_loc_list_path.is_file()
    assert policy.location_policy.cache_db_path == DEFAULT_LOC_CACHE_DB
    assert policy.isolation_source_prompt_policy.ontology_tsv_path == DEFAULT_ONTOLOGY_TSV
    assert policy.isolation_source_prompt_policy.ontology_tsv_path.is_file()
    assert policy.isolation_source_prompt_policy.cache_db_path == DEFAULT_ISOLATION_SOURCE_CACHE_DB


def test_effective_policy_loads_selected_policies_and_ignores_unselected_policy(
    tmp_path: Path,
) -> None:
    registry = load_pathogen_registry(PATHOGEN_REGISTRY_PATH)
    test_root = tmp_path / "selective-loading"
    configuration_root = test_root / "config"
    configuration_root.mkdir(parents=True)
    reference_root = test_root / "data" / "reference"
    reference_root.mkdir(parents=True)
    shutil.copyfile(
        ROOT / "data" / "reference" / "geo_loc_list.txt",
        reference_root / "geo_loc_list.txt",
    )
    selected_sources = {"host.yaml", "isolation_source.yaml"}
    for source_name in (
        "curation_schema.yaml",
        "host.yaml",
        "location.yaml",
        "isolation_source.yaml",
    ):
        destination = configuration_root / source_name
        if source_name in selected_sources:
            shutil.copyfile(CONFIG_DIR / source_name, destination)
        else:
            destination.write_text("malformed: [\n", encoding="utf-8")

    policy = load_effective_policy(
        pathogen_registry=registry,
        configuration_root=configuration_root,
        requested_standardization_targets=("iso",),
        extraction_required=False,
    )

    assert policy.curation_schema is None
    assert policy.host_policy is not None
    assert policy.location_policy is None
    assert policy.isolation_source_prompt_policy is not None


def test_effective_policy_loads_required_curation_schema() -> None:
    registry = load_pathogen_registry(PATHOGEN_REGISTRY_PATH)

    policy = load_effective_policy(
        pathogen_registry=registry,
        configuration_root=CONFIG_DIR,
        requested_standardization_targets=("date",),
        extraction_required=True,
    )

    assert policy.pathogen_registry is registry
    assert policy.curation_schema is not None
    assert policy.curation_schema.schema_version == 3
    with pytest.raises(AttributeError):
        policy.curation_schema = None  # type: ignore[misc]


def test_effective_policy_does_not_load_curation_schema_for_reused_bundle(
    tmp_path: Path,
) -> None:
    registry = load_pathogen_registry(PATHOGEN_REGISTRY_PATH)
    (tmp_path / "curation_schema.yaml").write_text("broken: [\n", encoding="utf-8")

    policy = load_effective_policy(
        pathogen_registry=registry,
        configuration_root=tmp_path,
        requested_standardization_targets=("date",),
        extraction_required=False,
    )

    assert policy.pathogen_registry is registry
    assert policy.curation_schema is None


def test_registry_scientific_name_change_updates_derived_host_rejection(
    tmp_path: Path,
) -> None:
    host_policy = {
        "schema_version": 3,
        "normalization": {"ignored_substrings": []},
        "routing": {"isolation_source_keywords": []},
        "curated_taxa": {},
        "value_rejections": {
            "exact": ["literal rejection", {"pathogen_key": "target"}],
        },
    }
    (tmp_path / "host.yaml").write_text(
        yaml.safe_dump(host_policy, sort_keys=False),
        encoding="utf-8",
    )
    first_registry_path = tmp_path / "first-pathogens.yaml"
    second_registry_path = tmp_path / "second-pathogens.yaml"
    registry_policy = {
        "schema_version": 1,
        "target": {
            "scientific_name": "Original target name",
            "ncbi_taxid": 1,
            "rank": "species",
        },
    }
    first_registry_path.write_text(
        yaml.safe_dump(registry_policy, sort_keys=False),
        encoding="utf-8",
    )
    registry_policy["target"]["scientific_name"] = "Renamed target"
    second_registry_path.write_text(
        yaml.safe_dump(registry_policy, sort_keys=False),
        encoding="utf-8",
    )

    first = load_effective_policy(
        pathogen_registry=load_pathogen_registry(first_registry_path),
        configuration_root=tmp_path,
        requested_standardization_targets=("host",),
        extraction_required=False,
    ).host_policy
    second = load_effective_policy(
        pathogen_registry=load_pathogen_registry(second_registry_path),
        configuration_root=tmp_path,
        requested_standardization_targets=("host",),
        extraction_required=False,
    ).host_policy

    assert first is not None
    assert second is not None
    assert first.value_rejections == ("literal rejection", "Original target name")
    assert second.value_rejections == ("literal rejection", "Renamed target")


def test_effective_policy_reports_unknown_host_key_with_source_and_dotted_key(
    tmp_path: Path,
) -> None:
    registry = load_pathogen_registry(PATHOGEN_REGISTRY_PATH)
    host_path = tmp_path / "host.yaml"
    host_path.write_text(
        yaml.safe_dump(
            {
                "schema_version": 3,
                "normalization": {
                    "ignored_substrings": [],
                    "unexpected": [],
                },
                "routing": {"isolation_source_keywords": []},
                "curated_taxa": {},
                "value_rejections": {"exact": []},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    with pytest.raises(PolicyConfigurationError) as error:
        load_effective_policy(
            pathogen_registry=registry,
            configuration_root=tmp_path,
            requested_standardization_targets=("host",),
            extraction_required=False,
        )

    assert str(host_path) in str(error.value)
    assert "normalization.unexpected" in str(error.value)


def test_host_policy_rejects_unknown_pathogen_key_reference(tmp_path: Path) -> None:
    registry = load_pathogen_registry(PATHOGEN_REGISTRY_PATH)
    host_path = tmp_path / "host.yaml"
    host_path.write_text(
        yaml.safe_dump(
            {
                "schema_version": 3,
                "normalization": {"ignored_substrings": []},
                "routing": {"isolation_source_keywords": []},
                "curated_taxa": {},
                "value_rejections": {
                    "exact": [{"pathogen_key": "not-a-target"}],
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    with pytest.raises(PolicyConfigurationError) as error:
        load_effective_policy(
            pathogen_registry=registry,
            configuration_root=tmp_path,
            requested_standardization_targets=("host",),
            extraction_required=False,
        )

    assert str(host_path) in str(error.value)
    assert "value_rejections.exact.0.pathogen_key" in str(error.value)
    assert "not-a-target" in str(error.value)


def test_host_policy_rejects_curated_match_overlapping_derived_rejection(
    tmp_path: Path,
) -> None:
    registry = load_pathogen_registry(PATHOGEN_REGISTRY_PATH)
    host_path = tmp_path / "host.yaml"
    host_path.write_text(
        yaml.safe_dump(
            {
                "schema_version": 3,
                "normalization": {"ignored_substrings": []},
                "routing": {"isolation_source_keywords": []},
                "curated_taxa": {
                    "9606": {
                        "scientific_name": "Homo sapiens",
                        "match_terms": {"exact": ["Escherichia coli"]},
                    },
                },
                "value_rejections": {
                    "exact": [{"pathogen_key": "ecoli"}],
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    with pytest.raises(PolicyConfigurationError) as error:
        load_effective_policy(
            pathogen_registry=registry,
            configuration_root=tmp_path,
            requested_standardization_targets=("host",),
            extraction_required=False,
        )

    message = str(error.value)
    assert str(host_path) in message
    assert "value_rejections.exact" in message
    assert "both a curated term and a value rejection" in message


def test_unselected_host_policy_is_not_loaded(tmp_path: Path) -> None:
    registry = load_pathogen_registry(PATHOGEN_REGISTRY_PATH)
    (tmp_path / "host.yaml").write_text("broken: [\n", encoding="utf-8")

    policy = load_effective_policy(
        pathogen_registry=registry,
        configuration_root=tmp_path,
        requested_standardization_targets=("date",),
        extraction_required=False,
    )

    assert policy.host_policy is None


def _write_location_policy(
    tmp_path: Path,
    overrides: dict[str, object] | None = None,
) -> Path:
    policy: dict[str, object] = {
        "schema_version": 1,
        "prompt_version": "test-v1",
        "coordinate_attributes": ["lat_lon", "latitude"],
        "llm_system_prompt": "Resolve a geographic location.",
        "llm_user_prompt_template": "Evidence: {attr_val_pairs}; literal {{brace}}",
        "insdc_country_map": {"United States": "USA"},
        "geo_loc_list_path": "reference/geo_loc_list.txt",
        "cache_db_path": "cache/location.db",
    }
    policy.update(overrides or {})
    path = tmp_path / "location.yaml"
    path.write_text(yaml.safe_dump(policy, sort_keys=False), encoding="utf-8")
    return path


def test_unselected_location_policy_is_not_loaded(tmp_path: Path) -> None:
    registry = load_pathogen_registry(PATHOGEN_REGISTRY_PATH)
    (tmp_path / "location.yaml").write_text("broken: [\n", encoding="utf-8")

    policy = load_effective_policy(
        pathogen_registry=registry,
        configuration_root=tmp_path,
        requested_standardization_targets=("date",),
        extraction_required=False,
    )

    assert policy.location_policy is None


def test_location_policy_preserves_relative_paths_and_escaped_literal_braces(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "reference").mkdir()
    (tmp_path / "reference" / "geo_loc_list.txt").write_text(
        "Germany\n",
        encoding="utf-8",
    )
    (tmp_path / "cache").mkdir()
    location_path = _write_location_policy(tmp_path)

    policy = LocationPolicy.load(location_path)

    assert policy.geo_loc_list_path == Path("reference/geo_loc_list.txt")
    assert policy.cache_db_path == Path("cache/location.db")
    assert policy.prompts.user_template.format(attr_val_pairs="geo_loc_name=Germany") == (
        "Evidence: geo_loc_name=Germany; literal {brace}"
    )


@pytest.mark.parametrize(
    ("overrides", "policy_key"),
    [
        ({"geo_loc_list_path": "missing-reference.txt"}, "geo_loc_list_path"),
        ({"cache_db_path": "missing-parent/location.db"}, "cache_db_path"),
    ],
)
def test_location_policy_rejects_unusable_resource_selection_before_standardization(
    tmp_path: Path,
    overrides: dict[str, object],
    policy_key: str,
) -> None:
    registry = load_pathogen_registry(PATHOGEN_REGISTRY_PATH)
    valid_reference = tmp_path / "geo_loc_list.txt"
    valid_reference.write_text("Germany\n", encoding="utf-8")
    location_path = _write_location_policy(
        tmp_path,
        {"geo_loc_list_path": valid_reference.as_posix(), **overrides},
    )

    with pytest.raises(PolicyConfigurationError) as error:
        load_effective_policy(
            pathogen_registry=registry,
            configuration_root=tmp_path,
            requested_standardization_targets=("loc",),
            extraction_required=False,
        )

    assert str(location_path) in str(error.value)
    assert policy_key in str(error.value)


def test_location_standardizer_does_not_reopen_loaded_policy_source(tmp_path: Path) -> None:
    geo_loc_list = tmp_path / "geo_loc_list.txt"
    geo_loc_list.write_text("Germany\n", encoding="utf-8")
    location_path = _write_location_policy(
        tmp_path,
        {
            "geo_loc_list_path": geo_loc_list.as_posix(),
            "cache_db_path": (tmp_path / "location-cache.db").as_posix(),
        },
    )
    policy = LocationPolicy.load(location_path)
    location_path.write_text("broken: [\n", encoding="utf-8")

    standardizer = LocationStandardizer(policy, client=None)
    try:
        outcome = standardizer.standardize(
            {
                "accession": "NO_REOPEN",
                "loc_attr_orig": "geo_loc_name",
                "loc_val_orig": "Germany",
            }
        )
    finally:
        standardizer.close()

    assert outcome.country == "Germany"


@pytest.mark.parametrize(
    ("overrides", "policy_key"),
    [
        ({"unexpected": True}, "top level.unexpected"),
        ({"schema_version": "1"}, "schema_version"),
        ({"prompt_version": 1}, "prompt_version"),
        ({"coordinate_attributes": "lat_lon"}, "coordinate_attributes"),
        ({"coordinate_attributes": ["lat_lon", "  "]}, "coordinate_attributes.1"),
        ({"llm_system_prompt": None}, "llm_system_prompt"),
        ({"llm_user_prompt_template": "literal only"}, "llm_user_prompt_template"),
        (
            {"llm_user_prompt_template": "{attr_val_pairs} {unsupported}"},
            "llm_user_prompt_template",
        ),
        (
            {"llm_user_prompt_template": "{attr_val_pairs} {attr_val_pairs}"},
            "llm_user_prompt_template",
        ),
        ({"insdc_country_map": []}, "insdc_country_map"),
        ({"insdc_country_map": {"": "USA"}}, "insdc_country_map"),
        ({"insdc_country_map": {"United States": 1}}, "insdc_country_map.United States"),
        ({"geo_loc_list_path": 1}, "geo_loc_list_path"),
        ({"cache_db_path": ""}, "cache_db_path"),
    ],
)
def test_location_policy_rejects_invalid_values_with_source_and_key(
    tmp_path: Path,
    overrides: dict[str, object],
    policy_key: str,
) -> None:
    registry = load_pathogen_registry(PATHOGEN_REGISTRY_PATH)
    location_path = _write_location_policy(tmp_path, overrides)

    with pytest.raises(PolicyConfigurationError) as error:
        load_effective_policy(
            pathogen_registry=registry,
            configuration_root=tmp_path,
            requested_standardization_targets=("loc",),
            extraction_required=False,
        )

    assert str(location_path) in str(error.value)
    assert policy_key in str(error.value)


@pytest.mark.parametrize(
    "missing_key",
    ["schema_version", "llm_system_prompt", "llm_user_prompt_template"],
)
def test_location_policy_rejects_missing_required_values(
    tmp_path: Path,
    missing_key: str,
) -> None:
    registry = load_pathogen_registry(PATHOGEN_REGISTRY_PATH)
    location_path = _write_location_policy(tmp_path)
    policy = yaml.safe_load(location_path.read_text(encoding="utf-8"))
    del policy[missing_key]
    location_path.write_text(yaml.safe_dump(policy, sort_keys=False), encoding="utf-8")

    with pytest.raises(PolicyConfigurationError) as error:
        load_effective_policy(
            pathogen_registry=registry,
            configuration_root=tmp_path,
            requested_standardization_targets=("loc",),
            extraction_required=False,
        )

    assert str(location_path) in str(error.value)
    assert missing_key in str(error.value)


def test_location_policy_rejects_malformed_user_prompt_braces(tmp_path: Path) -> None:
    registry = load_pathogen_registry(PATHOGEN_REGISTRY_PATH)
    location_path = _write_location_policy(
        tmp_path,
        {"llm_user_prompt_template": "{attr_val_pairs"},
    )

    with pytest.raises(PolicyConfigurationError) as error:
        load_effective_policy(
            pathogen_registry=registry,
            configuration_root=tmp_path,
            requested_standardization_targets=("loc",),
            extraction_required=False,
        )

    assert str(location_path) in str(error.value)
    assert "llm_user_prompt_template" in str(error.value)


def test_location_policy_unsupported_version_error_provides_migration_guidance(
    tmp_path: Path,
) -> None:
    registry = load_pathogen_registry(PATHOGEN_REGISTRY_PATH)
    location_path = _write_location_policy(tmp_path, {"schema_version": 2})

    with pytest.raises(PolicyConfigurationError) as error:
        load_effective_policy(
            pathogen_registry=registry,
            configuration_root=tmp_path,
            requested_standardization_targets=("loc",),
            extraction_required=False,
        )

    message = str(error.value)
    assert str(location_path) in message
    assert "schema_version" in message
    assert "unsupported schema version 2" in message
    assert "supported schema version is 1" in message
    assert "migrate" in message


def _write_isolation_source_prompt_policy(
    tmp_path: Path,
    overrides: dict[str, object] | None = None,
) -> Path:
    ontology = tmp_path / "ontology.tsv"
    ontology.write_text(
        "term_path\tdisplay_term\texternal_ontology_identifier\tcrosslink_targets\t"
        "synonyms\tcomment\texamples\n"
        "environmental\tenvironmental\t\t\t\t\t\n"
        "unspecified\tunspecified\t\t\t\tNo source named.\t\n",
        encoding="utf-8",
    )
    policy = {
        "schema_version": 1,
        "prompt_version": "isolation-v1",
        "system_prompt": "Classify with:\n{ontology_tree}",
        "user_prompt": "{metadata}\n{bioproject_context}",
        "bioproject_system_prompt": "Use study context carefully.",
        "bioproject_user_prompt": "Projects:\n{bioproject_context}",
        "ontology_tsv_path": ontology.as_posix(),
        "cache_db_path": (tmp_path / "isolation-cache.db").as_posix(),
    }
    policy.update(overrides or {})
    path = tmp_path / "isolation_source.yaml"
    path.write_text(yaml.safe_dump(policy, sort_keys=False), encoding="utf-8")
    (tmp_path / "host.yaml").write_text(
        (CONFIG_DIR / "host.yaml").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    return path


def test_unselected_isolation_source_prompt_policy_is_not_loaded(tmp_path: Path) -> None:
    registry = load_pathogen_registry(PATHOGEN_REGISTRY_PATH)
    (tmp_path / "isolation_source.yaml").write_text("broken: [\n", encoding="utf-8")

    policy = load_effective_policy(
        pathogen_registry=registry,
        configuration_root=tmp_path,
        requested_standardization_targets=("date",),
        extraction_required=False,
    )

    assert policy.isolation_source_prompt_policy is None


@pytest.mark.parametrize(
    ("overrides", "policy_key"),
    [
        ({"unexpected": True}, "top level.unexpected"),
        ({"schema_version": "1"}, "schema_version"),
        ({"prompt_version": 1}, "prompt_version"),
        ({"system_prompt": ""}, "system_prompt"),
        ({"system_prompt": "No ontology"}, "system_prompt"),
        ({"system_prompt": "{ontology_tree} {unsupported}"}, "system_prompt"),
        ({"system_prompt": "{ontology_tree} {unsupported.attr}"}, "system_prompt"),
        ({"user_prompt": "{metadata}"}, "user_prompt"),
        ({"user_prompt": "{metadata} {bioproject_context} {unsupported}"}, "user_prompt"),
        (
            {"user_prompt": "{metadata} {bioproject_context} {unsupported[0]}"},
            "user_prompt",
        ),
        ({"user_prompt": "{metadata} {bioproject_context} {0}"}, "user_prompt"),
        ({"user_prompt": "{metadata} {bioproject_context} {}"}, "user_prompt"),
        ({"user_prompt": "{metadata} {bioproject_context} {!r}"}, "user_prompt"),
        ({"user_prompt": "{metadata} {bioproject_context} {:<20}"}, "user_prompt"),
        ({"user_prompt": "{metadata} {bioproject_context} {.attribute}"}, "user_prompt"),
        ({"user_prompt": "{metadata} {bioproject_context} {[0]}"}, "user_prompt"),
        ({"user_prompt": "{metadata} {bioproject_context"}, "user_prompt"),
        (
            {"user_prompt": "{metadata} {metadata!r} {bioproject_context}"},
            "user_prompt",
        ),
        ({"bioproject_system_prompt": None}, "bioproject_system_prompt"),
        ({"bioproject_system_prompt": "Rules {unsupported!r}"}, "bioproject_system_prompt"),
        ({"bioproject_user_prompt": "No project placeholder"}, "bioproject_user_prompt"),
        ({"ontology_tsv_path": 1}, "ontology_tsv_path"),
        ({"ontology_tsv_path": None}, "ontology_tsv_path"),
        ({"cache_db_path": ""}, "cache_db_path"),
        ({"cache_db_path": None}, "cache_db_path"),
    ],
)
def test_isolation_source_prompt_policy_rejects_invalid_values_with_source_and_key(
    tmp_path: Path,
    overrides: dict[str, object],
    policy_key: str,
) -> None:
    registry = load_pathogen_registry(PATHOGEN_REGISTRY_PATH)
    policy_path = _write_isolation_source_prompt_policy(tmp_path, overrides)

    with pytest.raises(PolicyConfigurationError) as error:
        load_effective_policy(
            pathogen_registry=registry,
            configuration_root=tmp_path,
            requested_standardization_targets=("iso",),
            extraction_required=False,
        )

    assert str(policy_path) in str(error.value)
    assert policy_key in str(error.value)


@pytest.mark.parametrize(
    "missing_key",
    [
        "schema_version",
        "prompt_version",
        "system_prompt",
        "user_prompt",
        "bioproject_system_prompt",
        "bioproject_user_prompt",
    ],
)
def test_isolation_source_prompt_policy_rejects_missing_required_values(
    tmp_path: Path,
    missing_key: str,
) -> None:
    registry = load_pathogen_registry(PATHOGEN_REGISTRY_PATH)
    policy_path = _write_isolation_source_prompt_policy(tmp_path)
    policy = yaml.safe_load(policy_path.read_text(encoding="utf-8"))
    del policy[missing_key]
    policy_path.write_text(yaml.safe_dump(policy, sort_keys=False), encoding="utf-8")

    with pytest.raises(PolicyConfigurationError) as error:
        load_effective_policy(
            pathogen_registry=registry,
            configuration_root=tmp_path,
            requested_standardization_targets=("iso",),
            extraction_required=False,
        )

    assert str(policy_path) in str(error.value)
    assert missing_key in str(error.value)


def test_isolation_source_prompt_policy_unsupported_version_error_provides_migration_guidance(
    tmp_path: Path,
) -> None:
    registry = load_pathogen_registry(PATHOGEN_REGISTRY_PATH)
    policy_path = _write_isolation_source_prompt_policy(tmp_path, {"schema_version": 2})

    with pytest.raises(PolicyConfigurationError) as error:
        load_effective_policy(
            pathogen_registry=registry,
            configuration_root=tmp_path,
            requested_standardization_targets=("iso",),
            extraction_required=False,
        )

    message = str(error.value)
    assert str(policy_path) in message
    assert "schema_version" in message
    assert "unsupported schema version 2" in message
    assert "supported schema version is 1" in message
    assert "migrate" in message


def test_isolation_source_prompt_policy_preserves_configured_path_spelling(
    tmp_path: Path,
) -> None:
    policy_path = _write_isolation_source_prompt_policy(tmp_path)
    source_mapping = yaml.safe_load(policy_path.read_text(encoding="utf-8"))
    source_mapping["ontology_tsv_path"] = source_mapping["ontology_tsv_path"].replace("/", "\\")
    source_mapping["cache_db_path"] = source_mapping["cache_db_path"].replace("/", "\\")
    policy_path.write_text(yaml.safe_dump(source_mapping, sort_keys=False), encoding="utf-8")

    policy = IsolationSourcePromptPolicy.load(policy_path)

    assert policy.as_legacy_mapping()["ontology_tsv_path"] == source_mapping["ontology_tsv_path"]
    assert policy.as_legacy_mapping()["cache_db_path"] == source_mapping["cache_db_path"]
