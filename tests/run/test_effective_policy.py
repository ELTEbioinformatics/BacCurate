"""Public contract tests for selective effective-policy loading."""

import json
import shutil
from pathlib import Path

import pytest
import yaml

from baccurate.adapters.policy_yaml import PolicyConfigurationError
from baccurate.paths import (
    DEFAULT_ISOLATION_SOURCE_ONTOLOGY_DIRECTORY,
)
from baccurate.run.effective_policy import load_effective_policy
from baccurate.standardization.isolation_source import IsolationSourcePromptPolicy
from baccurate.standardization.location import LocationPolicy, LocationStandardizer
from baccurate.taxon_registry.registry import load_taxon_registry

ROOT = Path(__file__).parents[2]
CONFIG_DIR = ROOT / "config"
TAXON_REGISTRY_PATH = CONFIG_DIR / "taxa.yaml"
STANDARDIZATION_TARGETS = ("host", "date", "loc", "iso")


def test_effective_policy_loads_selected_policies_and_ignores_unselected_policy(
    tmp_path: Path,
) -> None:
    registry = load_taxon_registry(TAXON_REGISTRY_PATH)
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
        "selection.yaml",
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
        taxon_registry=registry,
        configuration_root=configuration_root,
        requested_standardization_targets=("iso",),
        extraction_required=False,
    )

    assert policy.selection_policy is None
    assert policy.host_policy is not None
    assert policy.location_policy is None
    assert policy.isolation_source_prompt_policy is not None


def test_effective_policy_loads_required_selection_policy() -> None:
    registry = load_taxon_registry(TAXON_REGISTRY_PATH)

    policy = load_effective_policy(
        taxon_registry=registry,
        configuration_root=CONFIG_DIR,
        requested_standardization_targets=("date",),
        extraction_required=True,
    )

    assert policy.taxon_registry is registry
    assert policy.selection_policy is not None
    assert policy.selection_policy.schema_version == 3
    with pytest.raises(AttributeError):
        policy.selection_policy = None  # type: ignore[misc]


def test_effective_policy_does_not_load_selection_policy_for_reused_bundle(
    tmp_path: Path,
) -> None:
    registry = load_taxon_registry(TAXON_REGISTRY_PATH)
    (tmp_path / "selection.yaml").write_text("broken: [\n", encoding="utf-8")

    policy = load_effective_policy(
        taxon_registry=registry,
        configuration_root=tmp_path,
        requested_standardization_targets=("date",),
        extraction_required=False,
    )

    assert policy.taxon_registry is registry
    assert policy.selection_policy is None


def test_registry_scientific_name_change_updates_derived_host_rejection(
    tmp_path: Path,
) -> None:
    host_policy = {
        "schema_version": 3,
        "normalization": {"ignored_substrings": []},
        "routing": {"isolation_source_keywords": []},
        "curated_taxa": {},
        "value_rejections": {
            "exact": ["literal rejection", {"taxon_key": "target"}],
        },
    }
    (tmp_path / "host.yaml").write_text(
        yaml.safe_dump(host_policy, sort_keys=False),
        encoding="utf-8",
    )
    first_registry_path = tmp_path / "first-taxa.yaml"
    second_registry_path = tmp_path / "second-taxa.yaml"
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
        taxon_registry=load_taxon_registry(first_registry_path),
        configuration_root=tmp_path,
        requested_standardization_targets=("host",),
        extraction_required=False,
    ).host_policy
    second = load_effective_policy(
        taxon_registry=load_taxon_registry(second_registry_path),
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
    registry = load_taxon_registry(TAXON_REGISTRY_PATH)
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
            taxon_registry=registry,
            configuration_root=tmp_path,
            requested_standardization_targets=("host",),
            extraction_required=False,
        )

    assert str(host_path) in str(error.value)
    assert "normalization.unexpected" in str(error.value)


def test_host_policy_rejects_unknown_taxon_key_reference(tmp_path: Path) -> None:
    registry = load_taxon_registry(TAXON_REGISTRY_PATH)
    host_path = tmp_path / "host.yaml"
    host_path.write_text(
        yaml.safe_dump(
            {
                "schema_version": 3,
                "normalization": {"ignored_substrings": []},
                "routing": {"isolation_source_keywords": []},
                "curated_taxa": {},
                "value_rejections": {
                    "exact": [{"taxon_key": "not-a-target"}],
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    with pytest.raises(PolicyConfigurationError) as error:
        load_effective_policy(
            taxon_registry=registry,
            configuration_root=tmp_path,
            requested_standardization_targets=("host",),
            extraction_required=False,
        )

    assert str(host_path) in str(error.value)
    assert "value_rejections.exact.0.taxon_key" in str(error.value)
    assert "not-a-target" in str(error.value)


def test_host_policy_rejects_curated_match_overlapping_derived_rejection(
    tmp_path: Path,
) -> None:
    registry = load_taxon_registry(TAXON_REGISTRY_PATH)
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
                    "exact": [{"taxon_key": "ecoli"}],
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    with pytest.raises(PolicyConfigurationError) as error:
        load_effective_policy(
            taxon_registry=registry,
            configuration_root=tmp_path,
            requested_standardization_targets=("host",),
            extraction_required=False,
        )

    message = str(error.value)
    assert str(host_path) in message
    assert "value_rejections.exact" in message
    assert "both a curated term and a value rejection" in message


def test_unselected_host_policy_is_not_loaded(tmp_path: Path) -> None:
    registry = load_taxon_registry(TAXON_REGISTRY_PATH)
    (tmp_path / "host.yaml").write_text("broken: [\n", encoding="utf-8")

    policy = load_effective_policy(
        taxon_registry=registry,
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
        "schema_version": 2,
        "coordinate_attributes": ["lat_lon", "latitude"],
        "insdc_country_map": {"United States": "USA"},
        "reviewed_mappings": {"uae": "United Arab Emirates"},
        "reviewed_unmapped": ["ncbs"],
        "geo_loc_list_path": "reference/geo_loc_list.txt",
    }
    policy.update(overrides or {})
    path = tmp_path / "location.yaml"
    path.write_text(yaml.safe_dump(policy, sort_keys=False), encoding="utf-8")
    return path


def test_unselected_location_policy_is_not_loaded(tmp_path: Path) -> None:
    registry = load_taxon_registry(TAXON_REGISTRY_PATH)
    (tmp_path / "location.yaml").write_text("broken: [\n", encoding="utf-8")

    policy = load_effective_policy(
        taxon_registry=registry,
        configuration_root=tmp_path,
        requested_standardization_targets=("date",),
        extraction_required=False,
    )

    assert policy.location_policy is None


def test_location_policy_preserves_relative_reference_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "reference").mkdir()
    (tmp_path / "reference" / "geo_loc_list.txt").write_text(
        "Germany\n",
        encoding="utf-8",
    )
    location_path = _write_location_policy(tmp_path)

    policy = LocationPolicy.load(location_path)

    assert policy.geo_loc_list_path == Path("reference/geo_loc_list.txt")
    assert policy.reviewed_mappings == {"uae": "United Arab Emirates"}
    assert policy.reviewed_unmapped == frozenset({"ncbs"})


def test_location_policy_rejects_unusable_resource_selection_before_standardization(
    tmp_path: Path,
) -> None:
    registry = load_taxon_registry(TAXON_REGISTRY_PATH)
    location_path = _write_location_policy(
        tmp_path,
        {"geo_loc_list_path": "missing-reference.txt"},
    )

    with pytest.raises(PolicyConfigurationError) as error:
        load_effective_policy(
            taxon_registry=registry,
            configuration_root=tmp_path,
            requested_standardization_targets=("loc",),
            extraction_required=False,
        )

    assert str(location_path) in str(error.value)
    assert "geo_loc_list_path" in str(error.value)


def test_location_standardizer_does_not_reopen_loaded_policy_source(tmp_path: Path) -> None:
    geo_loc_list = tmp_path / "geo_loc_list.txt"
    geo_loc_list.write_text("Germany\n", encoding="utf-8")
    location_path = _write_location_policy(
        tmp_path,
        {"geo_loc_list_path": geo_loc_list.as_posix()},
    )
    policy = LocationPolicy.load(location_path)
    location_path.write_text("broken: [\n", encoding="utf-8")

    outcome = LocationStandardizer(policy).standardize(
        {
            "accession": "NO_REOPEN",
            "loc_attr_orig": "geo_loc_name",
            "loc_val_orig": "Germany",
        }
    )

    assert outcome.country == "Germany"


@pytest.mark.parametrize(
    ("overrides", "policy_key"),
    [
        ({"unexpected": True}, "top level.unexpected"),
        ({"schema_version": "2"}, "schema_version"),
        ({"coordinate_attributes": "lat_lon"}, "coordinate_attributes"),
        ({"coordinate_attributes": ["lat_lon", "  "]}, "coordinate_attributes.1"),
        ({"insdc_country_map": []}, "insdc_country_map"),
        ({"insdc_country_map": {"": "USA"}}, "insdc_country_map"),
        ({"insdc_country_map": {"United States": 1}}, "insdc_country_map.United States"),
        ({"reviewed_mappings": []}, "reviewed_mappings"),
        ({"reviewed_mappings": {"uae": ""}}, "reviewed_mappings.uae"),
        ({"reviewed_unmapped": {"ncbs": True}}, "reviewed_unmapped"),
        ({"reviewed_unmapped": ["ncbs", " "]}, "reviewed_unmapped.1"),
        ({"geo_loc_list_path": 1}, "geo_loc_list_path"),
    ],
)
def test_location_policy_rejects_invalid_values_with_source_and_key(
    tmp_path: Path,
    overrides: dict[str, object],
    policy_key: str,
) -> None:
    registry = load_taxon_registry(TAXON_REGISTRY_PATH)
    location_path = _write_location_policy(tmp_path, overrides)

    with pytest.raises(PolicyConfigurationError) as error:
        load_effective_policy(
            taxon_registry=registry,
            configuration_root=tmp_path,
            requested_standardization_targets=("loc",),
            extraction_required=False,
        )

    assert str(location_path) in str(error.value)
    assert policy_key in str(error.value)


@pytest.mark.parametrize(
    "missing_key",
    ["schema_version", "reviewed_mappings", "reviewed_unmapped"],
)
def test_location_policy_rejects_missing_required_values(
    tmp_path: Path,
    missing_key: str,
) -> None:
    registry = load_taxon_registry(TAXON_REGISTRY_PATH)
    location_path = _write_location_policy(tmp_path)
    policy = yaml.safe_load(location_path.read_text(encoding="utf-8"))
    del policy[missing_key]
    location_path.write_text(yaml.safe_dump(policy, sort_keys=False), encoding="utf-8")

    with pytest.raises(PolicyConfigurationError) as error:
        load_effective_policy(
            taxon_registry=registry,
            configuration_root=tmp_path,
            requested_standardization_targets=("loc",),
            extraction_required=False,
        )

    assert str(location_path) in str(error.value)
    assert missing_key in str(error.value)


def test_location_policy_unsupported_version_error_provides_migration_guidance(
    tmp_path: Path,
) -> None:
    registry = load_taxon_registry(TAXON_REGISTRY_PATH)
    location_path = _write_location_policy(tmp_path, {"schema_version": 1})

    with pytest.raises(PolicyConfigurationError) as error:
        load_effective_policy(
            taxon_registry=registry,
            configuration_root=tmp_path,
            requested_standardization_targets=("loc",),
            extraction_required=False,
        )

    message = str(error.value)
    assert str(location_path) in message
    assert "schema_version" in message
    assert "unsupported schema version 1" in message
    assert "supported schema version is 2" in message
    assert "migrate" in message


def _write_isolation_source_prompt_policy(
    tmp_path: Path,
    overrides: dict[str, object] | None = None,
) -> Path:
    policy = {
        "schema_version": 3,
        "prompt_version": "isolation-v1",
        "system_prompt": "Classify with:\n{ontology_tree}",
        "user_prompt": "{metadata}",
        "ontology_directory": DEFAULT_ISOLATION_SOURCE_ONTOLOGY_DIRECTORY.as_posix(),
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
    registry = load_taxon_registry(TAXON_REGISTRY_PATH)
    (tmp_path / "isolation_source.yaml").write_text("broken: [\n", encoding="utf-8")

    policy = load_effective_policy(
        taxon_registry=registry,
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
        ({"user_prompt": "No metadata"}, "user_prompt"),
        ({"user_prompt": "{metadata} {unsupported}"}, "user_prompt"),
        (
            {"user_prompt": "{metadata} {unsupported[0]}"},
            "user_prompt",
        ),
        ({"user_prompt": "{metadata} {0}"}, "user_prompt"),
        ({"user_prompt": "{metadata} {}"}, "user_prompt"),
        ({"user_prompt": "{metadata} {!r}"}, "user_prompt"),
        ({"user_prompt": "{metadata} {:<20}"}, "user_prompt"),
        ({"user_prompt": "{metadata} {.attribute}"}, "user_prompt"),
        ({"user_prompt": "{metadata} {[0]}"}, "user_prompt"),
        ({"user_prompt": "{metadata"}, "user_prompt"),
        (
            {"user_prompt": "{metadata} {metadata!r}"},
            "user_prompt",
        ),
        ({"bioproject_system_prompt": "obsolete"}, "top level.bioproject_system_prompt"),
        ({"bioproject_user_prompt": "obsolete"}, "top level.bioproject_user_prompt"),
        ({"ontology_directory": 1}, "ontology_directory"),
        ({"ontology_directory": None}, "ontology_directory"),
        ({"ontology_tsv_path": "ontology.tsv"}, "top level.ontology_tsv_path"),
        ({"cache_db_path": ""}, "cache_db_path"),
        ({"cache_db_path": None}, "cache_db_path"),
    ],
)
def test_isolation_source_prompt_policy_rejects_invalid_values_with_source_and_key(
    tmp_path: Path,
    overrides: dict[str, object],
    policy_key: str,
) -> None:
    registry = load_taxon_registry(TAXON_REGISTRY_PATH)
    policy_path = _write_isolation_source_prompt_policy(tmp_path, overrides)

    with pytest.raises(PolicyConfigurationError) as error:
        load_effective_policy(
            taxon_registry=registry,
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
    ],
)
def test_isolation_source_prompt_policy_rejects_missing_required_values(
    tmp_path: Path,
    missing_key: str,
) -> None:
    registry = load_taxon_registry(TAXON_REGISTRY_PATH)
    policy_path = _write_isolation_source_prompt_policy(tmp_path)
    policy = yaml.safe_load(policy_path.read_text(encoding="utf-8"))
    del policy[missing_key]
    policy_path.write_text(yaml.safe_dump(policy, sort_keys=False), encoding="utf-8")

    with pytest.raises(PolicyConfigurationError) as error:
        load_effective_policy(
            taxon_registry=registry,
            configuration_root=tmp_path,
            requested_standardization_targets=("iso",),
            extraction_required=False,
        )

    assert str(policy_path) in str(error.value)
    assert missing_key in str(error.value)


def test_isolation_source_prompt_policy_preserves_configured_path_spelling(
    tmp_path: Path,
) -> None:
    policy_path = _write_isolation_source_prompt_policy(tmp_path)
    source_mapping = yaml.safe_load(policy_path.read_text(encoding="utf-8"))
    source_mapping["ontology_directory"] = source_mapping["ontology_directory"].replace("/", "\\")
    source_mapping["cache_db_path"] = source_mapping["cache_db_path"].replace("/", "\\")
    policy_path.write_text(yaml.safe_dump(source_mapping, sort_keys=False), encoding="utf-8")

    policy = IsolationSourcePromptPolicy.load(policy_path)

    assert policy.as_legacy_mapping()["ontology_directory"] == source_mapping["ontology_directory"]
    assert policy.as_legacy_mapping()["cache_db_path"] == source_mapping["cache_db_path"]


def test_isolation_source_prompt_policy_serializes_the_ontology_directory(
    tmp_path: Path,
) -> None:
    policy = IsolationSourcePromptPolicy.load(_write_isolation_source_prompt_policy(tmp_path))

    serialized = json.loads(policy.serialize())

    assert serialized["ontology_directory"] == (
        DEFAULT_ISOLATION_SOURCE_ONTOLOGY_DIRECTORY.as_posix()
    )
    assert "ontology_tsv_path" not in serialized


def test_isolation_source_prompt_version_is_independent_of_schema_version(
    tmp_path: Path,
) -> None:
    policy = IsolationSourcePromptPolicy.load(
        _write_isolation_source_prompt_policy(
            tmp_path,
            {"schema_version": 3, "prompt_version": "wording-revision-17"},
        )
    )

    assert policy.schema_version == 3
    assert policy.prompt_version == "wording-revision-17"


def test_isolation_source_prompt_policy_rejects_a_non_directory_ontology_path(
    tmp_path: Path,
) -> None:
    ontology_file = tmp_path / "ontology"
    ontology_file.write_text("not a directory\n", encoding="utf-8")
    policy_path = _write_isolation_source_prompt_policy(
        tmp_path,
        {"ontology_directory": ontology_file.as_posix()},
    )

    with pytest.raises(PolicyConfigurationError) as error:
        IsolationSourcePromptPolicy.load(policy_path)

    message = str(error.value)
    assert "ontology_directory" in message
    assert "readable directory" in message


@pytest.mark.parametrize(
    "missing_filename",
    ["facets.yaml", "terms.tsv", "mappings.sssom.tsv", "mappings.sssom.yml"],
)
def test_isolation_source_prompt_policy_names_a_missing_ontology_file(
    tmp_path: Path,
    missing_filename: str,
) -> None:
    ontology_directory = tmp_path / "ontology"
    shutil.copytree(DEFAULT_ISOLATION_SOURCE_ONTOLOGY_DIRECTORY, ontology_directory)
    (ontology_directory / missing_filename).unlink()
    policy_path = _write_isolation_source_prompt_policy(
        tmp_path,
        {"ontology_directory": ontology_directory.as_posix()},
    )

    with pytest.raises(PolicyConfigurationError) as error:
        IsolationSourcePromptPolicy.load(policy_path)

    message = str(error.value)
    assert "ontology_directory" in message
    assert missing_filename in message
