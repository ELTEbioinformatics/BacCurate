"""Public contract tests for selective run-config loading."""

import shutil
from pathlib import Path

import pytest
import yaml

from baccurate.adapters.config_yaml import ConfigError
from baccurate.paths import (
    DEFAULT_ISOLATION_SOURCE_ONTOLOGY_DIRECTORY,
)
from baccurate.run.config import load_run_config
from baccurate.standardization.isolation_source_config import (
    IsolationSourceConfig,
)
from baccurate.standardization.location import LocationConfig
from baccurate.taxon_registry.registry import load_taxon_registry

ROOT = Path(__file__).parents[2]
CONFIG_DIR = ROOT / "config"
ISOLATION_SOURCE_CONFIG = yaml.safe_load(
    (CONFIG_DIR / "isolation_source.yaml").read_text(encoding="utf-8")
)
FACETS = ISOLATION_SOURCE_CONFIG["facets"]
TAXON_REGISTRY_PATH = CONFIG_DIR / "taxa.yaml"
STANDARDIZATION_TARGETS = ("host", "date", "loc", "iso")


def test_run_config_loads_selected_configs_and_ignores_unselected_config(
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
    for config_name in (
        "selection.yaml",
        "host.yaml",
        "location.yaml",
        "isolation_source.yaml",
    ):
        destination = configuration_root / config_name
        if config_name in selected_sources:
            shutil.copyfile(CONFIG_DIR / config_name, destination)
        else:
            destination.write_text("malformed: [\n", encoding="utf-8")

    config = load_run_config(
        taxon_registry=registry,
        configuration_root=configuration_root,
        requested_standardization_targets=("iso",),
        extraction_required=False,
    )

    assert config.selection_config is None
    assert config.host_config is not None
    assert config.location_config is None
    assert config.isolation_source_config is not None


def test_run_config_loads_required_selection_config() -> None:
    registry = load_taxon_registry(TAXON_REGISTRY_PATH)

    config = load_run_config(
        taxon_registry=registry,
        configuration_root=CONFIG_DIR,
        requested_standardization_targets=("date",),
        extraction_required=True,
    )

    assert config.taxon_registry is registry
    assert config.selection_config is not None
    assert config.selection_config.schema_version == 4
    with pytest.raises(AttributeError):
        config.selection_config = None  # type: ignore[misc]


def test_run_config_does_not_load_selection_config_for_reused_extracted_metadata(
    tmp_path: Path,
) -> None:
    registry = load_taxon_registry(TAXON_REGISTRY_PATH)
    (tmp_path / "selection.yaml").write_text("broken: [\n", encoding="utf-8")

    config = load_run_config(
        taxon_registry=registry,
        configuration_root=tmp_path,
        requested_standardization_targets=("date",),
        extraction_required=False,
    )

    assert config.taxon_registry is registry
    assert config.selection_config is None


def test_registry_scientific_name_change_updates_derived_host_rejection(
    tmp_path: Path,
) -> None:
    host_config = {
        "schema_version": 3,
        "normalization": {"ignored_substrings": []},
        "routing": {"isolation_source_keywords": []},
        "curated_taxa": {},
        "value_rejections": {
            "exact": ["literal rejection", {"taxon_key": "target"}],
        },
    }
    (tmp_path / "host.yaml").write_text(
        yaml.safe_dump(host_config, sort_keys=False),
        encoding="utf-8",
    )
    first_registry_path = tmp_path / "first-taxa.yaml"
    second_registry_path = tmp_path / "second-taxa.yaml"
    registry_config = {
        "schema_version": 1,
        "target": {
            "scientific_name": "Original target name",
            "ncbi_taxid": 1,
            "rank": "species",
        },
    }
    first_registry_path.write_text(
        yaml.safe_dump(registry_config, sort_keys=False),
        encoding="utf-8",
    )
    registry_config["target"]["scientific_name"] = "Renamed target"
    second_registry_path.write_text(
        yaml.safe_dump(registry_config, sort_keys=False),
        encoding="utf-8",
    )

    first = load_run_config(
        taxon_registry=load_taxon_registry(first_registry_path),
        configuration_root=tmp_path,
        requested_standardization_targets=("host",),
        extraction_required=False,
    ).host_config
    second = load_run_config(
        taxon_registry=load_taxon_registry(second_registry_path),
        configuration_root=tmp_path,
        requested_standardization_targets=("host",),
        extraction_required=False,
    ).host_config

    assert first is not None
    assert second is not None
    assert first.value_rejections == ("literal rejection", "Original target name")
    assert second.value_rejections == ("literal rejection", "Renamed target")


def test_host_config_rejects_curated_match_overlapping_derived_rejection(
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

    with pytest.raises(ConfigError) as error:
        load_run_config(
            taxon_registry=registry,
            configuration_root=tmp_path,
            requested_standardization_targets=("host",),
            extraction_required=False,
        )

    message = str(error.value)
    assert str(host_path) in message
    assert "value_rejections.exact" in message


def test_unselected_host_config_is_not_loaded(tmp_path: Path) -> None:
    registry = load_taxon_registry(TAXON_REGISTRY_PATH)
    (tmp_path / "host.yaml").write_text("broken: [\n", encoding="utf-8")

    config = load_run_config(
        taxon_registry=registry,
        configuration_root=tmp_path,
        requested_standardization_targets=("date",),
        extraction_required=False,
    )

    assert config.host_config is None


def _write_location_config(
    tmp_path: Path,
    overrides: dict[str, object] | None = None,
) -> Path:
    config: dict[str, object] = {
        "schema_version": 2,
        "coordinate_attributes": ["lat_lon", "latitude"],
        "insdc_country_map": {"United States": "USA"},
        "reviewed_mappings": {"uae": "United Arab Emirates"},
        "reviewed_unmapped": ["ncbs"],
        "geo_loc_list_path": "reference/geo_loc_list.txt",
    }
    config.update(overrides or {})
    path = tmp_path / "location.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return path


def test_unselected_location_config_is_not_loaded(tmp_path: Path) -> None:
    registry = load_taxon_registry(TAXON_REGISTRY_PATH)
    (tmp_path / "location.yaml").write_text("broken: [\n", encoding="utf-8")

    config = load_run_config(
        taxon_registry=registry,
        configuration_root=tmp_path,
        requested_standardization_targets=("date",),
        extraction_required=False,
    )

    assert config.location_config is None


def test_location_config_preserves_relative_reference_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "reference").mkdir()
    (tmp_path / "reference" / "geo_loc_list.txt").write_text(
        "Germany\n",
        encoding="utf-8",
    )
    location_path = _write_location_config(tmp_path)

    config = LocationConfig.load(location_path)

    assert config.geo_loc_list_path == Path("reference/geo_loc_list.txt")
    assert config.reviewed_mappings == {"uae": "United Arab Emirates"}
    assert config.reviewed_unmapped == frozenset({"ncbs"})


def test_location_config_rejects_unusable_resource_selection_before_standardization(
    tmp_path: Path,
) -> None:
    registry = load_taxon_registry(TAXON_REGISTRY_PATH)
    location_path = _write_location_config(
        tmp_path,
        {"geo_loc_list_path": "missing-reference.txt"},
    )

    with pytest.raises(ConfigError) as error:
        load_run_config(
            taxon_registry=registry,
            configuration_root=tmp_path,
            requested_standardization_targets=("loc",),
            extraction_required=False,
        )

    assert str(location_path) in str(error.value)
    assert "geo_loc_list_path" in str(error.value)


def _write_isolation_source_config(
    tmp_path: Path,
    overrides: dict[str, object] | None = None,
) -> Path:
    config = {
        "schema_version": 3,
        "prompt_version": "isolation-v1",
        "system_prompt": "Classify with:\n{ontology_tree}",
        "user_prompt": "{metadata}",
        "ontology_directory": DEFAULT_ISOLATION_SOURCE_ONTOLOGY_DIRECTORY.as_posix(),
        "cache_db_path": (tmp_path / "isolation-cache.db").as_posix(),
        "facets": FACETS,
        "host_recovery_term_ids": ISOLATION_SOURCE_CONFIG["host_recovery_term_ids"],
        "direct_match_values": ISOLATION_SOURCE_CONFIG["direct_match_values"],
    }
    config.update(overrides or {})
    path = tmp_path / "isolation_source.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    (tmp_path / "host.yaml").write_text(
        (CONFIG_DIR / "host.yaml").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    return path


def test_unselected_isolation_source_config_is_not_loaded(tmp_path: Path) -> None:
    registry = load_taxon_registry(TAXON_REGISTRY_PATH)
    (tmp_path / "isolation_source.yaml").write_text("broken: [\n", encoding="utf-8")

    config = load_run_config(
        taxon_registry=registry,
        configuration_root=tmp_path,
        requested_standardization_targets=("date",),
        extraction_required=False,
    )

    assert config.isolation_source_config is None


@pytest.mark.parametrize(
    "missing_key",
    [
        "schema_version",
        "prompt_version",
        "system_prompt",
        "user_prompt",
        "facets",
        "host_recovery_term_ids",
        "direct_match_values",
    ],
)
def test_isolation_source_config_rejects_missing_required_values(
    tmp_path: Path,
    missing_key: str,
) -> None:
    registry = load_taxon_registry(TAXON_REGISTRY_PATH)
    config_path = _write_isolation_source_config(tmp_path)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    del config[missing_key]
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    with pytest.raises(ConfigError) as error:
        load_run_config(
            taxon_registry=registry,
            configuration_root=tmp_path,
            requested_standardization_targets=("iso",),
            extraction_required=False,
        )

    assert str(config_path) in str(error.value)
    assert missing_key in str(error.value)


def test_isolation_source_prompt_version_is_independent_of_schema_version(
    tmp_path: Path,
) -> None:
    config = IsolationSourceConfig.load(
        _write_isolation_source_config(
            tmp_path,
            {"schema_version": 3, "prompt_version": "wording-revision-17"},
        )
    )

    assert config.schema_version == 3
    assert config.prompt_version == "wording-revision-17"


@pytest.mark.parametrize(
    "missing_filename",
    [
        "facets.yaml",
        "terms.tsv",
        "obsolete_terms.tsv",
        "relations.tsv",
        "mappings.sssom.tsv",
        "mappings.sssom.yml",
    ],
)
def test_isolation_source_config_names_a_missing_ontology_file(
    tmp_path: Path,
    missing_filename: str,
) -> None:
    ontology_directory = tmp_path / "ontology"
    shutil.copytree(DEFAULT_ISOLATION_SOURCE_ONTOLOGY_DIRECTORY, ontology_directory)
    (ontology_directory / missing_filename).unlink()
    config_path = _write_isolation_source_config(
        tmp_path,
        {"ontology_directory": ontology_directory.as_posix()},
    )

    with pytest.raises(ConfigError) as error:
        IsolationSourceConfig.load(config_path)

    message = str(error.value)
    assert "ontology_directory" in message
    assert missing_filename in message
