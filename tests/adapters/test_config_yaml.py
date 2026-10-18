"""Shared contract for loading YAML-backed BacCurate config."""

from pathlib import Path

import pytest

from baccurate.adapters.config_yaml import ConfigError, load_config_mapping


def test_config_mapping_rejects_malformed_yaml(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text("schema_version: 1\nbroken: [\n", encoding="utf-8")

    with pytest.raises(ConfigError) as error:
        load_config_mapping(config_path)

    assert "<yaml>" in str(error.value)


def test_config_mapping_rejects_non_mapping_root(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text("- schema_version: 1\n", encoding="utf-8")

    with pytest.raises(ConfigError) as error:
        load_config_mapping(config_path)

    assert "<root>" in str(error.value)


def test_config_mapping_rejects_nested_duplicate_key(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "schema_version: 1\n"
        "targets:\n"
        "  host:\n"
        "    selected_attributes: {}\n"
        "    selected_attributes: {}\n",
        encoding="utf-8",
    )

    with pytest.raises(ConfigError) as error:
        load_config_mapping(config_path)

    assert "targets.host.selected_attributes" in str(error.value)


def test_config_mapping_error_identifies_its_source(tmp_path: Path) -> None:
    missing_config_path = tmp_path / "missing-config.yaml"

    with pytest.raises(ConfigError) as error:
        load_config_mapping(missing_config_path)

    assert str(missing_config_path) in str(error.value)
