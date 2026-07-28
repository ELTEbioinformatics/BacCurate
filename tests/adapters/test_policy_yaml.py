"""Shared contract for loading YAML-backed BacCurate policy."""

from pathlib import Path

import pytest

from baccurate.adapters.policy_yaml import PolicyConfigurationError, load_policy_mapping


def test_policy_mapping_rejects_malformed_yaml(tmp_path: Path) -> None:
    policy_path = tmp_path / "policy.yaml"
    policy_path.write_text("schema_version: 1\nbroken: [\n", encoding="utf-8")

    with pytest.raises(PolicyConfigurationError) as error:
        load_policy_mapping(policy_path)

    assert "<yaml>" in str(error.value)


def test_policy_mapping_rejects_non_mapping_root(tmp_path: Path) -> None:
    policy_path = tmp_path / "policy.yaml"
    policy_path.write_text("- schema_version: 1\n", encoding="utf-8")

    with pytest.raises(PolicyConfigurationError) as error:
        load_policy_mapping(policy_path)

    assert "<root>" in str(error.value)


def test_policy_mapping_rejects_nested_duplicate_key(tmp_path: Path) -> None:
    policy_path = tmp_path / "policy.yaml"
    policy_path.write_text(
        "schema_version: 1\n"
        "targets:\n"
        "  host:\n"
        "    selected_attributes: {}\n"
        "    selected_attributes: {}\n",
        encoding="utf-8",
    )

    with pytest.raises(PolicyConfigurationError) as error:
        load_policy_mapping(policy_path)

    assert "targets.host.selected_attributes" in str(error.value)


def test_policy_mapping_error_identifies_its_source(tmp_path: Path) -> None:
    missing_policy_path = tmp_path / "missing-policy.yaml"

    with pytest.raises(PolicyConfigurationError) as error:
        load_policy_mapping(missing_policy_path)

    assert str(missing_policy_path) in str(error.value)
