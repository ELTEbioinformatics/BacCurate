"""Public contract tests for the typed target-pathogen registry."""

from pathlib import Path
from typing import Any

import pytest
import yaml

from baccurate.pathogen_registry.registry import (
    Pathogen,
    PathogenRegistry,
    PathogenRegistryError,
    load_pathogen_registry,
)


def _write_policy(path: Path, policy: dict[Any, Any]) -> Path:
    path.write_text(yaml.safe_dump(policy, sort_keys=False), encoding="utf-8")
    return path


@pytest.mark.parametrize(
    ("policy", "policy_key"),
    [
        ("alpha: {}\n", "schema_version"),
        ("schema_version: 2\n", "schema_version"),
    ],
)
def test_registry_loader_rejects_missing_and_unsupported_versions(
    tmp_path: Path,
    policy: str,
    policy_key: str,
) -> None:
    path = tmp_path / "pathogens.yaml"
    path.write_text(policy, encoding="utf-8")

    with pytest.raises(PathogenRegistryError) as error:
        load_pathogen_registry(path)

    assert str(path) in str(error.value)
    assert policy_key in str(error.value)


@pytest.mark.parametrize(
    ("registry_entries", "policy_key"),
    [
        ({"alpha": "not-a-mapping"}, "alpha"),
        ({"alpha": {}}, "alpha"),
        (
            {
                "alpha": {
                    "scientific_name": "Alpha",
                    "ncbi_taxid": 1,
                    "rank": "genus",
                    "unexpected": True,
                }
            },
            "alpha.unexpected",
        ),
        ({"": {"scientific_name": "Alpha", "ncbi_taxid": 1, "rank": "genus"}}, "<key>"),
        (
            {"alpha": {"ncbi_taxid": 1, "rank": "genus"}},
            "alpha.scientific_name",
        ),
        (
            {"alpha": {"scientific_name": "", "ncbi_taxid": 1, "rank": "genus"}},
            "alpha.scientific_name",
        ),
        (
            {"alpha": {"scientific_name": "Alpha", "ncbi_taxid": 1, "rank": "family"}},
            "alpha.rank",
        ),
        (
            {"alpha": {"scientific_name": "Alpha", "ncbi_taxid": 0, "rank": "genus"}},
            "alpha.ncbi_taxid",
        ),
        (
            {
                "alpha": {
                    "scientific_name": "Alpha",
                    "ncbi_taxid": 1,
                    "rank": "genus",
                    "also_taxids": [2, 2],
                }
            },
            "alpha.also_taxids.1",
        ),
        (
            {
                "alpha": {
                    "scientific_name": "Alpha",
                    "ncbi_taxid": 1,
                    "rank": "genus",
                },
                "container": {
                    "alpha": {
                        "scientific_name": "Other alpha",
                        "ncbi_taxid": 2,
                        "rank": "species",
                    }
                },
            },
            "container.alpha",
        ),
        (
            {
                "alpha": {
                    "scientific_name": "Alpha",
                    "ncbi_taxid": 1,
                    "rank": "genus",
                },
                "beta": {
                    "scientific_name": "Beta",
                    "ncbi_taxid": 1,
                    "rank": "species",
                },
            },
            "beta.ncbi_taxid",
        ),
    ],
)
def test_registry_rejects_invalid_target_pathogen_policy(
    tmp_path: Path,
    registry_entries: dict[str, object],
    policy_key: str,
) -> None:
    path = _write_policy(
        tmp_path / "pathogens.yaml",
        {"schema_version": 1, **registry_entries},
    )

    with pytest.raises(PathogenRegistryError) as error:
        load_pathogen_registry(path)

    assert str(path) in str(error.value)
    assert policy_key in str(error.value)


def test_distinct_ancestor_and_descendant_target_taxa_remain_valid(
    tmp_path: Path,
) -> None:
    path = _write_policy(
        tmp_path / "pathogens.yaml",
        {
            "schema_version": 1,
            "enterobacter": {
                "scientific_name": "Enterobacter",
                "ncbi_taxid": 547,
                "rank": "genus",
            },
            "ecloacae": {
                "scientific_name": "Enterobacter cloacae",
                "ncbi_taxid": 550,
                "rank": "species",
            },
        },
    )

    registry = load_pathogen_registry(path)

    assert registry.target_taxa() == (("enterobacter", 547), ("ecloacae", 550))


def test_container_may_contain_a_pathogen_key_that_matches_a_policy_field(
    tmp_path: Path,
) -> None:
    path = _write_policy(
        tmp_path / "pathogens.yaml",
        {
            "schema_version": 1,
            "examples": {
                "rank": {
                    "scientific_name": "Example species",
                    "ncbi_taxid": 10,
                    "rank": "species",
                },
            },
        },
    )

    registry = load_pathogen_registry(path)

    assert registry.containers == {"examples": ("rank",)}
    assert registry.pathogen_keys == ("rank",)


def test_registry_canonical_serialization_preserves_policy_order(
    tmp_path: Path,
) -> None:
    path = _write_policy(
        tmp_path / "pathogens.yaml",
        {
            "schema_version": 1,
            "alpha": {
                "scientific_name": "Alpha",
                "ncbi_taxid": 10,
                "rank": "genus",
                "also_taxids": [12, 11],
            },
            "pair": {
                "beta": {
                    "scientific_name": "Beta",
                    "ncbi_taxid": 20,
                    "rank": "species",
                },
                "gamma": {
                    "scientific_name": "Gamma",
                    "ncbi_taxid": 30,
                    "rank": "species",
                },
            },
        },
    )

    registry = load_pathogen_registry(path)

    assert registry.serialize() == (
        '{"containers":[{"key":"pair","pathogen_keys":["beta","gamma"]}],'
        '"schema_version":1,'
        '"target_pathogens":['
        '{"also_taxids":[12,11],"container":null,"key":"alpha","ncbi_taxid":10,'
        '"rank":"genus","scientific_name":"Alpha"},'
        '{"also_taxids":[],"container":"pair","key":"beta","ncbi_taxid":20,'
        '"rank":"species","scientific_name":"Beta"},'
        '{"also_taxids":[],"container":"pair","key":"gamma","ncbi_taxid":30,'
        '"rank":"species","scientific_name":"Gamma"}]}'
    )
    assert registry.serialize() == load_pathogen_registry(path).serialize()


def test_registry_collections_are_immutable(tmp_path: Path) -> None:
    path = _write_policy(
        tmp_path / "pathogens.yaml",
        {
            "schema_version": 1,
            "alpha": {
                "scientific_name": "Alpha",
                "ncbi_taxid": 10,
                "rank": "genus",
            },
        },
    )
    registry = load_pathogen_registry(path)

    with pytest.raises(TypeError):
        registry.target_pathogens["beta"] = registry.target_pathogens["alpha"]  # type: ignore[index]
    with pytest.raises(AttributeError):
        registry.schema_version = 2  # type: ignore[misc]


def test_directly_constructed_registry_copies_mutable_collections() -> None:
    alpha = Pathogen("alpha", "Alpha", 10, "genus")
    target_pathogens = {"alpha": alpha}
    containers = {"examples": ("alpha",)}

    registry = PathogenRegistry(1, target_pathogens, containers)
    target_pathogens["beta"] = Pathogen("beta", "Beta", 20, "genus")
    containers["examples"] = ("beta",)

    assert registry.target_pathogens == {"alpha": alpha}
    assert registry.containers == {"examples": ("alpha",)}
