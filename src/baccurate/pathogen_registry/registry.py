"""Versioned target-pathogen registry policy."""

import json
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Literal, cast

from baccurate.adapters.policy_yaml import PolicyConfigurationError, load_policy_mapping
from baccurate.paths import PATHOGENS_YAML

PathogenRegistryError = PolicyConfigurationError

_SUPPORTED_SCHEMA_VERSION = 1
type PathogenRank = Literal["genus", "species"]

_SUPPORTED_RANKS: frozenset[PathogenRank] = frozenset({"genus", "species"})
_TARGET_PATHOGEN_KEYS = frozenset({"scientific_name", "ncbi_taxid", "rank", "also_taxids"})


@dataclass(frozen=True)
class Pathogen:
    """One target pathogen (a leaf entry in the registry)."""

    key: str
    scientific_name: str
    ncbi_taxid: int
    rank: PathogenRank
    group: str | None = None
    also_taxids: tuple[int, ...] = field(default_factory=tuple)

    @property
    def taxids(self) -> tuple[int, ...]:
        """Primary taxid plus any ``also_taxids``."""
        return (self.ncbi_taxid, *self.also_taxids)


@dataclass(frozen=True, slots=True)
class PathogenRegistry:
    """Immutable, ordered policy defining BacCurate's target pathogens."""

    schema_version: int
    target_pathogens: Mapping[str, Pathogen]
    pathogen_groups: Mapping[str, tuple[str, ...]]

    def __post_init__(self) -> None:
        """Detach registry collections from caller-owned mutable mappings."""
        object.__setattr__(
            self,
            "target_pathogens",
            MappingProxyType(dict(self.target_pathogens)),
        )
        object.__setattr__(
            self,
            "pathogen_groups",
            MappingProxyType(
                {
                    pathogen_group_key: tuple(pathogen_keys)
                    for pathogen_group_key, pathogen_keys in self.pathogen_groups.items()
                }
            ),
        )

    @property
    def pathogen_keys(self) -> tuple[str, ...]:
        """Return pathogen keys in registry order."""
        return tuple(self.target_pathogens)

    @property
    def pathogen_group_keys(self) -> tuple[str, ...]:
        """Return pathogen-group keys in registry order."""
        return tuple(self.pathogen_groups)

    @property
    def keywords(self) -> tuple[str, ...]:
        """Return every command keyword in compatibility order."""
        return (*self.pathogen_keys, *self.pathogen_group_keys)

    def expand(self, registry_identifiers: Sequence[str]) -> tuple[str, ...]:
        """Expand pathogen groups with first-seen pathogen-key deduplication."""
        expanded: list[str] = []
        for registry_identifier in registry_identifiers:
            for pathogen_key in self.pathogen_groups.get(
                registry_identifier,
                (registry_identifier,),
            ):
                if pathogen_key not in expanded:
                    expanded.append(pathogen_key)
        return tuple(expanded)

    def scientific_name(self, pathogen_key: str) -> str:
        """Return a target pathogen's scientific name, or an empty string if unknown."""
        pathogen = self.target_pathogens.get(pathogen_key)
        return pathogen.scientific_name if pathogen else ""

    def target_taxa(self) -> tuple[tuple[str, int], ...]:
        """Enumerate pathogen keys and their ordered target-taxon IDs."""
        return tuple(
            (pathogen.key, taxid)
            for pathogen in self.target_pathogens.values()
            for taxid in pathogen.taxids
        )

    def pathogen_key_table(self) -> str:
        """Serialize the flattened pathogen-key table used by preparation tooling."""
        lines = ["pathogen_key\ttaxids\trank\tgroup"]
        for pathogen in self.target_pathogens.values():
            taxids = " ".join(str(taxid) for taxid in pathogen.taxids)
            lines.append(f"{pathogen.key}\t{taxids}\t{pathogen.rank}\t{pathogen.group or ''}")
        return "\n".join(lines)

    def serialize(self) -> str:
        """Return deterministic, schema-versioned JSON preserving policy order."""
        return json.dumps(
            {
                "schema_version": self.schema_version,
                "target_pathogens": [
                    {
                        "key": pathogen.key,
                        "scientific_name": pathogen.scientific_name,
                        "ncbi_taxid": pathogen.ncbi_taxid,
                        "rank": pathogen.rank,
                        "group": pathogen.group,
                        "also_taxids": list(pathogen.also_taxids),
                    }
                    for pathogen in self.target_pathogens.values()
                ],
                "pathogen_groups": [
                    {
                        "key": pathogen_group_key,
                        "pathogen_keys": list(pathogen_keys),
                    }
                    for pathogen_group_key, pathogen_keys in self.pathogen_groups.items()
                ],
            },
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )


def _error(path: Path, policy_key: str, message: str) -> PathogenRegistryError:
    return PathogenRegistryError(f"{path}: {policy_key}: {message}")


def _validate_identifier(path: Path, registry_identifier: object, policy_key: str) -> str:
    if not isinstance(registry_identifier, str) or not registry_identifier.strip():
        raise _error(path, policy_key, "must be a non-empty string")
    return registry_identifier


def _is_target_pathogen(registry_entry_policy: Mapping[object, object]) -> bool:
    return not all(
        isinstance(nested_policy, Mapping) for nested_policy in registry_entry_policy.values()
    )


def _parse_target_pathogen(
    path: Path,
    pathogen_key: str,
    target_pathogen_policy: Mapping[object, object],
    pathogen_group_key: str | None,
    policy_key: str,
) -> Pathogen:
    unknown_keys = [
        target_pathogen_key
        for target_pathogen_key in target_pathogen_policy
        if target_pathogen_key not in _TARGET_PATHOGEN_KEYS
    ]
    if unknown_keys:
        raise _error(path, f"{policy_key}.{unknown_keys[0]}", "unknown policy key")

    scientific_name = target_pathogen_policy.get("scientific_name")
    if not isinstance(scientific_name, str) or not scientific_name.strip():
        raise _error(
            path,
            f"{policy_key}.scientific_name",
            "must be a non-empty string",
        )

    ncbi_taxid = target_pathogen_policy.get("ncbi_taxid")
    if type(ncbi_taxid) is not int or ncbi_taxid <= 0:
        raise _error(path, f"{policy_key}.ncbi_taxid", "must be a positive integer")

    rank = target_pathogen_policy.get("rank")
    if not isinstance(rank, str) or rank not in _SUPPORTED_RANKS:
        raise _error(
            path,
            f"{policy_key}.rank",
            f"must be one of {sorted(_SUPPORTED_RANKS)}",
        )

    raw_also_taxids = target_pathogen_policy.get("also_taxids", [])
    if not isinstance(raw_also_taxids, list):
        raise _error(path, f"{policy_key}.also_taxids", "must be a list")
    also_taxids: list[int] = []
    seen_taxids = {ncbi_taxid}
    for index, taxid in enumerate(raw_also_taxids):
        also_taxid_policy_key = f"{policy_key}.also_taxids.{index}"
        if type(taxid) is not int or taxid <= 0:
            raise _error(path, also_taxid_policy_key, "must be a positive integer")
        if taxid in seen_taxids:
            raise _error(
                path,
                also_taxid_policy_key,
                f"duplicates NCBI Taxonomy ID {taxid}",
            )
        seen_taxids.add(taxid)
        also_taxids.append(taxid)

    return Pathogen(
        key=pathogen_key,
        scientific_name=scientific_name,
        ncbi_taxid=ncbi_taxid,
        rank=cast(PathogenRank, rank),
        group=pathogen_group_key,
        also_taxids=tuple(also_taxids),
    )


def load_pathogen_registry(path: Path = PATHOGENS_YAML) -> PathogenRegistry:
    """Load the versioned target-pathogen registry from ``path``."""
    source = Path(path)
    registry_policy = load_policy_mapping(source)
    schema_version = registry_policy.get("schema_version")
    if type(schema_version) is not int or schema_version != _SUPPORTED_SCHEMA_VERSION:
        detail = (
            f"is required and must be {_SUPPORTED_SCHEMA_VERSION}"
            if schema_version is None
            else f"must be {_SUPPORTED_SCHEMA_VERSION}"
        )
        raise _error(source, "schema_version", detail)

    target_pathogens: dict[str, Pathogen] = {}
    pathogen_groups: dict[str, tuple[str, ...]] = {}
    taxid_owners: dict[int, str] = {}

    def register(pathogen: Pathogen, policy_key: str) -> None:
        if pathogen.key in target_pathogens or pathogen.key in pathogen_groups:
            raise _error(source, policy_key, f"pathogen key {pathogen.key!r} collides")
        for taxid in pathogen.taxids:
            owner = taxid_owners.get(taxid)
            if owner is not None:
                taxid_key = (
                    f"{policy_key}.ncbi_taxid"
                    if taxid == pathogen.ncbi_taxid
                    else f"{policy_key}.also_taxids"
                )
                raise _error(
                    source,
                    taxid_key,
                    f"NCBI Taxonomy ID {taxid} is already assigned to {owner!r}",
                )
            taxid_owners[taxid] = pathogen.key
        target_pathogens[pathogen.key] = pathogen

    for registry_entry_key, registry_entry_policy in registry_policy.items():
        if registry_entry_key == "schema_version":
            continue
        registry_identifier = _validate_identifier(source, registry_entry_key, "<key>")
        if not isinstance(registry_entry_policy, Mapping):
            raise _error(source, registry_identifier, "must be a mapping")
        if registry_entry_policy and _is_target_pathogen(registry_entry_policy):
            register(
                _parse_target_pathogen(
                    source,
                    registry_identifier,
                    registry_entry_policy,
                    pathogen_group_key=None,
                    policy_key=registry_identifier,
                ),
                registry_identifier,
            )
            continue

        if not registry_entry_policy:
            raise _error(
                source,
                registry_identifier,
                "pathogen group must not be empty",
            )
        if registry_identifier in target_pathogens:
            raise _error(
                source,
                registry_identifier,
                f"pathogen group key {registry_identifier!r} collides",
            )
        group_pathogen_keys: list[str] = []
        for pathogen_key, target_pathogen_policy in registry_entry_policy.items():
            child_key = _validate_identifier(
                source,
                pathogen_key,
                f"{registry_identifier}.<key>",
            )
            child_policy_key = f"{registry_identifier}.{child_key}"
            if not isinstance(target_pathogen_policy, Mapping):
                raise _error(source, child_policy_key, "must be a mapping")
            register(
                _parse_target_pathogen(
                    source,
                    child_key,
                    target_pathogen_policy,
                    pathogen_group_key=registry_identifier,
                    policy_key=child_policy_key,
                ),
                child_policy_key,
            )
            group_pathogen_keys.append(child_key)
        if registry_identifier in target_pathogens:
            raise _error(
                source,
                registry_identifier,
                f"pathogen group key {registry_identifier!r} collides with a pathogen key",
            )
        pathogen_groups[registry_identifier] = tuple(group_pathogen_keys)

    return PathogenRegistry(
        schema_version=_SUPPORTED_SCHEMA_VERSION,
        target_pathogens=target_pathogens,
        pathogen_groups=pathogen_groups,
    )


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(
        description="Emit the pathogen registry as a flat pathogen-key table."
    )
    ap.add_argument(
        "--pathogen-keys",
        action="store_true",
        help="emit pathogen_key/taxids/rank/group TSV",
    )
    ap.parse_args()
    registry = load_pathogen_registry()
    sys.stdout.reconfigure(newline="\n")
    sys.stdout.write(registry.pathogen_key_table() + "\n")
