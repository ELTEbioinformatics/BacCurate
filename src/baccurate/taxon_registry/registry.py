"""Versioned target-taxon registry policy."""

import json
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Literal, cast

from baccurate.adapters.policy_yaml import PolicyConfigurationError, load_policy_mapping
from baccurate.paths import TAXA_YAML

TaxonRegistryError = PolicyConfigurationError

_SUPPORTED_SCHEMA_VERSION = 1
type TaxonRank = Literal["genus", "species"]

_SUPPORTED_RANKS: frozenset[TaxonRank] = frozenset({"genus", "species"})
_TAXON_POLICY_KEYS = frozenset({"scientific_name", "ncbi_taxid", "rank", "also_taxids"})


@dataclass(frozen=True)
class Taxon:
    """One taxon (a leaf entry in the registry)."""

    key: str
    scientific_name: str
    ncbi_taxid: int
    rank: TaxonRank
    container: str | None = None
    also_taxids: tuple[int, ...] = field(default_factory=tuple)

    @property
    def taxids(self) -> tuple[int, ...]:
        """Primary taxid plus any ``also_taxids``."""
        return (self.ncbi_taxid, *self.also_taxids)


@dataclass(frozen=True, slots=True)
class TaxonRegistry:
    """Immutable, ordered policy defining the taxa that BacCurate includes."""

    schema_version: int
    included_taxa: Mapping[str, Taxon]
    containers: Mapping[str, tuple[str, ...]]

    def __post_init__(self) -> None:
        """Detach registry collections from caller-owned mutable mappings."""
        object.__setattr__(
            self,
            "included_taxa",
            MappingProxyType(dict(self.included_taxa)),
        )
        object.__setattr__(
            self,
            "containers",
            MappingProxyType(
                {
                    container_key: tuple(taxon_keys)
                    for container_key, taxon_keys in self.containers.items()
                }
            ),
        )

    @property
    def taxon_keys(self) -> tuple[str, ...]:
        """Return taxon keys in registry order."""
        return tuple(self.included_taxa)

    @property
    def container_keys(self) -> tuple[str, ...]:
        """Return container keys in registry order."""
        return tuple(self.containers)

    @property
    def keywords(self) -> tuple[str, ...]:
        """Return every command keyword in compatibility order."""
        return (*self.taxon_keys, *self.container_keys)

    def expand(self, registry_identifiers: Sequence[str]) -> tuple[str, ...]:
        """Expand containers with first-seen taxon-key deduplication."""
        expanded: list[str] = []
        for registry_identifier in registry_identifiers:
            for taxon_key in self.containers.get(
                registry_identifier,
                (registry_identifier,),
            ):
                if taxon_key not in expanded:
                    expanded.append(taxon_key)
        return tuple(expanded)

    def scientific_name(self, taxon_key: str) -> str:
        """Return a taxon's scientific name, or an empty string if unknown."""
        taxon = self.included_taxa.get(taxon_key)
        return taxon.scientific_name if taxon else ""

    def taxid_pairs(self) -> tuple[tuple[str, int], ...]:
        """Enumerate taxon keys paired with each of their ordered NCBI taxids."""
        return tuple(
            (taxon.key, taxid) for taxon in self.included_taxa.values() for taxid in taxon.taxids
        )

    def taxon_key_table(self) -> str:
        """Serialize the flattened taxon-key table used by preparation tooling."""
        lines = ["taxon_key\ttaxids\trank\tcontainer"]
        for taxon in self.included_taxa.values():
            taxids = " ".join(str(taxid) for taxid in taxon.taxids)
            lines.append(f"{taxon.key}\t{taxids}\t{taxon.rank}\t{taxon.container or ''}")
        return "\n".join(lines)

    def serialize(self) -> str:
        """Return deterministic, schema-versioned JSON preserving policy order."""
        return json.dumps(
            {
                "schema_version": self.schema_version,
                "included_taxa": [
                    {
                        "key": taxon.key,
                        "scientific_name": taxon.scientific_name,
                        "ncbi_taxid": taxon.ncbi_taxid,
                        "rank": taxon.rank,
                        "container": taxon.container,
                        "also_taxids": list(taxon.also_taxids),
                    }
                    for taxon in self.included_taxa.values()
                ],
                "containers": [
                    {
                        "key": container_key,
                        "taxon_keys": list(taxon_keys),
                    }
                    for container_key, taxon_keys in self.containers.items()
                ],
            },
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )


def _error(path: Path, policy_key: str, message: str) -> TaxonRegistryError:
    return TaxonRegistryError(f"{path}: {policy_key}: {message}")


def _validate_identifier(path: Path, registry_identifier: object, policy_key: str) -> str:
    if not isinstance(registry_identifier, str) or not registry_identifier.strip():
        raise _error(path, policy_key, "must be a non-empty string")
    return registry_identifier


def _is_taxon(registry_entry_policy: Mapping[object, object]) -> bool:
    return not all(
        isinstance(nested_policy, Mapping) for nested_policy in registry_entry_policy.values()
    )


def _parse_taxon(
    path: Path,
    taxon_key: str,
    taxon_policy: Mapping[object, object],
    container_key: str | None,
    policy_key: str,
) -> Taxon:
    unknown_keys = [
        policy_key for policy_key in taxon_policy if policy_key not in _TAXON_POLICY_KEYS
    ]
    if unknown_keys:
        raise _error(path, f"{policy_key}.{unknown_keys[0]}", "unknown policy key")

    scientific_name = taxon_policy.get("scientific_name")
    if not isinstance(scientific_name, str) or not scientific_name.strip():
        raise _error(
            path,
            f"{policy_key}.scientific_name",
            "must be a non-empty string",
        )

    ncbi_taxid = taxon_policy.get("ncbi_taxid")
    if type(ncbi_taxid) is not int or ncbi_taxid <= 0:
        raise _error(path, f"{policy_key}.ncbi_taxid", "must be a positive integer")

    rank = taxon_policy.get("rank")
    if not isinstance(rank, str) or rank not in _SUPPORTED_RANKS:
        raise _error(
            path,
            f"{policy_key}.rank",
            f"must be one of {sorted(_SUPPORTED_RANKS)}",
        )

    raw_also_taxids = taxon_policy.get("also_taxids", [])
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

    return Taxon(
        key=taxon_key,
        scientific_name=scientific_name,
        ncbi_taxid=ncbi_taxid,
        rank=cast(TaxonRank, rank),
        container=container_key,
        also_taxids=tuple(also_taxids),
    )


def load_taxon_registry(path: Path = TAXA_YAML) -> TaxonRegistry:
    """Load the versioned target-taxon registry from ``path``."""
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

    included_taxa: dict[str, Taxon] = {}
    containers: dict[str, tuple[str, ...]] = {}
    taxid_owners: dict[int, str] = {}

    def register(taxon: Taxon, policy_key: str) -> None:
        if taxon.key in included_taxa or taxon.key in containers:
            raise _error(source, policy_key, f"taxon key {taxon.key!r} collides")
        for taxid in taxon.taxids:
            owner = taxid_owners.get(taxid)
            if owner is not None:
                taxid_key = (
                    f"{policy_key}.ncbi_taxid"
                    if taxid == taxon.ncbi_taxid
                    else f"{policy_key}.also_taxids"
                )
                raise _error(
                    source,
                    taxid_key,
                    f"NCBI Taxonomy ID {taxid} is already assigned to {owner!r}",
                )
            taxid_owners[taxid] = taxon.key
        included_taxa[taxon.key] = taxon

    for registry_entry_key, registry_entry_policy in registry_policy.items():
        if registry_entry_key == "schema_version":
            continue
        registry_identifier = _validate_identifier(source, registry_entry_key, "<key>")
        if not isinstance(registry_entry_policy, Mapping):
            raise _error(source, registry_identifier, "must be a mapping")
        if registry_entry_policy and _is_taxon(registry_entry_policy):
            register(
                _parse_taxon(
                    source,
                    registry_identifier,
                    registry_entry_policy,
                    container_key=None,
                    policy_key=registry_identifier,
                ),
                registry_identifier,
            )
            continue

        if not registry_entry_policy:
            raise _error(
                source,
                registry_identifier,
                "container must not be empty",
            )
        if registry_identifier in included_taxa:
            raise _error(
                source,
                registry_identifier,
                f"container key {registry_identifier!r} collides",
            )
        container_taxon_keys: list[str] = []
        for taxon_key, taxon_policy in registry_entry_policy.items():
            child_key = _validate_identifier(
                source,
                taxon_key,
                f"{registry_identifier}.<key>",
            )
            child_policy_key = f"{registry_identifier}.{child_key}"
            if not isinstance(taxon_policy, Mapping):
                raise _error(source, child_policy_key, "must be a mapping")
            register(
                _parse_taxon(
                    source,
                    child_key,
                    taxon_policy,
                    container_key=registry_identifier,
                    policy_key=child_policy_key,
                ),
                child_policy_key,
            )
            container_taxon_keys.append(child_key)
        if registry_identifier in included_taxa:
            raise _error(
                source,
                registry_identifier,
                f"container key {registry_identifier!r} collides with a taxon key",
            )
        containers[registry_identifier] = tuple(container_taxon_keys)

    return TaxonRegistry(
        schema_version=_SUPPORTED_SCHEMA_VERSION,
        included_taxa=included_taxa,
        containers=containers,
    )


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Emit the taxon registry as a flat taxon-key table.")
    ap.add_argument(
        "--taxon-keys",
        action="store_true",
        help="emit taxon_key/taxids/rank/container TSV",
    )
    ap.parse_args()
    registry = load_taxon_registry()
    sys.stdout.reconfigure(newline="\n")
    sys.stdout.write(registry.taxon_key_table() + "\n")
