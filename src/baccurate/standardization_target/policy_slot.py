"""Effective-policy slots and their configuration filenames."""

from collections.abc import Mapping
from enum import StrEnum
from types import MappingProxyType


class PolicySlot(StrEnum):
    """A loadable domain policy slot in the effective policy."""

    CURATION_SCHEMA = "curation_schema"
    HOST = "host"
    LOCATION = "location"
    ISOLATION_SOURCE = "isolation_source"


POLICY_FILENAMES: Mapping[PolicySlot, str] = MappingProxyType(
    {
        PolicySlot.CURATION_SCHEMA: "curation_schema.yaml",
        PolicySlot.HOST: "host.yaml",
        PolicySlot.LOCATION: "location.yaml",
        PolicySlot.ISOLATION_SOURCE: "isolation_source.yaml",
    }
)
