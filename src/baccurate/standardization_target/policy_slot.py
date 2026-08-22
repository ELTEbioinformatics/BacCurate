"""Effective-policy slots and their configuration filenames."""

from collections.abc import Mapping
from enum import StrEnum
from types import MappingProxyType


class PolicySlot(StrEnum):
    """A loadable domain policy slot in the effective policy."""

    SELECTION = "selection"
    HOST = "host"
    LOCATION = "location"
    ISOLATION_SOURCE = "isolation_source"


POLICY_FILENAMES: Mapping[PolicySlot, str] = MappingProxyType(
    {
        PolicySlot.SELECTION: "selection.yaml",
        PolicySlot.HOST: "host.yaml",
        PolicySlot.LOCATION: "location.yaml",
        PolicySlot.ISOLATION_SOURCE: "isolation_source.yaml",
    }
)
