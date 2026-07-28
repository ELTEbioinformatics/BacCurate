"""Shared supporting attribute-value pair for standardized outcomes."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class SupportingAttributeValuePair:
    """Answer which submitted pair supports an emitted outcome, unlike HostOverflowContext."""

    attribute: str
    value: str
