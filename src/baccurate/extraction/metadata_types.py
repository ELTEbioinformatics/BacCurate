"""Types for metadata extraction."""

from dataclasses import dataclass

# This order fixes the extracted TSV column layout in tables.py
EXTRACTION_TARGET_ORDER = ("date", "iso", "host", "loc")


@dataclass(frozen=True, slots=True)
class TargetMatch:
    """One standardization target this raw pair supplies. Category is used only for date matches."""

    target: str
    category: str = ""
    # True when the `loc` target is identified by value-search only.
    matched_by_value: bool = False
