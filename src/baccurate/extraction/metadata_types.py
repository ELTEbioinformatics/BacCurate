"""Types for metadata extraction."""

from dataclasses import dataclass

# This order fixes the extracted TSV column layout in tables.py
ATTRIBUTES = ("date", "iso", "host", "loc")

DATE_SAMPLING = "s"
DATE_OTHER = "o"


@dataclass(frozen=True, slots=True)
class AttributeMatch:
    """One candidate identification. Category is used only for date matches."""

    target: str
    category: str = ""
