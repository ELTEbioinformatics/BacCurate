"""Build rows for extracted_metadata.tsv."""

from collections.abc import Iterable

from baccurate.extraction.metadata_types import ATTRIBUTES
from baccurate.extraction.policy import PolicyDecision

COLUMNS = ["accession", "bioproject", "pathogen", "package", "date_category"] + [
    f"{attribute}_{kind}_orig" for attribute in ATTRIBUTES for kind in ("attr", "val")
]

def record_row(
    *,
    accession: str,
    pathogen: str,
    package: str,
    bioproject: str,
    candidates: Iterable[PolicyDecision],
) -> list[str] | None:
    """Return one output row for one record, or None"""
    raw_pairs: dict[str, tuple[list[str], list[str]]] = {
        target: ([], []) for target in ATTRIBUTES
    }
    date_categories: list[str] = []
    found = False

    for decision in candidates:
        for match in decision.matches:
            found = True
            attributes, values = raw_pairs[match.target]
            attributes.append(decision.attribute or "")
            values.append(decision.value)
            if match.target == "date":
                date_categories.append(match.category)

    if not found:
        return None

    row = [accession, bioproject, pathogen, package, "||".join(date_categories)]
    for target in ATTRIBUTES:
        attributes, values = raw_pairs[target]
        row.extend(("||".join(attributes), "||".join(values)))
    return row
