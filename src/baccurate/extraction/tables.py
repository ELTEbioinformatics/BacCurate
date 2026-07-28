"""Build rows for extracted_metadata.tsv."""

from collections.abc import Iterable

from baccurate.extraction.curation import CurationDecision
from baccurate.extraction.metadata_types import EXTRACTION_TARGET_ORDER

COLUMNS = [
    "accession",
    "bioproject_id",
    "bioproject_accession",
    "pathogen",
    "date_category",
] + [f"{target}_{kind}_orig" for target in EXTRACTION_TARGET_ORDER for kind in ("attr", "val")]


def extracted_metadata_row(
    *,
    accession: str,
    pathogen: str,
    bioproject_id: str,
    bioproject_accession: str,
    decisions: Iterable[CurationDecision],
) -> list[str] | None:
    """Return one extracted metadata row for one BioSample record, or None."""
    raw_pairs: dict[str, tuple[list[str], list[str]]] = {
        target: ([], []) for target in EXTRACTION_TARGET_ORDER
    }
    date_categories: list[str] = []
    found = False

    for decision in decisions:
        for match in decision.matches:
            found = True
            attributes, values = raw_pairs[match.target]
            attributes.append(decision.attribute or "")
            values.append(decision.value)
            if match.target == "date":
                date_categories.append(match.category)

    # Keep rows that carry only a BioProject link, so unresolved-only samples
    # stay in the dataset, distinct from fully unlinked ones that we drop.
    if not found and not bioproject_id:
        return None

    extracted_metadata_values = [
        accession,
        bioproject_id,
        bioproject_accession,
        pathogen,
        "||".join(date_categories),
    ]
    for target in EXTRACTION_TARGET_ORDER:
        attributes, values = raw_pairs[target]
        extracted_metadata_values.extend(("||".join(attributes), "||".join(values)))
    return extracted_metadata_values
