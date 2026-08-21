"""Build rows for extracted_metadata.tsv."""

from collections.abc import Iterable

from baccurate.extraction.curation import CurationDecision
from baccurate.extraction.io import SEQUENCE_ACCESSION_COLUMNS, TaxonAssignment
from baccurate.extraction.metadata_types import EXTRACTION_TARGET_ORDER

COLUMNS = [
    "accession",
    "bioproject_accession",
    *SEQUENCE_ACCESSION_COLUMNS,
    "taxon_key",
    "ncbi_organism",
    "sylph_species",
    "biosample_last_update",
    "date_category",
] + [f"{target}_{kind}_orig" for target in EXTRACTION_TARGET_ORDER for kind in ("attr", "val")]

# The numeric BioProject IDs a BioSample links to exist only to look up their accessions in the
# BioProject XML, which happens after all BioSample rows are written to a temporary file. They
# are written alongside the row but dropped from the published table.
INTERMEDIATE_COLUMNS = ["bioproject_id", *COLUMNS]


def extracted_metadata_row(
    *,
    accession: str,
    assignment: TaxonAssignment,
    bioproject_id: str,
    bioproject_accession: str,
    decisions: Iterable[CurationDecision],
) -> list[str] | None:
    """Return one intermediate extracted-metadata row for a BioSample record, or None."""
    raw_pairs: dict[str, tuple[list[str], list[str]]] = {
        target: ([], []) for target in EXTRACTION_TARGET_ORDER
    }
    date_categories: list[str] = []
    biosample_last_update = ""
    found = False

    for decision in decisions:
        if decision.xml_element == "biosample_root" and decision.attribute == "last_update":
            biosample_last_update = decision.value
        for match in decision.matches:
            found = True
            attributes, values = raw_pairs[match.target]
            attributes.append(decision.attribute or "")
            values.append(decision.value)
            if match.target == "date":
                date_categories.append(match.category)

    if not found:
        return None

    extracted_metadata_values = [
        bioproject_id,
        accession,
        bioproject_accession,
        *assignment.sequence_accessions,
        assignment.taxon_key,
        assignment.ncbi_organism,
        assignment.sylph_species,
        biosample_last_update,
        "||".join(date_categories),
    ]
    for target in EXTRACTION_TARGET_ORDER:
        attributes, values = raw_pairs[target]
        extracted_metadata_values.extend(("||".join(attributes), "||".join(values)))
    return extracted_metadata_values
