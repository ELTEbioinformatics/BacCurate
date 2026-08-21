"""
Assemble and write the geographic-location review worklist for one run.

The worklist is used for adding approved mappings (or unmapped entries) for the next run.
"""

import csv
import json
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

from baccurate.standardization.location import (
    UnresolvedLocationInput,
    normalize_submitted_location_value,
)

LOCATION_REVIEW_WORKLIST_FILENAME = "location_review_worklist.tsv"

COLUMNS = (
    "normalized_submitted_value",
    "biosample_record_count",
    "occurrence_count",
    "taxon_counts",
    "submitted_attribute_counts",
    "representative_examples",
)

_REPRESENTATIVE_LIMIT = 3

# (biosample_accession, taxon_key, submitted_attribute, submitted_value)
type _Example = tuple[str, str, str, str]


@dataclass(frozen=True, slots=True)
class LocationReviewWorklistSummary:
    """Output path and size of the review worklist written by one run."""

    path: Path
    row_count: int
    occurrence_count: int
    biosample_record_count: int


@dataclass(slots=True)
class _WorklistRow:
    """Counts and examples for one normalized submitted value."""

    accessions: set[str] = field(default_factory=set)
    occurrence_count: int = 0
    taxon_counts: dict[str, int] = field(default_factory=dict)
    submitted_attribute_counts: dict[str, int] = field(default_factory=dict)
    examples: set[_Example] = field(default_factory=set)

    def observe(
        self,
        unresolved: UnresolvedLocationInput,
        *,
        accession: str,
        taxon_key: str,
    ) -> None:
        self.accessions.add(accession)
        self.occurrence_count += 1
        self.taxon_counts[taxon_key] = self.taxon_counts.get(taxon_key, 0) + 1
        self.submitted_attribute_counts[unresolved.attribute] = (
            self.submitted_attribute_counts.get(unresolved.attribute, 0) + 1
        )
        # Keeping the lexically smallest examples makes the set deterministic
        # regardless of input order.
        self.examples.add((accession, taxon_key, unresolved.attribute, unresolved.value))
        if len(self.examples) > _REPRESENTATIVE_LIMIT:
            self.examples.remove(max(self.examples))

    def render_examples(self) -> str:
        return _compact_json(
            [
                {
                    "biosample_accession": accession,
                    "taxon_key": taxon_key,
                    "submitted_attribute": attribute,
                    "submitted_value": value,
                }
                for accession, taxon_key, attribute, value in sorted(self.examples)
            ]
        )


class LocationReviewWorklist:
    """Collect distinct unresolved geographic-location inputs across one run."""

    def __init__(self) -> None:
        self._rows: dict[str, _WorklistRow] = {}

    def observe(
        self,
        unresolved_inputs: Iterable[UnresolvedLocationInput],
        *,
        accession: str,
        taxon_key: str,
    ) -> None:
        """Record one BioSample record's unresolved geographic-location inputs."""
        for unresolved in unresolved_inputs:
            normalized = normalize_submitted_location_value(unresolved.value)
            self._rows.setdefault(normalized, _WorklistRow()).observe(
                unresolved,
                accession=accession,
                taxon_key=taxon_key,
            )

    def write(self, destination: Path) -> LocationReviewWorklistSummary:
        """Write the worklist (header included) and return a summary for the run report."""
        ordered = sorted(
            self._rows.items(),
            key=lambda row: (-len(row[1].accessions), -row[1].occurrence_count, row[0]),
        )
        with destination.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream, delimiter="\t", lineterminator="\n")
            writer.writerow(COLUMNS)
            for normalized_value, row in ordered:
                writer.writerow(
                    (
                        normalized_value,
                        len(row.accessions),
                        row.occurrence_count,
                        _compact_json(dict(sorted(row.taxon_counts.items()))),
                        _compact_json(dict(sorted(row.submitted_attribute_counts.items()))),
                        row.render_examples(),
                    )
                )
        return LocationReviewWorklistSummary(
            path=destination,
            row_count=len(ordered),
            # Records and rows are many-to-many, so count distinct accessions
            # instead of summing per-row.
            biosample_record_count=len(
                {accession for _, row in ordered for accession in row.accessions}
            ),
            occurrence_count=sum(row.occurrence_count for _, row in ordered),
        )


def _compact_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
