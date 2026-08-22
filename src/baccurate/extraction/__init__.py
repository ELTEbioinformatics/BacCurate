"""BioSample XML metadata extraction stage."""

from baccurate.extraction.cli import ExtractionReport, cli, run_extraction
from baccurate.extraction.io import (
    SEQUENCE_ACCESSION_COLUMNS,
    InclusionRoute,
    TaxonAssignment,
    load_taxon_map,
    resolve_taxon_assignment,
)
from baccurate.extraction.metadata_types import TargetMatch
from baccurate.extraction.selection import (
    SelectionDecision,
    SelectionEvent,
    SelectionPolicy,
    SelectionPolicyError,
)
from baccurate.extraction.tables import COLUMNS
from baccurate.extraction.xml import SelectionCounters

__all__ = [
    "COLUMNS",
    "SEQUENCE_ACCESSION_COLUMNS",
    "ExtractionReport",
    "InclusionRoute",
    "SelectionCounters",
    "SelectionDecision",
    "SelectionEvent",
    "SelectionPolicy",
    "SelectionPolicyError",
    "TargetMatch",
    "TaxonAssignment",
    "cli",
    "load_taxon_map",
    "resolve_taxon_assignment",
    "run_extraction",
]
