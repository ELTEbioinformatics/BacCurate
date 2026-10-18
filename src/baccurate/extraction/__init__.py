"""BioSample XML metadata extraction stage."""

from baccurate.extraction.extract import ExtractionReport, run_extraction
from baccurate.extraction.io import (
    SEQUENCE_ACCESSION_COLUMNS,
    InclusionRoute,
    TaxonAssignment,
    load_taxon_map,
    resolve_taxon_assignment,
)
from baccurate.extraction.metadata_types import TargetMatch
from baccurate.extraction.selection import (
    SelectionConfig,
    SelectionConfigError,
    SelectionDecision,
    SelectionEvent,
)
from baccurate.extraction.tables import COLUMNS
from baccurate.extraction.xml import SelectionCounters

__all__ = [
    "COLUMNS",
    "SEQUENCE_ACCESSION_COLUMNS",
    "ExtractionReport",
    "InclusionRoute",
    "SelectionConfig",
    "SelectionConfigError",
    "SelectionCounters",
    "SelectionDecision",
    "SelectionEvent",
    "TargetMatch",
    "TaxonAssignment",
    "load_taxon_map",
    "resolve_taxon_assignment",
    "run_extraction",
]
