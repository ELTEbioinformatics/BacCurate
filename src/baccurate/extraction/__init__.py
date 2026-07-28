"""BioSample XML metadata extraction stage."""

from baccurate.extraction.cli import ExtractionReport, cli, run_extraction
from baccurate.extraction.curation import (
    CurationDecision,
    CurationEvent,
    CurationSchema,
    CurationSchemaError,
)
from baccurate.extraction.metadata_types import TargetMatch
from baccurate.extraction.tables import COLUMNS
from baccurate.extraction.xml import CurationCounters

__all__ = [
    "COLUMNS",
    "CurationCounters",
    "CurationDecision",
    "CurationEvent",
    "CurationSchema",
    "CurationSchemaError",
    "ExtractionReport",
    "TargetMatch",
    "cli",
    "run_extraction",
]
