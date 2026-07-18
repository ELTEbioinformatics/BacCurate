"""BioSample XML metadata extraction stage."""

from baccurate.extraction.cli import ExtractionReport, main
from baccurate.extraction.curation import (
    CurationDecision,
    CurationEvent,
    CurationSchema,
    CurationSchemaError,
)
from baccurate.extraction.metadata_types import (
    ATTRIBUTES,
    DATE_OTHER,
    DATE_SAMPLING,
    AttributeMatch,
)
from baccurate.extraction.xml import CandidateCounters

__all__ = [
    "ATTRIBUTES",
    "DATE_OTHER",
    "DATE_SAMPLING",
    "AttributeMatch",
    "CandidateCounters",
    "CurationDecision",
    "CurationEvent",
    "CurationSchema",
    "CurationSchemaError",
    "ExtractionReport",
    "main",
]
