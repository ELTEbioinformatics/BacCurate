"""BioSample XML metadata extraction stage."""

from baccurate.extraction.cli import main
from baccurate.extraction.metadata_types import (
    ATTRIBUTES,
    DATE_OTHER,
    DATE_SAMPLING,
    AttributeMatch,
)
from baccurate.extraction.policy import ExtractionPolicy, PolicyDecision, PolicyEvent
from baccurate.utils.xml import CandidateCounters

__all__ = [
    "ATTRIBUTES",
    "DATE_OTHER",
    "DATE_SAMPLING",
    "AttributeMatch",
    "CandidateCounters",
    "ExtractionPolicy",
    "PolicyDecision",
    "PolicyEvent",
    "main",
]
