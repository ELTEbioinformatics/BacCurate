"""BioSample XML metadata extraction stage."""

from baccurate.extraction.classifiers import (
    AttributeMatch,
    DateClassifier,
    KeywordClassifier,
    LocationClassifier,
    MetadataClassifier,
    build_metadata_classifier,
)
from baccurate.extraction.cli import main

__all__ = [
    "AttributeMatch",
    "DateClassifier",
    "KeywordClassifier",
    "LocationClassifier",
    "MetadataClassifier",
    "build_metadata_classifier",
    "main",
]
