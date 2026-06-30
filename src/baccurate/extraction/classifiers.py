"""Attribute classifiers used during BioSample metadata extraction."""

import re
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Protocol

from baccurate.utils.config import load_config
from baccurate.utils.text import normalize_keyword

# Attribute order here fixes the TSV column layout in tables.py.
ATTRIBUTES = ("date", "iso", "host", "loc")

DATE_SAMPLING = "s"  # collection/sampling date
DATE_OTHER = "o"  # any other date (submission, release, ...)


def _keyword_set(config: dict, key: str) -> set[str]:
    return {normalize_keyword(kw) for kw in config.get(key, [])}


def _within_scope(
    value: str,
    attr_name: str,
    blacklist: set[str],
    null_values: set[str],
) -> bool:
    """True when the value carries information and the attribute name isn't blacklisted."""
    return value not in null_values and attr_name not in blacklist


@dataclass(frozen=True, slots=True)
class AttributeMatch:
    """One hit on an attribute. Category is set only for date hits."""

    attribute: str
    category: str = ""


class AttributeClassifier(Protocol):
    def match(self, value: str, attr_name: str) -> AttributeMatch | None: ...


class KeywordClassifier:
    """Reports whether an attribute matches a keyword-defined attribute."""

    def __init__(self, attribute: str, config: dict) -> None:
        self.attribute = attribute
        self.keywords = _keyword_set(config, "keywords")
        self.blacklist = _keyword_set(config, "blacklist")
        self.null_values = _keyword_set(config, "null_values")

    def match(self, value: str, attr_name: str) -> AttributeMatch | None:
        if not _within_scope(value, attr_name, self.blacklist, self.null_values):
            return None
        if any(kw in attr_name for kw in self.keywords):
            return AttributeMatch(self.attribute)
        return None


class LocationClassifier:
    """Reports whether an attribute denotes a sampling location."""

    def __init__(self, config: dict) -> None:
        self.geo_keywords = _keyword_set(config, "geographic_keywords")
        self.coord_attributes = _keyword_set(config, "coordinate_attributes")
        self.blacklist = _keyword_set(config, "blacklist")
        self.null_values = _keyword_set(config, "null_values")
        patterns = config.get("patterns", {})
        self.pat_coord = re.compile(patterns.get("coordinates", r"^$"))
        self.pat_country_colon = re.compile(patterns.get("country_colon_city", r"^$"))
        self.pat_country_comma = re.compile(patterns.get("country_comma_city", r"^$"))

    def match(self, value: str, attr_name: str) -> AttributeMatch | None:
        if not _within_scope(value, attr_name, self.blacklist, self.null_values):
            return None
        by_attr = (
            any(kw in attr_name for kw in self.geo_keywords) or attr_name in self.coord_attributes
        )
        by_value = (
            self.pat_coord.match(value) is not None
            or self.pat_country_colon.match(value) is not None
            or self.pat_country_comma.match(value) is not None
        )
        return AttributeMatch("loc") if by_attr or by_value else None


class DateClassifier:
    """Classifies an attribute as a sampling date, other date, or neither."""

    def __init__(self, config: dict) -> None:
        self.sampling_keywords = _keyword_set(config, "sampling_keywords")
        self.other_keywords = _keyword_set(config, "other_date_keywords")
        self.blacklist = _keyword_set(config, "blacklist")
        self.null_values = _keyword_set(config, "null_values")

    def match(self, value: str, attr_name: str) -> AttributeMatch | None:
        if not _within_scope(value, attr_name, self.blacklist, self.null_values):
            return None
        if any(kw in attr_name for kw in self.sampling_keywords):
            return AttributeMatch("date", DATE_SAMPLING)
        if any(kw in attr_name for kw in self.other_keywords):
            return AttributeMatch("date", DATE_OTHER)
        return None


class MetadataClassifier:
    """Runs every classifier over one attribute."""

    def __init__(self, classifiers: list[AttributeClassifier]) -> None:
        self.classifiers = classifiers

    @cache
    def classify(self, value: str, attr_name: str | None = None) -> tuple[AttributeMatch, ...]:
        value = normalize_keyword(value)
        attr_name = normalize_keyword(attr_name or "")
        hits = (c.match(value, attr_name) for c in self.classifiers)
        return tuple(hit for hit in hits if hit is not None)


def build_metadata_classifier(config_dir: Path) -> MetadataClassifier:
    """Load the YAML configs and assemble a MetadataClassifier."""
    return MetadataClassifier(
        [
            KeywordClassifier("host", load_config(config_dir / "host.yaml")),
            KeywordClassifier("iso", load_config(config_dir / "isolation_source.yaml")),
            LocationClassifier(load_config(config_dir / "location.yaml")),
            DateClassifier(load_config(config_dir / "date.yaml")),
        ]
    )
