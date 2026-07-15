"""
Tests for baccurate.standardizers.location

Run with:
uv run pytest tests/test_stand_loc.py -v
or
pytest tests/test_stand_loc.py -v
"""

from __future__ import annotations

import gc
import weakref
from pathlib import Path

import pytest

from baccurate.standardizers.location import LocationMatch, LocationStandardizer

ROOT = Path(__file__).parent.parent
CONFIG_PATH = ROOT / "config" / "location.yaml"


@pytest.fixture(scope="session")
def standardizer() -> LocationStandardizer:
    return LocationStandardizer(CONFIG_PATH)


# =============================================================================
# INSDC remap + (_to_insdc)
# =============================================================================

def test_na_country_is_left_untouched(standardizer):
    result = standardizer._to_insdc(LocationMatch("NA", "NA", None))

    assert result.country == "NA"


def test_non_insdc_country_is_dropped_to_na(standardizer):
    result = standardizer._to_insdc(LocationMatch("Vatican", "Europe", None))

    assert result.country == "NA"
    assert result.continent == "NA"


def test_coco_false_positive_is_dropped_to_na(standardizer):
    result = standardizer._to_insdc(LocationMatch("water", "NA", None))

    assert result.country == "NA"


# =============================================================================
# find_best_location
# =============================================================================


def test_united_states_resolves_to_insdc_usa(standardizer):
    match = standardizer.find_best_location("TEST", "geo_loc_name", "United States")

    assert match.country == "USA"


def test_vietnam_resolves_to_insdc_viet_nam(standardizer):
    match = standardizer.find_best_location("TEST", "geo_loc_name", "Vietnam")

    assert match.country == "Viet Nam"


def test_conformant_country_unchanged_end_to_end(standardizer):
    match = standardizer.find_best_location("TEST", "geo_loc_name", "Germany")

    assert match.country == "Germany"


def test_country_colon_city_keeps_sublocation_after_remap(standardizer):
    match = standardizer.find_best_location("TEST", "geo_loc_name", "United States: Boston")

    assert match.country == "USA"
    assert match.sublocation == "Boston"


def test_country_conversion_cache_does_not_retain_standardizer():
    standardizer = LocationStandardizer(CONFIG_PATH)
    standardizer.find_best_location("TEST", "geo_loc_name", "Germany")
    standardizer_ref = weakref.ref(standardizer)

    del standardizer
    gc.collect()

    assert standardizer_ref() is None
