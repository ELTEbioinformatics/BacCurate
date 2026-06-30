"""
Tests for baccurate.standardizers.date.

Run with:
uv run pytest tests/test_stand_date.py -v
or
pytest tests/test_stand_date.py -v

"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from baccurate.standardizers.date import DateBounds, DateStandardizer

CONFIG_PATH = Path(__file__).parent.parent / "config" / "date.yaml"


@pytest.fixture
def standardizer() -> DateStandardizer:
    return DateStandardizer(CONFIG_PATH)


def parse(standardizer: DateStandardizer, value: str, category: str = "s"):
    """Run a single date string through the full dispatch logic.

    Returns the (DateBounds, attribute, value) tuple, or None if no usable
    date could be extracted.
    """
    return standardizer.find_best_date(
        accession="TEST",
        attributes_str="collection_date",
        date_values_str=value,
        categories_str=category,
    )


# =============================================================================
# Precise dates with day-level precision
# =============================================================================


@pytest.mark.parametrize(
    "input_string",
    [
        "2019-03-15",  # ISO 8601
        "2019/03/15",  # ISO with slash
        "March 15, 2019",  # US long form
        "15-Mar-2019",  # GenBank-style short form
    ],
)
def test_precise_dates_in_any_format_produce_a_single_day(standardizer, input_string):
    bounds, _, _ = parse(standardizer, input_string)

    assert bounds.start == date(2019, 3, 15)
    assert bounds.end == date(2019, 3, 15)
    assert bounds.score == 1.0


def test_iso_datetime_with_time_component_is_stripped_to_the_date(standardizer):
    """Test if the time component is dropped before parsing."""
    bounds, _, _ = parse(standardizer, "2019-03-15T12:00:00Z")

    assert bounds.start == date(2019, 3, 15)
    assert bounds.end == date(2019, 3, 15)
    assert bounds.score == 1.0


# =============================================================================
# Partial dates
# =============================================================================
#
# A partial date like "2019" or "March 2019" is a range of possible
# collection dates, not a single one. The goal is not to invent precision
# that the source didn't provide. The reliability score reflects parse
# confidence, the bound width reflects temporal precision. These are independent.


def test_year_only_date_spans_the_entire_year(standardizer):
    bounds, _, _ = parse(standardizer, "2019")

    assert bounds.start == date(2019, 1, 1)
    assert bounds.end == date(2019, 12, 31)
    assert bounds.score == 0.8


@pytest.mark.parametrize(
    "input_string",
    [
        "2019-03",
        "2019/03",
        "03-2019",
        "03/2019",
        "March 2019",
    ],
)
def test_year_month_date_spans_the_entire_month(standardizer, input_string):
    bounds, _, _ = parse(standardizer, input_string)

    assert bounds.start == date(2019, 3, 1)
    assert bounds.end == date(2019, 3, 31)
    assert bounds.score == 0.9


def test_february_in_a_leap_year_ends_on_the_29th(standardizer):
    """month bounds use the calendar, not a fixed 28/30/31 value."""
    bounds, _, _ = parse(standardizer, "2020-02")

    assert bounds.start == date(2020, 2, 1)
    assert bounds.end == date(2020, 2, 29)


# =============================================================================
# Ambiguous day/month formats
# =============================================================================


@pytest.mark.parametrize("input_string", ["03/03/2019", "03-03-2019"])
def test_ambiguous_day_month_format_assumed_to_be_day_first(standardizer, input_string):
    """'03/03/2019' and '03-03-2019' are interpreted as 3 March 2019, flagged with score 0.7."""
    bounds, _, _ = parse(standardizer, input_string)

    assert bounds.start == date(2019, 3, 3)
    assert bounds.end == date(2019, 3, 3)
    assert bounds.score == 0.7


# =============================================================================
# Intervals
# =============================================================================
#
# E.g. "2018/2020". We expand each endpoint to its own bounds and take
# the outer envelope: earliest possible start to latest possible end.

# Interval scores are based on span width. A wider
# possible-collection-date window means we know less about the actual
# collection time, so the score drops accordingly:
#     <= 1 year   -> 0.6
#     <= 3 years  -> 0.5
#     <= 6 years  -> 0.4
#     <= 10 years -> 0.3
#     >  10 years -> 0.2


@pytest.mark.parametrize("separator", ["/", " to ", " - "])
def test_interval_separators_all_produce_the_same_envelope(standardizer, separator):
    bounds, _, _ = parse(standardizer, f"2018{separator}2020")

    assert bounds.start == date(2018, 1, 1)
    assert bounds.end == date(2020, 12, 31)
    assert bounds.score == 0.5


@pytest.mark.parametrize(
    "input_string,expected_score",
    [
        ("2019-06-01/2019-12-01", 0.6),  # ~6 months - within 1 year
        ("2018/2020", 0.5),  # ~3 years
        ("2015/2020", 0.4),  # ~6 years
        ("2011/2020", 0.3),  # ~10 years
        ("1990/2020", 0.2),  # > 10 years
    ],
)
def test_interval_score_decreases_with_span_width(
    standardizer,
    input_string,
    expected_score,
):
    bounds, _, _ = parse(standardizer, input_string)
    assert bounds.score == expected_score


def test_interval_collapsing_to_single_date_keeps_pattern_score(standardizer):
    """'2019/2019' is a redundant encoding of '2019', which should not be penalized."""
    bounds, _, _ = parse(standardizer, "2019/2019")

    assert bounds.start == date(2019, 1, 1)
    assert bounds.end == date(2019, 12, 31)
    assert bounds.score == 0.8


def test_interval_with_mixed_precision_uses_each_endpoints_bounds(standardizer):
    bounds, _, _ = parse(standardizer, "2018/2020-06")

    assert bounds.start == date(2018, 1, 1)  # full year on the left
    assert bounds.end == date(2020, 6, 30)  # end of June on the right
    # Span: ~2.5 years
    assert bounds.score == 0.5


def test_reversed_interval_is_widened_to_the_outer_envelope(standardizer):
    """'2020/2018' has the endpoints in the wrong order."""
    bounds, _, _ = parse(standardizer, "2020/2018")

    assert bounds.start == date(2018, 1, 1)
    assert bounds.end == date(2020, 12, 31)


# =============================================================================
# Implausible dates
# =============================================================================
#
# The metadata metadata occasionally contains malformed or typo'd dates. The
# standardizer rejects anything outside [1800, current year].
# Some observed cases:


@pytest.mark.parametrize(
    "typo,description",
    [
        ("1799", "year too far in the past for modern microbiological samples"),
        ("2922-08-23", "year too far in the future. Probably typo for 2022"),
    ],
)
def test_typo_dates_are_rejected_rather_than_returning_a_garbage_date(
    standardizer,
    typo,
    description,
):
    assert parse(standardizer, typo) is None, description


# =============================================================================
# Multi-value records
# =============================================================================
#
# Some records list the collection date in multiple metadata fields.
# Three cases:
#   1. All values parse to the same bounds -> collapse silently.
#   2. Values differ but all are valid -> treat as an implicit interval.
#   3. Some values are invalid -> use the valid ones, ignore the rest.


def test_duplicate_sampling_dates_collapse_to_a_single_entry(standardizer):
    """When the same date appears in multiple fields, no special handling needed."""
    result = standardizer.find_best_date(
        accession="TEST",
        attributes_str="collection_date||collection_date",
        date_values_str="2019||2019",
        categories_str="s||s",
    )
    bounds, _, _ = result

    assert bounds.start == date(2019, 1, 1)
    assert bounds.end == date(2019, 12, 31)
    assert bounds.score == 0.8


def test_distinct_sampling_dates_become_an_implicit_interval(standardizer):
    """Real example: SAMEA1029569 had collection_date '1993' and '2009'.

    Hard to know which is correct, so we span the full range from the earliest
    possible date in 1993 to the latest possible date in 2009.
    The score is based on the span width.
    """
    result = standardizer.find_best_date(
        accession="SAMEA1029569",
        attributes_str="collection_date||collection_date",
        date_values_str="1993||2009",
        categories_str="s||s",
    )
    bounds, _, _ = result

    assert bounds.start == date(1993, 1, 1)
    assert bounds.end == date(2009, 12, 31)
    assert bounds.score == 0.2


# =============================================================================
# Sampling vs "other" dates - sampling should always win when valid
# =============================================================================


def test_sampling_date_takes_precedence_over_other_dates(standardizer):
    result = standardizer.find_best_date(
        accession="TEST",
        attributes_str="collection_date||upload_date",
        date_values_str="2019||2021-06-15",
        categories_str="s||o",
    )
    bounds, attribute, _ = result

    assert bounds.start == date(2019, 1, 1)
    assert attribute == "collection_date"


def test_other_date_is_ignored_when_any_sampling_date_is_valid(standardizer):
    """Test for when multiple sampling dates are merged into an envelope,
    unrelated 'other' dates (e.g. upload dates) are dropped.
    """
    result = standardizer.find_best_date(
        accession="TEST",
        attributes_str="collection_date||collection_date||upload_date",
        date_values_str="2019||2020||2021-06-15",
        categories_str="s||s||o",
    )
    bounds, _, _ = result

    assert bounds.start == date(2019, 1, 1)
    assert bounds.end == date(2020, 12, 31)


def test_other_date_is_used_only_when_no_valid_sampling_date_exists(standardizer):
    """Test for rejecting implausible sampling dates and falling back to non-sampling dates."""
    result = standardizer.find_best_date(
        accession="TEST",
        attributes_str="collection_date||upload_date",
        date_values_str="2922-08-23||2021-06-15",  # sampling year is a typo
        categories_str="s||o",
    )
    bounds, attribute, _ = result

    assert bounds.start == date(2021, 6, 15)
    assert bounds.score == 0.1
    assert attribute == "upload_date"


# =============================================================================
# No-result and invariant cases
# =============================================================================


def test_unparseable_value_yields_no_result(standardizer):
    """Test if a garbage string with no recognizable date format returns None."""
    assert parse(standardizer, "not a date") is None


def test_date_bounds_rejects_inverted_start_and_end():
    """Test if start <= end is enforced, so downstream code never sees a
    silently corrupt interval.
    """
    with pytest.raises(ValueError):
        DateBounds(start=date(2020, 1, 1), end=date(2019, 1, 1), score=1.0)
