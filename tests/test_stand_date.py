"""Tests for collection date standardization."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from baccurate.source_snapshot import SourceSnapshotManifest
from baccurate.standardizers._date_record import (
    DateBounds,
    DateDiagnostic,
    DateOrigin,
    DateOutcome,
    RecordDateStandardizer,
)

SOURCE_MANIFEST_PATH = Path(__file__).parent.parent / "config" / "source_snapshot.yaml"


@pytest.fixture
def standardizer() -> RecordDateStandardizer:
    manifest = SourceSnapshotManifest.load(SOURCE_MANIFEST_PATH)
    return RecordDateStandardizer(manifest.metadata_reference_date)


def standardize_record(
    standardizer: RecordDateStandardizer,
    *,
    accession: str = "TEST",
    attributes: str = "collection_date",
    values: str,
    categories: str = "s",
) -> DateOutcome | None:
    return standardizer.standardize(
        {
            "accession": accession,
            "date_attr_orig": attributes,
            "date_val_orig": values,
            "date_category": categories,
        }
    )


def standardize_date_value(
    standardizer: RecordDateStandardizer,
    value: str,
    category: str = "s",
) -> DateOutcome | None:
    """Standardize one date value through the typed record interface."""
    return standardize_record(standardizer, values=value, categories=category)


# =============================================================================
# Record input validation
# =============================================================================


def test_record_standardization_rejects_misaligned_date_candidate_counts(standardizer):
    with pytest.raises(
        ValueError,
        match=(r"SAMN_MISALIGNED.*date_attr_orig=2.*date_val_orig=1.*date_category=2"),
    ):
        standardize_record(
            standardizer,
            accession="SAMN_MISALIGNED",
            attributes="collection_date||submission_date",
            values="2020",
            categories="s||o",
        )


def test_record_standardization_rejects_unknown_date_candidate_category(standardizer):
    with pytest.raises(
        ValueError,
        match=r"SAMN_BAD_CATEGORY.*date_category\[0\]='x'.*expected 's' or 'o'",
    ):
        standardize_record(
            standardizer,
            accession="SAMN_BAD_CATEGORY",
            values="2020",
            categories="x",
        )


def test_record_standardization_distinguishes_empty_date_candidate_set(standardizer):
    with pytest.raises(ValueError, match="No date candidates for SAMN_EMPTY"):
        standardize_record(
            standardizer,
            accession="SAMN_EMPTY",
            attributes="",
            values="",
            categories="",
        )


# =============================================================================
# Record-level date selection
# =============================================================================


def test_date_standardizer_receives_source_snapshot_reference_date(standardizer):
    assert standardizer.interpreter.metadata_reference_date == date(2026, 7, 9)


def test_record_standardization_returns_typed_sampling_date_outcome(standardizer):
    outcome = standardize_record(
        standardizer,
        accession="SAMN00000001",
        values="2020-02",
    )

    assert outcome == DateOutcome(
        bounds=DateBounds(date(2020, 2, 1), date(2020, 2, 29), 0.9),
        origins=(DateOrigin("collection_date", "2020-02"),),
    )
    assert standardizer.diagnostic_counts == {DateDiagnostic.SAMPLING_SELECTION: 1}


def test_record_standardization_prefers_valid_sampling_date_over_all_fallbacks(standardizer):
    outcome = standardize_record(
        standardizer,
        accession="SAMN00000002",
        attributes="submission_date||collection_date||publication_date",
        values="2018-01-01||2020||2019-06-15",
        categories="o||s||o",
    )

    assert outcome == DateOutcome(
        DateBounds(date(2020, 1, 1), date(2020, 12, 31), 0.8),
        (DateOrigin("collection_date", "2020"),),
    )


def test_record_standardization_uses_oldest_fallback_when_sampling_dates_are_rejected(
    standardizer,
):
    outcome = standardize_record(
        standardizer,
        accession="SAMN00000003",
        attributes="collection_date||submission_date||publication_date",
        values="2922-08-23||2021-06-15||2019",
        categories="s||o||o",
    )

    assert outcome == DateOutcome(
        DateBounds(date(2019, 1, 1), date(2019, 12, 31), 0.1),
        (DateOrigin("publication_date", "2019"),),
    )
    assert standardizer.diagnostic_counts == {DateDiagnostic.FALLBACK_SELECTION: 1}


def test_sampling_date_takes_precedence_over_other_dates(standardizer):
    outcome = standardize_record(
        standardizer,
        attributes="collection_date||upload_date",
        values="2019||2021-06-15",
        categories="s||o",
    )

    assert outcome.bounds.start == date(2019, 1, 1)
    assert outcome.origins == (DateOrigin("collection_date", "2019"),)


def test_other_date_is_ignored_when_any_sampling_date_is_valid(standardizer):
    """Test for when multiple sampling dates are merged into an envelope,
    unrelated 'other' dates (e.g. upload dates) are dropped.
    """
    outcome = standardize_record(
        standardizer,
        attributes="collection_date||collection_date||upload_date",
        values="2019||2020||2021-06-15",
        categories="s||s||o",
    )

    assert outcome.bounds.start == date(2019, 1, 1)
    assert outcome.bounds.end == date(2020, 12, 31)


def test_other_date_is_used_only_when_no_valid_sampling_date_exists(standardizer):
    """Test for rejecting implausible sampling dates and falling back to non-sampling dates."""
    outcome = standardize_record(
        standardizer,
        attributes="collection_date||upload_date",
        values="2922-08-23||2021-06-15",  # sampling year is a typo
        categories="s||o",
    )

    assert outcome.bounds.start == date(2021, 6, 15)
    assert outcome.bounds.score == 0.1
    assert outcome.origins == (DateOrigin("upload_date", "2021-06-15"),)


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
    outcome = standardize_record(
        standardizer,
        attributes="collection_date||collection_date",
        values="2019||2019",
        categories="s||s",
    )

    assert outcome.bounds == DateBounds(date(2019, 1, 1), date(2019, 12, 31), 0.8)


def test_record_standardization_preserves_and_deduplicates_source_pairs(standardizer):
    outcome = standardize_record(
        standardizer,
        accession="SAMN_PAIRED_DATES",
        attributes="collection_date||sampling_date||collection_date||collection_date",
        values="1993||1993||2009||1993",
        categories="s||s||s||s",
    )

    assert outcome == DateOutcome(
        DateBounds(date(1993, 1, 1), date(2009, 12, 31), 0.2),
        (
            DateOrigin("collection_date", "1993"),
            DateOrigin("sampling_date", "1993"),
            DateOrigin("collection_date", "2009"),
        ),
    )


def test_equivalent_sampling_bounds_keep_first_provenance_and_lowest_score(
    standardizer,
    caplog,
):
    outcome = standardize_record(
        standardizer,
        accession="SAMN_EQUIVALENT_BOUNDS",
        attributes="collection_date||sampling_date",
        values="2019-03-04||04/03/2019",
        categories="s||s",
    )

    assert outcome == DateOutcome(
        DateBounds(date(2019, 3, 4), date(2019, 3, 4), 0.7),
        (DateOrigin("collection_date", "2019-03-04"),),
    )
    assert standardizer.diagnostic_counts == {
        DateDiagnostic.SAMPLING_SELECTION: 1,
        DateDiagnostic.EQUIVALENT_CANDIDATE_COLLAPSE: 1,
    }
    assert caplog.messages == []


def test_record_standardization_counts_conflicting_sampling_date_pairs(standardizer, caplog):
    """Real example: SAMEA1029569 reported collection_date '1993' and '2009'.

    Which is correct is unknowable, so the two are kept as a conflicting
    combination spanning 1993-01-01 to 2009-12-31 (bounds checked in the
    dedup test above).
    """
    standardize_record(
        standardizer,
        accession="SAMN_DATE_CONFLICT",
        attributes="collection_date||collection_date",
        values="1993||2009",
        categories="s||s",
    )

    assert standardizer.diagnostic_counts == {
        DateDiagnostic.SAMPLING_SELECTION: 1,
        DateDiagnostic.CONFLICTING_CANDIDATE_COMBINATION: 1,
    }
    assert caplog.messages == []


def test_explicit_intervals_preserve_record_selection_and_source_pairing(standardizer):
    sampling = standardize_record(
        standardizer,
        accession="SAMN_INTERVAL_SAMPLING",
        attributes="submission_date||collection_date",
        values="2018||Nov-2016/29-May-2017",
        categories="o||s",
    )
    conflict = standardize_record(
        standardizer,
        accession="SAMN_INTERVAL_CONFLICT",
        attributes="collection_date||sampling_date",
        values="Nov-2016/29-May-2017||2018",
        categories="s||s",
    )
    fallback = standardize_record(
        standardizer,
        accession="SAMN_INTERVAL_FALLBACK",
        attributes="submission_date||publication_date",
        values="Nov-2016/29-May-2017||2020",
        categories="o||o",
    )

    assert sampling == DateOutcome(
        DateBounds(date(2016, 11, 1), date(2017, 5, 29), 0.6),
        (DateOrigin("collection_date", "Nov-2016/29-May-2017"),),
    )
    assert conflict == DateOutcome(
        DateBounds(date(2016, 11, 1), date(2018, 12, 31), 0.5),
        (
            DateOrigin("collection_date", "Nov-2016/29-May-2017"),
            DateOrigin("sampling_date", "2018"),
        ),
    )
    assert fallback == DateOutcome(
        DateBounds(date(2016, 11, 1), date(2017, 5, 29), 0.1),
        (DateOrigin("submission_date", "Nov-2016/29-May-2017"),),
    )


# =============================================================================
# Diagnostics and reason reporting
# =============================================================================


def test_record_standardization_reports_deduplicated_details_and_reason_totals(
    standardizer, caplog
):
    outcomes = [
        standardize_record(standardizer, accession=accession, values=value)
        for accession, value in (
            ("SAMN_REJECT_1", "circa 2019"),
            ("SAMN_REJECT_2", "circa 2019"),
            ("SAMN_NOTICE", "2019-03-15T25:99"),
        )
    ]

    assert outcomes[:2] == [None, None]
    assert outcomes[2] is not None
    assert standardizer.rejection_counts == {"unsupported_format": 2}
    assert standardizer.notice_counts == {"malformed_time_suffix_ignored": 1}
    assert standardizer.diagnostic_counts == {
        DateDiagnostic.NO_USABLE_CANDIDATE: 2,
        DateDiagnostic.SAMPLING_SELECTION: 1,
    }
    assert caplog.messages == []


def test_explicit_interval_diagnostics_include_endpoint_context_and_debug_notices(
    standardizer, caplog
):
    assert standardize_date_value(standardizer, "2019 to 2027") is None
    reversed_bounds = standardize_date_value(standardizer, "2020 to 2018").bounds

    assert reversed_bounds == DateBounds(date(2018, 1, 1), date(2020, 12, 31), 0.5)
    assert standardizer.rejection_counts == {"after_metadata_reference_date": 1}
    assert standardizer.notice_counts == {"reversed_interval_normalized": 1}
    assert caplog.messages == []


# =============================================================================
# Value interpretation: precise dates
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
    bounds = standardize_date_value(standardizer, input_string).bounds

    assert bounds.start == date(2019, 3, 15)
    assert bounds.end == date(2019, 3, 15)
    assert bounds.score == 1.0


def test_iso_datetime_with_time_component_is_stripped_to_the_date(standardizer):
    """ISO timestamps are parsed from their date part."""
    bounds = standardize_date_value(standardizer, "2019-03-15T12:00:00Z").bounds

    assert bounds.start == date(2019, 3, 15)
    assert bounds.end == date(2019, 3, 15)
    assert bounds.score == 1.0


# =============================================================================
# Value interpretation: partial dates
# =============================================================================
#
# A partial date like "2019" or "March 2019" is a range of possible
# collection dates, not a single one. The goal is not to invent precision
# that the source didn't provide. The reliability score reflects parse
# confidence, the bound width reflects temporal precision. These are independent.


def test_year_only_date_spans_the_entire_year(standardizer):
    bounds = standardize_date_value(standardizer, "2019").bounds

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
    bounds = standardize_date_value(standardizer, input_string).bounds

    assert bounds.start == date(2019, 3, 1)
    assert bounds.end == date(2019, 3, 31)
    assert bounds.score == 0.9


def test_february_in_a_leap_year_ends_on_the_29th(standardizer):
    """month bounds use the calendar, not a fixed 28/30/31 value."""
    bounds = standardize_date_value(standardizer, "2020-02").bounds

    assert bounds.start == date(2020, 2, 1)
    assert bounds.end == date(2020, 2, 29)


# =============================================================================
# Value interpretation: ambiguous day/month formats
# =============================================================================


@pytest.mark.parametrize("input_string", ["03/03/2019", "03-03-2019"])
def test_equal_numeric_components_resolve_without_ambiguity(standardizer, input_string):
    bounds = standardize_date_value(standardizer, input_string).bounds

    assert bounds.start == date(2019, 3, 3)
    assert bounds.end == date(2019, 3, 3)
    assert bounds.score == 1.0


# =============================================================================
# Value interpretation: intervals
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
    bounds = standardize_date_value(standardizer, f"2018{separator}2020").bounds

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
    bounds = standardize_date_value(standardizer, input_string).bounds
    assert bounds.score == expected_score


def test_interval_collapsing_to_single_date_keeps_pattern_score(standardizer):
    """'2019/2019' is a redundant encoding of '2019', which should not be penalized."""
    bounds = standardize_date_value(standardizer, "2019/2019").bounds

    assert bounds.start == date(2019, 1, 1)
    assert bounds.end == date(2019, 12, 31)
    assert bounds.score == 0.8


def test_interval_with_mixed_precision_uses_each_endpoints_bounds(standardizer):
    bounds = standardize_date_value(standardizer, "2018/2020-06").bounds

    assert bounds.start == date(2018, 1, 1)  # full year on the left
    assert bounds.end == date(2020, 6, 30)  # end of June on the right
    # Span: ~2.5 years
    assert bounds.score == 0.5


def test_reversed_interval_is_widened_to_the_outer_envelope(standardizer):
    """'2020/2018' has the endpoints in the wrong order."""
    bounds = standardize_date_value(standardizer, "2020/2018").bounds

    assert bounds.start == date(2018, 1, 1)
    assert bounds.end == date(2020, 12, 31)


# =============================================================================
# Value interpretation: implausible and unrecognized values
# =============================================================================
#
# The metadata occasionally contains malformed or typo'd dates. The
# standardizer rejects anything outside [1800, current year], and returns no
# result for values it cannot parse at all.


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
    assert standardize_date_value(standardizer, typo) is None, description


def test_unparseable_value_yields_no_result(standardizer):
    """Unrecognized date strings produce no result."""
    assert standardize_date_value(standardizer, "not a date") is None


# =============================================================================
# DateBounds invariants
# =============================================================================


def test_date_bounds_rejects_inverted_start_and_end():
    """Date bounds reject silently corrupt intervals."""
    with pytest.raises(ValueError):
        DateBounds(start=date(2020, 1, 1), end=date(2019, 1, 1), score=1.0)
