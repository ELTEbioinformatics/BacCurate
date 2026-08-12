"""Pin the collection-date standardization contract."""

from __future__ import annotations

from datetime import date

import pytest

from baccurate.standardization.collection_date import (
    DateBounds,
    DateCategory,
    DateDiagnostic,
    DateOutcome,
    DatePrecision,
    DateStructure,
    RecordDateStandardizer,
)
from baccurate.standardization.supporting_attribute_value_pair import SupportingAttributeValuePair


@pytest.fixture
def standardizer() -> RecordDateStandardizer:
    return RecordDateStandardizer(metadata_reference_date=date(2026, 1, 1))


def standardize_record(
    standardizer: RecordDateStandardizer,
    *,
    accession: str = "TEST",
    attributes: str = "collection_date",
    values: str,
    categories: str = "c",
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
    category: str = "c",
) -> DateOutcome | None:
    """Standardize one date value through the typed record interface."""
    return standardize_record(standardizer, values=value, categories=category)


# =============================================================================
# Record input validation
# =============================================================================


def test_record_standardization_rejects_misaligned_date_pair_counts(standardizer):
    with pytest.raises(
        ValueError,
        match=(r"SAMN_MISALIGNED.*date_attr_orig=2.*date_val_orig=1.*date_category=2"),
    ):
        standardize_record(
            standardizer,
            accession="SAMN_MISALIGNED",
            attributes="collection_date||submission_date",
            values="2020",
            categories="c||f",
        )


def test_record_standardization_rejects_unknown_date_pair_category(standardizer):
    with pytest.raises(
        ValueError,
        match=r"SAMN_BAD_CATEGORY.*date_category\[0\]='x'.*expected 'c' or 'f'",
    ):
        standardize_record(
            standardizer,
            accession="SAMN_BAD_CATEGORY",
            values="2020",
            categories="x",
        )


def test_record_standardization_distinguishes_empty_date_pair_set(standardizer):
    with pytest.raises(ValueError, match="No date attribute-value pairs for SAMN_EMPTY"):
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


def test_record_standardization_returns_typed_collection_date_outcome(standardizer):
    outcome = standardize_record(
        standardizer,
        accession="SAMN00000001",
        values="2020-02",
    )

    assert outcome == DateOutcome(
        bounds=DateBounds(date(2020, 2, 1), date(2020, 2, 29)),
        category=DateCategory.SAMPLE_COLLECTION,
        structure=DateStructure.SINGLE_VALUE,
        precision=DatePrecision.MONTH,
        derivations=("direct",),
        supporting_pairs=(SupportingAttributeValuePair("collection_date", "2020-02"),),
    )

    assert standardizer.diagnostic_counts == {DateDiagnostic.COLLECTION_DATE_SELECTION: 1}


def test_record_standardization_prefers_valid_collection_date_over_all_fallback_dates(standardizer):
    outcome = standardize_record(
        standardizer,
        accession="SAMN00000002",
        attributes="submission_date||collection_date||publication_date",
        values="2018-01-01||2020||2019-06-15",
        categories="f||c||f",
    )

    assert outcome == DateOutcome(
        bounds=DateBounds(date(2020, 1, 1), date(2020, 12, 31)),
        category=DateCategory.SAMPLE_COLLECTION,
        structure=DateStructure.SINGLE_VALUE,
        precision=DatePrecision.YEAR,
        derivations=("direct",),
        supporting_pairs=(SupportingAttributeValuePair("collection_date", "2020"),),
    )


def test_record_standardization_uses_oldest_fallback_date_when_collection_dates_are_rejected(
    standardizer,
):
    outcome = standardize_record(
        standardizer,
        accession="SAMN00000003",
        attributes="collection_date||submission_date||publication_date",
        values="2922-08-23||2021-06-15||2019",
        categories="c||f||f",
    )

    assert outcome == DateOutcome(
        bounds=DateBounds(date(2019, 1, 1), date(2019, 12, 31)),
        category=DateCategory.FALLBACK,
        structure=DateStructure.SINGLE_VALUE,
        precision=DatePrecision.YEAR,
        derivations=("direct",),
        supporting_pairs=(SupportingAttributeValuePair("publication_date", "2019"),),
    )
    assert standardizer.diagnostic_counts == {DateDiagnostic.FALLBACK_DATE_SELECTION: 1}


# =============================================================================
# Multi-value records
# =============================================================================
#
# Some records list the collection date in multiple metadata fields.
# Three cases:
#   1. All values parse to the same bounds -> collapse silently.
#   2. Values differ but all are valid -> treat as an implicit interval.
#   3. Some values are invalid -> use the valid ones, ignore the rest.


def test_conflicting_collection_dates_preserve_and_deduplicate_supporting_pairs(
    standardizer,
    caplog,
):
    outcome = standardize_record(
        standardizer,
        accession="SAMN_PAIRED_DATES",
        attributes="collection_date||sampling_date||collection_date||collection_date",
        values="1993||1993||2009||1993",
        categories="c||c||c||c",
    )

    assert outcome == DateOutcome(
        bounds=DateBounds(date(1993, 1, 1), date(2009, 12, 31)),
        category=DateCategory.SAMPLE_COLLECTION,
        structure=DateStructure.CONFLICT_RANGE,
        precision=DatePrecision.YEAR,
        derivations=("direct",),
        supporting_pairs=(
            SupportingAttributeValuePair("collection_date", "1993"),
            SupportingAttributeValuePair("sampling_date", "1993"),
            SupportingAttributeValuePair("collection_date", "2009"),
        ),
    )
    assert standardizer.diagnostic_counts == {
        DateDiagnostic.COLLECTION_DATE_SELECTION: 1,
        DateDiagnostic.CONFLICTING_DATE_COMBINATION: 1,
    }
    assert caplog.messages == []


def test_equivalent_collection_date_bounds_keep_all_distinct_supporting_pairs(
    standardizer,
    caplog,
):
    outcome = standardize_record(
        standardizer,
        accession="SAMN_EQUIVALENT_BOUNDS",
        attributes="collection_date||sampling_date",
        values="2019-03-04||04/03/2019",
        categories="c||c",
    )

    assert outcome == DateOutcome(
        bounds=DateBounds(date(2019, 3, 4), date(2019, 3, 4)),
        category=DateCategory.SAMPLE_COLLECTION,
        structure=DateStructure.SINGLE_VALUE,
        precision=DatePrecision.DAY,
        derivations=("direct",),
        supporting_pairs=(
            SupportingAttributeValuePair("collection_date", "2019-03-04"),
            SupportingAttributeValuePair("sampling_date", "04/03/2019"),
        ),
    )
    assert standardizer.diagnostic_counts == {
        DateDiagnostic.COLLECTION_DATE_SELECTION: 1,
        DateDiagnostic.EQUIVALENT_DATE_COLLAPSE: 1,
    }
    assert caplog.messages == []


def test_equivalent_claims_union_exceptional_derivations_and_keep_their_evidence(
    standardizer,
):
    outcome = standardize_record(
        standardizer,
        accession="SAMN_EQUIVALENT_DERIVATIONS",
        attributes="collection_date||sampling_date",
        values="2019-03-04||2019-03-04T25:99",
        categories="c||c",
    )

    assert outcome.derivations == ("malformed_time_suffix",)
    assert outcome.supporting_pairs == (
        SupportingAttributeValuePair("collection_date", "2019-03-04"),
        SupportingAttributeValuePair("sampling_date", "2019-03-04T25:99"),
    )


def test_equivalent_single_value_and_interval_have_single_value_structure(standardizer):
    outcome = standardize_record(
        standardizer,
        accession="SAMN_EQUIVALENT_STRUCTURES",
        attributes="collection_date||sampling_date",
        values="2019||2019-01-01/2019-12-31",
        categories="c||c",
    )

    assert outcome.structure is DateStructure.SINGLE_VALUE
    assert outcome.supporting_pairs == (
        SupportingAttributeValuePair("collection_date", "2019"),
        SupportingAttributeValuePair("sampling_date", "2019-01-01/2019-12-31"),
    )


def test_explicit_interval_preserves_record_selection_and_supporting_pair(standardizer):
    outcome = standardize_record(
        standardizer,
        accession="SAMN_INTERVAL_SAMPLING",
        attributes="submission_date||collection_date",
        values="2018||Nov-2016/29-May-2017",
        categories="f||c",
    )

    assert outcome == DateOutcome(
        bounds=DateBounds(date(2016, 11, 1), date(2017, 5, 29)),
        category=DateCategory.SAMPLE_COLLECTION,
        structure=DateStructure.REPORTED_INTERVAL,
        precision=DatePrecision.MONTH,
        derivations=("direct",),
        supporting_pairs=(SupportingAttributeValuePair("collection_date", "Nov-2016/29-May-2017"),),
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
        DateDiagnostic.NO_USABLE_DATE: 2,
        DateDiagnostic.COLLECTION_DATE_SELECTION: 1,
    }
    assert caplog.messages == []


def test_metadata_reference_date_rejects_later_interval_endpoint_and_counts_notice(
    standardizer, caplog
):
    assert standardize_date_value(standardizer, "2019 to 2027") is None
    reversed_bounds = standardize_date_value(standardizer, "2020 to 2018").bounds

    assert reversed_bounds == DateBounds(date(2018, 1, 1), date(2020, 12, 31))
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
    outcome = standardize_date_value(standardizer, input_string)

    assert outcome.bounds == DateBounds(date(2019, 3, 15), date(2019, 3, 15))
    assert outcome.structure is DateStructure.SINGLE_VALUE
    assert outcome.precision is DatePrecision.DAY
    assert outcome.derivations == ("direct",)


def test_iso_datetime_with_time_component_is_stripped_to_the_date(standardizer):
    """ISO timestamps are parsed from their date part."""
    outcome = standardize_date_value(standardizer, "2019-03-15T12:00:00Z")

    assert outcome.bounds == DateBounds(date(2019, 3, 15), date(2019, 3, 15))


# =============================================================================
# Value interpretation: partial dates
# =============================================================================
#
# A partial date like "2019" or "March 2019" is a range of possible
# collection dates, not a single one. The goal is not to invent precision
# that the source didn't provide.


def test_year_only_date_spans_the_entire_year(standardizer):
    outcome = standardize_date_value(standardizer, "2019")

    assert outcome.bounds == DateBounds(date(2019, 1, 1), date(2019, 12, 31))
    assert outcome.precision is DatePrecision.YEAR


@pytest.mark.parametrize(
    "input_string",
    [
        "2019/03",
        "03-2019",
        "03/2019",
        "March 2019",
    ],
)
def test_year_month_date_spans_the_entire_month(standardizer, input_string):
    outcome = standardize_date_value(standardizer, input_string)

    assert outcome.bounds == DateBounds(date(2019, 3, 1), date(2019, 3, 31))
    assert outcome.precision is DatePrecision.MONTH


# =============================================================================
# Value interpretation: ambiguous day/month formats
# =============================================================================


@pytest.mark.parametrize("input_string", ["03/03/2019", "03-03-2019"])
def test_equal_numeric_components_resolve_without_ambiguity(standardizer, input_string):
    outcome = standardize_date_value(standardizer, input_string)

    assert outcome.bounds == DateBounds(date(2019, 3, 3), date(2019, 3, 3))
    assert outcome.precision is DatePrecision.DAY


# =============================================================================
# Value interpretation: intervals
# =============================================================================
#
# E.g. "2018/2020". We expand each endpoint to its own bounds and take
# the outer envelope: earliest possible start to latest possible end.


@pytest.mark.parametrize("separator", [" to ", " - "])
def test_interval_separators_all_produce_the_same_envelope(standardizer, separator):
    outcome = standardize_date_value(standardizer, f"2018{separator}2020")

    assert outcome.bounds == DateBounds(date(2018, 1, 1), date(2020, 12, 31))
    assert outcome.structure is DateStructure.REPORTED_INTERVAL
    assert outcome.precision is DatePrecision.YEAR


def test_interval_collapsing_to_single_date_has_single_value_structure(standardizer):
    outcome = standardize_date_value(standardizer, "2019/2019")

    assert outcome.bounds == DateBounds(date(2019, 1, 1), date(2019, 12, 31))
    assert outcome.structure is DateStructure.SINGLE_VALUE
    assert outcome.precision is DatePrecision.YEAR


def test_interval_with_mixed_precision_uses_each_endpoints_bounds(standardizer):
    outcome = standardize_date_value(standardizer, "2018/2020-06")

    assert outcome.bounds.start == date(2018, 1, 1)  # full year on the left
    assert outcome.bounds.end == date(2020, 6, 30)  # end of June on the right
    assert outcome.precision is DatePrecision.YEAR


def test_reversed_interval_is_widened_to_the_outer_envelope(standardizer):
    """'2020/2018' has the endpoints in the wrong order."""
    outcome = standardize_date_value(standardizer, "2020/2018")

    assert outcome.bounds == DateBounds(date(2018, 1, 1), date(2020, 12, 31))
    assert outcome.derivations == ("reversed_interval",)


def test_exceptional_derivations_have_deterministic_alphabetical_order(standardizer):
    outcome = standardize_date_value(standardizer, "2020T25:99/2018")

    assert outcome.derivations == (
        "malformed_time_suffix",
        "reversed_interval",
    )


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
        DateBounds(
            start=date(2020, 1, 1),
            end=date(2019, 1, 1),
        )
