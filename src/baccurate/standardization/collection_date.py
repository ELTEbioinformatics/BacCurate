"""Parse collection dates and pick the best one for a record."""

from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date
from enum import StrEnum

from baccurate.standardization._attribute_value_text import split_pipe_separated
from baccurate.standardization._collection_date_interpreter import (
    DateBounds,
    DateInterpreter,
    DateRejection,
    reliability_score_for_interval,
)
from baccurate.standardization.supporting_attribute_value_pair import SupportingAttributeValuePair
from baccurate.standardization_target.specifications import (
    COLLECTION_DATE_CATEGORY,
    FALLBACK_DATE_CATEGORY,
)

FALLBACK_SCORE = 0.1


class DateDiagnostic(StrEnum):
    """Record-selection vocabulary used by build diagnostics."""

    COLLECTION_DATE_SELECTION = "collection_date_selection"
    FALLBACK_DATE_SELECTION = "fallback_date_selection"
    EQUIVALENT_DATE_COLLAPSE = "equivalent_date_collapse"
    CONFLICTING_DATE_COMBINATION = "conflicting_date_combination"
    NO_USABLE_DATE = "no_usable_date"


@dataclass(frozen=True, slots=True)
class ParsedDate:
    """A parsed date with the attribute and value it came from."""

    bounds: DateBounds
    attribute: str
    value: str


@dataclass(frozen=True, slots=True)
class DateOutcome:
    """The chosen date and its supporting attribute-value pairs."""

    bounds: DateBounds
    supporting_pairs: tuple[SupportingAttributeValuePair, ...]

    def __post_init__(self) -> None:
        if not self.supporting_pairs:
            raise ValueError("A date outcome requires at least one supporting attribute-value pair")

    @property
    def reliability_score(self) -> float:
        return self.bounds.reliability_score


class RecordDateStandardizer:
    """Pick the best date from one extracted metadata record's selected values."""

    def __init__(self, metadata_reference_date: date):
        self.interpreter = DateInterpreter(metadata_reference_date)
        self.rejection_counts: Counter[str] = Counter()
        self.notice_counts: Counter[str] = Counter()
        self.diagnostic_counts: Counter[DateDiagnostic] = Counter()

    def standardize(self, extracted_record: Mapping[str, str]) -> DateOutcome | None:
        """Standardize the collection dates in one extracted metadata record."""
        accession = extracted_record.get("accession", "")
        attributes = split_pipe_separated(extracted_record.get("date_attr_orig", ""))
        values = split_pipe_separated(extracted_record.get("date_val_orig", ""))
        categories = split_pipe_separated(extracted_record.get("date_category", ""))
        pair_counts = (len(attributes), len(values), len(categories))
        if len(set(pair_counts)) != 1:
            raise ValueError(
                f"Malformed date attribute-value pairs for {accession}: "
                f"date_attr_orig={pair_counts[0]}, "
                f"date_val_orig={pair_counts[1]}, "
                f"date_category={pair_counts[2]}; counts must match"
            )
        if pair_counts[0] == 0:
            raise ValueError(f"No date attribute-value pairs for {accession}")
        for index, category in enumerate(categories):
            if category not in {COLLECTION_DATE_CATEGORY, FALLBACK_DATE_CATEGORY}:
                raise ValueError(
                    f"Malformed date attribute-value pairs for {accession}: "
                    f"date_category[{index}]={category!r}; expected "
                    f"{COLLECTION_DATE_CATEGORY!r} or {FALLBACK_DATE_CATEGORY!r}"
                )

        collection_dates: list[ParsedDate] = []
        fallback_dates: list[ParsedDate] = []
        for attribute, value, category in zip(attributes, values, categories, strict=False):
            parsed = self._parse_date(attribute, value)
            if parsed is None:
                continue
            (collection_dates if category == COLLECTION_DATE_CATEGORY else fallback_dates).append(
                parsed
            )

        if collection_dates:
            self.diagnostic_counts[DateDiagnostic.COLLECTION_DATE_SELECTION] += 1
            return self._select_collection_date(collection_dates)
        if fallback_dates:
            self.diagnostic_counts[DateDiagnostic.FALLBACK_DATE_SELECTION] += 1
            return self._select_fallback_date(fallback_dates)
        self.diagnostic_counts[DateDiagnostic.NO_USABLE_DATE] += 1
        return None

    def _parse_date(
        self,
        attribute: str,
        value: str,
    ) -> ParsedDate | None:
        result = self.interpreter.interpret(value)
        if isinstance(result, DateRejection):
            self.rejection_counts[result.reason] += 1
            return None
        for notice in result.notices:
            self.notice_counts[notice] += 1
        return ParsedDate(result.bounds, attribute, value)

    def _select_collection_date(
        self,
        parsed_dates: list[ParsedDate],
    ) -> DateOutcome:
        first = parsed_dates[0]
        if len(parsed_dates) == 1:
            return DateOutcome(
                first.bounds,
                (SupportingAttributeValuePair(first.attribute, first.value),),
            )

        temporal_bounds = {(parsed.bounds.start, parsed.bounds.end) for parsed in parsed_dates}
        if len(temporal_bounds) == 1:
            self.diagnostic_counts[DateDiagnostic.EQUIVALENT_DATE_COLLAPSE] += 1
            bounds = DateBounds(
                first.bounds.start,
                first.bounds.end,
                min(parsed.bounds.reliability_score for parsed in parsed_dates),
            )
            return DateOutcome(
                bounds,
                (SupportingAttributeValuePair(first.attribute, first.value),),
            )

        combined_start = min(parsed.bounds.start for parsed in parsed_dates)
        combined_end = max(parsed.bounds.end for parsed in parsed_dates)
        bounds = DateBounds(
            combined_start,
            combined_end,
            reliability_score_for_interval(combined_start, combined_end),
        )
        distinct_pairs = list(
            dict.fromkeys((parsed.attribute, parsed.value) for parsed in parsed_dates)
        )
        self.diagnostic_counts[DateDiagnostic.CONFLICTING_DATE_COMBINATION] += 1
        supporting_pairs = tuple(
            SupportingAttributeValuePair(attribute, value) for attribute, value in distinct_pairs
        )
        return DateOutcome(bounds, supporting_pairs)

    @staticmethod
    def _select_fallback_date(parsed_dates: list[ParsedDate]) -> DateOutcome:
        oldest = min(parsed_dates, key=lambda parsed: parsed.bounds.start)
        bounds = DateBounds(oldest.bounds.start, oldest.bounds.end, FALLBACK_SCORE)
        return DateOutcome(
            bounds,
            (SupportingAttributeValuePair(oldest.attribute, oldest.value),),
        )
