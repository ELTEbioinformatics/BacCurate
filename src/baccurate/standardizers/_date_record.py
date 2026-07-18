"""Parse collection dates and pick the best one for a record."""

from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date
from enum import StrEnum

from baccurate.standardizers._date_interpreter import (
    DateBounds,
    DateInterpreter,
    DateRejection,
    score_for_interval,
)
from baccurate.utils.text import split_pipe_separated

FALLBACK_SCORE = 0.1
SAMPLING_CATEGORY = "s"
FALLBACK_CATEGORY = "o"


class DateDiagnostic(StrEnum):
    """Record-selection vocabulary used by build diagnostics."""

    SAMPLING_SELECTION = "sampling_selection"
    FALLBACK_SELECTION = "fallback_selection"
    EQUIVALENT_CANDIDATE_COLLAPSE = "equivalent_candidate_collapse"
    CONFLICTING_CANDIDATE_COMBINATION = "conflicting_candidate_combination"
    NO_USABLE_CANDIDATE = "no_usable_candidate"


@dataclass(frozen=True, slots=True)
class ParsedDateCandidate:
    """A parsed date with the attribute and value it came from."""

    bounds: DateBounds
    attribute: str
    value: str


@dataclass(frozen=True, slots=True)
class DateOrigin:
    """One source attribute/value pair supporting a selected date."""

    attribute: str
    value: str


@dataclass(frozen=True, slots=True)
class DateOutcome:
    """The chosen date and its paired source origins."""

    bounds: DateBounds
    origins: tuple[DateOrigin, ...]

    def __post_init__(self) -> None:
        if not self.origins:
            raise ValueError("A date outcome requires at least one source origin")

    @property
    def score(self) -> float:
        return self.bounds.score


class RecordDateStandardizer:
    """Pick the best date for a record from its extracted candidates."""

    def __init__(self, metadata_reference_date: date):
        self.interpreter = DateInterpreter(metadata_reference_date)
        self.rejection_counts: Counter[str] = Counter()
        self.notice_counts: Counter[str] = Counter()
        self.diagnostic_counts: Counter[DateDiagnostic] = Counter()

    def standardize(self, record: Mapping[str, str]) -> DateOutcome | None:
        accession = record.get("accession", "")
        attributes = split_pipe_separated(record.get("date_attr_orig", ""))
        values = split_pipe_separated(record.get("date_val_orig", ""))
        categories = split_pipe_separated(record.get("date_category", ""))
        candidate_counts = (len(attributes), len(values), len(categories))
        if len(set(candidate_counts)) != 1:
            raise ValueError(
                f"Malformed date candidates for {accession}: "
                f"date_attr_orig={candidate_counts[0]}, "
                f"date_val_orig={candidate_counts[1]}, "
                f"date_category={candidate_counts[2]}; counts must match"
            )
        if candidate_counts[0] == 0:
            raise ValueError(f"No date candidates for {accession}")
        for index, category in enumerate(categories):
            if category not in {SAMPLING_CATEGORY, FALLBACK_CATEGORY}:
                raise ValueError(
                    f"Malformed date candidates for {accession}: "
                    f"date_category[{index}]={category!r}; expected "
                    f"{SAMPLING_CATEGORY!r} or {FALLBACK_CATEGORY!r}"
                )

        sampling: list[ParsedDateCandidate] = []
        fallback: list[ParsedDateCandidate] = []
        for attribute, value, category in zip(attributes, values, categories, strict=False):
            candidate = self._interpret_candidate(attribute, value)
            if candidate is None:
                continue
            (sampling if category == SAMPLING_CATEGORY else fallback).append(candidate)

        if sampling:
            self.diagnostic_counts[DateDiagnostic.SAMPLING_SELECTION] += 1
            return self._select_sampling(sampling)
        if fallback:
            self.diagnostic_counts[DateDiagnostic.FALLBACK_SELECTION] += 1
            return self._select_fallback(fallback)
        self.diagnostic_counts[DateDiagnostic.NO_USABLE_CANDIDATE] += 1
        return None

    def _interpret_candidate(
        self,
        attribute: str,
        value: str,
    ) -> ParsedDateCandidate | None:
        result = self.interpreter.interpret(value)
        if isinstance(result, DateRejection):
            self.rejection_counts[result.reason] += 1
            return None
        for notice in result.notices:
            self.notice_counts[notice] += 1
        return ParsedDateCandidate(result.bounds, attribute, value)

    def _select_sampling(
        self,
        candidates: list[ParsedDateCandidate],
    ) -> DateOutcome:
        first = candidates[0]
        if len(candidates) == 1:
            return DateOutcome(first.bounds, (DateOrigin(first.attribute, first.value),))

        temporal_bounds = {
            (candidate.bounds.start, candidate.bounds.end) for candidate in candidates
        }
        if len(temporal_bounds) == 1:
            self.diagnostic_counts[DateDiagnostic.EQUIVALENT_CANDIDATE_COLLAPSE] += 1
            bounds = DateBounds(
                first.bounds.start,
                first.bounds.end,
                min(candidate.bounds.score for candidate in candidates),
            )
            return DateOutcome(bounds, (DateOrigin(first.attribute, first.value),))

        combined_start = min(candidate.bounds.start for candidate in candidates)
        combined_end = max(candidate.bounds.end for candidate in candidates)
        bounds = DateBounds(
            combined_start,
            combined_end,
            score_for_interval(combined_start, combined_end),
        )
        candidate_pairs = list(
            dict.fromkeys((candidate.attribute, candidate.value) for candidate in candidates)
        )
        self.diagnostic_counts[DateDiagnostic.CONFLICTING_CANDIDATE_COMBINATION] += 1
        origins = tuple(DateOrigin(attribute, value) for attribute, value in candidate_pairs)
        return DateOutcome(bounds, origins)

    @staticmethod
    def _select_fallback(candidates: list[ParsedDateCandidate]) -> DateOutcome:
        oldest = min(candidates, key=lambda candidate: candidate.bounds.start)
        bounds = DateBounds(oldest.bounds.start, oldest.bounds.end, FALLBACK_SCORE)
        return DateOutcome(bounds, (DateOrigin(oldest.attribute, oldest.value),))
