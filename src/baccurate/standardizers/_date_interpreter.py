"""Interpretation of one date value."""

import calendar
import re
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date
from typing import Literal

MIN_YEAR = 1800

Notice = Literal["malformed_time_suffix_ignored", "reversed_interval_normalized"]
EndpointContext = Literal["start", "end"]
RejectionReason = Literal[
    "unsupported_format",
    "invalid_calendar_date",
    "before_supported_year",
    "after_metadata_reference_date",
    "invalid_interval_structure",
]
SingleDateFormatId = Literal[
    "year",
    "year_month",
    "month_year",
    "named_month_year",
    "year_first_day",
    "named_month_day",
    "day_named_month",
    "numeric_year_last_day",
]
InterpretationFormatId = (
    SingleDateFormatId
    | Literal[
        "abbreviated_year_interval",
        "explicit_interval",
    ]
)


@dataclass(frozen=True, slots=True)
class DateBounds:
    """Earliest and latest possible collection day, with a reliability score."""

    start: date
    end: date
    score: float

    def __post_init__(self) -> None:
        if self.start > self.end:
            raise ValueError(f"DateBounds: start {self.start} > end {self.end}")


@dataclass(frozen=True, slots=True)
class DateInterpretation:
    """A date parsed into the span of days it could refer to."""

    bounds: DateBounds
    format_id: InterpretationFormatId
    notices: tuple[Notice, ...] = ()


@dataclass(frozen=True, slots=True)
class DateRejection:
    """A date value that cannot be used and its reason."""

    reason: RejectionReason
    endpoint_context: EndpointContext | None = None


SingleParse = DateBounds | DateRejection
InterpretationResult = DateInterpretation | DateRejection


@dataclass(frozen=True, slots=True)
class FormatSpec:
    """One recognizable date form, with its id, the regex that matches it, and its parser."""

    format_id: SingleDateFormatId
    pattern: re.Pattern[str]
    parse: Callable[[re.Match[str]], SingleParse]


MONTHS = {
    "january": 1,
    "jan": 1,
    "february": 2,
    "feb": 2,
    "march": 3,
    "mar": 3,
    "april": 4,
    "apr": 4,
    "may": 5,
    "june": 6,
    "jun": 6,
    "july": 7,
    "jul": 7,
    "august": 8,
    "aug": 8,
    "september": 9,
    "sep": 9,
    "sept": 9,
    "october": 10,
    "oct": 10,
    "november": 11,
    "nov": 11,
    "december": 12,
    "dec": 12,
}
MONTH_TOKEN = "(?:" + "|".join(sorted(MONTHS, key=len, reverse=True)) + ")"

YEAR = re.compile(r"^(?P<year>\d{4})$")
YEAR_MONTH = re.compile(r"^(?P<year>\d{4})[-/](?P<month>\d{2})$")
MONTH_YEAR = re.compile(r"^(?P<month>\d{2})[-/](?P<year>\d{4})$")
NAMED_MONTH_YEAR = re.compile(
    rf"^(?P<month>{MONTH_TOKEN})(?: |-)(?P<year>\d{{4}})$",
    re.IGNORECASE | re.ASCII,
)
YEAR_FIRST_DAY = re.compile(
    r"^(?P<year>\d{4})(?P<separator>[-/])(?P<month>\d{2})"
    r"(?P=separator)(?P<day>\d{2})$"
)
NAMED_MONTH_DAY = re.compile(
    rf"^(?P<month>{MONTH_TOKEN}) (?P<day>\d{{1,2}}),\s*(?P<year>\d{{4}})$",
    re.IGNORECASE | re.ASCII,
)
DAY_NAMED_MONTH = re.compile(
    rf"^(?P<day>\d{{1,2}})-(?P<month>{MONTH_TOKEN})-(?P<year>\d{{4}})$",
    re.IGNORECASE | re.ASCII,
)
NUMERIC_YEAR_LAST_DAY = re.compile(
    r"^(?P<first>\d{1,2})(?P<separator>[-/])(?P<second>\d{1,2})"
    r"(?P=separator)(?P<year>\d{4})$"
)
VALID_TIME_SUFFIX = re.compile(
    r"^T(?:(?:[01]\d|2[0-3]):[0-5]\d(?:[:][0-5]\d(?:\.\d+)?)?"
    r"|(?:[01]\d|2[0-3])\.[0-5]\d\.[0-5]\d(?:\.\d+)?)"
    r"(?:Z|[+-](?:[01]\d|2[0-3]):?[0-5]\d)?$"
)
WORD_OR_HYPHEN_SEPARATOR = re.compile(r"\s+(?:to|-)\s+", re.IGNORECASE | re.ASCII)
WORD_OR_HYPHEN_INTERVAL_CLAIM = re.compile(r"(?:^|\s)(?:to|-)(?=\s|$)", re.IGNORECASE | re.ASCII)


def _month_bounds(year: int, month: int) -> tuple[date, date]:
    return date(year, month, 1), date(year, month, calendar.monthrange(year, month)[1])


def _calendar_date(year: int, month: int, day: int) -> date | None:
    try:
        return date(year, month, day)
    except ValueError:
        return None


def _parse_year(match: re.Match[str]) -> SingleParse:
    year = int(match["year"])
    try:
        return DateBounds(date(year, 1, 1), date(year, 12, 31), 0.8)
    except ValueError:
        return DateRejection("invalid_calendar_date")


def _parse_month(year: int, month: int) -> SingleParse:
    try:
        start, end = _month_bounds(year, month)
    except ValueError:
        return DateRejection("invalid_calendar_date")
    return DateBounds(start, end, 0.9)


def _parse_numeric_month(match: re.Match[str]) -> SingleParse:
    return _parse_month(int(match["year"]), int(match["month"]))


def _parse_named_month(match: re.Match[str]) -> SingleParse:
    return _parse_month(int(match["year"]), MONTHS[match["month"].casefold()])


def _parse_day(year: int, month: int, day: int) -> SingleParse:
    parsed = _calendar_date(year, month, day)
    if parsed is None:
        return DateRejection("invalid_calendar_date")
    return DateBounds(parsed, parsed, 1.0)


def _parse_year_first_day(match: re.Match[str]) -> SingleParse:
    return _parse_day(int(match["year"]), int(match["month"]), int(match["day"]))


def _parse_named_month_day(match: re.Match[str]) -> SingleParse:
    return _parse_day(
        int(match["year"]),
        MONTHS[match["month"].casefold()],
        int(match["day"]),
    )


def _parse_numeric_year_last_day(match: re.Match[str]) -> SingleParse:
    year = int(match["year"])
    first = int(match["first"])
    second = int(match["second"])
    day_first = _calendar_date(year, second, first)
    month_first = _calendar_date(year, first, second)
    if day_first is None and month_first is None:
        return DateRejection("invalid_calendar_date")
    if day_first is None:
        chosen = month_first
        score = 1.0
    elif month_first is None or day_first == month_first:
        chosen = day_first
        score = 1.0
    else:
        chosen = day_first
        score = 0.7
    assert chosen is not None
    return DateBounds(chosen, chosen, score)


def score_for_interval(start: date, end: date) -> float:
    """Score an interval using calendar-anniversary width thresholds."""
    for years, score in ((1, 0.6), (3, 0.5), (6, 0.4), (10, 0.3)):
        try:
            anniversary = start.replace(year=start.year + years)
        except ValueError:
            anniversary = start.replace(year=start.year + years, day=28)
        if end <= anniversary:
            return score
    return 0.2


def _is_recognized_endpoint(result: InterpretationResult) -> bool:
    return not (isinstance(result, DateRejection) and result.reason == "unsupported_format")


def _finalize_interval(
    start_result: InterpretationResult,
    end_result: InterpretationResult,
    format_id: InterpretationFormatId,
) -> InterpretationResult:
    if isinstance(start_result, DateRejection):
        return DateRejection(start_result.reason, "start")
    if isinstance(end_result, DateRejection):
        return DateRejection(end_result.reason, "end")

    combined_start = min(start_result.bounds.start, end_result.bounds.start)
    combined_end = max(start_result.bounds.end, end_result.bounds.end)
    if (
        start_result.bounds.start == end_result.bounds.start
        and start_result.bounds.end == end_result.bounds.end
    ):
        score = min(start_result.bounds.score, end_result.bounds.score)
    else:
        score = score_for_interval(combined_start, combined_end)
    notices = start_result.notices + end_result.notices
    if start_result.bounds.start > end_result.bounds.end:
        notices += ("reversed_interval_normalized",)
    return DateInterpretation(
        DateBounds(combined_start, combined_end, score),
        format_id,
        notices,
    )


SINGLE_DATE_FORMATS = (
    FormatSpec("year", YEAR, _parse_year),
    FormatSpec("year_month", YEAR_MONTH, _parse_numeric_month),
    FormatSpec("month_year", MONTH_YEAR, _parse_numeric_month),
    FormatSpec("named_month_year", NAMED_MONTH_YEAR, _parse_named_month),
    FormatSpec("year_first_day", YEAR_FIRST_DAY, _parse_year_first_day),
    FormatSpec("named_month_day", NAMED_MONTH_DAY, _parse_named_month_day),
    FormatSpec("day_named_month", DAY_NAMED_MONTH, _parse_named_month_day),
    FormatSpec("numeric_year_last_day", NUMERIC_YEAR_LAST_DAY, _parse_numeric_year_last_day),
)


class DateInterpreter:
    """Parse a reported date string into calendar bounds, or reject it.

    Recognizes single dates and intervals (abbreviated year ranges,
    "to"/hyphen ranges, and slash-separated pairs).

    Values after metadata_reference_date are rejected.
    """

    def __init__(self, metadata_reference_date: date):
        self.metadata_reference_date = metadata_reference_date

    def interpret(self, raw_value: str) -> InterpretationResult:
        value = raw_value.strip()
        single_result = self.interpret_single(value)
        if isinstance(single_result, DateInterpretation):
            return single_result

        abbreviated = re.fullmatch(r"(?P<start>\d{4})-(?P<end>\d{2})", value)
        if abbreviated is not None:
            start_year = int(abbreviated["start"])
            end_suffix = int(abbreviated["end"])
            end_year = start_year - (start_year % 100) + end_suffix
            if end_suffix >= 13 and end_year >= start_year:
                return _finalize_interval(
                    self.interpret_single(str(start_year)),
                    self.interpret_single(str(end_year)),
                    "abbreviated_year_interval",
                )

        word_or_hyphen_separators = list(WORD_OR_HYPHEN_SEPARATOR.finditer(value))
        if len(word_or_hyphen_separators) == 1:
            separator = word_or_hyphen_separators[0]
            return _finalize_interval(
                self.interpret_single(value[: separator.start()]),
                self.interpret_single(value[separator.end() :]),
                "explicit_interval",
            )
        if word_or_hyphen_separators or WORD_OR_HYPHEN_INTERVAL_CLAIM.search(value):
            return DateRejection("invalid_interval_structure")

        valid_decompositions: list[tuple[DateInterpretation, DateInterpretation]] = []
        rejected_decompositions: list[tuple[InterpretationResult, InterpretationResult]] = []
        slash_claimed = False
        for slash in (match.start() for match in re.finditer("/", value)):
            start_result = self.interpret_single(value[:slash])
            end_result = self.interpret_single(value[slash + 1 :])
            start_recognized = _is_recognized_endpoint(start_result)
            end_recognized = _is_recognized_endpoint(end_result)
            slash_claimed = slash_claimed or start_recognized or end_recognized
            if start_recognized and end_recognized:
                if isinstance(start_result, DateInterpretation) and isinstance(
                    end_result, DateInterpretation
                ):
                    valid_decompositions.append((start_result, end_result))
                else:
                    rejected_decompositions.append((start_result, end_result))

        preferred_decompositions = valid_decompositions or rejected_decompositions
        if len(preferred_decompositions) == 1:
            return _finalize_interval(*preferred_decompositions[0], "explicit_interval")
        if len(preferred_decompositions) > 1:
            return DateRejection("invalid_interval_structure")
        if single_result.reason != "unsupported_format":
            return single_result
        if slash_claimed:
            return DateRejection("invalid_interval_structure")
        return single_result

    def interpret_single(self, raw_value: str) -> DateInterpretation | DateRejection:
        value = raw_value.strip()
        parsed = self._parse_supported_shape(value)
        notices: tuple[Notice, ...] = ()
        if (
            isinstance(parsed, DateRejection)
            and parsed.reason == "unsupported_format"
            and "T" in value
        ):
            base, suffix = value.split("T", 1)
            if "/" in suffix or WORD_OR_HYPHEN_INTERVAL_CLAIM.search(suffix):
                return parsed
            prefix_result = self._parse_supported_shape(base)
            if not (
                isinstance(prefix_result, DateRejection)
                and prefix_result.reason == "unsupported_format"
            ):
                parsed = prefix_result
                if not VALID_TIME_SUFFIX.fullmatch(f"T{suffix}"):
                    notices = ("malformed_time_suffix_ignored",)
        if isinstance(parsed, DateRejection):
            return parsed

        bounds, format_id = parsed
        if bounds.end.year < MIN_YEAR:
            return DateRejection("before_supported_year")
        if bounds.start > self.metadata_reference_date:
            return DateRejection("after_metadata_reference_date")
        return DateInterpretation(bounds, format_id, notices)

    def _parse_supported_shape(
        self, value: str
    ) -> tuple[DateBounds, SingleDateFormatId] | DateRejection:
        matches = [
            (format_spec, match)
            for format_spec in SINGLE_DATE_FORMATS
            if (match := format_spec.pattern.fullmatch(value)) is not None
        ]
        if not matches:
            return DateRejection("unsupported_format")
        if len(matches) > 1:
            format_ids = ", ".join(spec.format_id for spec, _match in matches)
            raise RuntimeError(
                f"Single-date registry matched {value!r} more than once: {format_ids}"
            )

        format_spec, match = matches[0]
        result = format_spec.parse(match)
        if isinstance(result, DateRejection):
            return result
        return result, format_spec.format_id
