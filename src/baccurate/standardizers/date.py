"""
Parses sample collection date strings into [start, end] bounds against a
fixed set of named regex patterns.

See docs/collection_date.md for documentation.
"""

import calendar
import csv
import logging
import re
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

from dateutil import parser as dateutil_parser

from baccurate.paths import DATE_OUTPUT
from baccurate.utils.args import create_arg_parser
from baccurate.utils.config import load_config
from baccurate.utils.logging import setup_standardizer_logging
from baccurate.utils.progress import count_tsv_rows, make_inner_bar
from baccurate.utils.text import split_pipe_separated

logger = logging.getLogger(__name__)

# --- Constants ---

# Older/newer inputs are very likely to be typos.
MIN_YEAR = 1800
MAX_YEAR = date.today().year

# dateutil falls back to this when the parsed token doesn't carry a
# day or month. Using year 1 makes any unfilled component obviously
# wrong and easy to spot in logs.
DATEUTIL_DEFAULT = datetime(1, 1, 1)

# How a partial date is widened into a [start, end] bounds pair.
EXPAND_DAY = "day"  # day-precision [d, d]
EXPAND_MONTH = "month"  # month-precision [first, last day of month]
EXPAND_YEAR = "year"  # year-precision [Jan 1, Dec 31]

# Single-date formats tried in priority order:
# (pattern_name, score, expansion_strategy, dateutil_dayfirst).
# Pattern names refer to compiled regexes in the config.
# Score reflects parse precision.
SINGLE_DATE_FORMATS: list[tuple[str, float, str, bool]] = [
    ("yyyy_mm_dd", 1.0, EXPAND_DAY, False),
    ("yyyy_mm_dd_slash", 1.0, EXPAND_DAY, False),
    ("mon_dd_yyyy", 1.0, EXPAND_DAY, False),
    ("dd_mon_yyyy", 1.0, EXPAND_DAY, False),
    ("yyyy_mm", 0.9, EXPAND_MONTH, False),
    ("yyyy_mm_slash", 0.9, EXPAND_MONTH, False),
    ("mon_yyyy", 0.9, EXPAND_MONTH, False),
    ("mm_yyyy", 0.9, EXPAND_MONTH, False),
    ("mm_yyyy_slash", 0.9, EXPAND_MONTH, False),
    ("yyyy", 0.8, EXPAND_YEAR, False),
    # 0.7 - day-precision but ambiguous ordering, assuming day-first.
    ("dd_mm_yyyy_slash", 0.7, EXPAND_DAY, True),
    ("dd_mm_yyyy_dash", 0.7, EXPAND_DAY, True),
]

# Interval scoring by span
INTERVAL_SPAN_SCORES: list[tuple[float, float]] = [
    # (max_years_inclusive, score)
    (1.0, 0.6),
    (3.0, 0.5),
    (6.0, 0.4),
    (10.0, 0.3),
]
INTERVAL_SCORE_OVER_TEN_YEARS = 0.2

# Score for a non-sampling date used when no sampling date is available.
FALLBACK_SCORE = 0.1


def _score_for_interval(start: date, end: date) -> float:
    """Score an interval by its span in years."""
    span_years = (end - start).days / 365.25
    for max_years, score in INTERVAL_SPAN_SCORES:
        if span_years <= max_years:
            return score
    return INTERVAL_SCORE_OVER_TEN_YEARS


# --- Data structures ---


@dataclass(frozen=True, slots=True)
class DateBounds:
    """A [start, end] date interval and its reliability score."""

    start: date
    end: date
    score: float

    def __post_init__(self) -> None:
        if self.start > self.end:
            raise ValueError(f"DateBounds: start {self.start} > end {self.end}")


@dataclass(frozen=True, slots=True)
class DateRecord:
    """A successfully parsed date together with the (attribute, value) it came from."""

    bounds: DateBounds
    attribute: str
    value: str


# --- Parsing helpers ---


def _is_plausible_date(d: date) -> bool:
    return MIN_YEAR <= d.year <= MAX_YEAR


def _expand_bounds(dt: date, strategy: str) -> tuple[date, date]:
    """Widen a parsed date into [start, end] bounds at the parse precision."""
    if strategy == EXPAND_YEAR:
        return date(dt.year, 1, 1), date(dt.year, 12, 31)
    if strategy == EXPAND_MONTH:
        last_day = calendar.monthrange(dt.year, dt.month)[1]
        return date(dt.year, dt.month, 1), date(dt.year, dt.month, last_day)
    return dt, dt


def _out_of_range_month(pattern_name: str, value: str) -> int | None:
    """Extract the month from a year-month string and return it
    only if it is outside 1-12, otherwise return None.

    Returns None also when the pattern does not describe a
    year-month format."""

    month_position = {
        "yyyy_mm": 1,
        "yyyy_mm_slash": 1,
        "mm_yyyy": 0,
        "mm_yyyy_slash": 0,
    }.get(pattern_name)

    if month_position is None:
        return None

    month = int(re.split(r"[-/]", value)[month_position])

    if 1 <= month <= 12:
        return None

    return month

class DateParser:
    """Holds the compiled regex set and dispatches single-date / interval parsing."""

    def __init__(self, compiled_patterns: dict[str, re.Pattern]):
        self.patterns = compiled_patterns

    def _strip_time(self, date_str: str) -> str:
        return self.patterns["time_strip"].sub("", date_str).strip()

    def parse_single_date(self, date_str: str) -> DateBounds | None:
        val = self._strip_time(date_str)

        for name, score, strategy, dayfirst in SINGLE_DATE_FORMATS:
            if not self.patterns[name].fullmatch(val):
                continue
            invalid_month = _out_of_range_month(name, val)
            if invalid_month is not None:
                logger.warning(
                    "Invalid year-month %r: month %d is outside 1..12",
                    date_str,
                    invalid_month,
                )
                return None
            try:
                dt = dateutil_parser.parse(val, default=DATEUTIL_DEFAULT, dayfirst=dayfirst)
            except (ValueError, OverflowError) as e:
                logger.debug("Pattern %s matched %r but dateutil failed: %s", name, val, e)
                return None

            start, end = _expand_bounds(dt.date(), strategy)
            if not (_is_plausible_date(start) and _is_plausible_date(end)):
                logger.warning("Skipping implausible date %r", date_str)
                return None
            return DateBounds(start, end, score)

        return None

    def parse_interval(self, date_str: str) -> DateBounds | None:
        """
        Parse 'A/B', 'A to B', or 'A - B' into the union of A's and B's bounds.

        The combined score is the minimum of the two endpoint scores when
        both endpoints expand to identical bounds, otherwise it's derived
        from the span of the union.
        """
        m = self.patterns["interval"].match(date_str)
        if not m:
            return None

        start_bounds = self.parse_single_date(m.group("start").strip())
        end_bounds = self.parse_single_date(m.group("end").strip())

        if not (start_bounds and end_bounds):
            return None

        if start_bounds.start > end_bounds.end:
            logger.warning(
                "Reversed interval %r - standardizing to [%s, %s].",
                date_str,
                min(start_bounds.start, end_bounds.start).isoformat(),
                max(start_bounds.end, end_bounds.end).isoformat(),
            )

        combined_start = min(start_bounds.start, end_bounds.start)
        combined_end = max(start_bounds.end, end_bounds.end)

        if start_bounds.start == end_bounds.start and start_bounds.end == end_bounds.end:
            score = min(start_bounds.score, end_bounds.score)
        else:
            score = _score_for_interval(combined_start, combined_end)

        return DateBounds(combined_start, combined_end, score)


# --- Main class ---


class DateStandardizer:
    def __init__(self, config_path: Path | str):
        self.config = load_config(config_path)
        patterns_raw = self.config.get("patterns", {})
        self.patterns = {k: re.compile(v) for k, v in patterns_raw.items()}
        self.parser = DateParser(self.patterns)

    def _parse_attribute_value(self, attribute: str, value: str) -> DateRecord | None:
        """Try interval parsing first, fall back to single-date."""
        bounds = self.parser.parse_interval(value) or self.parser.parse_single_date(value)
        if bounds is None:
            return None
        return DateRecord(bounds, attribute, value)

    def find_best_date(
        self,
        accession: str,
        attributes_str: str,
        date_values_str: str,
        categories_str: str,
    ) -> tuple[DateBounds, str, str] | None:
        """
        Pick the best [start, end] bounds for a record.

        Sampling-category dates win over other categories; among multiple
        sampling dates, equivalent ones collapse silently and divergent
        ones are merged into a date-range envelope with a warning.
        """
        attributes = split_pipe_separated(attributes_str)
        values = split_pipe_separated(date_values_str)
        categories = split_pipe_separated(categories_str)

        sampling: list[DateRecord] = []
        other: list[DateRecord] = []

        for attr, val, category in zip(attributes, values, categories, strict=False):
            record = self._parse_attribute_value(attr, val)
            if record is None:
                continue
            (sampling if category == "s" else other).append(record)

        if len(sampling) > 1:
            unique_bounds = {r.bounds for r in sampling}

            if len(unique_bounds) == 1:
                first = sampling[0]
                if any(r.value != first.value for r in sampling[1:]):
                    logger.info(
                        "%s: %d sampling dates collapsed to one (equivalent values)",
                        accession,
                        len(sampling),
                    )
                return first.bounds, first.attribute, first.value

            combined_start = min(r.bounds.start for r in sampling)
            combined_end = max(r.bounds.end for r in sampling)
            score = _score_for_interval(combined_start, combined_end)
            envelope = DateBounds(combined_start, combined_end, score)

            logger.warning(
                "%s has different valid sampling dates across %d distinct attributes: "
                "%s > [%s, %s].",
                accession,
                len(unique_bounds),
                envelope.start.isoformat(),
                envelope.end.isoformat(),
                " | ".join(f"{r.attribute}={r.value!r}" for r in sampling),
            )
            combined_attrs = "||".join(dict.fromkeys(r.attribute for r in sampling))
            combined_vals = "||".join(dict.fromkeys(r.value for r in sampling))
            return envelope, combined_attrs, combined_vals

        if sampling:
            first = sampling[0]
            return first.bounds, first.attribute, first.value
        if other:
            return self._fallback_non_sampling_date(other)
        return None

    @staticmethod
    def _fallback_non_sampling_date(records: list[DateRecord]) -> tuple[DateBounds, str, str]:
        oldest = min(records, key=lambda r: r.bounds.start)
        bounds = DateBounds(oldest.bounds.start, oldest.bounds.end, FALLBACK_SCORE)
        return bounds, oldest.attribute, oldest.value

    def process_file(
        self,
        input_path: Path,
        output_path: Path,
        pathogen: str | None = None,
        disable_progress: bool = False,
    ) -> None:
        output_header = [
            "accession",
            "date_start",
            "date_end",
            "date_score",
            "date_attr_orig",
            "date_val_orig",
        ]

        total = count_tsv_rows(input_path)
        bar_desc = f"date [{pathogen}]" if pathogen else "date"

        with (
            input_path.open("r", encoding="utf-8", newline="") as infile,
            output_path.open("w", encoding="utf-8", newline="") as outfile,
            make_inner_bar(total, bar_desc, disable=disable_progress) as bar,
        ):
            reader = csv.DictReader(infile, delimiter="\t")
            writer = csv.writer(outfile, delimiter="\t")
            writer.writerow(output_header)

            for row in reader:
                if pathogen and row.get("pathogen") != pathogen:
                    bar.update(1)
                    continue

                accession = row.get("accession", "")
                res = self.find_best_date(
                    accession,
                    row.get("date_attr_orig", ""),
                    row.get("date_val_orig", ""),
                    row.get("date_category", ""),
                )
                bounds, attribute, value = res
                writer.writerow(
                    [
                        accession,
                        bounds.start.isoformat(),
                        bounds.end.isoformat(),
                        bounds.score,
                        attribute,
                        value,
                    ]
                )
                bar.update(1)


def main(
    input_path: Path,
    output_path: Path,
    config_path: Path,
    log_level: str = "INFO",
    pathogen: str | None = None,
    disable_progress: bool = False,
) -> None:
    setup_standardizer_logging(logger, output_path, "date_standardized", log_level)
    standardizer = DateStandardizer(config_path)
    standardizer.process_file(
        input_path, output_path, pathogen=pathogen, disable_progress=disable_progress
    )


if __name__ == "__main__":
    parser = create_arg_parser(default_config_path="config/date.yaml")
    args = parser.parse_args()

    in_path = Path(args.input_file)
    out_path = Path(args.output_dir) / DATE_OUTPUT
    cfg_path = Path(args.config)

    main(in_path, out_path, cfg_path, args.log_level)
