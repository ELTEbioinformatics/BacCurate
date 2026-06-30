"""Wrappers around `tqdm` for the pipeline's progress reporting.

Two stacked bars are used during a run:
- position=0: outer queue bar over (pathogen, attribute) slots
- position=1: inner per-iteration bar over rows / files

`progress_context()` redirects `logging` output through `tqdm.write` so log
lines from the standardizers don't interfere with the bars.
"""

from contextlib import contextmanager, nullcontext
from pathlib import Path

from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

# Format strings tuned for the pipeline. The inner format leaves room for a
# wall-clock ETA + rate; the outer one strips rate (queue throughput is not
# a useful number to look at).
_INNER_FMT = "{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"
_OUTER_FMT = "{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]"


def count_tsv_rows(path: Path) -> int:
    """
    Return the number of data rows (lines minus header) in a TSV.

    Used to set `total` on the inner bar so ETA is meaningful.
    """
    with path.open("r", encoding="utf-8") as f:
        count = sum(1 for _ in f)
    return max(count - 1, 0)


def make_outer_bar(total_slots: int, *, disable: bool = False) -> tqdm:
    """Pipeline-queue bar. One unit = one (pathogen, attribute) slot."""
    return tqdm(
        total=total_slots,
        desc="Pipeline queue",
        position=0,
        leave=True,
        disable=disable,
        bar_format=_OUTER_FMT,
        unit="slot",
    )


def make_inner_bar(total: int, desc: str, *, disable: bool = False) -> tqdm:
    """
    Per-iteration bar. One unit = one row (or file, for extraction).

    `unit_scale=True` applies SI suffixes (k/M/G) to the count and rate so
    millions of rows render as e.g. "599k/1.41M [178krec/s]" instead of the
    raw nine-digit numbers.
    """
    return tqdm(
        total=total,
        desc=desc,
        position=1,
        leave=False,
        disable=disable,
        bar_format=_INNER_FMT,
        unit="rec",
        unit_scale=True,
    )


@contextmanager
def progress_context(disable: bool = False):
    """
    Route `logging` output through `tqdm.write` for the duration.

    No-op when bars are disabled - keeps log formatting identical to a
    bar-less run.
    """
    if disable:
        with nullcontext():
            yield
    else:
        with logging_redirect_tqdm():
            yield
