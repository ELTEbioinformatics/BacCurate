"""Wrappers around `tqdm` for the pipeline's per-iteration progress reporting.

`progress_context()` redirects `logging` output through `tqdm.write` so log
lines from the standardizers don't interfere with the bar.
"""

from contextlib import contextmanager, nullcontext

from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

# Format string tuned for the pipeline's wall-clock ETA and rate.
_BAR_FORMAT = "{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"


def make_progress_bar(
    total: int,
    desc: str,
    *,
    disable: bool = False,
    position: int = 1,
) -> tqdm:
    """
    Per-iteration bar. One unit = one row (or file, for extraction).

    `unit_scale=True` applies SI suffixes (k/M/G) to the count and rate so
    millions of rows render as e.g. "599k/1.41M [178krec/s]" instead of the
    raw nine-digit numbers.
    """
    return tqdm(
        total=total,
        desc=desc,
        position=position,
        leave=False,
        disable=disable,
        bar_format=_BAR_FORMAT,
        unit="rec",
        unit_scale=True,
    )


@contextmanager
def progress_context(disable: bool = False):
    """
    Route `logging` output through `tqdm.write` while the context is active.

    No-op when bars are disabled - keeps log formatting identical to a
    bar-less run.
    """
    if disable:
        with nullcontext():
            yield
    else:
        with logging_redirect_tqdm():
            yield
