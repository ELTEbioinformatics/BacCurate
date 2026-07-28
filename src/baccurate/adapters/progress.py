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
    """Create a per-iteration bar, with one unit per row or extraction file.

    `unit_scale=True` adds SI suffixes (k/M/G) to the count and rate, so millions
    of rows appear as "599k/1.41M [178krec/s]" instead of raw nine-digit numbers.
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
    """Route `logging` output through `tqdm.write` while the context is active.

    When bars are disabled, the context does nothing and preserves the log
    formatting used in a run without progress bars.
    """
    if disable:
        with nullcontext():
            yield
    else:
        with logging_redirect_tqdm():
            yield
