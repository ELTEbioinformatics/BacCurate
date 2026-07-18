"""Logging helpers."""

import logging
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


class BoundedDebugFilter(logging.Filter):
    """Let each distinct DEBUG message through once per logger, dropping repeats."""

    def __init__(self) -> None:
        super().__init__()
        self._seen_debug_events: set[tuple[str, str]] = set()

    def filter(self, record: logging.LogRecord) -> bool:
        if getattr(record, "llm_trace_unbounded", False):
            return True
        if record.levelno != logging.DEBUG or not record.name.startswith("baccurate"):
            return True
        event = (record.name, str(record.msg))
        if event in self._seen_debug_events:
            return False
        self._seen_debug_events.add(event)
        return True


class ConsoleDetailFilter(logging.Filter):
    """Keep file-only LLM details out of console output."""

    def filter(self, record: logging.LogRecord) -> bool:
        return not getattr(record, "llm_file_only", False)


class LifecycleInfoFilter(logging.Filter):
    """Allow INFO only from the main CLI lifecycle logger."""

    def filter(self, record: logging.LogRecord) -> bool:
        return record.levelno != logging.INFO or record.name == "baccurate.main"


class LocalOffsetFormatter(logging.Formatter):
    """Format log timestamps in local time as ISO 8601 with the UTC offset."""

    def formatTime(self, record: logging.LogRecord, datefmt: str | None = None) -> str:
        del datefmt
        return datetime.fromtimestamp(record.created).astimezone().isoformat(timespec="seconds")


@dataclass(slots=True)
class RunLogging:
    """Installed handlers and prior logger state for one CLI invocation."""

    handlers: tuple[logging.Handler, logging.Handler]
    root_level: int
    package_level: int
    package_propagate: bool

    def close(self) -> None:
        root_logger = logging.getLogger()
        package_logger = logging.getLogger("baccurate")
        for handler in self.handlers:
            root_logger.removeHandler(handler)
            package_logger.removeHandler(handler)
            handler.close()
        root_logger.setLevel(self.root_level)
        package_logger.setLevel(self.package_level)
        package_logger.propagate = self.package_propagate


def configure_run_logging(log_path: Path, *, console_debug: bool) -> RunLogging:
    """Install file and console logging for a single pipeline run.

    The file handler at log_path records full DEBUG detail.

    The console handler shows INFO (or DEBUG when console_debug is set)
    and hides the file-only LLM traces. The root logger is pinned to
    WARNING so third-party libraries stay quiet unless something goes wrong.

    Returns the prior logger state so the caller can restore it with
    ``RunLogging.close()`` once the run finishes.
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(
        LocalOffsetFormatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    )
    file_handler.addFilter(BoundedDebugFilter())
    file_handler.addFilter(LifecycleInfoFilter())
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG if console_debug else logging.INFO)
    console_handler.setFormatter(logging.Formatter("%(levelname)s | %(message)s"))
    console_handler.addFilter(BoundedDebugFilter())
    console_handler.addFilter(LifecycleInfoFilter())
    console_handler.addFilter(ConsoleDetailFilter())

    root_logger = logging.getLogger()
    package_logger = logging.getLogger("baccurate")
    state = RunLogging(
        handlers=(file_handler, console_handler),
        root_level=root_logger.level,
        package_level=package_logger.level,
        package_propagate=package_logger.propagate,
    )
    root_logger.setLevel(logging.WARNING)
    package_logger.setLevel(logging.DEBUG)
    package_logger.propagate = False
    for handler in state.handlers:
        root_logger.addHandler(handler)
        package_logger.addHandler(handler)
    return state
