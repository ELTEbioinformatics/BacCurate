"""Logging filter and per-script logger setup."""

import logging
from pathlib import Path


class AccessionDedupFilter(logging.Filter):
    """Suppresses repeat log records that differ only in their accession ID."""

    def __init__(self) -> None:
        super().__init__()
        self._seen: set[tuple] = set()

    def filter(self, record: logging.LogRecord) -> bool:
        args = record.args
        if isinstance(args, tuple):
            fingerprint_args = repr(args[1:])
        else:
            fingerprint_args = repr(args)
        fingerprint = (record.levelno, record.msg, fingerprint_args)
        if fingerprint in self._seen:
            return False
        self._seen.add(fingerprint)
        return True


def setup_logging(output_dir: str | Path, script_name: str, log_level_name: str = "INFO"):
    """
    Attach a per-script log file to the root logger.

    Adds a `FileHandler` writing to `<output_dir>/logs/<script_name>.log` and
    sets the root logger's level. Existing stream handlers are left in place
    so any tqdm-aware redirect set up by the caller (e.g. the pipeline
    orchestrator's `progress_context`) stays in effect.

    If a file handler for the same path is already attached, it is removed
    first so re-runs start the log fresh.
    """
    output_dir = Path(output_dir)
    log_dir = output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    log_level = getattr(logging, log_level_name.upper(), logging.INFO)
    log_format = "%(asctime)s - %(levelname)s - %(message)s"

    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    log_file_path = log_dir / f"{script_name}.log"
    for handler in list(root_logger.handlers):
        if isinstance(handler, logging.FileHandler) and Path(handler.baseFilename) == log_file_path:
            root_logger.removeHandler(handler)
            handler.close()

    file_handler = logging.FileHandler(log_file_path, mode="w")
    file_handler.setFormatter(logging.Formatter(log_format))
    root_logger.addHandler(file_handler)


def setup_standardizer_logging(
    module_logger: logging.Logger,
    output_path: str | Path,
    script_name: str,
    log_level: str = "INFO",
) -> None:
    """logging setup for a standardizer script's main() function.

    Applies the shared stdout format, attaches a
    per-script ``<output_dir>/logs/<script_name>.log`` file handler via
    :func:`setup_logging`, and adds the accession-dedup filter to
    ``module_logger``. ``output_path`` is the standardizer's output file, its
    parent directory is used as the log location.
    """
    logging.basicConfig(
        level=log_level.upper(),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    setup_logging(Path(output_path).parent, script_name, log_level)
    module_logger.addFilter(AccessionDedupFilter())
