"""Shared utilities: logging, config, CLI helpers, XML parsing, IO writers."""

from baccurate.utils.args import create_arg_parser
from baccurate.utils.config import load_config
from baccurate.utils.logging import AccessionDedupFilter, setup_logging
from baccurate.utils.text import normalize_keyword

__all__ = [
    "AccessionDedupFilter",
    "create_arg_parser",
    "load_config",
    "normalize_keyword",
    "setup_logging",
]
