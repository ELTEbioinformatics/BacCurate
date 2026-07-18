"""Shared utilities for configuration, logging, and text handling."""

from baccurate.utils.config import load_config
from baccurate.utils.text import normalize_keyword

__all__ = [
    "load_config",
    "normalize_keyword",
]
