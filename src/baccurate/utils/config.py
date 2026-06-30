"""YAML configuration loader."""

import logging
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)


def load_config(config_path: str | Path) -> dict:
    """Load a YAML configuration file."""
    abs_config_path = Path(config_path).resolve()
    if not abs_config_path.exists():
        raise FileNotFoundError(f"Configuration file not found at: {abs_config_path}")
    try:
        with open(abs_config_path, encoding="utf-8") as f:
            return yaml.safe_load(f)
    except yaml.YAMLError as e:
        logger.error("Couldn't parse YAML configuration: %s", e)
        raise
