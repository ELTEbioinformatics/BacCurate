"""Shared interface for per-attribute standardizers."""

from pathlib import Path
from typing import Protocol


class StandardizerMain(Protocol):
    def __call__(
        self,
        input_path: Path,
        output_path: Path,
        config_path: Path,
        log_level: str = "INFO",
        pathogen: str | None = None,
    ) -> None: ...
