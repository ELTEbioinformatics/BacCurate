"""Streaming access to uncompressed and gzip-compressed files."""

import gzip
from pathlib import Path
from typing import IO, Literal


def open_text(
    path: str | Path,
    mode: Literal["r", "w", "a"] = "r",
    *,
    encoding: str = "utf-8",
    newline: str | None = None,
) -> IO[str]:
    """Open plain text or a gzip-compressed text file based on its suffix."""
    path = Path(path)
    if path.suffix == ".gz":
        return gzip.open(path, f"{mode}t", encoding=encoding, newline=newline)
    return path.open(mode, encoding=encoding, newline=newline)


def open_binary(path: str | Path, mode: Literal["r", "w", "a"] = "r") -> IO[bytes]:
    """Open a plain or gzip-compressed binary file based on its suffix."""
    path = Path(path)
    if path.suffix == ".gz":
        return gzip.GzipFile(filename=path, mode=mode)
    return path.open(f"{mode}b")
