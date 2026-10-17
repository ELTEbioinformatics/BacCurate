"""SHA-256 digests of files."""

import hashlib
from pathlib import Path


def sha256_file(path: Path) -> str:
    """Return the lowercase SHA-256 digest of a file's bytes."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()
