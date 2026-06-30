"""Shared SQLite key-value cache base for the standardizer caches."""

import hashlib
import sqlite3
from pathlib import Path
from typing import ClassVar


class SQLiteKVCache:
    """Base for the SQLite-backed standardizer caches.

    Handles connection setup, table creation, teardown, and
    the SHA-256 primitive.
    """

    _CREATE_TABLE_SQL: ClassVar[str]

    def __init__(self, db_path: Path | str) -> None:
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(db_path)
        self.cursor = self.conn.cursor()
        self.cursor.execute(self._CREATE_TABLE_SQL)
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()

    @staticmethod
    def _sha256(s: str) -> str:
        return hashlib.sha256(s.encode("utf-8")).hexdigest()
