"""Canonical identity for model requests whose results may be cached."""

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass


def _serialize_canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def canonical_json_sha256(value: object) -> str:
    """Hash an arbitrary JSON-compatible contract using canonical serialization."""
    return _sha256_text(_serialize_canonical_json(value))


@dataclass(frozen=True, slots=True)
class CanonicalLLMRequest:
    """The complete response-affecting contract for one model request."""

    model: str
    messages: tuple[Mapping[str, str], ...]
    parameters: Mapping[str, object]
    response_schema_id: str

    def serialize(self) -> str:
        """Return stable JSON while preserving the order of rendered messages."""
        return _serialize_canonical_json(
            {
                "messages": [dict(message) for message in self.messages],
                "model": self.model,
                "parameters": dict(self.parameters),
                "response_schema_id": self.response_schema_id,
            }
        )

    @property
    def fingerprint(self) -> str:
        """Return the SHA-256 cache key for the canonical serialization."""
        return _sha256_text(self.serialize())
