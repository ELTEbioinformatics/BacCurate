"""Build TSV reports for metadata that needs manual review."""

from __future__ import annotations

import csv
import json
import logging
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from baccurate.extraction.policy import PolicyDecision

_REPRESENTATIVE_LIMIT = 3
_UNREVIEWED_COLUMNS = (
    "target",
    "family",
    "normalized_attribute",
    "count",
    "representative_examples",
)
_UNCERTAIN_COLUMNS = (
    "target",
    "family",
    "normalized_value",
    "count",
    "representative_examples",
)


def _remember[Representative](
    representatives: set[Representative], example: Representative
) -> None:
    """Keep up to three examples, chosen consistently regardless of input order."""
    representatives.add(example)
    if len(representatives) > _REPRESENTATIVE_LIMIT:
        representatives.remove(max(representatives))


@dataclass(slots=True)
class _UnreviewedBucket:
    count: int = 0
    examples: set[tuple[str, str, str]] = field(default_factory=set)

    def add(self, decision: PolicyDecision, accession: str) -> None:
        self.count += 1
        _remember(self.examples, (decision.attribute, decision.value, accession))

    def render_examples(self) -> str:
        examples = [
            {"attribute": attribute, "value": value, "accession": accession}
            for attribute, value, accession in sorted(self.examples)
        ]
        return json.dumps(examples, ensure_ascii=False)


@dataclass(slots=True)
class _UncertainBucket:
    count: int = 0
    examples: set[tuple[str, str, str]] = field(default_factory=set)

    def add(self, decision: PolicyDecision, accession: str) -> None:
        self.count += 1
        _remember(self.examples, (decision.attribute, decision.value, accession))

    def render_examples(self) -> str:
        examples = [
            {"attribute": attribute, "value": value, "accession": accession}
            for attribute, value, accession in sorted(self.examples)
        ]
        return json.dumps(examples, ensure_ascii=False)


class ReviewReports:
    """Collect counts and a few examples for the reports written after extraction."""

    def __init__(self) -> None:
        self._unreviewed: dict[tuple[str, str, str], _UnreviewedBucket] = {}
        self._uncertain: dict[tuple[str, str, str], _UncertainBucket] = {}
        self._automatic_rejections: Counter[tuple[str, str]] = Counter()

    @property
    def has_unreviewed(self) -> bool:
        return bool(self._unreviewed)

    def observe(self, decision: PolicyDecision, *, accession: str) -> None:
        for event in decision.events:
            if event.kind == "unreviewed_attribute":
                key = (event.target, event.family, event.normalized_attribute)
                self._unreviewed.setdefault(key, _UnreviewedBucket()).add(decision, accession)
            elif event.kind == "uncertain_rejection":
                key = (event.target, event.family, event.normalized_value)
                self._uncertain.setdefault(key, _UncertainBucket()).add(decision, accession)
            elif event.kind == "rejected_value":
                self._automatic_rejections[(event.target, event.family)] += 1

    def write(self, directory: Path) -> None:
        self._write_unreviewed(directory / "unreviewed_attributes.tsv")
        self._write_uncertain(directory / "uncertain_rejections.tsv")

    def log_automatic_rejections(self, logger: logging.Logger) -> None:
        if not self._automatic_rejections:
            return
        summary = ", ".join(
            f"{target}/{family}={count}"
            for (target, family), count in sorted(self._automatic_rejections.items())
        )
        logger.info("Automatic metadata rejections: %s", summary)

    def _write_unreviewed(self, path: Path) -> None:
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream, delimiter="\t", lineterminator="\n")
            writer.writerow(_UNREVIEWED_COLUMNS)
            for (target, family, normalized), bucket in sorted(self._unreviewed.items()):
                writer.writerow(
                    (
                        target,
                        family,
                        normalized,
                        bucket.count,
                        bucket.render_examples(),
                    )
                )

    def _write_uncertain(self, path: Path) -> None:
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream, delimiter="\t", lineterminator="\n")
            writer.writerow(_UNCERTAIN_COLUMNS)
            for (target, family, normalized), bucket in sorted(self._uncertain.items()):
                writer.writerow(
                    (
                        target,
                        family,
                        normalized,
                        bucket.count,
                        bucket.render_examples(),
                    )
                )
