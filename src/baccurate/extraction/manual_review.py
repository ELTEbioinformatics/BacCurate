"""Build TSV worklists for metadata that needs manual review."""

from __future__ import annotations

import csv
import json
import logging
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from baccurate.extraction.selection import SelectionDecision

_REPRESENTATIVE_LIMIT = 3
_SELECTION_REVIEW_KINDS = ("unreviewed_attribute", "uncertain_rejection")
_SELECTION_REVIEW_COLUMNS = (
    "kind",
    "target",
    "family",
    "normalized_attribute",
    "normalized_value",
    "count",
    "representative_examples",
)


def keep_representative[Example](examples: set[Example], example: Example) -> None:
    """Keep up to three examples, chosen consistently regardless of input order."""
    examples.add(example)
    if len(examples) > _REPRESENTATIVE_LIMIT:
        examples.remove(max(examples))


@dataclass(slots=True)
class _ReviewBucket:
    count: int = 0
    examples: set[tuple[str, str, str]] = field(default_factory=set)

    def add(self, decision: SelectionDecision, accession: str) -> None:
        self.count += 1
        keep_representative(self.examples, (decision.attribute, decision.value, accession))

    def render_examples(self) -> str:
        examples = [
            {"attribute": attribute, "value": value, "accession": accession}
            for attribute, value, accession in sorted(self.examples)
        ]
        return json.dumps(examples, ensure_ascii=False)


class ReviewWorklists:
    """Collect counts and examples for the review worklists written after extraction."""

    def __init__(self) -> None:
        self._buckets: dict[tuple[str, str, str, str, str], _ReviewBucket] = {}
        self._automatic_rejections: Counter[tuple[str, str]] = Counter()

    @property
    def unreviewed_count(self) -> int:
        """Return the number of unreviewed attribute occurrences."""
        return self._count("unreviewed_attribute")

    @property
    def uncertain_count(self) -> int:
        """Return the number of uncertain rejection occurrences."""
        return self._count("uncertain_rejection")

    @property
    def automatic_rejection_counts(self) -> dict[str, dict[str, int]]:
        """Return automatic rejection counts grouped by target and family."""
        counts: dict[str, dict[str, int]] = {}
        for (target, family), count in sorted(self._automatic_rejections.items()):
            counts.setdefault(target, {})[family] = count
        return counts

    def observe(self, decision: SelectionDecision, *, accession: str) -> None:
        for event in decision.events:
            if event.kind in _SELECTION_REVIEW_KINDS:
                key = (
                    event.kind,
                    event.target,
                    event.family,
                    event.normalized_attribute,
                    event.normalized_value,
                )
                self._buckets.setdefault(key, _ReviewBucket()).add(decision, accession)
            elif event.kind == "rejected_value":
                self._automatic_rejections[(event.target, event.family)] += 1

    def write(self, directory: Path) -> dict[str, Path]:
        path = directory / "unreviewed_attributes.tsv"
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream, delimiter="\t", lineterminator="\n")
            writer.writerow(_SELECTION_REVIEW_COLUMNS)
            for key, bucket in sorted(self._buckets.items()):
                writer.writerow((*key, bucket.count, bucket.render_examples()))
        return {"unreviewed_attributes": path} if self._buckets else {}

    def log_automatic_rejections(self, logger: logging.Logger) -> None:
        if not self._automatic_rejections:
            return
        summary = ", ".join(
            f"{target}/{family}={count}"
            for (target, family), count in sorted(self._automatic_rejections.items())
        )
        logger.info("Automatic metadata rejections: %s", summary)

    def _count(self, kind: str) -> int:
        return sum(bucket.count for key, bucket in self._buckets.items() if key[0] == kind)
