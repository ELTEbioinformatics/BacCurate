"""BioSample XML streaming and attribute extraction helpers."""

from __future__ import annotations

import logging
import re
from collections.abc import Callable, Iterator
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING

from lxml import etree

from baccurate.utils.compressed_io import open_binary

if TYPE_CHECKING:
    from baccurate.extraction.policy import PolicyDecision

logger = logging.getLogger(__name__)

ROOT_DATE_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}")


@dataclass(slots=True)
class CandidateCounters:
    """Count each source attribute and value once,
    even when it affects several standardizers."""

    inspected: int = 0
    identified: int = 0
    rejected: int = 0
    unreviewed: int = 0
    found: int = 0
    filtered: int = 0
    multiply_matched: int = 0

    def record(self, decision: PolicyDecision) -> None:
        self.inspected += 1
        event_kinds = {event.kind for event in decision.events}
        if decision.matches or "rejected_value" in event_kinds:
            self.identified += 1
        if "rejected_value" in event_kinds:
            self.rejected += 1
        if "unreviewed_attribute" in event_kinds:
            self.unreviewed += 1
        if decision.matches:
            self.found += 1
        else:
            self.filtered += 1
        if len(decision.matches) > 1:
            self.multiply_matched += 1

    def summary(self) -> str:
        return (
            f"inspected={self.inspected}, identified={self.identified}, "
            f"found={self.found}, rejected={self.rejected}, "
            f"unreviewed={self.unreviewed}, filtered={self.filtered}, "
            f"multiply_matched={self.multiply_matched}"
        )


def _iter_biosample_candidates_with_source(
    record: etree._Element,
    *,
    include_root_dates: bool = False,
) -> Iterator[tuple[str, str, str]]:
    """Yield raw candidates while retaining their XML structural source."""
    if include_root_dates:
        for attr_name, value in record.attrib.items():
            if value and ROOT_DATE_PATTERN.match(str(value)):
                yield attr_name, str(value), "biosample_root"

    attributes_container = record.find("Attributes")
    if attributes_container is None:
        return

    for attr_elem in attributes_container.findall("Attribute"):
        attr_name = attr_elem.get("harmonized_name") or attr_elem.get("attribute")
        if any(isinstance(node, etree._Entity) for node in attr_elem.iterdescendants()):
            logger.warning("Skipping attribute %r containing an unresolved XML entity", attr_name)
            continue
        value = attr_elem.text
        if attr_name and value:
            yield attr_name, value, "attribute"


def _clear_record(record: etree._Element) -> None:
    """Release a streamed record and siblings already processed by libxml2."""
    record.clear()
    parent = record.getparent()
    if parent is not None:
        while record.getprevious() is not None:
            del parent[0]


def iter_biosample_records(input_file: str | Path) -> Iterator[etree._Element]:
    """Stream BioSample elements with bounded memory."""
    with open_binary(input_file) as stream:
        context = etree.iterparse(
            stream,
            events=("end",),
            tag="BioSample",
            load_dtd=False,
            no_network=True,
            resolve_entities=False,
            huge_tree=True,
            collect_ids=False,
        )
        try:
            for _event, record in context:
                try:
                    yield record
                finally:
                    _clear_record(record)
        finally:
            del context


def parse_xml(
    record: etree._Element,
    evaluate_function: Callable[..., PolicyDecision],
    check_root_attributes: bool = False,
    counters: CandidateCounters | None = None,
) -> list[PolicyDecision]:
    candidates = []

    for attr_name, value, xml_source in _iter_biosample_candidates_with_source(
        record,
        include_root_dates=check_root_attributes,
    ):
        decision = replace(
            evaluate_function(attribute=attr_name, value=value),
            xml_source=xml_source,
        )
        if counters is not None:
            counters.record(decision)
        candidates.append(decision)

    return candidates


def _extract_bioproject(elem: etree._Element) -> str:
    for link in elem.findall("./Links/Link"):
        if link.get("target") == "bioproject" and link.text:
            return link.text.strip()
    return ""


def process_biosample_xml(
    input_file: str | Path,
    evaluate_function: Callable[..., PolicyDecision],
    counters: CandidateCounters | None = None,
) -> Iterator[tuple[str, list[PolicyDecision], str, str]]:
    """Stream XML, yielding accession, candidate decisions, package, and BioProject."""
    records_processed = 0

    try:
        for elem in iter_biosample_records(input_file):
            records_processed += 1

            try:
                accession = elem.get("accession", "unknown")

                package_elem = elem.find("Package")
                package_name = package_elem.text if package_elem is not None else ""

                bioproject = _extract_bioproject(elem)

                candidates = parse_xml(
                    elem,
                    evaluate_function,
                    check_root_attributes=True,
                    counters=counters,
                )
                if candidates:
                    yield accession, candidates, package_name, bioproject

            except Exception as e:
                logger.error("Error parsing record %d: %s", records_processed, e, exc_info=True)

    except Exception as e:
        logger.critical(f"XML Iteration Error: {e}", exc_info=True)
        raise
