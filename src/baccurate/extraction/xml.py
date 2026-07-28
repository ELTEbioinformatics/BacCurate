"""BioSample XML streaming and attribute extraction helpers."""

from __future__ import annotations

import logging
import re
from collections.abc import Callable, Iterator
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING

from lxml import etree

from baccurate.adapters.compressed_io import open_binary
from baccurate.extraction.bioproject import require_numeric_bioproject_id

if TYPE_CHECKING:
    from baccurate.extraction.curation import CurationDecision

logger = logging.getLogger(__name__)

ROOT_DATE_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}")


@dataclass(slots=True)
class CurationCounters:
    """Count each BioSample attribute-value pair once.

    A pair may inform several standardization targets. The counters preserve
    the invariant ``inspected == selected + unselected``.
    """

    inspected: int = 0
    identified: int = 0
    automatically_rejected: int = 0
    unreviewed: int = 0
    selected: int = 0
    unselected: int = 0
    multiply_matched: int = 0

    def record(self, decision: CurationDecision) -> None:
        self.inspected += 1
        event_kinds = {event.kind for event in decision.events}
        if decision.matches or "rejected_value" in event_kinds:
            self.identified += 1
        if "rejected_value" in event_kinds:
            self.automatically_rejected += 1
        if "unreviewed_attribute" in event_kinds:
            self.unreviewed += 1
        if decision.matches:
            self.selected += 1
        else:
            self.unselected += 1
        if len(decision.matches) > 1:
            self.multiply_matched += 1

    def summary(self) -> str:
        return (
            f"inspected={self.inspected}, identified={self.identified}, "
            f"selected={self.selected}, automatically_rejected={self.automatically_rejected}, "
            f"unreviewed={self.unreviewed}, unselected={self.unselected}, "
            f"multiply_matched={self.multiply_matched}"
        )


def _iter_biosample_pairs_with_xml_element(
    biosample_record: etree._Element,
    *,
    include_root_dates: bool = False,
) -> Iterator[tuple[str, str, str]]:
    """Yield BioSample attribute-value pairs with their XML structural origin."""
    if include_root_dates:
        for attr_name, value in biosample_record.attrib.items():
            if value and ROOT_DATE_PATTERN.match(str(value)):
                yield attr_name, str(value), "biosample_root"

    attributes_container = biosample_record.find("Attributes")
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


def _clear_biosample_record(biosample_record: etree._Element) -> None:
    """Release a streamed BioSample record and previously processed siblings."""
    biosample_record.clear()
    parent = biosample_record.getparent()
    if parent is not None:
        while biosample_record.getprevious() is not None:
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
            for _event, biosample_record in context:
                try:
                    yield biosample_record
                finally:
                    _clear_biosample_record(biosample_record)
        finally:
            del context


def parse_xml(
    biosample_record: etree._Element,
    evaluate_function: Callable[..., CurationDecision],
    check_root_attributes: bool = False,
    counters: CurationCounters | None = None,
) -> list[CurationDecision]:
    decisions = []

    for attr_name, value, xml_element in _iter_biosample_pairs_with_xml_element(
        biosample_record,
        include_root_dates=check_root_attributes,
    ):
        decision = replace(
            evaluate_function(attribute=attr_name, value=value),
            xml_element=xml_element,
        )
        if counters is not None:
            counters.record(decision)
        decisions.append(decision)

    return decisions


def _extract_bioprojects(elem: etree._Element) -> tuple[str, ...]:
    project_ids = {
        link.text.strip()
        for link in elem.findall("./Links/Link")
        if link.get("target") == "bioproject" and link.text and link.text.strip()
    }
    ordered_ids = sorted(project_ids)
    for project_id in ordered_ids:
        require_numeric_bioproject_id(project_id)
    return tuple(ordered_ids)


def process_biosample_xml(
    input_file: str | Path,
    evaluate_function: Callable[..., CurationDecision],
    counters: CurationCounters | None = None,
) -> Iterator[tuple[str, list[CurationDecision], tuple[str, ...]]]:
    """Stream XML, yielding accession, curation decisions, and linked BioProjects."""
    for elem in iter_biosample_records(input_file):
        accession = elem.get("accession", "unknown")

        bioprojects = _extract_bioprojects(elem)

        decisions = parse_xml(
            elem,
            evaluate_function,
            check_root_attributes=True,
            counters=counters,
        )
        yield accession, decisions, bioprojects
