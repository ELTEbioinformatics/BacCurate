"""BioSample XML streaming and attribute extraction helpers."""

import logging
import re
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

from lxml import etree

from baccurate.utils.compressed_io import open_binary

logger = logging.getLogger(__name__)

ROOT_DATE_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}")


def iter_biosample_candidates(
    record: etree._Element,
    *,
    include_root_dates: bool = False,
) -> Iterator[tuple[str, str]]:
    """Yield attribute/value candidates in extraction order."""
    if include_root_dates:
        for attr_name, value in record.attrib.items():
            if value and ROOT_DATE_PATTERN.match(str(value)):
                yield attr_name, str(value)

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
            yield attr_name, value


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
    check_function: Callable[[str, str], bool],
    check_root_attributes: bool = False,
) -> list[dict[str, Any]]:
    found_attrs = []

    def add_finding(val: str, name: str, attr: bool):
        found_attrs.append({"attribute": name, "value": val, "is_valid_attr": attr})

    for attr_name, value in iter_biosample_candidates(
        record,
        include_root_dates=check_root_attributes,
    ):
        is_valid_attr = check_function(value, attr_name)
        if is_valid_attr:
            add_finding(value, attr_name, is_valid_attr)

    return found_attrs


def _extract_bioproject(elem: etree._Element) -> str:
    for link in elem.findall("./Links/Link"):
        if link.get("target") == "bioproject" and link.text:
            return link.text.strip()
    return ""


def process_biosample_xml(
    input_file: str | Path,
    check_function: Callable[[str, str], bool],
) -> Iterator[tuple[str, list[dict[str, Any]], str, str]]:
    """Stream XML, yielding accession, attributes, package, and BioProject per record."""
    records_processed = 0

    try:
        for elem in iter_biosample_records(input_file):
            records_processed += 1

            try:
                accession = elem.get("accession", "unknown")

                package_elem = elem.find("Package")
                package_name = package_elem.text if package_elem is not None else ""

                bioproject = _extract_bioproject(elem)

                found_attrs = parse_xml(elem, check_function, check_root_attributes=True)
                if found_attrs:
                    yield accession, found_attrs, package_name, bioproject

            except Exception as e:
                logger.error("Error parsing record %d: %s", records_processed, e, exc_info=True)

    except Exception as e:
        logger.critical(f"XML Iteration Error: {e}", exc_info=True)
        raise
