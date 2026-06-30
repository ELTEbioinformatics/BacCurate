"""BioSample XML streaming and attribute extraction helpers."""

import logging
import re
import xml.etree.ElementTree as ET
from collections.abc import Callable, Iterator
from typing import Any

logger = logging.getLogger(__name__)

ROOT_DATE_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}")


def parse_xml(
    record: str | ET.Element,
    check_function: Callable[[str, str], bool],
    check_root_attributes: bool = False,
) -> list[dict[str, Any]]:
    found_attrs = []

    def add_finding(val: str, name: str, attr: bool):
        found_attrs.append({"attribute": name, "value": val, "is_valid_attr": attr})

    if check_root_attributes:
        for attr_name, val in record.attrib.items():
            if val and ROOT_DATE_PATTERN.match(str(val)):
                is_valid_attr = check_function(val, attr_name)
                if is_valid_attr:
                    add_finding(val, attr_name, is_valid_attr)

    attributes_container = record.find("Attributes")

    if attributes_container is not None:
        for attr_elem in attributes_container.findall("Attribute"):
            attr = attr_elem.get("harmonized_name") or attr_elem.get("attribute")
            val = attr_elem.text

            if attr and val:
                is_valid_attr = check_function(val, attr)

                if is_valid_attr:
                    add_finding(val, attr, is_valid_attr)

    return found_attrs


def _extract_bioproject(elem: ET.Element) -> str:
    for link in elem.findall("./Links/Link"):
        if link.get("target") == "bioproject" and link.text:
            return link.text.strip()
    return ""


def process_biosample_xml(
    input_file: str,
    check_function: Callable[[str, str], bool],
) -> Iterator[tuple[str, list[dict[str, Any]], str, str]]:
    """Stream a BioSample XML file, yielding (accession, found_attrs, package_name, bioproject) per record."""
    records_processed = 0

    try:
        context = ET.iterparse(input_file, events=("end",))
        context = iter(context)

        for _event, elem in context:
            if elem.tag == "BioSample":
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
                    logger.error(f"Error parsing record {records_processed}: {e}", exc_info=True)

                finally:
                    elem.clear()

    except Exception as e:
        logger.critical(f"XML Iteration Error: {e}", exc_info=True)
        raise
