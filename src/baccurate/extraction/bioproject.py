"""Resolve linked BioProjects into compact context records."""

from __future__ import annotations

import html
import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from lxml import etree
from lxml import html as lxml_html

from baccurate.source_snapshot import SourceSnapshotError
from baccurate.utils.compressed_io import open_binary

_ORIGIN_RELEVANCE = ("Agricultural", "Environmental", "Veterinary")
_AFFIRMATIVE_VALUES = frozenset({"1", "true", "yes"})
_BLOCK_ELEMENTS = frozenset(
    {
        "article",
        "blockquote",
        "br",
        "dd",
        "div",
        "dl",
        "dt",
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
        "h6",
        "hr",
        "li",
        "ol",
        "p",
        "pre",
        "section",
        "table",
        "td",
        "th",
        "tr",
        "ul",
    }
)


@dataclass(frozen=True, slots=True)
class BioProjectContext:
    """Resolved context retained once for one linked BioProject."""

    id: str
    accession: str
    title: str
    description: str
    relevance: tuple[str, ...]

    def catalog_object(self) -> dict[str, object]:
        return {
            "id": self.id,
            "accession": self.accession,
            "title": self.title,
            "description": self.description,
            "relevance": list(self.relevance),
        }


def resolve_bioproject_contexts(
    input_file: str | Path, linked_ids: Iterable[str]
) -> dict[str, BioProjectContext]:
    """Stream BioProject XML and retain explicitly linked records."""
    wanted = frozenset(linked_ids)
    if not wanted:
        return {}

    resolved: dict[str, BioProjectContext] = {}
    with open_binary(input_file) as stream:
        context = etree.iterparse(
            stream,
            events=("end",),
            tag=("Package", "DocumentSummary"),
            load_dtd=False,
            no_network=True,
            resolve_entities=False,
            huge_tree=True,
            collect_ids=False,
        )
        try:
            for _event, record in context:
                try:
                    archive_id = record.find(".//ArchiveID")
                    project_id = (
                        (archive_id.get("id") or "").strip() if archive_id is not None else ""
                    )
                    if project_id not in wanted:
                        continue
                    resolved[project_id] = _project_context(record, archive_id)
                finally:
                    _clear_record(record)
        finally:
            del context
    return resolved


def write_bioproject_catalog(
    contexts: Iterable[BioProjectContext], destination: Path | str
) -> Path:
    """Write a JSON object per resolved project."""
    path = Path(destination)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for context in sorted(contexts, key=lambda item: item.accession):
            stream.write(
                json.dumps(
                    context.catalog_object(),
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
                + "\n"
            )
    return path


def _project_context(record: etree._Element, archive_id: etree._Element) -> BioProjectContext:
    project_id = (archive_id.get("id") or "").strip()
    accession = (archive_id.get("accession") or "").strip()
    title = _clean_project_text(record.find(".//ProjectDescr/Title"))
    description = _clean_project_text(record.find(".//ProjectDescr/Description"))
    if not project_id or not accession or not title:
        raise SourceSnapshotError(
            "Linked BioProject record requires numeric ID, canonical accession, and title"
        )

    relevance_element = record.find(".//ProjectDescr/Relevance")
    relevance = tuple(
        name
        for name in _ORIGIN_RELEVANCE
        if relevance_element is not None
        and (relevance_element.findtext(name) or "").strip().casefold() in _AFFIRMATIVE_VALUES
    )
    return BioProjectContext(
        id=project_id,
        accession=accession,
        title=title,
        description=description,
        relevance=relevance,
    )


def _clean_project_text(element: etree._Element | None) -> str:
    if element is None:
        return ""
    raw = "".join(element.itertext())
    unescaped = html.unescape(raw)
    try:
        fragment = lxml_html.fragment_fromstring(unescaped, create_parent="div")
        text = _readable_fragment_text(fragment)
    except etree.ParserError:
        text = unescaped
    return " ".join(text.split())


def _readable_fragment_text(element: etree._Element) -> str:
    parts = [element.text or ""]
    for child in element:
        if not isinstance(child.tag, str):
            parts.append(child.tail or "")
            continue
        is_block = child.tag.rsplit("}", 1)[-1].casefold() in _BLOCK_ELEMENTS
        if is_block:
            parts.append(" ")
        parts.append(_readable_fragment_text(child))
        if is_block:
            parts.append(" ")
        parts.append(child.tail or "")
    return "".join(parts)


def _clear_record(record: etree._Element) -> None:
    record.clear()
    parent = record.getparent()
    if parent is not None:
        while record.getprevious() is not None:
            del parent[0]
