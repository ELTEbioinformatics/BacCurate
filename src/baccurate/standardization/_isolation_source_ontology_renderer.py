"""Render the faceted isolation-source vocabulary for the LLM prompt."""

from __future__ import annotations

from collections import defaultdict

from baccurate.standardization.isolation_source_ontology import (
    IsolationSourceFacet,
    IsolationSourceOntology,
    IsolationSourceTerm,
)


def ordered_facets(ontology: IsolationSourceOntology) -> tuple[IsolationSourceFacet, ...]:
    """Return facet definitions in their declared render order."""
    return tuple(sorted(ontology.facets.values(), key=lambda facet: facet.render_order))


def _render_facet(ontology: IsolationSourceOntology, facet_key: str) -> list[str]:
    children: dict[str | None, list[IsolationSourceTerm]] = defaultdict(list)
    for term in ontology.terms.values():
        if term.facet == facet_key:
            children[term.parent_id].append(term)
    for siblings in children.values():
        siblings.sort(key=lambda term: term.term_id)

    lines = [f"## {facet_key}"]
    notes: list[str] = []

    def walk(parent_id: str | None, depth: int) -> None:
        for term in children.get(parent_id, []):
            lines.append(f"{'  ' * depth}- {term.label}")
            if term.curation_note:
                notes.append(f'- "{term.label}": {term.curation_note.rstrip(".")}.')
            walk(term.term_id, depth + 1)

    walk(None, 0)
    if notes:
        lines.extend(("", "### Term notes", *notes))
    return lines


def render_ontology(ontology: IsolationSourceOntology) -> str:
    """Render one ordered section per classifier response field."""
    sections = ["\n".join(_render_facet(ontology, facet.key)) for facet in ordered_facets(ontology)]
    return "\n\n".join(sections)
