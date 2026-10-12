"""Render the faceted isolation-source vocabulary for the LLM prompt."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping

from baccurate.standardization.isolation_source_ontology import (
    IsolationSourceOntology,
    IsolationSourceTerm,
)


def _render_facet(ontology: IsolationSourceOntology, facet_key: str, guidance: str) -> list[str]:
    children: dict[str | None, list[IsolationSourceTerm]] = defaultdict(list)
    for term in ontology.terms.values():
        if term.facet == facet_key:
            children[term.parent_id].append(term)
    for siblings in children.values():
        siblings.sort(key=lambda term: term.term_id)

    lines = [f"## {facet_key}", guidance]
    notes: list[str] = []

    def walk(parent_id: str | None, depth: int) -> None:
        for term in children.get(parent_id, []):
            lines.append(f"{'  ' * depth}- {term.label}")
            if term.note:
                notes.append(f'- "{term.label}": {term.note}')
            walk(term.term_id, depth + 1)

    walk(None, 0)
    if notes:
        lines.extend(("", "### Term notes", *notes))
    return lines


def render_ontology(ontology: IsolationSourceOntology, facet_guidance: Mapping[str, str]) -> str:
    """Render one section per facet, in the order of ``facet_guidance``, with the guidance
    under its heading."""
    return "\n\n".join(
        "\n".join(_render_facet(ontology, key, guidance))
        for key, guidance in facet_guidance.items()
    )
