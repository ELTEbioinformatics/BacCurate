"""
Render the ontology graph as an indented Markdown block for the LLM.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from baccurate.standardizers.isolation import OntologyManager

TOP_LEVEL_ORDER = (
    "environmental",
    "host-associated",
    "unspecified",
    "ambiguous",
)


def _format_node_line(*, indent: int, display_term: str) -> str:
    return "  " * indent + "- " + display_term


def _walk_tree(
    ont: OntologyManager,
    path: str,
    indent: int,
    out: list[str],
) -> None:
    meta = ont.node_metadata.get(path, {})
    # OntologyManager already falls back to the path tail when display_term
    # is empty, so this is non-empty for every node.
    display_term = (meta.get("display_term") or path.split(":")[-1]).strip()
    out.append(_format_node_line(indent=indent, display_term=display_term))
    for child in ont.children_map.get(path, []):
        _walk_tree(ont, child, indent + 1, out)


def _walk_notes(
    ont: OntologyManager,
    path: str,
    out: list[str],
) -> None:
    meta = ont.node_metadata.get(path, {})
    comment = (meta.get("comment") or "").strip()
    if comment:
        display_term = (meta.get("display_term") or path.split(":")[-1]).strip()
        out.append(f'- "{display_term}": {comment.rstrip(".")}.')
    for child in ont.children_map.get(path, []):
        _walk_notes(ont, child, out)


def _ordered_roots(ont: OntologyManager) -> list[str]:
    """Top-level branches in TOP_LEVEL_ORDER, with any extras appended."""
    roots = list(ont.children_map.get("", []))
    ordered: list[str] = []
    for name in TOP_LEVEL_ORDER:
        if name in roots:
            ordered.append(name)
    for p in roots:
        if p not in ordered:
            ordered.append(p)
    return ordered


def render_ontology(ont: OntologyManager) -> str:
    """Return the ontology as two sections: a tree and a notes list."""
    roots = _ordered_roots(ont)

    tree_lines: list[str] = []
    for root in roots:
        _walk_tree(ont, root, indent=0, out=tree_lines)

    note_lines: list[str] = []
    for root in roots:
        _walk_notes(ont, root, out=note_lines)

    parts = ["# Tree", "\n".join(tree_lines)]
    if note_lines:
        parts += ["", "# Node notes", "\n".join(note_lines)]
    return "\n".join(parts)


def valid_term_paths(ont: OntologyManager) -> list[str]:
    """All term paths the LLM is allowed to return, in render order."""
    seen: list[str] = []

    def collect(path: str) -> None:
        seen.append(path)
        for child in ont.children_map.get(path, []):
            collect(child)

    for root in _ordered_roots(ont):
        collect(root)
    return seen


def valid_display_terms(ont: OntologyManager) -> list[str]:
    """Every display name the LLM may emit, in the same order as the rendered tree."""
    paths = valid_term_paths(ont)
    out: list[str] = []
    for path in paths:
        meta = ont.node_metadata.get(path, {})
        display = (meta.get("display_term") or path.split(":")[-1]).strip()
        if display:
            out.append(display)
    return out
