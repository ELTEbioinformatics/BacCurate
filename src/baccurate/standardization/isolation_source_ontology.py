"""Load the versioned isolation-source ontology and its external mappings."""

from __future__ import annotations

import csv
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from types import MappingProxyType
from typing import Protocol

from baccurate.adapters.llm.request import canonical_json_sha256
from baccurate.adapters.policy_yaml import load_policy_mapping

FACETS_FILENAME = "facets.yaml"
TERMS_FILENAME = "terms.tsv"
MAPPINGS_FILENAME = "mappings.sssom.tsv"
MAPPING_SET_FILENAME = "mappings.sssom.yml"

_RESOLVING_MAPPING_PREDICATES = frozenset({"skos:exactMatch", "skos:narrowMatch"})


class IsolationSourceOntologyError(ValueError):
    """A source-aware error in the isolation-source ontology artifact."""


class FacetCardinality(StrEnum):
    """The number of terms a classification can select from one facet."""

    SINGLE = "single"
    MULTIPLE = "multiple"


@dataclass(frozen=True, slots=True)
class IsolationSourceFacet:
    """Definition and classifier guidance for one isolation-source facet."""

    key: str
    cardinality: FacetCardinality
    output_column: str
    render_order: int
    standards_bindings: Mapping[str, tuple[str, ...]]
    meaning: str
    classifier_guidance: str


@dataclass(frozen=True, slots=True)
class IsolationSourceTerm:
    """One stable term in an isolation-source facet."""

    term_id: str
    label: str
    facet: str
    parent_id: str | None
    synonyms: tuple[str, ...]
    crosslink_target_ids: tuple[str, ...]
    curation_note: str
    enables_host_recovery: bool


@dataclass(frozen=True, slots=True)
class OntologyTermMapping:
    """One stated relation from a BacCurate term to an external ontology class."""

    subject_id: str
    subject_label: str
    predicate_id: str
    object_id: str
    mapping_justification: str
    object_source: str
    object_source_version: str
    comment: str
    row_number: int


@dataclass(frozen=True, slots=True)
class IsolationSourceMappingSet:
    """Versioned SSSOM metadata and mappings for the isolation-source vocabulary."""

    mapping_set_id: str
    mapping_set_version: str
    mapping_set_title: str
    mapping_set_description: str
    mapping_date: str
    license: str
    subject_source: str
    subject_source_version: str
    comment: str
    curie_map: Mapping[str, str]
    mappings: tuple[OntologyTermMapping, ...]


@dataclass(frozen=True, slots=True)
class IsolationSourceOntology:
    """The loaded facets, terms, relationships, and external mappings."""

    schema_version: int
    vocabulary_version: str
    retired_term_ids: tuple[str, ...]
    facets: Mapping[str, IsolationSourceFacet]
    terms: Mapping[str, IsolationSourceTerm]
    mapping_set: IsolationSourceMappingSet
    resolved_mapping_terms: Mapping[str, IsolationSourceTerm]

    @property
    def vocabulary_fingerprint(self) -> str:
        """Return the semantic fingerprint of the vocabulary files."""
        return ontology_semantics_fingerprint(self)

    @property
    def mapping_set_fingerprint(self) -> str:
        """Return the semantic fingerprint of the mapping-set files."""
        return _mapping_set_semantics_fingerprint(self)

    @classmethod
    def load(cls, ontology_directory: Path | str) -> IsolationSourceOntology:
        """Load the four fixed files in an ontology directory as one vocabulary."""
        directory = Path(ontology_directory)
        facets_path = directory / FACETS_FILENAME
        terms_path = directory / TERMS_FILENAME
        mappings_path = directory / MAPPINGS_FILENAME
        mapping_set_path = directory / MAPPING_SET_FILENAME
        facet_document = load_policy_mapping(facets_path)
        mapping_set_document = load_policy_mapping(mapping_set_path)

        facets = _load_facets(facet_document, facets_path)
        terms = _load_terms(terms_path)
        _validate_declared_facets(terms, facets, terms_path)
        _validate_term_references(terms, terms_path)
        _validate_parent_acyclicity(terms, terms_path)
        _validate_crosslink_cardinality(terms, facets, terms_path)
        mappings = _load_mappings(mappings_path)
        _validate_mapping_subjects(terms, mappings, mappings_path)
        mapping_set = _load_mapping_set(
            mapping_set_document,
            mappings,
            mapping_set_path,
        )
        resolved_mapping_terms = {
            mapping.object_id.upper(): terms[mapping.subject_id]
            for mapping in mappings
            if mapping.predicate_id in _RESOLVING_MAPPING_PREDICATES
        }

        return cls(
            schema_version=_required_integer(facet_document, "schema_version", facets_path),
            vocabulary_version=_required_text(facet_document, "vocabulary_version", facets_path),
            retired_term_ids=_string_sequence(
                facet_document.get("retired_term_ids"),
                facets_path,
                "retired_term_ids",
            ),
            facets=MappingProxyType(facets),
            terms=MappingProxyType(terms),
            mapping_set=mapping_set,
            resolved_mapping_terms=MappingProxyType(resolved_mapping_terms),
        )


class _LegacyOntologyGraph(Protocol):
    """The graph shape retained until the faceted classifier swap."""

    node_metadata: Mapping[str, Mapping[str, object]]
    children_map: Mapping[str, list[str]]
    crosslink_map: Mapping[str, list[str]]


def ontology_semantics_fingerprint(
    ontology: IsolationSourceOntology | _LegacyOntologyGraph | None,
) -> str:
    """Fingerprint parsed vocabulary content independently of file formatting."""
    if ontology is None:
        return _legacy_ontology_semantics_fingerprint({}, {}, {})
    if not isinstance(ontology, IsolationSourceOntology):
        return _legacy_ontology_semantics_fingerprint(
            ontology.node_metadata,
            ontology.children_map,
            ontology.crosslink_map,
        )
    facets = {
        key: {
            "cardinality": facet.cardinality.value,
            "classifier_guidance": facet.classifier_guidance,
            "meaning": facet.meaning,
            "output_column": facet.output_column,
            "render_order": facet.render_order,
            "standards_bindings": {
                binding: sorted(values)
                for binding, values in sorted(facet.standards_bindings.items())
            },
        }
        for key, facet in sorted(ontology.facets.items())
    }
    terms = {
        term_id: {
            "crosslink_target_ids": sorted(term.crosslink_target_ids),
            "curation_note": term.curation_note,
            "enables_host_recovery": term.enables_host_recovery,
            "facet": term.facet,
            "label": term.label,
            "parent_id": term.parent_id,
            "synonyms": sorted(term.synonyms),
        }
        for term_id, term in sorted(ontology.terms.items())
    }
    return canonical_json_sha256(
        {
            "facets": facets,
            "retired_term_ids": sorted(ontology.retired_term_ids),
            "schema_version": ontology.schema_version,
            "terms": terms,
        }
    )


def _legacy_ontology_semantics_fingerprint(
    node_metadata: Mapping[str, Mapping[str, object]],
    children_map: Mapping[str, list[str]],
    crosslink_map: Mapping[str, list[str]],
) -> str:
    """Preserve current run identity until the directory policy becomes active."""
    nodes = {
        term_path: {
            **metadata,
            "synonyms": sorted(metadata.get("synonyms", [])),
        }
        for term_path, metadata in sorted(node_metadata.items())
    }
    hierarchy = {parent: sorted(children) for parent, children in sorted(children_map.items())}
    crosslinks = {
        source_term_path: sorted(targets)
        for source_term_path, targets in sorted(crosslink_map.items())
    }
    return canonical_json_sha256({"nodes": nodes, "hierarchy": hierarchy, "crosslinks": crosslinks})


def _mapping_set_semantics_fingerprint(ontology: IsolationSourceOntology) -> str:
    """Fingerprint parsed mapping-set content independently of file formatting."""
    mapping_set = ontology.mapping_set
    mappings = sorted(
        (
            {
                "comment": mapping.comment,
                "mapping_justification": mapping.mapping_justification,
                "object_id": mapping.object_id,
                "object_source": mapping.object_source,
                "object_source_version": mapping.object_source_version,
                "predicate_id": mapping.predicate_id,
                "subject_id": mapping.subject_id,
                "subject_label": mapping.subject_label,
            }
            for mapping in mapping_set.mappings
        ),
        key=lambda mapping: tuple(mapping.values()),
    )
    return canonical_json_sha256(
        {
            "comment": mapping_set.comment,
            "curie_map": dict(sorted(mapping_set.curie_map.items())),
            "license": mapping_set.license,
            "mapping_date": mapping_set.mapping_date,
            "mapping_set_description": mapping_set.mapping_set_description,
            "mapping_set_id": mapping_set.mapping_set_id,
            "mapping_set_title": mapping_set.mapping_set_title,
            "mappings": mappings,
            "subject_source": mapping_set.subject_source,
            "subject_source_version": mapping_set.subject_source_version,
        }
    )


def _ontology_error(path: Path, location: str, message: str) -> IsolationSourceOntologyError:
    return IsolationSourceOntologyError(f"{path}: {location}: {message}")


def _required_text(source: Mapping[object, object], key: str, path: Path) -> str:
    value = source.get(key)
    if not isinstance(value, str) or not value.strip():
        raise _ontology_error(path, key, "must be a non-empty string")
    return value.strip()


def _required_integer(source: Mapping[object, object], key: str, path: Path) -> int:
    value = source.get(key)
    if type(value) is not int:
        raise _ontology_error(path, key, "must be an integer")
    return value


def _required_mapping(
    source: Mapping[object, object], key: str, path: Path
) -> Mapping[object, object]:
    value = source.get(key)
    if not isinstance(value, Mapping):
        raise _ontology_error(path, key, "must be a mapping")
    return value


def _string_sequence(value: object, path: Path, location: str) -> tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, str):
        raise _ontology_error(path, location, "must be a sequence of strings")
    if not all(isinstance(entry, str) and entry.strip() for entry in value):
        raise _ontology_error(path, location, "must contain only non-empty strings")
    return tuple(entry.strip() for entry in value)


def _load_facets(document: Mapping[object, object], path: Path) -> dict[str, IsolationSourceFacet]:
    raw_facets = _required_mapping(document, "facets", path)
    facets: dict[str, IsolationSourceFacet] = {}
    for raw_key, raw_definition in raw_facets.items():
        if not isinstance(raw_key, str) or not raw_key.strip():
            raise _ontology_error(path, "facets", "facet keys must be non-empty strings")
        key = raw_key.strip()
        if not isinstance(raw_definition, Mapping):
            raise _ontology_error(path, f"facets.{key}", "must be a mapping")
        try:
            cardinality = FacetCardinality(_required_text(raw_definition, "cardinality", path))
        except ValueError as error:
            raise _ontology_error(
                path,
                f"facets.{key}.cardinality",
                "must be 'single' or 'multiple'",
            ) from error

        raw_bindings = _required_mapping(raw_definition, "standards_bindings", path)
        standards_bindings = {
            str(binding_key): _string_sequence(
                binding_values,
                path,
                f"facets.{key}.standards_bindings.{binding_key}",
            )
            for binding_key, binding_values in raw_bindings.items()
        }
        facets[key] = IsolationSourceFacet(
            key=key,
            cardinality=cardinality,
            output_column=_required_text(raw_definition, "output_column", path),
            render_order=_required_integer(raw_definition, "render_order", path),
            standards_bindings=MappingProxyType(standards_bindings),
            meaning=_required_text(raw_definition, "meaning", path),
            classifier_guidance=_required_text(raw_definition, "classifier_guidance", path),
        )
    return facets


def _load_tsv_rows(
    path: Path, expected_columns: tuple[str, ...]
) -> list[tuple[int, dict[str, str]]]:
    try:
        with path.open("r", encoding="utf-8", newline="") as source:
            reader = csv.DictReader(source, delimiter="\t")
            if tuple(reader.fieldnames or ()) != expected_columns:
                raise _ontology_error(
                    path,
                    "header",
                    f"must contain these columns in order: {', '.join(expected_columns)}",
                )
            return [
                (row_number, {key: (value or "").strip() for key, value in row.items()})
                for row_number, row in enumerate(reader, start=2)
            ]
    except IsolationSourceOntologyError:
        raise
    except (OSError, UnicodeError, csv.Error) as error:
        raise _ontology_error(path, "file", str(error)) from error


def _semicolon_values(value: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in value.split(";") if part.strip())


def _required_tsv_value(
    row: Mapping[str, str],
    key: str,
    path: Path,
    row_number: int,
    domain_name: str,
) -> str:
    value = row[key]
    if not value:
        raise _ontology_error(
            path,
            f"row {row_number} {key}",
            f"must be a non-empty {domain_name}",
        )
    return value


def _load_terms(path: Path) -> dict[str, IsolationSourceTerm]:
    columns = (
        "term_id",
        "label",
        "facet",
        "parent_id",
        "synonyms",
        "crosslink_targets",
        "curation_note",
        "enables_host_recovery",
    )
    terms: dict[str, IsolationSourceTerm] = {}
    term_rows: dict[str, int] = {}
    labels: dict[str, tuple[str, str, int]] = {}
    for row_number, row in _load_tsv_rows(path, columns):
        term_id = _required_tsv_value(row, "term_id", path, row_number, "term identifier")
        if term_id in terms:
            raise _ontology_error(
                path,
                f"row {row_number} term_id",
                f"duplicate term identifier {term_id!r}; first declared at row "
                f"{term_rows[term_id]}",
            )
        label = _required_tsv_value(row, "label", path, row_number, "ontology display term")
        facet = _required_tsv_value(row, "facet", path, row_number, "facet key")
        normalized_label = label.casefold()
        if normalized_label in labels:
            previous_term_id, previous_facet, previous_row = labels[normalized_label]
            facet_message = (
                f"across facets {previous_facet!r} and {facet!r}"
                if previous_facet != facet
                else f"within facet {facet!r}"
            )
            raise _ontology_error(
                path,
                f"row {row_number} label",
                f"duplicate label {label!r} {facet_message}; first declared by "
                f"{previous_term_id!r} at row {previous_row}",
            )
        recovery_text = row["enables_host_recovery"].lower()
        if recovery_text not in {"true", "false"}:
            raise _ontology_error(
                path,
                f"row {row_number} enables_host_recovery",
                "must be true or false",
            )
        term = IsolationSourceTerm(
            term_id=term_id,
            label=label,
            facet=facet,
            parent_id=row["parent_id"] or None,
            synonyms=_semicolon_values(row["synonyms"]),
            crosslink_target_ids=_semicolon_values(row["crosslink_targets"]),
            curation_note=row["curation_note"],
            enables_host_recovery=recovery_text == "true",
        )
        terms[term.term_id] = term
        term_rows[term.term_id] = row_number
        labels[normalized_label] = (term.term_id, term.facet, row_number)
    return terms


def _load_mappings(path: Path) -> tuple[OntologyTermMapping, ...]:
    columns = (
        "subject_id",
        "subject_label",
        "predicate_id",
        "object_id",
        "mapping_justification",
        "object_source",
        "object_source_version",
        "comment",
    )
    mappings: list[OntologyTermMapping] = []
    for row_number, row in _load_tsv_rows(path, columns):
        _required_tsv_value(row, "subject_id", path, row_number, "term identifier")
        _required_tsv_value(row, "predicate_id", path, row_number, "mapping predicate")
        _required_tsv_value(
            row,
            "object_id",
            path,
            row_number,
            "external ontology identifier",
        )
        mappings.append(OntologyTermMapping(row_number=row_number, **row))
    return tuple(mappings)


def _validate_declared_facets(
    terms: Mapping[str, IsolationSourceTerm],
    facets: Mapping[str, IsolationSourceFacet],
    terms_path: Path,
) -> None:
    for term in terms.values():
        if term.facet not in facets:
            raise _ontology_error(
                terms_path,
                f"term {term.term_id!r} facet",
                f"undeclared facet {term.facet!r}",
            )


def _validate_term_references(terms: Mapping[str, IsolationSourceTerm], terms_path: Path) -> None:
    for term in terms.values():
        if term.parent_id is not None:
            if term.parent_id not in terms:
                raise _ontology_error(
                    terms_path,
                    f"term {term.term_id!r} parent_id",
                    f"unresolvable parent {term.parent_id!r}",
                )
            parent = terms[term.parent_id]
            if parent.facet != term.facet:
                raise _ontology_error(
                    terms_path,
                    f"term {term.term_id!r} parent_id",
                    f"parent {parent.term_id!r} belongs to facet {parent.facet!r}; "
                    f"parents must belong to {term.facet!r}",
                )
        for target_id in term.crosslink_target_ids:
            if target_id not in terms:
                raise _ontology_error(
                    terms_path,
                    f"term {term.term_id!r} crosslink_targets",
                    f"unresolvable crosslink target {target_id!r}",
                )


def _validate_mapping_subjects(
    terms: Mapping[str, IsolationSourceTerm],
    mappings: tuple[OntologyTermMapping, ...],
    mappings_path: Path,
) -> None:
    for mapping in mappings:
        if mapping.subject_id not in terms:
            raise _ontology_error(
                mappings_path,
                f"row {mapping.row_number} subject_id",
                f"mapping subject {mapping.subject_id!r} is not an ontology term",
            )


def _validate_parent_acyclicity(terms: Mapping[str, IsolationSourceTerm], terms_path: Path) -> None:
    completed: set[str] = set()
    for start_id in terms:
        route: list[str] = []
        route_positions: dict[str, int] = {}
        current_id: str | None = start_id
        while current_id is not None and current_id not in completed:
            if current_id in route_positions:
                cycle = [*route[route_positions[current_id] :], current_id]
                raise _ontology_error(
                    terms_path,
                    f"term {start_id!r} parent_id",
                    f"parent cycle: {' -> '.join(cycle)}",
                )
            route_positions[current_id] = len(route)
            route.append(current_id)
            current_id = terms[current_id].parent_id
        completed.update(route)


def _validate_crosslink_cardinality(
    terms: Mapping[str, IsolationSourceTerm],
    facets: Mapping[str, IsolationSourceFacet],
    terms_path: Path,
) -> None:
    for term in terms.values():
        assigned_ids_by_facet: dict[str, set[str]] = {term.facet: {term.term_id}}
        for target_id in term.crosslink_target_ids:
            target = terms[target_id]
            assigned_ids_by_facet.setdefault(target.facet, set()).add(target_id)
        for facet_key, assigned_ids in assigned_ids_by_facet.items():
            if facets[facet_key].cardinality is FacetCardinality.SINGLE and len(assigned_ids) > 1:
                identifiers = ", ".join(sorted(assigned_ids))
                raise _ontology_error(
                    terms_path,
                    f"term {term.term_id!r} crosslink_targets",
                    f"assigns multiple values to single-valued facet {facet_key!r}: {identifiers}",
                )


def _load_mapping_set(
    document: Mapping[object, object],
    mappings: tuple[OntologyTermMapping, ...],
    path: Path,
) -> IsolationSourceMappingSet:
    raw_curie_map = _required_mapping(document, "curie_map", path)
    curie_map = {
        str(prefix): str(expansion)
        for prefix, expansion in raw_curie_map.items()
        if str(prefix) and str(expansion)
    }
    return IsolationSourceMappingSet(
        mapping_set_id=_required_text(document, "mapping_set_id", path),
        mapping_set_version=_required_text(document, "mapping_set_version", path),
        mapping_set_title=_required_text(document, "mapping_set_title", path),
        mapping_set_description=_required_text(document, "mapping_set_description", path),
        mapping_date=_required_text(document, "mapping_date", path),
        license=_required_text(document, "license", path),
        subject_source=_required_text(document, "subject_source", path),
        subject_source_version=_required_text(document, "subject_source_version", path),
        comment=_required_text(document, "comment", path),
        curie_map=MappingProxyType(curie_map),
        mappings=mappings,
    )
