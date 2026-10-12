"""Load the versioned isolation-source ontology and its external mappings."""

from __future__ import annotations

import csv
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from enum import StrEnum
from pathlib import Path
from types import MappingProxyType

from baccurate.adapters.llm.request import canonical_json_sha256
from baccurate.adapters.policy_yaml import load_policy_mapping
from baccurate.standardization._attribute_value_text import normalize_keyword

FACETS_FILENAME = "facets.yaml"
TERMS_FILENAME = "terms.tsv"
SYNONYMS_FILENAME = "synonyms.tsv"
OBSOLETE_TERMS_FILENAME = "obsolete_terms.tsv"
RELATIONS_FILENAME = "relations.tsv"
MAPPINGS_FILENAME = "mappings.sssom.tsv"
MAPPING_SET_FILENAME = "mappings.sssom.yml"

SOURCE_TYPE_FACET = "source_type"
HOST_DISEASE_FACET = "host_disease"

HOST_ASSOCIATED_TERM_ID = "BACC:0000001"
ANIMAL_HOST_TERM_ID = "BACC:0000002"
HUMAN_HOST_TERM_ID = "BACC:0000122"
PLANT_HOST_TERM_ID = "BACC:0000003"
ENVIRONMENTAL_TERM_ID = "BACC:0000004"
FOOD_OR_FEED_TERM_ID = "BACC:0000007"
LABORATORY_TERM_ID = "BACC:0000008"
MEDICAL_DEVICE_TERM_ID = "BACC:0000091"
SOURCE_TYPE_BY_IMPLYING_FACET = {
    "body_product": HOST_ASSOCIATED_TERM_ID,
    "body_site": HOST_ASSOCIATED_TERM_ID,
    "lesion": HOST_ASSOCIATED_TERM_ID,
    "environmental_material": ENVIRONMENTAL_TERM_ID,
    "sampled_object": ENVIRONMENTAL_TERM_ID,
    "food_type": FOOD_OR_FEED_TERM_ID,
    HOST_DISEASE_FACET: HOST_ASSOCIATED_TERM_ID,
}

_RESOLVING_MAPPING_PREDICATES = frozenset({"skos:exactMatch", "skos:narrowMatch"})
_FACET_FIELDS = frozenset({"meaning", "standards_bindings"})


class IsolationSourceOntologyError(ValueError):
    """A source-aware error in the isolation-source ontology artifact."""


class ParentRelation(StrEnum):
    """How a term relates to its parent."""

    KIND_OF = "kind_of"
    PART_OF = "part_of"


class SynonymType(StrEnum):
    """The oboInOwl scope of a synonym relative to its term."""

    EXACT = "exact"
    NARROW = "narrow"
    BROAD = "broad"
    RELATED = "related"


class RelationType(StrEnum):
    """A directed, non-hierarchical link from a subject term to an object term."""

    LOCATED_IN = "located_in"
    PRODUCED_BY = "produced_by"
    COLLECTED_FROM = "collected_from"
    PART_OF = "part_of"
    HAS_PART = "has_part"
    IMPLIES_SOURCE_TYPE = "implies_source_type"


@dataclass(frozen=True, slots=True)
class IsolationSourceFacet:
    """The meaning and standards bindings of one isolation-source facet."""

    key: str
    standards_bindings: Mapping[str, tuple[str, ...]]
    meaning: str


@dataclass(frozen=True, slots=True)
class Synonym:
    """One name of a term and its scope."""

    text: str
    synonym_type: SynonymType


@dataclass(frozen=True, slots=True)
class IsolationSourceTerm:
    """One stable term in an isolation-source facet."""

    term_id: str
    label: str
    facet: str
    parent_id: str | None
    parent_relation: ParentRelation | None
    definition: str
    scope_note: str
    synonyms: tuple[Synonym, ...] = ()

    @property
    def note(self) -> str:
        """Return the definition and the scope note as one text, in the form the prompt shows."""
        return " ".join(text for text in (self.definition, self.scope_note) if text)


@dataclass(frozen=True, slots=True)
class ObsoleteTerm:
    """One retired term and the active terms, if any, that together replace it."""

    term_id: str
    label: str
    replaced_by: tuple[str, ...]
    note: str


@dataclass(frozen=True, slots=True)
class TermRelation:
    """One typed relation from a subject term to an object term."""

    subject_id: str
    relation: RelationType
    object_id: str


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
    """The loaded facets, terms, relations, and external mappings.

    ``relation_object_ids`` holds, for every term, the relation objects that term derivation adds
    when the term is explicit: the objects of its own relations and the objects of the
    ``implies_source_type`` relations of its ancestors. ``match_keys`` maps the
    ``normalize_keyword`` form of every label and synonym to its term identifier.
    """

    schema_version: int
    vocabulary_version: str
    facets: Mapping[str, IsolationSourceFacet]
    terms: Mapping[str, IsolationSourceTerm]
    obsolete_terms: Mapping[str, ObsoleteTerm]
    relations: tuple[TermRelation, ...]
    relation_object_ids: Mapping[str, tuple[str, ...]]
    mapping_set: IsolationSourceMappingSet
    resolved_mapping_terms: Mapping[str, IsolationSourceTerm]
    exact_match_ids: Mapping[str, str]
    match_keys: Mapping[str, str]

    @property
    def vocabulary_fingerprint(self) -> str:
        """Return the semantic fingerprint of the vocabulary files."""
        return _ontology_semantics_fingerprint(self)

    @property
    def mapping_set_fingerprint(self) -> str:
        """Return the semantic fingerprint of the mapping-set files."""
        return _mapping_set_semantics_fingerprint(self)

    def implied_source_type_ids(self, term_ids: Iterable[str]) -> frozenset[str]:
        """Return the source types that the facets of the terms imply.

        A medical device is placed in or on a patient, so it and its descendants imply no source
        type, although their facet does.
        """
        facets = {
            self.terms[term_id].facet
            for term_id in term_ids
            if MEDICAL_DEVICE_TERM_ID not in (term_id, *_ancestor_ids(self.terms, term_id))
        }
        return frozenset(
            source_type_id
            for facet, source_type_id in SOURCE_TYPE_BY_IMPLYING_FACET.items()
            if facet in facets
        )

    def nearest_common_ancestors(self, term_ids: Iterable[str]) -> frozenset[str]:
        """Replace the terms under each root with their nearest common ancestor.

        A term that shares no root with another term stays as it is.
        """
        lineages_by_root: dict[str, list[list[str]]] = {}
        for term_id in term_ids:
            lineage = [term_id, *_ancestor_ids(self.terms, term_id)]
            lineages_by_root.setdefault(lineage[-1], []).append(lineage)
        return frozenset(
            next(
                term_id
                for term_id in lineages[0]
                if all(term_id in lineage for lineage in lineages[1:])
            )
            for lineages in lineages_by_root.values()
        )

    def derive(self, term_ids: Iterable[str]) -> frozenset[str]:
        """Return the terms that production publishes for a set of explicit terms."""
        derived = set(term_ids)
        derived.update(
            object_id
            for term_id in tuple(derived)
            for object_id in self.relation_object_ids[term_id]
        )
        derived |= self.implied_source_type_ids(derived)
        derived.update(
            ancestor_id
            for term_id in tuple(derived)
            for ancestor_id in _ancestor_ids(self.terms, term_id)
        )
        return frozenset(derived)

    @classmethod
    def load(cls, ontology_directory: Path | str) -> IsolationSourceOntology:
        """Load the seven fixed files in an ontology directory as one vocabulary."""
        directory = Path(ontology_directory)
        facets_path = directory / FACETS_FILENAME
        terms_path = directory / TERMS_FILENAME
        synonyms_path = directory / SYNONYMS_FILENAME
        obsolete_terms_path = directory / OBSOLETE_TERMS_FILENAME
        relations_path = directory / RELATIONS_FILENAME
        mappings_path = directory / MAPPINGS_FILENAME
        mapping_set_path = directory / MAPPING_SET_FILENAME
        facet_document = load_policy_mapping(facets_path)
        mapping_set_document = load_policy_mapping(mapping_set_path)

        facets = _load_facets(facet_document, facets_path)
        terms = _load_synonyms(synonyms_path, _load_terms(terms_path))
        obsolete_terms = _load_obsolete_terms(obsolete_terms_path, terms)
        _validate_declared_facets(terms, facets, terms_path)
        _validate_term_references(terms, terms_path)
        _validate_parent_acyclicity(terms, terms_path)
        relations = _load_relations(relations_path, terms)
        relation_object_ids = _resolve_relation_objects(terms, relations)
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
        exact_match_ids = {
            mapping.subject_id: mapping.object_id
            for mapping in mappings
            if mapping.predicate_id == "skos:exactMatch"
        }
        _validate_definitions(terms, exact_match_ids, terms_path)
        vocabulary_version = _required_text(facet_document, "vocabulary_version", facets_path)
        _validate_versions(vocabulary_version, mapping_set, mapping_set_path)

        return cls(
            schema_version=_required_integer(facet_document, "schema_version", facets_path),
            vocabulary_version=vocabulary_version,
            facets=MappingProxyType(facets),
            terms=MappingProxyType(terms),
            obsolete_terms=MappingProxyType(obsolete_terms),
            relations=relations,
            relation_object_ids=MappingProxyType(relation_object_ids),
            mapping_set=mapping_set,
            resolved_mapping_terms=MappingProxyType(resolved_mapping_terms),
            exact_match_ids=MappingProxyType(exact_match_ids),
            match_keys=MappingProxyType(_match_keys(terms, synonyms_path)),
        )


def _ontology_semantics_fingerprint(ontology: IsolationSourceOntology) -> str:
    """Fingerprint parsed vocabulary content independently of file formatting."""
    facets = {
        key: {
            "meaning": facet.meaning,
            "standards_bindings": {
                binding: sorted(values)
                for binding, values in sorted(facet.standards_bindings.items())
            },
        }
        for key, facet in sorted(ontology.facets.items())
    }
    terms = {
        term_id: {
            "definition": term.definition,
            "facet": term.facet,
            "label": term.label,
            "parent_id": term.parent_id,
            "parent_relation": term.parent_relation,
            "scope_note": term.scope_note,
            "synonyms": sorted([synonym.text, synonym.synonym_type] for synonym in term.synonyms),
        }
        for term_id, term in sorted(ontology.terms.items())
    }
    return canonical_json_sha256(
        {
            "facets": facets,
            "relations": sorted(
                [relation.subject_id, relation.relation.value, relation.object_id]
                for relation in ontology.relations
            ),
            "obsolete_terms": {
                term_id: {
                    "label": term.label,
                    "note": term.note,
                    "replaced_by": sorted(term.replaced_by),
                }
                for term_id, term in sorted(ontology.obsolete_terms.items())
            },
            "schema_version": ontology.schema_version,
            "terms": terms,
        }
    )


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
        unknown_fields = sorted(str(field) for field in set(raw_definition) - _FACET_FIELDS)
        if unknown_fields:
            raise _ontology_error(path, f"facets.{key}.{unknown_fields[0]}", "unknown facet field")

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
            standards_bindings=MappingProxyType(standards_bindings),
            meaning=_required_text(raw_definition, "meaning", path),
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
        "parent_relation",
        "definition",
        "scope_note",
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
        term = IsolationSourceTerm(
            term_id=term_id,
            label=label,
            facet=facet,
            parent_id=row["parent_id"] or None,
            parent_relation=_parent_relation(row, path, row_number),
            definition=row["definition"],
            scope_note=row["scope_note"],
        )
        terms[term.term_id] = term
        term_rows[term.term_id] = row_number
        labels[normalized_label] = (term.term_id, term.facet, row_number)
    return terms


def _load_synonyms(
    path: Path, terms: Mapping[str, IsolationSourceTerm]
) -> dict[str, IsolationSourceTerm]:
    synonyms: dict[str, list[Synonym]] = {term_id: [] for term_id in terms}
    synonym_rows: dict[tuple[str, str], int] = {}
    for row_number, row in _load_tsv_rows(path, ("term_id", "synonym", "synonym_type")):
        term_id = _required_tsv_value(row, "term_id", path, row_number, "term identifier")
        if term_id not in terms:
            raise _ontology_error(path, f"row {row_number} term_id", f"unknown term {term_id!r}")
        text = _required_tsv_value(row, "synonym", path, row_number, "synonym")
        try:
            synonym_type = SynonymType(row["synonym_type"])
        except ValueError as error:
            raise _ontology_error(
                path,
                f"row {row_number} synonym_type",
                f"unknown synonym type {row['synonym_type']!r}",
            ) from error
        if (term_id, text) in synonym_rows:
            raise _ontology_error(
                path, f"row {row_number}", f"duplicate of row {synonym_rows[term_id, text]}"
            )
        synonym_rows[term_id, text] = row_number
        synonyms[term_id].append(Synonym(text, synonym_type))
    return {
        term_id: replace(term, synonyms=tuple(synonyms[term_id])) for term_id, term in terms.items()
    }


def _validate_definitions(
    terms: Mapping[str, IsolationSourceTerm], exact_match_ids: Mapping[str, str], terms_path: Path
) -> None:
    for term in terms.values():
        if not term.definition and term.term_id not in exact_match_ids:
            raise _ontology_error(
                terms_path,
                f"term {term.term_id!r} definition",
                "required for a term without a skos:exactMatch mapping",
            )


def _validate_versions(
    vocabulary_version: str, mapping_set: IsolationSourceMappingSet, mapping_set_path: Path
) -> None:
    for key in ("mapping_set_version", "subject_source_version"):
        version = getattr(mapping_set, key)
        if version != vocabulary_version:
            raise _ontology_error(
                mapping_set_path,
                key,
                f"{version!r} differs from the vocabulary_version {vocabulary_version!r}",
            )


def _match_keys(terms: Mapping[str, IsolationSourceTerm], synonyms_path: Path) -> dict[str, str]:
    match_keys: dict[str, str] = {}
    for term in terms.values():
        for name in (term.label, *(synonym.text for synonym in term.synonyms)):
            key = normalize_keyword(name)
            if key in match_keys:
                raise _ontology_error(
                    synonyms_path,
                    f"term {term.term_id!r} name {name!r}",
                    f"match key {key!r} already names {match_keys[key]!r}",
                )
            match_keys[key] = term.term_id
    return match_keys


def _load_obsolete_terms(
    path: Path, terms: Mapping[str, IsolationSourceTerm]
) -> dict[str, ObsoleteTerm]:
    obsolete_terms: dict[str, ObsoleteTerm] = {}
    for row_number, row in _load_tsv_rows(path, ("term_id", "label", "replaced_by", "note")):
        term_id = _required_tsv_value(row, "term_id", path, row_number, "term identifier")
        location = f"row {row_number} term_id"
        if term_id in terms:
            raise _ontology_error(path, location, f"{term_id!r} is an active term")
        if term_id in obsolete_terms:
            raise _ontology_error(path, location, f"duplicate term identifier {term_id!r}")
        replaced_by = _semicolon_values(row["replaced_by"])
        for replacement_id in replaced_by:
            if replacement_id not in terms:
                raise _ontology_error(
                    path,
                    f"row {row_number} replaced_by",
                    f"replacement {replacement_id!r} is not an active term",
                )
        obsolete_terms[term_id] = ObsoleteTerm(term_id, row["label"], replaced_by, row["note"])
    return obsolete_terms


def _parent_relation(row: Mapping[str, str], path: Path, row_number: int) -> ParentRelation | None:
    location = f"row {row_number} parent_relation"
    if not row["parent_id"]:
        if row["parent_relation"]:
            raise _ontology_error(path, location, "must be empty when parent_id is empty")
        return None
    try:
        return ParentRelation(row["parent_relation"])
    except ValueError as error:
        raise _ontology_error(path, location, "must be 'kind_of' or 'part_of'") from error


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


def _ancestor_ids(terms: Mapping[str, IsolationSourceTerm], term_id: str) -> list[str]:
    ancestor_ids = []
    parent_id = terms[term_id].parent_id
    while parent_id is not None:
        ancestor_ids.append(parent_id)
        parent_id = terms[parent_id].parent_id
    return ancestor_ids


def _load_relations(
    path: Path, terms: Mapping[str, IsolationSourceTerm]
) -> tuple[TermRelation, ...]:
    relation_rows: dict[TermRelation, int] = {}
    for row_number, row in _load_tsv_rows(path, ("subject_id", "relation", "object_id")):
        for key in ("subject_id", "object_id"):
            term_id = _required_tsv_value(row, key, path, row_number, "term identifier")
            if term_id not in terms:
                raise _ontology_error(path, f"row {row_number} {key}", f"unknown term {term_id!r}")
        try:
            relation_type = RelationType(row["relation"])
        except ValueError as error:
            raise _ontology_error(
                path,
                f"row {row_number} relation",
                f"unknown relation type {row['relation']!r}",
            ) from error
        relation = TermRelation(row["subject_id"], relation_type, row["object_id"])
        if relation in relation_rows:
            raise _ontology_error(
                path, f"row {row_number}", f"duplicate of row {relation_rows[relation]}"
            )
        _validate_relation_object(relation, terms, path, row_number)
        relation_rows[relation] = row_number
    return tuple(relation_rows)


def _validate_relation_object(
    relation: TermRelation,
    terms: Mapping[str, IsolationSourceTerm],
    path: Path,
    row_number: int,
) -> None:
    location = f"row {row_number} object_id"
    implies_source_type = relation.relation is RelationType.IMPLIES_SOURCE_TYPE
    if implies_source_type != (terms[relation.object_id].facet == SOURCE_TYPE_FACET):
        requirement = "must" if implies_source_type else "must not"
        raise _ontology_error(
            path,
            location,
            f"object {relation.object_id!r} of {relation.relation.value!r} {requirement} be a "
            f"{SOURCE_TYPE_FACET!r} term",
        )
    if relation.object_id in {
        relation.subject_id,
        *_ancestor_ids(terms, relation.subject_id),
    } or relation.subject_id in _ancestor_ids(terms, relation.object_id):
        raise _ontology_error(
            path,
            location,
            f"object {relation.object_id!r} is the subject {relation.subject_id!r} or one of "
            "its ancestors or descendants",
        )


def _resolve_relation_objects(
    terms: Mapping[str, IsolationSourceTerm], relations: tuple[TermRelation, ...]
) -> dict[str, tuple[str, ...]]:
    own_object_ids: dict[str, set[str]] = {term_id: set() for term_id in terms}
    implied_source_type_ids: dict[str, set[str]] = {term_id: set() for term_id in terms}
    for relation in relations:
        own_object_ids[relation.subject_id].add(relation.object_id)
        if relation.relation is RelationType.IMPLIES_SOURCE_TYPE:
            implied_source_type_ids[relation.subject_id].add(relation.object_id)
    return {
        term_id: tuple(
            sorted(
                own_object_ids[term_id].union(
                    *(
                        implied_source_type_ids[ancestor_id]
                        for ancestor_id in _ancestor_ids(terms, term_id)
                    )
                )
            )
        )
        for term_id in terms
    }


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
        comment=str(document.get("comment") or "").strip(),
        curie_map=MappingProxyType(curie_map),
        mappings=mappings,
    )
