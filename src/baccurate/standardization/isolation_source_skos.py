"""Export the isolation-source vocabulary and its mapping set as one SKOS graph."""

from __future__ import annotations

from collections.abc import Mapping

from rdflib import RDF, RDFS, SKOS, Graph, Literal, Namespace, URIRef
from rdflib.namespace import DCTERMS, OWL

from baccurate.standardization.isolation_source_ontology import (
    IsolationSourceOntology,
    ParentRelation,
    RelationType,
    SynonymType,
)

BACC = Namespace("https://w3id.org/baccurate/BACC_")
BACCV = Namespace("https://w3id.org/baccurate/vocabulary/")
ISO_THES = Namespace("http://purl.org/iso25964/skos-thes#")
OBO_IN_OWL = Namespace("http://www.geneontology.org/formats/oboInOwl#")

SCHEME = URIRef("https://w3id.org/baccurate/vocabulary")
SCHEME_TITLE = "BacCurate isolation-source vocabulary"
SCHEME_DESCRIPTION = (
    "The BacCurate isolation-source vocabulary describes the sources of bacterial isolates in "
    "NCBI BioSample metadata."
)
LICENSE = URIRef("https://creativecommons.org/licenses/by/4.0/")

RELATION_MEANINGS = {
    RelationType.LOCATED_IN: "The material is inside the site.",
    RelationType.PRODUCED_BY: "The site produces the material.",
    RelationType.COLLECTED_FROM: "The specimen is taken from the site.",
    RelationType.PART_OF: "The site is part of a second whole outside its parent path.",
    RelationType.HAS_PART: "The subject contains the object.",
    RelationType.IMPLIES_SOURCE_TYPE: (
        "Every sample under the subject belongs to the source-type term."
    ),
}

BROADER_PROPERTIES = {
    ParentRelation.KIND_OF: ISO_THES.broaderGeneric,
    ParentRelation.PART_OF: ISO_THES.broaderPartitive,
}

SYNONYM_PROPERTIES = {
    SynonymType.EXACT: OBO_IN_OWL.hasExactSynonym,
    SynonymType.NARROW: OBO_IN_OWL.hasNarrowSynonym,
    SynonymType.BROAD: OBO_IN_OWL.hasBroadSynonym,
    SynonymType.RELATED: OBO_IN_OWL.hasRelatedSynonym,
}


def term_iri(term_id: str) -> URIRef:
    """Return the w3id IRI of a ``BACC:`` term identifier."""
    return BACC[term_id.removeprefix("BACC:")]


def build_skos_graph(ontology: IsolationSourceOntology) -> Graph:
    """Build the SKOS release graph of the vocabulary and its mapping set."""
    graph = Graph(bind_namespaces="none")
    for prefix, namespace in (
        ("bacc", BACC),
        ("baccv", BACCV),
        ("skos", SKOS),
        ("iso-thes", ISO_THES),
        ("oboInOwl", OBO_IN_OWL),
        ("dcterms", DCTERMS),
        ("owl", OWL),
        ("rdfs", RDFS),
    ):
        graph.bind(prefix, namespace)

    graph.add((SCHEME, RDF.type, SKOS.ConceptScheme))
    graph.add((SCHEME, DCTERMS.title, Literal(SCHEME_TITLE)))
    graph.add((SCHEME, DCTERMS.description, Literal(SCHEME_DESCRIPTION)))
    graph.add((SCHEME, DCTERMS.license, LICENSE))
    graph.add((SCHEME, OWL.versionInfo, Literal(ontology.vocabulary_version)))

    for key, facet in ontology.facets.items():
        graph.add((BACCV[key], RDF.type, SKOS.Collection))
        graph.add((BACCV[key], SKOS.prefLabel, _english(key.replace("_", " "))))
        graph.add((BACCV[key], SKOS.definition, _english(facet.meaning)))

    for relation_type, meaning in RELATION_MEANINGS.items():
        relation = BACCV[relation_type.value]
        graph.add((relation, RDFS.subPropertyOf, SKOS.related))
        graph.add((relation, RDFS.label, _english(relation_type.value.replace("_", " "))))
        graph.add((relation, SKOS.definition, _english(meaning)))

    for term in ontology.terms.values():
        concept = term_iri(term.term_id)
        graph.add((concept, RDF.type, SKOS.Concept))
        graph.add((concept, SKOS.inScheme, SCHEME))
        graph.add((concept, SKOS.prefLabel, _english(term.label)))
        graph.add((concept, SKOS.notation, Literal(term.term_id)))
        graph.add((BACCV[term.facet], SKOS.member, concept))
        if term.parent_id is None:
            graph.add((concept, SKOS.topConceptOf, SCHEME))
            graph.add((SCHEME, SKOS.hasTopConcept, concept))
        else:
            parent = term_iri(term.parent_id)
            graph.add((concept, SKOS.broader, parent))
            graph.add((parent, SKOS.narrower, concept))
            graph.add((concept, BROADER_PROPERTIES[term.parent_relation], parent))
        for synonym in term.synonyms:
            graph.add((concept, SYNONYM_PROPERTIES[synonym.synonym_type], _english(synonym.text)))
            if synonym.synonym_type is SynonymType.EXACT:
                graph.add((concept, SKOS.altLabel, _english(synonym.text)))
        if term.definition:
            graph.add((concept, SKOS.definition, _english(term.definition)))
        if term.scope_note:
            graph.add((concept, SKOS.scopeNote, _english(term.scope_note)))

    for relation in ontology.relations:
        subject, target = term_iri(relation.subject_id), term_iri(relation.object_id)
        graph.add((subject, BACCV[relation.relation.value], target))
        graph.add((subject, SKOS.related, target))

    for obsolete in ontology.obsolete_terms.values():
        concept = term_iri(obsolete.term_id)
        graph.add((concept, RDF.type, SKOS.Concept))
        graph.add((concept, SKOS.inScheme, SCHEME))
        graph.add((concept, SKOS.prefLabel, _english(obsolete.label)))
        graph.add((concept, OWL.deprecated, Literal(True)))
        graph.add((concept, SKOS.historyNote, _english(obsolete.note)))
        for replacement_id in obsolete.replaced_by:
            graph.add((concept, DCTERMS.isReplacedBy, term_iri(replacement_id)))

    curie_map = ontology.mapping_set.curie_map
    for mapping in ontology.mapping_set.mappings:
        graph.add(
            (
                term_iri(mapping.subject_id),
                _expand(mapping.predicate_id, curie_map),
                _expand(mapping.object_id, curie_map),
            )
        )
    return graph


def _english(text: str) -> Literal:
    return Literal(text, lang="en")


def _expand(curie: str, curie_map: Mapping[str, str]) -> URIRef:
    prefix, _, local_id = curie.partition(":")
    return URIRef(curie_map[prefix] + local_id)
