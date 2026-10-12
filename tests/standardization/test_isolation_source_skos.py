"""Protect the SKOS release file of the isolation-source vocabulary."""

from collections import Counter
from pathlib import Path

import pytest
from rdflib import RDF, SKOS, Graph, URIRef
from rdflib.compare import isomorphic

from baccurate.standardization.isolation_source_ontology import IsolationSourceOntology
from baccurate.standardization.isolation_source_skos import build_skos_graph

ONTOLOGY_DIRECTORY = Path(__file__).parents[2] / "data" / "reference" / "ontology"
RELEASE_PATH = ONTOLOGY_DIRECTORY / "baccurate.ttl"

SCHEME = URIRef("https://w3id.org/baccurate/vocabulary")

CEREBROSPINAL_FLUID_EXAMPLE = """\
@prefix bacc: <https://w3id.org/baccurate/BACC_> .
@prefix baccv: <https://w3id.org/baccurate/vocabulary/> .
@prefix iso-thes: <http://purl.org/iso25964/skos-thes#> .
@prefix oboInOwl: <http://www.geneontology.org/formats/oboInOwl#> .
@prefix skos: <http://www.w3.org/2004/02/skos/core#> .

bacc:0000013 a skos:Concept ;
    skos:inScheme <https://w3id.org/baccurate/vocabulary> ;
    skos:prefLabel "cerebrospinal fluid"@en ;
    skos:notation "BACC:0000013" ;
    skos:altLabel "csf"@en ;
    oboInOwl:hasExactSynonym "csf"@en ;
    skos:broader bacc:0000009 ;
    iso-thes:broaderGeneric bacc:0000009 ;
    baccv:located_in bacc:0000052 ;
    skos:related bacc:0000052 ;
    skos:exactMatch <http://purl.obolibrary.org/obo/UBERON_0001359> .

bacc:0000009 skos:narrower bacc:0000013 .
baccv:body_product skos:member bacc:0000013 .
"""


@pytest.fixture(scope="module")
def ontology() -> IsolationSourceOntology:
    return IsolationSourceOntology.load(ONTOLOGY_DIRECTORY)


@pytest.fixture(scope="module")
def release() -> Graph:
    return Graph().parse(RELEASE_PATH, format="turtle")


def _iri(ontology: IsolationSourceOntology, curie: str) -> URIRef:
    prefix, _, local_id = curie.partition(":")
    return URIRef(ontology.mapping_set.curie_map[prefix] + local_id)


def test_committed_release_equals_a_new_generation(
    ontology: IsolationSourceOntology, release: Graph
) -> None:
    assert isomorphic(release, build_skos_graph(ontology))


def test_every_concept_is_in_the_scheme_and_every_root_is_a_top_concept(
    ontology: IsolationSourceOntology, release: Graph
) -> None:
    concepts = set(release.subjects(RDF.type, SKOS.Concept))

    assert concepts == {_iri(ontology, term_id) for term_id in ontology.terms} | {
        _iri(ontology, term_id) for term_id in ontology.obsolete_terms
    }
    assert concepts == set(release.subjects(SKOS.inScheme, SCHEME))
    assert set(release.subjects(SKOS.topConceptOf, SCHEME)) == {
        _iri(ontology, term_id)
        for term_id, term in ontology.terms.items()
        if term.parent_id is None
    }


def test_release_holds_the_cerebrospinal_fluid_example(release: Graph) -> None:
    example = Graph().parse(data=CEREBROSPINAL_FLUID_EXAMPLE, format="turtle")

    assert set(example) - set(release) == set()


def test_skos_integrity_conditions_hold(release: Graph) -> None:
    concepts = set(release.subjects(RDF.type, SKOS.Concept))
    for concept in concepts:
        preferred = list(release.objects(concept, SKOS.prefLabel))
        alternative = set(release.objects(concept, SKOS.altLabel))
        assert alternative.isdisjoint(preferred), f"S13 {concept}"
        assert max(Counter(label.language for label in preferred).values()) == 1, f"S14 {concept}"

    for subject, target in release.subject_objects(SKOS.related):
        assert target not in release.transitive_objects(subject, SKOS.broader), f"S27 {subject}"
        assert subject not in release.transitive_objects(target, SKOS.broader), f"S27 {subject}"

    for subject, target in release.subject_objects(SKOS.exactMatch):
        for predicate in (SKOS.broadMatch, SKOS.relatedMatch):
            assert (subject, predicate, target) not in release, f"S46 {subject} {target}"
