"""Protect the public isolation-source ontology loading contract."""

import csv
from pathlib import Path

import pytest
import yaml

from baccurate.standardization.isolation_source_ontology import (
    FacetCardinality,
    IsolationSourceOntology,
    IsolationSourceOntologyError,
)

ONTOLOGY_DIRECTORY = Path(__file__).parents[2] / "data" / "reference" / "ontology"

FACETS_YAML = """\
schema_version: 1
vocabulary_version: "test"
retired_term_ids: []

facets:
  source_type:
    cardinality: single
    output_column: iso_source_type
    render_order: 1
    standards_bindings:
      mixs: []
      pha4ge_geneepio: []
      external_ontology_namespaces:
        - ENVO
    meaning: The broad kind of isolation source.
    classifier_guidance: Select one when another facet is populated.

  environmental_material:
    cardinality: multiple
    output_column: iso_environmental_material
    render_order: 2
    standards_bindings:
      mixs:
        - env_medium
      pha4ge_geneepio:
        - environmental material
      external_ontology_namespaces:
        - ENVO
    meaning: The non-host material sampled.
    classifier_guidance: Select only a material named by the metadata.
"""

TERMS_TSV = "\n".join(
    (
        "term_id\tlabel\tfacet\tparent_id\tsynonyms\tcrosslink_targets\tcuration_note\t"
        "enables_host_recovery",
        "BACC:0000001\tenvironmental\tsource_type\t\tenvironment\t\t\tfalse",
        "BACC:0000002\tbuilt environment\tsource_type\tBACC:0000001\t\t\t\tfalse",
        "BACC:0000003\tenvironmental material\tenvironmental_material\t\tmaterial\t\t\tfalse",
        "BACC:0000004\tsoil\tenvironmental_material\tBACC:0000003\tdirt\tBACC:0000001\t"
        "Use for sampled soil.\ttrue",
        "",
    )
)

MAPPINGS_TSV = "\n".join(
    (
        "subject_id\tsubject_label\tpredicate_id\tobject_id\tmapping_justification\t"
        "object_source\tobject_source_version\tcomment",
        "BACC:0000004\tsoil\tskos:exactMatch\tENVO:00000001\t"
        'semapv:ManualMappingCuration\tobo:envo.owl\t2026-01-01\t""',
        "BACC:0000004\tsoil\tskos:narrowMatch\tENVO:00000002\t"
        "semapv:ManualMappingCuration\tobo:envo.owl\t2026-01-01\tNarrower class.",
        "BACC:0000004\tsoil\tskos:broadMatch\tENVO:00000003\t"
        "semapv:ManualMappingCuration\tobo:envo.owl\t2026-01-01\tBroader class.",
        "BACC:0000004\tsoil\tskos:closeMatch\tENVO:00000004\t"
        "semapv:ManualMappingCuration\tobo:envo.owl\t2026-01-01\tRelated class.",
        "",
    )
)

MAPPING_SET_YAML = """\
mapping_set_id: https://example.org/baccurate/test-mappings.tsv
mapping_set_version: "test"
mapping_set_title: Test isolation-source mappings
mapping_set_description: Minimal mapping set for the public loader contract.
mapping_date: "2026-01-01"
license: https://opensource.org/license/mit/
subject_source: baccurate:terms.tsv
subject_source_version: "test"
comment: BACC identifiers are local to this test vocabulary.
curie_map:
  BACC: https://example.org/baccurate/terms#BACC_
  baccurate: https://example.org/baccurate/
  ENVO: http://purl.obolibrary.org/obo/ENVO_
  obo: http://purl.obolibrary.org/obo/
  semapv: https://w3id.org/semapv/vocab/
  skos: http://www.w3.org/2004/02/skos/core#
"""


@pytest.fixture
def ontology_directory(tmp_path: Path) -> Path:
    directory = tmp_path / "ontology"
    directory.mkdir()
    for filename, content in {
        "facets.yaml": FACETS_YAML,
        "terms.tsv": TERMS_TSV,
        "mappings.sssom.tsv": MAPPINGS_TSV,
        "mappings.sssom.yml": MAPPING_SET_YAML,
    }.items():
        (directory / filename).write_text(content, encoding="utf-8", newline="\n")
    return directory


def _read_tsv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8", newline="") as source:
        reader = csv.DictReader(source, delimiter="\t")
        assert reader.fieldnames is not None
        return reader.fieldnames, list(reader)


def _write_tsv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as destination:
        writer = csv.DictWriter(
            destination,
            fieldnames=fieldnames,
            delimiter="\t",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def _update_tsv_row(
    path: Path,
    *,
    match_column: str,
    match_value: str,
    changes: dict[str, str],
) -> None:
    fieldnames, rows = _read_tsv(path)
    matching_rows = [row for row in rows if row[match_column] == match_value]
    assert len(matching_rows) == 1
    matching_rows[0].update(changes)
    _write_tsv(path, fieldnames, rows)


def test_authored_isolation_source_ontology_loads() -> None:
    IsolationSourceOntology.load(ONTOLOGY_DIRECTORY)


def test_authored_ontology_preserves_host_recovery_eligible_term_set() -> None:
    ontology = IsolationSourceOntology.load(ONTOLOGY_DIRECTORY)

    assert {term.term_id for term in ontology.terms.values() if term.enables_host_recovery} == {
        "BACC:0000001",  # host-associated
        "BACC:0000002",  # animal host
        "BACC:0000003",  # plant host
        "BACC:0000094",  # plant food product
        "BACC:0000095",  # animal food product
        "BACC:0000096",  # dairy product
        "BACC:0000097",  # meat product
        "BACC:0000098",  # seafood product
    }


def test_load_returns_complete_facet_term_and_mapping_shapes(ontology_directory: Path) -> None:
    ontology = IsolationSourceOntology.load(ontology_directory)

    assert set(ontology.facets) == {"source_type", "environmental_material"}
    assert ontology.facets["source_type"].cardinality is FacetCardinality.SINGLE
    assert ontology.facets["environmental_material"].output_column == ("iso_environmental_material")
    assert set(ontology.terms) == {
        "BACC:0000001",
        "BACC:0000002",
        "BACC:0000003",
        "BACC:0000004",
    }
    soil = ontology.terms["BACC:0000004"]
    assert (
        soil.label,
        soil.facet,
        soil.parent_id,
        soil.synonyms,
        soil.crosslink_target_ids,
        soil.curation_note,
        soil.enables_host_recovery,
    ) == (
        "soil",
        "environmental_material",
        "BACC:0000003",
        ("dirt",),
        ("BACC:0000001",),
        "Use for sampled soil.",
        True,
    )
    assert ontology.mapping_set.mapping_set_version == "test"
    assert len(ontology.mapping_set.mappings) == 4


def test_only_exact_and_narrow_mappings_resolve(ontology_directory: Path) -> None:
    ontology = IsolationSourceOntology.load(ontology_directory)

    assert ontology.resolved_mapping_terms["ENVO:00000001"].term_id == "BACC:0000004"
    assert ontology.resolved_mapping_terms["ENVO:00000002"].term_id == "BACC:0000004"
    assert "ENVO:00000003" not in ontology.resolved_mapping_terms
    assert "ENVO:00000004" not in ontology.resolved_mapping_terms


def test_duplicate_term_identifier_names_the_term(ontology_directory: Path) -> None:
    terms_path = ontology_directory / "terms.tsv"
    fieldnames, rows = _read_tsv(terms_path)
    rows.append({**rows[0], "label": "duplicate environmental source"})
    _write_tsv(terms_path, fieldnames, rows)

    with pytest.raises(
        IsolationSourceOntologyError,
        match=r"row \d+ term_id: duplicate term identifier 'BACC:0000001'",
    ):
        IsolationSourceOntology.load(ontology_directory)


def test_label_duplicated_across_facets_names_both_facets(ontology_directory: Path) -> None:
    _update_tsv_row(
        ontology_directory / "terms.tsv",
        match_column="term_id",
        match_value="BACC:0000003",
        changes={"label": "environmental"},
    )

    with pytest.raises(
        IsolationSourceOntologyError,
        match=(
            r"row \d+ label: duplicate label 'environmental' across facets "
            r"'source_type' and 'environmental_material'"
        ),
    ):
        IsolationSourceOntology.load(ontology_directory)


def test_term_naming_undeclared_facet_raises(ontology_directory: Path) -> None:
    _update_tsv_row(
        ontology_directory / "terms.tsv",
        match_column="term_id",
        match_value="BACC:0000001",
        changes={"facet": "undeclared_facet"},
    )

    with pytest.raises(
        IsolationSourceOntologyError,
        match=r"term 'BACC:0000001' facet: undeclared facet 'undeclared_facet'",
    ):
        IsolationSourceOntology.load(ontology_directory)


def test_unresolvable_parent_names_the_term(ontology_directory: Path) -> None:
    _update_tsv_row(
        ontology_directory / "terms.tsv",
        match_column="term_id",
        match_value="BACC:0000002",
        changes={"parent_id": "BACC:9999999"},
    )

    with pytest.raises(
        IsolationSourceOntologyError,
        match=r"term 'BACC:0000002' parent_id: unresolvable parent 'BACC:9999999'",
    ):
        IsolationSourceOntology.load(ontology_directory)


def test_parent_relation_cannot_cross_facets(ontology_directory: Path) -> None:
    _update_tsv_row(
        ontology_directory / "terms.tsv",
        match_column="term_id",
        match_value="BACC:0000003",
        changes={"parent_id": "BACC:0000001"},
    )

    with pytest.raises(
        IsolationSourceOntologyError,
        match=(
            r"term 'BACC:0000003' parent_id: parent 'BACC:0000001' belongs to facet "
            r"'source_type'; parents must belong to 'environmental_material'"
        ),
    ):
        IsolationSourceOntology.load(ontology_directory)


def test_unresolvable_crosslink_names_the_term(ontology_directory: Path) -> None:
    _update_tsv_row(
        ontology_directory / "terms.tsv",
        match_column="term_id",
        match_value="BACC:0000004",
        changes={"crosslink_targets": "BACC:9999999"},
    )

    with pytest.raises(
        IsolationSourceOntologyError,
        match=(
            r"term 'BACC:0000004' crosslink_targets: unresolvable crosslink target "
            r"'BACC:9999999'"
        ),
    ):
        IsolationSourceOntology.load(ontology_directory)


def test_mapping_subject_that_is_not_a_term_names_the_row(ontology_directory: Path) -> None:
    _update_tsv_row(
        ontology_directory / "mappings.sssom.tsv",
        match_column="object_id",
        match_value="ENVO:00000001",
        changes={"subject_id": "BACC:9999999"},
    )

    with pytest.raises(
        IsolationSourceOntologyError,
        match=r"row \d+ subject_id: mapping subject 'BACC:9999999' is not an ontology term",
    ):
        IsolationSourceOntology.load(ontology_directory)


@pytest.mark.parametrize(
    ("filename", "match_column", "match_value", "empty_column", "message"),
    [
        (
            "terms.tsv",
            "term_id",
            "BACC:0000001",
            "term_id",
            r"row \d+ term_id: must be a non-empty term identifier",
        ),
        (
            "terms.tsv",
            "term_id",
            "BACC:0000001",
            "label",
            r"row \d+ label: must be a non-empty ontology display term",
        ),
        (
            "mappings.sssom.tsv",
            "object_id",
            "ENVO:00000001",
            "predicate_id",
            r"row \d+ predicate_id: must be a non-empty mapping predicate",
        ),
        (
            "mappings.sssom.tsv",
            "object_id",
            "ENVO:00000001",
            "object_id",
            r"row \d+ object_id: must be a non-empty external ontology identifier",
        ),
    ],
)
def test_required_term_and_mapping_fields_cannot_be_empty(
    ontology_directory: Path,
    filename: str,
    match_column: str,
    match_value: str,
    empty_column: str,
    message: str,
) -> None:
    _update_tsv_row(
        ontology_directory / filename,
        match_column=match_column,
        match_value=match_value,
        changes={empty_column: ""},
    )

    with pytest.raises(IsolationSourceOntologyError, match=message):
        IsolationSourceOntology.load(ontology_directory)


def test_cycle_in_parent_relation_names_the_terms(ontology_directory: Path) -> None:
    _update_tsv_row(
        ontology_directory / "terms.tsv",
        match_column="term_id",
        match_value="BACC:0000001",
        changes={"parent_id": "BACC:0000002"},
    )

    with pytest.raises(
        IsolationSourceOntologyError,
        match=(
            r"term 'BACC:0000001' parent_id: parent cycle: "
            r"BACC:0000001 -> BACC:0000002 -> BACC:0000001"
        ),
    ):
        IsolationSourceOntology.load(ontology_directory)


def test_term_cannot_crosslink_to_multiple_values_in_single_facet(
    ontology_directory: Path,
) -> None:
    _update_tsv_row(
        ontology_directory / "terms.tsv",
        match_column="term_id",
        match_value="BACC:0000004",
        changes={"crosslink_targets": "BACC:0000001;BACC:0000002"},
    )

    with pytest.raises(
        IsolationSourceOntologyError,
        match=(
            r"term 'BACC:0000004' crosslink_targets: assigns multiple values "
            r"to single-valued facet 'source_type': BACC:0000001, BACC:0000002"
        ),
    ):
        IsolationSourceOntology.load(ontology_directory)


def test_semantic_fingerprints_ignore_reference_file_formatting(
    ontology_directory: Path,
) -> None:
    original = IsolationSourceOntology.load(ontology_directory)

    for filename in ("facets.yaml", "mappings.sssom.yml"):
        path = ontology_directory / filename
        document = yaml.safe_load(path.read_text(encoding="utf-8"))
        path.write_text(yaml.safe_dump(document, sort_keys=True), encoding="utf-8")
    for filename in ("terms.tsv", "mappings.sssom.tsv"):
        path = ontology_directory / filename
        text = path.read_text(encoding="utf-8")
        path.write_bytes(text.replace("\n", "\r\n").encode("utf-8"))

    reformatted = IsolationSourceOntology.load(ontology_directory)

    assert original.vocabulary_fingerprint == reformatted.vocabulary_fingerprint
    assert original.mapping_set_fingerprint == reformatted.mapping_set_fingerprint


def test_term_edit_changes_only_vocabulary_fingerprint(ontology_directory: Path) -> None:
    original = IsolationSourceOntology.load(ontology_directory)
    _update_tsv_row(
        ontology_directory / "terms.tsv",
        match_column="term_id",
        match_value="BACC:0000004",
        changes={"curation_note": "Use only when soil was sampled."},
    )

    changed_vocabulary = IsolationSourceOntology.load(ontology_directory)

    assert original.vocabulary_fingerprint != changed_vocabulary.vocabulary_fingerprint
    assert original.mapping_set_fingerprint == changed_vocabulary.mapping_set_fingerprint


def test_facet_edit_changes_only_vocabulary_fingerprint(ontology_directory: Path) -> None:
    original = IsolationSourceOntology.load(ontology_directory)
    facets_path = ontology_directory / "facets.yaml"
    document = yaml.safe_load(facets_path.read_text(encoding="utf-8"))
    document["facets"]["environmental_material"]["meaning"] = "The environmental substance sampled."
    facets_path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")

    changed_vocabulary = IsolationSourceOntology.load(ontology_directory)

    assert original.vocabulary_fingerprint != changed_vocabulary.vocabulary_fingerprint
    assert original.mapping_set_fingerprint == changed_vocabulary.mapping_set_fingerprint


def test_mapping_edit_changes_only_mapping_set_fingerprint(ontology_directory: Path) -> None:
    original = IsolationSourceOntology.load(ontology_directory)
    _update_tsv_row(
        ontology_directory / "mappings.sssom.tsv",
        match_column="object_id",
        match_value="ENVO:00000001",
        changes={"predicate_id": "skos:broadMatch"},
    )

    changed_mapping_set = IsolationSourceOntology.load(ontology_directory)

    assert original.vocabulary_fingerprint == changed_mapping_set.vocabulary_fingerprint
    assert original.mapping_set_fingerprint != changed_mapping_set.mapping_set_fingerprint
