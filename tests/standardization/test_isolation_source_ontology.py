"""Protect the public isolation-source ontology loading contract."""

import csv
import re
from pathlib import Path

import pytest
import yaml

from baccurate.standardization.isolation_source_ontology import (
    IsolationSourceOntology,
    IsolationSourceOntologyError,
    ParentRelation,
    RelationType,
    Synonym,
    SynonymType,
    TermRelation,
)

ONTOLOGY_DIRECTORY = Path(__file__).parents[2] / "data" / "reference" / "ontology"
FIXTURE_DIRECTORY = Path(__file__).parents[1] / "fixtures" / "standardization" / "ontology"

FACETS_YAML = """\
schema_version: 1
vocabulary_version: "test"

facets:
  source_type:
    standards_bindings:
      mixs: []
      pha4ge_geneepio: []
      external_ontology_namespaces:
        - ENVO
    meaning: The broad kind of isolation source.

  environmental_material:
    standards_bindings:
      mixs:
        - env_medium
      pha4ge_geneepio:
        - environmental material
      external_ontology_namespaces:
        - ENVO
    meaning: The non-host material sampled.
"""

TERMS_TSV = "\n".join(
    (
        "term_id\tlabel\tfacet\tparent_id\tparent_relation\tdefinition\tscope_note",
        "BACC:0000001\tenvironmental\tsource_type\t\t\tMaterial from the environment.\t",
        "BACC:0000002\tbuilt environment\tsource_type\tBACC:0000001\tkind_of\t"
        "Material from a built environment.\t",
        "BACC:0000003\tenvironmental material\tenvironmental_material\t\t\t"
        "Material that was sampled.\t",
        "BACC:0000004\tsoil\tenvironmental_material\tBACC:0000003\tkind_of\tSoil material.\t"
        "Use for sampled soil.",
        "",
    )
)

SYNONYMS_TSV = (
    "term_id\tsynonym\tsynonym_type\n"
    "BACC:0000001\tenvironment\texact\n"
    "BACC:0000003\tmaterial\tbroad\n"
    "BACC:0000004\tdirt\texact\n"
)

OBSOLETE_TERMS_TSV = (
    "term_id\tlabel\treplaced_by\tnote\n"
    "BACC:0000005\tnatural environment\tBACC:0000001\tObsolete since 2026-01-01.\n"
)

RELATIONS_TSV = "subject_id\trelation\tobject_id\nBACC:0000004\timplies_source_type\tBACC:0000001\n"

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
subject_source: baccurate:vocabulary
subject_source_version: "test"
curie_map:
  BACC: https://w3id.org/baccurate/BACC_
  baccurate: https://w3id.org/baccurate/
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
        "synonyms.tsv": SYNONYMS_TSV,
        "obsolete_terms.tsv": OBSOLETE_TERMS_TSV,
        "relations.tsv": RELATIONS_TSV,
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


def test_authored_identifiers_expand_to_w3id_iris() -> None:
    ontology = IsolationSourceOntology.load(ONTOLOGY_DIRECTORY)
    curie_map = ontology.mapping_set.curie_map

    for term_id in (*ontology.terms, *ontology.obsolete_terms):
        prefix, _, local_id = term_id.partition(":")
        assert re.fullmatch(r"\d{7}", local_id), term_id
        assert curie_map[prefix] + local_id == f"https://w3id.org/baccurate/BACC_{local_id}"


def test_authored_mappings_bind_entity_terms_to_at_most_one_exact_class() -> None:
    ontology = IsolationSourceOntology.load(ONTOLOGY_DIRECTORY)
    mappings = ontology.mapping_set.mappings
    exact_subjects = [m.subject_id for m in mappings if m.predicate_id == "skos:exactMatch"]

    assert {
        m.subject_id for m in mappings if ontology.terms[m.subject_id].facet == "source_type"
    } == set()
    assert {s for s in exact_subjects if exact_subjects.count(s) > 1} == set()


def test_each_authored_term_without_an_exact_match_has_a_definition() -> None:
    ontology = IsolationSourceOntology.load(ONTOLOGY_DIRECTORY)

    for term_id, term in ontology.terms.items():
        if term_id not in ontology.exact_match_ids:
            assert term.definition, term_id


def test_load_returns_complete_facet_term_and_mapping_shapes(ontology_directory: Path) -> None:
    ontology = IsolationSourceOntology.load(ontology_directory)

    assert set(ontology.facets) == {"source_type", "environmental_material"}
    assert ontology.facets["environmental_material"].meaning == "The non-host material sampled."
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
        soil.parent_relation,
        soil.synonyms,
        soil.definition,
        soil.scope_note,
    ) == (
        "soil",
        "environmental_material",
        "BACC:0000003",
        ParentRelation.KIND_OF,
        (Synonym("dirt", SynonymType.EXACT),),
        "Soil material.",
        "Use for sampled soil.",
    )
    assert ontology.match_keys["dirt"] == "BACC:0000004"
    assert ontology.relations == (
        TermRelation("BACC:0000004", RelationType.IMPLIES_SOURCE_TYPE, "BACC:0000001"),
    )
    assert ontology.relation_object_ids["BACC:0000004"] == ("BACC:0000001",)
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


def test_parent_cannot_cross_facets(ontology_directory: Path) -> None:
    _update_tsv_row(
        ontology_directory / "terms.tsv",
        match_column="term_id",
        match_value="BACC:0000003",
        changes={"parent_id": "BACC:0000001", "parent_relation": "kind_of"},
    )

    with pytest.raises(
        IsolationSourceOntologyError,
        match=(
            r"term 'BACC:0000003' parent_id: parent 'BACC:0000001' belongs to facet "
            r"'source_type'; parents must belong to 'environmental_material'"
        ),
    ):
        IsolationSourceOntology.load(ontology_directory)


@pytest.mark.parametrize(
    ("term_id", "parent_relation", "message"),
    [
        ("BACC:0000002", "", "must be 'kind_of' or 'part_of'"),
        ("BACC:0000001", "kind_of", "must be empty when parent_id is empty"),
        ("BACC:0000002", "broader", "must be 'kind_of' or 'part_of'"),
    ],
    ids=["missing", "without parent", "unknown"],
)
def test_parent_relation_is_set_exactly_for_a_parent(
    ontology_directory: Path, term_id: str, parent_relation: str, message: str
) -> None:
    _update_tsv_row(
        ontology_directory / "terms.tsv",
        match_column="term_id",
        match_value=term_id,
        changes={"parent_relation": parent_relation},
    )

    with pytest.raises(IsolationSourceOntologyError, match=rf"row \d+ parent_relation: {message}"):
        IsolationSourceOntology.load(ontology_directory)


@pytest.mark.parametrize(
    ("row", "message"),
    [
        (
            "BACC:9999999\thas_part\tBACC:0000003",
            r"row 3 subject_id: unknown term 'BACC:9999999'",
        ),
        (
            "BACC:0000004\thas_part\tBACC:9999999",
            r"row 3 object_id: unknown term 'BACC:9999999'",
        ),
        (
            "BACC:0000004\trelated\tBACC:0000002",
            r"row 3 relation: unknown relation type 'related'",
        ),
        (
            "BACC:0000004\timplies_source_type\tBACC:0000001",
            r"row 3: duplicate of row 2",
        ),
        (
            "BACC:0000002\timplies_source_type\tBACC:0000003",
            r"row 3 object_id: object 'BACC:0000003' of 'implies_source_type' must be a "
            r"'source_type' term",
        ),
        (
            "BACC:0000003\tlocated_in\tBACC:0000001",
            r"row 3 object_id: object 'BACC:0000001' of 'located_in' must not be a "
            r"'source_type' term",
        ),
        (
            "BACC:0000004\tpart_of\tBACC:0000003",
            r"row 3 object_id: object 'BACC:0000003' is the subject 'BACC:0000004' or one of "
            r"its ancestors or descendants",
        ),
        (
            "BACC:0000003\thas_part\tBACC:0000004",
            r"row 3 object_id: object 'BACC:0000004' is the subject 'BACC:0000003' or one of "
            r"its ancestors or descendants",
        ),
    ],
    ids=[
        "unknown subject",
        "unknown object",
        "unknown relation type",
        "duplicate",
        "implied object outside source_type",
        "other object in source_type",
        "object is an ancestor",
        "object is a descendant",
    ],
)
def test_invalid_relation_names_the_row(ontology_directory: Path, row: str, message: str) -> None:
    with (ontology_directory / "relations.tsv").open("a", encoding="utf-8") as relations:
        relations.write(f"{row}\n")

    with pytest.raises(IsolationSourceOntologyError, match=message):
        IsolationSourceOntology.load(ontology_directory)


def test_only_implies_source_type_relations_apply_to_descendants() -> None:
    ontology = IsolationSourceOntology.load(FIXTURE_DIRECTORY)
    tested_inheritance = set()

    for relation in ontology.relations:
        assert relation.object_id in ontology.relation_object_ids[relation.subject_id]
        for term_id in ontology.terms:
            if relation.subject_id not in _ancestor_ids(ontology, term_id):
                continue
            inherited = relation.relation is RelationType.IMPLIES_SOURCE_TYPE
            applied = relation.object_id in ontology.relation_object_ids[term_id]
            assert applied == inherited, (relation, term_id)
            tested_inheritance.add(inherited)

    assert tested_inheritance == {True, False}


def _ancestor_ids(ontology: IsolationSourceOntology, term_id: str) -> set[str]:
    ancestor_ids = set()
    parent_id = ontology.terms[term_id].parent_id
    while parent_id is not None:
        ancestor_ids.add(parent_id)
        parent_id = ontology.terms[parent_id].parent_id
    return ancestor_ids


@pytest.mark.parametrize(
    ("filename", "line", "message"),
    [
        (
            "obsolete_terms.tsv",
            "BACC:0000004\tsoil\tBACC:0000003\tObsolete.",
            r"row 3 term_id: 'BACC:0000004' is an active term",
        ),
        (
            "obsolete_terms.tsv",
            "BACC:0000005\tnatural environment\tBACC:0000001\tObsolete.",
            r"row 3 term_id: duplicate term identifier 'BACC:0000005'",
        ),
        (
            "obsolete_terms.tsv",
            "BACC:0000006\tmud\tBACC:0000004;BACC:0000005\tObsolete.",
            r"row 3 replaced_by: replacement 'BACC:0000005' is not an active term",
        ),
    ],
    ids=["active term", "duplicate", "obsolete replacement"],
)
def test_invalid_obsolete_term_is_rejected(
    ontology_directory: Path, filename: str, line: str, message: str
) -> None:
    with (ontology_directory / filename).open("a", encoding="utf-8") as source:
        source.write(f"{line}\n")

    with pytest.raises(IsolationSourceOntologyError, match=message):
        IsolationSourceOntology.load(ontology_directory)


def test_obsolete_term_without_replacement_is_accepted(ontology_directory: Path) -> None:
    with (ontology_directory / "obsolete_terms.tsv").open("a", encoding="utf-8") as source:
        source.write("BACC:0000006\tmud\t\tObsolete.\n")

    ontology = IsolationSourceOntology.load(ontology_directory)

    assert ontology.obsolete_terms["BACC:0000006"].replaced_by == ()


@pytest.mark.parametrize("key", ["mapping_set_version", "subject_source_version"])
def test_mapping_set_version_must_equal_the_vocabulary_version(
    ontology_directory: Path, key: str
) -> None:
    mapping_set_path = ontology_directory / "mappings.sssom.yml"
    document = yaml.safe_load(mapping_set_path.read_text(encoding="utf-8"))
    document[key] = "other"
    mapping_set_path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")

    with pytest.raises(
        IsolationSourceOntologyError,
        match=rf"{key}: 'other' differs from the vocabulary_version 'test'",
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


def test_cycle_of_parents_names_the_terms(ontology_directory: Path) -> None:
    _update_tsv_row(
        ontology_directory / "terms.tsv",
        match_column="term_id",
        match_value="BACC:0000001",
        changes={"parent_id": "BACC:0000002", "parent_relation": "kind_of"},
    )

    with pytest.raises(
        IsolationSourceOntologyError,
        match=(
            r"term 'BACC:0000001' parent_id: parent cycle: "
            r"BACC:0000001 -> BACC:0000002 -> BACC:0000001"
        ),
    ):
        IsolationSourceOntology.load(ontology_directory)


@pytest.mark.parametrize(
    "field", ["cardinality", "output_column", "render_order", "classifier_guidance"]
)
def test_facet_rejects_a_pipeline_field(ontology_directory: Path, field: str) -> None:
    facets_path = ontology_directory / "facets.yaml"
    document = yaml.safe_load(facets_path.read_text(encoding="utf-8"))
    document["facets"]["source_type"][field] = 1
    facets_path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")

    with pytest.raises(
        IsolationSourceOntologyError,
        match=rf"facets\.source_type\.{field}: unknown facet field",
    ):
        IsolationSourceOntology.load(ontology_directory)


def test_terms_reject_the_host_recovery_column(ontology_directory: Path) -> None:
    terms_path = ontology_directory / "terms.tsv"
    fieldnames, rows = _read_tsv(terms_path)
    _write_tsv(
        terms_path,
        [*fieldnames, "enables_host_recovery"],
        [{**row, "enables_host_recovery": "false"} for row in rows],
    )

    with pytest.raises(IsolationSourceOntologyError, match="header: must contain these columns"):
        IsolationSourceOntology.load(ontology_directory)


def test_semantic_fingerprints_ignore_reference_file_formatting(
    ontology_directory: Path,
) -> None:
    original = IsolationSourceOntology.load(ontology_directory)

    for filename in ("facets.yaml", "mappings.sssom.yml"):
        path = ontology_directory / filename
        document = yaml.safe_load(path.read_text(encoding="utf-8"))
        path.write_text(yaml.safe_dump(document, sort_keys=True), encoding="utf-8")
    for filename in (
        "terms.tsv",
        "synonyms.tsv",
        "obsolete_terms.tsv",
        "relations.tsv",
        "mappings.sssom.tsv",
    ):
        path = ontology_directory / filename
        text = path.read_text(encoding="utf-8")
        path.write_bytes(text.replace("\n", "\r\n").encode("utf-8"))

    reformatted = IsolationSourceOntology.load(ontology_directory)

    assert original.vocabulary_fingerprint == reformatted.vocabulary_fingerprint
    assert original.mapping_set_fingerprint == reformatted.mapping_set_fingerprint


@pytest.mark.parametrize(
    ("filename", "changes"),
    [
        ("terms.tsv", {"definition": "Soil that was sampled."}),
        ("terms.tsv", {"scope_note": "Use only when soil was sampled."}),
        ("terms.tsv", {"parent_relation": "part_of"}),
        ("synonyms.tsv", {"synonym_type": "related"}),
    ],
)
def test_term_edit_changes_only_vocabulary_fingerprint(
    ontology_directory: Path, filename: str, changes: dict[str, str]
) -> None:
    original = IsolationSourceOntology.load(ontology_directory)
    _update_tsv_row(
        ontology_directory / filename,
        match_column="term_id",
        match_value="BACC:0000004",
        changes=changes,
    )

    changed_vocabulary = IsolationSourceOntology.load(ontology_directory)

    assert original.vocabulary_fingerprint != changed_vocabulary.vocabulary_fingerprint
    assert original.mapping_set_fingerprint == changed_vocabulary.mapping_set_fingerprint


def test_obsolete_term_edit_changes_only_vocabulary_fingerprint(ontology_directory: Path) -> None:
    original = IsolationSourceOntology.load(ontology_directory)
    _update_tsv_row(
        ontology_directory / "obsolete_terms.tsv",
        match_column="term_id",
        match_value="BACC:0000005",
        changes={"replaced_by": "BACC:0000001;BACC:0000002"},
    )

    changed_vocabulary = IsolationSourceOntology.load(ontology_directory)

    assert original.vocabulary_fingerprint != changed_vocabulary.vocabulary_fingerprint
    assert original.mapping_set_fingerprint == changed_vocabulary.mapping_set_fingerprint


def test_relation_edit_changes_only_vocabulary_fingerprint(ontology_directory: Path) -> None:
    original = IsolationSourceOntology.load(ontology_directory)
    _update_tsv_row(
        ontology_directory / "relations.tsv",
        match_column="subject_id",
        match_value="BACC:0000004",
        changes={"object_id": "BACC:0000002"},
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
