"""Protect the public isolation-source ontology loading contract."""

import csv
from pathlib import Path

import pytest
import yaml

from baccurate.paths import DEFAULT_ISOLATION_SOURCE_ONTOLOGY_DIRECTORY
from baccurate.standardization.isolation_source_ontology import (
    IsolationSourceOntology,
    IsolationSourceOntologyError,
    ParentRelation,
    RelationType,
    Synonym,
    SynonymType,
    TermRelation,
    render_ontology,
)

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


def test_prompt_writes_one_term_note_for_each_relation() -> None:
    ontology = IsolationSourceOntology.load(DEFAULT_ISOLATION_SOURCE_ONTOLOGY_DIRECTORY)
    sections = {
        facet_key: render_ontology(ontology, {facet_key: ""}).splitlines()
        for facet_key in ontology.facets
    }

    assert ontology.relations
    for relation in ontology.relations:
        subject = ontology.terms[relation.subject_id]
        relation_name = relation.relation.value.replace("_", " ")
        note = f'- "{subject.label}": {relation_name} {ontology.terms[relation.object_id].label}'
        section = sections[subject.facet]
        assert section.count(note) == 1, relation
        assert section.index(note) > section.index("### Term notes"), relation


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


def _reformat_every_file(directory: Path) -> None:
    for filename in ("facets.yaml", "mappings.sssom.yml"):
        path = directory / filename
        document = yaml.safe_load(path.read_text(encoding="utf-8"))
        path.write_text(yaml.safe_dump(document, sort_keys=True), encoding="utf-8")
    for filename in (
        "terms.tsv",
        "synonyms.tsv",
        "obsolete_terms.tsv",
        "relations.tsv",
        "mappings.sssom.tsv",
    ):
        path = directory / filename
        text = path.read_text(encoding="utf-8")
        path.write_bytes(text.replace("\n", "\r\n").encode("utf-8"))


def _edit_facet_meaning(directory: Path) -> None:
    facets_path = directory / "facets.yaml"
    document = yaml.safe_load(facets_path.read_text(encoding="utf-8"))
    document["facets"]["environmental_material"]["meaning"] = "The environmental substance sampled."
    facets_path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")


def _edit_row(filename: str, match_column: str, match_value: str, **changes: str):
    return lambda directory: _update_tsv_row(
        directory / filename,
        match_column=match_column,
        match_value=match_value,
        changes=changes,
    )


@pytest.mark.parametrize(
    ("edit", "vocabulary_changes", "mapping_set_changes"),
    [
        pytest.param(_reformat_every_file, False, False, id="formatting"),
        pytest.param(
            _edit_row("terms.tsv", "term_id", "BACC:0000004", definition="Soil that was sampled."),
            True,
            False,
            id="definition",
        ),
        pytest.param(
            _edit_row(
                "terms.tsv", "term_id", "BACC:0000004", scope_note="Use only when soil was sampled."
            ),
            True,
            False,
            id="scope note",
        ),
        pytest.param(
            _edit_row("terms.tsv", "term_id", "BACC:0000004", parent_relation="part_of"),
            True,
            False,
            id="parent relation",
        ),
        pytest.param(
            _edit_row("synonyms.tsv", "term_id", "BACC:0000004", synonym_type="related"),
            True,
            False,
            id="synonym",
        ),
        pytest.param(
            _edit_row(
                "obsolete_terms.tsv",
                "term_id",
                "BACC:0000005",
                replaced_by="BACC:0000001;BACC:0000002",
            ),
            True,
            False,
            id="obsolete term",
        ),
        pytest.param(
            _edit_row("relations.tsv", "subject_id", "BACC:0000004", object_id="BACC:0000002"),
            True,
            False,
            id="relation",
        ),
        pytest.param(_edit_facet_meaning, True, False, id="facet"),
        pytest.param(
            _edit_row(
                "mappings.sssom.tsv", "object_id", "ENVO:00000001", predicate_id="skos:broadMatch"
            ),
            False,
            True,
            id="mapping",
        ),
    ],
)
def test_fingerprints_change_only_with_their_semantic_content(
    ontology_directory: Path, edit, vocabulary_changes: bool, mapping_set_changes: bool
) -> None:
    original = IsolationSourceOntology.load(ontology_directory)
    edit(ontology_directory)
    edited = IsolationSourceOntology.load(ontology_directory)

    assert (original.vocabulary_fingerprint != edited.vocabulary_fingerprint) is vocabulary_changes
    assert (
        original.mapping_set_fingerprint != edited.mapping_set_fingerprint
    ) is mapping_set_changes
