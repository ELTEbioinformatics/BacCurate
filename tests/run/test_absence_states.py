"""Protect the distinct absence states serialized in the standardized dataset."""

from __future__ import annotations

import csv
import shutil
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import date
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from instructor.core import InstructorRetryException
from pydantic import ValidationError

from baccurate.adapters.llm.client import LLMSettings
from baccurate.extraction import (
    SEQUENCE_ACCESSION_COLUMNS,
    SelectionDecision,
    SelectionEvent,
    SelectionPolicy,
)
from baccurate.provenance.source_snapshot import (
    DerivedBundleProvenance,
    SourceSnapshotManifest,
    provenance_path_for,
    sha256_file,
)
from baccurate.run.dataset_builder import (
    DatasetBuilder,
    DatasetBuildRequest,
)
from baccurate.run.statistics import DatasetBuildStatistics, InventedLabelStatistics
from baccurate.standardization.host import HostPolicy, HostStandardizer
from baccurate.standardization.host_lineage import HostLineage
from baccurate.standardization.isolation_source import (
    IsolationSourceDiagnostic,
    IsolationSourcePromptPolicy,
    IsolationSourceStandardizer,
)
from baccurate.standardization.isolation_source_ontology import IsolationSourceOntology
from baccurate.standardization.location import (
    LocationDiagnostic,
    LocationPolicy,
    LocationResolutionRoute,
    LocationStandardizer,
)
from baccurate.standardization_target.specifications import (
    LOCATION_NAME_MATCH,
    StandardizationTarget,
)
from baccurate.taxon_registry.registry import load_taxon_registry

ROOT = Path(__file__).parents[2]
SELECTION_POLICY_PATH = ROOT / "config" / "selection.yaml"
LOCATION_POLICY_PATH = ROOT / "config" / "location.yaml"
ISOLATION_SOURCE_POLICY_PATH = ROOT / "config" / "isolation_source.yaml"

DATE_COLUMNS = (
    "date_category",
    "date_structure",
    "date_precision",
    "date_start",
    "date_end",
    "date_diagnostics",
    "date_attr_orig",
    "date_val_orig",
)
LOCATION_COLUMNS = (
    "loc_attr_orig",
    "loc_val_orig",
    "loc_selected_pair",
    "loc_resolution",
    "loc_country",
    "loc_un_region",
    "loc_sublocation",
    "loc_latitude",
    "loc_longitude",
    "loc_diagnostics",
)
LOCATION_ANSWER_COLUMNS = LOCATION_COLUMNS[2:-1]
ISOLATION_SOURCE_COLUMNS = (
    "iso_attr_orig",
    "iso_val_orig",
    "iso_source_type",
    "iso_body_product",
    "iso_body_site",
    "iso_lesion",
    "iso_environmental_material",
    "iso_facility",
    "iso_sampled_object",
    "iso_food_type",
    "iso_term_ids",
)
HOST_COLUMNS = (
    "host_attr_orig",
    "host_val_orig",
    "host_taxid",
    "host_sci_name",
    "host_common_names",
    "host_lineage_names",
    "host_lineage_taxids",
    "host_match_quality_score",
    "host_needs_review",
)
EXTRACTED_COLUMNS = (
    "accession",
    "taxon_key",
    "ncbi_organism",
    "sylph_species",
    "bioproject_accession",
    *SEQUENCE_ACCESSION_COLUMNS,
    "biosample_last_update",
    "date_attr_orig",
    "date_val_orig",
    "date_category",
    "loc_matched_by",
    "loc_attr_orig",
    "loc_val_orig",
    "iso_attr_orig",
    "iso_val_orig",
    "host_attr_orig",
    "host_val_orig",
)


@dataclass(frozen=True, slots=True)
class _BuiltDataset:
    columns: tuple[str, ...]
    records: tuple[dict[str, str], ...]
    statistics: DatasetBuildStatistics
    content: bytes


@pytest.fixture(scope="module")
def selection_policy() -> SelectionPolicy:
    return SelectionPolicy.load(SELECTION_POLICY_PATH)


def _write_source_manifest(
    path: Path,
    *,
    snapshot_id: str,
    sha256_fill_character: str,
) -> SourceSnapshotManifest:
    manifest = SourceSnapshotManifest(
        snapshot_id=snapshot_id,
        provider="absence-characterization",
        retrieved_on=date(2026, 1, 1),
        file={
            "name": f"{snapshot_id}.xml.gz",
            "sha256": sha256_fill_character * 64,
        },
    )
    path.write_text(
        yaml.safe_dump(manifest.model_dump(mode="json"), sort_keys=False),
        encoding="utf-8",
    )
    return manifest


def _write_extracted_bundle(
    tmp_path: Path,
    rows: Sequence[Mapping[str, str]],
) -> tuple[Path, Path, Path]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    extracted_path = tmp_path / "extracted.tsv"
    with extracted_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=EXTRACTED_COLUMNS,
            delimiter="\t",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)

    biosample_manifest_path = tmp_path / "biosample-snapshot.yaml"
    bioproject_manifest_path = tmp_path / "bioproject-snapshot.yaml"
    biosample_manifest = _write_source_manifest(
        biosample_manifest_path,
        snapshot_id="biosample-absence-characterization",
        sha256_fill_character="0",
    )
    bioproject_manifest = _write_source_manifest(
        bioproject_manifest_path,
        snapshot_id="bioproject-absence-characterization",
        sha256_fill_character="1",
    )
    DerivedBundleProvenance(
        biosample_snapshot_id=biosample_manifest.snapshot_id,
        biosample_manifest_sha256=sha256_file(biosample_manifest_path),
        bioproject_snapshot_id=bioproject_manifest.snapshot_id,
        bioproject_manifest_sha256=sha256_file(bioproject_manifest_path),
        extracted_metadata_sha256=sha256_file(extracted_path),
    ).write(provenance_path_for(extracted_path))
    return extracted_path, biosample_manifest_path, bioproject_manifest_path


def _build_dataset(
    tmp_path: Path,
    rows: Sequence[Mapping[str, str]],
    targets: tuple[StandardizationTarget, ...],
    *,
    location_policy: LocationPolicy | None = None,
    host_policy: HostPolicy | None = None,
    isolation_source_policy: IsolationSourcePromptPolicy | None = None,
    location_standardizer_factory: Callable[..., LocationStandardizer] | None = None,
    host_standardizer_factory: Callable[..., HostStandardizer] | None = None,
    isolation_source_standardizer_factory: Callable[..., IsolationSourceStandardizer] | None = None,
) -> _BuiltDataset:
    extracted_path, biosample_manifest, bioproject_manifest = _write_extracted_bundle(
        tmp_path,
        rows,
    )
    destination = tmp_path / "final.tsv"
    statistics = DatasetBuilder(
        location_standardizer_factory=location_standardizer_factory,
        host_standardizer_factory=host_standardizer_factory,
        host_lineage_factory=lambda *_paths: SimpleNamespace(
            enrich=lambda _taxid: HostLineage("", "", ""),
            is_descendant_or_self=lambda _taxid, _ancestor: False,
        ),
        isolation_source_standardizer_factory=isolation_source_standardizer_factory,
    ).build(
        DatasetBuildRequest(
            extracted_metadata=extracted_path,
            biosample_snapshot_manifest=biosample_manifest,
            bioproject_snapshot_manifest=bioproject_manifest,
            requested_taxa=("ecoli",),
            requested_targets=targets,
            final_destination=destination,
            taxon_registry=load_taxon_registry(),
            host_policy=host_policy,
            location_policy=location_policy,
            isolation_source_prompt_policy=isolation_source_policy,
            llm_settings=LLMSettings(None, None, "test-model"),
            disable_progress=True,
        )
    )
    with destination.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        records = tuple(dict(record) for record in reader)
        columns = tuple(reader.fieldnames or ())
    return _BuiltDataset(columns, records, statistics, destination.read_bytes())


def _dated_record(accession: str, **metadata: str) -> dict[str, str]:
    return {
        "accession": accession,
        "taxon_key": "ecoli",
        "bioproject_accession": "",
        "date_attr_orig": "collection_date",
        "date_val_orig": "2020-01-02",
        "date_category": "c",
        "loc_matched_by": _name_match_flags(metadata.get("loc_attr_orig", "")),
        **metadata,
    }


def _name_match_flags(loc_attr_orig: str) -> str:
    """Flag every location pair as matched by its attribute name."""
    if not loc_attr_orig:
        return ""
    return "||".join(LOCATION_NAME_MATCH for _ in loc_attr_orig.split("||"))


def test_ncbi_organism_and_sylph_species_pass_through_from_extracted_metadata(
    tmp_path: Path,
) -> None:
    built = _build_dataset(
        tmp_path,
        [
            _dated_record(
                "AGREEING",
                ncbi_organism="Escherichia coli",
                sylph_species="Escherichia coli",
            ),
            _dated_record(
                "REASSIGNED",
                ncbi_organism="Klebsiella pneumoniae",
                sylph_species="Escherichia coli",
            ),
        ],
        (StandardizationTarget.DATE,),
    )

    assert [record["ncbi_organism"] for record in built.records] == [
        "Escherichia coli",
        "Klebsiella pneumoniae",
    ]
    assert [record["sylph_species"] for record in built.records] == [
        "Escherichia coli",
        "Escherichia coli",
    ]


def test_sequence_accessions_pass_through_from_extracted_metadata(tmp_path: Path) -> None:
    built = _build_dataset(
        tmp_path,
        [
            _dated_record(
                "LINKED", sra_run_accessions="SRR1||SRR2", refseq_assembly_accessions="GCF_1.1"
            ),
            _dated_record("UNLINKED"),
        ],
        (StandardizationTarget.DATE,),
    )

    linked, unlinked = built.records
    assert linked["sra_run_accessions"] == "SRR1||SRR2"
    assert linked["genbank_assembly_accessions"] == ""
    assert linked["refseq_assembly_accessions"] == "GCF_1.1"
    assert unlinked["sra_run_accessions"] == ""


def _location_policy(_tmp_path: Path) -> LocationPolicy:
    return LocationPolicy.load(LOCATION_POLICY_PATH)


def _location_standardizer(policy: LocationPolicy, _logger: object) -> LocationStandardizer:
    return LocationStandardizer(policy)


def _minimal_host_components(tmp_path: Path) -> tuple[HostPolicy, Path]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    taxonomy_path = tmp_path / "taxonomy.tsv"
    taxonomy_path.write_text(
        "taxid\trank\tscientific_name\tsynonym\tgenbank_common_name\tcommon_name\tcomments\n"
        "9606\tspecies\tHomo sapiens\t\t\t\t\n",
        encoding="utf-8",
    )
    return (
        HostPolicy(
            schema_version=3,
            ignored_substrings=(),
            isolation_source_keywords=(),
            curated_taxa=(),
            value_rejection_entries=(),
            value_rejections=(),
        ),
        taxonomy_path,
    )


def test_published_fallback_date_projects_resolved_columns_and_equal_bounds_evidence(
    tmp_path: Path,
) -> None:
    built = _build_dataset(
        tmp_path,
        [
            _dated_record(
                "FALLBACK_DATE",
                date_attr_orig="submission_date||publication_date",
                date_val_orig="2019||2019",
                date_category="f||f",
            )
        ],
        (StandardizationTarget.DATE,),
    )

    assert tuple(built.records[0][column] for column in DATE_COLUMNS) == (
        "fallback",
        "single_value",
        "year",
        "2019-01-01",
        "2019-12-31",
        "",
        "submission_date||publication_date",
        "2019||2019",
    )


def _isolation_source_policy(tmp_path: Path) -> IsolationSourcePromptPolicy:
    tmp_path.mkdir(parents=True, exist_ok=True)
    return replace(
        IsolationSourcePromptPolicy.load(
            ROOT / "tests" / "fixtures" / "standardization" / "isolation_source.yaml"
        ),
        cache_db_path=tmp_path / "isolation-source-cache.db",
    )


def _build_isolation_source_dataset(
    tmp_path: Path,
    rows: Sequence[Mapping[str, str]],
) -> _BuiltDataset:
    host_policy, taxonomy_path = _minimal_host_components(tmp_path)
    isolation_source_policy = _isolation_source_policy(tmp_path)
    return _build_dataset(
        tmp_path,
        rows,
        (StandardizationTarget.DATE, StandardizationTarget.ISOLATION_SOURCE),
        host_policy=host_policy,
        isolation_source_policy=isolation_source_policy,
        host_standardizer_factory=lambda policy, _logger: HostStandardizer(
            policy,
            taxonomy_path,
        ),
        isolation_source_standardizer_factory=(
            lambda policy, logger: IsolationSourceStandardizer(
                policy,
                client=None,
                llm_settings=LLMSettings(None, None, "test-model"),
                result_logger=logger,
            )
        ),
    )


def test_isolation_source_outcome_projects_eleven_columns_in_facet_order(
    tmp_path: Path,
) -> None:
    host_policy, taxonomy_path = _minimal_host_components(tmp_path)
    isolation_source_policy = replace(
        IsolationSourcePromptPolicy.load(ISOLATION_SOURCE_POLICY_PATH),
        cache_db_path=tmp_path / "faceted-isolation-source-cache.db",
    )
    built = _build_dataset(
        tmp_path,
        [
            _dated_record(
                "FACETED_SOURCE",
                iso_attr_orig="material||site||lesion",
                iso_val_orig="pus||liver||abscess",
            )
        ],
        (StandardizationTarget.DATE, StandardizationTarget.ISOLATION_SOURCE),
        host_policy=host_policy,
        isolation_source_policy=isolation_source_policy,
        host_standardizer_factory=lambda policy, _logger: HostStandardizer(
            policy,
            taxonomy_path,
        ),
        isolation_source_standardizer_factory=(
            lambda policy, logger: IsolationSourceStandardizer(
                policy,
                client=None,
                llm_settings=LLMSettings(None, None, "test-model"),
                result_logger=logger,
            )
        ),
    )

    assert tuple(column for column in built.columns if column.startswith("iso_")) == (
        "iso_attr_orig",
        "iso_val_orig",
        "iso_source_type",
        "iso_body_product",
        "iso_body_site",
        "iso_lesion",
        "iso_environmental_material",
        "iso_facility",
        "iso_sampled_object",
        "iso_food_type",
        "iso_term_ids",
    )
    assert tuple(built.records[0][column] for column in ISOLATION_SOURCE_COLUMNS) == (
        "material||site||lesion",
        "pus||liver||abscess",
        "host-associated||animal host",
        "body fluid||pus",
        "liver",
        "abscess",
        "NA",
        "NA",
        "NA",
        "NA",
        ("BACC:0000001||BACC:0000002||BACC:0000009||BACC:0000017||BACC:0000057||BACC:0000063"),
    )
    facet_labels = tuple(
        label
        for column in ISOLATION_SOURCE_COLUMNS[2:-1]
        for label in built.records[0][column].split("||")
        if label != "NA"
    )
    term_id_by_label = {
        term.label: term.term_id for term in isolation_source_policy.ontology.terms.values()
    }
    assert tuple(term_id_by_label[label] for label in facet_labels) == tuple(
        built.records[0]["iso_term_ids"].split("||")
    )


def test_equivalent_vocabulary_order_produces_byte_stable_dataset(tmp_path: Path) -> None:
    source_ontology = ROOT / "tests" / "fixtures" / "standardization" / "ontology"
    built_datasets: list[_BuiltDataset] = []

    for name, reverse_order in (("ordered_a", False), ("ordered_b", True)):
        build_root = tmp_path / name
        ontology_directory = build_root / "ontology"
        shutil.copytree(source_ontology, ontology_directory)
        if reverse_order:
            terms_path = ontology_directory / "terms.tsv"
            lines = terms_path.read_text(encoding="utf-8").splitlines()
            terms_path.write_text(
                "\n".join((lines[0], *reversed(lines[1:]))) + "\n",
                encoding="utf-8",
            )
            facets_path = ontology_directory / "facets.yaml"
            facet_document = yaml.safe_load(facets_path.read_text(encoding="utf-8"))
            facet_document["facets"] = dict(reversed(tuple(facet_document["facets"].items())))
            facets_path.write_text(
                yaml.safe_dump(facet_document, sort_keys=False),
                encoding="utf-8",
            )

        host_policy, taxonomy_path = _minimal_host_components(build_root)
        isolation_source_policy = replace(
            _isolation_source_policy(build_root),
            ontology_directory=ontology_directory,
            ontology=IsolationSourceOntology.load(ontology_directory),
        )
        built_datasets.append(
            _build_dataset(
                build_root,
                [
                    _dated_record(
                        "DETERMINISTIC_ENRICHMENT",
                        iso_attr_orig="specimen||isolation_source||anatomical_site",
                        iso_val_orig="blood||stool||rectal swab",
                    )
                ],
                (StandardizationTarget.DATE, StandardizationTarget.ISOLATION_SOURCE),
                host_policy=host_policy,
                isolation_source_policy=isolation_source_policy,
                host_standardizer_factory=lambda policy, _logger, taxonomy_path=taxonomy_path: (
                    HostStandardizer(
                        policy,
                        taxonomy_path,
                    )
                ),
                isolation_source_standardizer_factory=(
                    lambda policy, logger: IsolationSourceStandardizer(
                        policy,
                        client=None,
                        llm_settings=LLMSettings(None, None, "test-model"),
                        result_logger=logger,
                    )
                ),
            )
        )

    assert built_datasets[0].content == built_datasets[1].content


def test_failed_isolation_source_classification_skips_record_and_continues(
    tmp_path: Path,
) -> None:
    calls: list[dict[str, object]] = []

    def create(**kwargs: object) -> object:
        calls.append(kwargs)
        response_model = kwargs["response_model"]
        invalid_answer = {
            "reasoning": "The sample names an unsupported anatomical site.",
            "evidence_level": "sample",
            "source_type": ["animal host"],
            "body_product": [],
            "body_site": ["nasal cavity"],
            "lesion": [],
            "environmental_material": [],
            "facility": [],
            "sampled_object": [],
            "food_type": [],
        }
        if len(calls) == 1:
            for _ in range(3):
                with pytest.raises(ValidationError):
                    response_model.model_validate(invalid_answer)
            raise InstructorRetryException(
                "invalid structured response",
                n_attempts=4,
                total_usage=0,
            )
        with pytest.raises(ValidationError):
            response_model.model_validate(invalid_answer)
        return response_model.model_validate(
            {
                "reasoning": "The second response is valid.",
                "evidence_level": "sample",
                "source_type": ["environmental"],
                "body_product": [],
                "body_site": [],
                "lesion": [],
                "environmental_material": [],
                "facility": [],
                "sampled_object": [],
                "food_type": [],
            }
        )

    client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create)),
        close=lambda: None,
    )
    host_policy, taxonomy_path = _minimal_host_components(tmp_path)
    isolation_source_policy = _isolation_source_policy(tmp_path)

    def isolation_source_factory(
        policy: IsolationSourcePromptPolicy,
        logger: object,
    ) -> IsolationSourceStandardizer:
        standardizer = IsolationSourceStandardizer(
            policy,
            client=None,
            llm_settings=LLMSettings(None, None, "test-model"),
            result_logger=logger,
        )
        standardizer.pipeline.client = client
        return standardizer

    built = _build_dataset(
        tmp_path,
        [
            _dated_record(
                "FAILED_CLASSIFICATION",
                iso_attr_orig="isolation_source",
                iso_val_orig="unresolved material",
            ),
            _dated_record(
                "RECOVERED_CLASSIFICATION",
                iso_attr_orig="isolation_source",
                iso_val_orig="unresolved material",
            ),
        ],
        (StandardizationTarget.DATE, StandardizationTarget.ISOLATION_SOURCE),
        host_policy=host_policy,
        isolation_source_policy=isolation_source_policy,
        host_standardizer_factory=lambda policy, _logger: HostStandardizer(
            policy,
            taxonomy_path,
        ),
        isolation_source_standardizer_factory=isolation_source_factory,
    )

    records = {record["accession"]: record for record in built.records}
    assert tuple(records) == ("FAILED_CLASSIFICATION", "RECOVERED_CLASSIFICATION")
    assert tuple(
        records["FAILED_CLASSIFICATION"][column] for column in ISOLATION_SOURCE_COLUMNS
    ) == ("",) * len(ISOLATION_SOURCE_COLUMNS)
    assert records["RECOVERED_CLASSIFICATION"]["iso_source_type"] == "environmental"
    assert records["RECOVERED_CLASSIFICATION"]["iso_term_ids"] == "BACC:0000004"
    assert [call["max_retries"] for call in calls] == [3, 3]
    assert built.statistics.isolation_source is not None
    assert built.statistics.isolation_source.aggregate.rejected == 1
    assert built.statistics.isolation_source.aggregate.standardized == 1
    assert built.statistics.isolation_source.aggregate.diagnostics == {
        IsolationSourceDiagnostic.CLASSIFICATION_FAILURE: 1,
        IsolationSourceDiagnostic.LLM_CALL: 1,
    }
    assert built.statistics.isolation_source.aggregate.invented_labels == {
        "body_site": {
            "nasal cavity": InventedLabelStatistics(
                occurrences=4,
                accessions={
                    "FAILED_CLASSIFICATION": 3,
                    "RECOVERED_CLASSIFICATION": 1,
                },
            )
        }
    }


def test_unrequested_targets_omit_columns_instead_of_serializing_empty(
    tmp_path: Path,
) -> None:
    built = _build_dataset(
        tmp_path,
        [_dated_record("DATE_ONLY")],
        (StandardizationTarget.DATE,),
    )

    assert built.columns == (
        "accession",
        "taxon",
        "ncbi_organism",
        "sylph_species",
        "bioproject",
        *SEQUENCE_ACCESSION_COLUMNS,
        *DATE_COLUMNS,
    )
    unrequested_columns = LOCATION_COLUMNS + ISOLATION_SOURCE_COLUMNS + HOST_COLUMNS
    assert not (set(unrequested_columns) & set(built.columns))


def test_absent_bioproject_link_serializes_as_empty_not_na(tmp_path: Path) -> None:
    built = _build_dataset(
        tmp_path,
        [_dated_record("NO_BIOPROJECT")],
        (StandardizationTarget.DATE,),
    )

    assert built.records[0]["bioproject"] == ""


def test_absent_sublocation_serializes_as_na_not_empty(tmp_path: Path) -> None:
    built = _build_dataset(
        tmp_path,
        [
            _dated_record(
                "NO_SUBLOCATION",
                loc_attr_orig="geo_loc_name",
                loc_val_orig="Germany",
            ),
            _dated_record(
                "WITH_SUBLOCATION",
                loc_attr_orig="geo_loc_name",
                loc_val_orig="Germany: Berlin",
            ),
        ],
        (StandardizationTarget.DATE, StandardizationTarget.LOCATION),
        location_policy=_location_policy(tmp_path),
        location_standardizer_factory=_location_standardizer,
    )
    records = {record["accession"]: record for record in built.records}

    assert records["NO_SUBLOCATION"]["loc_sublocation"] == "NA"
    assert records["WITH_SUBLOCATION"]["loc_sublocation"] == "Berlin"


def test_absent_un_region_serializes_as_na_not_empty_for_resolved_country(
    tmp_path: Path,
) -> None:
    """A reviewed mapping may name a water body, which has no UN region."""
    built = _build_dataset(
        tmp_path,
        [
            _dated_record(
                "NO_UN_REGION",
                loc_attr_orig="geo_loc_name",
                loc_val_orig="Baltic Sea",
            )
        ],
        (StandardizationTarget.DATE, StandardizationTarget.LOCATION),
        location_policy=_location_policy(tmp_path),
        location_standardizer_factory=_location_standardizer,
    )

    assert built.records[0]["loc_country"] == "Baltic Sea"
    assert built.records[0]["loc_un_region"] == "NA"


def test_published_resolution_route_names_the_step_that_produced_the_country(
    tmp_path: Path,
) -> None:
    built = _build_dataset(
        tmp_path,
        [
            _dated_record("INSDC", loc_attr_orig="geo_loc_name", loc_val_orig="Germany"),
            _dated_record(
                "CONVERTED",
                loc_attr_orig="geo_loc_name",
                loc_val_orig="United States of America",
            ),
            _dated_record("COORDINATE", loc_attr_orig="lat_lon", loc_val_orig="52.52, 13.405"),
        ],
        (StandardizationTarget.DATE, StandardizationTarget.LOCATION),
        location_policy=_location_policy(tmp_path),
        location_standardizer_factory=_location_standardizer,
    )

    assert {record["accession"]: record["loc_resolution"] for record in built.records} == {
        "INSDC": "insdc_term",
        "CONVERTED": "country_conversion",
        "COORDINATE": "coordinate",
    }
    assert built.statistics.location is not None
    assert built.statistics.location.aggregate.resolution_routes == {
        LocationResolutionRoute.COORDINATE: 1,
        LocationResolutionRoute.COUNTRY_CONVERSION: 1,
        LocationResolutionRoute.INSDC_TERM: 1,
    }


def test_published_coordinate_is_filled_whenever_a_value_parses(tmp_path: Path) -> None:
    built = _build_dataset(
        tmp_path,
        [
            _dated_record(
                "COORDINATE",
                loc_attr_orig="lat_lon",
                loc_val_orig="6°12'52\"S 106°50'42\"E",
            ),
            _dated_record("NO_LOCATION", loc_attr_orig="lat_lon", loc_val_orig="0.0, 0.0"),
            _dated_record("INSDC", loc_attr_orig="geo_loc_name", loc_val_orig="Germany"),
        ],
        (StandardizationTarget.DATE, StandardizationTarget.LOCATION),
        location_policy=_location_policy(tmp_path),
        location_standardizer_factory=_location_standardizer,
    )

    published = {
        record["accession"]: (record["loc_latitude"], record["loc_longitude"])
        for record in built.records
    }
    assert published == {
        "COORDINATE": ("-6.21444", "106.845"),
        "NO_LOCATION": ("0", "0"),
        "INSDC": ("", ""),
    }


def test_published_selected_pair_positions_index_the_published_pairs(tmp_path: Path) -> None:
    built = _build_dataset(
        tmp_path,
        [
            _dated_record(
                "EMPTY_FIRST_VALUE",
                loc_attr_orig="collection_site||geo_loc_name",
                loc_val_orig="||Germany",
            )
        ],
        (StandardizationTarget.DATE, StandardizationTarget.LOCATION),
        location_policy=_location_policy(tmp_path),
        location_standardizer_factory=_location_standardizer,
    )

    record = built.records[0]
    assert record["loc_val_orig"] == "||Germany"
    assert record["loc_selected_pair"] == "2"


def test_rejected_locations_publish_evidence_and_diagnostics_with_empty_answer_columns(
    tmp_path: Path,
) -> None:
    built = _build_dataset(
        tmp_path,
        [
            _dated_record(
                "UNRESOLVED",
                loc_attr_orig="lat_lon",
                loc_val_orig="999, 999",
            ),
            _dated_record("ABSENT", loc_attr_orig="", loc_val_orig=""),
            _dated_record(
                "OUTSIDE_INSDC",
                loc_attr_orig="geo_loc_name",
                loc_val_orig="Vatican",
            ),
        ],
        (StandardizationTarget.DATE, StandardizationTarget.LOCATION),
        location_policy=_location_policy(tmp_path),
        location_standardizer_factory=_location_standardizer,
    )

    assert len(built.records) == 3
    records = {record["accession"]: record for record in built.records}

    for record in built.records:
        assert tuple(record[column] for column in LOCATION_ANSWER_COLUMNS) == ("",) * len(
            LOCATION_ANSWER_COLUMNS
        )
        assert record["date_start"] == "2020-01-02"

    # A rejected record keeps its selected pairs and its diagnostics.
    assert records["UNRESOLVED"]["loc_attr_orig"] == "lat_lon"
    assert records["UNRESOLVED"]["loc_val_orig"] == "999, 999"
    assert records["UNRESOLVED"]["loc_diagnostics"] == "unresolved_place"

    assert records["ABSENT"]["loc_attr_orig"] == ""
    assert records["ABSENT"]["loc_val_orig"] == ""
    assert records["ABSENT"]["loc_diagnostics"] == "absent_values"

    assert records["OUTSIDE_INSDC"]["loc_attr_orig"] == "geo_loc_name"
    assert records["OUTSIDE_INSDC"]["loc_val_orig"] == "Vatican"
    assert records["OUTSIDE_INSDC"]["loc_diagnostics"] == "unmappable_result||unresolved_place"

    assert built.statistics.location is not None
    assert built.statistics.location.aggregate.rejected == 3
    assert built.statistics.location.aggregate.diagnostics == {
        LocationDiagnostic.ABSENT_VALUES: 1,
        LocationDiagnostic.UNMAPPABLE_RESULT: 1,
        LocationDiagnostic.UNRESOLVED_PLACE: 2,
    }


def test_standardized_location_publishes_its_diagnostics_in_the_same_column(
    tmp_path: Path,
) -> None:
    built = _build_dataset(
        tmp_path,
        [
            _dated_record(
                "STANDARDIZED_WITH_CONFLICT",
                loc_attr_orig="geo_loc_name||collection_site",
                loc_val_orig="Germany||unreviewed site 8841",
            )
        ],
        (StandardizationTarget.DATE, StandardizationTarget.LOCATION),
        location_policy=_location_policy(tmp_path),
        location_standardizer_factory=_location_standardizer,
    )

    record = built.records[0]
    assert record["loc_country"] == "Germany"
    assert record["loc_diagnostics"] == "unresolved_place"


def test_standardized_location_reports_a_pair_resolved_outside_the_insdc_list(
    tmp_path: Path,
) -> None:
    # The coordinate lies inside the Vatican map unit. The Vatican is not a valid
    # INSDC location, so standardization uses `geo_loc_name` alone.
    built = _build_dataset(
        tmp_path,
        [
            _dated_record(
                "STANDARDIZED_WITH_UNMAPPABLE_PAIR",
                loc_attr_orig="geo_loc_name||lat_lon",
                loc_val_orig="Italy: Milan||41.9038 N 12.4536 E",
            )
        ],
        (StandardizationTarget.DATE, StandardizationTarget.LOCATION),
        location_policy=_location_policy(tmp_path),
        location_standardizer_factory=_location_standardizer,
    )

    record = built.records[0]
    assert record["loc_selected_pair"] == "1"
    assert record["loc_resolution"] == "insdc_term"
    assert (record["loc_country"], record["loc_sublocation"]) == ("Italy", "Milan")
    assert (record["loc_latitude"], record["loc_longitude"]) == ("41.9038", "12.4536")
    assert record["loc_diagnostics"] == "unmappable_result||unresolved_place"


def test_unfilled_isolation_source_facets_serialize_as_na_not_empty(
    tmp_path: Path,
) -> None:
    built = _build_isolation_source_dataset(
        tmp_path,
        [
            _dated_record(
                "NO_IDENTIFIER",
                iso_attr_orig="isolation_source",
                iso_val_orig="environmental",
            )
        ],
    )

    assert built.records[0]["iso_source_type"] == "environmental"
    assert (
        tuple(built.records[0][column] for column in ISOLATION_SOURCE_COLUMNS[3:-1]) == ("NA",) * 7
    )
    assert built.records[0]["iso_term_ids"] == "BACC:0000004"


def test_classified_source_with_no_applicable_facet_serializes_as_na(
    tmp_path: Path,
) -> None:
    built = _build_isolation_source_dataset(
        tmp_path,
        [
            _dated_record(
                "EMPTY_FACETS",
                iso_attr_orig="isolation_source",
                iso_val_orig="unmapped submitted material",
            )
        ],
    )

    assert (
        tuple(built.records[0][column] for column in ISOLATION_SOURCE_COLUMNS[2:-1]) == ("NA",) * 8
    )
    assert built.records[0]["iso_term_ids"] == ""
    assert built.statistics.isolation_source is not None
    assert built.statistics.isolation_source.aggregate.diagnostics == {
        IsolationSourceDiagnostic.UNSPECIFIED: 1,
    }


def test_host_absence_and_non_resolution_share_empty_columns_but_run_report_separates_them(
    tmp_path: Path,
) -> None:
    host_policy, taxonomy_path = _minimal_host_components(tmp_path)
    built = _build_dataset(
        tmp_path,
        [
            _dated_record(
                "HOST_ABSENT",
                host_attr_orig="",
                host_val_orig="",
            ),
            _dated_record(
                "HOST_UNRESOLVED",
                host_attr_orig="host",
                host_val_orig="zzzxxyy submitted host",
            ),
        ],
        (StandardizationTarget.DATE, StandardizationTarget.HOST),
        host_policy=host_policy,
        host_standardizer_factory=lambda policy, _logger: HostStandardizer(
            policy,
            taxonomy_path,
        ),
    )

    assert len(built.records) == 2
    assert all(
        tuple(record[column] for column in HOST_COLUMNS) == ("",) * 9 for record in built.records
    )
    assert built.statistics.host is not None
    assert built.statistics.host.aggregate.rejected == 1
    assert built.statistics.host.aggregate.overflow == 1


def test_missing_metadata_is_rejected_before_selection_not_serialized_as_absence(
    selection_policy: SelectionPolicy,
) -> None:
    decision = selection_policy.evaluate(attribute="host", value="unknown")

    assert decision == SelectionDecision(
        "host",
        "unknown",
        (),
        (SelectionEvent("rejected_value", "host", "universal_missing", "host", "unknown"),),
    )
