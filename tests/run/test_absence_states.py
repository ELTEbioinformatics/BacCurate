"""Protect the distinct absence states serialized in the standardized dataset."""

from __future__ import annotations

import csv
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import date
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from baccurate.adapters.llm.client import LLMSettings
from baccurate.extraction import CurationDecision, CurationEvent, CurationSchema
from baccurate.pathogen_registry.registry import load_pathogen_registry
from baccurate.provenance.source_snapshot import (
    ArtifactReference,
    DerivedArtifactReferences,
    DerivedBundleProvenance,
    ManifestReference,
    PairedManifestReferences,
    SourceSnapshotManifest,
    bioproject_catalog_path_for,
    provenance_path_for,
    sha256_file,
)
from baccurate.run.dataset_builder import (
    DatasetBuilder,
    DatasetBuildRequest,
)
from baccurate.run.statistics import DatasetBuildStatistics
from baccurate.standardization.host import HostPolicy, HostStandardizer
from baccurate.standardization.host_lineage import HostLineage
from baccurate.standardization.isolation_source import (
    IsolationSourcePromptPolicy,
    IsolationSourceStandardizer,
)
from baccurate.standardization.location import (
    LocationDiagnostic,
    LocationPolicy,
    LocationStandardizer,
)
from baccurate.standardization_target.specifications import StandardizationTarget

ROOT = Path(__file__).parents[2]
CURATION_SCHEMA_PATH = ROOT / "config" / "curation_schema.yaml"
LOCATION_POLICY_PATH = ROOT / "config" / "location.yaml"
ISOLATION_SOURCE_POLICY_PATH = ROOT / "config" / "isolation_source.yaml"

DATE_COLUMNS = (
    "date_attr_orig",
    "date_val_orig",
    "date_start",
    "date_end",
    "date_reliability_score",
)
LOCATION_COLUMNS = (
    "loc_attr_orig",
    "loc_val_orig",
    "loc_UNregion",
    "loc_country",
    "loc_sublocation",
)
ISOLATION_SOURCE_COLUMNS = (
    "iso_attr_orig",
    "iso_val_orig",
    "iso_term_paths",
    "iso_display_terms",
    "iso_external_ontology_identifiers",
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
    "pathogen",
    "bioproject_id",
    "bioproject_accession",
    "date_attr_orig",
    "date_val_orig",
    "date_category",
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


@pytest.fixture(scope="module")
def curation_schema() -> CurationSchema:
    return CurationSchema.load(CURATION_SCHEMA_PATH)


def _write_source_manifest(
    path: Path,
    *,
    snapshot_id: str,
    sha256_fill_character: str,
) -> SourceSnapshotManifest:
    manifest = SourceSnapshotManifest(
        manifest_version=1,
        snapshot_id=snapshot_id,
        provider="absence-characterization",
        retrieved_on=date(2026, 1, 1),
        metadata_reference_date=date(2026, 1, 1),
        files=(
            {
                "name": f"{snapshot_id}.xml.gz",
                "sha256": sha256_fill_character * 64,
            },
        ),
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

    bioproject_catalog = bioproject_catalog_path_for(extracted_path)
    bioproject_catalog.write_text("", encoding="utf-8")
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
        bundle_version=1,
        source_manifests=PairedManifestReferences(
            biosample=ManifestReference(
                snapshot_id=biosample_manifest.snapshot_id,
                path=str(biosample_manifest_path),
                sha256=sha256_file(biosample_manifest_path),
            ),
            bioproject=ManifestReference(
                snapshot_id=bioproject_manifest.snapshot_id,
                path=str(bioproject_manifest_path),
                sha256=sha256_file(bioproject_manifest_path),
            ),
        ),
        artifacts=DerivedArtifactReferences(
            extracted_metadata=ArtifactReference(
                path=extracted_path.name,
                sha256=sha256_file(extracted_path),
            ),
            bioproject_context=ArtifactReference(
                path=bioproject_catalog.name,
                sha256=sha256_file(bioproject_catalog),
            ),
        ),
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
    atb_index = tmp_path / "atb.tsv"
    atb_index.write_text("accession\tpathogen_ATB\tin_ATB\n", encoding="utf-8")
    destination = tmp_path / "final.tsv"
    statistics = DatasetBuilder(
        location_standardizer_factory=location_standardizer_factory,
        host_standardizer_factory=host_standardizer_factory,
        host_lineage_factory=lambda *_paths: SimpleNamespace(
            enrich=lambda _taxid: HostLineage("", "", "")
        ),
        isolation_source_standardizer_factory=isolation_source_standardizer_factory,
    ).build(
        DatasetBuildRequest(
            extracted_metadata=extracted_path,
            biosample_snapshot_manifest=biosample_manifest,
            bioproject_snapshot_manifest=bioproject_manifest,
            requested_pathogens=("ecoli",),
            requested_targets=targets,
            final_destination=destination,
            atb_index=atb_index,
            pathogen_registry=load_pathogen_registry(),
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
    return _BuiltDataset(columns, records, statistics)


def _dated_record(accession: str, **metadata: str) -> dict[str, str]:
    return {
        "accession": accession,
        "pathogen": "ecoli",
        "bioproject_accession": "",
        "date_attr_orig": "collection_date",
        "date_val_orig": "2020-01-02",
        "date_category": "c",
        **metadata,
    }


def _location_policy(tmp_path: Path) -> LocationPolicy:
    return replace(
        LocationPolicy.load(LOCATION_POLICY_PATH),
        cache_db_path=tmp_path / "location-cache.db",
    )


def _location_without_llm(
    policy: LocationPolicy,
    _logger: object,
) -> LocationStandardizer:
    return LocationStandardizer(
        policy,
        client=None,
        llm_settings=LLMSettings(None, None, "test-model"),
    )


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


def _isolation_source_policy(tmp_path: Path) -> IsolationSourcePromptPolicy:
    tmp_path.mkdir(parents=True, exist_ok=True)
    ontology_path = tmp_path / "ontology.tsv"
    ontology_path.write_text(
        "term_path\tdisplay_term\texternal_ontology_identifier\t"
        "crosslink_targets\tsynonyms\tcomment\n"
        "environmental:identifierless\tidentifierless node\t\t\t\t\n",
        encoding="utf-8",
    )
    return replace(
        IsolationSourcePromptPolicy.load(ISOLATION_SOURCE_POLICY_PATH),
        ontology_tsv_path=ontology_path,
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
            lambda policy, bundle_path, logger: IsolationSourceStandardizer(
                policy,
                bundle_path,
                client=None,
                llm_settings=LLMSettings(None, None, "test-model"),
                result_logger=logger,
            )
        ),
    )


def test_requested_target_without_outcome_serializes_exact_empty_columns_without_shifting_others(
    tmp_path: Path,
) -> None:
    date_path = tmp_path / "date"
    date_policy = _location_policy(date_path)
    date_without_outcome = _build_dataset(
        date_path,
        [
            _dated_record(
                "DATE_WITHOUT_OUTCOME",
                date_val_orig="not a date",
                loc_attr_orig="geo_loc_name",
                loc_val_orig="Germany",
            )
        ],
        (StandardizationTarget.DATE, StandardizationTarget.LOCATION),
        location_policy=date_policy,
        location_standardizer_factory=_location_without_llm,
    )

    location_path = tmp_path / "location"
    location_without_outcome = _build_dataset(
        location_path,
        [_dated_record("LOCATION_WITHOUT_OUTCOME", loc_attr_orig="", loc_val_orig="")],
        (StandardizationTarget.DATE, StandardizationTarget.LOCATION),
        location_policy=_location_policy(location_path),
        location_standardizer_factory=_location_without_llm,
    )

    host_path = tmp_path / "host"
    host_policy, taxonomy_path = _minimal_host_components(host_path)
    host_without_outcome = _build_dataset(
        host_path,
        [_dated_record("HOST_WITHOUT_OUTCOME", host_attr_orig="", host_val_orig="")],
        (StandardizationTarget.DATE, StandardizationTarget.HOST),
        host_policy=host_policy,
        host_standardizer_factory=lambda policy, _logger: HostStandardizer(
            policy,
            taxonomy_path,
        ),
    )

    isolation_source_without_outcome = _build_isolation_source_dataset(
        tmp_path / "isolation-source",
        [_dated_record("ISOLATION_SOURCE_WITHOUT_OUTCOME")],
    )

    cases = (
        (date_without_outcome, DATE_COLUMNS, "loc_country", "Germany"),
        (location_without_outcome, LOCATION_COLUMNS, "date_start", "2020-01-02"),
        (host_without_outcome, HOST_COLUMNS, "date_start", "2020-01-02"),
        (
            isolation_source_without_outcome,
            ISOLATION_SOURCE_COLUMNS,
            "date_start",
            "2020-01-02",
        ),
    )
    for built, absent_columns, preserved_column, preserved_value in cases:
        assert len(built.records) == 1
        record = built.records[0]
        assert tuple(record[column] for column in absent_columns) == ("",) * len(absent_columns)
        assert record[preserved_column] == preserved_value


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
        "pathogen_scientific_name",
        "in_ATB",
        "bioproject",
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
        location_standardizer_factory=_location_without_llm,
    )
    records = {record["accession"]: record for record in built.records}

    assert records["NO_SUBLOCATION"]["loc_sublocation"] == "NA"
    assert records["WITH_SUBLOCATION"]["loc_sublocation"] == "Berlin"


def test_absent_un_region_serializes_as_na_not_empty_for_resolved_country(
    tmp_path: Path,
) -> None:
    response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content='{"country": "Arctic Ocean"}'))]
    )
    client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=lambda **_kwargs: response),
        ),
        close=lambda: None,
    )
    built = _build_dataset(
        tmp_path,
        [
            _dated_record(
                "NO_UN_REGION",
                loc_attr_orig="geo_loc_name",
                loc_val_orig="model-only place 739105",
            )
        ],
        (StandardizationTarget.DATE, StandardizationTarget.LOCATION),
        location_policy=_location_policy(tmp_path),
        location_standardizer_factory=lambda policy, _logger: LocationStandardizer(
            policy,
            client=client,
            llm_settings=LLMSettings(None, None, "test-model"),
        ),
    )

    assert built.records[0]["loc_country"] == "Arctic Ocean"
    assert built.records[0]["loc_UNregion"] == "NA"


def test_rejected_locations_serialize_as_empty_not_na_while_diagnostics_preserve_reasons(
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
        location_standardizer_factory=_location_without_llm,
    )

    assert len(built.records) == 3
    for record in built.records:
        assert tuple(record[column] for column in LOCATION_COLUMNS) == ("",) * 5
        assert record["loc_country"] != "NA"
    assert built.statistics.location is not None
    assert built.statistics.location.aggregate.rejected == 3
    assert built.statistics.location.aggregate.diagnostics == {
        LocationDiagnostic.ABSENT_VALUES: 1,
        LocationDiagnostic.UNMAPPABLE_RESULT: 1,
        LocationDiagnostic.UNRESOLVED_PLACE: 1,
    }


def test_absent_external_ontology_identifier_serializes_as_na_not_empty(
    tmp_path: Path,
) -> None:
    built = _build_isolation_source_dataset(
        tmp_path,
        [
            _dated_record(
                "NO_IDENTIFIER",
                iso_attr_orig="isolation_source",
                iso_val_orig="identifierless node",
            )
        ],
    )

    assert built.records[0]["iso_term_paths"] == "environmental:identifierless"
    assert built.records[0]["iso_external_ontology_identifiers"] == "NA"


def test_unspecified_isolation_source_serializes_as_explicit_term_not_empty(
    tmp_path: Path,
) -> None:
    built = _build_isolation_source_dataset(
        tmp_path,
        [
            _dated_record(
                "UNSPECIFIED",
                iso_attr_orig="isolation_source",
                iso_val_orig="unmapped submitted material",
            )
        ],
    )

    assert built.records[0]["iso_term_paths"] == "unspecified"
    assert built.records[0]["iso_display_terms"] == "unspecified"
    assert built.records[0]["iso_external_ontology_identifiers"] == "NA"


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
    curation_schema: CurationSchema,
) -> None:
    decision = curation_schema.evaluate(attribute="host", value="unknown")

    assert decision == CurationDecision(
        "host",
        "unknown",
        (),
        (CurationEvent("rejected_value", "host", "universal_missing", "host", "unknown"),),
    )
