"""End-to-end golden run covering every standardization target."""

from __future__ import annotations

import csv
import json
import logging
import shutil
from datetime import date
from pathlib import Path
from types import SimpleNamespace

import yaml
from pytest import MonkeyPatch

import baccurate.standardization.isolation_source as isolation_source_module
import baccurate.standardization.location as location_module
from baccurate.adapters.llm.client import LLMSettings
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
from baccurate.run.outputs import RunOutputs
from baccurate.run.report import RunContext, RunReport, RunStatus
from baccurate.run.statistics import DatasetBuildProgress
from baccurate.standardization.host import HostPolicy, HostStandardizer
from baccurate.standardization.host_lineage import HostLineage
from baccurate.standardization.isolation_source import (
    IsolationSourcePromptPolicy,
    IsolationSourceStandardizer,
)
from baccurate.standardization.location import LocationPolicy, LocationStandardizer
from baccurate.standardization_target.specifications import StandardizationTarget

STANDARDIZATION_TARGETS = tuple(StandardizationTarget)
PATHOGENS = ("ecoli", "abaumannii")
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


class _LocationCompletions:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content='{"country": "Germany"}'))]
        )


class _IsolationSourceCompletions:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return kwargs["response_model"].model_validate(
            {
                "reasoning": "The submitted lesion is wound material.",
                "evidence_level": "sample_and_project",
                "source_type": "animal host",
                "body_product": [],
                "body_site": [],
                "lesion": ["wound"],
                "environmental_material": [],
                "facility": [],
                "sampled_object": [],
                "food_type": [],
            }
        )


class _FakeLocationClient:
    def __init__(self) -> None:
        self.completions = _LocationCompletions()
        self.chat = SimpleNamespace(completions=self.completions)

    def close(self) -> None:
        pass


class _FakeIsolationSourceClient:
    def __init__(self) -> None:
        self.completions = _IsolationSourceCompletions()
        self.chat = SimpleNamespace(completions=self.completions)

    def close(self) -> None:
        pass


def _write_manifest(
    path: Path,
    *,
    snapshot_id: str,
    sha256_fill_character: str,
) -> SourceSnapshotManifest:
    manifest = SourceSnapshotManifest(
        manifest_version=1,
        snapshot_id=snapshot_id,
        provider="golden-run",
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


def _prepare_bundle(tmp_path: Path, golden_run_fixture_dir: Path) -> tuple[Path, Path, Path]:
    extracted = tmp_path / "extracted.tsv"
    shutil.copyfile(golden_run_fixture_dir / "extracted.tsv", extracted)
    catalog = bioproject_catalog_path_for(extracted)
    shutil.copyfile(golden_run_fixture_dir / "bioproject_context.jsonl", catalog)
    biosample_manifest_path = tmp_path / "biosample_snapshot.yaml"
    bioproject_manifest_path = tmp_path / "bioproject_snapshot.yaml"
    biosample_manifest = _write_manifest(
        biosample_manifest_path,
        snapshot_id="biosample-golden",
        sha256_fill_character="0",
    )
    bioproject_manifest = _write_manifest(
        bioproject_manifest_path,
        snapshot_id="bioproject-golden",
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
                path=extracted.name,
                sha256=sha256_file(extracted),
            ),
            bioproject_context=ArtifactReference(
                path=catalog.name,
                sha256=sha256_file(catalog),
            ),
        ),
    ).write(provenance_path_for(extracted))
    return extracted, biosample_manifest_path, bioproject_manifest_path


def _write_runtime_config(
    tmp_path: Path,
    golden_run_fixture_dir: Path,
    source_name: str,
    **paths: Path,
) -> Path:
    config = yaml.safe_load((golden_run_fixture_dir / source_name).read_text(encoding="utf-8"))
    config.update({key: value.as_posix() for key, value in paths.items()})
    destination = tmp_path / source_name
    destination.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return destination


def _golden_bytes(golden_run_fixture_dir: Path, name: str) -> bytes:
    return (golden_run_fixture_dir / "expected" / name).read_bytes()


def _tsv_rows(path: Path) -> tuple[tuple[str, ...], ...]:
    with path.open(encoding="utf-8", newline="") as stream:
        return tuple(tuple(row) for row in csv.reader(stream, delimiter="\t"))


def _pad_trailing_empty_fields(
    rows: tuple[tuple[str, ...], ...],
) -> tuple[tuple[str, ...], ...]:
    if not rows:
        return rows
    column_count = len(rows[0])
    return tuple(row + ("",) * max(0, column_count - len(row)) for row in rows)


def test_all_standardization_targets_match_golden_run(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
    golden_run_fixture_dir: Path,
) -> None:
    monkeypatch.setattr(
        location_module.reverse_geocode,
        "get",
        lambda _coordinates: (_ for _ in ()).throw(RuntimeError("synthetic service failure")),
    )
    monkeypatch.setattr(isolation_source_module.instructor, "from_openai", lambda client: client)
    extracted, biosample_manifest, bioproject_manifest = _prepare_bundle(
        tmp_path, golden_run_fixture_dir
    )
    location_config = _write_runtime_config(
        tmp_path,
        golden_run_fixture_dir,
        "location.yaml",
        geo_loc_list_path=golden_run_fixture_dir / "geo_loc_list.txt",
        cache_db_path=tmp_path / "location-cache.db",
    )
    location_policy = LocationPolicy.load(location_config)
    isolation_source_config = _write_runtime_config(
        tmp_path,
        golden_run_fixture_dir,
        "isolation_source.yaml",
        ontology_directory=golden_run_fixture_dir / "ontology",
        cache_db_path=tmp_path / "isolation-cache.db",
    )
    isolation_source_prompt_policy = IsolationSourcePromptPolicy.load(isolation_source_config)
    location_client = _FakeLocationClient()
    isolation_source_client = _FakeIsolationSourceClient()

    def location_factory(policy: LocationPolicy, logger: logging.Logger) -> LocationStandardizer:
        return LocationStandardizer(
            policy,
            client=location_client,
            llm_settings=LLMSettings(None, None, "golden-model"),
            result_logger=logger,
        )

    def host_factory(policy: HostPolicy, logger: logging.Logger) -> HostStandardizer:
        return HostStandardizer(
            policy,
            ncbi_table_path=golden_run_fixture_dir / "taxonomy.tsv",
            result_logger=logger,
        )

    def isolation_source_factory(
        policy: IsolationSourcePromptPolicy,
        bundle_path: Path,
        logger: logging.Logger,
    ) -> IsolationSourceStandardizer:
        standardizer = IsolationSourceStandardizer(
            policy,
            bundle_path,
            client=isolation_source_client,
            llm_settings=LLMSettings(None, None, "golden-model"),
            result_logger=logger,
        )
        return standardizer

    destination = tmp_path / "final.tsv"
    reasoning = tmp_path / "isolation_source_reasoning.jsonl"
    run_outputs = RunOutputs.plan(
        output_dir=tmp_path,
        run_name="golden-run",
        output_file=destination,
        include_isolation_source=True,
        include_prompt_snapshot=False,
    )
    run_report_writer = RunReport(
        run_outputs,
        RunContext(
            requested_pathogens=PATHOGENS,
            requested_standardization_targets=tuple(
                target.value for target in STANDARDIZATION_TARGETS
            ),
            extracted_metadata=extracted,
            options={},
            configuration_paths=(),
            skip_llm=False,
            model_identifiers={
                "location": "golden-model",
                "isolation_source": "golden-model",
            },
        ),
    )
    build_statistics = DatasetBuilder(
        location_standardizer_factory=location_factory,
        host_standardizer_factory=host_factory,
        host_lineage_factory=lambda *_: SimpleNamespace(
            enrich=lambda taxid: {
                9606: HostLineage("human", "Eukaryota||Metazoa||Homo sapiens", "2759||33208||9606"),
                9031: HostLineage(
                    "chicken", "Eukaryota||Metazoa||Gallus gallus", "2759||33208||9031"
                ),
            }[taxid]
        ),
        isolation_source_standardizer_factory=isolation_source_factory,
    ).build(
        DatasetBuildRequest(
            extracted_metadata=extracted,
            biosample_snapshot_manifest=biosample_manifest,
            bioproject_snapshot_manifest=bioproject_manifest,
            requested_pathogens=PATHOGENS,
            requested_targets=STANDARDIZATION_TARGETS,
            final_destination=destination,
            atb_index=golden_run_fixture_dir / "atb.tsv",
            pathogen_registry=load_pathogen_registry(),
            host_policy=HostPolicy.load(
                golden_run_fixture_dir / "host.yaml", load_pathogen_registry()
            ),
            location_policy=location_policy,
            isolation_source_prompt_policy=isolation_source_prompt_policy,
            isolation_source_reasoning_destination=reasoning,
            llm_settings=LLMSettings(None, None, "golden-model"),
            disable_progress=True,
        )
    )
    run_report_writer.finish(
        RunStatus.SUCCEEDED,
        progress=DatasetBuildProgress(
            processed_rows=build_statistics.rows_written,
            rows_written=build_statistics.rows_written,
            statistics=build_statistics,
        ),
        statistics=build_statistics,
    )
    run_report = json.loads(run_outputs.run_report.read_text(encoding="utf-8"))

    dataset_tsv_rows = _tsv_rows(destination)
    assert dataset_tsv_rows
    assert all(len(row) == len(dataset_tsv_rows[0]) for row in dataset_tsv_rows)
    expected_dataset_tsv_rows = _tsv_rows(golden_run_fixture_dir / "expected" / "final.tsv")
    assert dataset_tsv_rows == _pad_trailing_empty_fields(expected_dataset_tsv_rows)
    assert reasoning.read_bytes() == _golden_bytes(
        golden_run_fixture_dir, "isolation_source_reasoning.jsonl"
    )
    with destination.open(encoding="utf-8", newline="") as stream:
        dataset_rows = {row["accession"]: row for row in csv.DictReader(stream, delimiter="\t")}
    reasoning_rows = {
        row["accession"]: row
        for row in (json.loads(line) for line in reasoning.read_text(encoding="utf-8").splitlines())
    }
    # This pair protects fresh and cached answers from diverging during deterministic
    # post-processing while allowing their route diagnostics and record identities to differ.
    assert tuple(dataset_rows["ECO_MODEL"][column] for column in ISOLATION_SOURCE_COLUMNS) == tuple(
        dataset_rows["ABA_CACHE"][column] for column in ISOLATION_SOURCE_COLUMNS
    )
    assert {
        key: value
        for key, value in reasoning_rows["ECO_MODEL"].items()
        if key not in {"accession", "pathogen", "diagnostics"}
    } == {
        key: value
        for key, value in reasoning_rows["ABA_CACHE"].items()
        if key not in {"accession", "pathogen", "diagnostics"}
    }
    assert len(location_client.completions.calls) == 1
    assert len(isolation_source_client.completions.calls) == 1
    llm_entries = run_report["llm"]["by_target_and_model"]
    assert [entry["target"] for entry in llm_entries] == ["isolation_source", "location"]
    assert llm_entries[0]["model"] == "golden-model"
    assert llm_entries[0]["cache_hits"] == 1
