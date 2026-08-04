"""Shared, fixture-sized resources for BacCurate behavior tests."""

from __future__ import annotations

import csv
import gzip
import shutil
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Protocol

import pytest
import yaml

from baccurate.extraction import COLUMNS
from baccurate.pathogen_registry.registry import PathogenRegistry, load_pathogen_registry
from baccurate.provenance.source_snapshot import (
    DerivedBundleProvenance,
    SourceSnapshotManifest,
    provenance_path_for,
    sha256_file,
    validate_extracted_metadata_bundle,
    validate_paired_source_contract,
)
from baccurate.run.dataset_builder import DatasetBuilder
from baccurate.standardization.host import HostPolicy, HostStandardizer
from baccurate.standardization.host_lineage import HostLineage
from baccurate.standardization.isolation_source import (
    IsolationSourcePromptPolicy,
)
from baccurate.standardization.location import LocationPolicy

# --- Fixture data structures ---


@dataclass(frozen=True, slots=True)
class StandardizationFixtureResources:
    """Paths to the compact reference and policy resources shared by tests."""

    root: Path

    @property
    def ncbi_taxonomy_reference_table(self) -> Path:
        return self.root / "taxonomy.tsv"

    @property
    def isolation_source_ontology_directory(self) -> Path:
        return self.root / "ontology"

    @property
    def geographic_locations(self) -> Path:
        return self.root / "geo_loc_list.txt"

    @property
    def pathogen_registry(self) -> Path:
        return self.root / "pathogens.yaml"

    @property
    def host_policy(self) -> Path:
        return self.root / "host.yaml"

    @property
    def location_policy(self) -> Path:
        return self.root / "location.yaml"

    @property
    def isolation_source_policy(self) -> Path:
        return self.root / "isolation_source.yaml"

    @property
    def extracted_metadata(self) -> Path:
        return self.root / "extracted.tsv"

    @property
    def atb_index(self) -> Path:
        return self.root / "atb.tsv"


@dataclass(frozen=True, slots=True)
class ExtractedMetadataBundle:
    """Paths making up one validated, provenance-bound extracted metadata bundle."""

    extracted_metadata: Path
    biosample_snapshot_manifest: Path
    bioproject_snapshot_manifest: Path
    provenance: Path


class ExtractedMetadataBundleFactory(Protocol):
    """Build a named extracted metadata bundle from optional custom records."""

    def __call__(
        self,
        name: str = "bundle",
        extracted_rows: Sequence[Mapping[str, object]] | None = None,
    ) -> ExtractedMetadataBundle: ...


@dataclass(frozen=True, slots=True)
class PairedSourceSnapshots:
    """A valid paired BioSample and BioProject acquired source snapshot."""

    biosample: Path
    bioproject: Path
    biosample_manifest: Path
    bioproject_manifest: Path

    def replace_contents(self, *, biosample_xml: bytes, bioproject_xml: bytes) -> None:
        """Replace both source snapshots while keeping their manifests valid."""
        _write_compressed_snapshot(self.biosample, biosample_xml)
        _write_compressed_snapshot(self.bioproject, bioproject_xml)
        _write_source_manifest(
            self.biosample_manifest,
            snapshot_id="biosample-test",
            source=self.biosample,
            snapshot_date=date(2026, 7, 19),
        )
        _write_source_manifest(
            self.bioproject_manifest,
            snapshot_id="bioproject-test",
            source=self.bioproject,
            snapshot_date=date(2026, 7, 19),
        )


# --- Reference and policy fixtures ---


@pytest.fixture(scope="session")
def standardization_fixture_resources() -> StandardizationFixtureResources:
    """Return the compact resources used by standardization and dataset-build tests."""
    return StandardizationFixtureResources(Path(__file__).parent / "fixtures" / "standardization")


@pytest.fixture(scope="session")
def fixture_pathogen_registry(
    standardization_fixture_resources: StandardizationFixtureResources,
) -> PathogenRegistry:
    """Load the fixture-sized target-pathogen registry through its public loader."""
    return load_pathogen_registry(standardization_fixture_resources.pathogen_registry)


@pytest.fixture(scope="session")
def fixture_host_policy(
    standardization_fixture_resources: StandardizationFixtureResources,
    fixture_pathogen_registry: PathogenRegistry,
) -> HostPolicy:
    """Load fixture host policy through the same validation boundary as production policy."""
    return HostPolicy.load(
        standardization_fixture_resources.host_policy,
        fixture_pathogen_registry,
    )


@pytest.fixture
def fixture_location_policy(
    tmp_path: Path,
    standardization_fixture_resources: StandardizationFixtureResources,
) -> LocationPolicy:
    """Load geographic-location policy against fixture locations and a temporary cache."""
    path = _write_runtime_policy(
        standardization_fixture_resources.location_policy,
        tmp_path / "location.yaml",
        geo_loc_list_path=standardization_fixture_resources.geographic_locations,
        cache_db_path=tmp_path / "location-cache.db",
    )
    return LocationPolicy.load(path)


@pytest.fixture
def fixture_isolation_source_prompt_policy(
    tmp_path: Path,
    standardization_fixture_resources: StandardizationFixtureResources,
) -> IsolationSourcePromptPolicy:
    """Load isolation-source policy against the fixture ontology and a temporary cache."""
    path = _write_runtime_policy(
        standardization_fixture_resources.isolation_source_policy,
        tmp_path / "isolation_source.yaml",
        ontology_directory=standardization_fixture_resources.isolation_source_ontology_directory,
        cache_db_path=tmp_path / "isolation-source-cache.db",
    )
    return IsolationSourcePromptPolicy.load(path)


# --- Extracted metadata bundle fixtures ---


@pytest.fixture
def paired_source_snapshots(tmp_path: Path) -> PairedSourceSnapshots:
    """Return the default valid paired acquired source snapshots."""
    sources = PairedSourceSnapshots(
        biosample=tmp_path / "biosamples.xml.gz",
        bioproject=tmp_path / "bioproject.xml.gz",
        biosample_manifest=tmp_path / "biosample_snapshot.yaml",
        bioproject_manifest=tmp_path / "bioproject_snapshot.yaml",
    )
    sources.replace_contents(
        biosample_xml=b"<BioSampleSet />",
        bioproject_xml=b"<PackageSet />",
    )
    return sources


@pytest.fixture
def extracted_metadata_bundle_factory(
    tmp_path: Path,
    standardization_fixture_resources: StandardizationFixtureResources,
    fixture_isolation_source_prompt_policy: IsolationSourcePromptPolicy,
) -> ExtractedMetadataBundleFactory:
    """Build deterministic extracted metadata bundles with valid source and artifact provenance."""

    def build(
        name: str = "bundle",
        extracted_rows: Sequence[Mapping[str, object]] | None = None,
    ) -> ExtractedMetadataBundle:
        bundle_dir = tmp_path / name
        bundle_dir.mkdir()
        extracted_metadata = bundle_dir / "extracted.tsv"

        if extracted_rows is None:
            shutil.copyfile(
                standardization_fixture_resources.extracted_metadata,
                extracted_metadata,
            )
        else:
            _write_extracted_metadata(extracted_metadata, extracted_rows)
        biosample_source = bundle_dir / "biosample.xml.gz"
        biosample_source.write_bytes(b"fixture BioSample source\n")
        bioproject_source = bundle_dir / "bioproject.xml.gz"
        bioproject_source.write_bytes(b"fixture BioProject source\n")
        biosample_manifest = _write_source_manifest(
            bundle_dir / "biosample_snapshot.yaml",
            snapshot_id="fixture-biosample-2026-01-01",
            source=biosample_source,
        )
        bioproject_manifest = _write_source_manifest(
            bundle_dir / "bioproject_snapshot.yaml",
            snapshot_id="fixture-bioproject-2026-01-01",
            source=bioproject_source,
        )
        source_contract = validate_paired_source_contract(
            biosample_path=biosample_source,
            bioproject_path=bioproject_source,
            biosample_manifest_path=biosample_manifest,
            bioproject_manifest_path=bioproject_manifest,
        )
        provenance = DerivedBundleProvenance.create(
            source_contract=source_contract,
            biosample_manifest_path=biosample_manifest,
            bioproject_manifest_path=bioproject_manifest,
            extracted_metadata_path=extracted_metadata,
        ).write(provenance_path_for(extracted_metadata))
        validate_extracted_metadata_bundle(
            extracted_metadata,
            biosample_manifest,
            bioproject_manifest,
        )
        return ExtractedMetadataBundle(
            extracted_metadata=extracted_metadata,
            biosample_snapshot_manifest=biosample_manifest,
            bioproject_snapshot_manifest=bioproject_manifest,
            provenance=provenance,
        )

    return build


@pytest.fixture
def extracted_metadata_bundle(
    extracted_metadata_bundle_factory: ExtractedMetadataBundleFactory,
) -> ExtractedMetadataBundle:
    """Return the default one-record extracted metadata bundle."""
    return extracted_metadata_bundle_factory()


# --- DatasetBuilder fixture ---


@pytest.fixture
def fixture_dataset_builder_factory(
    standardization_fixture_resources: StandardizationFixtureResources,
) -> Callable[..., DatasetBuilder]:
    """Build DatasetBuilder instances with compact defaults and optional collaborators."""
    lineage_by_taxid = {
        9606: HostLineage(
            "human",
            "Eukaryota||Metazoa||Homo sapiens",
            "2759||33208||9606",
        ),
        9031: HostLineage(
            "chicken",
            "Eukaryota||Metazoa||Gallus gallus",
            "2759||33208||9031",
        ),
    }

    class FixtureHostLineageEnricher:
        def enrich(self, taxid: int) -> HostLineage:
            return lineage_by_taxid[taxid]

        @staticmethod
        def is_descendant_or_self(taxid: int, ancestor_taxid: int) -> bool:
            return (taxid, ancestor_taxid) in {(9606, 33208), (9031, 33208)}

    def build(
        *,
        host_standardizer_factory=None,
        host_lineage_factory=None,
        isolation_source_standardizer_factory=None,
    ) -> DatasetBuilder:
        return DatasetBuilder(
            host_standardizer_factory=host_standardizer_factory
            or (
                lambda policy, result_logger: HostStandardizer(
                    policy,
                    ncbi_table_path=(
                        standardization_fixture_resources.ncbi_taxonomy_reference_table
                    ),
                    result_logger=result_logger,
                )
            ),
            host_lineage_factory=host_lineage_factory
            or (lambda _names, _nodes: FixtureHostLineageEnricher()),
            isolation_source_standardizer_factory=isolation_source_standardizer_factory,
        )

    return build


@pytest.fixture
def fixture_dataset_builder(fixture_dataset_builder_factory) -> DatasetBuilder:
    """Return a DatasetBuilder wired only to fixture-sized host references."""
    return fixture_dataset_builder_factory()


# --- Golden run fixture ---


@pytest.fixture
def golden_run_fixture_dir() -> Path:
    """Return the fixture-only resources for the end-to-end golden run."""
    return Path(__file__).parent / "fixtures" / "golden_run"


# --- Artifact writers ---


def _write_extracted_metadata(
    destination: Path,
    rows: Sequence[Mapping[str, object]],
) -> None:
    with destination.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, COLUMNS, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _write_source_manifest(
    destination: Path,
    *,
    snapshot_id: str,
    source: Path,
    snapshot_date: date = date(2026, 1, 1),
) -> Path:
    manifest = SourceSnapshotManifest(
        snapshot_id=snapshot_id,
        provider="BacCurate test fixture",
        retrieved_on=snapshot_date,
        file={"name": source.name, "sha256": sha256_file(source)},
    )
    destination.write_text(
        yaml.safe_dump(manifest.model_dump(mode="json"), sort_keys=False),
        encoding="utf-8",
    )
    SourceSnapshotManifest.load(destination)
    return destination


def _write_compressed_snapshot(destination: Path, contents: bytes) -> None:
    with (
        destination.open("wb") as raw_stream,
        gzip.GzipFile(fileobj=raw_stream, mode="wb", mtime=0) as stream,
    ):
        stream.write(contents)


def _write_runtime_policy(source: Path, destination: Path, **paths: Path) -> Path:
    policy = yaml.safe_load(source.read_text(encoding="utf-8"))
    policy.update({key: value.as_posix() for key, value in paths.items()})
    destination.write_text(yaml.safe_dump(policy, sort_keys=False), encoding="utf-8")
    return destination
