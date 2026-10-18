"""Shared, fixture-sized resources for BacCurate behavior tests."""

from __future__ import annotations

import csv
import gzip
import shutil
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from baccurate.extraction import COLUMNS
from baccurate.provenance.extracted_metadata import (
    ExtractedMetadataProvenance,
    provenance_path_for,
    validate_extracted_metadata,
)
from baccurate.provenance.sha256 import sha256_file
from baccurate.provenance.snapshot_manifest import SnapshotManifest
from baccurate.run.dataset_builder import DatasetBuilder
from baccurate.standardization.host import HostPolicy, HostStandardizer
from baccurate.standardization.host_lineage import HostLineage
from baccurate.standardization.isolation_source import (
    IsolationSourcePromptPolicy,
)
from baccurate.standardization.location import LocationPolicy
from baccurate.taxon_registry.registry import TaxonRegistry, load_taxon_registry

CLASSIFIER_FACETS = (
    "source_type",
    "body_product",
    "body_site",
    "lesion",
    "environmental_material",
    "facility",
    "sampled_object",
    "food_type",
    "host_disease",
)

FACET_BY_LABEL = {
    "host-associated": "source_type",
    "animal host": "source_type",
    "environmental": "source_type",
    "blood": "body_product",
    "feces": "body_product",
    "wound": "lesion",
    "soil": "environmental_material",
    "meat product": "food_type",
}

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
    def taxon_registry(self) -> Path:
        return self.root / "taxa.yaml"

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


@dataclass(frozen=True, slots=True)
class ExtractedMetadataFiles:
    """An extracted metadata TSV, its provenance file and the two manifests it records."""

    extracted_metadata_path: Path
    biosample_manifest_path: Path
    bioproject_manifest_path: Path
    provenance_path: Path


@dataclass(frozen=True, slots=True)
class PairedSnapshots:
    """Valid BioSample and BioProject snapshots with their manifests."""

    biosample_snapshot_path: Path
    bioproject_snapshot_path: Path
    biosample_manifest_path: Path
    bioproject_manifest_path: Path

    def replace_contents(self, *, biosample_xml: bytes, bioproject_xml: bytes) -> None:
        """Replace both snapshots and rewrite their manifests to match."""
        _write_compressed_snapshot(self.biosample_snapshot_path, biosample_xml)
        _write_compressed_snapshot(self.bioproject_snapshot_path, bioproject_xml)
        _write_manifest(
            self.biosample_manifest_path,
            snapshot_id="biosample-test",
            snapshot_path=self.biosample_snapshot_path,
            retrieved_on=date(2026, 7, 19),
        )
        _write_manifest(
            self.bioproject_manifest_path,
            snapshot_id="bioproject-test",
            snapshot_path=self.bioproject_snapshot_path,
            retrieved_on=date(2026, 7, 19),
        )


class FakeLLMClient:
    """Answer every structured-output request with one scripted behavior and record the request.

    The behavior is a classifier answer, an exception to raise, or a callable that receives the
    response model.
    """

    def __init__(self) -> None:
        self.calls: list[dict] = []
        self.behavior: object | None = None
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        behavior = self.behavior
        if behavior is None:
            raise AssertionError("LLM create() was called unexpectedly")
        if isinstance(behavior, Exception):
            raise behavior
        if callable(behavior):
            return behavior(kwargs["response_model"])
        return kwargs["response_model"].model_validate(behavior)

    @staticmethod
    def answer(
        *,
        reasoning: str = "because",
        evidence_level: str = "none",
        **facets: str | list[str] | None,
    ) -> dict[str, object]:
        """Return a classifier answer with an empty list for every facet that is not given."""
        if unknown := facets.keys() - set(CLASSIFIER_FACETS):
            raise TypeError(f"unknown classifier facets: {sorted(unknown)}")
        answer: dict[str, object] = {"reasoning": reasoning, "evidence_level": evidence_level}
        for facet in CLASSIFIER_FACETS:
            labels = facets.get(facet) or []
            answer[facet] = [labels] if isinstance(labels, str) else list(labels)
        return answer

    def respond_with_facets(self, **answer: object) -> None:
        self.behavior = self.answer(**answer)

    def respond_with(self, labels: Sequence[str], **answer: object) -> None:
        facets: dict[str, list[str]] = {}
        for label in labels:
            facets.setdefault(FACET_BY_LABEL[label], []).append(label)
        self.respond_with_facets(**facets, **answer)

    def close(self) -> None:
        pass


@pytest.fixture
def fake_llm_client() -> FakeLLMClient:
    """Return a fake LLM client that fails the test if a request reaches it unscripted."""
    return FakeLLMClient()


# --- Reference and policy fixtures ---


@pytest.fixture(scope="session")
def standardization_fixture_resources() -> StandardizationFixtureResources:
    """Return the compact resources used by standardization and dataset-build tests."""
    return StandardizationFixtureResources(Path(__file__).parent / "fixtures" / "standardization")


@pytest.fixture(scope="session")
def fixture_taxon_registry(
    standardization_fixture_resources: StandardizationFixtureResources,
) -> TaxonRegistry:
    """Load the fixture-sized target-taxon registry through its public loader."""
    return load_taxon_registry(standardization_fixture_resources.taxon_registry)


@pytest.fixture(scope="session")
def fixture_host_policy(
    standardization_fixture_resources: StandardizationFixtureResources,
    fixture_taxon_registry: TaxonRegistry,
) -> HostPolicy:
    """Load fixture host policy through the same validation boundary as production policy."""
    return HostPolicy.load(
        standardization_fixture_resources.host_policy,
        fixture_taxon_registry,
    )


@pytest.fixture
def fixture_location_policy(
    tmp_path: Path,
    standardization_fixture_resources: StandardizationFixtureResources,
) -> LocationPolicy:
    """Load geographic-location policy against the fixture INSDC location vocabulary."""
    path = _write_runtime_policy(
        standardization_fixture_resources.location_policy,
        tmp_path / "location.yaml",
        geo_loc_list_path=standardization_fixture_resources.geographic_locations,
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


# --- Extracted metadata fixtures ---


@pytest.fixture
def paired_snapshots(tmp_path: Path) -> PairedSnapshots:
    """Return the default valid snapshot pair on disk."""
    snapshots = PairedSnapshots(
        biosample_snapshot_path=tmp_path / "biosamples.xml.gz",
        bioproject_snapshot_path=tmp_path / "bioproject.xml.gz",
        biosample_manifest_path=tmp_path / "biosample.yaml",
        bioproject_manifest_path=tmp_path / "bioproject.yaml",
    )
    snapshots.replace_contents(
        biosample_xml=b"<BioSampleSet />",
        bioproject_xml=b"<PackageSet />",
    )
    return snapshots


@pytest.fixture
def make_extracted_metadata_files(
    tmp_path: Path,
    standardization_fixture_resources: StandardizationFixtureResources,
    fixture_isolation_source_prompt_policy: IsolationSourcePromptPolicy,
) -> Callable[..., ExtractedMetadataFiles]:
    """Build deterministic extracted metadata files with a valid provenance file."""

    def build(
        name: str = "extracted",
        extracted_rows: Sequence[Mapping[str, object]] | None = None,
    ) -> ExtractedMetadataFiles:
        files_dir = tmp_path / name
        files_dir.mkdir()
        extracted_metadata = files_dir / "extracted.tsv"

        if extracted_rows is None:
            shutil.copyfile(
                standardization_fixture_resources.extracted_metadata,
                extracted_metadata,
            )
        else:
            _write_extracted_metadata(extracted_metadata, extracted_rows)
        biosample_snapshot_path = files_dir / "biosample.xml.gz"
        biosample_snapshot_path.write_bytes(b"fixture BioSample source\n")
        bioproject_snapshot_path = files_dir / "bioproject.xml.gz"
        bioproject_snapshot_path.write_bytes(b"fixture BioProject source\n")
        biosample_manifest_path = _write_manifest(
            files_dir / "biosample.yaml",
            snapshot_id="fixture-biosample-2026-01-01",
            snapshot_path=biosample_snapshot_path,
        )
        bioproject_manifest_path = _write_manifest(
            files_dir / "bioproject.yaml",
            snapshot_id="fixture-bioproject-2026-01-01",
            snapshot_path=bioproject_snapshot_path,
        )
        provenance_path = provenance_path_for(extracted_metadata)
        ExtractedMetadataProvenance.create(
            biosample_manifest_path=biosample_manifest_path,
            bioproject_manifest_path=bioproject_manifest_path,
            extracted_metadata_path=extracted_metadata,
        ).write(provenance_path)
        validate_extracted_metadata(
            extracted_metadata,
            biosample_manifest_path,
            bioproject_manifest_path,
        )
        return ExtractedMetadataFiles(
            extracted_metadata_path=extracted_metadata,
            biosample_manifest_path=biosample_manifest_path,
            bioproject_manifest_path=bioproject_manifest_path,
            provenance_path=provenance_path,
        )

    return build


@pytest.fixture
def extracted_metadata_files(
    make_extracted_metadata_files: Callable[..., ExtractedMetadataFiles],
) -> ExtractedMetadataFiles:
    """Return the default one-record extracted metadata files."""
    return make_extracted_metadata_files()


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
            return (taxid, ancestor_taxid) in {(9606, 9606), (9606, 33208), (9031, 33208)}

    def build() -> DatasetBuilder:
        return DatasetBuilder(
            host_standardizer_factory=lambda policy, result_logger: HostStandardizer(
                policy,
                ncbi_table_path=standardization_fixture_resources.ncbi_taxonomy_reference_table,
                result_logger=result_logger,
            ),
            host_lineage_factory=lambda _names, _nodes: FixtureHostLineageEnricher(),
        )

    return build


@pytest.fixture
def fixture_dataset_builder(fixture_dataset_builder_factory) -> DatasetBuilder:
    """Return a DatasetBuilder wired only to fixture-sized host references."""
    return fixture_dataset_builder_factory()


# --- Artifact writers ---


def _write_extracted_metadata(
    destination: Path,
    rows: Sequence[Mapping[str, object]],
) -> None:
    with destination.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, COLUMNS, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _write_manifest(
    destination: Path,
    *,
    snapshot_id: str,
    snapshot_path: Path,
    retrieved_on: date = date(2026, 1, 1),
) -> Path:
    manifest = SnapshotManifest(
        snapshot_id=snapshot_id,
        provider="BacCurate test fixture",
        retrieved_on=retrieved_on,
        file={"name": snapshot_path.name, "sha256": sha256_file(snapshot_path)},
    )
    manifest.write(destination)
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
