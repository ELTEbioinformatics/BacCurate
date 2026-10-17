"""Validate snapshots against their manifests and publish extracted metadata bundles."""

import hashlib
import os
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field


class SnapshotError(ValueError):
    """A snapshot, snapshot manifest or bundle provenance record is invalid."""


class SnapshotFile(BaseModel):
    """The file name and SHA-256 of one snapshot."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(min_length=1)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class SnapshotManifest(BaseModel):
    """The identity, retrieval date and file of one snapshot."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    snapshot_id: str = Field(min_length=1)
    provider: str = Field(min_length=1)
    url: str | None = None
    retrieved_on: date
    snapshot_as_of: date | None = None
    file: SnapshotFile

    @property
    def metadata_reference_date(self) -> date:
        return self.snapshot_as_of or self.retrieved_on

    @classmethod
    def load(cls, path: Path | str) -> "SnapshotManifest":
        """Load and validate a snapshot manifest from YAML."""
        return cls.model_validate(yaml.safe_load(Path(path).read_text(encoding="utf-8")))

    def write(self, path: Path | str) -> Path:
        """Write this manifest as YAML in field declaration order."""
        destination = Path(path)
        destination.write_text(
            yaml.safe_dump(self.model_dump(), sort_keys=False),
            encoding="utf-8",
            newline="\n",
        )
        return destination


@dataclass(frozen=True, slots=True)
class SnapshotPair:
    """Validated BioSample and BioProject snapshot manifests."""

    biosample: SnapshotManifest
    bioproject: SnapshotManifest

    @property
    def metadata_reference_date(self) -> date:
        return self.biosample.metadata_reference_date


def validate_snapshot_pair(
    *,
    biosample_snapshot_path: Path | str,
    bioproject_snapshot_path: Path | str,
    biosample_manifest_path: Path | str,
    bioproject_manifest_path: Path | str,
) -> SnapshotPair:
    """Load both manifests and check each snapshot against its manifest."""
    biosample_manifest = SnapshotManifest.load(biosample_manifest_path)
    if sha256_file(biosample_snapshot_path) != biosample_manifest.file.sha256:
        raise SnapshotError("BioSample snapshot checksum mismatch")
    bioproject_manifest = SnapshotManifest.load(bioproject_manifest_path)
    if sha256_file(bioproject_snapshot_path) != bioproject_manifest.file.sha256:
        raise SnapshotError("BioProject snapshot checksum mismatch")
    return SnapshotPair(
        biosample=biosample_manifest,
        bioproject=bioproject_manifest,
    )


class DerivedBundleProvenance(BaseModel):
    """Hash binding between a snapshot pair and an extracted metadata bundle."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    biosample_snapshot_id: str = Field(min_length=1)
    biosample_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    bioproject_snapshot_id: str = Field(min_length=1)
    bioproject_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    extracted_metadata_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @classmethod
    def create(
        cls,
        *,
        snapshot_pair: SnapshotPair,
        biosample_manifest_path: Path | str,
        bioproject_manifest_path: Path | str,
        extracted_metadata_path: Path | str,
    ) -> "DerivedBundleProvenance":
        """Bind validated snapshot manifests to completed temporary artifacts."""
        return cls(
            biosample_snapshot_id=snapshot_pair.biosample.snapshot_id,
            biosample_manifest_sha256=sha256_file(biosample_manifest_path),
            bioproject_snapshot_id=snapshot_pair.bioproject.snapshot_id,
            bioproject_manifest_sha256=sha256_file(bioproject_manifest_path),
            extracted_metadata_sha256=sha256_file(extracted_metadata_path),
        )

    def write(self, path: Path | str) -> Path:
        """Write this provenance record as deterministic YAML."""
        destination = Path(path)
        destination.write_text(
            yaml.safe_dump(self.model_dump(mode="json"), sort_keys=False),
            encoding="utf-8",
        )
        return destination


def sha256_file(path: Path | str) -> str:
    """Return the lowercase SHA-256 digest of a file's bytes."""
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def provenance_path_for(extracted_metadata_path: Path | str) -> Path:
    """Return the bundle provenance record path for an extracted metadata TSV."""
    path = Path(extracted_metadata_path)
    return path.with_name(f"{path.stem}.provenance.yaml")


def load_derived_bundle_provenance(
    extracted_metadata_path: Path | str,
) -> DerivedBundleProvenance:
    """Load provenance and validate an extracted metadata bundle."""
    extracted_metadata_path = Path(extracted_metadata_path)
    provenance_path = provenance_path_for(extracted_metadata_path)
    bundle = DerivedBundleProvenance.model_validate(
        yaml.safe_load(provenance_path.read_text(encoding="utf-8"))
    )
    if sha256_file(extracted_metadata_path) != bundle.extracted_metadata_sha256:
        raise SnapshotError("Derived extracted TSV checksum mismatch")
    return bundle


def _publish_bundle(
    *,
    temporary_output_path: Path,
    output_path: Path,
    temporary_provenance_path: Path,
    provenance_path: Path,
) -> None:
    """Publish bundle members with provenance last as the bundle-validity marker."""
    provenance_path.unlink(missing_ok=True)
    try:
        os.replace(temporary_output_path, output_path)
        os.replace(temporary_provenance_path, provenance_path)
    except Exception:
        provenance_path.unlink(missing_ok=True)
        raise


def validate_extracted_metadata_bundle(
    extracted_metadata_path: Path | str,
    biosample_manifest_path: Path | str,
    bioproject_manifest_path: Path | str,
) -> SnapshotPair:
    """Validate an extracted metadata bundle and return its snapshot pair."""
    biosample_manifest = SnapshotManifest.load(biosample_manifest_path)
    bioproject_manifest = SnapshotManifest.load(bioproject_manifest_path)
    bundle = load_derived_bundle_provenance(extracted_metadata_path)

    if (
        bundle.biosample_snapshot_id != biosample_manifest.snapshot_id
        or bundle.biosample_manifest_sha256 != sha256_file(biosample_manifest_path)
    ):
        raise SnapshotError("Derived BioSample snapshot manifest mismatch")
    if (
        bundle.bioproject_snapshot_id != bioproject_manifest.snapshot_id
        or bundle.bioproject_manifest_sha256 != sha256_file(bioproject_manifest_path)
    ):
        raise SnapshotError("Derived BioProject snapshot manifest mismatch")
    return SnapshotPair(
        biosample=biosample_manifest,
        bioproject=bioproject_manifest,
    )
