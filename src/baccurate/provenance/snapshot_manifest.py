"""Validate snapshots against their manifests."""

from dataclasses import dataclass
from datetime import date
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field

from baccurate.provenance.sha256 import sha256_file


class SnapshotError(ValueError):
    """A snapshot or snapshot manifest is invalid."""


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
    def load(cls, path: Path) -> "SnapshotManifest":
        """Load and validate a snapshot manifest from YAML."""
        return cls.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))

    def write(self, path: Path) -> None:
        """Write this manifest as YAML in field declaration order."""
        path.write_text(
            yaml.safe_dump(self.model_dump(), sort_keys=False),
            encoding="utf-8",
            newline="\n",
        )


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
    biosample_snapshot_path: Path,
    bioproject_snapshot_path: Path,
    biosample_manifest_path: Path,
    bioproject_manifest_path: Path,
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
