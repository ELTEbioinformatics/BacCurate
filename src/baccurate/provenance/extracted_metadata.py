"""Record and check the snapshot manifests that an extracted metadata TSV came from."""

from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field

from baccurate.provenance.sha256 import sha256_file
from baccurate.provenance.snapshot_manifest import SnapshotManifest, SnapshotPair


class ExtractedMetadataError(ValueError):
    """Extracted metadata does not match its provenance file."""


class ExtractedMetadataProvenance(BaseModel):
    """The SHA-256 values of an extracted metadata TSV and of its two snapshot manifests."""

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
        biosample_manifest_path: Path,
        bioproject_manifest_path: Path,
        extracted_metadata_path: Path,
    ) -> "ExtractedMetadataProvenance":
        """Record the manifests and the extracted metadata TSV as they are on disk."""
        return cls(
            biosample_snapshot_id=SnapshotManifest.load(biosample_manifest_path).snapshot_id,
            biosample_manifest_sha256=sha256_file(biosample_manifest_path),
            bioproject_snapshot_id=SnapshotManifest.load(bioproject_manifest_path).snapshot_id,
            bioproject_manifest_sha256=sha256_file(bioproject_manifest_path),
            extracted_metadata_sha256=sha256_file(extracted_metadata_path),
        )

    @classmethod
    def load(cls, path: Path) -> "ExtractedMetadataProvenance":
        """Load an extracted metadata provenance file from YAML."""
        return cls.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))

    def write(self, path: Path) -> None:
        """Write this provenance file as deterministic YAML."""
        path.write_text(
            yaml.safe_dump(self.model_dump(mode="json"), sort_keys=False),
            encoding="utf-8",
        )


def provenance_path_for(extracted_metadata_path: Path) -> Path:
    """Return the provenance file path for an extracted metadata TSV."""
    return extracted_metadata_path.with_name(f"{extracted_metadata_path.stem}.provenance.yaml")


def validate_extracted_metadata(
    extracted_metadata_path: Path,
    biosample_manifest_path: Path,
    bioproject_manifest_path: Path,
) -> SnapshotPair:
    """Check an extracted metadata TSV and both manifests against its provenance file."""
    provenance = ExtractedMetadataProvenance.load(provenance_path_for(extracted_metadata_path))
    if sha256_file(extracted_metadata_path) != provenance.extracted_metadata_sha256:
        raise ExtractedMetadataError("Extracted metadata checksum mismatch")
    if sha256_file(biosample_manifest_path) != provenance.biosample_manifest_sha256:
        raise ExtractedMetadataError("BioSample manifest checksum mismatch")
    if sha256_file(bioproject_manifest_path) != provenance.bioproject_manifest_sha256:
        raise ExtractedMetadataError("BioProject manifest checksum mismatch")
    return SnapshotPair(
        biosample=SnapshotManifest.load(biosample_manifest_path),
        bioproject=SnapshotManifest.load(bioproject_manifest_path),
    )
