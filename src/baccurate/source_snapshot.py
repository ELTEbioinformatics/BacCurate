"""Check a raw BioSample snapshot and the TSVs derived from it."""

import hashlib
from collections import Counter
from datetime import date
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator


class SourceSnapshotError(ValueError):
    """A source snapshot manifest or derived source record is invalid."""


class SourceFile(BaseModel):
    """One raw file belonging to a source snapshot."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(min_length=1)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class SourceSnapshotManifest(BaseModel):
    """Versioned description of the raw inputs used for extraction."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    manifest_version: Literal[1]
    snapshot_id: str = Field(min_length=1)
    provider: str = Field(min_length=1)
    retrieved_on: date
    metadata_reference_date: date
    files: tuple[SourceFile, ...] = Field(min_length=1)
    snapshot_as_of: date | None = None
    source_url: str | None = None
    notes: str | None = None

    @model_validator(mode="after")
    def validate_snapshot_identity(self) -> "SourceSnapshotManifest":
        """Check relationships and uniqueness that span manifest fields."""
        expected_reference = self.snapshot_as_of or self.retrieved_on
        if self.metadata_reference_date != expected_reference:
            source_field = "snapshot_as_of" if self.snapshot_as_of else "retrieved_on"
            raise ValueError(
                "metadata_reference_date must equal "
                f"{source_field} ({expected_reference.isoformat()})"
            )
        if self.snapshot_as_of and self.snapshot_as_of > self.retrieved_on:
            raise ValueError("snapshot_as_of cannot be later than retrieved_on")

        names = [source_file.name for source_file in self.files]
        duplicate_names = sorted(name for name in set(names) if names.count(name) > 1)
        if duplicate_names:
            raise ValueError(f"duplicate file name: {duplicate_names[0]}")

        hashes = [source_file.sha256 for source_file in self.files]
        duplicate_hashes = sorted(value for value in set(hashes) if hashes.count(value) > 1)
        if duplicate_hashes:
            raise ValueError(f"duplicate SHA-256: {duplicate_hashes[0]}")
        return self

    @classmethod
    def load(cls, path: Path | str) -> "SourceSnapshotManifest":
        """Load and validate a source snapshot manifest from YAML."""
        manifest_path = Path(path)
        try:
            raw = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
            return cls.model_validate(raw)
        except (OSError, yaml.YAMLError, ValidationError) as exc:
            raise SourceSnapshotError(
                f"Invalid source snapshot manifest {manifest_path}: {exc}"
            ) from exc

    def validate_raw_inputs(self, input_paths: list[Path]) -> None:
        """Require the resolved extraction inputs to match this manifest exactly."""
        resolved = [path.resolve() for path in input_paths]
        input_name_counts = Counter(path.name for path in resolved)
        duplicate_inputs = sorted(name for name, count in input_name_counts.items() if count > 1)
        if duplicate_inputs:
            raise SourceSnapshotError(f"duplicate resolved input: {duplicate_inputs[0]}")
        actual_by_name = {path.name: path for path in resolved}
        expected_by_name = {source_file.name: source_file for source_file in self.files}

        missing = sorted(expected_by_name.keys() - actual_by_name.keys())
        extra = sorted(actual_by_name.keys() - expected_by_name.keys())
        if missing or extra:
            details = []
            if missing:
                details.append(f"missing inputs: {', '.join(missing)}")
            if extra:
                details.append(f"unlisted inputs: {', '.join(extra)}")
            raise SourceSnapshotError(
                "Raw extraction input set does not match source snapshot manifest ("
                + "; ".join(details)
                + ")"
            )

        for name, source_file in expected_by_name.items():
            actual_hash = sha256_file(actual_by_name[name])
            if actual_hash != source_file.sha256:
                raise SourceSnapshotError(
                    f"Raw input checksum mismatch for {name}: expected "
                    f"{source_file.sha256}, calculated {actual_hash}"
                )


class DerivedSourceRecord(BaseModel):
    """Reference from an extracted TSV back to its source snapshot manifest."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    source_snapshot_id: str = Field(min_length=1)
    source_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @classmethod
    def from_manifest(
        cls, manifest: SourceSnapshotManifest, manifest_path: Path | str
    ) -> "DerivedSourceRecord":
        return cls(
            source_snapshot_id=manifest.snapshot_id,
            source_manifest_sha256=sha256_file(manifest_path),
        )

    def write_for(self, extracted_metadata_path: Path | str) -> Path:
        """Write this record beside an extracted metadata TSV."""
        provenance_path = provenance_path_for(extracted_metadata_path)
        provenance_path.write_text(
            yaml.safe_dump(self.model_dump(mode="json"), sort_keys=False),
            encoding="utf-8",
        )
        return provenance_path

    @classmethod
    def load_for(cls, extracted_metadata_path: Path | str) -> "DerivedSourceRecord":
        """Load the derived source record beside an extracted metadata TSV."""
        provenance_path = provenance_path_for(extracted_metadata_path)
        if not provenance_path.exists():
            raise SourceSnapshotError(f"derived source record not found: {provenance_path}")
        try:
            raw = yaml.safe_load(provenance_path.read_text(encoding="utf-8"))
            return cls.model_validate(raw)
        except (OSError, yaml.YAMLError, ValidationError) as exc:
            raise SourceSnapshotError(
                f"Invalid derived source record {provenance_path}: {exc}"
            ) from exc


def sha256_file(path: Path | str) -> str:
    """Return the lowercase SHA-256 digest of a file's bytes."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def provenance_path_for(extracted_metadata_path: Path | str) -> Path:
    """Return the derived source record path for an extracted metadata TSV."""
    path = Path(extracted_metadata_path)
    return path.with_name(f"{path.stem}.provenance.yaml")


def validate_derived_metadata_source(
    extracted_metadata_path: Path | str, manifest_path: Path | str
) -> SourceSnapshotManifest:
    """Validate a derived TSV's reference and return its source manifest."""
    manifest = SourceSnapshotManifest.load(manifest_path)
    record = DerivedSourceRecord.load_for(extracted_metadata_path)
    if record.source_snapshot_id != manifest.snapshot_id:
        raise SourceSnapshotError(
            "Derived metadata snapshot identity mismatch: expected "
            f"{manifest.snapshot_id}, found {record.source_snapshot_id}"
        )
    expected_hash = sha256_file(manifest_path)
    if record.source_manifest_sha256 != expected_hash:
        raise SourceSnapshotError(
            "Derived metadata source manifest checksum mismatch: expected "
            f"{expected_hash}, found {record.source_manifest_sha256}"
        )
    return manifest
