"""Validate raw extraction snapshots and the metadata derived from them."""

import hashlib
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
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
                details.append(f"unexpected inputs: {', '.join(extra)}")
            raise SourceSnapshotError(
                "Raw extraction input set does not match source snapshot manifest ("
                + "; ".join(details)
                + ")"
            )

        absent_on_disk = sorted(name for name, path in actual_by_name.items() if not path.is_file())
        if absent_on_disk:
            raise SourceSnapshotError(f"missing input files: {', '.join(absent_on_disk)}")

        for name, source_file in expected_by_name.items():
            actual_hash = sha256_file(actual_by_name[name])
            if actual_hash != source_file.sha256:
                raise SourceSnapshotError(
                    f"Raw input checksum mismatch for {name}: expected "
                    f"{source_file.sha256}, calculated {actual_hash}"
                )


@dataclass(frozen=True, slots=True)
class PairedSourceContract:
    """Validated BioSample and BioProject source snapshot identities."""

    biosample: SourceSnapshotManifest
    bioproject: SourceSnapshotManifest

    @property
    def metadata_reference_date(self) -> date:
        return self.biosample.metadata_reference_date


def validate_paired_source_contract(
    *,
    biosample_path: Path | str | Iterable[Path | str],
    bioproject_path: Path | str,
    biosample_manifest_path: Path | str,
    bioproject_manifest_path: Path | str,
) -> PairedSourceContract:
    """Load and validate the two independently versioned extraction snapshots."""
    biosample_paths = (
        [Path(biosample_path)]
        if isinstance(biosample_path, (str, Path))
        else [Path(path) for path in biosample_path]
    )
    biosample_manifest = _validate_snapshot_source(
        role="BioSample",
        source_paths=biosample_paths,
        manifest_path=biosample_manifest_path,
    )
    bioproject_manifest = _validate_snapshot_source(
        role="BioProject",
        source_paths=[Path(bioproject_path)],
        manifest_path=bioproject_manifest_path,
    )
    return PairedSourceContract(
        biosample=biosample_manifest,
        bioproject=bioproject_manifest,
    )


def _validate_snapshot_source(
    *,
    role: Literal["BioSample", "BioProject"],
    source_paths: list[Path],
    manifest_path: Path | str,
) -> SourceSnapshotManifest:
    try:
        manifest = SourceSnapshotManifest.load(manifest_path)
        manifest.validate_raw_inputs(source_paths)
    except SourceSnapshotError as exc:
        raise SourceSnapshotError(f"{role} source validation failed: {exc}") from exc
    return manifest


class ManifestReference(BaseModel):
    """One raw manifest bound into a derived bundle."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    snapshot_id: str = Field(min_length=1)
    path: str = Field(min_length=1)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class PairedManifestReferences(BaseModel):
    """The independently versioned sources used by extraction."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    biosample: ManifestReference
    bioproject: ManifestReference


class ArtifactReference(BaseModel):
    """One published member of a derived bundle."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    path: str = Field(min_length=1)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class DerivedArtifactReferences(BaseModel):
    """The data artifacts committed by one provenance record."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    extracted_metadata: ArtifactReference
    bioproject_context: ArtifactReference


class DerivedBundleProvenance(BaseModel):
    """Hash binding for a paired-source, two-artifact extraction bundle."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    bundle_version: Literal[1]
    source_manifests: PairedManifestReferences
    artifacts: DerivedArtifactReferences

    @classmethod
    def create(
        cls,
        *,
        source_contract: PairedSourceContract,
        biosample_manifest_path: Path | str,
        bioproject_manifest_path: Path | str,
        extracted_metadata_path: Path | str,
        bioproject_context_path: Path | str,
    ) -> "DerivedBundleProvenance":
        """Bind validated raw manifests to completed temporary artifacts."""
        biosample_manifest_path = Path(biosample_manifest_path)
        bioproject_manifest_path = Path(bioproject_manifest_path)
        extracted_metadata_path = Path(extracted_metadata_path)
        bioproject_context_path = Path(bioproject_context_path)
        return cls(
            bundle_version=1,
            source_manifests=PairedManifestReferences(
                biosample=ManifestReference(
                    snapshot_id=source_contract.biosample.snapshot_id,
                    path=str(biosample_manifest_path),
                    sha256=sha256_file(biosample_manifest_path),
                ),
                bioproject=ManifestReference(
                    snapshot_id=source_contract.bioproject.snapshot_id,
                    path=str(bioproject_manifest_path),
                    sha256=sha256_file(bioproject_manifest_path),
                ),
            ),
            artifacts=DerivedArtifactReferences(
                extracted_metadata=ArtifactReference(
                    path=extracted_metadata_path.name,
                    sha256=sha256_file(extracted_metadata_path),
                ),
                bioproject_context=ArtifactReference(
                    path=bioproject_context_path.name,
                    sha256=sha256_file(bioproject_context_path),
                ),
            ),
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
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def provenance_path_for(extracted_metadata_path: Path | str) -> Path:
    """Return the derived source record path for an extracted metadata TSV."""
    path = Path(extracted_metadata_path)
    return path.with_name(f"{path.stem}.provenance.yaml")


def bioproject_catalog_path_for(extracted_metadata_path: Path | str) -> Path:
    """Return the BioProject context catalog paired with an extracted TSV."""
    path = Path(extracted_metadata_path)
    return path.with_name(f"{path.stem}.bioproject_context.jsonl")


def validate_derived_metadata_source(
    extracted_metadata_path: Path | str,
    biosample_manifest_path: Path | str,
    bioproject_manifest_path: Path | str,
) -> PairedSourceContract:
    """Validate a derived bundle and return both source manifest identities."""
    biosample_manifest = SourceSnapshotManifest.load(biosample_manifest_path)
    bioproject_manifest = SourceSnapshotManifest.load(bioproject_manifest_path)
    provenance_path = provenance_path_for(extracted_metadata_path)
    try:
        raw = yaml.safe_load(provenance_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise SourceSnapshotError(
            f"Invalid derived bundle provenance {provenance_path}: {exc}"
        ) from exc
    try:
        bundle = DerivedBundleProvenance.model_validate(raw)
    except ValidationError as exc:
        raise SourceSnapshotError(
            f"Invalid derived bundle provenance {provenance_path}: {exc}"
        ) from exc

    _validate_manifest_reference(
        role="BioSample",
        manifest_path=Path(biosample_manifest_path),
        manifest=biosample_manifest,
        reference=bundle.source_manifests.biosample,
    )
    _validate_manifest_reference(
        role="BioProject",
        manifest_path=Path(bioproject_manifest_path),
        manifest=bioproject_manifest,
        reference=bundle.source_manifests.bioproject,
    )
    _validate_bundle_artifact(
        role="extracted TSV",
        expected_path=Path(extracted_metadata_path),
        reference=bundle.artifacts.extracted_metadata,
    )
    _validate_bundle_artifact(
        role="BioProject context catalog",
        expected_path=bioproject_catalog_path_for(extracted_metadata_path),
        reference=bundle.artifacts.bioproject_context,
    )
    return PairedSourceContract(
        biosample=biosample_manifest,
        bioproject=bioproject_manifest,
    )


def _validate_manifest_reference(
    *,
    role: Literal["BioSample", "BioProject"],
    manifest_path: Path,
    manifest: SourceSnapshotManifest,
    reference: ManifestReference,
) -> None:
    if reference.path != str(manifest_path):
        raise SourceSnapshotError(
            f"Derived {role} source manifest path mismatch: expected "
            f"{manifest_path}, found {reference.path}"
        )
    if reference.snapshot_id != manifest.snapshot_id:
        raise SourceSnapshotError(
            f"Derived {role} snapshot identity mismatch: expected "
            f"{manifest.snapshot_id}, found {reference.snapshot_id}"
        )
    expected_hash = sha256_file(manifest_path)
    if reference.sha256 != expected_hash:
        raise SourceSnapshotError(
            f"Derived {role} source manifest checksum mismatch: expected "
            f"{expected_hash}, found {reference.sha256}"
        )


def _validate_bundle_artifact(
    *, role: str, expected_path: Path, reference: ArtifactReference
) -> None:
    if reference.path != expected_path.name:
        raise SourceSnapshotError(
            f"Derived {role} path mismatch: expected {expected_path.name}, found {reference.path}"
        )
    if not expected_path.is_file():
        raise SourceSnapshotError(f"Derived {role} not found: {expected_path}")
    actual_hash = sha256_file(expected_path)
    if reference.sha256 != actual_hash:
        raise SourceSnapshotError(
            f"Derived {role} checksum mismatch: expected {reference.sha256}, "
            f"calculated {actual_hash}"
        )
