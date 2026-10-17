"""Snapshot pair and extracted metadata bundle provenance contracts."""

from datetime import date

import pytest

from baccurate.provenance.snapshot_manifest import (
    SnapshotError,
    validate_extracted_metadata_bundle,
    validate_snapshot_pair,
)


def test_snapshot_pair_preserves_both_snapshot_identities(
    paired_snapshots,
) -> None:
    snapshots = paired_snapshots
    snapshot_pair = validate_snapshot_pair(
        biosample_snapshot_path=snapshots.biosample_snapshot_path,
        bioproject_snapshot_path=snapshots.bioproject_snapshot_path,
        biosample_manifest_path=snapshots.biosample_manifest_path,
        bioproject_manifest_path=snapshots.bioproject_manifest_path,
    )

    assert snapshot_pair.biosample.snapshot_id == "biosample-test"
    assert snapshot_pair.bioproject.snapshot_id == "bioproject-test"
    assert snapshot_pair.metadata_reference_date == date(2026, 7, 19)


@pytest.mark.parametrize("database", ["biosample", "bioproject"])
def test_snapshot_pair_rejects_changed_snapshot_for_each_database(
    paired_snapshots,
    database: str,
) -> None:
    snapshots = paired_snapshots
    changed_snapshot_path = getattr(snapshots, f"{database}_snapshot_path")
    changed_snapshot_path.write_bytes(
        changed_snapshot_path.read_bytes() + b"changed compressed bytes"
    )

    role_label = "BioSample" if database == "biosample" else "BioProject"
    with pytest.raises(SnapshotError, match=rf"{role_label} snapshot checksum mismatch"):
        validate_snapshot_pair(
            biosample_snapshot_path=(
                changed_snapshot_path
                if database == "biosample"
                else snapshots.biosample_snapshot_path
            ),
            bioproject_snapshot_path=(
                changed_snapshot_path
                if database == "bioproject"
                else snapshots.bioproject_snapshot_path
            ),
            biosample_manifest_path=snapshots.biosample_manifest_path,
            bioproject_manifest_path=snapshots.bioproject_manifest_path,
        )


def test_extracted_metadata_bundle_rejects_changed_extracted_metadata(
    extracted_metadata_bundle_factory,
) -> None:
    bundle = extracted_metadata_bundle_factory("invalid-artifact")
    artifact = bundle.extracted_metadata
    artifact.write_bytes(artifact.read_bytes() + b"changed")

    with pytest.raises(SnapshotError, match="Derived extracted TSV checksum mismatch"):
        validate_extracted_metadata_bundle(
            bundle.extracted_metadata,
            bundle.biosample_manifest_path,
            bundle.bioproject_manifest_path,
        )


@pytest.mark.parametrize("database", ["biosample", "bioproject"])
def test_extracted_metadata_bundle_rejects_each_changed_manifest(
    extracted_metadata_bundle_factory,
    database: str,
) -> None:
    bundle = extracted_metadata_bundle_factory("changed-manifest")
    changed_manifest_path = getattr(bundle, f"{database}_manifest_path")
    changed_manifest_path.write_text(
        changed_manifest_path.read_text(encoding="utf-8") + "# changed after extraction\n",
        encoding="utf-8",
    )

    role_label = "BioSample" if database == "biosample" else "BioProject"
    with pytest.raises(
        SnapshotError,
        match=f"Derived {role_label} snapshot manifest mismatch",
    ):
        validate_extracted_metadata_bundle(
            bundle.extracted_metadata,
            bundle.biosample_manifest_path,
            bundle.bioproject_manifest_path,
        )
