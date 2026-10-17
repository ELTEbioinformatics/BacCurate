"""Snapshot pair contracts."""

from datetime import date

import pytest

from baccurate.provenance.snapshot_manifest import SnapshotError, validate_snapshot_pair


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
