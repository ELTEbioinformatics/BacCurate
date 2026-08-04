"""Acquired source snapshot and extracted metadata bundle provenance contracts."""

from datetime import date

import pytest

from baccurate.provenance.source_snapshot import (
    SourceSnapshotError,
    validate_extracted_metadata_bundle,
    validate_paired_source_contract,
)


def test_paired_source_contract_preserves_both_snapshot_identities(
    paired_source_snapshots,
) -> None:
    sources = paired_source_snapshots
    source_contract = validate_paired_source_contract(
        biosample_path=sources.biosample,
        bioproject_path=sources.bioproject,
        biosample_manifest_path=sources.biosample_manifest,
        bioproject_manifest_path=sources.bioproject_manifest,
    )

    assert source_contract.biosample.snapshot_id == "biosample-test"
    assert source_contract.bioproject.snapshot_id == "bioproject-test"
    assert source_contract.metadata_reference_date == date(2026, 7, 19)


@pytest.mark.parametrize("source_role", ["biosample", "bioproject"])
def test_paired_source_contract_rejects_changed_snapshot_for_each_source(
    paired_source_snapshots,
    source_role: str,
) -> None:
    sources = paired_source_snapshots
    source_path = getattr(sources, source_role)
    source_path.write_bytes(source_path.read_bytes() + b"changed compressed bytes")

    role_label = "BioSample" if source_role == "biosample" else "BioProject"
    with pytest.raises(SourceSnapshotError, match=rf"{role_label} source checksum mismatch"):
        validate_paired_source_contract(
            biosample_path=(source_path if source_role == "biosample" else sources.biosample),
            bioproject_path=(source_path if source_role == "bioproject" else sources.bioproject),
            biosample_manifest_path=sources.biosample_manifest,
            bioproject_manifest_path=sources.bioproject_manifest,
        )


def test_extracted_metadata_bundle_rejects_changed_extracted_metadata(
    extracted_metadata_bundle_factory,
) -> None:
    bundle = extracted_metadata_bundle_factory("invalid-artifact")
    artifact = bundle.extracted_metadata
    artifact.write_bytes(artifact.read_bytes() + b"changed")

    with pytest.raises(SourceSnapshotError, match="Derived extracted TSV checksum mismatch"):
        validate_extracted_metadata_bundle(
            bundle.extracted_metadata,
            bundle.biosample_snapshot_manifest,
            bundle.bioproject_snapshot_manifest,
        )


@pytest.mark.parametrize("manifest_role", ["biosample", "bioproject"])
def test_extracted_metadata_bundle_rejects_each_changed_source_manifest(
    extracted_metadata_bundle_factory,
    manifest_role: str,
) -> None:
    bundle = extracted_metadata_bundle_factory("changed-manifest")
    changed_manifest = getattr(bundle, f"{manifest_role}_snapshot_manifest")
    changed_manifest.write_text(
        changed_manifest.read_text(encoding="utf-8") + "# changed after extraction\n",
        encoding="utf-8",
    )

    role_label = "BioSample" if manifest_role == "biosample" else "BioProject"
    with pytest.raises(
        SourceSnapshotError,
        match=f"Derived {role_label} source manifest mismatch",
    ):
        validate_extracted_metadata_bundle(
            bundle.extracted_metadata,
            bundle.biosample_snapshot_manifest,
            bundle.bioproject_snapshot_manifest,
        )
