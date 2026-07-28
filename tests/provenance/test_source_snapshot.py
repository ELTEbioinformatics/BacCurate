"""Acquired source snapshot and extracted metadata bundle provenance contracts."""

import gzip
from datetime import date

import pytest
import yaml

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
@pytest.mark.parametrize("failure", ["missing", "unexpected", "checksum-mismatch"])
def test_paired_source_contract_rejects_invalid_snapshot_for_each_source(
    paired_source_snapshots,
    source_role: str,
    failure: str,
) -> None:
    sources = paired_source_snapshots
    source_path = getattr(sources, source_role)
    expected_detail = {
        "missing": "missing",
        "unexpected": "unexpected",
        "checksum-mismatch": "checksum mismatch",
    }[failure]

    if failure == "missing":
        source_path.unlink()
    elif failure == "unexpected":
        unexpected = source_path.with_name("unexpected.xml.gz")
        unexpected.write_bytes(gzip.compress(b"<Unexpected />", mtime=0))
        source_path = unexpected
    else:
        source_path.write_bytes(source_path.read_bytes() + b"changed compressed bytes")

    role_label = "BioSample" if source_role == "biosample" else "BioProject"
    with pytest.raises(SourceSnapshotError, match=rf"{role_label}.*{expected_detail}"):
        validate_paired_source_contract(
            biosample_path=(source_path if source_role == "biosample" else sources.biosample),
            bioproject_path=(source_path if source_role == "bioproject" else sources.bioproject),
            biosample_manifest_path=sources.biosample_manifest,
            bioproject_manifest_path=sources.bioproject_manifest,
        )


@pytest.mark.parametrize("member", ["extracted_metadata", "bioproject_context"])
@pytest.mark.parametrize("failure", ["missing", "changed", "mismatched"])
def test_extracted_metadata_bundle_rejects_each_invalid_artifact(
    extracted_metadata_bundle_factory,
    member: str,
    failure: str,
) -> None:
    bundle = extracted_metadata_bundle_factory("invalid-artifact")
    artifact = getattr(bundle, member)
    role = "extracted TSV" if member == "extracted_metadata" else "BioProject context catalog"

    if failure == "missing":
        artifact.unlink()
        detail = "not found"
    elif failure == "changed":
        artifact.write_bytes(artifact.read_bytes() + b"changed")
        detail = "checksum mismatch"
    else:
        provenance_record = yaml.safe_load(bundle.provenance.read_text(encoding="utf-8"))
        provenance_record["artifacts"][member]["path"] = f"wrong-{artifact.name}"
        bundle.provenance.write_text(
            yaml.safe_dump(provenance_record, sort_keys=False),
            encoding="utf-8",
        )
        detail = "path mismatch"

    with pytest.raises(SourceSnapshotError, match=f"Derived {role} {detail}"):
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
        match=f"Derived {role_label} source manifest checksum mismatch",
    ):
        validate_extracted_metadata_bundle(
            bundle.extracted_metadata,
            bundle.biosample_snapshot_manifest,
            bundle.bioproject_snapshot_manifest,
        )
