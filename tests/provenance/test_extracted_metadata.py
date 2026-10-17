"""Extracted metadata provenance contracts."""

import pytest

from baccurate.provenance.extracted_metadata import (
    ExtractedMetadataError,
    validate_extracted_metadata,
)


def test_extracted_metadata_rejects_changed_extracted_metadata(
    make_extracted_metadata_files,
) -> None:
    files = make_extracted_metadata_files("invalid-artifact")
    artifact = files.extracted_metadata_path
    artifact.write_bytes(artifact.read_bytes() + b"changed")

    with pytest.raises(ExtractedMetadataError, match="Extracted metadata checksum mismatch"):
        validate_extracted_metadata(
            files.extracted_metadata_path,
            files.biosample_manifest_path,
            files.bioproject_manifest_path,
        )


@pytest.mark.parametrize("database", ["biosample", "bioproject"])
def test_extracted_metadata_rejects_each_changed_manifest(
    make_extracted_metadata_files,
    database: str,
) -> None:
    files = make_extracted_metadata_files("changed-manifest")
    changed_manifest_path = getattr(files, f"{database}_manifest_path")
    changed_manifest_path.write_text(
        changed_manifest_path.read_text(encoding="utf-8") + "# changed after extraction\n",
        encoding="utf-8",
    )

    role_label = "BioSample" if database == "biosample" else "BioProject"
    with pytest.raises(
        ExtractedMetadataError,
        match=f"{role_label} manifest checksum mismatch",
    ):
        validate_extracted_metadata(
            files.extracted_metadata_path,
            files.biosample_manifest_path,
            files.bioproject_manifest_path,
        )
