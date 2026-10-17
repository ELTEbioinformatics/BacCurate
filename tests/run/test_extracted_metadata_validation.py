"""Extracted metadata validation at the dataset streaming boundary."""

from pathlib import Path

import pytest

from baccurate.provenance.extracted_metadata import ExtractedMetadataError
from baccurate.run.dataset_builder import DatasetBuilder, DatasetBuildRequest
from baccurate.standardization_target.specifications import StandardizationTarget


def test_dataset_builder_rejects_changed_extracted_metadata_before_streaming(
    tmp_path: Path,
    extracted_metadata_files,
    fixture_dataset_builder: DatasetBuilder,
    fixture_taxon_registry,
    standardization_fixture_resources,
) -> None:
    files = extracted_metadata_files
    files.extracted_metadata_path.write_text("invalid extracted metadata\n", encoding="utf-8")

    with pytest.raises(ExtractedMetadataError, match="Extracted metadata checksum mismatch"):
        fixture_dataset_builder.build(
            DatasetBuildRequest(
                extracted_metadata=files.extracted_metadata_path,
                biosample_manifest_path=files.biosample_manifest_path,
                bioproject_manifest_path=files.bioproject_manifest_path,
                requested_taxa=("ecoli",),
                requested_targets=(StandardizationTarget.DATE,),
                final_destination=tmp_path / "standardized.tsv",
                taxon_registry=fixture_taxon_registry,
                disable_progress=True,
            )
        )
