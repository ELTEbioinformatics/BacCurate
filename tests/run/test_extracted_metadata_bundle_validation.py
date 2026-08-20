"""Extracted metadata bundle validation at the dataset streaming boundary."""

from pathlib import Path

import pytest

from baccurate.provenance.source_snapshot import SourceSnapshotError
from baccurate.run.dataset_builder import DatasetBuilder, DatasetBuildRequest
from baccurate.standardization_target.specifications import StandardizationTarget


def test_dataset_builder_rejects_changed_extracted_metadata_before_streaming(
    tmp_path: Path,
    extracted_metadata_bundle,
    fixture_dataset_builder: DatasetBuilder,
    fixture_pathogen_registry,
    standardization_fixture_resources,
) -> None:
    bundle = extracted_metadata_bundle
    bundle.extracted_metadata.write_text("invalid extracted metadata\n", encoding="utf-8")

    with pytest.raises(SourceSnapshotError, match="Derived extracted TSV checksum mismatch"):
        fixture_dataset_builder.build(
            DatasetBuildRequest(
                extracted_metadata=bundle.extracted_metadata,
                biosample_snapshot_manifest=bundle.biosample_snapshot_manifest,
                bioproject_snapshot_manifest=bundle.bioproject_snapshot_manifest,
                requested_pathogens=("ecoli",),
                requested_targets=(StandardizationTarget.DATE,),
                final_destination=tmp_path / "standardized.tsv",
                pathogen_registry=fixture_pathogen_registry,
                disable_progress=True,
            )
        )
