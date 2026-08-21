"""Demonstrate the fast, provenance-valid DatasetBuilder fixture environment."""

from pathlib import Path

from baccurate.run.dataset_builder import DatasetBuilder, DatasetBuildRequest
from baccurate.standardization.host import HostPolicy
from baccurate.standardization_target.specifications import StandardizationTarget


def test_fixture_dataset_builder_standardizes_host_without_production_taxonomy(
    tmp_path: Path,
    extracted_metadata_bundle,
    fixture_dataset_builder: DatasetBuilder,
    fixture_host_policy: HostPolicy,
    fixture_taxon_registry,
    standardization_fixture_resources,
) -> None:
    destination = tmp_path / "standardized.tsv"

    statistics = fixture_dataset_builder.build(
        DatasetBuildRequest(
            extracted_metadata=extracted_metadata_bundle.extracted_metadata,
            biosample_snapshot_manifest=extracted_metadata_bundle.biosample_snapshot_manifest,
            bioproject_snapshot_manifest=extracted_metadata_bundle.bioproject_snapshot_manifest,
            requested_taxa=("ecoli",),
            requested_targets=(StandardizationTarget.HOST,),
            final_destination=destination,
            taxon_registry=fixture_taxon_registry,
            host_policy=fixture_host_policy,
            disable_progress=True,
        )
    )

    assert destination.read_text(encoding="utf-8") == (
        "accession\ttaxon_scientific_name\tncbi_organism\tsylph_species\t"
        "bioproject\t"
        "sra_run_accessions\tgenbank_assembly_accessions\t"
        "refseq_assembly_accessions\t"
        "host_attr_orig\thost_val_orig\thost_taxid\thost_sci_name\t"
        "host_common_names\thost_lineage_names\thost_lineage_taxids\t"
        "host_match_quality_score\thost_needs_review\n"
        "FIXTURE_HUMAN\tEscherichia coli\tEscherichia coli\tEscherichia coli\t"
        "PRJNA1\t\t\t\t"
        "host\thuman\t9606\t"
        "Homo sapiens\thuman\tEukaryota||Metazoa||Homo sapiens\t"
        "2759||33208||9606\t0.95\tFalse\n"
    )
    assert statistics.rows_written == 1
