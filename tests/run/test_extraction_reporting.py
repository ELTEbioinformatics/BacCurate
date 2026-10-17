"""Run-report contracts for performed and reused extraction."""

import json
from datetime import date
from pathlib import Path

import yaml

from baccurate.extraction import ExtractionReport, SelectionCounters
from baccurate.run.outputs import RunOutputs
from baccurate.run.report import (
    SEQUENCE_ACCESSION_MANIFEST_PATHS,
    RunContext,
    RunReport,
)


def _expected_sequence_accession_provenance() -> dict[str, dict[str, str]]:
    """Load expected snapshot provenance from raw YAML, bypassing the report code path."""
    expected = {}
    for name, path in SEQUENCE_ACCESSION_MANIFEST_PATHS.items():
        manifest = yaml.safe_load(path.read_text(encoding="utf-8"))
        expected[name] = {
            "snapshot_id": manifest["snapshot_id"],
            # Mirror the fallback in SnapshotManifest.metadata_reference_date.
            "metadata_reference_date": str(manifest["snapshot_as_of"] or manifest["retrieved_on"]),
        }
    return expected


def _run_report_for_extraction(
    tmp_path: Path,
    *,
    run_name: str,
    extracted_metadata: Path,
) -> tuple[RunOutputs, RunReport]:
    outputs = RunOutputs.plan(
        output_dir=tmp_path,
        run_name=run_name,
        output_file=None,
        include_isolation_source=False,
    )
    outputs.initialize()
    return outputs, RunReport(
        outputs,
        RunContext(
            requested_taxa=("ecoli",),
            requested_standardization_targets=("date",),
            extracted_metadata=extracted_metadata,
            options={},
            configuration_paths=(),
            skip_llm=True,
            model_identifiers={},
        ),
    )


def test_performed_extraction_report_publishes_mode_paths_and_paired_provenance(
    tmp_path: Path,
    paired_snapshots,
) -> None:
    snapshots = paired_snapshots
    extracted_metadata = tmp_path / "extracted.tsv"
    outputs, run_report = _run_report_for_extraction(
        tmp_path,
        run_name="performed-extraction",
        extracted_metadata=extracted_metadata,
    )
    run_report.begin_performed_extraction(
        biosample_input_path=snapshots.biosample_snapshot_path,
        bioproject_input_path=snapshots.bioproject_snapshot_path,
        extracted_metadata_path=extracted_metadata,
        biosample_manifest_path=snapshots.biosample_manifest_path,
        bioproject_manifest_path=snapshots.bioproject_manifest_path,
    )
    run_report.record_performed_extraction(
        ExtractionReport(
            prepared_input_paths=(
                snapshots.biosample_snapshot_path,
                snapshots.bioproject_snapshot_path,
            ),
            extracted_metadata_path=extracted_metadata,
            extracted_record_count=0,
            counters=SelectionCounters(),
            automatic_rejection_counts={},
            unreviewed_count=0,
            uncertain_count=0,
            review_worklist_paths={},
            biosample_snapshot_id="biosample-test",
            bioproject_snapshot_id="bioproject-test",
            metadata_reference_date=date(2026, 7, 19),
            extracted_metadata_provenance_path=tmp_path / "extracted.provenance.yaml",
            inclusion_route_counts={
                "biosample_taxonomy": 10,
                "allthebacteria": 2,
            },
        ),
        elapsed_seconds=1.0,
    )

    document = json.loads(outputs.run_report.read_text(encoding="utf-8"))
    extraction = document["extraction"]
    assert document["schema_version"] == 9
    assert extraction["mode"] == "performed"
    assert extraction["prepared_input_paths"] == [
        str(snapshots.biosample_snapshot_path),
        str(snapshots.bioproject_snapshot_path),
    ]
    assert extraction["extracted_metadata_provenance_path"] == str(
        tmp_path / "extracted.provenance.yaml"
    )
    assert extraction["inclusion_route_counts"] == {
        "biosample_taxonomy": 10,
        "allthebacteria": 2,
    }
    assert document["provenance"] == _expected_sequence_accession_provenance() | {
        "biosample": {
            "snapshot_id": "biosample-test",
            "metadata_reference_date": "2026-07-19",
        },
        "bioproject": {"snapshot_id": "bioproject-test"},
    }


def test_reused_extraction_report_publishes_mode_and_acquired_snapshot_files(
    tmp_path: Path,
    extracted_metadata_files,
) -> None:
    files = extracted_metadata_files
    outputs, run_report = _run_report_for_extraction(
        tmp_path,
        run_name="reused-extraction",
        extracted_metadata=files.extracted_metadata_path,
    )

    run_report.record_reused_extraction(
        extracted_metadata_path=files.extracted_metadata_path,
        biosample_manifest_path=files.biosample_manifest_path,
        bioproject_manifest_path=files.bioproject_manifest_path,
    )

    document = json.loads(outputs.run_report.read_text(encoding="utf-8"))
    extraction = document["extraction"]
    assert extraction["mode"] == "reused"
    assert extraction["acquired_snapshot_files"] == ["biosample.xml.gz", "bioproject.xml.gz"]
    assert extraction["extracted_metadata_provenance_path"] == str(files.provenance_path)
    assert "inclusion_route_counts" not in extraction
    assert document["provenance"] == _expected_sequence_accession_provenance() | {
        "biosample": {
            "snapshot_id": "fixture-biosample-2026-01-01",
            "metadata_reference_date": "2026-01-01",
        },
        "bioproject": {"snapshot_id": "fixture-bioproject-2026-01-01"},
    }
