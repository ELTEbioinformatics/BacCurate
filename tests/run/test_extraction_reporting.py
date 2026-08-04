"""Run-report contracts for performed and reused extraction."""

import json
from datetime import date
from pathlib import Path

from baccurate.extraction import CurationCounters, ExtractionReport
from baccurate.run.outputs import RunOutputs
from baccurate.run.report import RunContext, RunReport


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
            requested_pathogens=("ecoli",),
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
    paired_source_snapshots,
) -> None:
    sources = paired_source_snapshots
    extracted_metadata = tmp_path / "extracted.tsv"
    outputs, run_report = _run_report_for_extraction(
        tmp_path,
        run_name="performed-extraction",
        extracted_metadata=extracted_metadata,
    )
    run_report.begin_performed_extraction(
        biosample_input_path=sources.biosample,
        bioproject_input_path=sources.bioproject,
        extracted_metadata_path=extracted_metadata,
        biosample_manifest_path=sources.biosample_manifest,
        bioproject_manifest_path=sources.bioproject_manifest,
    )
    run_report.record_performed_extraction(
        ExtractionReport(
            prepared_input_paths=(sources.biosample, sources.bioproject),
            extracted_metadata_path=extracted_metadata,
            extracted_record_count=0,
            counters=CurationCounters(),
            automatic_rejection_counts={},
            unreviewed_count=0,
            uncertain_count=0,
            review_worklist_paths={},
            biosample_snapshot_id="biosample-test",
            bioproject_snapshot_id="bioproject-test",
            metadata_reference_date=date(2026, 7, 19),
            bundle_provenance_path=tmp_path / "extracted.provenance.yaml",
        ),
        elapsed_seconds=1.0,
    )

    document = json.loads(outputs.run_report.read_text(encoding="utf-8"))
    extraction = document["extraction"]
    assert extraction["mode"] == "performed"
    assert extraction["prepared_input_paths"] == [str(sources.biosample), str(sources.bioproject)]
    assert extraction["bundle_provenance_path"] == str(tmp_path / "extracted.provenance.yaml")
    assert document["provenance"] == {
        "biosample": {
            "snapshot_id": "biosample-test",
            "metadata_reference_date": "2026-07-19",
        },
        "bioproject": {"snapshot_id": "bioproject-test"},
    }


def test_reused_extraction_report_publishes_mode_and_acquired_snapshot_files(
    tmp_path: Path,
    extracted_metadata_bundle,
) -> None:
    bundle = extracted_metadata_bundle
    outputs, run_report = _run_report_for_extraction(
        tmp_path,
        run_name="reused-extraction",
        extracted_metadata=bundle.extracted_metadata,
    )

    run_report.record_reused_extraction(
        extracted_metadata_path=bundle.extracted_metadata,
        biosample_manifest_path=bundle.biosample_snapshot_manifest,
        bioproject_manifest_path=bundle.bioproject_snapshot_manifest,
    )

    document = json.loads(outputs.run_report.read_text(encoding="utf-8"))
    extraction = document["extraction"]
    assert extraction["mode"] == "reused"
    assert extraction["acquired_snapshot_files"] == ["biosample.xml.gz", "bioproject.xml.gz"]
    assert extraction["bundle_provenance_path"] == str(bundle.provenance)
    assert document["provenance"] == {
        "biosample": {
            "snapshot_id": "fixture-biosample-2026-01-01",
            "metadata_reference_date": "2026-01-01",
        },
        "bioproject": {"snapshot_id": "fixture-bioproject-2026-01-01"},
    }
