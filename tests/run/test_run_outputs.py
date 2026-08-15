import hashlib
import json
from pathlib import Path

import pytest

from baccurate.run.location_review_worklist import LocationReviewWorklistSummary
from baccurate.run.outputs import RunOutputs
from baccurate.run.prompt_snapshot import write_prompt_snapshot
from baccurate.run.report import RunContext, RunReport, RunStatus
from baccurate.run.statistics import (
    DatasetBuildProgress,
    DatasetBuildStatistics,
    DateBuildStatistics,
    DateStatistics,
    HostBuildStatistics,
    HostStatistics,
    InventedLabelStatistics,
    IsolationSourceBuildStatistics,
    IsolationSourceStatistics,
    LocationBuildStatistics,
    LocationStatistics,
)
from baccurate.standardization.collection_date import (
    DateCategory,
    DatePrecision,
    DateStructure,
)
from baccurate.standardization.isolation_source import (
    IsolationSourcePromptPolicy,
)

ROOT = Path(__file__).parents[2]


def _write_prompt_configs(tmp_path: Path) -> IsolationSourcePromptPolicy:
    isolation_source = tmp_path / "isolation.yaml"
    isolation_source.write_text(
        "schema_version: 3\n"
        f'ontology_directory: "{(ROOT / "tests" / "fixtures" / "standardization" / "ontology").as_posix()}"\n'
        "prompt_version: isolation-v3\n"
        "system_prompt: |-\n"
        "  Classify using:\n"
        "  {ontology_tree}\n"
        "user_prompt: |-\n"
        "  Sample:\n"
        "  {metadata}\n",
        encoding="utf-8",
    )
    return IsolationSourcePromptPolicy.load(isolation_source)


def test_prompt_snapshot_omits_unselected_pipeline(tmp_path: Path) -> None:
    destination = tmp_path / "prompts.txt"

    write_prompt_snapshot(destination, model_identifiers={})

    contents = destination.read_text(encoding="utf-8")
    assert "[location]" not in contents
    assert "[isolation_source]" not in contents


def test_prompt_snapshot_uses_loaded_effective_isolation_source_prompts(tmp_path: Path) -> None:
    isolation_source_policy = _write_prompt_configs(tmp_path)
    source = tmp_path / "isolation.yaml"
    source.write_text("broken: [\n", encoding="utf-8")
    destination = tmp_path / "prompts.txt"

    write_prompt_snapshot(
        destination,
        model_identifiers={"isolation_source": "model"},
        isolation_source_prompt_policy=isolation_source_policy,
    )

    contents = destination.read_text(encoding="utf-8")
    assert "[isolation_source]" in contents
    assert "model_identifier: model" in contents
    assert "prompt_version: isolation-v3" in contents
    assert "{ontology_tree}" not in contents
    assert "environmental" in contents


@pytest.mark.parametrize("include_prompt_snapshot", [False, True])
def test_prompt_snapshot_participates_in_output_planning_only_when_enabled(
    tmp_path: Path, include_prompt_snapshot: bool
) -> None:
    outputs = RunOutputs.plan(
        output_dir=tmp_path,
        run_name="run",
        output_file=None,
        include_isolation_source=False,
        include_prompt_snapshot=include_prompt_snapshot,
    )

    expected = tmp_path / "run" / "prompts.txt" if include_prompt_snapshot else None
    assert outputs.prompt_snapshot == expected
    assert (expected in outputs.paths()) is include_prompt_snapshot
    if expected is not None:
        expected.parent.mkdir(parents=True)
        expected.write_text("existing", encoding="utf-8")
        assert outputs.collision() == expected

def test_run_report_uses_published_output_name_and_participates_in_collision_detection(
    tmp_path: Path,
) -> None:
    outputs = RunOutputs.plan(
        output_dir=tmp_path,
        run_name="run",
        output_file=None,
        include_isolation_source=False,
    )

    expected = tmp_path / "run" / "run_report.json"
    assert outputs.run_report == expected
    assert expected in outputs.paths()
    expected.parent.mkdir(parents=True)
    expected.write_text("existing", encoding="utf-8")
    assert outputs.collision() == expected


def test_isolation_source_reasoning_uses_published_output_name(tmp_path: Path) -> None:
    outputs = RunOutputs.plan(
        output_dir=tmp_path,
        run_name="run",
        output_file=None,
        include_isolation_source=True,
        include_prompt_snapshot=False,
    )
    outputs.initialize()

    RunReport(outputs, _run_context(tmp_path))

    run_report = json.loads(outputs.run_report.read_text(encoding="utf-8"))
    assert outputs.isolation_source_reasoning == (
        tmp_path / "run" / "isolation_source_reasoning.jsonl"
    )
    assert run_report["outputs"]["isolation_source_reasoning"] == str(
        outputs.isolation_source_reasoning
    )


def test_prompt_snapshot_cannot_alias_an_explicit_dataset_path(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="aliases another run output"):
        RunOutputs.plan(
            output_dir=tmp_path,
            run_name="unused",
            output_file=tmp_path / "prompts.txt",
            include_isolation_source=False,
            include_prompt_snapshot=True,
        )


def test_run_report_cannot_alias_an_explicit_dataset_path(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="aliases another run output"):
        RunOutputs.plan(
            output_dir=tmp_path,
            run_name="unused",
            output_file=tmp_path / "run_report.json",
            include_isolation_source=False,
        )


def test_run_report_records_prompt_snapshot_path_and_hash(tmp_path: Path) -> None:
    outputs = RunOutputs.plan(
        output_dir=tmp_path,
        run_name="run",
        output_file=None,
        include_isolation_source=False,
        include_prompt_snapshot=True,
    )
    outputs.initialize()
    assert outputs.prompt_snapshot is not None
    outputs.prompt_snapshot.write_text("effective prompts\n", encoding="utf-8")

    RunReport(outputs, _run_context(tmp_path))

    run_report = json.loads(outputs.run_report.read_text(encoding="utf-8"))
    assert set(run_report["outputs"]) == {
        "dataset",
        "log",
        "run_report",
        "extracted_input",
        "isolation_source_reasoning",
        "prompt_snapshot",
    }
    assert run_report["outputs"]["run_report"] == str(outputs.run_report)
    assert run_report["outputs"]["prompt_snapshot"] == str(outputs.prompt_snapshot)
    assert run_report["prompt_artifact"] == {
        "path": str(outputs.prompt_snapshot),
        "sha256": hashlib.sha256(outputs.prompt_snapshot.read_bytes()).hexdigest(),
    }


def test_run_report_omits_prompt_artifact_when_snapshot_is_not_planned(tmp_path: Path) -> None:
    outputs = RunOutputs.plan(
        output_dir=tmp_path,
        run_name="run",
        output_file=None,
        include_isolation_source=False,
        include_prompt_snapshot=False,
    )
    outputs.initialize()

    RunReport(outputs, _run_context(tmp_path))

    run_report = json.loads(outputs.run_report.read_text(encoding="utf-8"))
    assert outputs.prompt_snapshot is None
    assert run_report["outputs"]["prompt_snapshot"] is None
    assert "prompt_artifact" not in run_report


def test_run_report_publishes_the_full_scientific_target_key_set(tmp_path: Path) -> None:
    outputs = RunOutputs.plan(
        output_dir=tmp_path,
        run_name="run",
        output_file=None,
        include_isolation_source=False,
        include_prompt_snapshot=False,
    )
    outputs.initialize()
    statistics = _complete_build_statistics(tmp_path)
    run_report_writer = RunReport(outputs, _run_context(tmp_path))

    run_report_writer.finish(
        RunStatus.SUCCEEDED,
        progress=DatasetBuildProgress(processed_rows=1, rows_written=1, statistics=statistics),
        statistics=statistics,
    )

    run_report = json.loads(outputs.run_report.read_text(encoding="utf-8"))
    assert set(run_report["scientific"]) == {
        "date",
        "location",
        "host",
        "isolation_source",
    }
    assert run_report["scientific"]["isolation_source"]["aggregate"]["invented_labels"] == {
        "body_site": {
            "nasal cavity": {
                "occurrences": 2,
                "accessions": {"SAMN00000001": 2},
            }
        }
    }
    assert run_report["scientific"]["location"]["review_worklist"] == {
        "path": str(tmp_path / "run" / "location_review_worklist.tsv"),
        "row_count": 1,
        "occurrence_count": 2,
        "biosample_record_count": 2,
    }
    assert run_report["scientific"]["date"]["aggregate"] == {
        "processed": 1,
        "standardized": 1,
        "rejected": 0,
        "categories": {"sample_collection": 1},
        "structures": {"single_value": 1},
        "precisions": {"day": 1},
        "derivations": {"direct": 1},
        "diagnostics": {},
        "parsed_date_rejections": {},
        "notices": {},
    }


def _complete_build_statistics(tmp_path: Path) -> DatasetBuildStatistics:
    date_statistics = DateStatistics(
        processed=1,
        standardized=1,
        rejected=0,
        categories={DateCategory.SAMPLE_COLLECTION: 1},
        structures={DateStructure.SINGLE_VALUE: 1},
        precisions={DatePrecision.DAY: 1},
        derivations={"direct": 1},
        diagnostics={},
        parsed_date_rejections={},
        notices={},
    )
    location_statistics = LocationStatistics(
        processed=1,
        standardized=1,
        rejected=0,
        coordinate_decodes=0,
        direct_matches=1,
        reviewed_mapping_matches=0,
        diagnostics={},
    )
    host_statistics = HostStatistics(
        processed=1,
        standardized=1,
        rejected=0,
        overflow=0,
        needs_review=0,
        host_recovery_passes=0,
        diagnostics={},
    )
    isolation_source_statistics = IsolationSourceStatistics(
        processed=1,
        standardized=1,
        rejected=0,
        exact_matches=0,
        cache_hits=0,
        llm_calls=0,
        host_recovery_passes=0,
        evidence_levels={},
        diagnostics={},
        invented_labels={
            "body_site": {
                "nasal cavity": InventedLabelStatistics(
                    occurrences=2,
                    accessions={"SAMN00000001": 2},
                )
            }
        },
    )
    return DatasetBuildStatistics(
        final_destination=tmp_path / "run" / "run.tsv",
        rows_written=1,
        date=DateBuildStatistics(
            aggregate=date_statistics,
            by_pathogen={"ecoli": date_statistics},
        ),
        location=LocationBuildStatistics(
            aggregate=location_statistics,
            by_pathogen={"ecoli": location_statistics},
            review_worklist=LocationReviewWorklistSummary(
                path=tmp_path / "run" / "location_review_worklist.tsv",
                row_count=1,
                occurrence_count=2,
                biosample_record_count=2,
            ),
        ),
        host=HostBuildStatistics(
            aggregate=host_statistics,
            by_pathogen={"ecoli": host_statistics},
        ),
        isolation_source=IsolationSourceBuildStatistics(
            aggregate=isolation_source_statistics,
            by_pathogen={"ecoli": isolation_source_statistics},
        ),
    )


def _run_context(tmp_path: Path) -> RunContext:
    return RunContext(
        requested_pathogens=("ecoli",),
        requested_standardization_targets=("date",),
        extracted_metadata=tmp_path / "extracted.tsv",
        options={},
        configuration_paths=(),
        skip_llm=True,
        model_identifiers={},
    )
