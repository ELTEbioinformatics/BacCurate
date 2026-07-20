import hashlib
import json
from pathlib import Path

import pytest

from baccurate.prompt_snapshot import write_prompt_snapshot
from baccurate.run_outputs import RunContext, RunDiagnostics, RunOutputs


def _write_prompt_configs(tmp_path: Path) -> tuple[Path, Path]:
    ontology = tmp_path / "ontology.tsv"
    ontology.write_text(
        "term\tdisplay_term\tontology_link\tcrosslink_term\tsynonyms\tcomment\texamples\n"
        "environmental\tenvironmental\t\t\t\t\t\n"
        "unspecified\tunspecified\t\t\t\tNo source named.\t\n",
        encoding="utf-8",
    )
    location = tmp_path / "location.yaml"
    location.write_text(
        "prompt_version: location-v2\n"
        "llm_system_prompt: |\n"
        "  Locate café.\n"
        "llm_user_prompt_template: |-\n"
        "  {attr_val_pairs}\n",
        encoding="utf-8",
    )
    isolation = tmp_path / "isolation.yaml"
    isolation.write_text(
        f'ontology_tsv_path: "{ontology.as_posix()}"\n'
        "prompt_version: isolation-v3\n"
        "system_prompt: |-\n"
        "  Classify using:\n"
        "  {ontology_tree}\n"
        "user_prompt: |-\n"
        "  Sample:\n"
        "  {metadata}\n"
        "  {bioproject_context}\n"
        "bioproject_system_prompt: |-\n"
        "  Project rules.\n"
        "bioproject_user_prompt: |-\n"
        "  Projects: {bioproject_context}\n",
        encoding="utf-8",
    )
    return location, isolation


def test_prompt_snapshot_omits_unselected_pipeline_and_optional_version(tmp_path: Path) -> None:
    location, _ = _write_prompt_configs(tmp_path)
    location.write_text(
        "llm_system_prompt: system\nllm_user_prompt_template: user {attr_val_pairs}\n",
        encoding="utf-8",
    )
    destination = tmp_path / "prompts.txt"

    write_prompt_snapshot(
        destination,
        model_identifiers={"location": "model"},
        location_config=location,
    )

    contents = destination.read_text(encoding="utf-8")
    assert "[location]" in contents
    assert "prompt_version:" not in contents
    assert "[isolation]" not in contents


@pytest.mark.parametrize("include_prompt_snapshot", [False, True])
def test_prompt_snapshot_participates_in_output_planning_only_when_enabled(
    tmp_path: Path, include_prompt_snapshot: bool
) -> None:
    outputs = RunOutputs.plan(
        output_dir=tmp_path,
        run_name="run",
        output_file=None,
        include_isolation=False,
        include_prompt_snapshot=include_prompt_snapshot,
    )

    expected = tmp_path / "run" / "prompts.txt" if include_prompt_snapshot else None
    assert outputs.prompt_snapshot == expected
    assert (expected in outputs.paths()) is include_prompt_snapshot
    if expected is not None:
        expected.parent.mkdir(parents=True)
        expected.write_text("existing", encoding="utf-8")
        assert outputs.collision() == expected


def test_prompt_snapshot_cannot_alias_an_explicit_dataset_path(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="aliases another run output"):
        RunOutputs.plan(
            output_dir=tmp_path,
            run_name="unused",
            output_file=tmp_path / "prompts.txt",
            include_isolation=False,
            include_prompt_snapshot=True,
        )


def test_diagnostics_records_prompt_snapshot_path_and_hash(tmp_path: Path) -> None:
    outputs = RunOutputs.plan(
        output_dir=tmp_path,
        run_name="run",
        output_file=None,
        include_isolation=False,
        include_prompt_snapshot=True,
    )
    outputs.initialize()
    assert outputs.prompt_snapshot is not None
    outputs.prompt_snapshot.write_text("effective prompts\n", encoding="utf-8")

    RunDiagnostics(outputs, _run_context(tmp_path))

    diagnostics = json.loads(outputs.diagnostics.read_text(encoding="utf-8"))
    assert diagnostics["prompt_artifact"] == {
        "path": str(outputs.prompt_snapshot),
        "sha256": hashlib.sha256(outputs.prompt_snapshot.read_bytes()).hexdigest(),
    }


def test_diagnostics_omits_prompt_artifact_when_snapshot_is_not_planned(tmp_path: Path) -> None:
    outputs = RunOutputs.plan(
        output_dir=tmp_path,
        run_name="run",
        output_file=None,
        include_isolation=False,
        include_prompt_snapshot=False,
    )
    outputs.initialize()

    RunDiagnostics(outputs, _run_context(tmp_path))

    diagnostics = json.loads(outputs.diagnostics.read_text(encoding="utf-8"))
    assert outputs.prompt_snapshot is None
    assert "prompt_artifact" not in diagnostics


def _run_context(tmp_path: Path) -> RunContext:
    return RunContext(
        requested_pathogens=("ecoli",),
        requested_attributes=("date",),
        extracted_metadata=tmp_path / "extracted.tsv",
        options={},
        configuration_paths=(),
        skip_llm=True,
        model_identifiers={},
    )
