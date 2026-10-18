"""Command-line contract tests for resolving and executing a BacCurate run."""

import csv
import json
import shutil
from dataclasses import asdict
from pathlib import Path

import pytest

import baccurate.run.invocation as invocation_module
import baccurate.run.main as main_module
from baccurate.adapters.policy_yaml import PolicyConfigurationError
from baccurate.extraction import COLUMNS, SelectionPolicyError
from baccurate.provenance.extracted_metadata import (
    ExtractedMetadataProvenance,
    provenance_path_for,
)
from baccurate.provenance.sha256 import sha256_file
from baccurate.provenance.snapshot_manifest import SnapshotManifest
from baccurate.run.main import main as pipeline_cli
from baccurate.standardization.isolation_source import IsolationSourcePromptPolicy
from baccurate.taxon_registry.registry import Taxon, TaxonRegistry


def _fail_if_run_resources_started(*_args: object, **_kwargs: object) -> None:
    pytest.fail("invalid selected policy reached run resource initialization")


def _guard_run_resources(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(invocation_module.RunOutputs, "initialize", _fail_if_run_resources_started)
    monkeypatch.setattr(main_module, "run_extraction", _fail_if_run_resources_started)
    monkeypatch.setattr(main_module.DatasetBuilder, "build", _fail_if_run_resources_started)


def _example_registry() -> TaxonRegistry:
    return TaxonRegistry(
        schema_version=1,
        included_taxa={
            "zeta": Taxon("zeta", "Zeta example", 30, "species"),
            "alpha": Taxon("alpha", "Alpha", 10, "genus"),
        },
        containers={"examples": ("alpha", "zeta")},
    )


def test_pipeline_cli_discovers_taxa_in_supplied_registry_order(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    index = tmp_path / "index.tsv"
    index.write_text(
        "accession\ttaxon_biosample\tsylph_species\n"
        "SAMN1\talpha\tNA\n"
        "SAMN2\tnot-targeted\tNA\n"
        "SAMN3\tNA\tZeta example\n"
        "SAMN4\talpha\tNA\n",
        encoding="utf-8",
    )
    extracted_metadata = _prepare_empty_extracted_metadata(tmp_path, monkeypatch)
    monkeypatch.setattr(invocation_module, "DEFAULT_INDEX_TSV", index)

    pipeline_cli(
        [
            "--standardize",
            "date",
            "--extracted-metadata",
            str(extracted_metadata),
            "--output-dir",
            str(tmp_path / "runs"),
            "--run-name",
            "registry-order",
            "--skip-llm",
            "--quiet",
        ],
        taxon_registry=_example_registry(),
    )

    run_report = json.loads(
        (tmp_path / "runs" / "registry-order" / "run_report.json").read_text(encoding="utf-8")
    )
    assert run_report["request"]["taxa"] == ["zeta", "alpha"]


def test_pipeline_output_label_uses_supplied_registry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    extracted_metadata = _prepare_empty_extracted_metadata(
        tmp_path,
        monkeypatch,
        records=[
            {
                "accession": "SAMN1",
                "taxon_key": "zeta",
                "date_category": "c",
                "date_attr_orig": "collection_date",
                "date_val_orig": "2020",
            }
        ],
    )

    pipeline_cli(
        [
            "zeta",
            "--standardize",
            "date",
            "--extracted-metadata",
            str(extracted_metadata),
            "--output-dir",
            str(tmp_path / "runs"),
            "--run-name",
            "supplied-registry",
            "--skip-llm",
            "--quiet",
        ],
        taxon_registry=_example_registry(),
    )

    output_path = tmp_path / "runs" / "supplied-registry" / "supplied-registry.tsv"
    with output_path.open(encoding="utf-8", newline="") as stream:
        row = next(csv.DictReader(stream, delimiter="\t"))
    assert row["taxon"] == "Zeta example"


def test_pipeline_cli_uses_custom_extracted_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    extracted_metadata = _prepare_empty_extracted_metadata(tmp_path, monkeypatch)

    pipeline_cli(
        [
            "ecoli",
            "--standardize",
            "date",
            "--extracted-metadata",
            str(extracted_metadata),
            "--output-dir",
            str(tmp_path / "runs"),
            "--run-name",
            "custom-files",
            "--skip-llm",
            "--quiet",
        ]
    )

    run_dir = tmp_path / "runs" / "custom-files"
    assert (run_dir / "custom-files.tsv").exists()
    run_report = json.loads((run_dir / "run_report.json").read_text(encoding="utf-8"))
    assert run_report["outputs"]["extracted_input"] == str(extracted_metadata)
    options = run_report["runtime"]["options"]
    assert options["standardization_targets"] == ["date"]


def test_pipeline_cli_validates_required_selection_before_output_initialization(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    selection_path = config_dir / "selection.yaml"
    selection_path.write_text(
        "schema_version: 4\ntargets:\n  host:\n    unexpected: true\n",
        encoding="utf-8",
    )
    output_dir = tmp_path / "runs"
    _guard_run_resources(monkeypatch)

    with pytest.raises(SelectionPolicyError) as error:
        pipeline_cli(
            [
                "ecoli",
                "--standardize",
                "date",
                "--config-dir",
                str(config_dir),
                "--extracted-metadata",
                str(tmp_path / "missing.tsv"),
                "--output-dir",
                str(output_dir),
                "--run-name",
                "invalid-policy",
                "--skip-llm",
                "--quiet",
            ]
        )

    assert str(selection_path) in str(error.value)
    assert "targets.host.unexpected" in str(error.value)
    assert not output_dir.exists()


def test_pipeline_cli_validates_selected_standardization_policy_before_resources(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    extracted_metadata = _prepare_empty_extracted_metadata(tmp_path, monkeypatch)
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    invalid_path = config_dir / "location.yaml"
    invalid_path.write_text("schema_version: [\n", encoding="utf-8")
    output_dir = tmp_path / "runs"
    _guard_run_resources(monkeypatch)

    with pytest.raises(PolicyConfigurationError) as error:
        pipeline_cli(
            [
                "ecoli",
                "--standardize",
                "loc",
                "--config-dir",
                str(config_dir),
                "--extracted-metadata",
                str(extracted_metadata),
                "--output-dir",
                str(output_dir),
                "--run-name",
                "invalid-selected-policy",
                "--quiet",
            ]
        )

    assert str(invalid_path) in str(error.value)
    assert "<yaml>" in str(error.value)
    assert not output_dir.exists()
    assert not (tmp_path / "isolation-cache.db").exists()


def test_pipeline_cli_reuses_extracted_metadata_without_selection_policy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    extracted_metadata = _prepare_empty_extracted_metadata(tmp_path, monkeypatch)
    config_dir = tmp_path / "missing-config"

    pipeline_cli(
        [
            "ecoli",
            "--standardize",
            "date",
            "--config-dir",
            str(config_dir),
            "--extracted-metadata",
            str(extracted_metadata),
            "--output-dir",
            str(tmp_path / "runs"),
            "--run-name",
            "reuse-without-selection",
            "--skip-llm",
            "--quiet",
        ]
    )

    assert (tmp_path / "runs" / "reuse-without-selection" / "reuse-without-selection.tsv").exists()


def _prepare_fixture_backed_cli_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fixture_dataset_builder_factory,
    standardization_fixture_resources,
) -> tuple[Path, Path]:
    monkeypatch.setattr(main_module, "DatasetBuilder", fixture_dataset_builder_factory)
    extracted_metadata = _prepare_empty_extracted_metadata(tmp_path, monkeypatch)
    config_dir = tmp_path / "config"
    shutil.copytree(invocation_module.CONFIG_DIR, config_dir)
    shutil.copyfile(standardization_fixture_resources.host_policy, config_dir / "host.yaml")
    with (config_dir / "isolation_source.yaml").open("a", encoding="utf-8") as policy:
        policy.write(f'cache_db_path: "{(tmp_path / "isolation-source-cache.db").as_posix()}"\n')
    return extracted_metadata, config_dir


def test_isolation_source_run_report_adds_loaded_policy_provenance_to_snapshot_provenance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fixture_dataset_builder_factory,
    standardization_fixture_resources,
) -> None:
    extracted_metadata, config_dir = _prepare_fixture_backed_cli_run(
        tmp_path,
        monkeypatch,
        fixture_dataset_builder_factory,
        standardization_fixture_resources,
    )

    pipeline_cli(
        [
            "ecoli",
            "--standardize",
            "iso",
            "--config-dir",
            str(config_dir),
            "--extracted-metadata",
            str(extracted_metadata),
            "--output-dir",
            str(tmp_path / "runs"),
            "--run-name",
            "isolation-source-provenance",
            "--skip-llm",
            "--quiet",
        ]
    )

    run_report = json.loads(
        (tmp_path / "runs" / "isolation-source-provenance" / "run_report.json").read_text(
            encoding="utf-8"
        )
    )
    expected = asdict(
        IsolationSourcePromptPolicy.load(config_dir / "isolation_source.yaml").provenance
    )
    assert set(run_report["provenance"]) == {
        "biosample",
        "bioproject",
        "sra",
        "genbank_assembly",
        "refseq_assembly",
        "isolation_source",
    }
    assert run_report["provenance"]["isolation_source"] == expected


def test_isolation_source_only_run_report_changes_when_borrowed_host_policy_changes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fixture_dataset_builder_factory,
    standardization_fixture_resources,
) -> None:
    extracted_metadata, config_dir = _prepare_fixture_backed_cli_run(
        tmp_path,
        monkeypatch,
        fixture_dataset_builder_factory,
        standardization_fixture_resources,
    )
    output_dir = tmp_path / "runs"

    def run_configuration_hashes(run_name: str) -> dict[str, str]:
        pipeline_cli(
            [
                "ecoli",
                "--standardize",
                "iso",
                "--config-dir",
                str(config_dir),
                "--extracted-metadata",
                str(extracted_metadata),
                "--output-dir",
                str(output_dir),
                "--run-name",
                run_name,
                "--skip-llm",
                "--quiet",
            ]
        )
        run_report = json.loads(
            (output_dir / run_name / "run_report.json").read_text(encoding="utf-8")
        )
        return run_report["runtime"]["configuration_sha256"]

    before = run_configuration_hashes("before-host-policy-change")
    host_policy_path = config_dir / "host.yaml"
    host_policy = host_policy_path.read_text(encoding="utf-8")
    changed_host_policy = host_policy.replace("    - meat\n", "    - meat product\n")
    assert changed_host_policy != host_policy
    host_policy_path.write_text(changed_host_policy, encoding="utf-8")

    after = run_configuration_hashes("after-host-policy-change")

    host_policy_key = str(host_policy_path)
    assert set(before) == set(after)
    assert before[host_policy_key] != after[host_policy_key]
    assert {path: sha256 for path, sha256 in before.items() if path != host_policy_key} == {
        path: sha256 for path, sha256 in after.items() if path != host_policy_key
    }


@pytest.mark.parametrize(
    ("standardization_target", "skip_llm", "expects_snapshot", "expected_model_keys"),
    [
        ("loc", False, False, set()),
        ("loc", True, False, set()),
        ("iso", True, False, {"isolation_source"}),
        ("date", False, False, set()),
    ],
)
def test_pipeline_cli_scopes_prompt_snapshot_to_active_llm_standardization_targets(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fixture_dataset_builder_factory,
    standardization_fixture_resources,
    standardization_target: str,
    skip_llm: bool,
    expects_snapshot: bool,
    expected_model_keys: set[str],
) -> None:
    extracted_metadata, config_dir = _prepare_fixture_backed_cli_run(
        tmp_path,
        monkeypatch,
        fixture_dataset_builder_factory,
        standardization_fixture_resources,
    )
    arguments = [
        "ecoli",
        "--standardize",
        standardization_target,
        "--config-dir",
        str(config_dir),
        "--extracted-metadata",
        str(extracted_metadata),
        "--output-dir",
        str(tmp_path / "runs"),
        "--run-name",
        "prompt-scope",
        "--quiet",
    ]
    if skip_llm:
        arguments.append("--skip-llm")

    pipeline_cli(arguments)

    run_dir = tmp_path / "runs" / "prompt-scope"
    run_report = json.loads((run_dir / "run_report.json").read_text(encoding="utf-8"))
    assert (run_dir / "prompts.txt").exists() is expects_snapshot
    assert ("prompt_artifact" in run_report) is expects_snapshot
    assert set(run_report["runtime"]["model_identifiers"]) == expected_model_keys


def _prepare_empty_extracted_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    records: list[dict[str, str]] | None = None,
) -> Path:
    extracted_metadata = tmp_path / "custom_metadata.tsv"
    rows = ["\t".join(record.get(column, "") for column in COLUMNS) for record in records or []]
    extracted_metadata.write_text("\n".join(("\t".join(COLUMNS), *rows)) + "\n", encoding="utf-8")
    biosample_manifest_path = invocation_module.DEFAULT_BIOSAMPLE_MANIFEST_PATH
    bioproject_manifest_path = invocation_module.DEFAULT_BIOPROJECT_MANIFEST_PATH
    biosample_manifest = SnapshotManifest.load(biosample_manifest_path)
    bioproject_manifest = SnapshotManifest.load(bioproject_manifest_path)
    ExtractedMetadataProvenance(
        biosample_snapshot_id=biosample_manifest.snapshot_id,
        biosample_manifest_sha256=sha256_file(biosample_manifest_path),
        bioproject_snapshot_id=bioproject_manifest.snapshot_id,
        bioproject_manifest_sha256=sha256_file(bioproject_manifest_path),
        extracted_metadata_sha256=sha256_file(extracted_metadata),
    ).write(provenance_path_for(extracted_metadata))
    prepared_index = tmp_path / "biosample_index.tsv"
    prepared_index.write_text("accession\ttaxon_biosample\tsylph_species\n", encoding="utf-8")
    monkeypatch.setattr(invocation_module, "DEFAULT_INDEX_TSV", prepared_index)
    return extracted_metadata
