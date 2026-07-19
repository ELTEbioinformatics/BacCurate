import json
from collections.abc import Callable, Sequence
from pathlib import Path

import pytest

import baccurate.main as main_module
from baccurate.extraction.cli import cli as extraction_cli
from baccurate.extraction.tables import COLUMNS
from baccurate.main import main as pipeline_cli
from baccurate.source_snapshot import (
    ArtifactReference,
    DerivedArtifactReferences,
    DerivedBundleProvenance,
    ManifestReference,
    PairedManifestReferences,
    SourceSnapshotManifest,
    bioproject_catalog_path_for,
    provenance_path_for,
    sha256_file,
)

RETIRED_RAW_SOURCE_OPTIONS = ("--input", "--source-manifest", "--uncompressed")


@pytest.mark.parametrize("option", RETIRED_RAW_SOURCE_OPTIONS)
def test_pipeline_cli_rejects_raw_source_options(
    option: str, capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit) as exit_info:
        pipeline_cli([option])

    assert exit_info.value.code == 2
    assert f"unrecognized arguments: {option}" in capsys.readouterr().err


@pytest.mark.parametrize("option", RETIRED_RAW_SOURCE_OPTIONS)
def test_extraction_cli_rejects_raw_source_options(
    option: str, capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit) as exit_info:
        extraction_cli(["--output", "unused.tsv", option])

    assert exit_info.value.code == 2
    assert f"unrecognized arguments: {option}" in capsys.readouterr().err


@pytest.mark.parametrize("entry_point", [pipeline_cli, extraction_cli])
def test_cli_help_advertises_only_the_supported_raw_input_contract(
    entry_point: Callable[[Sequence[str]], None], capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit) as exit_info:
        entry_point(["--help"])

    assert exit_info.value.code == 0
    help_text = capsys.readouterr().out
    assert "--input" not in help_text
    assert "--source-manifest" not in help_text
    assert "--uncompressed" not in help_text


def test_pipeline_cli_keeps_extracted_metadata_option(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as exit_info:
        pipeline_cli(["--help"])

    assert exit_info.value.code == 0
    assert "--extracted-metadata" in capsys.readouterr().out


def test_pipeline_cli_uses_custom_extracted_metadata_bundle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    extracted_metadata = tmp_path / "custom_metadata.tsv"
    extracted_metadata.write_text("\t".join(COLUMNS) + "\n", encoding="utf-8")
    catalog = bioproject_catalog_path_for(extracted_metadata)
    catalog.write_text("", encoding="utf-8")
    biosample_manifest_path = main_module.DEFAULT_BIOSAMPLE_SNAPSHOT_MANIFEST
    bioproject_manifest_path = main_module.DEFAULT_BIOPROJECT_SNAPSHOT_MANIFEST
    biosample_manifest = SourceSnapshotManifest.load(biosample_manifest_path)
    bioproject_manifest = SourceSnapshotManifest.load(bioproject_manifest_path)
    DerivedBundleProvenance(
        bundle_version=1,
        source_manifests=PairedManifestReferences(
            biosample=ManifestReference(
                snapshot_id=biosample_manifest.snapshot_id,
                path=str(biosample_manifest_path),
                sha256=sha256_file(biosample_manifest_path),
            ),
            bioproject=ManifestReference(
                snapshot_id=bioproject_manifest.snapshot_id,
                path=str(bioproject_manifest_path),
                sha256=sha256_file(bioproject_manifest_path),
            ),
        ),
        artifacts=DerivedArtifactReferences(
            extracted_metadata=ArtifactReference(
                path=extracted_metadata.name,
                sha256=sha256_file(extracted_metadata),
            ),
            bioproject_context=ArtifactReference(
                path=catalog.name,
                sha256=sha256_file(catalog),
            ),
        ),
    ).write(provenance_path_for(extracted_metadata))
    atb_index = tmp_path / "biosample_index.tsv"
    atb_index.write_text("accession\tin_ATB\tpathogen_ATB\n", encoding="utf-8")
    monkeypatch.setattr(main_module, "DEFAULT_INDEX_TSV", atb_index)

    pipeline_cli(
        [
            "ecoli",
            "--attribute",
            "date",
            "--extracted-metadata",
            str(extracted_metadata),
            "--output-dir",
            str(tmp_path / "runs"),
            "--run-name",
            "custom-bundle",
            "--skip-llm",
            "--quiet",
        ]
    )

    run_dir = tmp_path / "runs" / "custom-bundle"
    assert (run_dir / "custom-bundle.tsv").exists()
    diagnostics = json.loads((run_dir / "diagnostics.json").read_text(encoding="utf-8"))
    assert diagnostics["outputs"]["extracted_input"] == str(extracted_metadata)
    options = diagnostics["runtime"]["options"]
    assert "input" not in options
    assert "uncompressed" not in options
    assert "source_manifest" not in options
