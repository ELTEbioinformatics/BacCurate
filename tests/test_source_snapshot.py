import gzip
import hashlib
import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import pytest
import yaml

from baccurate.extraction.cli import ExtractionReport
from baccurate.extraction.cli import main as run_extraction
from baccurate.extraction.xml import CandidateCounters
from baccurate.paths import (
    DEFAULT_BIOPROJECT_SNAPSHOT_MANIFEST,
    DEFAULT_BIOPROJECT_XML_INPUT,
    DEFAULT_BIOSAMPLE_SNAPSHOT_MANIFEST,
    DEFAULT_BIOSAMPLE_XML_INPUT,
)
from baccurate.run_outputs import RunContext, RunDiagnostics, RunOutputs
from baccurate.source_snapshot import SourceSnapshotError, validate_paired_source_contract


def _write_snapshot(path: Path, contents: bytes) -> str:
    with (
        path.open("wb") as raw_stream,
        gzip.GzipFile(fileobj=raw_stream, mode="wb", mtime=0) as stream,
    ):
        stream.write(contents)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_manifest(path: Path, *, snapshot_id: str, source: Path, sha256: str) -> None:
    path.write_text(
        yaml.safe_dump(
            {
                "manifest_version": 1,
                "snapshot_id": snapshot_id,
                "provider": "NCBI",
                "retrieved_on": "2026-07-19",
                "metadata_reference_date": "2026-07-19",
                "files": [{"name": source.name, "sha256": sha256}],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )


@dataclass(frozen=True, slots=True)
class _PairedSources:
    biosample: Path
    bioproject: Path
    biosample_manifest: Path
    bioproject_manifest: Path


def _paired_sources(tmp_path: Path) -> _PairedSources:
    biosample = tmp_path / "biosamples.xml.gz"
    bioproject = tmp_path / "bioproject.xml.gz"
    biosample_hash = _write_snapshot(biosample, b"<BioSampleSet />")
    bioproject_hash = _write_snapshot(bioproject, b"<PackageSet />")
    biosample_manifest = tmp_path / "biosample_snapshot.yaml"
    bioproject_manifest = tmp_path / "bioproject_snapshot.yaml"
    _write_manifest(
        biosample_manifest,
        snapshot_id="biosample-test",
        source=biosample,
        sha256=biosample_hash,
    )
    _write_manifest(
        bioproject_manifest,
        snapshot_id="bioproject-test",
        source=bioproject,
        sha256=bioproject_hash,
    )
    return _PairedSources(
        biosample=biosample,
        bioproject=bioproject,
        biosample_manifest=biosample_manifest,
        bioproject_manifest=bioproject_manifest,
    )


def test_paired_source_contract_validates_both_compressed_snapshots(tmp_path: Path) -> None:
    sources = _paired_sources(tmp_path)

    source = validate_paired_source_contract(
        biosample_path=sources.biosample,
        bioproject_path=sources.bioproject,
        biosample_manifest_path=sources.biosample_manifest,
        bioproject_manifest_path=sources.bioproject_manifest,
    )

    assert source.biosample.snapshot_id == "biosample-test"
    assert source.bioproject.snapshot_id == "bioproject-test"
    assert source.metadata_reference_date == date(2026, 7, 19)


@pytest.mark.parametrize(
    ("missing_role", "message"),
    [("biosample", "BioSample.*missing"), ("bioproject", "BioProject.*missing")],
)
def test_paired_source_contract_reports_each_missing_snapshot(
    tmp_path: Path, missing_role: str, message: str
) -> None:
    sources = _paired_sources(tmp_path)
    getattr(sources, missing_role).unlink()

    with pytest.raises(SourceSnapshotError, match=message):
        validate_paired_source_contract(
            biosample_path=sources.biosample,
            bioproject_path=sources.bioproject,
            biosample_manifest_path=sources.biosample_manifest,
            bioproject_manifest_path=sources.bioproject_manifest,
        )


def test_default_paired_source_locations_are_internal_compressed_paths() -> None:
    assert DEFAULT_BIOSAMPLE_XML_INPUT.name == "biosamples.xml.gz"
    assert DEFAULT_BIOPROJECT_XML_INPUT.name == "bioproject.xml.gz"
    assert DEFAULT_BIOSAMPLE_SNAPSHOT_MANIFEST.name == "biosample_snapshot.yaml"
    assert DEFAULT_BIOPROJECT_SNAPSHOT_MANIFEST.name == "bioproject_snapshot.yaml"


def test_extraction_validates_paired_source_before_other_work(tmp_path: Path) -> None:
    sources = _paired_sources(tmp_path)
    sources.bioproject.unlink()
    output = tmp_path / "extracted.tsv"

    with pytest.raises(SourceSnapshotError, match=r"BioProject.*missing"):
        run_extraction(
            input_path=sources.biosample,
            bioproject_input_path=sources.bioproject,
            output_path=output,
            index_path=tmp_path / "missing-index.tsv.gz",
            config_dir=tmp_path / "missing-config",
            source_manifest_path=sources.biosample_manifest,
            bioproject_manifest_path=sources.bioproject_manifest,
            disable_progress=True,
        )

    assert not output.exists()


def test_extraction_report_carries_both_validated_source_identities(tmp_path: Path) -> None:
    sources = _paired_sources(tmp_path)
    index = tmp_path / "biosample_index.tsv"
    index.write_text("accession\tpathogen_biosample\n", encoding="utf-8")

    report = run_extraction(
        input_path=sources.biosample,
        bioproject_input_path=sources.bioproject,
        output_path=tmp_path / "extracted.tsv",
        index_path=index,
        source_manifest_path=sources.biosample_manifest,
        bioproject_manifest_path=sources.bioproject_manifest,
        disable_progress=True,
    )

    assert report.source_snapshot_id == "biosample-test"
    assert report.bioproject_snapshot_id == "bioproject-test"
    assert report.metadata_reference_date == date(2026, 7, 19)
    assert report.source_xml_paths == (sources.biosample, sources.bioproject)


@pytest.mark.parametrize(
    ("unexpected_role", "message"),
    [("biosample", "BioSample.*unexpected"), ("bioproject", "BioProject.*unexpected")],
)
def test_paired_source_contract_reports_each_unexpected_snapshot(
    tmp_path: Path, unexpected_role: str, message: str
) -> None:
    sources = _paired_sources(tmp_path)
    unexpected = tmp_path / "alternate.xml.gz"
    _write_snapshot(unexpected, b"<Unexpected />")

    with pytest.raises(SourceSnapshotError, match=message):
        validate_paired_source_contract(
            biosample_path=(unexpected if unexpected_role == "biosample" else sources.biosample),
            bioproject_path=(unexpected if unexpected_role == "bioproject" else sources.bioproject),
            biosample_manifest_path=sources.biosample_manifest,
            bioproject_manifest_path=sources.bioproject_manifest,
        )


@pytest.mark.parametrize(
    ("changed_role", "message"),
    [
        ("biosample", "BioSample.*checksum mismatch"),
        ("bioproject", "BioProject.*checksum mismatch"),
    ],
)
def test_paired_source_contract_reports_each_checksum_mismatch(
    tmp_path: Path, changed_role: str, message: str
) -> None:
    sources = _paired_sources(tmp_path)
    changed = getattr(sources, changed_role)
    changed.write_bytes(changed.read_bytes() + b"changed compressed bytes")

    with pytest.raises(SourceSnapshotError, match=message):
        validate_paired_source_contract(
            biosample_path=sources.biosample,
            bioproject_path=sources.bioproject,
            biosample_manifest_path=sources.biosample_manifest,
            bioproject_manifest_path=sources.bioproject_manifest,
        )


def test_run_source_reporting_preserves_both_identities_and_biosample_date(
    tmp_path: Path,
) -> None:
    outputs = RunOutputs.plan(
        output_dir=tmp_path,
        run_name="paired-source",
        output_file=None,
        include_isolation=False,
    )
    outputs.initialize()
    diagnostics = RunDiagnostics(
        outputs,
        RunContext(
            requested_pathogens=("ecoli",),
            requested_attributes=("date",),
            extracted_metadata=tmp_path / "extracted.tsv",
            options={},
            configuration_paths=(),
            skip_llm=True,
            model_identifiers={},
        ),
    )
    diagnostics.record_performed_extraction(
        ExtractionReport(
            source_xml_paths=(
                tmp_path / "biosamples.xml.gz",
                tmp_path / "bioproject.xml.gz",
            ),
            extracted_metadata_path=tmp_path / "extracted.tsv",
            extracted_record_count=0,
            counters=CandidateCounters(),
            automatic_rejection_counts={},
            unreviewed_count=0,
            uncertain_count=0,
            review_artifact_paths={},
            source_snapshot_id="biosample-test",
            bioproject_snapshot_id="bioproject-test",
            metadata_reference_date=date(2026, 7, 9),
            source_record_path=tmp_path / "extracted.provenance.yaml",
        ),
        elapsed_seconds=1.0,
    )

    document = json.loads(outputs.diagnostics.read_text(encoding="utf-8"))
    assert document["source"]["biosample"]["snapshot_id"] == "biosample-test"
    assert document["source"]["bioproject"]["snapshot_id"] == "bioproject-test"
    assert document["source"]["metadata_reference_date"] == "2026-07-09"
