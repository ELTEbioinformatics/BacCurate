import csv
import gzip
import hashlib
import json
from dataclasses import dataclass
from datetime import date
from os import replace
from pathlib import Path

import pytest
import yaml

from baccurate import paths
from baccurate.extraction.cli import ExtractionReport, run_extraction
from baccurate.extraction.cli import cli as run_extraction_cli
from baccurate.extraction.xml import CandidateCounters
from baccurate.paths import (
    DEFAULT_BIOPROJECT_SNAPSHOT_MANIFEST,
    DEFAULT_BIOPROJECT_XML_INPUT,
    DEFAULT_BIOSAMPLE_SNAPSHOT_MANIFEST,
    DEFAULT_BIOSAMPLE_XML_INPUT,
)
from baccurate.run_outputs import RunContext, RunDiagnostics, RunOutputs
from baccurate.source_snapshot import (
    SourceSnapshotError,
    bioproject_catalog_path_for,
    provenance_path_for,
    sha256_file,
    validate_derived_metadata_source,
    validate_paired_source_contract,
)


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


def _configure_internal_paths(monkeypatch: pytest.MonkeyPatch, sources: _PairedSources) -> None:
    monkeypatch.setattr(paths, "DEFAULT_BIOSAMPLE_XML_INPUT", sources.biosample)
    monkeypatch.setattr(paths, "DEFAULT_BIOPROJECT_XML_INPUT", sources.bioproject)
    monkeypatch.setattr(paths, "DEFAULT_BIOSAMPLE_SNAPSHOT_MANIFEST", sources.biosample_manifest)
    monkeypatch.setattr(paths, "DEFAULT_BIOPROJECT_SNAPSHOT_MANIFEST", sources.bioproject_manifest)


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


def test_extraction_uses_internal_paired_sources_without_raw_arguments(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sources = _paired_sources(tmp_path)
    index = tmp_path / "biosample_index.tsv"
    index.write_text("accession\tpathogen_biosample\n", encoding="utf-8")
    _configure_internal_paths(monkeypatch, sources)

    report = run_extraction(
        output_path=tmp_path / "extracted.tsv",
        index_path=index,
        disable_progress=True,
    )

    assert report.source_xml_paths == (sources.biosample, sources.bioproject)
    assert report.source_snapshot_id == "biosample-test"
    assert report.bioproject_snapshot_id == "bioproject-test"


def test_extraction_cli_accepts_simplified_supported_invocation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sources = _paired_sources(tmp_path)
    _configure_internal_paths(monkeypatch, sources)
    index = tmp_path / "biosample_index.tsv"
    index.write_text("accession\tpathogen_biosample\n", encoding="utf-8")
    output = tmp_path / "extracted.tsv"

    run_extraction_cli(["--output", str(output), "--index", str(index), "--quiet"])

    assert output.exists()


def test_extraction_validates_paired_source_before_other_work(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sources = _paired_sources(tmp_path)
    sources.bioproject.unlink()
    _configure_internal_paths(monkeypatch, sources)
    output = tmp_path / "extracted.tsv"

    with pytest.raises(SourceSnapshotError, match=r"BioProject.*missing"):
        run_extraction(
            output_path=output,
            index_path=tmp_path / "missing-index.tsv.gz",
            config_dir=tmp_path / "missing-config",
            disable_progress=True,
        )

    assert not output.exists()


def test_extraction_report_carries_both_validated_source_identities(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sources = _paired_sources(tmp_path)
    _configure_internal_paths(monkeypatch, sources)
    index = tmp_path / "biosample_index.tsv"
    index.write_text("accession\tpathogen_biosample\n", encoding="utf-8")

    report = run_extraction(
        output_path=tmp_path / "extracted.tsv",
        index_path=index,
        disable_progress=True,
    )

    assert report.source_snapshot_id == "biosample-test"
    assert report.bioproject_snapshot_id == "bioproject-test"
    assert report.metadata_reference_date == date(2026, 7, 19)
    assert report.source_xml_paths == (sources.biosample, sources.bioproject)


def test_extraction_publishes_provenance_bound_bioproject_context_bundle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sources = _paired_sources(tmp_path)
    biosample_xml = b"""\
<BioSampleSet>
  <BioSample accession="SAMN00000001">
    <Attributes>
      <Attribute attribute="isolation_source">farm soil</Attribute>
    </Attributes>
    <Links><Link target="bioproject">1050647</Link></Links>
  </BioSample>
</BioSampleSet>
"""
    bioproject_xml = b"""\
<PackageSet>
  <Package>
    <Project>
      <Project>
        <ProjectID>
          <ArchiveID accession="PRJNA1050647" archive="NCBI" id="1050647" />
        </ProjectID>
        <ProjectDescr>
          <Title>&lt;b&gt;One &amp;amp; Health&lt;/b&gt; study</Title>
          <Description>&lt;p&gt;Farm &amp;amp;&lt;/p&gt;&lt;p&gt;soil context.&lt;/p&gt;</Description>
          <Relevance>
            <Agricultural>yes</Agricultural>
            <Environmental>Yes</Environmental>
            <Veterinary>no</Veterinary>
            <Medical>yes</Medical>
            <Evolution>yes</Evolution>
          </Relevance>
        </ProjectDescr>
      </Project>
    </Project>
  </Package>
</PackageSet>
"""
    biosample_hash = _write_snapshot(sources.biosample, biosample_xml)
    bioproject_hash = _write_snapshot(sources.bioproject, bioproject_xml)
    _write_manifest(
        sources.biosample_manifest,
        snapshot_id="biosample-test",
        source=sources.biosample,
        sha256=biosample_hash,
    )
    _write_manifest(
        sources.bioproject_manifest,
        snapshot_id="bioproject-test",
        source=sources.bioproject,
        sha256=bioproject_hash,
    )
    _configure_internal_paths(monkeypatch, sources)
    index = tmp_path / "biosample_index.tsv"
    index.write_text(
        "accession\tpathogen_biosample\nSAMN00000001\tecoli\n",
        encoding="utf-8",
    )
    extracted = tmp_path / "custom_metadata.tsv"

    report = run_extraction(
        output_path=extracted,
        index_path=index,
        disable_progress=True,
    )

    catalog = bioproject_catalog_path_for(extracted)
    provenance = provenance_path_for(extracted)
    assert report.bioproject_catalog_path == catalog
    assert report.bundle_provenance_path == provenance
    assert catalog.name == "custom_metadata.bioproject_context.jsonl"
    assert provenance.name == "custom_metadata.provenance.yaml"

    with extracted.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))
    assert len(rows) == 1
    assert rows[0]["bioproject_id"] == "1050647"
    assert rows[0]["bioproject_accession"] == "PRJNA1050647"
    assert "title" not in rows[0]
    assert "description" not in rows[0]

    catalog_objects = [
        json.loads(line) for line in catalog.read_text(encoding="utf-8").splitlines()
    ]
    assert catalog_objects == [
        {
            "id": "1050647",
            "accession": "PRJNA1050647",
            "title": "One & Health study",
            "description": "Farm & soil context.",
            "relevance": ["Agricultural", "Environmental"],
        }
    ]

    provenance_record = yaml.safe_load(provenance.read_text(encoding="utf-8"))
    assert provenance_record["bundle_version"] == 1
    assert provenance_record["source_manifests"] == {
        "biosample": {
            "snapshot_id": "biosample-test",
            "path": str(sources.biosample_manifest),
            "sha256": sha256_file(sources.biosample_manifest),
        },
        "bioproject": {
            "snapshot_id": "bioproject-test",
            "path": str(sources.bioproject_manifest),
            "sha256": sha256_file(sources.bioproject_manifest),
        },
    }
    assert provenance_record["artifacts"] == {
        "extracted_metadata": {
            "path": extracted.name,
            "sha256": sha256_file(extracted),
        },
        "bioproject_context": {
            "path": catalog.name,
            "sha256": sha256_file(catalog),
        },
    }


def test_interrupted_bundle_publication_cannot_leave_valid_provenance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sources = _paired_sources(tmp_path)
    _configure_internal_paths(monkeypatch, sources)
    index = tmp_path / "biosample_index.tsv"
    index.write_text("accession\tpathogen_biosample\n", encoding="utf-8")
    extracted = tmp_path / "interrupted.tsv"
    catalog = bioproject_catalog_path_for(extracted)
    real_replace = replace

    def interrupt_catalog_publication(source: Path | str, destination: Path | str) -> None:
        if Path(destination) == catalog:
            raise OSError("simulated publication interruption")
        real_replace(source, destination)

    monkeypatch.setattr("baccurate.extraction.cli.os.replace", interrupt_catalog_publication)

    with pytest.raises(OSError, match="simulated publication interruption"):
        run_extraction(
            output_path=extracted,
            index_path=index,
            disable_progress=True,
        )

    assert not provenance_path_for(extracted).exists()
    with pytest.raises(SourceSnapshotError, match="derived bundle provenance"):
        validate_derived_metadata_source(extracted, sources.biosample_manifest)


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
            bundle_provenance_path=tmp_path / "extracted.provenance.yaml",
            bioproject_catalog_path=tmp_path / "extracted.bioproject_context.jsonl",
        ),
        elapsed_seconds=1.0,
    )

    document = json.loads(outputs.diagnostics.read_text(encoding="utf-8"))
    assert document["source"]["biosample"]["snapshot_id"] == "biosample-test"
    assert document["source"]["bioproject"]["snapshot_id"] == "bioproject-test"
    assert document["source"]["metadata_reference_date"] == "2026-07-09"
