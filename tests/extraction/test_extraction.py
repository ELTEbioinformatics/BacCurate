"""BioSample metadata extraction and publication contracts."""

import csv
from datetime import date
from os import replace
from pathlib import Path
from typing import Any, Protocol

import pytest

import baccurate.extraction as extraction
from baccurate import paths
from baccurate.extraction import (
    CurationSchema,
    ExtractionReport,
)
from baccurate.paths import (
    DEFAULT_BIOPROJECT_SNAPSHOT_MANIFEST,
    DEFAULT_BIOPROJECT_XML_INPUT,
    DEFAULT_BIOSAMPLE_SNAPSHOT_MANIFEST,
    DEFAULT_BIOSAMPLE_XML_INPUT,
)
from baccurate.provenance.source_snapshot import (
    provenance_path_for,
    validate_extracted_metadata_bundle,
)

ROOT = Path(__file__).parents[2]
# Ruff resolves attributes on the original import binding as same-named submodules.
extraction_facade = extraction


class _PairedSourceSnapshotPaths(Protocol):
    biosample: Path
    bioproject: Path
    biosample_manifest: Path
    bioproject_manifest: Path


def run_extraction(**kwargs: Any) -> ExtractionReport:
    """Call extraction through its injected curation-schema interface."""
    kwargs.setdefault(
        "curation_schema",
        CurationSchema.load(ROOT / "config" / "curation_schema.yaml"),
    )
    return extraction_facade.run_extraction(**kwargs)


def _configure_internal_paths(
    monkeypatch: pytest.MonkeyPatch,
    sources: _PairedSourceSnapshotPaths,
) -> None:
    monkeypatch.setattr(paths, "DEFAULT_BIOSAMPLE_XML_INPUT", sources.biosample)
    monkeypatch.setattr(paths, "DEFAULT_BIOPROJECT_XML_INPUT", sources.bioproject)
    monkeypatch.setattr(paths, "DEFAULT_BIOSAMPLE_SNAPSHOT_MANIFEST", sources.biosample_manifest)
    monkeypatch.setattr(paths, "DEFAULT_BIOPROJECT_SNAPSHOT_MANIFEST", sources.bioproject_manifest)


def test_default_paired_source_locations_are_internal_compressed_paths() -> None:
    assert DEFAULT_BIOSAMPLE_XML_INPUT.name == "biosamples.xml.gz"
    assert DEFAULT_BIOPROJECT_XML_INPUT.name == "bioproject.xml.gz"
    assert DEFAULT_BIOSAMPLE_SNAPSHOT_MANIFEST.name == "biosample_snapshot.yaml"
    assert DEFAULT_BIOPROJECT_SNAPSHOT_MANIFEST.name == "bioproject_snapshot.yaml"


def test_extraction_uses_internal_paired_sources_without_raw_arguments(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    paired_source_snapshots,
) -> None:
    sources = paired_source_snapshots
    index = tmp_path / "biosample_index.tsv"
    index.write_text("accession\tpathogen_biosample\n", encoding="utf-8")
    _configure_internal_paths(monkeypatch, sources)

    report = run_extraction(
        output_path=tmp_path / "extracted.tsv",
        index_path=index,
        disable_progress=True,
    )

    assert report.prepared_input_paths == (sources.biosample, sources.bioproject)
    assert report.biosample_snapshot_id == "biosample-test"
    assert report.bioproject_snapshot_id == "bioproject-test"


def test_extraction_cli_accepts_simplified_supported_invocation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    paired_source_snapshots,
) -> None:
    sources = paired_source_snapshots
    _configure_internal_paths(monkeypatch, sources)
    index = tmp_path / "biosample_index.tsv"
    index.write_text("accession\tpathogen_biosample\n", encoding="utf-8")
    output = tmp_path / "extracted.tsv"

    extraction_facade.cli(["--output", str(output), "--index", str(index), "--quiet"])

    assert output.exists()


def test_extraction_validates_paired_source_before_other_work(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    paired_source_snapshots,
) -> None:
    sources = paired_source_snapshots
    sources.bioproject.unlink()
    _configure_internal_paths(monkeypatch, sources)
    output = tmp_path / "extracted.tsv"

    with pytest.raises(FileNotFoundError):
        run_extraction(
            output_path=output,
            index_path=tmp_path / "missing-index.tsv.gz",
            disable_progress=True,
        )

    assert not output.exists()


def test_extraction_report_carries_both_validated_source_identities(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    paired_source_snapshots,
) -> None:
    sources = paired_source_snapshots
    _configure_internal_paths(monkeypatch, sources)
    index = tmp_path / "biosample_index.tsv"
    index.write_text("accession\tpathogen_biosample\n", encoding="utf-8")

    report = run_extraction(
        output_path=tmp_path / "extracted.tsv",
        index_path=index,
        disable_progress=True,
    )

    assert report.biosample_snapshot_id == "biosample-test"
    assert report.bioproject_snapshot_id == "bioproject-test"
    assert report.metadata_reference_date == date(2026, 7, 19)
    assert report.prepared_input_paths == (sources.biosample, sources.bioproject)


def test_extraction_reports_inclusion_routes_for_records_present_in_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    paired_source_snapshots,
) -> None:
    sources = paired_source_snapshots
    sources.replace_contents(
        biosample_xml=b"""\
<BioSampleSet>
  <BioSample accession="taxonomy"><Attributes>
    <Attribute attribute="isolation_source">soil</Attribute>
  </Attributes></BioSample>
  <BioSample accession="atb"><Attributes>
    <Attribute attribute="isolation_source">water</Attribute>
  </Attributes></BioSample>
</BioSampleSet>
""",
        bioproject_xml=b"<PackageSet />",
    )
    _configure_internal_paths(monkeypatch, sources)
    index = tmp_path / "biosample_index.tsv"
    index.write_text(
        "accession\tpathogen_biosample\tpathogen_ATB\n"
        "taxonomy\tecoli\tNA\n"
        "atb\tNA\tecoli\n"
        "absent-from-snapshot\tNA\tecoli\n",
        encoding="utf-8",
    )

    report = run_extraction(
        output_path=tmp_path / "extracted.tsv",
        index_path=index,
        disable_progress=True,
    )

    assert report.extracted_record_count == 2
    assert report.inclusion_route_counts == {
        "biosample_taxonomy": 1,
        "allthebacteria": 1,
    }


def test_extraction_report_preserves_production_curation_review_counters(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    paired_source_snapshots,
) -> None:
    sources = paired_source_snapshots
    biosample_xml = b"""\
<BioSampleSet>
  <BioSample accession="SAMN1"><Attributes>
    <Attribute attribute="host">unknown</Attribute>
  </Attributes></BioSample>
  <BioSample accession="SAMN2"><Attributes>
    <Attribute attribute="collection_date">missing</Attribute>
  </Attributes></BioSample>
  <BioSample accession="SAMN3"><Attributes>
    <Attribute attribute="isolation_source">GENOMIC</Attribute>
  </Attributes></BioSample>
  <BioSample accession="SAMN4"><Attributes>
    <Attribute attribute="host">unkmowm</Attribute>
  </Attributes></BioSample>
  <BioSample accession="SAMN5"><Attributes>
    <Attribute attribute="host_environment">human</Attribute>
  </Attributes></BioSample>
  <BioSample accession="SAMN6"><Attributes>
    <Attribute attribute="collection_date">aerobic</Attribute>
  </Attributes></BioSample>
</BioSampleSet>
"""
    sources.replace_contents(
        biosample_xml=biosample_xml,
        bioproject_xml=b"<PackageSet />",
    )
    _configure_internal_paths(monkeypatch, sources)
    index = tmp_path / "biosample_index.tsv"
    index.write_text("accession\tpathogen_biosample\n", encoding="utf-8")

    report = run_extraction(
        output_path=tmp_path / "extracted.tsv",
        index_path=index,
        disable_progress=True,
        curation_schema=CurationSchema.load(ROOT / "config" / "curation_schema.yaml"),
    )

    assert report.unreviewed_count == 2
    assert report.uncertain_count == 1
    assert report.automatic_rejection_counts == {
        "date": {"non_date_evidence": 1, "universal_missing": 1},
        "host": {"universal_missing": 1},
        "iso": {"non_discriminative_process": 1},
    }


@pytest.mark.parametrize(
    "value",
    [
        "missing:",
        "missing: control sample",
        ' " MISSING: unavailable from submitter " ',
    ],
)
def test_universal_missing_marker_rejects_the_whole_value(value: str) -> None:
    schema = CurationSchema.load(ROOT / "config" / "curation_schema.yaml")

    decision = schema.evaluate(attribute="collection_date", value=value)

    assert decision.matches == ()
    assert decision.events[0].family == "universal_missing"


@pytest.mark.parametrize(
    "value",
    [
        "missingness: control sample",
        "missing-control sample",
        "record missing: control sample",
    ],
)
def test_universal_missing_marker_does_not_match_nearby_syntax(value: str) -> None:
    schema = CurationSchema.load(ROOT / "config" / "curation_schema.yaml")

    decision = schema.evaluate(attribute="collection_date", value=value)

    assert [match.target for match in decision.matches] == ["date"]
    assert decision.events == ()


def test_extraction_publishes_provenance_bound_metadata_bundle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    paired_source_snapshots,
) -> None:
    sources = paired_source_snapshots
    biosample_xml = b"""\
<BioSampleSet>
  <BioSample accession="SAMN00000001" last_update="2025-06-30T12:34:56.000">
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
    sources.replace_contents(
        biosample_xml=biosample_xml,
        bioproject_xml=bioproject_xml,
    )
    _configure_internal_paths(monkeypatch, sources)
    index = tmp_path / "biosample_index.tsv"
    index.write_text(
        "accession\tpathogen_biosample\tsra_run_accessions\t"
        "genbank_assembly_accessions\trefseq_assembly_accessions\n"
        "SAMN00000001\tecoli\tSRR1,SRR2\tGCA_1.1\tNA\n",
        encoding="utf-8",
    )
    extracted = tmp_path / "custom_metadata.tsv"

    report = run_extraction(
        output_path=extracted,
        index_path=index,
        disable_progress=True,
    )

    provenance = provenance_path_for(extracted)
    assert report.bundle_provenance_path == provenance
    assert provenance.name == "custom_metadata.provenance.yaml"

    with extracted.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))
    assert len(rows) == 1
    assert rows[0]["bioproject_accession"] == "PRJNA1050647"
    assert "bioproject_id" not in rows[0]
    assert rows[0]["sra_run_accessions"] == "SRR1||SRR2"
    assert rows[0]["genbank_assembly_accessions"] == "GCA_1.1"
    assert rows[0]["refseq_assembly_accessions"] == ""
    assert rows[0]["biosample_last_update"] == "2025-06-30T12:34:56.000"
    assert "title" not in rows[0]
    assert "description" not in rows[0]

    source_contract = validate_extracted_metadata_bundle(
        extracted,
        sources.biosample_manifest,
        sources.bioproject_manifest,
    )
    assert source_contract.biosample.snapshot_id == "biosample-test"
    assert source_contract.bioproject.snapshot_id == "bioproject-test"


def test_extraction_preserves_linked_project_sets_and_unresolved_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    paired_source_snapshots,
) -> None:
    sources = paired_source_snapshots
    samples = "".join(
        f"""
  <BioSample accession="{accession}">
    <Attributes><Attribute attribute="isolation_source">soil</Attribute></Attributes>
    {links}
  </BioSample>"""
        for accession, links in (
            (
                "SAMN00000001",
                '<Links><Link target="bioproject">3</Link><Link target="bioproject">2</Link>'
                '<Link target="bioproject">2</Link><Link target="bioproject">1</Link></Links>',
            ),
            (
                "SAMN00000002",
                '<Links><Link target="bioproject">99</Link><Link target="bioproject">2</Link>'
                '<Link target="bioproject">99</Link></Links>',
            ),
            ("SAMN00000003", '<Links><Link target="bioproject">98</Link></Links>'),
            ("SAMN00000004", ""),
        )
    )
    samples += """
  <BioSample accession="SAMN00000005">
    <Links><Link target="bioproject">1</Link><Link target="bioproject">99</Link></Links>
  </BioSample>"""
    sources.replace_contents(
        biosample_xml=f"<BioSampleSet>{samples}</BioSampleSet>".encode(),
        bioproject_xml=b"""\
<PackageSet>
  <Package><Project><Project><ProjectID><ArchiveID accession="PRJNA300" id="3" /></ProjectID>
    <ProjectDescr><Title>Third</Title></ProjectDescr></Project></Project></Package>
  <Package><Project><Project><ProjectID><ArchiveID accession="PRJNA100" id="1" /></ProjectID>
    <ProjectDescr><Title>First</Title></ProjectDescr></Project></Project></Package>
  <Package><Project><Project><ProjectID><ArchiveID accession="PRJNA200" id="2" /></ProjectID>
    <ProjectDescr><Title>Second</Title></ProjectDescr></Project></Project></Package>
  <Package><Project><Project><ProjectID><ArchiveID id="98" /></ProjectID>
    <ProjectDescr><Title>Missing accession</Title></ProjectDescr></Project></Project></Package>
</PackageSet>
""",
    )
    _configure_internal_paths(monkeypatch, sources)
    index = tmp_path / "biosample_index.tsv"
    index.write_text(
        "accession\tpathogen_biosample\n"
        "SAMN00000001\tecoli\n"
        "SAMN00000002\tecoli\n"
        "SAMN00000003\tecoli\n"
        "SAMN00000004\tecoli\n"
        "SAMN00000005\tecoli\n",
        encoding="utf-8",
    )
    extracted = tmp_path / "linked_sets.tsv"

    report = run_extraction(
        output_path=extracted,
        index_path=index,
        disable_progress=True,
    )

    with extracted.open(newline="", encoding="utf-8") as stream:
        rows = {row["accession"]: row for row in csv.DictReader(stream, delimiter="\t")}
    assert rows["SAMN00000001"]["bioproject_accession"] == ("PRJNA100||PRJNA200||PRJNA300")
    assert rows["SAMN00000002"]["bioproject_accession"] == "PRJNA200"
    assert rows["SAMN00000003"]["bioproject_accession"] == ""
    assert rows["SAMN00000004"]["bioproject_accession"] == ""
    assert "SAMN00000005" not in rows
    unresolved = tmp_path / "unresolved_bioproject_links.tsv"
    assert report.review_worklist_paths["unresolved_bioproject_links"] == unresolved
    assert unresolved.read_text(encoding="utf-8") == (
        "bioproject_id\tcount\trepresentative_biosample_accessions\n"
        "98\t1\tSAMN00000003\n"
        "99\t1\tSAMN00000002\n"
    )


def test_equivalent_link_and_project_order_produces_byte_stable_data_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    paired_source_snapshots,
) -> None:
    sources = paired_source_snapshots
    _configure_internal_paths(monkeypatch, sources)
    index = tmp_path / "biosample_index.tsv"
    index.write_text(
        "accession\tpathogen_biosample\nSAMN00000001\tecoli\n",
        encoding="utf-8",
    )
    sample_template = """\
<BioSampleSet><BioSample accession="SAMN00000001">
  <Attributes><Attribute attribute="isolation_source">soil</Attribute></Attributes>
  <Links>{links}</Links>
</BioSample></BioSampleSet>
"""
    long_tail = "Z" * 5000
    project_one = f"""\
<Package><ArchiveID accession="PRJNA1" id="1" /><ProjectDescr>
  <Title>Mixed <i>Case</i> &amp; complete</Title>
  <Description><p>Alpha &amp; beta</p><ul><li>Gamma</li><li>Delta</li></ul> {long_tail}</Description>
</ProjectDescr></Package>
"""
    project_two = """\
<Package><ArchiveID accession="PRJNA2" id="2" /><ProjectDescr>
  <Title>Second</Title><Description />
  <Relevance><Agricultural>true</Agricultural><Medical>yes</Medical></Relevance>
</ProjectDescr></Package>
"""

    outputs = []
    for stem, links, projects in (
        (
            "ordered_a",
            '<Link target="bioproject">2</Link><Link target="bioproject">1</Link>',
            project_two + project_one,
        ),
        (
            "ordered_b",
            '<Link target="bioproject">1</Link><Link target="bioproject">2</Link>',
            project_one + project_two,
        ),
    ):
        sources.replace_contents(
            biosample_xml=sample_template.format(links=links).encode(),
            bioproject_xml=f"<PackageSet>{projects}</PackageSet>".encode(),
        )
        extracted = tmp_path / f"{stem}.tsv"
        run_extraction(output_path=extracted, index_path=index, disable_progress=True)
        outputs.append(extracted.read_bytes())

    assert outputs[0] == outputs[1]


def test_interrupted_bundle_publication_cannot_leave_valid_provenance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    paired_source_snapshots,
) -> None:
    sources = paired_source_snapshots
    _configure_internal_paths(monkeypatch, sources)
    index = tmp_path / "biosample_index.tsv"
    index.write_text("accession\tpathogen_biosample\n", encoding="utf-8")
    extracted = tmp_path / "interrupted.tsv"
    provenance = provenance_path_for(extracted)
    real_replace = replace

    def interrupt_provenance_publication(source: Path | str, destination: Path | str) -> None:
        if Path(destination) == provenance:
            raise OSError("simulated publication interruption")
        real_replace(source, destination)

    monkeypatch.setattr(
        "baccurate.provenance.source_snapshot.os.replace", interrupt_provenance_publication
    )

    with pytest.raises(OSError, match="simulated publication interruption"):
        run_extraction(
            output_path=extracted,
            index_path=index,
            disable_progress=True,
        )

    assert not provenance_path_for(extracted).exists()
