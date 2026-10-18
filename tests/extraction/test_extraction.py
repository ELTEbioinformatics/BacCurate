"""BioSample metadata extraction and publication contracts."""

import csv
from datetime import date
from pathlib import Path
from typing import Any, Protocol

import pytest

import baccurate.extraction as extraction
from baccurate import paths
from baccurate.extraction import (
    ExtractionReport,
    SelectionPolicy,
)
from baccurate.provenance.extracted_metadata import (
    provenance_path_for,
    validate_extracted_metadata,
)

ROOT = Path(__file__).parents[2]


class _PairedSnapshotPaths(Protocol):
    biosample_snapshot_path: Path
    bioproject_snapshot_path: Path
    biosample_manifest_path: Path
    bioproject_manifest_path: Path


def run_extraction(**kwargs: Any) -> ExtractionReport:
    """Call extraction through its injected selection-policy interface."""
    kwargs.setdefault(
        "selection_policy",
        SelectionPolicy.load(ROOT / "config" / "selection.yaml"),
    )
    return extraction.run_extraction(**kwargs)


def _configure_internal_paths(
    monkeypatch: pytest.MonkeyPatch,
    snapshots: _PairedSnapshotPaths,
) -> None:
    monkeypatch.setattr(paths, "DEFAULT_BIOSAMPLE_SNAPSHOT_PATH", snapshots.biosample_snapshot_path)
    monkeypatch.setattr(
        paths, "DEFAULT_BIOPROJECT_SNAPSHOT_PATH", snapshots.bioproject_snapshot_path
    )
    monkeypatch.setattr(paths, "DEFAULT_BIOSAMPLE_MANIFEST_PATH", snapshots.biosample_manifest_path)
    monkeypatch.setattr(
        paths, "DEFAULT_BIOPROJECT_MANIFEST_PATH", snapshots.bioproject_manifest_path
    )


def test_extraction_uses_internal_snapshots_without_raw_arguments(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    paired_snapshots,
) -> None:
    snapshots = paired_snapshots
    index = tmp_path / "biosample_index.tsv"
    index.write_text("accession\ttaxon_biosample\n", encoding="utf-8")
    _configure_internal_paths(monkeypatch, snapshots)

    report = run_extraction(
        output_path=tmp_path / "extracted.tsv",
        index_path=index,
    )

    assert report.prepared_input_paths == (
        snapshots.biosample_snapshot_path,
        snapshots.bioproject_snapshot_path,
    )
    assert report.biosample_snapshot_id == "biosample-test"
    assert report.bioproject_snapshot_id == "bioproject-test"


def test_extraction_validates_snapshot_pair_before_other_work(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    paired_snapshots,
) -> None:
    snapshots = paired_snapshots
    snapshots.bioproject_snapshot_path.unlink()
    _configure_internal_paths(monkeypatch, snapshots)
    output = tmp_path / "extracted.tsv"

    with pytest.raises(FileNotFoundError):
        run_extraction(
            output_path=output,
            index_path=tmp_path / "missing-index.tsv.gz",
        )

    assert not output.exists()


def test_extraction_report_carries_both_validated_source_identities(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    paired_snapshots,
) -> None:
    snapshots = paired_snapshots
    _configure_internal_paths(monkeypatch, snapshots)
    index = tmp_path / "biosample_index.tsv"
    index.write_text("accession\ttaxon_biosample\n", encoding="utf-8")

    report = run_extraction(
        output_path=tmp_path / "extracted.tsv",
        index_path=index,
    )

    assert report.biosample_snapshot_id == "biosample-test"
    assert report.bioproject_snapshot_id == "bioproject-test"
    assert report.metadata_reference_date == date(2026, 7, 19)
    assert report.prepared_input_paths == (
        snapshots.biosample_snapshot_path,
        snapshots.bioproject_snapshot_path,
    )


def test_extraction_reports_inclusion_routes_for_records_present_in_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    paired_snapshots,
) -> None:
    snapshots = paired_snapshots
    snapshots.replace_contents(
        biosample_xml=b"""\
<BioSampleSet>
  <BioSample accession="taxonomy">
    <Description><Organism taxonomy_name="Escherichia coli" /></Description>
    <Attributes><Attribute attribute_name="isolation_source">soil</Attribute></Attributes>
  </BioSample>
  <BioSample accession="atb">
    <Description><Organism taxonomy_name="Escherichia coli O157" /></Description>
    <Attributes><Attribute attribute_name="isolation_source">water</Attribute></Attributes>
  </BioSample>
  <BioSample accession="reassigned">
    <Description><Organism taxonomy_name="Enterococcus faecium" /></Description>
    <Attributes><Attribute attribute_name="isolation_source">blood</Attribute></Attributes>
  </BioSample>
  <BioSample accession="agreeing"><Attributes>
    <Attribute attribute_name="isolation_source">stool</Attribute>
  </Attributes></BioSample>
</BioSampleSet>
""",
        bioproject_xml=b"<PackageSet />",
    )
    _configure_internal_paths(monkeypatch, snapshots)
    index = tmp_path / "biosample_index.tsv"
    index.write_text(
        "accession\ttaxon_biosample\torganism_value\tsylph_species\n"
        "taxonomy\tecoli\tEscherichia coli\tNA\n"
        "atb\tNA\tNA\tEscherichia coli\n"
        "reassigned\tefaecium\tEnterococcus faecium\tEscherichia coli\n"
        "agreeing\tecoli\tEscherichia coli\tEscherichia coli\n"
        "absent-from-snapshot\tNA\tNA\tEscherichia coli\n",
        encoding="utf-8",
    )
    extracted = tmp_path / "extracted.tsv"

    report = run_extraction(
        output_path=extracted,
        index_path=index,
    )

    # The agreeing record reassigns nothing, so it counts against the NCBI taxonomy route.
    assert report.extracted_record_count == 4
    assert report.inclusion_route_counts == {
        "biosample_taxonomy": 2,
        "allthebacteria": 2,
    }
    with extracted.open(newline="", encoding="utf-8") as stream:
        published = {row["accession"]: row for row in csv.DictReader(stream, delimiter="\t")}
    assert published["reassigned"]["taxon_key"] == "ecoli"
    assert published["reassigned"]["ncbi_organism"] == "Enterococcus faecium"
    assert published["reassigned"]["sylph_species"] == "Escherichia coli"
    assert published["taxonomy"]["sylph_species"] == "NA"
    assert published["atb"]["ncbi_organism"] == "Escherichia coli O157"
    assert published["agreeing"]["ncbi_organism"] == "NA"


def test_extraction_report_preserves_production_selection_review_counters(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    paired_snapshots,
) -> None:
    snapshots = paired_snapshots
    biosample_xml = b"""\
<BioSampleSet>
  <BioSample accession="SAMN1"><Attributes>
    <Attribute attribute_name="host">unknown</Attribute>
  </Attributes></BioSample>
  <BioSample accession="SAMN2"><Attributes>
    <Attribute attribute_name="collection_date">missing</Attribute>
  </Attributes></BioSample>
  <BioSample accession="SAMN3"><Attributes>
    <Attribute attribute_name="isolation_source">GENOMIC</Attribute>
  </Attributes></BioSample>
  <BioSample accession="SAMN4"><Attributes>
    <Attribute attribute_name="host">unkmowm</Attribute>
  </Attributes></BioSample>
  <BioSample accession="SAMN5"><Attributes>
    <Attribute attribute_name="host_environment">human</Attribute>
  </Attributes></BioSample>
  <BioSample accession="SAMN6"><Attributes>
    <Attribute attribute_name="collection_date">aerobic</Attribute>
  </Attributes></BioSample>
  <BioSample accession="SAMN7"><Attributes>
    <Attribute attribute_name="nat-host" harmonized_name="host">unknown</Attribute>
  </Attributes></BioSample>
</BioSampleSet>
"""
    snapshots.replace_contents(
        biosample_xml=biosample_xml,
        bioproject_xml=b"<PackageSet />",
    )
    _configure_internal_paths(monkeypatch, snapshots)
    index = tmp_path / "biosample_index.tsv"
    index.write_text("accession\ttaxon_biosample\n", encoding="utf-8")

    report = run_extraction(
        output_path=tmp_path / "extracted.tsv",
        index_path=index,
        selection_policy=SelectionPolicy.load(ROOT / "config" / "selection.yaml"),
    )

    assert report.unreviewed_count == 2
    assert report.uncertain_count == 1
    assert report.automatic_rejection_counts == {
        "date": {"non_date_evidence": 1, "universal_missing": 1},
        "host": {"universal_missing": 2},
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
    policy = SelectionPolicy.load(ROOT / "config" / "selection.yaml")

    decision = policy.evaluate(attribute="collection_date", value=value)

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
    policy = SelectionPolicy.load(ROOT / "config" / "selection.yaml")

    decision = policy.evaluate(attribute="collection_date", value=value)

    assert [match.target for match in decision.matches] == ["date"]
    assert decision.events == ()


def test_extraction_writes_extracted_metadata_provenance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    paired_snapshots,
) -> None:
    snapshots = paired_snapshots
    biosample_xml = b"""\
<BioSampleSet>
  <BioSample accession="SAMN00000001" last_update="2025-06-30T12:34:56.000">
    <Attributes>
      <Attribute attribute_name="isolation_source">farm soil</Attribute>
      <Attribute attribute_name="strain" harmonized_name="strain">K-12 MG1655</Attribute>
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
    snapshots.replace_contents(
        biosample_xml=biosample_xml,
        bioproject_xml=bioproject_xml,
    )
    _configure_internal_paths(monkeypatch, snapshots)
    index = tmp_path / "biosample_index.tsv"
    index.write_text(
        "accession\ttaxon_biosample\tsra_run_accessions\t"
        "genbank_assembly_accessions\trefseq_assembly_accessions\n"
        "SAMN00000001\tecoli\tSRR1,SRR2\tGCA_1.1\tNA\n",
        encoding="utf-8",
    )
    extracted = tmp_path / "custom_metadata.tsv"

    report = run_extraction(
        output_path=extracted,
        index_path=index,
    )

    provenance = provenance_path_for(extracted)
    assert report.extracted_metadata_provenance_path == provenance
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
    assert rows[0]["strain"] == "K-12 MG1655"
    assert "title" not in rows[0]
    assert "description" not in rows[0]

    snapshot_pair = validate_extracted_metadata(
        extracted,
        snapshots.biosample_manifest_path,
        snapshots.bioproject_manifest_path,
    )
    assert snapshot_pair.biosample.snapshot_id == "biosample-test"
    assert snapshot_pair.bioproject.snapshot_id == "bioproject-test"


def test_extraction_preserves_linked_project_sets_and_logs_unresolved_count(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    paired_snapshots,
) -> None:
    snapshots = paired_snapshots
    samples = "".join(
        f"""
  <BioSample accession="{accession}">
    <Attributes><Attribute attribute_name="isolation_source">soil</Attribute></Attributes>
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
    snapshots.replace_contents(
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
    _configure_internal_paths(monkeypatch, snapshots)
    index = tmp_path / "biosample_index.tsv"
    index.write_text(
        "accession\ttaxon_biosample\n"
        "SAMN00000001\tecoli\n"
        "SAMN00000002\tecoli\n"
        "SAMN00000003\tecoli\n"
        "SAMN00000004\tecoli\n"
        "SAMN00000005\tecoli\n",
        encoding="utf-8",
    )
    extracted = tmp_path / "linked_sets.tsv"

    with caplog.at_level("INFO", logger="baccurate.extraction.extract"):
        report = run_extraction(
            output_path=extracted,
            index_path=index,
        )

    with extracted.open(newline="", encoding="utf-8") as stream:
        rows = {row["accession"]: row for row in csv.DictReader(stream, delimiter="\t")}
    assert rows["SAMN00000001"]["bioproject_accession"] == ("PRJNA100||PRJNA200||PRJNA300")
    assert rows["SAMN00000002"]["bioproject_accession"] == "PRJNA200"
    assert rows["SAMN00000003"]["bioproject_accession"] == ""
    assert rows["SAMN00000004"]["bioproject_accession"] == ""
    assert "SAMN00000005" not in rows
    assert not (tmp_path / "unresolved_bioproject_links.tsv").exists()
    assert "unresolved_bioproject_links" not in report.review_worklist_paths
    assert [
        record.getMessage() for record in caplog.records if "BioProject IDs" in record.getMessage()
    ] == ["Linked BioProject IDs without an accession in the BioProject XML: 2"]


def test_equivalent_link_and_project_order_produces_byte_stable_data_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    paired_snapshots,
) -> None:
    snapshots = paired_snapshots
    _configure_internal_paths(monkeypatch, snapshots)
    index = tmp_path / "biosample_index.tsv"
    index.write_text(
        "accession\ttaxon_biosample\nSAMN00000001\tecoli\n",
        encoding="utf-8",
    )
    sample_template = """\
<BioSampleSet><BioSample accession="SAMN00000001">
  <Attributes><Attribute attribute_name="isolation_source">soil</Attribute></Attributes>
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
        snapshots.replace_contents(
            biosample_xml=sample_template.format(links=links).encode(),
            bioproject_xml=f"<PackageSet>{projects}</PackageSet>".encode(),
        )
        extracted = tmp_path / f"{stem}.tsv"
        run_extraction(output_path=extracted, index_path=index)
        outputs.append(extracted.read_bytes())

    assert outputs[0] == outputs[1]


def test_extraction_flags_how_each_location_pair_matched(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    paired_snapshots,
) -> None:
    snapshots = paired_snapshots
    snapshots.replace_contents(
        biosample_xml=b"""\
<BioSampleSet>
  <BioSample accession="matched"><Attributes>
    <Attribute attribute_name="note">Germany: Berlin</Attribute>
    <Attribute attribute_name="geo_loc_name">USA: CA</Attribute>
  </Attributes></BioSample>
</BioSampleSet>
""",
        bioproject_xml=b"<PackageSet />",
    )
    _configure_internal_paths(monkeypatch, snapshots)
    index = tmp_path / "biosample_index.tsv"
    index.write_text(
        "accession\ttaxon_biosample\nmatched\tecoli\n",
        encoding="utf-8",
    )
    extracted = tmp_path / "extracted.tsv"

    run_extraction(output_path=extracted, index_path=index)

    with extracted.open(newline="", encoding="utf-8") as stream:
        row = next(csv.DictReader(stream, delimiter="\t"))
    assert row["loc_attr_orig"] == "note||geo_loc_name"
    assert row["loc_matched_by"] == "value||name"


@pytest.mark.parametrize(
    "attribute",
    [
        "geographic location (latitude)",
        "geographic location (longitude)",
        "original geographic location (latitude)",
        "original geographic location (longitude)",
        "latitude",
        "longitude",
        "lat",
        "lon",
    ],
)
def test_coordinate_halves_are_selected_for_location(attribute: str) -> None:
    policy = SelectionPolicy.load(ROOT / "config" / "selection.yaml")

    decision = policy.evaluate(attribute=attribute, value="51.34")

    assert [match.target for match in decision.matches] == ["loc"]


@pytest.mark.parametrize(
    "attribute",
    ["latitude start", "latitude end", "longitude start", "longitude end"],
)
def test_transect_endpoints_are_not_selected_for_location(attribute: str) -> None:
    policy = SelectionPolicy.load(ROOT / "config" / "selection.yaml")

    decision = policy.evaluate(attribute=attribute, value="51.34")

    assert decision.matches == ()


def test_a_quoted_duplicate_pair_is_yielded_once() -> None:
    from lxml import etree

    from baccurate.extraction.xml import _iter_biosample_pairs_with_xml_element

    record = etree.fromstring(
        "<BioSample><Attributes>"
        "<Attribute attribute_name='geo_loc_name'>USA: Madison, WI</Attribute>"
        "<Attribute attribute_name=\"'geo_loc_name'\">USA: Madison, WI</Attribute>"
        "<Attribute attribute_name=\"'*host'\">Homo sapiens</Attribute>"
        "<Attribute attribute_name='collection_date'>2020-01-02</Attribute>"
        "<Attribute attribute_name='collection_date'>2021-03-04</Attribute>"
        "</Attributes></BioSample>"
    )

    pairs = [(name, value) for name, value, _ in _iter_biosample_pairs_with_xml_element(record)]

    assert pairs == [
        ("geo_loc_name", "USA: Madison, WI"),
        ("'*host'", "Homo sapiens"),
        ("collection_date", "2020-01-02"),
        ("collection_date", "2021-03-04"),
    ]


def test_record_level_dates_are_not_counted_as_submitted_pairs() -> None:
    from lxml import etree

    from baccurate.extraction.xml import SelectionCounters, parse_xml

    record = etree.fromstring(
        '<BioSample accession="SAMN1"'
        ' submission_date="2026-08-15T14:07:09.233"'
        ' last_update="2026-08-13T13:04:31.000"'
        ' publication_date="2026-08-13T00:00:00.000">'
        "<Attributes>"
        "<Attribute attribute_name='collection_date'>2020-01-02</Attribute>"
        "</Attributes></BioSample>"
    )
    policy = SelectionPolicy.load(ROOT / "config" / "selection.yaml")
    counters = SelectionCounters()

    decisions = parse_xml(record, policy.evaluate, check_root_attributes=True, counters=counters)

    assert len(decisions) == 4
    assert counters.inspected == 1
    assert counters.selected == 1
    assert counters.identified == 1
