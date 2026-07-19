"""
Tests for isolation source standardziation

Uses a monkeypatch instead of calling an actual LLM API.
"""

from __future__ import annotations

import csv
import json
from datetime import date
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

import baccurate.standardizers.isolation as isolation_module
from baccurate.dataset_builder import (
    DatasetBuilder,
    DatasetBuildRequest,
    StandardizationAttribute,
)
from baccurate.extraction.tables import COLUMNS
from baccurate.llm.client import LLMSettings
from baccurate.source_snapshot import (
    ArtifactReference,
    DerivedArtifactReferences,
    DerivedBundleProvenance,
    ManifestReference,
    PairedManifestReferences,
    SourceSnapshotError,
    SourceSnapshotManifest,
    bioproject_catalog_path_for,
    provenance_path_for,
    sha256_file,
)
from baccurate.standardizers.host import HostOverflowContext
from baccurate.standardizers.iso_renderer import render_ontology
from baccurate.standardizers.isolation import (
    IsolationDiagnostic,
    IsolationEvidenceLevel,
    IsolationOutcome,
    IsoStandardizer,
    LLMClassifier,
    OntologyManager,
    SQLiteCache,
    StandardizedSource,
)
from baccurate.utils.config import load_config
from baccurate.utils.text import normalize_keyword

ROOT = Path(__file__).parent.parent
CONFIG_PATH = ROOT / "config" / "isolation_source.yaml"
ONTOLOGY_PATH = ROOT / "data" / "reference" / "ontology_terms.tsv"

# term paths used across the tests.
FECES = "host-associated:animal host:feces"
RECTUM = "host-associated:animal host:digestive tract:intestine:rectum"
BLOOD = "host-associated:animal host:bodily fluid:blood"
WOUND = "host-associated:animal host:wound"
SKIN = "host-associated:animal host:skin"
RHIZOSPHERE = "host-associated:plant host:rhizosphere"
SOIL = "environmental:natural environment:terrestrial:soil"

# =============================================================================
# Fake LLM client
# =============================================================================


class _FakeCompletions:
    def __init__(self, parent: FakeClient) -> None:
        self._parent = parent

    def create(self, **kwargs):
        self._parent.calls.append(kwargs)
        behavior = self._parent.behavior
        if behavior is None:
            raise AssertionError("LLM create() was called unexpectedly")
        if isinstance(behavior, Exception):
            raise behavior
        return behavior


class FakeClient:
    def __init__(self) -> None:
        self.calls: list[dict] = []
        self.behavior: object | None = None
        self.chat = SimpleNamespace(completions=_FakeCompletions(self))

    def respond_with(
        self,
        terms,
        reasoning: str = "because",
        evidence_level: str = "sample",
    ) -> None:
        self.behavior = SimpleNamespace(
            terms=list(terms),
            reasoning=reasoning,
            evidence_level=evidence_level,
        )

    def fail_with(self, exc: Exception) -> None:
        self.behavior = exc


def _write_derived_bundle(
    tmp_path: Path,
    *,
    rows: list[dict[str, str]],
    projects: list[dict[str, object]],
) -> tuple[Path, Path]:
    manifest = SourceSnapshotManifest(
        manifest_version=1,
        snapshot_id="biosample-test",
        provider="test",
        retrieved_on=date(2026, 1, 1),
        metadata_reference_date=date(2026, 1, 1),
        files=(
            {
                "name": "biosample.xml.gz",
                "sha256": "0" * 64,
            },
        ),
    )
    manifest_path = tmp_path / "biosample_snapshot.yaml"
    manifest_path.write_text(
        yaml.safe_dump(manifest.model_dump(mode="json"), sort_keys=False),
        encoding="utf-8",
    )

    extracted = tmp_path / "extracted.tsv"
    with extracted.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=COLUMNS, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow(dict.fromkeys(COLUMNS, "") | row)

    catalog = bioproject_catalog_path_for(extracted)
    catalog.write_text(
        "".join(
            json.dumps(project, ensure_ascii=False, separators=(",", ":")) + "\n"
            for project in projects
        ),
        encoding="utf-8",
    )
    DerivedBundleProvenance(
        bundle_version=1,
        source_manifests=PairedManifestReferences(
            biosample=ManifestReference(
                snapshot_id=manifest.snapshot_id,
                path=str(manifest_path),
                sha256=sha256_file(manifest_path),
            ),
            bioproject=ManifestReference(
                snapshot_id="bioproject-test",
                path="bioproject_snapshot.yaml",
                sha256="1" * 64,
            ),
        ),
        artifacts=DerivedArtifactReferences(
            extracted_metadata=ArtifactReference(
                path=extracted.name,
                sha256=sha256_file(extracted),
            ),
            bioproject_context=ArtifactReference(
                path=catalog.name,
                sha256=sha256_file(catalog),
            ),
        ),
    ).write(provenance_path_for(extracted))
    return extracted, manifest_path


def _build_isolation_run(
    tmp_path: Path,
    *,
    rows: list[dict[str, str]],
    projects: list[dict[str, object]],
    fake: FakeClient,
) -> SimpleNamespace:
    tmp_path.mkdir(parents=True, exist_ok=True)
    extracted, manifest = _write_derived_bundle(
        tmp_path,
        rows=rows,
        projects=projects,
    )
    config_path = _isolation_config(tmp_path)
    atb_index = tmp_path / "atb.tsv"
    atb_index.write_text("accession\tin_ATB\tpathogen_ATB\n", encoding="utf-8")

    def isolation_factory(config: Path, bundle: Path, result_logger) -> IsoStandardizer:
        standardizer = IsoStandardizer(
            config,
            bundle,
            result_logger=result_logger,
            client=None,
            llm_settings=LLMSettings(None, None, "test-model"),
        )
        standardizer.pipeline.client = fake
        return standardizer

    final_destination = tmp_path / "final.tsv"
    reasoning_destination = tmp_path / "isolation_reasoning.jsonl"
    report = DatasetBuilder(isolation_standardizer_factory=isolation_factory).build(
        DatasetBuildRequest(
            extracted_metadata=extracted,
            source_snapshot_manifest=manifest,
            requested_pathogens=("ecoli",),
            requested_attributes=(StandardizationAttribute.ISOLATION_SOURCE,),
            final_destination=final_destination,
            isolation_reasoning_destination=reasoning_destination,
            atb_index=atb_index,
            isolation_config=config_path,
            disable_progress=True,
        )
    )
    return SimpleNamespace(
        dataset=final_destination,
        reasoning=reasoning_destination,
        report=report,
    )


def _build_isolation_dataset(
    tmp_path: Path,
    *,
    rows: list[dict[str, str]],
    projects: list[dict[str, object]],
    fake: FakeClient,
) -> Path:
    return _build_isolation_run(
        tmp_path,
        rows=rows,
        projects=projects,
        fake=fake,
    ).dataset


def _empty_extracted_bundle(tmp_path: Path) -> Path:
    extracted = tmp_path / "unit-extracted.tsv"
    bioproject_catalog_path_for(extracted).write_text("", encoding="utf-8")
    return extracted


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture(scope="session")
def ontology() -> OntologyManager:
    return OntologyManager(ONTOLOGY_PATH)


@pytest.fixture(scope="session")
def config() -> dict:
    return load_config(CONFIG_PATH)


@pytest.fixture
def cache(tmp_path) -> SQLiteCache:
    c = SQLiteCache(tmp_path / "iso_cache.db")
    yield c
    c.close()


@pytest.fixture
def feces_source() -> StandardizedSource:
    return StandardizedSource(
        categories="host-associated",
        display_terms="feces",
        ontology_links="ENVO:00002003",
        term_paths=FECES,
        reasoning=[],
    )


@pytest.fixture
def classifier(config, ontology, cache, monkeypatch) -> LLMClassifier:
    # Creaty dummy values for LLMClassifier.__init__
    monkeypatch.setenv("API_KEY", "test")
    monkeypatch.setenv("SERVER", "http://localhost:0")
    clf = LLMClassifier(config, ontology, cache)
    fake = FakeClient()
    clf.client = fake
    clf._fake = fake
    return clf


# =============================================================================
# Record-level outcomes
# =============================================================================

def test_dataset_build_conditionally_supplies_every_resolved_project_in_accession_order(
    tmp_path: Path,
) -> None:
    fake = FakeClient()
    fake.respond_with(["wound"])
    _build_isolation_dataset(
        tmp_path,
        rows=[
            {
                "accession": "SAMN00000002",
                "pathogen": "ecoli",
                "bioproject_id": "3||1||999",
                "bioproject_accession": "PRJNA300||PRJNA100",
                "iso_attr_orig": "isolation_source",
                "iso_val_orig": "wound patient 314159",
            }
        ],
        projects=[
            {
                "id": "3",
                "accession": "PRJNA300",
                "title": "Veterinary wound survey",
                "description": "Animal and environmental sampling.",
                "relevance": ["Environmental", "Veterinary"],
            },
            {
                "id": "1",
                "accession": "PRJNA100",
                "title": "Agricultural surveillance",
                "description": "Farm sampling.",
                "relevance": ["Agricultural"],
            },
        ],
        fake=fake,
    )

    messages = fake.calls[0]["messages"]
    assert "# BioProject context rules" in messages[0]["content"]
    expected_json = json.dumps(
        [
            {
                "id": "1",
                "accession": "PRJNA100",
                "title": "Agricultural surveillance",
                "description": "Farm sampling.",
                "relevance": ["Agricultural"],
            },
            {
                "id": "3",
                "accession": "PRJNA300",
                "title": "Veterinary wound survey",
                "description": "Animal and environmental sampling.",
                "relevance": ["Environmental", "Veterinary"],
            },
        ],
        ensure_ascii=False,
        indent=2,
    )
    assert messages[1]["content"].endswith(f"# BioProject context\n{expected_json}\n")


def test_dataset_build_project_context_changes_the_canonical_request_fingerprint(
    tmp_path: Path,
) -> None:
    fake = FakeClient()
    fake.respond_with(["wound"])
    _build_isolation_dataset(
        tmp_path,
        rows=[
            {
                "accession": "SAMN00000003",
                "pathogen": "ecoli",
                "bioproject_id": "1",
                "bioproject_accession": "PRJNA100",
                "iso_attr_orig": "isolation_source",
                "iso_val_orig": "wound patient 161803",
            },
            {
                "accession": "SAMN00000004",
                "pathogen": "ecoli",
                "bioproject_id": "2",
                "bioproject_accession": "PRJNA200",
                "iso_attr_orig": "isolation_source",
                "iso_val_orig": "wound patient 161803",
            },
        ],
        projects=[
            {
                "id": "1",
                "accession": "PRJNA100",
                "title": "Farm study",
                "description": "Agricultural sampling.",
                "relevance": ["Agricultural"],
            },
            {
                "id": "2",
                "accession": "PRJNA200",
                "title": "Veterinary study",
                "description": "Animal sampling.",
                "relevance": ["Veterinary"],
            },
        ],
        fake=fake,
    )

    # Both samples carry identical isolation evidence and differ only in the
    # linked project. If the rendered context did not reach the fingerprint the
    # second sample would serve a cache hit from the first; two model calls
    # prove the context participates in the cache key.
    assert len(fake.calls) == 2


def test_dataset_build_resolved_project_context_does_not_bypass_direct_sample_matches(
    tmp_path: Path,
) -> None:
    fake = FakeClient()
    final_dataset = _build_isolation_dataset(
        tmp_path,
        rows=[
            {
                "accession": "SAMN00000005",
                "pathogen": "ecoli",
                "bioproject_id": "1",
                "bioproject_accession": "PRJNA100",
                "iso_attr_orig": "isolation_source||specimen",
                "iso_val_orig": "stool||blood",
            }
        ],
        projects=[
            {
                "id": "1",
                "accession": "PRJNA100",
                "title": "Environmental surveillance",
                "description": "A broad One Health study.",
                "relevance": ["Environmental"],
            }
        ],
        fake=fake,
    )

    with final_dataset.open(encoding="utf-8", newline="") as stream:
        row = next(csv.DictReader(stream, delimiter="\t"))
    assert fake.calls == []
    assert row["iso_terms"] == f"{BLOOD}||{FECES}"


def test_dataset_build_classifies_project_only_records(tmp_path: Path) -> None:
    fake = FakeClient()
    fake.respond_with(
        ["soil"],
        reasoning="The project explicitly describes soil sampling.",
        evidence_level="project",
    )

    final_dataset = _build_isolation_dataset(
        tmp_path,
        rows=[
            {
                "accession": "SAMN00000007",
                "pathogen": "ecoli",
                "bioproject_id": "1",
                "bioproject_accession": "PRJNA100",
            }
        ],
        projects=[
            {
                "id": "1",
                "accession": "PRJNA100",
                "title": "Soil isolate survey",
                "description": "Bacteria isolated from agricultural soil.",
                "relevance": ["Agricultural", "Environmental"],
            }
        ],
        fake=fake,
    )

    with final_dataset.open(encoding="utf-8", newline="") as stream:
        row = next(csv.DictReader(stream, delimiter="\t"))
    assert len(fake.calls) == 1
    evidence_schema = fake.calls[0]["response_model"].model_json_schema()["properties"][
        "evidence_level"
    ]
    assert evidence_schema["enum"] == ["project", "none"]
    assert row["iso_terms"] == SOIL


def test_dataset_build_persists_and_aggregates_project_evidence(tmp_path: Path) -> None:
    fake = FakeClient()
    fake.respond_with(["soil"], evidence_level="project")

    run = _build_isolation_run(
        tmp_path,
        rows=[
            {
                "accession": "SAMN00000008",
                "pathogen": "ecoli",
                "bioproject_accession": "PRJNA100",
            }
        ],
        projects=[
            {
                "id": "1",
                "accession": "PRJNA100",
                "title": "Soil survey",
                "description": "Soil sampling.",
                "relevance": ["Environmental"],
            }
        ],
        fake=fake,
    )

    reasoning = json.loads(run.reasoning.read_text(encoding="utf-8"))
    assert reasoning["evidence_level"] == "project"
    assert run.report.isolation.aggregate.evidence_levels == {IsolationEvidenceLevel.PROJECT: 1}


@pytest.mark.parametrize("invalid_evidence_level", ["sample", "sample_and_project", "none"])
def test_dataset_build_rejects_invalid_project_only_evidence_claim(
    tmp_path: Path,
    invalid_evidence_level: str,
) -> None:
    fake = FakeClient()
    fake.respond_with(["soil"], evidence_level=invalid_evidence_level)

    with pytest.raises(
        RuntimeError,
        match="Isolation-source LLM failed for accession SAMN00000009",
    ):
        _build_isolation_run(
            tmp_path,
            rows=[
                {
                    "accession": "SAMN00000009",
                    "pathogen": "ecoli",
                    "bioproject_accession": "PRJNA100",
                }
            ],
            projects=[
                {
                    "id": "1",
                    "accession": "PRJNA100",
                    "title": "Soil survey",
                    "description": "Soil sampling.",
                    "relevance": ["Environmental"],
                }
            ],
            fake=fake,
        )


def test_dataset_build_combines_deterministic_sample_and_project_evidence(
    tmp_path: Path,
) -> None:
    fake = FakeClient()
    fake.respond_with(["soil"], evidence_level="project")

    run = _build_isolation_run(
        tmp_path,
        rows=[
            {
                "accession": "SAMN00000013",
                "pathogen": "ecoli",
                "bioproject_accession": "PRJNA100",
                "iso_attr_orig": "isolation_source||environment",
                "iso_val_orig": "stool||farm material 777",
            }
        ],
        projects=[
            {
                "id": "1",
                "accession": "PRJNA100",
                "title": "Farm soil survey",
                "description": "Soil and animal sampling.",
                "relevance": ["Agricultural", "Environmental"],
            }
        ],
        fake=fake,
    )

    reasoning = json.loads(run.reasoning.read_text(encoding="utf-8"))
    assert reasoning["evidence_level"] == "sample_and_project"
    term_paths = reasoning["term_paths"].split("||")
    assert FECES in term_paths  # deterministic sample match
    assert SOIL in term_paths  # model's project-derived term


def test_dataset_build_preserves_combined_evidence_level(tmp_path: Path) -> None:
    fake = FakeClient()
    fake.respond_with(["wound"], evidence_level="sample_and_project")

    run = _build_isolation_run(
        tmp_path,
        rows=[
            {
                "accession": "SAMN00000010",
                "pathogen": "ecoli",
                "bioproject_accession": "PRJNA100",
                "iso_attr_orig": "isolation_source",
                "iso_val_orig": "clinical material 424242",
            }
        ],
        projects=[
            {
                "id": "1",
                "accession": "PRJNA100",
                "title": "Wound infection survey",
                "description": "Clinical wound isolates.",
                "relevance": ["Veterinary"],
            }
        ],
        fake=fake,
    )

    reasoning = json.loads(run.reasoning.read_text(encoding="utf-8"))
    assert reasoning["evidence_level"] == "sample_and_project"
    evidence_schema = fake.calls[0]["response_model"].model_json_schema()["properties"][
        "evidence_level"
    ]
    assert evidence_schema["enum"] == [
        "sample",
        "project",
        "sample_and_project",
        "none",
    ]


def test_dataset_build_keeps_uninformative_project_only_context_unspecified(
    tmp_path: Path,
) -> None:
    fake = FakeClient()
    fake.respond_with(["unspecified"], evidence_level="project")

    run = _build_isolation_run(
        tmp_path,
        rows=[
            {
                "accession": "SAMN00000011",
                "pathogen": "ecoli",
                "bioproject_accession": "PRJNA100||PRJNA200",
            }
        ],
        projects=[
            {
                "id": "1",
                "accession": "PRJNA100",
                "title": "Mixed surveillance",
                "description": "Clinical and environmental isolates.",
                "relevance": ["Environmental"],
            },
            {
                "id": "2",
                "accession": "PRJNA200",
                "title": "Genome sequencing",
                "description": "No sample origin is reported.",
                "relevance": [],
            },
        ],
        fake=fake,
    )

    with run.dataset.open(encoding="utf-8", newline="") as stream:
        row = next(csv.DictReader(stream, delimiter="\t"))
    reasoning = json.loads(run.reasoning.read_text(encoding="utf-8"))
    assert row["iso_display_term"] == "unspecified"
    assert reasoning["evidence_level"] == "none"
    assert run.report.isolation.aggregate.evidence_levels == {IsolationEvidenceLevel.NONE: 1}


def test_dataset_build_without_sample_or_resolved_project_keeps_no_candidates(
    tmp_path: Path,
) -> None:
    fake = FakeClient()

    run = _build_isolation_run(
        tmp_path,
        rows=[
            {
                "accession": "SAMN00000012",
                "pathogen": "ecoli",
                "bioproject_id": "999",
            }
        ],
        projects=[],
        fake=fake,
    )

    assert fake.calls == []
    assert run.reasoning.read_text(encoding="utf-8") == ""
    assert run.report.isolation.aggregate.rejected == 1
    assert run.report.isolation.aggregate.diagnostics == {IsolationDiagnostic.NO_CANDIDATES: 1}


def test_dataset_build_rejects_invalid_project_context_before_classification(
    tmp_path: Path,
) -> None:
    fake = FakeClient()
    with pytest.raises(
        SourceSnapshotError,
        match="relevance must contain distinct Agricultural, Environmental, or Veterinary",
    ):
        _build_isolation_dataset(
            tmp_path,
            rows=[
                {
                    "accession": "SAMN00000006",
                    "pathogen": "ecoli",
                    "bioproject_id": "1",
                    "bioproject_accession": "PRJNA100",
                    "iso_attr_orig": "isolation_source",
                    "iso_val_orig": "wound patient 141421",
                }
            ],
            projects=[
                {
                    "id": "1",
                    "accession": "PRJNA100",
                    "title": "Medical study",
                    "description": "Clinical sampling.",
                    "relevance": ["Medical"],
                }
            ],
            fake=fake,
        )

    assert fake.calls == []


# =============================================================================
# BioProject catalog loading and validation
# =============================================================================


_VALID_PROJECT: dict[str, object] = {
    "id": "1",
    "accession": "PRJNA100",
    "title": "Farm study",
    "description": "Agricultural sampling.",
    "relevance": ["Agricultural"],
}


def _write_catalog(extracted: Path, text: str) -> Path:
    bioproject_catalog_path_for(extracted).write_text(text, encoding="utf-8")
    return extracted


def _catalog_lines(*records: object) -> str:
    return "".join(
        json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n" for record in records
    )


def test_load_bioproject_catalog_requires_a_paired_catalog_file(tmp_path: Path) -> None:
    with pytest.raises(SourceSnapshotError, match="Invalid BioProject context catalog"):
        isolation_module._load_bioproject_catalog(tmp_path / "extracted.tsv")


def test_load_bioproject_catalog_parses_valid_records(tmp_path: Path) -> None:
    extracted = _write_catalog(
        tmp_path / "extracted.tsv",
        _catalog_lines(
            _VALID_PROJECT,
            {
                "id": "2",
                "accession": "PRJNA200",
                "title": "Vet study",
                "description": "",
                "relevance": ["Environmental", "Veterinary"],
            },
        ),
    )

    catalog = isolation_module._load_bioproject_catalog(extracted)

    assert set(catalog) == {"PRJNA100", "PRJNA200"}
    project = catalog["PRJNA200"]
    assert isinstance(project, isolation_module.ResolvedBioProjectContext)
    assert project.relevance == ("Environmental", "Veterinary")
    assert project.description == ""


@pytest.mark.parametrize(
    ("catalog_text", "match"),
    [
        ("{not valid json}\n", "line 1"),
        (_catalog_lines([1, 2, 3]), "expected fields"),
        (_catalog_lines("PRJNA100"), "expected fields"),
        (
            _catalog_lines({k: v for k, v in _VALID_PROJECT.items() if k != "relevance"}),
            "expected fields",
        ),
        (_catalog_lines(_VALID_PROJECT | {"extra": "x"}), "expected fields"),
        (_catalog_lines(_VALID_PROJECT | {"id": "PRJNA1"}), "project ID must be a numeric string"),
        (_catalog_lines(_VALID_PROJECT | {"id": "١٢٣"}), "project ID must be a numeric string"),
        (
            _catalog_lines(_VALID_PROJECT | {"accession": ""}),
            "accession must be a non-empty string",
        ),
        (_catalog_lines(_VALID_PROJECT | {"title": ""}), "title must be a non-empty string"),
        (_catalog_lines(_VALID_PROJECT | {"title": 5}), "title must be a non-empty string"),
        (_catalog_lines(_VALID_PROJECT | {"description": 0}), "description must be a string"),
        (
            _catalog_lines(_VALID_PROJECT | {"relevance": "Agricultural"}),
            "relevance must contain distinct",
        ),
        (
            _catalog_lines(_VALID_PROJECT | {"relevance": ["Medical"]}),
            "relevance must contain distinct",
        ),
        (
            _catalog_lines(_VALID_PROJECT | {"relevance": ["Agricultural", "Agricultural"]}),
            "relevance must contain distinct",
        ),
        (
            _catalog_lines(_VALID_PROJECT, _VALID_PROJECT | {"accession": "PRJNA200"}),
            "duplicate project ID",
        ),
        (
            _catalog_lines(_VALID_PROJECT, _VALID_PROJECT | {"id": "2"}),
            "duplicate project accession",
        ),
    ],
)
def test_load_bioproject_catalog_rejects_invalid_records(
    tmp_path: Path,
    catalog_text: str,
    match: str,
) -> None:
    extracted = _write_catalog(tmp_path / "extracted.tsv", catalog_text)
    with pytest.raises(SourceSnapshotError, match=match):
        isolation_module._load_bioproject_catalog(extracted)


def test_typed_record_outcome_preserves_context_origins_and_diagnostics(tmp_path, monkeypatch):
    config_path = tmp_path / "isolation.yaml"
    config_path.write_text(
        CONFIG_PATH.read_text(encoding="utf-8")
        + f'\ncache_db_path: "{(tmp_path / "iso-cache.db").as_posix()}"\n',
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "baccurate.standardizers.isolation.load_llm_client",
        lambda *_args: (None, None),
    )
    standardizer = IsoStandardizer(config_path, _empty_extracted_bundle(tmp_path))

    try:
        result = standardizer.standardize(
            {
                "accession": "SAME_ACCESSION",
                "iso_attr_orig": "isolation_source",
                "iso_val_orig": "stool",
            },
            host_context="Homo sapiens",
            overflow=HostOverflowContext(
                attribute="host sample",
                value="blood",
            ),
        )
        with pytest.raises(ValueError, match="2 attributes for 1 values"):
            standardizer.standardize(
                {
                    "accession": "MALFORMED",
                    "iso_attr_orig": "isolation_source||tissue",
                    "iso_val_orig": "blood",
                },
                host_context="",
            )
        fake_client = FakeClient()
        fake_client.fail_with(RuntimeError("boom"))
        standardizer.pipeline.client = fake_client
        with pytest.raises(
            RuntimeError,
            match="Isolation-source LLM failed for accession TYPED_FAILURE",
        ):
            standardizer.standardize(
                {
                    "accession": "TYPED_FAILURE",
                    "iso_attr_orig": "isolation_source",
                    "iso_val_orig": "venous draw",
                },
                host_context="",
            )
    finally:
        standardizer.close()

    assert isinstance(result, IsolationOutcome)
    assert result.host_context == "Homo sapiens"
    assert [(origin.attribute, origin.value) for origin in result.origins] == [
        ("isolation_source", "stool"),
        ("host sample", "blood"),
    ]
    assert FECES in result.term_paths.split("||")
    assert BLOOD in result.term_paths.split("||")
    assert result.display_terms != "unspecified"
    assert result.ontology_links
    assert result.exact_matches == 2
    assert result.cache_hits == 0
    assert result.llm_calls == 0
    assert result.evidence_level is IsolationEvidenceLevel.SAMPLE
    assert result.diagnostics == (IsolationDiagnostic.EXACT_MATCH,)


def test_partial_deterministic_result_uses_sample_evidence_when_llm_is_disabled(
    tmp_path: Path,
) -> None:
    config_path = _isolation_config(tmp_path)
    standardizer = IsoStandardizer(
        config_path,
        _empty_extracted_bundle(tmp_path),
        client=None,
        llm_settings=LLMSettings(None, None, "test-model"),
    )

    try:
        result = standardizer.standardize(
            {
                "accession": "PARTIAL_DETERMINISTIC",
                "iso_attr_orig": "isolation_source||environment",
                "iso_val_orig": "stool||unresolved material 12345",
            },
            host_context="",
        )
    finally:
        standardizer.close()

    assert isinstance(result, IsolationOutcome)
    assert result.term_paths == FECES
    assert result.evidence_level is IsolationEvidenceLevel.SAMPLE


def test_typed_record_outcome_preserves_model_reasoning_on_exact_cache_hit(tmp_path, monkeypatch):
    config_path = tmp_path / "isolation.yaml"
    config_path.write_text(
        CONFIG_PATH.read_text(encoding="utf-8")
        + f'\ncache_db_path: "{(tmp_path / "iso-cache.db").as_posix()}"\n',
        encoding="utf-8",
    )
    monkeypatch.setenv("LLM_MODEL", "test-model")
    extracted = _empty_extracted_bundle(tmp_path)
    first_standardizer = IsoStandardizer(config_path, extracted, client=None)
    fake_client = FakeClient()
    fake_client.respond_with(["wound"], reasoning="clinical wound")
    first_standardizer.pipeline.client = fake_client

    try:
        modelled = first_standardizer.standardize(
            {
                "accession": "MODELLED",
                "iso_attr_orig": "isolation_source",
                "iso_val_orig": "wound patient 1",
            },
            host_context="Homo sapiens",
        )
    finally:
        first_standardizer.close()

    cached_standardizer = IsoStandardizer(config_path, extracted, client=None)
    try:
        cached = cached_standardizer.standardize(
            {
                "accession": "CACHED",
                "iso_attr_orig": "isolation_source",
                "iso_val_orig": "wound patient 1",
            },
            host_context="Homo sapiens",
        )
    finally:
        cached_standardizer.close()

    assert isinstance(modelled, IsolationOutcome)
    assert modelled.evidence_level is IsolationEvidenceLevel.SAMPLE
    assert modelled.diagnostics == (IsolationDiagnostic.LLM_CALL,)
    assert modelled.reasoning == (
        {
            "node": "classifier",
            "reasoning": "clinical wound",
            "selections": [WOUND],
            "selected_terms": ["wound"],
        },
    )
    assert isinstance(cached, IsolationOutcome)
    assert cached.term_paths == modelled.term_paths
    assert cached.evidence_level is IsolationEvidenceLevel.SAMPLE
    assert cached.reasoning == modelled.reasoning
    assert cached.diagnostics == (IsolationDiagnostic.CACHE_HIT,)


def _isolation_config(tmp_path: Path, *, suffix: str = "") -> Path:
    config_path = tmp_path / f"isolation-{len(list(tmp_path.glob('isolation-*.yaml')))}.yaml"
    config_path.write_text(
        CONFIG_PATH.read_text(encoding="utf-8")
        + f'\ncache_db_path: "{(tmp_path / "isolation-cache.db").as_posix()}"\n'
        + suffix,
        encoding="utf-8",
    )
    return config_path


def _standardize_model_isolation(
    config_path: Path,
    fake: FakeClient,
    *,
    model: str = "test-model",
    schema_override: object | None = None,
) -> IsolationOutcome:
    standardizer = IsoStandardizer(
        config_path,
        _empty_extracted_bundle(config_path.parent),
        client=None,
        llm_settings=LLMSettings(None, None, model),
    )
    standardizer.pipeline.client = fake
    if schema_override is not None:
        standardizer.pipeline._response_schemas[
            isolation_module._IsolationRequestMode.SAMPLE_ONLY
        ] = schema_override
    try:
        result = standardizer.standardize(
            {
                "accession": "REQUEST_FINGERPRINT",
                "iso_attr_orig": "isolation_source",
                "iso_val_orig": "wound patient 739105",
            },
            host_context="Homo sapiens",
        )
    finally:
        standardizer.close()
    assert isinstance(result, IsolationOutcome)
    return result


def test_isolation_cache_reuses_identical_request_when_only_prompt_metadata_changes(tmp_path):
    fake = FakeClient()
    fake.respond_with(["wound"], reasoning="clinical wound")
    first_config = _isolation_config(tmp_path, suffix="prompt_version: first\n")
    second_config = _isolation_config(tmp_path, suffix="prompt_version: second\n")

    first = _standardize_model_isolation(first_config, fake)
    second = _standardize_model_isolation(second_config, fake)

    assert first.diagnostics == (IsolationDiagnostic.LLM_CALL,)
    assert second.diagnostics == (IsolationDiagnostic.CACHE_HIT,)
    assert second.reasoning == first.reasoning
    assert len(fake.calls) == 1


@pytest.mark.parametrize(
    "changed_component",
    ["message", "model", "parameter", "schema_id", "schema_contract"],
)
def test_isolation_cache_misses_when_canonical_request_changes(
    tmp_path, monkeypatch, changed_component
):
    fake = FakeClient()
    fake.respond_with(["wound"], reasoning="clinical wound")
    first_config = _isolation_config(tmp_path)
    _standardize_model_isolation(first_config, fake)

    second_config = _isolation_config(tmp_path)
    second_model = "test-model"
    schema_override = None
    if changed_component == "message":
        second_config = _isolation_config(
            tmp_path,
            suffix="system_prompt: A changed fully rendered system prompt.\n",
        )
    elif changed_component == "model":
        second_model = "changed-model"
    elif changed_component == "parameter":
        monkeypatch.setattr(
            isolation_module,
            "ISOLATION_LLM_PARAMETERS",
            {"temperature": 0, "seed": 101},
        )
    elif changed_component == "schema_id":
        monkeypatch.setattr(
            isolation_module,
            "ISOLATION_RESPONSE_SCHEMA_ID",
            "baccurate.isolation.classification.changed",
        )
    else:

        class ChangedResponseModel:
            @classmethod
            def model_json_schema(cls):
                return {
                    "type": "object",
                    "properties": {"changed": {"type": "string"}},
                    "required": ["changed"],
                }

        schema_override = ChangedResponseModel

    second = _standardize_model_isolation(
        second_config,
        fake,
        model=second_model,
        schema_override=schema_override,
    )

    assert second.diagnostics == (IsolationDiagnostic.LLM_CALL,)
    assert len(fake.calls) == 2


# =============================================================================
# SQLite cache
# =============================================================================


def test_cache_does_not_reuse_changed_request_fingerprint(cache, feces_source):
    cache.set("fingerprint-a", feces_source)

    assert cache.get("fingerprint-b") is None


def test_cache_reuses_the_same_request_fingerprint(cache, feces_source):
    cache.set("fingerprint", feces_source)

    assert cache.get("fingerprint") == feces_source


def test_set_then_get_round_trips_the_record(cache):
    rec = StandardizedSource(
        categories="host-associated",
        display_terms="feces",
        ontology_links="ENVO:00002003",
        term_paths=FECES,
        reasoning=[{"node": "direct_match", "reasoning": "x", "selections": [FECES]}],
    )
    cache.set("fingerprint", rec)

    got = cache.get("fingerprint")

    assert got is not None
    assert got.categories == "host-associated"
    assert got.display_terms == "feces"
    assert got.ontology_links == "ENVO:00002003"
    assert got.term_paths == FECES
    assert got.reasoning == [{"node": "direct_match", "reasoning": "x", "selections": [FECES]}]


def test_get_on_unknown_key_returns_none(cache):
    assert cache.get("never-stored") is None


# =============================================================================
# Ontology parsing
# =============================================================================


def test_tree_edges_follow_the_colon_structure(ontology):
    assert "host-associated" in ontology.children_map[""]
    assert FECES in ontology.children_map["host-associated:animal host"]


def test_node_metadata_captures_display_link_and_synonyms(ontology):
    meta = ontology.node_metadata[FECES]

    assert meta["display_term"] == "feces"
    assert "ENVO:00002003" in meta["ontology_link"]
    assert "stool" in meta["synonyms"]


def test_exact_match_index_resolves_display_terms_and_synonyms(ontology):
    assert ontology.exact_match_index[normalize_keyword("feces")] == FECES
    assert ontology.exact_match_index[normalize_keyword("stool")] == FECES  # synonym
    assert ontology.exact_match_index[normalize_keyword("rectal swab")] == RECTUM  # synonym


def test_id_match_index_is_upper_cased_and_split(ontology):
    assert ontology.id_match_index["ENVO:02000020"] == BLOOD


def test_display_term_to_path_maps_back_to_canonical_path(ontology):
    assert ontology.display_term_to_path[normalize_keyword("blood")] == BLOOD


def test_crosslinks_resolve_to_target_paths(ontology):
    assert SOIL in ontology.crosslink_map[RHIZOSPHERE]


# =============================================================================
# Direct match
# =============================================================================


def test_direct_match_by_exact_display_term(classifier):
    assert classifier._direct_match("feces") == FECES


def test_direct_match_by_synonym(classifier):
    assert classifier._direct_match("stool") == FECES
    assert classifier._direct_match("rectal swab") == RECTUM


def test_direct_match_by_ontology_id_in_free_text(classifier):
    assert classifier._direct_match("sample drawn, ENVO:02000020 present") == BLOOD


def test_direct_match_returns_none_for_unknown_value(classifier):
    assert classifier._direct_match("totally unknown value") is None


# =============================================================================
# Metadata formatting
# =============================================================================


def test_format_metadata_includes_host_when_present():
    out = LLMClassifier._format_metadata(["isolation_source"], ["blood"], "Homo sapiens")

    assert "isolation_source = blood" in out
    assert "host = Homo sapiens" in out


def test_format_metadata_omits_blank_host():
    out = LLMClassifier._format_metadata(["isolation_source"], ["blood"], "   ")

    assert "host =" not in out


def test_system_prompt_renders_ontology_without_altering_json_examples(classifier):
    assert "{ontology_tree}" not in classifier.system_prompt
    assert '{"reasoning":' in classifier.system_prompt


# =============================================================================
# standardize_record: empty and trusted inputs
# =============================================================================


def test_standardizer_does_not_reapply_extraction_rejection(classifier):
    classifier._fake.respond_with(["unspecified"])

    res = classifier.standardize_record("T", "isolation_source", "GENOMIC", "")

    assert res.display_terms == "unspecified"
    assert res.categories == "unspecified"
    assert res.ontology_links == "NA"
    assert classifier.stats["llm_calls"] == 1
    assert len(classifier._fake.calls) == 1


def test_empty_value_resolves_to_unspecified_without_llm(classifier):
    res = classifier.standardize_record("T", "isolation_source", "   ", "")

    assert res.display_terms == "unspecified"
    assert classifier._fake.calls == []


# =============================================================================
# standardize_record: direct matching
# =============================================================================


def test_direct_feces_match_does_not_infer_anatomical_site(classifier):
    res = classifier.standardize_record("T", "isolation_source", "stool", "")

    terms = res.display_terms.split("||")
    paths = res.term_paths.split("||")
    assert "feces" in terms
    assert "rectum" not in terms
    assert FECES in paths
    assert RECTUM not in paths
    assert res.categories == "host-associated"
    assert classifier.stats["llm_calls"] == 0
    assert classifier.stats["exact_matches"] >= 1
    assert classifier._fake.calls == []


def test_all_values_direct_matching_skips_llm(classifier):
    res = classifier.standardize_record("T", "isolation_source||tissue", "stool||blood", "")

    paths = res.term_paths.split("||")
    assert FECES in paths
    assert BLOOD in paths
    assert classifier.stats["llm_calls"] == 0
    assert classifier._fake.calls == []


def test_synonymous_values_direct_matching_the_same_path_skip_llm(classifier):
    res = classifier.standardize_record(
        "T", "isolation_source||sample_material", "stool||feces", ""
    )

    assert FECES in res.term_paths.split("||")
    assert classifier.stats["exact_matches"] == 2
    assert classifier.stats["llm_calls"] == 0
    assert classifier._fake.calls == []


# =============================================================================
# standardize_record: LLM branch
# =============================================================================


def test_llm_selection_is_mapped_to_term_path(classifier):
    classifier._fake.respond_with(["blood"], reasoning="venous blood")
    res = classifier.standardize_record("T", "isolation_source", "venous draw", "")

    assert classifier.stats["llm_calls"] == 1
    user_prompt = classifier._fake.calls[0]["messages"][1]["content"]
    assert "isolation_source = venous draw" in user_prompt
    assert "{metadata}" not in user_prompt
    assert "{bioproject_context}" not in user_prompt
    assert "blood" in res.display_terms.split("||")
    assert BLOOD in res.term_paths.split("||")
    nodes = [h.get("node") for h in res.reasoning]
    assert "classifier" in nodes
    classifier_entry = next(h for h in res.reasoning if h.get("node") == "classifier")
    assert classifier_entry["selected_terms"] == ["blood"]


def test_llm_feces_selection_does_not_infer_anatomical_site(classifier):
    classifier._fake.respond_with(["feces"])
    res = classifier.standardize_record("T", "isolation_source", "gut contents", "")

    terms = res.display_terms.split("||")
    assert "feces" in terms
    assert "rectum" not in terms
    nodes = [h.get("node") for h in res.reasoning]
    assert "crosslink" not in nodes


def test_direct_and_llm_paths_are_unioned(classifier):
    classifier._fake.respond_with(["blood"])
    res = classifier.standardize_record("T", "isolation_source||tissue", "stool||venous draw", "")

    paths = res.term_paths.split("||")
    assert FECES in paths  # from direct match
    assert BLOOD in paths  # from LLM
    assert classifier.stats["llm_calls"] == 1


# =============================================================================
# standardize_record: expected-mismatch warnings
# =============================================================================


def test_expected_node_absence_is_not_logged_per_record(classifier, caplog):
    classifier._fake.respond_with(["blood"])
    with caplog.at_level("WARNING"):
        res = classifier.standardize_record("ACC", "food_source", "mystery", "")

    assert not any("attribute_shortcut mismatch" in r.getMessage() for r in caplog.records)
    assert "food" not in res.display_terms.split("||")


def test_no_warning_for_descendant_term(classifier, caplog):
    classifier._fake.respond_with(["meat product"])
    with caplog.at_level("WARNING"):
        classifier.standardize_record("ACC", "food_source", "ground beef", "")

    assert not any("attribute_shortcut mismatch" in r.getMessage() for r in caplog.records)


# =============================================================================
# standardize_record: LLM failure handling
# =============================================================================


def test_model_failure_aborts_with_accession(classifier):
    classifier._fake.fail_with(RuntimeError("boom"))

    with pytest.raises(
        RuntimeError,
        match="Isolation-source LLM failed for accession EXTERNAL_FAILURE",
    ):
        classifier.standardize_record(
            "EXTERNAL_FAILURE",
            "isolation_source",
            "venous draw",
            "",
        )


# =============================================================================
# standardize_record: caching
# =============================================================================


def test_identical_model_evidence_hits_cache_without_reinvoking_model(classifier):
    classifier._fake.respond_with(["wound"], reasoning="clinical wound")
    first = classifier.standardize_record("T1", "isolation_source", "wound patient 1", "")
    classifier._fake.behavior = None

    second = classifier.standardize_record("T2", "isolation_source", "wound patient 1", "")

    assert classifier.stats["cache_hits"] == 1
    assert len(classifier._fake.calls) == 1
    assert second == first
    assert second.reasoning == [
        {
            "node": "classifier",
            "reasoning": "clinical wound",
            "selections": [WOUND],
            "selected_terms": ["wound"],
        }
    ]


def test_changed_numeric_evidence_uses_model_again(classifier):
    classifier._fake.respond_with(["wound"])
    classifier.standardize_record("T", "isolation_source", "wound patient 1", "")
    classifier.standardize_record("T", "isolation_source", "wound patient 2", "")

    assert len(classifier._fake.calls) == 2
    assert classifier.stats["cache_hits"] == 0
