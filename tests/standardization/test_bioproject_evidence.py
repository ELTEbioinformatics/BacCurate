"""Protect BioProject evidence and cross-target dataset-build behavior."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from baccurate.adapters.llm.client import LLMSettings
from baccurate.provenance.source_snapshot import SourceSnapshotError
from baccurate.run.dataset_builder import DatasetBuildRequest
from baccurate.standardization.host import HostPolicy, HostStandardizer
from baccurate.standardization.isolation_source import (
    IsolationSourceDiagnostic,
    IsolationSourceEvidenceLevel,
    IsolationSourceOutcome,
    IsolationSourcePromptPolicy,
    IsolationSourceStandardizer,
)
from baccurate.standardization.supporting_attribute_value_pair import SupportingAttributeValuePair
from baccurate.standardization_target.specifications import StandardizationTarget

FECES = "host-associated:animal host:feces"
SOIL = "environmental:natural environment:terrestrial:soil"


class _ScriptedCompletions:
    def __init__(self, parent: ScriptedClient) -> None:
        self._parent = parent

    def create(self, **kwargs):
        self._parent.calls.append(kwargs)
        if not self._parent.responses:
            raise AssertionError("LLM create() was called unexpectedly")
        response_index = min(len(self._parent.calls) - 1, len(self._parent.responses) - 1)
        return self._parent.responses[response_index]


class ScriptedClient:
    """Return deterministic typed responses while retaining rendered model requests."""

    def __init__(self) -> None:
        self.calls: list[dict] = []
        self.responses: list[object] = []
        self.chat = SimpleNamespace(completions=_ScriptedCompletions(self))

    def respond_with(
        self,
        terms: list[str],
        *,
        evidence_level: str,
        reasoning: str = "fixture classification",
    ) -> None:
        self.responses.append(
            SimpleNamespace(
                terms=terms,
                reasoning=reasoning,
                evidence_level=evidence_level,
            )
        )

    def close(self) -> None:
        pass


def _build_isolation_source_run(
    tmp_path: Path,
    *,
    rows: list[dict[str, str]],
    bioproject_context_entries: list[dict[str, object]],
    client: ScriptedClient,
    extracted_metadata_bundle_factory,
    fixture_dataset_builder_factory,
    fixture_host_policy: HostPolicy,
    fixture_isolation_source_prompt_policy: IsolationSourcePromptPolicy,
    fixture_pathogen_registry,
    standardization_fixture_resources,
    targets: tuple[StandardizationTarget, ...] = (StandardizationTarget.ISOLATION_SOURCE,),
):
    bundle = extracted_metadata_bundle_factory(
        "bioproject-evidence",
        extracted_rows=rows,
        bioproject_context_entries=bioproject_context_entries,
    )

    def isolation_source_factory(policy, extracted_metadata, result_logger):
        standardizer = IsolationSourceStandardizer(
            policy,
            extracted_metadata,
            result_logger=result_logger,
            client=None,
            llm_settings=LLMSettings(None, None, "fixture-model"),
        )
        standardizer.pipeline.client = client
        return standardizer

    dataset = tmp_path / "standardized.tsv"
    reasoning = tmp_path / "isolation_source_reasoning.jsonl"
    statistics = fixture_dataset_builder_factory(
        isolation_source_standardizer_factory=isolation_source_factory
    ).build(
        DatasetBuildRequest(
            extracted_metadata=bundle.extracted_metadata,
            biosample_snapshot_manifest=bundle.biosample_snapshot_manifest,
            bioproject_snapshot_manifest=bundle.bioproject_snapshot_manifest,
            requested_pathogens=("ecoli",),
            requested_targets=targets,
            final_destination=dataset,
            isolation_source_reasoning_destination=reasoning,
            atb_index=standardization_fixture_resources.atb_index,
            pathogen_registry=fixture_pathogen_registry,
            host_policy=fixture_host_policy,
            isolation_source_prompt_policy=fixture_isolation_source_prompt_policy,
            disable_progress=True,
        )
    )
    return SimpleNamespace(dataset=dataset, reasoning=reasoning, statistics=statistics)


def _dataset_rows(path: Path) -> dict[str, dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return {row["accession"]: row for row in csv.DictReader(stream, delimiter="\t")}


def _reasoning_rows(path: Path) -> dict[str, dict[str, object]]:
    return {
        record["accession"]: record
        for line in path.read_text(encoding="utf-8").splitlines()
        if (record := json.loads(line))
    }


def test_dataset_build_preserves_evidence_levels_artifacts_and_statistics(
    tmp_path: Path,
    extracted_metadata_bundle_factory,
    fixture_dataset_builder_factory,
    fixture_host_policy,
    fixture_isolation_source_prompt_policy,
    fixture_pathogen_registry,
    standardization_fixture_resources,
) -> None:
    client = ScriptedClient()
    client.respond_with(["soil"], evidence_level="project")
    client.respond_with(["host-associated"], evidence_level="sample")
    client.respond_with(["soil"], evidence_level="project")
    client.respond_with(["unspecified"], evidence_level="project")
    run = _build_isolation_source_run(
        tmp_path,
        rows=[
            {
                "accession": "SAMPLE_ONLY",
                "pathogen": "ecoli",
                "bioproject_accession": "PRJNA1",
                "iso_attr_orig": "isolation_source",
                "iso_val_orig": "stool",
            },
            {
                "accession": "PROJECT_ONLY",
                "pathogen": "ecoli",
                "bioproject_accession": "PRJNA2",
            },
            {
                "accession": "HOST_ONLY",
                "pathogen": "ecoli",
                "host_attr_orig": "host",
                "host_val_orig": "Homo sapiens",
            },
            {
                "accession": "SAMPLE_AND_PROJECT",
                "pathogen": "ecoli",
                "bioproject_accession": "PRJNA3",
                "iso_attr_orig": "isolation_source||environment",
                "iso_val_orig": "stool||farm material 777",
            },
            {
                "accession": "UNINFORMATIVE_PROJECT",
                "pathogen": "ecoli",
                "bioproject_accession": "PRJNA4",
            },
            {
                "accession": "UNRESOLVED_PROJECT",
                "pathogen": "ecoli",
                "bioproject_id": "999",
            },
        ],
        bioproject_context_entries=[
            {
                "id": "1",
                "accession": "PRJNA1",
                "title": "Environmental surveillance",
                "description": "A broad One Health study.",
                "relevance": ["Environmental"],
            },
            {
                "id": "2",
                "accession": "PRJNA2",
                "title": "Soil isolate survey",
                "description": "Bacteria isolated from agricultural soil.",
                "relevance": ["Agricultural", "Environmental"],
            },
            {
                "id": "3",
                "accession": "PRJNA3",
                "title": "Farm soil survey",
                "description": "Soil and animal sampling.",
                "relevance": ["Agricultural", "Environmental"],
            },
            {
                "id": "4",
                "accession": "PRJNA4",
                "title": "Genome sequencing",
                "description": "No sample origin is reported.",
                "relevance": [],
            },
        ],
        client=client,
        extracted_metadata_bundle_factory=extracted_metadata_bundle_factory,
        fixture_dataset_builder_factory=fixture_dataset_builder_factory,
        fixture_host_policy=fixture_host_policy,
        fixture_isolation_source_prompt_policy=fixture_isolation_source_prompt_policy,
        fixture_pathogen_registry=fixture_pathogen_registry,
        standardization_fixture_resources=standardization_fixture_resources,
    )

    rows = _dataset_rows(run.dataset)
    reasoning = _reasoning_rows(run.reasoning)
    assert len(client.calls) == 4
    assert rows["SAMPLE_ONLY"]["iso_term_paths"] == FECES
    assert rows["PROJECT_ONLY"]["iso_term_paths"] == SOIL
    assert rows["HOST_ONLY"]["iso_term_paths"] == "host-associated"
    assert set(rows["SAMPLE_AND_PROJECT"]["iso_term_paths"].split("||")) == {FECES, SOIL}
    assert rows["UNINFORMATIVE_PROJECT"]["iso_display_terms"] == "unspecified"
    assert "UNRESOLVED_PROJECT" not in rows
    assert {accession: record["evidence_level"] for accession, record in reasoning.items()} == {
        "SAMPLE_ONLY": "sample",
        "PROJECT_ONLY": "project",
        "HOST_ONLY": "sample",
        "SAMPLE_AND_PROJECT": "sample_and_project",
        "UNINFORMATIVE_PROJECT": "none",
    }
    assert run.statistics.isolation_source.aggregate.evidence_levels == {
        IsolationSourceEvidenceLevel.SAMPLE: 2,
        IsolationSourceEvidenceLevel.PROJECT: 1,
        IsolationSourceEvidenceLevel.SAMPLE_AND_PROJECT: 1,
        IsolationSourceEvidenceLevel.NONE: 1,
    }
    assert run.statistics.isolation_source.aggregate.rejected == 1
    assert (
        run.statistics.isolation_source.aggregate.diagnostics[
            IsolationSourceDiagnostic.UNRESOLVED_BIOPROJECT_LINK
        ]
        == 1
    )
    assert (
        run.statistics.isolation_source.aggregate.diagnostics[
            IsolationSourceDiagnostic.NO_CLASSIFICATION_INPUT
        ]
        == 1
    )


def test_dataset_build_orders_resolved_bioproject_context_and_includes_it_in_request_identity(
    tmp_path: Path,
    extracted_metadata_bundle_factory,
    fixture_dataset_builder_factory,
    fixture_host_policy,
    fixture_isolation_source_prompt_policy,
    fixture_pathogen_registry,
    standardization_fixture_resources,
) -> None:
    client = ScriptedClient()
    client.respond_with(["wound"], evidence_level="sample_and_project")
    client.respond_with(["wound"], evidence_level="sample_and_project")
    _build_isolation_source_run(
        tmp_path,
        rows=[
            {
                "accession": "ORDERED_CONTEXT",
                "pathogen": "ecoli",
                "bioproject_id": "3||1||999",
                "bioproject_accession": "PRJNA300||PRJNA100",
                "iso_attr_orig": "isolation_source",
                "iso_val_orig": "wound patient 161803",
            },
            {
                "accession": "DIFFERENT_CONTEXT",
                "pathogen": "ecoli",
                "bioproject_id": "2",
                "bioproject_accession": "PRJNA200",
                "iso_attr_orig": "isolation_source",
                "iso_val_orig": "wound patient 161803",
            },
        ],
        bioproject_context_entries=[
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
            {
                "id": "2",
                "accession": "PRJNA200",
                "title": "Veterinary study",
                "description": "Animal sampling.",
                "relevance": ["Veterinary"],
            },
        ],
        client=client,
        extracted_metadata_bundle_factory=extracted_metadata_bundle_factory,
        fixture_dataset_builder_factory=fixture_dataset_builder_factory,
        fixture_host_policy=fixture_host_policy,
        fixture_isolation_source_prompt_policy=fixture_isolation_source_prompt_policy,
        fixture_pathogen_registry=fixture_pathogen_registry,
        standardization_fixture_resources=standardization_fixture_resources,
    )

    assert len(client.calls) == 2
    first_context = json.loads(
        client.calls[0]["messages"][1]["content"].split("BioProject context:\n", 1)[1]
    )
    assert [project["accession"] for project in first_context] == ["PRJNA100", "PRJNA300"]


@pytest.mark.parametrize("invalid_evidence_level", ["sample", "sample_and_project", "none"])
def test_dataset_build_rejects_invalid_project_only_evidence_claim(
    tmp_path: Path,
    invalid_evidence_level: str,
    extracted_metadata_bundle_factory,
    fixture_dataset_builder_factory,
    fixture_host_policy,
    fixture_isolation_source_prompt_policy,
    fixture_pathogen_registry,
    standardization_fixture_resources,
) -> None:
    client = ScriptedClient()
    client.respond_with(["soil"], evidence_level=invalid_evidence_level)
    with pytest.raises(
        RuntimeError, match="Isolation-source LLM failed for accession PROJECT_ONLY"
    ):
        _build_isolation_source_run(
            tmp_path,
            rows=[
                {
                    "accession": "PROJECT_ONLY",
                    "pathogen": "ecoli",
                    "bioproject_accession": "PRJNA1",
                }
            ],
            bioproject_context_entries=[
                {
                    "id": "1",
                    "accession": "PRJNA1",
                    "title": "Soil survey",
                    "description": "Soil sampling.",
                    "relevance": ["Environmental"],
                }
            ],
            client=client,
            extracted_metadata_bundle_factory=extracted_metadata_bundle_factory,
            fixture_dataset_builder_factory=fixture_dataset_builder_factory,
            fixture_host_policy=fixture_host_policy,
            fixture_isolation_source_prompt_policy=fixture_isolation_source_prompt_policy,
            fixture_pathogen_registry=fixture_pathogen_registry,
            standardization_fixture_resources=standardization_fixture_resources,
        )


@pytest.mark.parametrize(
    ("bioproject_context_entries", "match"),
    [
        (
            [{"id": "1", "accession": "PRJNA1"}],
            "expected fields",
        ),
        (
            [
                {
                    "id": "1",
                    "accession": "PRJNA1",
                    "title": "Medical study",
                    "description": "Clinical sampling.",
                    "relevance": ["Medical"],
                }
            ],
            "relevance must contain distinct Agricultural, Environmental, or Veterinary",
        ),
        (
            [
                {
                    "id": "1",
                    "accession": "PRJNA1",
                    "title": "Soil study",
                    "description": "Soil sampling.",
                    "relevance": ["Environmental"],
                },
                {
                    "id": "1",
                    "accession": "PRJNA2",
                    "title": "Farm study",
                    "description": "Farm sampling.",
                    "relevance": ["Agricultural"],
                },
            ],
            "duplicate project ID",
        ),
        (
            [
                {
                    "id": "1",
                    "accession": "PRJNA1",
                    "title": "Soil study",
                    "description": "Soil sampling.",
                    "relevance": ["Environmental"],
                },
                {
                    "id": "2",
                    "accession": "PRJNA1",
                    "title": "Farm study",
                    "description": "Farm sampling.",
                    "relevance": ["Agricultural"],
                },
            ],
            "duplicate project accession",
        ),
    ],
    ids=["missing-fields", "invalid-relevance", "duplicate-id", "duplicate-accession"],
)
def test_malformed_bioproject_context_is_rejected_before_classification(
    extracted_metadata_bundle_factory,
    bioproject_context_entries: list[dict[str, object]],
    match: str,
) -> None:
    with pytest.raises(SourceSnapshotError, match=match):
        extracted_metadata_bundle_factory(
            "invalid-project-context",
            bioproject_context_entries=bioproject_context_entries,
        )


def test_host_recovery_uses_record_pairs_and_not_project_only_evidence(
    tmp_path: Path,
    extracted_metadata_bundle_factory,
    fixture_dataset_builder_factory,
    fixture_host_policy,
    fixture_isolation_source_prompt_policy,
    fixture_pathogen_registry,
    standardization_fixture_resources,
) -> None:
    bundle = extracted_metadata_bundle_factory(
        "host-recovery",
        extracted_rows=[
            {
                "accession": "RECORD_EVIDENCE",
                "pathogen": "ecoli",
                "bioproject_accession": "PRJNA1",
                "iso_attr_orig": "food_source",
                "iso_val_orig": "chicken meat",
            },
            {
                "accession": "PROJECT_ONLY",
                "pathogen": "ecoli",
                "bioproject_accession": "PRJNA2",
            },
        ],
        bioproject_context_entries=[
            {
                "id": "1",
                "accession": "PRJNA1",
                "title": "Human microbiome project",
                "description": "Human clinical samples.",
                "relevance": ["Veterinary"],
            },
            {
                "id": "2",
                "accession": "PRJNA2",
                "title": "Chicken survey",
                "description": "Poultry meat surveillance.",
                "relevance": ["Agricultural"],
            },
        ],
    )

    class ProjectSupportedIsolationSourceStandardizer:
        def standardize(self, record, *, host_context, overflow=None):
            supporting_pairs = (
                (SupportingAttributeValuePair("food_source", "chicken meat"),)
                if record["accession"] == "RECORD_EVIDENCE"
                else ()
            )
            return IsolationSourceOutcome(
                term_path_roots="environmental",
                display_terms="meat product",
                external_ontology_identifiers="FOODON:00001006",
                term_paths=(
                    "environmental:anthropogenic environment:food:animal product:meat product"
                ),
                evidence_level=IsolationSourceEvidenceLevel.PROJECT,
                host_context=host_context,
                supporting_pairs=supporting_pairs,
                host_recovery_pairs=supporting_pairs,
                reasoning=(),
                diagnostics=(IsolationSourceDiagnostic.LLM_CALL,),
                exact_matches=0,
                cache_hits=0,
                llm_calls=1,
            )

        def close(self) -> None:
            pass

    class RecordingHostStandardizer(HostStandardizer):
        def __init__(self, policy: HostPolicy, result_logger) -> None:
            super().__init__(
                policy,
                standardization_fixture_resources.ncbi_taxonomy_reference_table,
                result_logger=result_logger,
            )
            self.recovery_pass_calls: list[tuple[str, str, str]] = []

        def recovery_pass(self, accession: str, attributes: str, values: str):
            self.recovery_pass_calls.append((accession, attributes, values))
            return super().recovery_pass(accession, attributes, values)

    host_standardizer = None

    def host_factory(policy, result_logger):
        nonlocal host_standardizer
        host_standardizer = RecordingHostStandardizer(policy, result_logger)
        return host_standardizer

    dataset = tmp_path / "host-recovery.tsv"
    statistics = fixture_dataset_builder_factory(
        host_standardizer_factory=host_factory,
        isolation_source_standardizer_factory=lambda *_args: (
            ProjectSupportedIsolationSourceStandardizer()
        ),
    ).build(
        DatasetBuildRequest(
            extracted_metadata=bundle.extracted_metadata,
            biosample_snapshot_manifest=bundle.biosample_snapshot_manifest,
            bioproject_snapshot_manifest=bundle.bioproject_snapshot_manifest,
            requested_pathogens=("ecoli",),
            requested_targets=(StandardizationTarget.HOST, StandardizationTarget.ISOLATION_SOURCE),
            final_destination=dataset,
            atb_index=standardization_fixture_resources.atb_index,
            pathogen_registry=fixture_pathogen_registry,
            host_policy=fixture_host_policy,
            isolation_source_prompt_policy=fixture_isolation_source_prompt_policy,
            disable_progress=True,
        )
    )

    assert host_standardizer is not None
    assert host_standardizer.recovery_pass_calls == [
        ("RECORD_EVIDENCE", "food_source", "chicken meat")
    ]
    rows = _dataset_rows(dataset)
    assert rows["RECORD_EVIDENCE"]["host_sci_name"] == "Gallus gallus"
    assert rows["PROJECT_ONLY"]["host_sci_name"] == ""
    assert statistics.host.aggregate.host_recovery_passes == 1
    assert statistics.isolation_source.aggregate.host_recovery_passes == 1
