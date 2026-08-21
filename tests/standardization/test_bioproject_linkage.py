"""Protect BioSample evidence and BioProject-link dataset behavior."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from types import SimpleNamespace

from baccurate.adapters.llm.client import LLMSettings
from baccurate.run.dataset_builder import DatasetBuildRequest
from baccurate.standardization.host import HostPolicy, HostStandardizer
from baccurate.standardization.isolation_source import (
    IsolationSourceDiagnostic,
    IsolationSourceEvidenceLevel,
    IsolationSourceOutcome,
    IsolationSourcePromptPolicy,
    IsolationSourceRejection,
    IsolationSourceStandardizer,
    SelectedTerm,
)
from baccurate.standardization.supporting_attribute_value_pair import SupportingAttributeValuePair
from baccurate.standardization_target.specifications import StandardizationTarget

FECES = "BACC:0000025"
SOIL = "BACC:0000064"

FACET_BY_LABEL = {
    "host-associated": "source_type",
    "soil": "environmental_material",
    "wound": "lesion",
}


class _ScriptedCompletions:
    def __init__(self, parent: ScriptedClient) -> None:
        self._parent = parent

    def create(self, **kwargs):
        self._parent.calls.append(kwargs)
        if not self._parent.responses:
            raise AssertionError("LLM create() was called unexpectedly")
        response_index = min(len(self._parent.calls) - 1, len(self._parent.responses) - 1)
        response = self._parent.responses[response_index]
        if isinstance(response, Exception):
            raise response
        return kwargs["response_model"].model_validate(response)


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
        answer: dict[str, object] = {
            "reasoning": reasoning,
            "evidence_level": evidence_level,
            "source_type": [],
            "body_product": [],
            "body_site": [],
            "lesion": [],
            "environmental_material": [],
            "facility": [],
            "sampled_object": [],
            "food_type": [],
        }
        for term in terms:
            facet = FACET_BY_LABEL[term]
            if facet == "source_type":
                answer[facet].append(term)
            else:
                answer[facet].append(term)
        self.responses.append(answer)

    def close(self) -> None:
        pass


def _build_isolation_source_run(
    tmp_path: Path,
    *,
    rows: list[dict[str, str]],
    client: ScriptedClient,
    extracted_metadata_bundle_factory,
    fixture_dataset_builder_factory,
    fixture_host_policy: HostPolicy,
    fixture_isolation_source_prompt_policy: IsolationSourcePromptPolicy,
    fixture_taxon_registry,
    standardization_fixture_resources,
    targets: tuple[StandardizationTarget, ...] = (StandardizationTarget.ISOLATION_SOURCE,),
):
    bundle = extracted_metadata_bundle_factory(
        "bioproject-linkage",
        extracted_rows=rows,
    )

    def isolation_source_factory(policy, result_logger):
        standardizer = IsolationSourceStandardizer(
            policy,
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
            requested_taxa=("ecoli",),
            requested_targets=targets,
            final_destination=dataset,
            isolation_source_reasoning_destination=reasoning,
            taxon_registry=fixture_taxon_registry,
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


def test_dataset_build_limits_evidence_to_biosample_pairs(
    tmp_path: Path,
    extracted_metadata_bundle_factory,
    fixture_dataset_builder_factory,
    fixture_host_policy,
    fixture_isolation_source_prompt_policy,
    fixture_taxon_registry,
    standardization_fixture_resources,
) -> None:
    client = ScriptedClient()
    client.respond_with(["soil"], evidence_level="sample")
    run = _build_isolation_source_run(
        tmp_path,
        rows=[
            {
                "accession": "SAMPLE_ONLY",
                "taxon_key": "ecoli",
                "bioproject_accession": "PRJNA1",
                "iso_attr_orig": "isolation_source",
                "iso_val_orig": "stool",
            },
            {
                "accession": "PROJECT_ONLY",
                "taxon_key": "ecoli",
                "bioproject_accession": "PRJNA2",
            },
            {
                "accession": "HOST_ONLY",
                "taxon_key": "ecoli",
                "host_attr_orig": "host",
                "host_val_orig": "Homo sapiens",
            },
            {
                "accession": "SAMPLE_AND_PROJECT",
                "taxon_key": "ecoli",
                "bioproject_accession": "PRJNA3",
                "iso_attr_orig": "isolation_source||environment",
                "iso_val_orig": "stool||farm material 777",
            },
            {
                "accession": "UNINFORMATIVE_PROJECT",
                "taxon_key": "ecoli",
                "bioproject_accession": "PRJNA4",
            },
            {
                "accession": "UNRESOLVED_PROJECT",
                "taxon_key": "ecoli",
                "bioproject_accession": "",
            },
        ],
        client=client,
        extracted_metadata_bundle_factory=extracted_metadata_bundle_factory,
        fixture_dataset_builder_factory=fixture_dataset_builder_factory,
        fixture_host_policy=fixture_host_policy,
        fixture_isolation_source_prompt_policy=fixture_isolation_source_prompt_policy,
        fixture_taxon_registry=fixture_taxon_registry,
        standardization_fixture_resources=standardization_fixture_resources,
    )

    rows = _dataset_rows(run.dataset)
    reasoning = _reasoning_rows(run.reasoning)
    assert len(client.calls) == 1
    assert rows["SAMPLE_ONLY"]["iso_body_product"] == "feces"
    assert rows["SAMPLE_ONLY"]["iso_term_ids"] == (f"BACC:0000001||BACC:0000002||{FECES}")
    assert rows["HOST_ONLY"]["iso_source_type"] == "host-associated||animal host"
    assert rows["SAMPLE_AND_PROJECT"]["iso_term_ids"] == (
        f"BACC:0000001||BACC:0000002||BACC:0000004||{FECES}||{SOIL}"
    )
    assert "PROJECT_ONLY" not in rows
    assert "UNINFORMATIVE_PROJECT" not in rows
    assert "UNRESOLVED_PROJECT" not in rows
    assert {accession: record["evidence_level"] for accession, record in reasoning.items()} == {
        "SAMPLE_ONLY": "sample",
        "HOST_ONLY": "sample",
        "SAMPLE_AND_PROJECT": "sample",
    }
    assert run.statistics.isolation_source.aggregate.evidence_levels == {
        IsolationSourceEvidenceLevel.SAMPLE: 3,
    }
    assert run.statistics.isolation_source.aggregate.rejected == 3
    assert (
        run.statistics.isolation_source.aggregate.diagnostics[
            IsolationSourceDiagnostic.NO_CLASSIFICATION_INPUT
        ]
        == 3
    )


def test_dataset_build_request_identity_uses_only_biosample_pairs(
    tmp_path: Path,
    extracted_metadata_bundle_factory,
    fixture_dataset_builder_factory,
    fixture_host_policy,
    fixture_isolation_source_prompt_policy,
    fixture_taxon_registry,
    standardization_fixture_resources,
) -> None:
    client = ScriptedClient()
    client.respond_with(["wound"], evidence_level="sample")
    run = _build_isolation_source_run(
        tmp_path,
        rows=[
            {
                "accession": "FIRST_CONTEXT",
                "taxon_key": "ecoli",
                "bioproject_accession": "PRJNA300||PRJNA100",
                "iso_attr_orig": "isolation_source",
                "iso_val_orig": "wound patient 161803",
            },
            {
                "accession": "SECOND_CONTEXT",
                "taxon_key": "ecoli",
                "bioproject_accession": "PRJNA200",
                "iso_attr_orig": "isolation_source",
                "iso_val_orig": "wound patient 161803",
            },
        ],
        client=client,
        extracted_metadata_bundle_factory=extracted_metadata_bundle_factory,
        fixture_dataset_builder_factory=fixture_dataset_builder_factory,
        fixture_host_policy=fixture_host_policy,
        fixture_isolation_source_prompt_policy=fixture_isolation_source_prompt_policy,
        fixture_taxon_registry=fixture_taxon_registry,
        standardization_fixture_resources=standardization_fixture_resources,
    )

    assert len(client.calls) == 1
    messages = json.dumps(client.calls[0]["messages"])
    assert "BioProject" not in messages
    assert "PRJNA" not in messages
    assert "Veterinary" not in messages
    assert run.statistics.isolation_source.aggregate.cache_hits == 1


def test_dataset_build_rejects_bioproject_only_record_without_model_call(
    tmp_path: Path,
    extracted_metadata_bundle_factory,
    fixture_dataset_builder_factory,
    fixture_host_policy,
    fixture_isolation_source_prompt_policy,
    fixture_taxon_registry,
    standardization_fixture_resources,
) -> None:
    client = ScriptedClient()
    run = _build_isolation_source_run(
        tmp_path,
        rows=[
            {
                "accession": "PROJECT_ONLY",
                "taxon_key": "ecoli",
                "bioproject_accession": "PRJNA1",
            }
        ],
        client=client,
        extracted_metadata_bundle_factory=extracted_metadata_bundle_factory,
        fixture_dataset_builder_factory=fixture_dataset_builder_factory,
        fixture_host_policy=fixture_host_policy,
        fixture_isolation_source_prompt_policy=fixture_isolation_source_prompt_policy,
        fixture_taxon_registry=fixture_taxon_registry,
        standardization_fixture_resources=standardization_fixture_resources,
    )

    assert _dataset_rows(run.dataset) == {}
    assert client.calls == []
    assert run.statistics.isolation_source.aggregate.rejected == 1
    assert run.statistics.isolation_source.aggregate.diagnostics == {
        IsolationSourceDiagnostic.NO_CLASSIFICATION_INPUT: 1
    }


def test_host_recovery_uses_biosample_pairs_not_bioproject_linkage(
    tmp_path: Path,
    extracted_metadata_bundle_factory,
    fixture_dataset_builder_factory,
    fixture_host_policy,
    fixture_isolation_source_prompt_policy,
    fixture_taxon_registry,
    standardization_fixture_resources,
) -> None:
    bundle = extracted_metadata_bundle_factory(
        "host-recovery",
        extracted_rows=[
            {
                "accession": "RECORD_EVIDENCE",
                "taxon_key": "ecoli",
                "bioproject_accession": "PRJNA1",
                "iso_attr_orig": "food_source",
                "iso_val_orig": "chicken meat",
            },
            {
                "accession": "PROJECT_ONLY",
                "taxon_key": "ecoli",
                "bioproject_accession": "PRJNA2",
            },
        ],
    )

    class BioSampleIsolationSourceStandardizer:
        def standardize(self, record, *, overflow=None):
            supporting_pairs = (
                (SupportingAttributeValuePair("food_source", "chicken meat"),)
                if record["accession"] == "RECORD_EVIDENCE"
                else ()
            )
            if not supporting_pairs:
                return IsolationSourceRejection(
                    (IsolationSourceDiagnostic.NO_CLASSIFICATION_INPUT,)
                )
            return IsolationSourceOutcome(
                selected_terms=(SelectedTerm("BACC:0000097", "food_type", "meat product"),),
                evidence_level=IsolationSourceEvidenceLevel.SAMPLE,
                supporting_pairs=supporting_pairs,
                host_recovery_pairs=supporting_pairs,
                reasoning=(),
                diagnostics=(IsolationSourceDiagnostic.LLM_CALL,),
                exact_matches=0,
                cache_hits=0,
                llm_calls=1,
                host_recovery_eligible=True,
            )

        @staticmethod
        def refine_source_type_from_host_lineage(outcome, **_lineage):
            return outcome

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
        isolation_source_standardizer_factory=lambda *_args: BioSampleIsolationSourceStandardizer(),
    ).build(
        DatasetBuildRequest(
            extracted_metadata=bundle.extracted_metadata,
            biosample_snapshot_manifest=bundle.biosample_snapshot_manifest,
            bioproject_snapshot_manifest=bundle.bioproject_snapshot_manifest,
            requested_taxa=("ecoli",),
            requested_targets=(StandardizationTarget.HOST, StandardizationTarget.ISOLATION_SOURCE),
            final_destination=dataset,
            taxon_registry=fixture_taxon_registry,
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
    assert "PROJECT_ONLY" not in rows
    assert statistics.host.aggregate.host_recovery_passes == 1
    assert statistics.isolation_source.aggregate.host_recovery_passes == 1
