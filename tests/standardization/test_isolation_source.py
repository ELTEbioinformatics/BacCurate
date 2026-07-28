"""Pin the isolation-source standardization contract."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

import baccurate.standardization.isolation_source as isolation_source_module
from baccurate.adapters.llm.client import LLMSettings
from baccurate.adapters.llm.diagnostics import LLMObservability
from baccurate.provenance.source_snapshot import bioproject_catalog_path_for
from baccurate.standardization.host import HostOverflowContext
from baccurate.standardization.isolation_source import (
    IsolationSourceDiagnostic,
    IsolationSourceEvidenceLevel,
    IsolationSourceOutcome,
    IsolationSourcePromptPolicy,
    IsolationSourceReasoningStep,
    IsolationSourceStandardizer,
    SQLiteCache,
    StandardizedIsolationSource,
)

ROOT = Path(__file__).parents[2]
CONFIG_PATH = ROOT / "config" / "isolation_source.yaml"
FIXTURE_ROOT = ROOT / "tests" / "fixtures" / "standardization"


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

    def close(self) -> None:
        pass


def _empty_extracted_bundle(tmp_path: Path) -> Path:
    extracted = tmp_path / "unit-extracted.tsv"
    bioproject_catalog_path_for(extracted).write_text("", encoding="utf-8")
    return extracted


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def cache(tmp_path) -> SQLiteCache:
    c = SQLiteCache(tmp_path / "iso_cache.db")
    yield c
    c.close()


# =============================================================================
# Record-level outcomes
# =============================================================================


def test_typed_record_outcome_preserves_context_supporting_pairs_and_diagnostics(
    tmp_path,
    monkeypatch,
):
    config_path = _isolation_source_config(tmp_path)
    monkeypatch.setattr(
        "baccurate.standardization.isolation_source.load_llm_client",
        lambda *_args: (None, None),
    )
    standardizer = IsolationSourceStandardizer(
        IsolationSourcePromptPolicy.load(config_path),
        _empty_extracted_bundle(tmp_path),
    )

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
        with pytest.raises(
            ValueError,
            match=(
                "Malformed isolation-source selected attribute-value pairs for accession "
                "MALFORMED: 2 attributes for 1 values"
            ),
        ):
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

    assert isinstance(result, IsolationSourceOutcome)
    assert result.host_context == "Homo sapiens"
    assert [(pair.attribute, pair.value) for pair in result.supporting_pairs] == [
        ("isolation_source", "stool"),
        ("host sample", "blood"),
    ]
    assert [(pair.attribute, pair.value) for pair in result.host_recovery_pairs] == [
        ("isolation_source", "stool")
    ]
    assert FECES in result.term_paths.split("||")
    assert BLOOD in result.term_paths.split("||")
    assert result.term_path_roots == "host-associated"
    assert result.display_terms == "blood||feces"
    assert result.external_ontology_identifiers == "ENVO:02000020||ENVO:00002003"
    assert result.reasoning == (
        IsolationSourceReasoningStep(
            node="direct_match",
            reasoning="All values resolved manually.",
            selected_term_paths=(BLOOD, FECES),
        ),
    )
    assert result.exact_matches == 2
    assert result.cache_hits == 0
    assert result.llm_calls == 0
    assert result.evidence_level is IsolationSourceEvidenceLevel.SAMPLE
    assert result.diagnostics == (IsolationSourceDiagnostic.EXACT_MATCH,)


def _classify_fixture_record(
    policy: IsolationSourcePromptPolicy,
    tmp_path: Path,
    *,
    values: str,
    attributes: str = "isolation_source",
    fake: FakeClient | None = None,
    monkeypatch: pytest.MonkeyPatch | None = None,
    host_context: str = "",
) -> IsolationSourceOutcome:
    if fake is not None:
        assert monkeypatch is not None
        monkeypatch.setattr(
            isolation_source_module.instructor, "from_openai", lambda client: client
        )
    standardizer = IsolationSourceStandardizer(
        policy,
        _empty_extracted_bundle(tmp_path),
        client=fake,
        llm_settings=LLMSettings(None, None, "test-model"),
    )
    try:
        result = standardizer.standardize(
            {
                "accession": "PUBLIC_CLASSIFICATION",
                "iso_attr_orig": attributes,
                "iso_val_orig": values,
            },
            host_context=host_context,
        )
    finally:
        standardizer.close()
    assert isinstance(result, IsolationSourceOutcome)
    return result


@pytest.mark.parametrize(
    ("submitted_value", "term_paths", "display_terms", "external_identifiers"),
    [
        ("feces", FECES, "feces", "ENVO:00002003"),
        ("rectal swab", RECTUM, "rectum", "UBERON:0001052"),
        ("sample drawn, envo:02000020 present", BLOOD, "blood", "ENVO:02000020"),
        (
            "rhizosphere",
            f"{SOIL}||{RHIZOSPHERE}",
            "soil||rhizosphere",
            "ENVO:00001998||ENVO:00005801",
        ),
    ],
)
def test_public_classification_resolves_ontology_terms_and_crosslinks(
    fixture_isolation_source_prompt_policy: IsolationSourcePromptPolicy,
    tmp_path: Path,
    submitted_value: str,
    term_paths: str,
    display_terms: str,
    external_identifiers: str,
) -> None:
    result = _classify_fixture_record(
        fixture_isolation_source_prompt_policy,
        tmp_path,
        values=submitted_value,
    )

    assert result.term_paths == term_paths
    assert result.display_terms == display_terms
    assert result.external_ontology_identifiers == external_identifiers
    assert result.evidence_level is IsolationSourceEvidenceLevel.SAMPLE
    assert result.exact_matches == 1
    assert result.llm_calls == 0
    assert result.diagnostics == (IsolationSourceDiagnostic.EXACT_MATCH,)


def test_public_classification_unions_direct_and_model_terms(
    fixture_isolation_source_prompt_policy: IsolationSourcePromptPolicy,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = FakeClient()
    fake.respond_with(["blood"], reasoning="The unresolved value describes blood.")

    result = _classify_fixture_record(
        fixture_isolation_source_prompt_policy,
        tmp_path,
        attributes="isolation_source||tissue",
        values="stool||venous draw",
        fake=fake,
        monkeypatch=monkeypatch,
    )

    assert result.term_paths == f"{BLOOD}||{FECES}"
    assert result.display_terms == "blood||feces"
    assert result.exact_matches == 1
    assert result.llm_calls == 1
    assert result.evidence_level is IsolationSourceEvidenceLevel.SAMPLE
    assert result.reasoning == (
        IsolationSourceReasoningStep(
            node="classifier",
            reasoning="The unresolved value describes blood.",
            selected_term_paths=(BLOOD,),
            selected_display_terms=("blood",),
        ),
    )


def test_model_selected_feces_does_not_infer_anatomical_origin(
    fixture_isolation_source_prompt_policy: IsolationSourcePromptPolicy,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = FakeClient()
    fake.respond_with(["feces"], reasoning="The evidence describes fecal material.")

    result = _classify_fixture_record(
        fixture_isolation_source_prompt_policy,
        tmp_path,
        values="gut contents",
        fake=fake,
        monkeypatch=monkeypatch,
    )

    assert result.term_paths == FECES
    assert RECTUM not in result.standardized_term_paths
    assert result.reasoning == (
        IsolationSourceReasoningStep(
            node="classifier",
            reasoning="The evidence describes fecal material.",
            selected_term_paths=(FECES,),
            selected_display_terms=("feces",),
        ),
    )
    assert result.diagnostics == (IsolationSourceDiagnostic.LLM_CALL,)


def test_explicit_unspecified_model_result_remains_a_typed_outcome(
    fixture_isolation_source_prompt_policy: IsolationSourcePromptPolicy,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = FakeClient()
    fake.respond_with(["unspecified"], reasoning="No specific source is supported.")

    result = _classify_fixture_record(
        fixture_isolation_source_prompt_policy,
        tmp_path,
        values="GENOMIC",
        fake=fake,
        monkeypatch=monkeypatch,
    )

    assert result.term_path_roots == "unspecified"
    assert result.term_paths == "unspecified"
    assert result.display_terms == "unspecified"
    assert result.external_ontology_identifiers == "NA"
    assert result.evidence_level is IsolationSourceEvidenceLevel.NONE
    assert result.diagnostics == (
        IsolationSourceDiagnostic.LLM_CALL,
        IsolationSourceDiagnostic.UNSPECIFIED,
    )


def test_partial_deterministic_result_uses_sample_evidence_when_llm_is_disabled(
    tmp_path: Path,
) -> None:
    config_path = _isolation_source_config(tmp_path)
    standardizer = IsolationSourceStandardizer(
        IsolationSourcePromptPolicy.load(config_path),
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

    assert isinstance(result, IsolationSourceOutcome)
    assert result.term_paths == FECES
    assert result.evidence_level is IsolationSourceEvidenceLevel.SAMPLE


def test_typed_record_outcome_preserves_model_reasoning_on_exact_cache_hit(tmp_path, monkeypatch):
    config_path = _isolation_source_config(tmp_path)
    monkeypatch.setenv("LLM_MODEL", "test-model")
    extracted = _empty_extracted_bundle(tmp_path)
    policy = IsolationSourcePromptPolicy.load(config_path)
    first_standardizer = IsolationSourceStandardizer(policy, extracted, client=None)
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

    cached_standardizer = IsolationSourceStandardizer(policy, extracted, client=None)
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

    assert isinstance(modelled, IsolationSourceOutcome)
    assert modelled.evidence_level is IsolationSourceEvidenceLevel.SAMPLE
    assert modelled.diagnostics == (IsolationSourceDiagnostic.LLM_CALL,)
    assert modelled.reasoning == (
        IsolationSourceReasoningStep(
            node="classifier",
            reasoning="clinical wound",
            selected_term_paths=(WOUND,),
            selected_display_terms=("wound",),
        ),
    )
    assert isinstance(cached, IsolationSourceOutcome)
    assert cached.term_paths == modelled.term_paths
    assert cached.evidence_level is IsolationSourceEvidenceLevel.SAMPLE
    assert cached.reasoning == modelled.reasoning
    assert cached.diagnostics == (IsolationSourceDiagnostic.CACHE_HIT,)
    assert len(fake_client.calls) == 1


def _isolation_source_config(
    tmp_path: Path,
    *,
    overrides: dict[str, object] | None = None,
) -> Path:
    config_path = tmp_path / f"isolation-{len(list(tmp_path.glob('isolation-*.yaml')))}.yaml"
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    config.update(overrides or {})
    if not overrides or "ontology_tsv_path" not in overrides:
        config["ontology_tsv_path"] = (FIXTURE_ROOT / "ontology.tsv").as_posix()
    config["cache_db_path"] = (tmp_path / "isolation-cache.db").as_posix()
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return config_path


def _standardize_model_isolation_source(
    config_path: Path,
    fake: FakeClient,
    *,
    model: str = "test-model",
) -> IsolationSourceOutcome:
    standardizer = IsolationSourceStandardizer(
        IsolationSourcePromptPolicy.load(config_path),
        _empty_extracted_bundle(config_path.parent),
        client=None,
        llm_settings=LLMSettings(None, None, model),
    )
    standardizer.pipeline.client = fake
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
    assert isinstance(result, IsolationSourceOutcome)
    return result


def test_isolation_source_llm_observability_uses_the_published_target_key(
    tmp_path: Path,
) -> None:
    fake = FakeClient()
    fake.respond_with(["wound"])
    observability = LLMObservability({"isolation_source": "test-model"})
    observability.start()
    try:
        _standardize_model_isolation_source(_isolation_source_config(tmp_path), fake)
        snapshot = observability.snapshot()
    finally:
        observability.close()

    assert [entry["target"] for entry in snapshot["by_target_and_model"]] == ["isolation_source"]


def test_isolation_source_standardizer_does_not_reopen_loaded_policy_source(tmp_path: Path) -> None:
    config_path = _isolation_source_config(tmp_path)
    policy = IsolationSourcePromptPolicy.load(config_path)
    config_path.write_text("broken: [\n", encoding="utf-8")

    standardizer = IsolationSourceStandardizer(
        policy,
        _empty_extracted_bundle(tmp_path),
        client=None,
    )
    try:
        outcome = standardizer.standardize(
            {
                "accession": "NO_REOPEN",
                "iso_attr_orig": "isolation_source",
                "iso_val_orig": "stool",
            },
            host_context="",
        )
    finally:
        standardizer.close()

    assert isinstance(outcome, IsolationSourceOutcome)
    assert outcome.term_paths == FECES


def test_isolation_source_cache_reuses_identical_request_when_only_prompt_metadata_changes(
    tmp_path,
):
    fake = FakeClient()
    fake.respond_with(["wound"], reasoning="clinical wound")
    first_config = _isolation_source_config(tmp_path, overrides={"prompt_version": "first"})
    second_config = _isolation_source_config(tmp_path, overrides={"prompt_version": "second"})

    first = _standardize_model_isolation_source(first_config, fake)
    second = _standardize_model_isolation_source(second_config, fake)

    assert first.diagnostics == (IsolationSourceDiagnostic.LLM_CALL,)
    assert second.diagnostics == (IsolationSourceDiagnostic.CACHE_HIT,)
    assert second.reasoning == first.reasoning
    assert len(fake.calls) == 1


def test_isolation_source_request_uses_rendered_synthetic_prompts(tmp_path: Path) -> None:
    fake = FakeClient()
    fake.respond_with(["wound"], reasoning="clinical wound")
    config_path = _isolation_source_config(
        tmp_path,
        overrides={
            "prompt_version": "synthetic-v1",
            "system_prompt": "Synthetic ontology:\n{ontology_tree}",
            "user_prompt": "Synthetic sample:\n{metadata}\n{bioproject_context}",
            "bioproject_system_prompt": "Synthetic BioProject rules.",
            "bioproject_user_prompt": "Synthetic projects:\n{bioproject_context}",
            "ontology_tsv_path": (
                ROOT / "tests" / "fixtures" / "standardization" / "ontology.tsv"
            ).as_posix(),
        },
    )
    standardizer = IsolationSourceStandardizer(
        IsolationSourcePromptPolicy.load(config_path),
        _empty_extracted_bundle(tmp_path),
        client=None,
        llm_settings=LLMSettings(None, None, "test-model"),
    )
    standardizer.pipeline.client = fake
    try:
        direct = standardizer.standardize(
            {
                "accession": "DIRECT",
                "iso_attr_orig": "isolation_source",
                "iso_val_orig": "stool",
            },
            host_context="Homo sapiens",
        )
        assert fake.calls == []
        outcome = standardizer.standardize(
            {
                "accession": "REQUEST_FINGERPRINT",
                "iso_attr_orig": "isolation_source",
                "iso_val_orig": "wound patient 739105",
            },
            host_context="Homo sapiens",
        )
    finally:
        standardizer.close()

    assert isinstance(direct, IsolationSourceOutcome)
    assert direct.diagnostics == (IsolationSourceDiagnostic.EXACT_MATCH,)
    assert fake.calls[0]["messages"][0]["role"] == "system"
    assert fake.calls[0]["messages"][0]["content"].startswith("Synthetic ontology:\n## Tree\n")
    assert "- wound" in fake.calls[0]["messages"][0]["content"]
    assert fake.calls[0]["messages"][1] == {
        "role": "user",
        "content": (
            "Synthetic sample:\n"
            "Metadata:\n"
            "isolation_source = wound patient 739105\n"
            "host = Homo sapiens\n"
        ),
    }
    assert fake.calls[0]["model"] == "test-model"
    assert fake.calls[0]["temperature"] == 0
    assert fake.calls[0]["seed"] == 100
    assert fake.calls[0]["response_model"].__name__ == "IsolationSourceClassification"
    assert outcome.request_fingerprint


@pytest.mark.parametrize(
    "changed_component",
    ["message", "model", "parameter", "schema_id"],
)
def test_isolation_source_cache_misses_when_canonical_request_changes(
    tmp_path, monkeypatch, changed_component
):
    fake = FakeClient()
    fake.respond_with(["wound"], reasoning="clinical wound")
    first_config = _isolation_source_config(tmp_path)
    _standardize_model_isolation_source(first_config, fake)

    second_config = _isolation_source_config(tmp_path)
    second_model = "test-model"
    if changed_component == "message":
        second_config = _isolation_source_config(
            tmp_path,
            overrides={"system_prompt": "{ontology_tree}\nA changed fully rendered system prompt."},
        )
    elif changed_component == "model":
        second_model = "changed-model"
    elif changed_component == "parameter":
        monkeypatch.setattr(
            isolation_source_module,
            "ISOLATION_SOURCE_LLM_PARAMETERS",
            {"temperature": 0, "seed": 101},
        )
    elif changed_component == "schema_id":
        monkeypatch.setattr(
            isolation_source_module,
            "ISOLATION_SOURCE_RESPONSE_SCHEMA_ID",
            "baccurate.isolation.classification.changed",
        )

    second = _standardize_model_isolation_source(
        second_config,
        fake,
        model=second_model,
    )

    assert second.diagnostics == (IsolationSourceDiagnostic.LLM_CALL,)
    assert len(fake.calls) == 2


# --- Cache persistence adapter ---


def test_cache_round_trip_preserves_classification_audit_fields(cache):
    rec = StandardizedIsolationSource(
        term_path_roots="host-associated",
        display_terms="feces",
        external_ontology_identifiers="ENVO:00002003",
        term_paths=FECES,
        reasoning=[{"node": "direct_match", "reasoning": "x", "selected_term_paths": [FECES]}],
        evidence_level=IsolationSourceEvidenceLevel.SAMPLE,
    )
    cache.set("fingerprint", rec)

    got = cache.get("fingerprint")

    assert got is not None
    assert got.term_path_roots == "host-associated"
    assert got.display_terms == "feces"
    assert got.external_ontology_identifiers == "ENVO:00002003"
    assert got.term_paths == FECES
    assert got.reasoning == [
        {"node": "direct_match", "reasoning": "x", "selected_term_paths": [FECES]}
    ]
    assert got.evidence_level is IsolationSourceEvidenceLevel.SAMPLE
