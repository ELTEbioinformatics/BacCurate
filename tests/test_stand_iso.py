"""
Tests for isolation source standardziation

Uses a monkeypatch instead of calling an actual LLM API.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from baccurate.standardizers.host import HostOverflowContext
from baccurate.standardizers.isolation import (
    IsolationDiagnostic,
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
MANURE = "environmental:anthropogenic environment:farm:manure"


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

    def respond_with(self, terms, reasoning: str = "because") -> None:
        self.behavior = SimpleNamespace(terms=list(terms), reasoning=reasoning)

    def fail_with(self, exc: Exception) -> None:
        self.behavior = exc


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
    standardizer = IsoStandardizer(config_path)

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
    assert result.diagnostics == (IsolationDiagnostic.EXACT_MATCH,)


def test_typed_record_outcome_preserves_model_reasoning_on_exact_cache_hit(tmp_path, monkeypatch):
    config_path = tmp_path / "isolation.yaml"
    config_path.write_text(
        CONFIG_PATH.read_text(encoding="utf-8")
        + f'\ncache_db_path: "{(tmp_path / "iso-cache.db").as_posix()}"\n',
        encoding="utf-8",
    )
    monkeypatch.setenv("LLM_MODEL", "test-model")
    first_standardizer = IsoStandardizer(config_path, client=None)
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

    cached_standardizer = IsoStandardizer(config_path, client=None)
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
    assert cached.reasoning == modelled.reasoning
    assert cached.diagnostics == (IsolationDiagnostic.CACHE_HIT,)


# =============================================================================
# SQLite cache
# =============================================================================


@pytest.mark.parametrize(
    "stored_value,changed_value",
    [
        ("sampled 2020-02-01", "sampled 2020-02-02"),
        ("37.9 N", "38.0 N"),
        ("10 mg", "11 mg"),
        ("1x4 cm", "1x5 cm"),
        ("Patient-1", "Patient-2"),
        ("50%", "90%"),
        ("patient 1", "patient 2"),
        ("patient1", "patient2"),
        ("ENVO:00002003", "ENVO:00002005"),
    ],
)
def test_cache_does_not_reuse_changed_evidence(cache, feces_source, stored_value, changed_value):
    cache.set("isolation_source", stored_value, "human", feces_source, "m1")

    assert cache.get("isolation_source", changed_value, "human", "m1") is None


def test_cache_reuses_the_same_normalized_evidence(cache, feces_source):
    cache.set(" Isolation Source ", " Stool Sample ", " Homo sapiens ", feces_source, " M1 ")

    assert cache.get("isolation_source", "stool sample", "homo sapiens", "m1") == feces_source


def test_set_then_get_round_trips_the_record(cache):
    rec = StandardizedSource(
        categories="host-associated",
        display_terms="feces",
        ontology_links="ENVO:00002003",
        term_paths=FECES,
        reasoning=[{"node": "direct_match", "reasoning": "x", "selections": [FECES]}],
    )
    cache.set("isolation_source", "stool", "human", rec, "m1")

    got = cache.get("isolation_source", "stool", "human", "m1")

    assert got is not None
    assert got.categories == "host-associated"
    assert got.display_terms == "feces"
    assert got.ontology_links == "ENVO:00002003"
    assert got.term_paths == FECES
    assert got.reasoning == [{"node": "direct_match", "reasoning": "x", "selections": [FECES]}]


def test_get_on_unknown_key_returns_none(cache):
    assert cache.get("isolation_source", "never stored", "human", "m1") is None


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


@pytest.mark.parametrize(
    "source,target",
    [
        (FECES, RECTUM),
        (MANURE, FECES),
    ],
)
def test_crosslinks_resolve_to_target_paths(ontology, source, target):
    assert target in ontology.crosslink_map[source]


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


def test_direct_match_skips_llm_and_applies_crosslink(classifier):
    res = classifier.standardize_record("T", "isolation_source", "stool", "")

    terms = res.display_terms.split("||")
    paths = res.term_paths.split("||")
    assert "feces" in terms
    assert "rectum" in terms  # crosslink feces -> rectum
    assert FECES in paths
    assert RECTUM in paths
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
    assert "blood" in res.display_terms.split("||")
    assert BLOOD in res.term_paths.split("||")
    nodes = [h.get("node") for h in res.reasoning]
    assert "classifier" in nodes
    classifier_entry = next(h for h in res.reasoning if h.get("node") == "classifier")
    assert classifier_entry["selected_terms"] == ["blood"]


def test_llm_selection_expands_along_crosslinks(classifier):
    classifier._fake.respond_with(["feces"])
    res = classifier.standardize_record("T", "isolation_source", "gut contents", "")

    terms = res.display_terms.split("||")
    assert "feces" in terms
    assert "rectum" in terms  # crosslink feces -> rectum
    nodes = [h.get("node") for h in res.reasoning]
    assert "crosslink" in nodes


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
