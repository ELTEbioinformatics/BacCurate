"""
Tests for baccurate.standardizers.isolation.

Uses a monkeypatch instead of calling an actual LLM API.

Run with:
uv run pytest tests/test_stand_iso.py -v
or
pytest tests/test_stand_iso.py -v
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from baccurate.standardizers.isolation import (
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
# 1. Value masking (SQLiteCache._mask_value)
# =============================================================================


@pytest.mark.parametrize("value", ["2020-02-01", "12/07/2025", "Jan 2011"])
def test_dates_are_masked(value):
    assert SQLiteCache._mask_value(value) == "<DATE>"


def test_coordinate_is_masked():
    assert SQLiteCache._mask_value("37.9 N") == "<COORD>"


def test_lab_unit_is_masked():
    assert SQLiteCache._mask_value("10 mg") == "<MEASURE>"


def test_dimension_is_masked():
    assert SQLiteCache._mask_value("1x4") == "<DIMENSION>"


def test_percentage_number_is_masked():
    assert SQLiteCache._mask_value("50%") == SQLiteCache._mask_value("90%")


@pytest.mark.parametrize("value", ["Patient-1", "A1"])
def test_identifiers_are_masked(value):
    assert SQLiteCache._mask_value(value) == "<ID>"


def test_bare_number_is_masked():
    assert SQLiteCache._mask_value("patient 1") == "patient <NUM>"


def test_number_glued_to_a_word_is_masked():
    a = SQLiteCache._mask_value("patient1")
    b = SQLiteCache._mask_value("patient2")

    assert a == b == "patient<NUM>"


def test_different_stems_with_glued_numbers_stay_distinct():
    assert SQLiteCache._mask_value("patient1") != SQLiteCache._mask_value("sample1")


def test_values_differing_only_by_number_mask_identically():
    a = SQLiteCache._mask_value("Stool Sample Patient 1")
    b = SQLiteCache._mask_value("stool sample patient 2")

    assert a == b
    assert a == "stool sample patient <NUM>"


def test_ontology_id_digits_are_not_masked_as_numbers():
    masked = SQLiteCache._mask_value("ENVO:00002003")

    assert masked == "envo:00002003"
    assert "<NUM>" not in masked


def test_distinct_ontology_ids_do_not_collide_after_masking():
    """Different ontology IDs should not share a cache key."""
    assert SQLiteCache._mask_value("ENVO:00002003") != SQLiteCache._mask_value("ENVO:00002005")


# =============================================================================
# 2. Cache hashing + round-trip (SQLiteCache)
# =============================================================================


def test_hash_is_stable_for_identical_inputs(cache):
    h1 = cache._generate_hash("isolation_source", "blood", "human", "m1")
    h2 = cache._generate_hash("isolation_source", "blood", "human", "m1")

    assert h1 == h2


@pytest.mark.parametrize(
    "attr,value,host,model",
    [
        ("tissue", "blood", "human", "m1"),            # different attr
        ("isolation_source", "blood", "mouse", "m1"),  # different host
        ("isolation_source", "blood", "human", "m2"),  # different model
    ],
)
def test_hash_differs_when_any_keyed_field_differs(cache, attr, value, host, model):
    base = cache._generate_hash("isolation_source", "blood", "human", "m1")

    assert cache._generate_hash(attr, value, host, model) != base


def test_masking_equivalent_values_share_a_hash(cache):
    h1 = cache._generate_hash("iso", "patient 1", "human", "m1")
    h2 = cache._generate_hash("iso", "patient 2", "human", "m1")

    assert h1 == h2


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
# 3. Ontology parsing (OntologyManager)
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
# 4. Direct match (LLMClassifier._direct_match)
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
# 5. Metadata formatting (LLMClassifier._format_metadata)
# =============================================================================


def test_format_metadata_includes_host_when_present():
    out = LLMClassifier._format_metadata(["isolation_source"], ["blood"], "Homo sapiens")

    assert "isolation_source = blood" in out
    assert "host = Homo sapiens" in out


def test_format_metadata_omits_blank_host():
    out = LLMClassifier._format_metadata(["isolation_source"], ["blood"], "   ")

    assert "host =" not in out


# =============================================================================
# 6. standardize_record
# =============================================================================

# --- Null / trivial inputs -------------------------------------------------


def test_null_value_resolves_to_unspecified_without_llm(classifier):
    res = classifier.standardize_record("T", "isolation_source", "missing", "", "")

    assert res.display_terms == "unspecified"
    assert res.categories == "unspecified"
    assert res.ontology_links == "NA"
    assert classifier.stats["llm_calls"] == 0
    assert classifier._fake.calls == []


def test_empty_value_resolves_to_unspecified_without_llm(classifier):
    res = classifier.standardize_record("T", "isolation_source", "   ", "", "")

    assert res.display_terms == "unspecified"
    assert classifier._fake.calls == []


# --- Direct-match -------------------------------------


def test_direct_match_skips_llm_and_applies_crosslink(classifier):
    res = classifier.standardize_record("T", "isolation_source", "stool", "", "")

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
    res = classifier.standardize_record(
        "T", "isolation_source||tissue", "stool||blood", "", ""
    )

    paths = res.term_paths.split("||")
    assert FECES in paths
    assert BLOOD in paths
    assert classifier.stats["llm_calls"] == 0
    assert classifier._fake.calls == []


# --- LLM branch -------------------------------------------


def test_llm_selection_is_mapped_to_term_path(classifier):
    classifier._fake.respond_with(["blood"], reasoning="venous blood")
    res = classifier.standardize_record("T", "isolation_source", "venous draw", "", "")

    assert classifier.stats["llm_calls"] == 1
    assert "blood" in res.display_terms.split("||")
    assert BLOOD in res.term_paths.split("||")
    nodes = [h.get("node") for h in res.reasoning]
    assert "classifier" in nodes
    classifier_entry = next(h for h in res.reasoning if h.get("node") == "classifier")
    assert classifier_entry["selected_terms"] == ["blood"]


def test_llm_selection_expands_along_crosslinks(classifier):
    classifier._fake.respond_with(["feces"])
    res = classifier.standardize_record("T", "isolation_source", "gut contents", "", "")

    terms = res.display_terms.split("||")
    assert "feces" in terms
    assert "rectum" in terms  # crosslink feces -> rectum
    nodes = [h.get("node") for h in res.reasoning]
    assert "crosslink" in nodes


def test_direct_and_llm_paths_are_unioned(classifier):
    classifier._fake.respond_with(["blood"])
    res = classifier.standardize_record(
        "T", "isolation_source||tissue", "stool||venous draw", "", ""
    )

    paths = res.term_paths.split("||")
    assert FECES in paths  # from direct match
    assert BLOOD in paths  # from LLM
    assert classifier.stats["llm_calls"] == 1


# --- Expected mismatch warning) -----------------------------------


def test_warn_when_expected_node_absent(classifier, caplog):
    classifier._fake.respond_with(["blood"])
    with caplog.at_level("WARNING"):
        res = classifier.standardize_record("ACC", "food_source", "mystery", "", "")

    assert any("attribute_shortcut mismatch" in r.getMessage() for r in caplog.records)
    assert "food" not in res.display_terms.split("||")


def test_no_warning_for_descendant_term(classifier, caplog):
    classifier._fake.respond_with(["meat product"])
    with caplog.at_level("WARNING"):
        classifier.standardize_record("ACC", "food_source", "ground beef", "", "")

    assert not any("attribute_shortcut mismatch" in r.getMessage() for r in caplog.records)


# --- LLM failure handling --------------------------------------------------


def test_llm_failure_with_no_direct_match_falls_back_to_unspecified(classifier):
    classifier._fake.fail_with(RuntimeError("boom"))
    res = classifier.standardize_record("T", "isolation_source", "venous draw", "", "")

    assert res.display_terms == "unspecified"
    assert any("error" in h for h in res.reasoning)


def test_llm_failure_still_keeps_direct_matches(classifier):
    classifier._fake.fail_with(RuntimeError("boom"))
    res = classifier.standardize_record(
        "T", "isolation_source||tissue", "stool||venous draw", "", ""
    )

    assert "feces" in res.display_terms.split("||")
    assert res.display_terms != "unspecified"


# --- Caching within standardize_record -------------------------------------


def test_identical_call_hits_cache_without_reinvoking_llm(classifier):
    classifier.standardize_record("T", "isolation_source", "stool", "", "")
    classifier.standardize_record("T", "isolation_source", "stool", "", "")

    assert classifier.stats["cache_hits"] == 1
    assert classifier._fake.calls == []


def test_masking_equivalent_value_hits_cache(classifier):
    classifier._fake.respond_with(["wound"])
    classifier.standardize_record("T", "isolation_source", "wound patient 1", "", "")
    assert len(classifier._fake.calls) == 1

    classifier.standardize_record("T", "isolation_source", "wound patient 2", "", "")

    assert len(classifier._fake.calls) == 1  # no second LLM call
    assert classifier.stats["cache_hits"] >= 1
