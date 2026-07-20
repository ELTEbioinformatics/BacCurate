"""Tests for host standardization."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from baccurate.standardizers.host import HostDiagnostic, HostStandardizer

ROOT = Path(__file__).parent.parent
CONFIG_PATH = ROOT / "config" / "host.yaml"
NCBI_PATH = ROOT / "data" / "reference" / "taxonomy" / "taxids_ncbi.tsv"


@pytest.fixture(scope="session")
def standardizer() -> HostStandardizer:
    return HostStandardizer(CONFIG_PATH, NCBI_PATH)


@pytest.fixture
def minimal_ncbi_path(tmp_path: Path) -> Path:
    path = tmp_path / "taxids_ncbi.tsv"
    path.write_text(
        "\n".join(
            [
                "taxid\trank\tscientific_name\tsynonym\tgenbank_common_name\tcommon_name\tcomments",
                "9606\tspecies\tHomo sapiens\t\t\t\t",
                "9913\tspecies\tBos taurus\t\t\t\t",
                "9520\tgenus\tSaimiri\t\t\t\t",
                "9605\tgenus\tHomo\t\t\t\t",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def write_host_config(
    tmp_path: Path,
    *,
    curated_taxa: dict[str, object] | None = None,
    value_rejections: list[str] | None = None,
    schema_version: int = 2,
    extra: dict[str, object] | None = None,
) -> Path:
    config: dict[str, object] = {
        "schema_version": schema_version,
        "normalization": {"ignored_substrings": []},
        "routing": {"isolation_source_keywords": []},
        "curated_taxa": curated_taxa or {},
        "value_rejections": {"exact": value_rejections or []},
    }
    config.update(extra or {})
    path = tmp_path / "host.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return path


def classify(standardizer: HostStandardizer, value: str, attribute: str = "host"):
    """Standardize a single (attribute, value)."""
    return standardizer.classify_row("TEST", attribute, value)


# =============================================================================
# Record-level outcomes
# =============================================================================


def test_record_standardization_rejects_misaligned_host_candidate_counts(standardizer):
    with pytest.raises(
        ValueError,
        match=r"MALFORMED.*host_attr_orig=2.*host_val_orig=1.*counts must match",
    ):
        standardizer.standardize(
            {
                "accession": "MALFORMED",
                "host_attr_orig": "host||host_taxid",
                "host_val_orig": "human",
            }
        )


def test_record_outcome_distinguishes_standardized_host_overflow_and_absence(standardizer):
    matched = standardizer.standardize(
        {
            "accession": "MATCHED",
            "host_attr_orig": "host_taxid||host",
            "host_val_orig": "9606||human",
        }
    )
    overflow = standardizer.standardize(
        {
            "accession": "OVERFLOW",
            "host_attr_orig": "host",
            "host_val_orig": "no host",
        }
    )
    absent = standardizer.standardize(
        {
            "accession": "ABSENT",
            "host_attr_orig": "",
            "host_val_orig": "",
        }
    )
    context_bearing_bacterium = standardizer.standardize(
        {
            "accession": "BACTERIAL_CONTEXT",
            "host_attr_orig": "host",
            "host_val_orig": "Escherichia coli str. K-12",
        }
    )
    broad_host = standardizer.standardize(
        {
            "accession": "BROAD",
            "host_attr_orig": "host",
            "host_val_orig": "swine",
        }
    )

    assert matched.standardized.taxid == 9606
    assert matched.standardized.scientific_name == "Homo sapiens"
    assert matched.score == 1.0
    assert matched.low_confidence is False
    assert matched.diagnostics == (HostDiagnostic.MATCHED,)
    assert [(origin.attribute, origin.value) for origin in matched.origins] == [
        ("host_taxid", "9606")
    ]
    assert matched.overflow is None
    assert overflow.standardized is None
    assert overflow.overflow.attribute == "host"
    assert overflow.overflow.value == "no host"
    assert absent.standardized is None
    assert absent.overflow is None
    assert context_bearing_bacterium.standardized is None
    assert context_bearing_bacterium.overflow.value == "Escherichia coli str. K-12"
    assert broad_host.standardized.taxid == 9823
    assert broad_host.score == 0.7


def test_retry_uses_eligible_isolation_origins_without_iso_keyword_preemption(standardizer):
    outcome = standardizer.retry("RETRY", "food_source", "chicken meat")

    assert outcome.standardized.taxid == 9031
    assert outcome.standardized.scientific_name == "Gallus gallus"
    assert len(outcome.origins) == 1
    assert (outcome.origins[0].attribute, outcome.origins[0].value) == (
        "food_source",
        "chicken meat",
    )
    assert outcome.retry_eligible


# =============================================================================
# Match score tiers
# =============================================================================


def test_numeric_taxid_under_host_taxid_attribute_resolves_directly(standardizer):
    """A bare taxid carried in the `host_taxid` attribute should skip text matching."""
    match = classify(standardizer, "9606", attribute="host_taxid")

    assert match.info.taxid == 9606
    assert match.info.scientific_name == "Homo sapiens"
    assert match.score == 1.0
    assert match.low_confidence is False


def test_scientific_name_match_returns_full_score(standardizer):
    match = classify(standardizer, "Homo sapiens")

    assert match.info.taxid == 9606
    assert match.score == 1.0


def test_ncbi_synonym_match_returns_full_score(standardizer):
    """'Bos bovis' is an NCBI synonym for Bos taurus (taxid 9913)."""
    match = classify(standardizer, "Bos bovis")

    assert match.info.taxid == 9913
    assert match.score == 1.0


def test_curated_term_match_scores_0_95(standardizer):
    """'Holstein' is a locally curated subset term for Bos taurus."""
    match = classify(standardizer, "Holstein")

    assert match.info.taxid == 9913
    assert match.score == 0.95


def test_genbank_common_name_match_scores_0_90(standardizer):
    """'human' is the NCBI genbank_common_name for Homo sapiens."""
    match = classify(standardizer, "human")

    assert match.info.taxid == 9606
    assert match.score == 0.9


def test_broad_common_name_match_scores_0_70(standardizer):
    """'swine' is one of the broad NCBI common_names for Sus scrofa."""
    match = classify(standardizer, "swine")

    assert match.info.taxid == 9823
    assert match.score == 0.7


def test_multiword_subset_match_scores_0_70(standardizer):
    """A recognized binomial embedded in a longer phrase resolves via subset."""
    match = classify(standardizer, "Homo sapiens sample")

    assert match.info.taxid == 9606
    assert match.score == 0.7
    assert match.match_tier == "multi-word"
    assert match.low_confidence is True
    assert match.diagnostics == (HostDiagnostic.SUBSET_MATCH,)


def test_singleword_subset_match_scores_0_50(standardizer):
    """A single recognized word in an otherwise unrecognized phrase."""
    match = classify(standardizer, "elderly human hospitalised")

    assert match.info.taxid == 9606
    assert match.score == 0.5
    assert match.match_tier == "single-word"
    assert match.low_confidence is True


def test_curated_subset_term_can_target_a_subspecies(standardizer: HostStandardizer) -> None:
    match = classify(standardizer, "sample Canis lupus familaris")

    assert match.info.taxid == 9615
    assert match.score == 0.7
    assert match.low_confidence is True


# =============================================================================
# Normalization test
# =============================================================================


@pytest.mark.parametrize(
    "input_string",
    [
        "Homo sapiens",
        "HOMO SAPIENS",
        "homo sapiens",
        "Homo_sapiens",
        "Homo-sapiens",
        "  Homo   sapiens  ",
        "Homo sapiens.",
    ],
)
def test_normalization_collapses_surface_variants(standardizer, input_string):
    match = classify(standardizer, input_string)

    assert match.info.taxid == 9606
    assert match.score == 1.0


# =============================================================================
# Trusted candidates and ignored substrings
# =============================================================================


def test_standardizer_does_not_reapply_extraction_rejection(standardizer):
    """lab strains retained by extraction reach host matching."""
    match = classify(standardizer, "Anopheles gambiae G3 strain, lab colony")

    assert match.info.taxid == 7165


def test_ignored_substrings_are_stripped_before_matching(standardizer):
    """'healthy' and ', adult' should be removed. The remaining string ('human') is resolved."""
    match = classify(standardizer, "healthy human, adult")

    assert match.info.taxid == 9606
    assert match.score == 0.9


def test_value_consisting_only_of_ignored_substrings_returns_no_match(standardizer):
    """'healthy' alone is stripped, leaving an empty value to match."""
    assert classify(standardizer, "healthy") is None


# =============================================================================
# Numeric value gating
# =============================================================================


def test_numeric_value_under_non_taxid_attribute_is_rejected(standardizer):
    """A bare number in the `host` attribute is not a meaningful host name.

    The same number under `host_taxid` resolves — see
    test_numeric_taxid_under_host_taxid_attribute_resolves_directly.
    """
    assert classify(standardizer, "9606", attribute="host") is None


# =============================================================================
# Iso-source keyword preemption
# =============================================================================
#
# Values matching an iso_keyword (food, meat, environment, soil) should be forwarded
# to the iso-source pipeline rather than host-matched.


def test_value_equal_to_iso_keyword_is_preempted(standardizer):
    assert classify(standardizer, "meat") is None


def test_value_containing_iso_keyword_as_whole_word_is_preempted(standardizer):
    """'duck meat' contains 'meat' as a whole word and is forwarded to iso."""
    assert classify(standardizer, "duck meat") is None


def test_iso_keyword_as_substring_inside_another_word_does_not_preempt(standardizer):
    """The value falls through to host matching, where 'human' (single-word
    subset) provides the only signal.
    """
    match = classify(standardizer, "seafood and human sample")  # 'food' is not whole-word.

    assert match is not None
    assert match.info.taxid == 9606


# =============================================================================
# Preemptive host decisions
# =============================================================================


def test_force_term_beats_default_matching(standardizer):
    """`Squirrel monkey` would otherwise resolve to Sciurus (genus) via
    single-word subset. The force term selects Saimiri (taxid 9520).
    """
    match = classify(standardizer, "Squirrel monkey")

    assert match.info.taxid == 9520
    assert match.score == 1.0
    assert match.low_confidence is False


def test_value_rejection_forwards_to_iso_and_returns_no_match(standardizer):
    """`cancer` is rejected as a host and retained for isolation interpretation."""
    assert classify(standardizer, "cancer") is None


# =============================================================================
# Subspecies trinomial guard
# =============================================================================
#
# Subspecies (e.g. Gallus gallus gallus, taxid 208526) are kept in the
# exact-match lookups but excluded from the subset-match index so that a
# 2-word input like 'Gallus gallus' cannot accidentally resolve to a 3-word
# subspecies trinomial.


def test_species_binomial_does_not_resolve_to_a_subspecies(standardizer):
    """'Gallus gallus' should resolve to the species (9031), not the subspecies."""
    match = classify(standardizer, "Gallus gallus")

    assert match.info.taxid == 9031
    assert match.score == 1.0


def test_subspecies_trinomial_still_resolves_via_exact_match(standardizer):
    """The full trinomial should resolve to the subspecies."""
    match = classify(standardizer, "Gallus gallus gallus")

    assert match.info.taxid == 208526
    assert match.score == 1.0


# =============================================================================
# Multi-attribute records and the low_confidence flag
# =============================================================================
#
# When a record has multiple host-related attributes, all candidates are
# ranked and one winner is chosen. `low_confidence` is orthogonal to the
# score - a 1.0 match can still be flagged if attributes disagree, and any
# subset match (0.7 or 0.5) is flagged regardless of agreement.


def test_worked_example_from_documentation(standardizer):
    """Example from docs/host.md: taxid + free-text both pointing at Homo sapiens.

    host_taxid should win on score. low_confidence should stay False because both
    candidates resolve to the same taxon.
    """
    match = standardizer.classify_row(
        accession="TEST",
        attr_str="host_taxid||host",
        val_str="9606||healthy human, adult",
    )

    assert match.info.taxid == 9606
    assert match.score == 1.0
    assert match.attribute == "host_taxid"
    assert match.value == "9606"
    assert match.low_confidence is False


def test_attribute_disagreement_flags_low_confidence_but_keeps_best_score(standardizer):
    """If host_taxid and host point at different taxa, low_confidence should be True.

    The higher-scoring candidate still wins, but the row is marked for review.
    """
    match = standardizer.classify_row(
        accession="TEST",
        attr_str="host_taxid||host",
        val_str="9606||Bos taurus",
    )

    assert match.info.taxid == 9606
    assert match.score == 1.0
    assert match.low_confidence is True
    assert HostDiagnostic.ATTRIBUTE_DISAGREEMENT in match.diagnostics


def test_host_taxid_wins_tiebreaker_over_host_at_equal_score(standardizer):
    """Both attributes should resolve to Homo sapiens with score 1.0.
    host_taxid should win on attribute priority.
    """
    match = standardizer.classify_row(
        accession="TEST",
        attr_str="host||host_taxid",
        val_str="Homo sapiens||9606",
    )

    assert match.info.taxid == 9606
    assert match.attribute == "host_taxid"
    assert match.low_confidence is False


def test_high_score_match_with_no_disagreement_is_not_low_confidence(standardizer):
    """A clean 0.9 genbank_common_name match should not be flagged."""
    match = classify(standardizer, "human")

    assert match.score == 0.9
    assert match.low_confidence is False
