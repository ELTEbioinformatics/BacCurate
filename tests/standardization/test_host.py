"""Protect host standardization with compact domain scenarios and one production smoke test."""

from __future__ import annotations

from pathlib import Path

import pytest

from baccurate.standardization.host import (
    HostDiagnostic,
    HostPolicy,
    HostStandardizer,
    MatchRoute,
)
from baccurate.standardization.host_lineage import HostLineageEnricher
from baccurate.taxon_registry.registry import load_taxon_registry

ROOT = Path(__file__).parents[2]
CONFIG_PATH = ROOT / "config" / "host.yaml"
TAXON_REGISTRY_PATH = ROOT / "config" / "taxa.yaml"
NCBI_PATH = ROOT / "data" / "reference" / "taxonomy" / "taxids_ncbi.tsv"


@pytest.fixture(scope="session")
def standardizer(fixture_host_policy, standardization_fixture_resources) -> HostStandardizer:
    """Use the compact taxonomy for every detailed host scenario."""
    return HostStandardizer(
        fixture_host_policy,
        standardization_fixture_resources.ncbi_taxonomy_reference_table,
    )


def classify(standardizer: HostStandardizer, value: str, attribute: str = "host"):
    """Standardize a single BioSample attribute-value pair."""
    return standardizer.classify_extracted_record("TEST", attribute, value)


def test_production_host_policy_and_taxonomy_have_required_semantics() -> None:
    """Smoke-test production resources once, including rejection vocabulary semantics."""
    registry = load_taxon_registry(TAXON_REGISTRY_PATH)
    standardizer = HostStandardizer(HostPolicy.load(CONFIG_PATH, registry), NCBI_PATH)

    taxonomic_match = classify(standardizer, "Homo sapiens")
    assert taxonomic_match.info.taxid == 9606
    assert taxonomic_match.match_quality_score == 1.0

    for submitted_value in (
        "Bovine_sWine-Pool!",  # normalized rejection
        "Escherichia coli",  # target-taxon-derived rejection
        "cancer",  # deliberately curated rejection
    ):
        outcome = standardizer.standardize(
            {
                "accession": "PRODUCTION_REJECTION",
                "host_attr_orig": "host",
                "host_val_orig": submitted_value,
            }
        )
        assert outcome.standardized is None
        assert outcome.overflow is not None
        assert outcome.overflow.value == submitted_value
        assert outcome.diagnostics == (HostDiagnostic.OVERRIDE_REJECTION,)

    intentionally_retained = standardizer.standardize(
        {
            "accession": "PRODUCTION_NON_REJECTION",
            "host_attr_orig": "host",
            "host_val_orig": "Klebsiella variicola",
        }
    )
    assert intentionally_retained.overflow is not None
    assert intentionally_retained.diagnostics == (HostDiagnostic.UNMATCHED,)


def test_record_outcomes_preserve_standardized_overflow_and_missing_metadata(
    standardizer: HostStandardizer,
) -> None:
    matched = standardizer.standardize(
        {
            "accession": "MATCHED",
            "host_attr_orig": "host_taxid||host",
            "host_val_orig": "9606||human",
        }
    )
    rejected = standardizer.standardize(
        {
            "accession": "REJECTED",
            "host_attr_orig": "host",
            "host_val_orig": "no host",
        }
    )
    unmatched = standardizer.standardize(
        {
            "accession": "UNMATCHED",
            "host_attr_orig": "host",
            "host_val_orig": "Escherichia coli str. K-12",
        }
    )
    missing_metadata = standardizer.standardize(
        {
            "accession": "MISSING_METADATA",
            "host_attr_orig": "",
            "host_val_orig": "",
        }
    )

    assert matched.standardized.taxid == 9606
    assert matched.standardized.scientific_name == "Homo sapiens"
    assert matched.match_quality_score == 1.0
    assert matched.needs_review is False
    assert matched.diagnostics == (HostDiagnostic.MATCHED,)
    assert [(pair.attribute, pair.value) for pair in matched.supporting_pairs] == [
        ("host_taxid", "9606")
    ]
    assert matched.overflow is None

    assert rejected.standardized is None
    assert rejected.supporting_pairs == ()
    assert rejected.overflow.attribute == "host"
    assert rejected.overflow.value == "no host"
    assert rejected.diagnostics == (HostDiagnostic.OVERRIDE_REJECTION,)

    assert unmatched.standardized is None
    assert unmatched.overflow.value == "Escherichia coli str. K-12"
    assert unmatched.diagnostics == (HostDiagnostic.UNMATCHED,)

    assert missing_metadata.standardized is None
    assert missing_metadata.overflow is None
    assert missing_metadata.diagnostics == (HostDiagnostic.UNMATCHED,)


def test_exact_matches_report_their_route_and_score(
    standardizer: HostStandardizer,
) -> None:
    scenarios = (
        ("9606", "host_taxid", 9606, MatchRoute.TAXID, 1.0),
        ("Homo sapiens", "host", 9606, MatchRoute.SCIENTIFIC_NAME, 1.0),
        ("Bos bovis", "host", 9913, MatchRoute.SYNONYM, 1.0),
        ("Holstein", "host", 9913, MatchRoute.CURATED_TERM, 0.95),
        ("domestic cattle", "host", 9913, MatchRoute.CURATED_COMMON_NAME, 0.9),
        ("swine", "host", 9823, MatchRoute.BROAD_COMMON_NAME, 0.7),
    )

    for value, attribute, expected_taxid, expected_route, expected_score in scenarios:
        match = classify(standardizer, value, attribute)
        assert match.info.taxid == expected_taxid
        assert match.route == expected_route
        assert match.match_quality_score == expected_score
        assert match.needs_review is False


def test_subset_and_taxonomic_rank_scenarios_retain_scores_and_review_decisions(
    standardizer: HostStandardizer,
) -> None:
    scenarios = (
        ("Homo sapiens sample", 9606, 0.7, MatchRoute.SUBSET_MULTI_WORD, True),
        ("elderly human hospitalised", 9606, 0.5, MatchRoute.SUBSET_SINGLE_WORD, True),
        ("sample Canis lupus familaris", 9615, 0.7, MatchRoute.SUBSET_MULTI_WORD, True),
        ("Gallus gallus", 9031, 1.0, MatchRoute.SCIENTIFIC_NAME, False),
        ("Gallus gallus gallus", 208526, 1.0, MatchRoute.SCIENTIFIC_NAME, False),
    )

    for value, expected_taxid, expected_score, expected_route, needs_review in scenarios:
        match = classify(standardizer, value)
        assert match.info.taxid == expected_taxid
        assert match.match_quality_score == expected_score
        assert match.route == expected_route
        assert match.needs_review is needs_review

    subset = classify(standardizer, "Homo sapiens sample")
    assert subset.diagnostics == (HostDiagnostic.SUBSET_MATCH,)

    broad_common = classify(standardizer, "swine")
    multi_word = classify(standardizer, "Homo sapiens sample")
    assert broad_common.match_quality_score == multi_word.match_quality_score
    assert broad_common.route != multi_word.route


def test_normalization_collapses_host_surface_variants(standardizer: HostStandardizer) -> None:
    for submitted_value in (
        "Homo sapiens",
        "HOMO SAPIENS",
        "homo sapiens",
        "Homo_sapiens",
        "Homo-sapiens",
        "  Homo   sapiens  ",
        "Homo sapiens.",
    ):
        match = classify(standardizer, submitted_value)
        assert match.info.taxid == 9606
        assert match.match_quality_score == 1.0


def test_ignored_substrings_do_not_hide_trusted_or_empty_host_values(
    standardizer: HostStandardizer,
) -> None:
    trusted_lab_strain = classify(
        standardizer,
        "Anopheles gambiae G3 strain, lab colony",
    )
    cleaned_host = classify(standardizer, "healthy domestic cattle, adult")

    assert trusted_lab_strain.info.taxid == 7165
    assert cleaned_host.info.taxid == 9913
    assert cleaned_host.match_quality_score == 0.9
    assert classify(standardizer, "healthy") is None


def test_numeric_values_are_gated_by_host_taxid_attribute(
    standardizer: HostStandardizer,
) -> None:
    assert classify(standardizer, "9606", attribute="host") is None
    taxid_match = classify(standardizer, "9606", attribute="host_taxid")
    assert taxid_match.info.taxid == 9606
    assert taxid_match.match_quality_score == 1.0


def test_numeric_general_host_value_falls_through_to_text_matching(
    fixture_host_policy,
    standardization_fixture_resources,
    tmp_path,
) -> None:
    taxonomy_path = tmp_path / "taxonomy.tsv"
    taxonomy_path.write_text(
        standardization_fixture_resources.ncbi_taxonomy_reference_table.read_text(encoding="utf-8")
        + "111\tspecies\t9606\t\t\t\t\n",
        encoding="utf-8",
    )
    numeric_name_standardizer = HostStandardizer(
        fixture_host_policy,
        ncbi_table_path=taxonomy_path,
    )

    text_match = classify(numeric_name_standardizer, "9606", attribute="host")
    taxid_match = classify(numeric_name_standardizer, "9606", attribute="host_taxid")

    assert text_match.info.taxid == 111
    assert taxid_match.info.taxid == 9606


def test_prefixed_taxonomy_identifier_resolves_under_any_host_attribute(
    standardizer: HostStandardizer,
) -> None:
    for attribute in ("host_taxid", "host"):
        match = classify(standardizer, "NCBITaxon:9031", attribute=attribute)

        assert match.info.taxid == 9031
        assert match.match_quality_score == 1.0
        assert match.needs_review is False


def test_taxonomy_identifier_resolves_in_each_label_pairing_convention(
    standardizer: HostStandardizer,
) -> None:
    for submitted_value in (
        "Chicken [NCBITaxon:9031]",
        "Chicken (NCBITaxon:9031)",
        "Chicken NCBITaxon:9031",
        "NCBITaxon:9031 Chicken",
        "Chicken [ncbitaxon:9031]",
    ):
        match = classify(standardizer, submitted_value)

        assert match.info.taxid == 9031
        assert match.match_quality_score == 1.0
        assert match.needs_review is False


def test_taxonomy_identifier_wins_a_disagreeing_label_with_review_flag(
    standardizer: HostStandardizer,
) -> None:
    match = classify(standardizer, "Human [NCBITaxon:9031]")

    assert match.info.taxid == 9031
    assert match.match_quality_score == 1.0
    assert match.needs_review is True
    assert match.diagnostics == (HostDiagnostic.IDENTIFIER_DISAGREEMENT,)


def test_prefixed_taxonomy_identifier_requires_a_recognized_whole_value_shape(
    standardizer: HostStandardizer,
) -> None:
    buried_identifier = classify(
        standardizer,
        "sample NCBITaxon:9031 collected from human",
    )

    assert buried_identifier.info.taxid == 9606
    assert buried_identifier.match_quality_score == 0.5
    assert classify(standardizer, "Chicken [NCBITaxon:999999]") is None


def test_isolation_source_routing_uses_whole_keywords_and_recovery_bypasses_preemption(
    standardizer: HostStandardizer,
) -> None:
    assert HostDiagnostic.ISOLATION_SOURCE_KEYWORD_PREEMPTION.value == (
        "isolation_source_keyword_preemption"
    )
    assert classify(standardizer, "meat") is None
    assert classify(standardizer, "duck meat") is None

    embedded_keyword = classify(standardizer, "seafood and human sample")
    assert embedded_keyword.info.taxid == 9606

    recovered = standardizer.recovery_pass("RECOVER", "food_source", "chicken meat")
    assert recovered.standardized.taxid == 9031
    assert recovered.standardized.scientific_name == "Gallus gallus"
    assert [(pair.attribute, pair.value) for pair in recovered.supporting_pairs] == [
        ("food_source", "chicken meat")
    ]
    assert recovered.from_recovery_pass is True


def test_force_terms_and_value_rejections_preempt_default_matching(
    standardizer: HostStandardizer,
) -> None:
    forced = classify(standardizer, "Squirrel monkey")
    assert forced.info.taxid == 9520
    assert forced.match_quality_score == 1.0
    assert forced.needs_review is False
    assert classify(standardizer, "cancer") is None


def test_multi_attribute_disagreement_tie_breaking_and_review_flags_are_preserved(
    standardizer: HostStandardizer,
) -> None:
    agreeing = standardizer.classify_extracted_record(
        accession="AGREEING",
        attr_str="host_taxid||host",
        val_str="9606||healthy human, adult",
    )
    disagreeing = standardizer.classify_extracted_record(
        accession="DISAGREEING",
        attr_str="host_taxid||host",
        val_str="9606||Bos taurus",
    )
    tied = standardizer.classify_extracted_record(
        accession="TIED",
        attr_str="host||host_taxid",
        val_str="Homo sapiens||9606",
    )

    assert agreeing.info.taxid == 9606
    assert agreeing.attribute == "host_taxid"
    assert agreeing.value == "9606"
    assert agreeing.match_quality_score == 1.0
    assert agreeing.needs_review is False

    assert disagreeing.info.taxid == 9606
    assert disagreeing.match_quality_score == 1.0
    assert disagreeing.needs_review is True
    assert HostDiagnostic.ATTRIBUTE_DISAGREEMENT in disagreeing.diagnostics

    assert tied.info.taxid == 9606
    assert tied.attribute == "host_taxid"
    assert tied.needs_review is False


def test_record_standardization_rejects_misaligned_host_pair_counts(
    standardizer: HostStandardizer,
) -> None:
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


def test_lineage_membership_uses_taxonomy_parents_and_stops_on_broken_cycles(
    tmp_path: Path,
) -> None:
    names = tmp_path / "names.dmp"
    names.write_text("", encoding="utf-8")
    nodes = tmp_path / "nodes.dmp"
    nodes.write_text(
        "1 | 1 | no rank |\n"
        "2759 | 1 | superkingdom |\n"
        "33208 | 2759 | kingdom |\n"
        "9606 | 33208 | species |\n"
        "33090 | 2759 | kingdom |\n"
        "3702 | 33090 | species |\n"
        "4932 | 2759 | species |\n"
        "10 | 11 | no rank |\n"
        "11 | 10 | no rank |\n",
        encoding="utf-8",
    )
    lineage = HostLineageEnricher(names, nodes)

    assert lineage.is_descendant_or_self(33208, 33208)
    assert lineage.is_descendant_or_self(9606, 33208)
    assert lineage.is_descendant_or_self(3702, 33090)
    assert not lineage.is_descendant_or_self(4932, 33208)
    assert not lineage.is_descendant_or_self(999999, 33208)
    assert not lineage.is_descendant_or_self(10, 33208)


def test_record_diagnostics_name_every_review_reason(standardizer: HostStandardizer) -> None:
    scenarios = (
        ("host", "Homo sapiens sample", (HostDiagnostic.SUBSET_MATCH,)),
        (
            "host",
            "human and chicken sample",
            (HostDiagnostic.SUBSET_MATCH, HostDiagnostic.AMBIGUOUS_SUBSET),
        ),
        ("host", "Human [NCBITaxon:9031]", (HostDiagnostic.IDENTIFIER_DISAGREEMENT,)),
        ("host_taxid||host", "9606||Bos taurus", (HostDiagnostic.ATTRIBUTE_DISAGREEMENT,)),
        ("host", "Homo sapiens", ()),
    )

    for attributes, values, expected in scenarios:
        outcome = standardizer.standardize(
            {"accession": "TEST", "host_attr_orig": attributes, "host_val_orig": values}
        )
        assert outcome.record_diagnostics == expected, values
        assert outcome.needs_review is bool(expected), values
        # The build-level values never reach the published field.
        assert HostDiagnostic.MATCHED in outcome.diagnostics, values
