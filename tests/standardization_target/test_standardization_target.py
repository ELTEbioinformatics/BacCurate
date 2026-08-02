"""Contracts for standardization-target routing metadata."""

import pytest

from baccurate.standardization_target.policy_slot import POLICY_FILENAMES, PolicySlot
from baccurate.standardization_target.specifications import (
    DATASET_COLUMN_ORDER,
    TARGET_SPECS,
    StandardizationTarget,
    required_policy_slots,
    run_policy_slots,
)


def test_every_policy_slot_has_a_configuration_filename() -> None:
    assert set(POLICY_FILENAMES) == set(PolicySlot)
    assert all(isinstance(filename, str) for filename in POLICY_FILENAMES.values())


def test_llm_targets_own_their_published_model_identifier_keys() -> None:
    """These keys are published in the run report and key the per-target LLM metrics."""
    assert {target: spec.model_identifier_key for target, spec in TARGET_SPECS.items()} == {
        StandardizationTarget.HOST: None,
        StandardizationTarget.DATE: None,
        StandardizationTarget.LOCATION: "location",
        StandardizationTarget.ISOLATION_SOURCE: "isolation_source",
    }


def test_standardization_targets_own_their_published_diagnostics_keys() -> None:
    assert {target: spec.published_key for target, spec in TARGET_SPECS.items()} == {
        StandardizationTarget.HOST: "host",
        StandardizationTarget.DATE: "date",
        StandardizationTarget.LOCATION: "location",
        StandardizationTarget.ISOLATION_SOURCE: "isolation_source",
    }


def test_standardization_targets_keep_user_facing_identity_and_order() -> None:
    assert tuple(StandardizationTarget) == (
        StandardizationTarget.HOST,
        StandardizationTarget.DATE,
        StandardizationTarget.LOCATION,
        StandardizationTarget.ISOLATION_SOURCE,
    )
    assert tuple(target.value for target in StandardizationTarget) == (
        "host",
        "date",
        "loc",
        "iso",
    )


@pytest.mark.parametrize(
    ("target", "expected_policy_slots"),
    [
        (StandardizationTarget.HOST, frozenset({PolicySlot.HOST})),
        (StandardizationTarget.DATE, frozenset()),
        (StandardizationTarget.LOCATION, frozenset({PolicySlot.LOCATION})),
        (
            StandardizationTarget.ISOLATION_SOURCE,
            frozenset({PolicySlot.HOST, PolicySlot.ISOLATION_SOURCE}),
        ),
    ],
)
def test_each_standardization_target_declares_its_required_policy_slots(
    target: StandardizationTarget,
    expected_policy_slots: frozenset[PolicySlot],
) -> None:
    assert required_policy_slots((target,)) == expected_policy_slots


def test_extraction_adds_curation_schema_to_required_run_policy_slots() -> None:
    assert run_policy_slots(
        (StandardizationTarget.DATE,),
        extraction_required=True,
    ) == (PolicySlot.CURATION_SCHEMA,)


def test_mixed_target_policy_slots_follow_run_report_order_without_duplicates() -> None:
    assert run_policy_slots(
        (StandardizationTarget.LOCATION, StandardizationTarget.ISOLATION_SOURCE),
        extraction_required=False,
    ) == (
        PolicySlot.HOST,
        PolicySlot.LOCATION,
        PolicySlot.ISOLATION_SOURCE,
    )


def test_dataset_header_keeps_its_target_order_contract() -> None:
    assert DATASET_COLUMN_ORDER == (
        StandardizationTarget.DATE,
        StandardizationTarget.LOCATION,
        StandardizationTarget.ISOLATION_SOURCE,
        StandardizationTarget.HOST,
    )
    assert tuple(
        column for target in DATASET_COLUMN_ORDER for column in TARGET_SPECS[target].output_columns
    ) == (
        "date_attr_orig",
        "date_val_orig",
        "date_start",
        "date_end",
        "date_reliability_score",
        "loc_attr_orig",
        "loc_val_orig",
        "loc_UNregion",
        "loc_country",
        "loc_sublocation",
        "iso_attr_orig",
        "iso_val_orig",
        "iso_source_type",
        "iso_body_product",
        "iso_body_site",
        "iso_lesion",
        "iso_environmental_material",
        "iso_facility",
        "iso_sampled_object",
        "iso_food_type",
        "iso_term_ids",
        "host_attr_orig",
        "host_val_orig",
        "host_taxid",
        "host_sci_name",
        "host_common_names",
        "host_lineage_names",
        "host_lineage_taxids",
        "host_match_quality_score",
        "host_needs_review",
    )
