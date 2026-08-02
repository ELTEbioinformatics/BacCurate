"""Verify fixture policies select only compact reference resources."""

from baccurate.standardization.isolation_source import IsolationSourcePromptPolicy
from baccurate.standardization.location import LocationPolicy


def test_fixture_policies_select_fixture_sized_reference_resources(
    fixture_location_policy: LocationPolicy,
    fixture_isolation_source_prompt_policy: IsolationSourcePromptPolicy,
    standardization_fixture_resources,
) -> None:
    assert (
        fixture_location_policy.geo_loc_list_path
        == standardization_fixture_resources.geographic_locations
    )
    assert (
        fixture_isolation_source_prompt_policy.ontology_directory
        == standardization_fixture_resources.isolation_source_ontology_directory
    )
