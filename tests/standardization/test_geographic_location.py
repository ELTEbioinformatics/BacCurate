"""Pin the geographic-location standardization contract."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
import yaml

import baccurate.standardization.location as location_module
from baccurate.standardization.location import (
    LocationDiagnostic,
    LocationOutcome,
    LocationPolicy,
    LocationRejection,
    LocationStandardizer,
    UnresolvedLocationInput,
    normalize_submitted_location_value,
)
from baccurate.standardization.supporting_attribute_value_pair import (
    SupportingAttributeValuePair,
)


def _location_policy(
    tmp_path: Path,
    fixture_policy: LocationPolicy,
    **overrides: object,
) -> LocationPolicy:
    config: dict[str, object] = {
        "schema_version": 2,
        "coordinate_attributes": ["lat_lon"],
        "insdc_country_map": {"United States": "USA", "Vietnam": "Viet Nam"},
        "reviewed_mappings": {},
        "reviewed_unmapped": [],
        "geo_loc_list_path": fixture_policy.geo_loc_list_path.as_posix(),
    }
    config.update(overrides)
    policy_path = tmp_path / f"reviewed-{len(list(tmp_path.glob('reviewed-*.yaml')))}.yaml"
    policy_path.write_text(
        yaml.safe_dump(config, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    return LocationPolicy.load(policy_path)


def _standardize(policy: LocationPolicy, **record: str) -> LocationOutcome | LocationRejection:
    return LocationStandardizer(policy).standardize({"accession": "RECORD", **record})


@pytest.fixture
def standardizer(fixture_location_policy: LocationPolicy):
    return LocationStandardizer(
        replace(
            fixture_location_policy,
            coordinate_attributes=("lat_lon",),
            insdc_country_map={"United States": "USA", "Vietnam": "Viet Nam"},
        )
    )


# =============================================================================
# INSDC normalization
# =============================================================================


@pytest.mark.parametrize(
    ("submitted", "country", "sublocation"),
    [
        pytest.param("United States", "USA", None, id="insdc-remap"),
        pytest.param("Germany", "Germany", None, id="already-insdc"),
        pytest.param("United States: Boston", "USA", "Boston", id="sublocation"),
    ],
)
def test_record_standardization_normalizes_countries_to_insdc(
    standardizer, submitted, country, sublocation
):
    outcome = standardizer.standardize(
        {
            "accession": "INSDC_NORMALIZATION",
            "loc_attr_orig": "geo_loc_name",
            "loc_val_orig": submitted,
        }
    )

    assert isinstance(outcome, LocationOutcome)
    assert outcome.country == country
    assert outcome.sublocation == sublocation
    assert outcome.direct_matches == 1
    assert outcome.diagnostics == (LocationDiagnostic.DIRECT_RESOLUTION,)


def test_record_standardization_rejects_non_insdc_country(standardizer):
    outcome = standardizer.standardize(
        {
            "accession": "INSDC_REJECTION",
            "loc_attr_orig": "geo_loc_name",
            "loc_val_orig": "Vatican",
        }
    )

    assert isinstance(outcome, LocationRejection)
    assert outcome.direct_matches == 1
    assert outcome.diagnostics == (
        LocationDiagnostic.UNMAPPABLE_RESULT,
        LocationDiagnostic.UNRESOLVED_PLACE,
    )
    assert outcome.unresolved_inputs == (UnresolvedLocationInput("geo_loc_name", "Vatican"),)


# =============================================================================
# Record-level deterministic resolution
# =============================================================================


def test_record_standardization_returns_typed_location_with_supporting_pairs_and_diagnostics(
    standardizer,
):
    outcome = standardizer.standardize(
        {
            "accession": "TEST",
            "loc_attr_orig": "geo_loc_name",
            "loc_val_orig": "Germany: Berlin",
        }
    )

    assert outcome == LocationOutcome(
        un_region="Western Europe",
        country="Germany",
        sublocation="Berlin",
        supporting_pairs=(SupportingAttributeValuePair("geo_loc_name", "Germany: Berlin"),),
        direct_matches=1,
        diagnostics=(LocationDiagnostic.DIRECT_RESOLUTION,),
    )


def test_record_standardization_preserves_coordinate_decoding(monkeypatch, standardizer):
    monkeypatch.setattr(
        location_module.reverse_geocode,
        "get",
        lambda _coordinates: {"country": "Germany", "city": "Berlin"},
    )

    outcome = standardizer.standardize(
        {
            "accession": "COORDINATE",
            "loc_attr_orig": "lat_lon",
            "loc_val_orig": "52.52, 13.405",
        }
    )

    assert isinstance(outcome, LocationOutcome)
    assert outcome.country == "Germany"
    assert outcome.sublocation == "Berlin"
    assert outcome.coordinate_decodes == 1
    assert outcome.diagnostics == (LocationDiagnostic.COORDINATE_RESOLUTION,)


# =============================================================================
# Coordinate failures
# =============================================================================


def test_record_standardization_distinguishes_coordinate_service_failure(
    monkeypatch, standardizer, caplog
):
    def fail(_coordinates):
        raise RuntimeError("reverse geocoder unavailable")

    monkeypatch.setattr(location_module.reverse_geocode, "get", fail)

    outcome = standardizer.standardize(
        {
            "accession": "COORDINATE_FAILURE",
            "loc_attr_orig": "lat_lon",
            "loc_val_orig": "48.2, 16.3",
        }
    )

    assert isinstance(outcome, LocationRejection)
    assert outcome.diagnostics == (
        LocationDiagnostic.RECOVERABLE_COORDINATE_FAILURE,
        LocationDiagnostic.UNRESOLVED_PLACE,
    )
    assert caplog.messages == []


def test_record_standardization_retains_coordinate_failure_when_place_name_resolves(
    monkeypatch, standardizer
):
    def fail(_coordinates):
        raise RuntimeError("reverse geocoder unavailable")

    monkeypatch.setattr(location_module.reverse_geocode, "get", fail)

    outcome = standardizer.standardize(
        {
            "accession": "MIXED_COORDINATE_FAILURE",
            "loc_attr_orig": "lat_lon||geo_loc_name",
            "loc_val_orig": "47.5, 19.0||Germany",
        }
    )

    assert isinstance(outcome, LocationOutcome)
    assert outcome.country == "Germany"
    assert outcome.diagnostics == (
        LocationDiagnostic.RECOVERABLE_COORDINATE_FAILURE,
        LocationDiagnostic.DIRECT_RESOLUTION,
        LocationDiagnostic.UNRESOLVED_PLACE,
    )


# =============================================================================
# Absent and unresolved values
# =============================================================================


def test_record_standardization_distinguishes_absent_and_unresolved_values(
    fixture_location_policy,
):
    standardizer = LocationStandardizer(fixture_location_policy)

    absent = standardizer.standardize(
        {"accession": "ABSENT", "loc_attr_orig": "", "loc_val_orig": ""}
    )
    unresolved = standardizer.standardize(
        {
            "accession": "UNRESOLVED",
            "loc_attr_orig": "geo_loc_name",
            "loc_val_orig": "not a resolvable place 764321",
        }
    )

    assert isinstance(absent, LocationRejection)
    assert absent.diagnostics == (LocationDiagnostic.ABSENT_VALUES,)
    assert absent.unresolved_inputs == ()
    assert isinstance(unresolved, LocationRejection)
    assert unresolved.diagnostics == (LocationDiagnostic.UNRESOLVED_PLACE,)
    assert unresolved.unresolved_inputs == (
        UnresolvedLocationInput("geo_loc_name", "not a resolvable place 764321"),
    )


# =============================================================================
# Reviewed geographic-location fallback
# =============================================================================


def test_reviewed_mapping_standardizes_a_country_alias_without_a_sublocation(
    tmp_path, fixture_location_policy
):
    policy = _location_policy(
        tmp_path,
        fixture_location_policy,
        reviewed_mappings={"uae": "United Arab Emirates"},
    )

    outcome = _standardize(policy, loc_attr_orig="geo_loc_name", loc_val_orig="UAE")

    assert isinstance(outcome, LocationOutcome)
    assert outcome.country == "United Arab Emirates"
    assert outcome.sublocation is None
    assert outcome.reviewed_mapping_matches == 1
    assert outcome.unresolved_inputs == ()
    assert outcome.diagnostics == (LocationDiagnostic.REVIEWED_MAPPING_RESOLUTION,)


def test_reviewed_mapping_to_a_water_body_publishes_no_un_region(tmp_path, fixture_location_policy):
    policy = _location_policy(
        tmp_path,
        fixture_location_policy,
        reviewed_mappings={"baltic sea": "Baltic Sea"},
    )

    outcome = _standardize(policy, loc_attr_orig="geo_loc_name", loc_val_orig="Baltic Sea")

    assert isinstance(outcome, LocationOutcome)
    assert outcome.country == "Baltic Sea"
    assert outcome.un_region == "NA"


def test_reviewed_unmapped_value_is_rejected_and_stays_out_of_the_review_worklist(
    tmp_path, fixture_location_policy
):
    policy = _location_policy(tmp_path, fixture_location_policy, reviewed_unmapped=["ncbs"])

    outcome = _standardize(policy, loc_attr_orig="geo_loc_name", loc_val_orig="NCBS")

    assert isinstance(outcome, LocationRejection)
    assert outcome.diagnostics == (LocationDiagnostic.REVIEWED_UNMAPPED,)
    assert outcome.unresolved_inputs == ()


def test_disagreeing_reviewed_mappings_reject_the_record_but_not_its_unreviewed_evidence(
    tmp_path, fixture_location_policy
):
    policy = _location_policy(
        tmp_path,
        fixture_location_policy,
        reviewed_mappings={"uae": "United Arab Emirates", "cambodge": "Cambodia"},
    )

    outcome = _standardize(
        policy,
        loc_attr_orig="geo_loc_name||collection_site||isolation_country",
        loc_val_orig="UAE||Cambodge||unreviewed site 5512",
    )

    assert isinstance(outcome, LocationRejection)
    assert outcome.diagnostics == (
        LocationDiagnostic.REVIEWED_MAPPING_CONFLICT,
        LocationDiagnostic.UNRESOLVED_PLACE,
    )
    assert outcome.unresolved_inputs == (
        UnresolvedLocationInput("isolation_country", "unreviewed site 5512"),
    )


def test_agreeing_reviewed_mappings_are_not_cancelled_by_other_values(
    tmp_path, fixture_location_policy
):
    policy = _location_policy(
        tmp_path,
        fixture_location_policy,
        reviewed_mappings={
            "not provided: cologne": "Germany",
            "not provided, tübingen": "Germany",
        },
        reviewed_unmapped=["water"],
    )

    outcome = _standardize(
        policy,
        loc_attr_orig="geo_loc_name||collection_site||isolation_source",
        loc_val_orig="Not Provided: Cologne||not provided,  Tübingen||water",
    )

    assert isinstance(outcome, LocationOutcome)
    assert outcome.country == "Germany"
    assert outcome.reviewed_mapping_matches == 2
    assert outcome.diagnostics == (LocationDiagnostic.REVIEWED_MAPPING_RESOLUTION,)


def test_deterministic_resolution_beside_an_unreviewed_value_still_reports_it(
    tmp_path, fixture_location_policy
):
    policy = _location_policy(tmp_path, fixture_location_policy)

    outcome = _standardize(
        policy,
        loc_attr_orig="geo_loc_name||collection_site",
        loc_val_orig="Germany||unreviewed site 8841",
    )

    assert isinstance(outcome, LocationOutcome)
    assert outcome.country == "Germany"
    assert outcome.diagnostics == (
        LocationDiagnostic.DIRECT_RESOLUTION,
        LocationDiagnostic.UNRESOLVED_PLACE,
    )
    assert outcome.unresolved_inputs == (
        UnresolvedLocationInput("collection_site", "unreviewed site 8841"),
    )


def test_deterministic_resolution_does_not_report_a_reviewed_unmapped_neighbour(
    tmp_path, fixture_location_policy
):
    """Diagnostics record what the resolution did, not what the reviewed policy contains."""
    policy = _location_policy(tmp_path, fixture_location_policy, reviewed_unmapped=["water"])

    outcome = _standardize(
        policy,
        loc_attr_orig="geo_loc_name||isolation_source",
        loc_val_orig="Germany||water",
    )

    assert isinstance(outcome, LocationOutcome)
    assert outcome.diagnostics == (LocationDiagnostic.DIRECT_RESOLUTION,)
    assert outcome.unresolved_inputs == ()


@pytest.mark.parametrize(
    ("submitted", "normalized"),
    [
        pytest.param("  UAE  ", "uae", id="trimmed"),
        pytest.param("Baltic\tSea", "baltic sea", id="whitespace-run"),
        pytest.param("Not Provided: Cologne", "not provided: cologne", id="lowercased"),
        pytest.param("Perú", "perú", id="accents-kept"),
    ],
)
def test_reviewed_keys_are_normalized_without_dropping_punctuation_or_accents(
    submitted, normalized
):
    assert normalize_submitted_location_value(submitted) == normalized


def test_reviewed_matching_is_whole_key_equality_only(tmp_path, fixture_location_policy):
    policy = _location_policy(
        tmp_path,
        fixture_location_policy,
        reviewed_mappings={"uae": "United Arab Emirates"},
    )

    outcome = _standardize(policy, loc_attr_orig="geo_loc_name", loc_val_orig="UAE hospital 4471")

    assert isinstance(outcome, LocationRejection)
    assert outcome.diagnostics == (LocationDiagnostic.UNRESOLVED_PLACE,)
