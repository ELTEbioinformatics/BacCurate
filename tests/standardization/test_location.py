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
    LocationResolutionRoute,
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
            insdc_country_map={
                "United States": "USA",
                "Vietnam": "Viet Nam",
                "Brunei Darussalam": "Brunei",
            },
        )
    )


# =============================================================================
# INSDC normalization
# =============================================================================


@pytest.mark.parametrize(
    ("submitted", "country", "sublocation", "route"),
    [
        pytest.param(
            "United States",
            "USA",
            None,
            LocationResolutionRoute.INSDC_TERM,
            id="insdc-remap",
        ),
        pytest.param(
            "Germany",
            "Germany",
            None,
            LocationResolutionRoute.INSDC_TERM,
            id="already-insdc",
        ),
        pytest.param(
            "United States: Boston",
            "USA",
            "Boston",
            LocationResolutionRoute.INSDC_TERM,
            id="sublocation",
        ),
    ],
)
def test_country_normalized_to_insdc(standardizer, submitted, country, sublocation, route):
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
    assert outcome.route == route
    assert outcome.diagnostics == ()


def test_non_insdc_country_rejected(standardizer):
    outcome = standardizer.standardize(
        {
            "accession": "INSDC_REJECTION",
            "loc_attr_orig": "geo_loc_name",
            "loc_val_orig": "Vatican",
        }
    )

    assert isinstance(outcome, LocationRejection)
    assert outcome.country_conversion_matches == 1
    assert outcome.diagnostics == (
        LocationDiagnostic.UNMAPPABLE_RESULT,
        LocationDiagnostic.UNRESOLVED_PLACE,
    )
    assert outcome.unresolved_inputs == (UnresolvedLocationInput("geo_loc_name", "Vatican"),)


# =============================================================================
# Record-level deterministic resolution
# =============================================================================


def test_outcome_carries_supporting_pairs_and_diagnostics(
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
        selected_pair_positions=(1,),
        route=LocationResolutionRoute.INSDC_TERM,
        insdc_term_matches=1,
    )


def test_selected_pair_position_names_the_pair_that_produced_the_country(standardizer):
    outcome = standardizer.standardize(
        {
            "accession": "SECOND_PAIR",
            "loc_attr_orig": "collection_site||geo_loc_name",
            "loc_val_orig": "unreviewed site 8841||Germany",
        }
    )

    assert isinstance(outcome, LocationOutcome)
    assert outcome.selected_pair_positions == (2,)


def test_selected_pair_position_counts_a_published_pair_with_an_empty_value(standardizer):
    outcome = standardizer.standardize(
        {
            "accession": "EMPTY_VALUE",
            "loc_attr_orig": "collection_site||geo_loc_name",
            "loc_val_orig": "||Germany",
        }
    )

    assert isinstance(outcome, LocationOutcome)
    assert outcome.selected_pair_positions == (2,)


def test_selected_pair_position_follows_the_sublocation_preference(standardizer):
    outcome = standardizer.standardize(
        {
            "accession": "SUBLOCATION_PREFERENCE",
            "loc_attr_orig": "geo_loc_name||collection_site",
            "loc_val_orig": "Germany||Germany: Berlin",
        }
    )

    assert isinstance(outcome, LocationOutcome)
    assert outcome.sublocation == "Berlin"
    assert outcome.selected_pair_positions == (2,)


def test_coordinate_decodes_to_country_and_city(monkeypatch, standardizer):
    monkeypatch.setattr(
        location_module.reverse_geocode,
        "get",
        lambda _coordinates: {"country_code": "DE", "city": "Berlin"},
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
    assert outcome.route == LocationResolutionRoute.COORDINATE
    assert outcome.coordinate == (52.52, 13.405)
    assert outcome.diagnostics == ()


# =============================================================================
# Coordinate resolution by map-unit containment
# =============================================================================


@pytest.mark.parametrize(
    ("submitted", "country"),
    [
        # King George Island. The nearest city is on the Falkland Islands, 1,200 km away.
        pytest.param("62.13 S 58.57 W", "Antarctica", id="antarctic-island"),
        # Brunei. The nearest city lies across the border, in Malaysia.
        pytest.param("4.33764957 N 114.44534163 E", "Brunei", id="enclaved-country"),
    ],
)
def test_coordinate_names_the_map_unit_that_contains_it(standardizer, submitted, country):
    outcome = standardizer.standardize(
        {"accession": "CONTAINMENT", "loc_attr_orig": "lat_lon", "loc_val_orig": submitted}
    )

    assert isinstance(outcome, LocationOutcome)
    assert outcome.country == country
    assert outcome.route == LocationResolutionRoute.COORDINATE


@pytest.mark.parametrize(
    ("submitted", "coordinate", "reason"),
    [
        pytest.param(
            "0.0, 0.0",
            (0.0, 0.0),
            "no map unit contains a point in the Gulf of Guinea",
            id="water",
        ),
        pytest.param(
            "9.56 N 44.06 E",
            (9.56, 44.06),
            "Somaliland carries no ISO code",
            id="map-unit-without-code",
        ),
    ],
)
def test_coordinate_outside_a_coded_map_unit_names_no_country(
    standardizer, submitted, coordinate, reason
):
    outcome = standardizer.standardize(
        {"accession": "NO_COUNTRY", "loc_attr_orig": "lat_lon", "loc_val_orig": submitted}
    )

    assert isinstance(outcome, LocationRejection), reason
    assert outcome.coordinate_decodes == 0
    assert outcome.unresolved_inputs == (UnresolvedLocationInput("lat_lon", submitted),)
    assert LocationDiagnostic.COORDINATE_WITHOUT_COUNTRY in outcome.diagnostics
    assert outcome.coordinate == pytest.approx(coordinate)


def test_a_city_outside_the_resolved_country_is_no_sublocation(standardizer):
    # A point in Italy near the border of Vatican City. The nearest city, Vatican City, is its
    # own country, so it must not become an italian sublocation.
    outcome = standardizer.standardize(
        {"accession": "BORDER", "loc_attr_orig": "lat_lon", "loc_val_orig": "41.9 N 12.46 E"}
    )

    assert isinstance(outcome, LocationOutcome)
    assert outcome.country == "Italy"
    assert outcome.sublocation is None


@pytest.mark.parametrize(
    ("submitted", "parsed", "submitted_text", "country", "sublocation"),
    [
        pytest.param("0.0, 0.0", (0.0, 0.0), "Germany", "Germany", None, id="gulf-of-guinea"),
        # Offshore antarctic coordinate.
        pytest.param(
            "65.9 S 110.0 E",
            (-65.9, 110.0),
            "Antarctica: Warriner Island",
            "Antarctica",
            "Warriner Island",
            id="offshore-antarctica",
        ),
    ],
)
def test_coordinate_that_names_no_country_leaves_the_record_to_its_text(
    standardizer, submitted, parsed, submitted_text, country, sublocation
):
    outcome = standardizer.standardize(
        {
            "accession": "NO_COUNTRY_AND_TEXT",
            "loc_attr_orig": "lat_lon||geo_loc_name",
            "loc_val_orig": f"{submitted}||{submitted_text}",
        }
    )

    assert isinstance(outcome, LocationOutcome)
    assert (outcome.country, outcome.sublocation) == (country, sublocation)
    assert outcome.route == LocationResolutionRoute.INSDC_TERM
    assert LocationDiagnostic.COORDINATE_WITHOUT_COUNTRY in outcome.diagnostics
    assert outcome.coordinate == pytest.approx(parsed)


def test_the_first_parsed_coordinate_is_published_when_no_coordinate_resolved(standardizer):
    outcome = standardizer.standardize(
        {
            "accession": "TWO_COORDINATES",
            "loc_attr_orig": "lat_lon||lat_lon||geo_loc_name",
            "loc_val_orig": "0.0, 0.0||9.56 N 44.06 E||Germany",
        }
    )

    assert isinstance(outcome, LocationOutcome)
    assert outcome.route == LocationResolutionRoute.INSDC_TERM
    assert outcome.coordinate == pytest.approx((0.0, 0.0))


def test_a_coordinate_country_outranks_submitted_text(standardizer):
    # Coordinate is Belgium, next to a Dutch place name.
    # The coordinate wins even though the text carries a sublocation.
    outcome = standardizer.standardize(
        {
            "accession": "ROUTE_ORDER",
            "loc_attr_orig": "geo_loc_name||lat_lon",
            "loc_val_orig": "Netherlands: Breda||51.34 N 4.48 E",
        }
    )

    assert isinstance(outcome, LocationOutcome)
    assert outcome.country == "Belgium"
    assert outcome.route == LocationResolutionRoute.COORDINATE
    assert outcome.selected_pair_positions == (2,)
    assert LocationDiagnostic.COUNTRY_CONFLICT in outcome.diagnostics


# =============================================================================
# Coordinate parsing - full-value patterns
# =============================================================================


@pytest.mark.parametrize(
    ("submitted", "coordinates"),
    [
        pytest.param("51.9194 N 19.1451 E", (51.9194, 19.1451), id="hemisphere-decimal"),
        pytest.param("6°12'52\"S_106°50'42\"E", (-6.2144444, 106.845), id="sexagesimal"),
        pytest.param("47°59.754′ N 7°51.1332′ E", (47.9959, 7.85222), id="degrees-and-minutes"),  # noqa: RUF001 - these marks are the submitted characters
        pytest.param("(-34.6037; -58.3816)", (-34.6037, -58.3816), id="parenthesized-decimal"),
        pytest.param("?41.4808_2.23782", (41.4808, 2.23782), id="corrupted-leading-mark"),
        pytest.param("6°12'52\"_106°50'42\"", None, id="sexagesimal-without-hemisphere"),
        pytest.param("Latitude: 47°59.754′ N Longitude: 7°51.1332′ E", None, id="labelled"),  # noqa: RUF001 - these marks are the submitted characters
        pytest.param("51.9194N19.1451E", None, id="components-without-a-separator"),
        pytest.param("Permoserstrasse 15, 04318 Leipzig", None, id="postal-address"),
        pytest.param("523702_48952", None, id="out-of-range-pair"),
        pytest.param("2° 30′ 55″ sud _ 28° 50′ 42″ est", None, id="non-english-hemisphere"),  # noqa: RUF001 - these marks are the submitted characters
    ],
)
def test_coordinate_parsed_only_on_full_value_match(
    monkeypatch, standardizer, submitted, coordinates
):
    decoded: list[tuple[float, float]] = []

    def record(coordinate_pair):
        decoded.append(coordinate_pair)
        return {"country_code": "DE", "city": "Berlin"}

    monkeypatch.setattr(location_module.reverse_geocode, "get", record)

    standardizer.standardize(
        {"accession": "COORDINATE_GRAMMAR", "loc_attr_orig": "lat_lon", "loc_val_orig": submitted}
    )

    if coordinates is None:
        assert decoded == []
    else:
        assert decoded == [pytest.approx(coordinates)]


def test_text_on_coordinate_attribute_resolves(standardizer):
    outcome = standardizer.standardize(
        {
            "accession": "TEXT_ON_A_COORDINATE_ATTRIBUTE",
            "loc_attr_orig": "lat_lon||lat_lon",
            "loc_val_orig": "Germany:Berlin||523702_48952",
        }
    )

    assert isinstance(outcome, LocationOutcome)
    assert (outcome.country, outcome.sublocation) == ("Germany", "Berlin")
    # A value with no letters gets no textual reading, so only its numbers could have matched.
    assert outcome.unresolved_inputs == (UnresolvedLocationInput("lat_lon", "523702_48952"),)


# =============================================================================
# INSDC matching and country lookup on the text before the first delimiter
# =============================================================================


@pytest.mark.parametrize(
    ("submitted", "country", "sublocation"),
    [
        pytest.param("Gaza Strip", "Gaza Strip", None, id="insdc-verbatim"),
        pytest.param(
            "Atlantic Ocean: CORK U1383C-Deep at North Pond",
            "Atlantic Ocean",
            "CORK U1383C-Deep at North Pond",
            id="insdc-before-the-first-colon",
        ),
        pytest.param("Germany, CO213, Tübingen", "Germany", None, id="first-segment-only"),
        pytest.param("Germany, , Tübingen", "Germany", None, id="comma-is-not-a-sublocation"),
    ],
)
def test_insdc_match_before_country_lookup(standardizer, submitted, country, sublocation):
    outcome = standardizer.standardize(
        {"accession": "INSDC_ROUTES", "loc_attr_orig": "geo_loc_name", "loc_val_orig": submitted}
    )

    assert isinstance(outcome, LocationOutcome)
    assert (outcome.country, outcome.sublocation) == (country, sublocation)


def test_a_later_segment_is_never_a_country_lookup_key(standardizer):
    """
    `KY` matches the ISO alpha-2 code for the Cayman Islands.
    Only the text before the first delimiter is used as a country lookup key.
    """
    outcome = standardizer.standardize(
        {
            "accession": "LATE_SEGMENT",
            "loc_attr_orig": "geo_loc_name",
            "loc_val_orig": "Morehead, KY",
        }
    )

    assert isinstance(outcome, LocationRejection)
    assert outcome.unresolved_inputs == (UnresolvedLocationInput("geo_loc_name", "Morehead, KY"),)


# =============================================================================
# Selection after INSDC country mapping
# =============================================================================


def test_unmappable_coordinate_country_does_not_contribute_sublocation(standardizer):
    """The Vatican map unit converts to a name outside the INSDC location vocabulary."""
    outcome = standardizer.standardize(
        {
            "accession": "UNMAPPABLE_COORDINATE",
            "loc_attr_orig": "lat_lon||geo_loc_name",
            "loc_val_orig": "41.9038/12.4536||Italy",
        }
    )

    assert isinstance(outcome, LocationOutcome)
    assert (outcome.country, outcome.sublocation) == ("Italy", None)
    assert outcome.route == LocationResolutionRoute.INSDC_TERM
    assert outcome.diagnostics == (
        LocationDiagnostic.UNMAPPABLE_RESULT,
        LocationDiagnostic.UNRESOLVED_PLACE,
    )
    assert outcome.coordinate_decodes == 1


def test_conflicting_countries_pick_coordinate_and_flag_conflict(monkeypatch, standardizer):
    monkeypatch.setattr(
        location_module.reverse_geocode,
        "get",
        lambda _coordinates: {"country_code": "IT", "city": "Rome"},
    )

    outcome = standardizer.standardize(
        {
            "accession": "CONTRADICTION",
            "loc_attr_orig": "lat_lon||geo_loc_name",
            "loc_val_orig": "41.90/12.49||Germany",
        }
    )

    assert isinstance(outcome, LocationOutcome)
    assert (outcome.country, outcome.sublocation) == ("Italy", "Rome")
    assert outcome.route == LocationResolutionRoute.COORDINATE
    assert outcome.diagnostics == (LocationDiagnostic.COUNTRY_CONFLICT,)


# =============================================================================
# Coordinate failures
# =============================================================================


def test_coordinate_service_failure_costs_only_the_sublocation(monkeypatch, standardizer, caplog):

    def fail(_coordinates):
        raise RuntimeError("reverse geocoder unavailable")

    monkeypatch.setattr(location_module.reverse_geocode, "get", fail)

    outcome = standardizer.standardize(
        {
            "accession": "COORDINATE_FAILURE",
            "loc_attr_orig": "lat_lon",
            "loc_val_orig": "52.52, 13.405",
        }
    )

    assert isinstance(outcome, LocationOutcome)
    assert (outcome.country, outcome.sublocation) == ("Germany", None)
    assert outcome.route == LocationResolutionRoute.COORDINATE
    assert outcome.diagnostics == (LocationDiagnostic.RECOVERABLE_COORDINATE_FAILURE,)
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
    assert outcome.route == LocationResolutionRoute.INSDC_TERM
    assert outcome.diagnostics == (
        LocationDiagnostic.RECOVERABLE_COORDINATE_FAILURE,
        LocationDiagnostic.UNMAPPABLE_RESULT,
        LocationDiagnostic.UNRESOLVED_PLACE,
    )


# =============================================================================
# Absent and unresolved values
# =============================================================================


def test_absent_vs_unresolved_values(
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
    assert outcome.route == LocationResolutionRoute.REVIEWED_MAPPING
    assert outcome.diagnostics == ()


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
    assert outcome.supporting_pairs == (
        SupportingAttributeValuePair("geo_loc_name", "UAE"),
        SupportingAttributeValuePair("collection_site", "Cambodge"),
        SupportingAttributeValuePair("isolation_country", "unreviewed site 5512"),
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
        loc_attr_orig="geo_loc_name||isolation_source||collection_site",
        loc_val_orig="Not Provided: Cologne||water||not provided,  Tübingen",
    )

    assert isinstance(outcome, LocationOutcome)
    assert outcome.country == "Germany"
    assert outcome.reviewed_mapping_matches == 2
    assert outcome.route == LocationResolutionRoute.REVIEWED_MAPPING
    assert outcome.selected_pair_positions == (1, 3)
    assert outcome.diagnostics == ()


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
    assert outcome.route == LocationResolutionRoute.INSDC_TERM
    assert outcome.diagnostics == (LocationDiagnostic.UNRESOLVED_PLACE,)
    assert outcome.unresolved_inputs == (
        UnresolvedLocationInput("collection_site", "unreviewed site 8841"),
    )


def test_deterministic_resolution_does_not_report_a_reviewed_unmapped_neighbour(
    tmp_path, fixture_location_policy
):
    """Diagnostics record what the record needed, not what the reviewed policy contains."""
    policy = _location_policy(tmp_path, fixture_location_policy, reviewed_unmapped=["water"])

    outcome = _standardize(
        policy,
        loc_attr_orig="geo_loc_name||isolation_source",
        loc_val_orig="Germany||water",
    )

    assert isinstance(outcome, LocationOutcome)
    assert outcome.route == LocationResolutionRoute.INSDC_TERM
    assert outcome.diagnostics == ()
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
def test_reviewed_key_normalization(submitted, normalized):
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
