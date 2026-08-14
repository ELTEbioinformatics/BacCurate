"""Pin the geographic-location standardization contract."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import httpx
import openai
import pytest
import yaml

import baccurate.standardization.location as location_module
from baccurate.adapters.llm.client import LLMSettings
from baccurate.standardization.location import (
    LocationDiagnostic,
    LocationOutcome,
    LocationPolicy,
    LocationRejection,
    LocationStandardizer,
)
from baccurate.standardization.supporting_attribute_value_pair import (
    SupportingAttributeValuePair,
)


def _location_policy(
    tmp_path: Path,
    fixture_policy: LocationPolicy,
    *,
    overrides: dict[str, object] | None = None,
) -> LocationPolicy:
    policy_path = tmp_path / f"location-{len(list(tmp_path.glob('location-*.yaml')))}.yaml"
    config = {
        "schema_version": 1,
        "prompt_version": "synthetic-v1",
        "coordinate_attributes": ["synthetic_coordinate"],
        "llm_system_prompt": "Resolve one standardized country.",
        "llm_user_prompt_template": "Synthetic evidence:\n{attr_val_pairs}\n",
        "insdc_country_map": {"United States": "USA"},
        "geo_loc_list_path": fixture_policy.geo_loc_list_path.as_posix(),
        "cache_db_path": (tmp_path / "location-cache.db").as_posix(),
    }
    config.update(overrides or {})
    policy_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return LocationPolicy.load(policy_path)


def _location_client(calls: list[dict], *, country: str = "Germany") -> SimpleNamespace:
    def create(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(
            choices=[
                SimpleNamespace(message=SimpleNamespace(content=f'{{"country": "{country}"}}'))
            ]
        )

    return SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create)),
        close=lambda: None,
    )


def _standardize_model_location(
    policy: LocationPolicy,
    calls: list[dict],
    *,
    model: str = "test-model",
    country: str = "Germany",
) -> LocationOutcome | LocationRejection:
    standardizer = LocationStandardizer(
        policy,
        client=_location_client(calls, country=country),
        llm_settings=LLMSettings(None, None, model),
    )
    try:
        return standardizer.standardize(
            {
                "accession": "REQUEST_FINGERPRINT",
                "loc_attr_orig": "geo_loc_name",
                "loc_val_orig": "model-only place 739105",
            }
        )
    finally:
        standardizer.close()


@pytest.fixture
def standardizer(fixture_location_policy: LocationPolicy):
    policy = replace(
        fixture_location_policy,
        coordinate_attributes=("lat_lon",),
        insdc_country_map={"United States": "USA", "Vietnam": "Viet Nam"},
    )
    instance = LocationStandardizer(policy, client=None)
    try:
        yield instance
    finally:
        instance.close()


# =============================================================================
# INSDC normalization
# =============================================================================


@pytest.mark.parametrize(
    ("submitted", "country", "sublocation"),
    [
        pytest.param("United States", "USA", None, id="insdc-remap"),
        pytest.param("Vietnam", "Viet Nam", None, id="country-alias"),
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
    assert outcome.diagnostics == (LocationDiagnostic.UNMAPPABLE_RESULT,)


def test_record_standardization_rejects_known_false_positive_model_country(
    tmp_path, fixture_location_policy
):
    calls: list[dict] = []
    policy = _location_policy(tmp_path, fixture_location_policy)

    outcome = _standardize_model_location(policy, calls, country="water")

    assert isinstance(outcome, LocationRejection)
    assert outcome.llm_calls == 1
    assert outcome.diagnostics == (LocationDiagnostic.UNMAPPABLE_RESULT,)


# =============================================================================
# Record-level resolution
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
    assert outcome.diagnostics == (
        LocationDiagnostic.RECOVERABLE_COORDINATE_FAILURE,
        LocationDiagnostic.DIRECT_RESOLUTION,
    )


def test_record_standardization_retains_later_coordinate_failure_after_unresolved_value(
    monkeypatch, standardizer
):
    def fail(_coordinates):
        raise RuntimeError("reverse geocoder unavailable")

    monkeypatch.setattr(location_module.reverse_geocode, "get", fail)

    outcome = standardizer.standardize(
        {
            "accession": "REJECTION_ONLY_COORDINATE_FAILURE",
            "loc_attr_orig": "lat_lon||lat_lon",
            "loc_val_orig": "999, 999||46.0, 18.0",
        }
    )

    assert isinstance(outcome, LocationRejection)
    assert outcome.diagnostics == (
        LocationDiagnostic.RECOVERABLE_COORDINATE_FAILURE,
        LocationDiagnostic.UNRESOLVED_PLACE,
    )


# =============================================================================
# Absent, unresolved, and disabled values
# =============================================================================


def test_record_standardization_distinguishes_absent_unresolved_and_disabled_values(
    monkeypatch, fixture_location_policy
):
    standardizer = LocationStandardizer(fixture_location_policy, client=None)
    monkeypatch.setattr(standardizer.cache, "get", lambda _context: None)
    monkeypatch.setattr(standardizer.cache, "set", lambda *_args: None)
    try:
        absent = standardizer.standardize(
            {"accession": "ABSENT", "loc_attr_orig": "", "loc_val_orig": ""}
        )
        disabled = standardizer.standardize(
            {
                "accession": "DISABLED",
                "loc_attr_orig": "geo_loc_name",
                "loc_val_orig": "not a resolvable place 764321",
            }
        )
        unresolved = standardizer.standardize(
            {
                "accession": "UNRESOLVED",
                "loc_attr_orig": "lat_lon",
                "loc_val_orig": "999, 999",
            }
        )
    finally:
        standardizer.close()

    assert isinstance(absent, LocationRejection)
    assert absent.diagnostics == (LocationDiagnostic.ABSENT_VALUES,)
    assert isinstance(disabled, LocationRejection)
    assert disabled.diagnostics == (LocationDiagnostic.LLM_DISABLED,)
    assert disabled.llm_calls == 0
    assert isinstance(unresolved, LocationRejection)
    assert unresolved.diagnostics == (LocationDiagnostic.UNRESOLVED_PLACE,)


def test_record_standardization_counts_unmappable_direct_result_without_logging(
    fixture_location_policy, caplog
):
    standardizer = LocationStandardizer(fixture_location_policy, client=None)
    try:
        outcome = standardizer.standardize(
            {
                "accession": "UNMAPPABLE",
                "loc_attr_orig": "geo_loc_name",
                "loc_val_orig": "Vatican",
            }
        )
    finally:
        standardizer.close()

    assert isinstance(outcome, LocationRejection)
    assert outcome.diagnostics == (LocationDiagnostic.UNMAPPABLE_RESULT,)
    assert caplog.messages == []


# =============================================================================
# Model resolution and failures
# =============================================================================


@pytest.mark.parametrize(
    ("content", "expected", "writes_cache"),
    [
        (
            '{"country": "Germany"}',
            LocationDiagnostic.LLM_RESOLUTION,
            True,
        ),
        ("not json", LocationDiagnostic.INVALID_LLM_RESPONSE, False),
        ("{}", LocationDiagnostic.INVALID_LLM_RESPONSE, False),
        ('{"country": ["Germany"]}', LocationDiagnostic.INVALID_LLM_RESPONSE, False),
    ],
)
def test_record_standardization_distinguishes_model_resolution_and_invalid_response(
    monkeypatch,
    fixture_location_policy,
    content,
    expected,
    writes_cache,
    caplog,
):
    response = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])
    client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=lambda **_kwargs: response),
        ),
        close=lambda: None,
    )
    standardizer = LocationStandardizer(
        fixture_location_policy,
        client=client,
        llm_settings=LLMSettings(None, None, "test-model"),
    )
    monkeypatch.setattr(standardizer.cache, "get", lambda _context: None)
    cache_writes = []
    monkeypatch.setattr(standardizer.cache, "set", lambda *_args: cache_writes.append(_args))
    try:
        outcome = standardizer.standardize(
            {
                "accession": "MODEL",
                "loc_attr_orig": "geo_loc_name",
                "loc_val_orig": "model-only place 918273",
            }
        )
    finally:
        standardizer.close()

    assert outcome.diagnostics == (expected,)
    assert outcome.llm_calls == 1
    assert bool(cache_writes) is writes_cache
    if writes_cache:
        fingerprint, country = cache_writes[0]
        assert len(fingerprint) == 64
        assert country == "Germany"
    assert caplog.messages == []


def test_record_standardization_distinguishes_cache_resolution(
    monkeypatch, fixture_location_policy
):
    standardizer = LocationStandardizer(fixture_location_policy, client=None)
    monkeypatch.setattr(standardizer.cache, "get", lambda _context: "Germany")
    try:
        outcome = standardizer.standardize(
            {
                "accession": "CACHE",
                "loc_attr_orig": "geo_loc_name",
                "loc_val_orig": "cached place 192837",
            }
        )
    finally:
        standardizer.close()

    assert outcome.diagnostics == (LocationDiagnostic.CACHE_RESOLUTION,)


def test_mapped_model_country_is_identical_on_first_call_and_on_cache_hit(
    tmp_path, fixture_location_policy
):
    """A standardized country must not depend on whether the model answer was cached."""
    calls: list[dict] = []
    policy = _location_policy(tmp_path, fixture_location_policy)

    first = _standardize_model_location(policy, calls, country="United States")
    second = _standardize_model_location(policy, calls, country="United States")

    assert len(calls) == 1
    assert second.diagnostics == (LocationDiagnostic.CACHE_RESOLUTION,)
    assert first.country == "USA"
    assert (second.country, second.un_region) == (
        first.country,
        first.un_region,
    )


def test_cached_country_outside_insdc_vocabulary_is_rejected(
    monkeypatch, tmp_path, fixture_location_policy
):
    """The INSDC vocabulary bounds cached model answers as well as fresh ones."""
    standardizer = LocationStandardizer(
        _location_policy(tmp_path, fixture_location_policy), client=None
    )
    monkeypatch.setattr(standardizer.cache, "get", lambda _fingerprint: "Vatican")
    try:
        outcome = standardizer.standardize(
            {
                "accession": "CACHE_NON_INSDC",
                "loc_attr_orig": "geo_loc_name",
                "loc_val_orig": "cached place 445566",
            }
        )
    finally:
        standardizer.close()

    assert isinstance(outcome, LocationRejection)
    assert outcome.diagnostics == (LocationDiagnostic.UNMAPPABLE_RESULT,)
    assert outcome.cache_hits == 1


def test_location_cache_reuses_identical_request_when_only_prompt_metadata_changes(
    tmp_path, fixture_location_policy
):
    calls: list[dict] = []
    first_policy = _location_policy(
        tmp_path, fixture_location_policy, overrides={"prompt_version": "first"}
    )
    second_policy = _location_policy(
        tmp_path, fixture_location_policy, overrides={"prompt_version": "second"}
    )

    first = _standardize_model_location(first_policy, calls)
    second = _standardize_model_location(second_policy, calls)

    assert first.diagnostics == (LocationDiagnostic.LLM_RESOLUTION,)
    assert second.diagnostics == (LocationDiagnostic.CACHE_RESOLUTION,)
    assert len(calls) == 1


def test_location_request_uses_rendered_synthetic_prompts(
    tmp_path: Path, fixture_location_policy: LocationPolicy
) -> None:
    policy = _location_policy(tmp_path, fixture_location_policy)
    calls: list[dict] = []
    standardizer = LocationStandardizer(
        policy,
        client=_location_client(calls),
        llm_settings=LLMSettings(None, None, "test-model"),
    )
    try:
        direct = standardizer.standardize(
            {
                "accession": "DIRECT",
                "loc_attr_orig": "geo_loc_name",
                "loc_val_orig": "Germany: Berlin",
            }
        )
        assert calls == []
        modelled = standardizer.standardize(
            {
                "accession": "REQUEST_FINGERPRINT",
                "loc_attr_orig": "geo_loc_name",
                "loc_val_orig": "model-only place 739105",
            }
        )
    finally:
        standardizer.close()

    assert isinstance(direct, LocationOutcome)
    assert isinstance(modelled, LocationOutcome)
    assert [message["role"] for message in calls[0]["messages"]] == ["system", "user"]
    assert "Resolve one standardized country" in calls[0]["messages"][0]["content"]
    assert "geo_loc_name=model-only place 739105" in calls[0]["messages"][1]["content"]
    assert calls[0]["model"] == "test-model"
    assert calls[0]["temperature"] == 0
    assert calls[0]["seed"] == 100
    assert calls[0]["response_format"] == {"type": "json_object"}


@pytest.mark.parametrize("changed_component", ["message", "model", "parameter", "schema"])
def test_location_cache_misses_when_canonical_request_changes(
    tmp_path, monkeypatch, fixture_location_policy, changed_component
):
    calls: list[dict] = []
    first_policy = _location_policy(tmp_path, fixture_location_policy)
    _standardize_model_location(first_policy, calls)

    second_policy = _location_policy(tmp_path, fixture_location_policy)
    second_model = "test-model"
    if changed_component == "message":
        second_policy = _location_policy(
            tmp_path,
            fixture_location_policy,
            overrides={"llm_system_prompt": "A changed fully rendered system prompt."},
        )
    elif changed_component == "model":
        second_model = "changed-model"
    elif changed_component == "parameter":
        monkeypatch.setattr(
            location_module,
            "LOCATION_LLM_PARAMETERS",
            {"temperature": 1, "seed": 100},
        )
    else:
        monkeypatch.setattr(
            location_module,
            "LOCATION_RESPONSE_SCHEMA_ID",
            "baccurate.location.country.changed",
        )

    second = _standardize_model_location(second_policy, calls, model=second_model)

    assert second.diagnostics == (LocationDiagnostic.LLM_RESOLUTION,)
    assert len(calls) == 2


@pytest.mark.parametrize(
    "failure",
    [
        pytest.param(
            openai.APITimeoutError(request=httpx.Request("POST", "https://example.test")),
            id="timeout",
        ),
        pytest.param(
            openai.APIError(
                "service unavailable",
                request=httpx.Request("POST", "https://example.test"),
                body=None,
            ),
            id="api-error",
        ),
    ],
)
def test_recoverable_model_failure_can_be_retried(monkeypatch, fixture_location_policy, failure):
    successful_response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content='{"country": "Germany"}'))]
    )
    responses = iter(
        [
            failure,
            successful_response,
        ]
    )
    calls = 0

    def complete(**_kwargs):
        nonlocal calls
        calls += 1
        response = next(responses)
        if isinstance(response, Exception):
            raise response
        return response

    client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=complete)),
        close=lambda: None,
    )
    standardizer = LocationStandardizer(
        fixture_location_policy,
        client=client,
        llm_settings=LLMSettings(None, None, "test-model"),
    )
    monkeypatch.setattr(standardizer.cache, "get", lambda _context: None)
    cached_results = []
    monkeypatch.setattr(standardizer.cache, "set", lambda *_args: cached_results.append(_args))
    record = {
        "accession": "RETRY",
        "loc_attr_orig": "geo_loc_name",
        "loc_val_orig": "model-only place 24681357",
    }
    try:
        first = standardizer.standardize(record)
        second = standardizer.standardize(record)
    finally:
        standardizer.close()

    assert first.diagnostics == (LocationDiagnostic.RECOVERABLE_LLM_FAILURE,)
    assert first.llm_calls == 1
    assert second.diagnostics == (LocationDiagnostic.LLM_RESOLUTION,)
    assert second.llm_calls == 1
    assert calls == 2
    assert len(cached_results) == 1
    fingerprint, country = cached_results[0]
    assert len(fingerprint) == 64
    assert country == "Germany"
