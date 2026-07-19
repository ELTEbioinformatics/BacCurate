"""Tests for geographical location standardization.
Uses a monkeypatch instead of calling an actual LLM API.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import httpx
import openai
import pytest

import baccurate.standardizers.location as location_module
from baccurate.llm.client import LLMSettings
from baccurate.standardizers.location import (
    LocationDiagnostic,
    LocationMatch,
    LocationOrigin,
    LocationOutcome,
    LocationRejection,
    LocationStandardizer,
)

ROOT = Path(__file__).parent.parent
CONFIG_PATH = ROOT / "config" / "location.yaml"


def _location_config(tmp_path: Path, *, suffix: str = "") -> Path:
    config_path = tmp_path / f"location-{len(list(tmp_path.glob('location-*.yaml')))}.yaml"
    config_path.write_text(
        CONFIG_PATH.read_text(encoding="utf-8")
        + f'\ncache_db_path: "{(tmp_path / "location-cache.db").as_posix()}"\n'
        + suffix,
        encoding="utf-8",
    )
    return config_path


def _location_client(calls: list[dict]) -> SimpleNamespace:
    def create(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content='{"country": "Germany"}'))]
        )

    return SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create)),
        close=lambda: None,
    )


def _standardize_model_location(
    config_path: Path,
    calls: list[dict],
    *,
    model: str = "test-model",
) -> LocationOutcome | LocationRejection:
    standardizer = LocationStandardizer(
        config_path,
        client=_location_client(calls),
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


@pytest.fixture(scope="session")
def standardizer() -> LocationStandardizer:
    return LocationStandardizer(CONFIG_PATH)


# =============================================================================
# INSDC remapping
# =============================================================================


def test_na_country_is_left_untouched(standardizer):
    result = standardizer._to_insdc(LocationMatch("NA", "NA", None))

    assert result.country == "NA"


def test_non_insdc_country_is_dropped_to_na(standardizer):
    result = standardizer._to_insdc(LocationMatch("Vatican", "Europe", None))

    assert result.country == "NA"
    assert result.continent == "NA"


def test_coco_false_positive_is_dropped_to_na(standardizer):
    result = standardizer._to_insdc(LocationMatch("water", "NA", None))

    assert result.country == "NA"


# =============================================================================
# find_best_location
# =============================================================================


def test_united_states_resolves_to_insdc_usa(standardizer):
    match = standardizer.find_best_location("TEST", "geo_loc_name", "United States")

    assert match.country == "USA"


def test_vietnam_resolves_to_insdc_viet_nam(standardizer):
    match = standardizer.find_best_location("TEST", "geo_loc_name", "Vietnam")

    assert match.country == "Viet Nam"


def test_conformant_country_unchanged_end_to_end(standardizer):
    match = standardizer.find_best_location("TEST", "geo_loc_name", "Germany")

    assert match.country == "Germany"


def test_country_colon_city_keeps_sublocation_after_remap(standardizer):
    match = standardizer.find_best_location("TEST", "geo_loc_name", "United States: Boston")

    assert match.country == "USA"
    assert match.sublocation == "Boston"


# =============================================================================
# Record-level resolution
# =============================================================================


def test_record_standardization_returns_typed_location_with_origins_and_diagnostics(standardizer):
    outcome = standardizer.standardize(
        {
            "accession": "TEST",
            "loc_attr_orig": "geo_loc_name",
            "loc_val_orig": "Germany: Berlin",
        }
    )

    assert outcome == LocationOutcome(
        continent="Europe",
        un_region="Western Europe",
        country="Germany",
        sublocation="Berlin",
        origins=(LocationOrigin("geo_loc_name", "Germany: Berlin"),),
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
# Absent, unresolved, and disabled candidates
# =============================================================================


def test_record_standardization_distinguishes_absent_unresolved_and_disabled_candidates(
    monkeypatch,
):
    standardizer = LocationStandardizer(CONFIG_PATH, client=None)
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
    assert absent.diagnostics == (LocationDiagnostic.ABSENT_CANDIDATES,)
    assert isinstance(disabled, LocationRejection)
    assert disabled.diagnostics == (LocationDiagnostic.LLM_DISABLED,)
    assert disabled.llm_calls == 0
    assert isinstance(unresolved, LocationRejection)
    assert unresolved.diagnostics == (LocationDiagnostic.UNRESOLVED_PLACE,)


def test_record_standardization_counts_unmappable_direct_result_without_logging(caplog):
    standardizer = LocationStandardizer(CONFIG_PATH, client=None)
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
        ('result: {"country": "Germany"}', LocationDiagnostic.INVALID_LLM_RESPONSE, False),
        ("{}", LocationDiagnostic.INVALID_LLM_RESPONSE, False),
        (
            '{"country": "Germany", "continent": "Europe"}',
            LocationDiagnostic.INVALID_LLM_RESPONSE,
            False,
        ),
        ('{"country": ["Germany"]}', LocationDiagnostic.INVALID_LLM_RESPONSE, False),
        ('{"country": {"name": "Germany"}}', LocationDiagnostic.INVALID_LLM_RESPONSE, False),
        ('{"country": 123}', LocationDiagnostic.INVALID_LLM_RESPONSE, False),
        ('{"country": "  "}', LocationDiagnostic.INVALID_LLM_RESPONSE, False),
    ],
)
def test_record_standardization_distinguishes_model_resolution_and_invalid_response(
    monkeypatch, content, expected, writes_cache, caplog
):
    response = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])
    client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=lambda **_kwargs: response),
        ),
        close=lambda: None,
    )
    standardizer = LocationStandardizer(
        CONFIG_PATH, client=client, llm_settings=LLMSettings(None, None, "test-model")
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
        fingerprint, country, continent = cache_writes[0]
        assert len(fingerprint) == 64
        assert (country, continent) == ("Germany", "Europe")
    assert caplog.messages == []


def test_record_standardization_distinguishes_cache_resolution(monkeypatch):
    standardizer = LocationStandardizer(CONFIG_PATH, client=None)
    monkeypatch.setattr(standardizer.cache, "get", lambda _context: ("Germany", "Europe"))
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


def test_location_cache_reuses_identical_request_when_only_prompt_metadata_changes(tmp_path):
    calls: list[dict] = []
    first_config = _location_config(tmp_path, suffix="prompt_version: first\n")
    second_config = _location_config(tmp_path, suffix="prompt_version: second\n")

    first = _standardize_model_location(first_config, calls)
    second = _standardize_model_location(second_config, calls)

    assert first.diagnostics == (LocationDiagnostic.LLM_RESOLUTION,)
    assert second.diagnostics == (LocationDiagnostic.CACHE_RESOLUTION,)
    assert len(calls) == 1


@pytest.mark.parametrize("changed_component", ["message", "model", "parameter", "schema"])
def test_location_cache_misses_when_canonical_request_changes(
    tmp_path, monkeypatch, changed_component
):
    calls: list[dict] = []
    first_config = _location_config(tmp_path)
    _standardize_model_location(first_config, calls)

    second_config = _location_config(tmp_path)
    second_model = "test-model"
    if changed_component == "message":
        second_config = _location_config(
            tmp_path,
            suffix="llm_system_prompt: A changed fully rendered system prompt.\n",
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

    second = _standardize_model_location(second_config, calls, model=second_model)

    assert second.diagnostics == (LocationDiagnostic.LLM_RESOLUTION,)
    assert len(calls) == 2


def test_record_standardization_counts_recoverable_model_failure_once_per_record(
    monkeypatch, caplog
):
    def timeout(**_kwargs):
        raise openai.APITimeoutError(request=httpx.Request("POST", "https://example.test"))

    client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=timeout)),
        close=lambda: None,
    )
    standardizer = LocationStandardizer(
        CONFIG_PATH, client=client, llm_settings=LLMSettings(None, None, "test-model")
    )
    monkeypatch.setattr(standardizer.cache, "get", lambda _context: None)
    monkeypatch.setattr(standardizer.cache, "set", lambda *_args: None)
    try:
        outcome = standardizer.standardize(
            {
                "accession": "TIMEOUT",
                "loc_attr_orig": "geo_loc_name",
                "loc_val_orig": "timeout place 564738",
            }
        )
    finally:
        standardizer.close()

    assert outcome.diagnostics == (LocationDiagnostic.RECOVERABLE_LLM_FAILURE,)
    assert caplog.messages == []


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
def test_recoverable_model_failure_can_be_retried(monkeypatch, failure):
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
        CONFIG_PATH, client=client, llm_settings=LLMSettings(None, None, "test-model")
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
    fingerprint, country, continent = cached_results[0]
    assert len(fingerprint) == 64
    assert (country, continent) == ("Germany", "Europe")
