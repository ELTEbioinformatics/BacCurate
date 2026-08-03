"""
Map location annotations from sample metadata to standardized country names. Falls back to an LLM
for values that country_converter and reverse_geocode cannot resolve.

See location.md for the documentation.
"""

import json
import logging
import os
import re
import string
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from functools import lru_cache
from pathlib import Path
from types import MappingProxyType

import country_converter as coco
import openai
import reverse_geocode

from baccurate.adapters.llm.client import LLMSettings, load_llm_client
from baccurate.adapters.llm.diagnostics import (
    LLMFailureCategory,
    observe_llm_call,
)
from baccurate.adapters.llm.request import CanonicalLLMRequest
from baccurate.adapters.policy_yaml import PolicyConfigurationError, load_policy_mapping
from baccurate.paths import DEFAULT_GEO_LOC_LIST, DEFAULT_LOC_CACHE_DB
from baccurate.standardization._attribute_value_text import split_pipe_separated
from baccurate.standardization._cache import SQLiteKVCache
from baccurate.standardization.supporting_attribute_value_pair import SupportingAttributeValuePair

logger = logging.getLogger(__name__)
_LOAD_CONFIGURED_CLIENT = object()

LOCATION_LLM_PARAMETERS: dict[str, object] = {"temperature": 0, "seed": 100}
# This needs to be bumped by hand whenever parsing/response changes
LOCATION_RESPONSE_SCHEMA_ID = "baccurate.location.country.v1"


@dataclass(frozen=True, slots=True)
class LocationPrompts:
    """The geographic location prompt text used in canonical LLM requests."""

    system: str
    user_template: str


@dataclass(frozen=True, slots=True)
class LocationPolicy:
    """Validated geographic-location standardization policy."""

    schema_version: int
    prompt_version: str
    coordinate_attributes: tuple[str, ...]
    prompts: LocationPrompts
    insdc_country_map: Mapping[str, str]
    geo_loc_list_path: Path
    cache_db_path: Path

    @classmethod
    def load(cls, path: Path | str) -> "LocationPolicy":
        """Strictly load geographic-location policy from one YAML source."""
        source = Path(path)
        return _parse_location_policy(load_policy_mapping(source), source)

    def serialize(self) -> str:
        """Return deterministic canonical JSON without changing legacy identities."""
        return json.dumps(
            {
                "schema_version": self.schema_version,
                "prompt_version": self.prompt_version,
                "coordinate_attributes": list(self.coordinate_attributes),
                "llm_system_prompt": self.prompts.system,
                "llm_user_prompt_template": self.prompts.user_template,
                "insdc_country_map": dict(self.insdc_country_map),
                "geo_loc_list_path": self.geo_loc_list_path.as_posix(),
                "cache_db_path": self.cache_db_path.as_posix(),
            },
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )


def _location_policy_error(source: Path, key: str, message: str) -> PolicyConfigurationError:
    return PolicyConfigurationError(f"{source}: {key}: {message}")


def _require_location_string(
    config: Mapping[object, object],
    key: str,
    source: Path,
) -> str:
    value = config.get(key)
    if not isinstance(value, str) or not value.strip():
        raise _location_policy_error(source, key, "must be a non-empty string")
    return value


def _validate_location_user_prompt(template: str, source: Path) -> None:
    try:
        fields = [
            (field_name, format_spec, conversion)
            for _, field_name, format_spec, conversion in string.Formatter().parse(template)
            if field_name is not None
        ]
    except ValueError as error:
        raise _location_policy_error(
            source, "llm_user_prompt_template", f"malformed format string: {error}"
        ) from error
    if fields != [("attr_val_pairs", "", None)]:
        raise _location_policy_error(
            source,
            "llm_user_prompt_template",
            "must contain exactly one {attr_val_pairs} placeholder and no other placeholders",
        )


def _parse_location_policy(
    config: Mapping[object, object],
    source: Path,
) -> LocationPolicy:
    allowed = {
        "schema_version",
        "prompt_version",
        "coordinate_attributes",
        "llm_system_prompt",
        "llm_user_prompt_template",
        "insdc_country_map",
        "geo_loc_list_path",
        "cache_db_path",
    }
    unknown = set(config) - allowed
    if unknown:
        key = sorted(str(value) for value in unknown)[0]
        raise _location_policy_error(source, f"top level.{key}", "unknown policy key")

    schema_version = config.get("schema_version")
    if type(schema_version) is not int:
        raise _location_policy_error(source, "schema_version", "must be integer version 1")
    if schema_version != 1:
        raise _location_policy_error(
            source,
            "schema_version",
            f"unsupported schema version {schema_version}; supported schema version is 1; "
            "migrate this location policy before retrying",
        )
    prompt_version = _require_location_string(config, "prompt_version", source)

    coordinate_attributes = config.get("coordinate_attributes")
    if not isinstance(coordinate_attributes, list):
        raise _location_policy_error(source, "coordinate_attributes", "must be a list of strings")
    for index, attribute in enumerate(coordinate_attributes):
        if not isinstance(attribute, str) or not attribute.strip():
            raise _location_policy_error(
                source,
                f"coordinate_attributes.{index}",
                "must be a non-empty string",
            )

    system_prompt = _require_location_string(config, "llm_system_prompt", source)
    user_prompt = _require_location_string(config, "llm_user_prompt_template", source)
    _validate_location_user_prompt(user_prompt, source)

    raw_insdc_map = config.get("insdc_country_map")
    if not isinstance(raw_insdc_map, Mapping):
        raise _location_policy_error(source, "insdc_country_map", "must be a mapping")
    insdc_map: dict[str, str] = {}
    for submitted_name, standardized_name in raw_insdc_map.items():
        if not isinstance(submitted_name, str) or not submitted_name.strip():
            raise _location_policy_error(
                source, "insdc_country_map", "keys must be non-empty strings"
            )
        if not isinstance(standardized_name, str) or not standardized_name.strip():
            raise _location_policy_error(
                source,
                f"insdc_country_map.{submitted_name}",
                "must be a non-empty string",
            )
        insdc_map[submitted_name] = standardized_name

    geo_loc_list_path = config.get("geo_loc_list_path", str(DEFAULT_GEO_LOC_LIST))
    if not isinstance(geo_loc_list_path, str) or not geo_loc_list_path.strip():
        raise _location_policy_error(source, "geo_loc_list_path", "must be a non-empty string")
    cache_db_path = config.get("cache_db_path", str(DEFAULT_LOC_CACHE_DB))
    if not isinstance(cache_db_path, str) or not cache_db_path.strip():
        raise _location_policy_error(source, "cache_db_path", "must be a non-empty string")
    geo_loc_path = Path(geo_loc_list_path)
    try:
        with geo_loc_path.open("r", encoding="utf-8"):
            pass
    except (OSError, UnicodeError) as error:
        raise _location_policy_error(
            source,
            "geo_loc_list_path",
            f"must select a readable file: {error}",
        ) from error
    cache_path = Path(cache_db_path)
    cache_parent = cache_path.parent
    if not cache_parent.is_dir():
        raise _location_policy_error(
            source,
            "cache_db_path",
            f"parent directory does not exist: {cache_parent}",
        )
    writable_cache_target = cache_path if cache_path.exists() else cache_parent
    if (cache_path.exists() and not cache_path.is_file()) or not os.access(
        writable_cache_target, os.W_OK
    ):
        raise _location_policy_error(
            source,
            "cache_db_path",
            "must select a writable database file",
        )

    return LocationPolicy(
        schema_version=1,
        prompt_version=prompt_version,
        coordinate_attributes=tuple(coordinate_attributes),
        prompts=LocationPrompts(system=system_prompt, user_template=user_prompt),
        insdc_country_map=MappingProxyType(insdc_map),
        geo_loc_list_path=geo_loc_path,
        cache_db_path=cache_path,
    )


# --- Coordinate patterns ---

# "DD.DDD N/S DD.DDD E/W" e.g. "51.9194 N 19.1451 E"
COORD_NS_EW_PATTERN = re.compile(
    r"(-?\d+\.?\d*)\s*([NS])\s*[,/\s]*\s*(-?\d+\.?\d*)\s*([EW])", re.IGNORECASE
)

# "lat,lon" or "lat/lon" e.g. "43.51/16.44", "-34.6037, -58.3816"
COORD_LAT_LON_PATTERN = re.compile(r"(-?\d+\.?\d*)\s*[,/]\s*(-?\d+\.?\d*)")

# Combined check for the is_coordinate test.
COORD_PATTERN = re.compile(
    r"(-?\d+\.?\d*)\s*([NS])\s*[,/\s]*\s*(-?\d+\.?\d*)\s*([EW])|"
    r"(-?\d+\.?\d*)\s*[,/]\s*(-?\d+\.?\d*)",
    re.IGNORECASE,
)

# --- Helpers ---


def _normalize_coordinates(coord_str: str) -> tuple[float | None, float | None]:
    """Parse a coordinate string into (lat, lon); (None, None) on failure."""
    if not isinstance(coord_str, str) or not coord_str.strip():
        return None, None

    match = COORD_NS_EW_PATTERN.search(coord_str)
    if match:
        lat, lat_dir, lon, lon_dir = match.groups()
        lat = float(lat)
        lon = float(lon)
        if lat_dir.upper() == "S":
            lat = -lat
        if lon_dir.upper() == "W":
            lon = -lon
        return lat, lon

    match = COORD_LAT_LON_PATTERN.search(coord_str)
    if match:
        lat, lon = match.groups()
        return float(lat), float(lon)

    return None, None


def _is_valid_coord(lat: float | None, lon: float | None) -> bool:
    if lat is None or lon is None:
        return False
    return -90 <= lat <= 90 and -180 <= lon <= 180


def _is_coordinate(value: str) -> bool:
    if not isinstance(value, str):
        return False
    return bool(COORD_PATTERN.search(value))


def _extract_string(value) -> str | None:
    """Extract a non-empty string from a value that may be a list/tuple/array."""
    if value is None:
        return None
    if isinstance(value, str):
        s = value.strip()
        return s if s else None
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, (list, tuple, set)):
        for v in value:
            if isinstance(v, str) and v.strip():
                return v.strip()
        return None
    s = str(value).strip()
    return s if s else None


# --- Data structure ---


class LocationDiagnostic(StrEnum):
    """location-resolution vocabulary used by build diagnostics."""

    ABSENT_VALUES = "absent_values"
    UNRESOLVED_PLACE = "unresolved_place"
    LLM_DISABLED = "llm_disabled"
    RECOVERABLE_LLM_FAILURE = "recoverable_llm_failure"
    RECOVERABLE_COORDINATE_FAILURE = "recoverable_coordinate_failure"
    INVALID_LLM_RESPONSE = "invalid_llm_response"
    UNMAPPABLE_RESULT = "unmappable_result"
    COORDINATE_RESOLUTION = "coordinate_resolution"
    DIRECT_RESOLUTION = "direct_resolution"
    CACHE_RESOLUTION = "cache_resolution"
    LLM_RESOLUTION = "llm_resolution"


@dataclass(frozen=True, slots=True)
class LocationMatch:
    """Standardization result for one extracted metadata record."""

    country: str
    sublocation: str | None
    used_llm: bool = False
    diagnostics: tuple[LocationDiagnostic, ...] = ()


@dataclass(frozen=True, slots=True)
class _LLMResponse:
    country: str | None
    diagnostic: LocationDiagnostic | None = None


@dataclass(frozen=True, slots=True)
class LocationOutcome:
    """A standardized geographic location with supporting attribute-value pairs and diagnostics."""

    un_region: str
    country: str
    sublocation: str | None
    supporting_pairs: tuple[SupportingAttributeValuePair, ...]
    coordinate_decodes: int = 0
    direct_matches: int = 0
    cache_hits: int = 0
    llm_calls: int = 0
    diagnostics: tuple[LocationDiagnostic, ...] = ()


@dataclass(frozen=True, slots=True)
class LocationRejection:
    """An extracted metadata record with no usable location, plus diagnostics."""

    coordinate_decodes: int = 0
    direct_matches: int = 0
    cache_hits: int = 0
    llm_calls: int = 0
    diagnostics: tuple[LocationDiagnostic, ...] = ()


# --- Cache ---


class SQLiteCache(SQLiteKVCache):
    _CREATE_TABLE_SQL = """
        CREATE TABLE IF NOT EXISTS cache (
            hash_id TEXT PRIMARY KEY,
            country TEXT
        )
    """

    def __init__(self, db_path: Path | str = DEFAULT_LOC_CACHE_DB) -> None:
        super().__init__(db_path)

    def get(self, request_fingerprint: str) -> str | None:
        self.cursor.execute(
            "SELECT country FROM cache WHERE hash_id=?",
            (request_fingerprint,),
        )
        cache_entry = self.cursor.fetchone()
        return cache_entry[0] if cache_entry else None

    def set(self, request_fingerprint: str, country: str) -> None:
        self.cursor.execute(
            "INSERT OR REPLACE INTO cache (hash_id, country) VALUES (?, ?)",
            (request_fingerprint, country),
        )
        self.conn.commit()


# --- Main class ---


class LocationStandardizer:
    def __init__(
        self,
        policy: LocationPolicy,
        *,
        client: openai.OpenAI | object | None = _LOAD_CONFIGURED_CLIENT,
        llm_settings: LLMSettings | None = None,
        result_logger: logging.Logger | None = None,
    ) -> None:
        self.logger = result_logger or logger
        self.policy = policy
        self.coordinate_attributes = set(policy.coordinate_attributes)
        self.insdc_map = dict(policy.insdc_country_map)
        self.insdc_names = self._load_insdc_names(policy.geo_loc_list_path)
        self.llm_system_prompt = policy.prompts.system
        self.llm_user_prompt_template = policy.prompts.user_template

        if client is _LOAD_CONFIGURED_CLIENT:
            self.client, self.llm_model = load_llm_client(llm_settings)
        else:
            self.client = client
            self.llm_model = llm_settings.model if llm_settings else None

        self.cc = coco.CountryConverter()
        logging.getLogger("country_converter").setLevel(logging.CRITICAL)

        # Cache on this instance so the cache stays small
        # and is freed with the standardizer
        self._country_convert = lru_cache(maxsize=4096)(self._country_convert)
        self._country_to_unregion = lru_cache(maxsize=4096)(self._country_to_unregion)
        self.decode_coordinates = lru_cache(maxsize=4096)(self.decode_coordinates)

        self.cache = SQLiteCache(policy.cache_db_path)

        self.stats = {
            "coordinate_decodes": 0,
            "direct_matches": 0,
            "cache_hits": 0,
            "llm_calls": 0,
        }

    @staticmethod
    def _load_insdc_names(path: Path | str) -> set[str]:
        """Load the INSDC geo_loc_name vocabulary."""
        with Path(path).open("r", encoding="utf-8") as f:
            return {line.strip() for line in f if line.strip()}

    def _to_insdc(self, match: LocationMatch) -> LocationMatch:
        """Remap a coco country name to INSDC."""
        if match.country == "NA":
            return match
        mapped = self.insdc_map.get(match.country, match.country)
        if mapped not in self.insdc_names:
            return LocationMatch(
                "NA",
                match.sublocation,
                match.used_llm,
                (LocationDiagnostic.UNMAPPABLE_RESULT,),
            )
        if mapped == match.country:
            return match
        return LocationMatch(
            mapped,
            match.sublocation,
            match.used_llm,
            match.diagnostics,
        )

    # --- Per-value matching ---

    def _country_converter_lookup(self, value: str, to: str) -> str | None:
        """
        Look one value up in country_converter, reporting a failed lookup as None.

        country_converter signals a failed lookup by echoing back whatever `not_found`
        receives, so the sentinel is translated here and never travels further. Callers
        that need a serialized "NA" supply it themselves, per that column's contract.
        """
        converted = _extract_string(self.cc.convert(names=value, to=to, not_found="NA"))
        return None if converted == "NA" else converted

    def _country_convert(self, loc: str) -> str | None:
        """
        Look up a single string via country_converter.

        The input is split on colon/comma/semicolon and each part is tried
        in order, since submitter values often contain trailing extras
        like "France: Paris" or "USA, California". A failed lookup is
        reported as None.
        """
        raw = re.sub(r"\s+", " ", loc).strip()
        for part in re.split(r"[:;,]", raw):
            token = part.strip()
            if not token:
                continue
            if name := self._country_converter_lookup(token, "name_short"):
                return name
        return None

    def _country_to_unregion(self, country: str) -> str:
        """Map a standardized country name to its UN region via country_converter."""
        if not country or country == "NA":
            return "NA"
        return self._country_converter_lookup(country, "UNregion") or "NA"

    def decode_coordinates(self, coord_str: str) -> tuple[str | None, str | None]:
        """Decode a coordinate string to (raw_country, city) via reverse_geocode."""
        lat, lon = _normalize_coordinates(coord_str)
        if not _is_valid_coord(lat, lon):
            return None, None
        info = reverse_geocode.get((lat, lon))
        return info.get("country"), info.get("city")

    def _try_coordinate(self, val: str, attr: str) -> LocationMatch | None:
        """Decode and standardize if the value or attribute looks like a coordinate."""
        if not (_is_coordinate(val) or attr in self.coordinate_attributes):
            return None

        try:
            raw_country, city = self.decode_coordinates(val)
        except Exception:
            return LocationMatch(
                "NA",
                None,
                diagnostics=(LocationDiagnostic.RECOVERABLE_COORDINATE_FAILURE,),
            )
        if raw_country is None:
            return LocationMatch("NA", None, diagnostics=(LocationDiagnostic.UNRESOLVED_PLACE,))

        country = self._country_converter_lookup(raw_country, "name_short") or raw_country

        self.stats["coordinate_decodes"] += 1
        return LocationMatch(
            country,
            city,
            diagnostics=(LocationDiagnostic.COORDINATE_RESOLUTION,),
        )

    def _try_country_converter(self, val: str) -> LocationMatch | None:
        """
        Run country_converter on a non-coordinate value.

        Returns None if country_converter fails so the caller can queue the
        already-identified value for LLM fallback.
        """
        loc_lower = val.strip().lower()
        if not loc_lower:
            return LocationMatch("NA", None)

        # Peel off "Country:City" sublocation before lookup.
        sublocation = None
        loc_clean = val
        if ":" in val:
            country_part, sub = val.split(":", 1)
            loc_clean = country_part.strip()
            sublocation = sub.strip() or None

        country = self._country_convert(loc_clean)
        if country is None:
            return None
        self.stats["direct_matches"] += 1
        return LocationMatch(
            country,
            sublocation,
            diagnostics=(LocationDiagnostic.DIRECT_RESOLUTION,),
        )

    # --- LLM fallback ---

    def _call_llm(
        self,
        accession: str,
        request: CanonicalLLMRequest,
        timeout: int = 30,
    ) -> _LLMResponse:
        try:
            with observe_llm_call(
                accession=accession,
                target="location",
                model=request.model,
            ) as call:
                response = self.client.chat.completions.create(
                    model=request.model,
                    messages=list(request.messages),
                    **request.parameters,
                    timeout=timeout,
                )
        except openai.APITimeoutError:
            return _LLMResponse(None, diagnostic=LocationDiagnostic.RECOVERABLE_LLM_FAILURE)
        except openai.APIError:
            return _LLMResponse(None, diagnostic=LocationDiagnostic.RECOVERABLE_LLM_FAILURE)
        except Exception as e:
            call.failed(LLMFailureCategory.UNEXPECTED)
            raise RuntimeError(f"Unexpected location LLM failure: {e}") from e

        if response is None:
            call.failed(LLMFailureCategory.INVALID_MODEL_RESPONSE)
            return _LLMResponse(None, diagnostic=LocationDiagnostic.INVALID_LLM_RESPONSE)

        try:
            content = response.choices[0].message.content
        except (AttributeError, IndexError):
            call.failed(LLMFailureCategory.INVALID_MODEL_RESPONSE)
            return _LLMResponse(None, diagnostic=LocationDiagnostic.INVALID_LLM_RESPONSE)

        if not content:
            call.failed(LLMFailureCategory.INVALID_MODEL_RESPONSE)
            return _LLMResponse(None, diagnostic=LocationDiagnostic.INVALID_LLM_RESPONSE)

        content = content.strip()

        try:
            parsed = json.loads(content)
        except json.JSONDecodeError:
            call.failed(LLMFailureCategory.INVALID_MODEL_RESPONSE)
            return _LLMResponse(None, diagnostic=LocationDiagnostic.INVALID_LLM_RESPONSE)

        if not isinstance(parsed, dict) or set(parsed) != {"country"}:
            call.failed(LLMFailureCategory.INVALID_MODEL_RESPONSE)
            return _LLMResponse(None, diagnostic=LocationDiagnostic.INVALID_LLM_RESPONSE)

        country = parsed["country"]
        if not isinstance(country, str) or not country.strip():
            call.failed(LLMFailureCategory.INVALID_MODEL_RESPONSE)
            return _LLMResponse(None, diagnostic=LocationDiagnostic.INVALID_LLM_RESPONSE)

        call.accepted()
        return _LLMResponse(country.strip())

    def _llm_fallback(self, accession: str, context_string: str) -> LocationMatch:
        user_prompt = self.llm_user_prompt_template.format(attr_val_pairs=context_string)
        messages = []
        if self.llm_system_prompt:
            messages.append({"role": "system", "content": self.llm_system_prompt})
        messages.append({"role": "user", "content": user_prompt})
        request = CanonicalLLMRequest(
            model=self.llm_model or "",
            messages=tuple(messages),
            parameters=LOCATION_LLM_PARAMETERS,
            response_schema_id=LOCATION_RESPONSE_SCHEMA_ID,
        )
        cached = self.cache.get(request.fingerprint)
        if cached is not None:
            self.stats["cache_hits"] += 1
            diagnostic = (
                LocationDiagnostic.CACHE_RESOLUTION
                if cached != "NA"
                else LocationDiagnostic.UNRESOLVED_PLACE
            )
            # The cache memoizes the model's country_converter name, not the standardized
            # INSDC name, so INSDC remapping has to be applied on this route too. Without
            # it a mapped country (United States -> USA) resolves differently on the first
            # classification than on every later cache hit, and a country absent from the
            # INSDC vocabulary reaches the dataset instead of becoming NA.
            return self._to_insdc(LocationMatch(cached, None, True, (diagnostic,)))

        if self.client is None:
            return LocationMatch("NA", None, True, (LocationDiagnostic.LLM_DISABLED,))

        response = self._call_llm(accession, request)
        self.stats["llm_calls"] += 1

        if response.diagnostic is not None:
            return LocationMatch("NA", None, True, (response.diagnostic,))

        llm_country = response.country
        if llm_country is None:
            raise AssertionError("Successful LLM response has no country")

        if llm_country == "NA":
            self.cache.set(request.fingerprint, "NA")
            return LocationMatch("NA", None, True, (LocationDiagnostic.UNMAPPABLE_RESULT,))

        # Standardize the LLM's country through cc
        resolved_country = self._country_converter_lookup(llm_country, "name_short") or llm_country

        self.cache.set(request.fingerprint, resolved_country)
        return self._to_insdc(
            LocationMatch(
                resolved_country,
                None,
                True,
                (LocationDiagnostic.LLM_RESOLUTION,),
            )
        )

    # --- Per-record dispatch ---

    def standardize(
        self, extracted_record: Mapping[str, str]
    ) -> LocationOutcome | LocationRejection:
        """Standardize one extracted metadata record without performing persistence."""
        accession = extracted_record.get("accession", "")
        attributes = tuple(split_pipe_separated(extracted_record.get("loc_attr_orig", "")))
        values = tuple(split_pipe_separated(extracted_record.get("loc_val_orig", "")))
        if len(attributes) != len(values):
            raise ValueError(
                f"Malformed location attribute-value pairs for {accession}: "
                f"loc_attr_orig={len(attributes)}, loc_val_orig={len(values)}; counts must match"
            )
        before = self.stats.copy()
        match = self.find_best_location(
            accession,
            extracted_record.get("loc_attr_orig", ""),
            extracted_record.get("loc_val_orig", ""),
        )
        diagnostics = {
            "coordinate_decodes": self.stats["coordinate_decodes"] - before["coordinate_decodes"],
            "direct_matches": self.stats["direct_matches"] - before["direct_matches"],
            "cache_hits": self.stats["cache_hits"] - before["cache_hits"],
            "llm_calls": self.stats["llm_calls"] - before["llm_calls"],
            "diagnostics": match.diagnostics,
        }
        if match.country == "NA":
            return LocationRejection(**diagnostics)
        # loc_UNregion carries the UN region derived from the standardized country.
        return LocationOutcome(
            un_region=self._country_to_unregion(match.country),
            country=match.country,
            sublocation=match.sublocation,
            supporting_pairs=tuple(
                SupportingAttributeValuePair(attribute, value)
                for attribute, value in zip(attributes, values, strict=True)
            ),
            **diagnostics,
        )

    def close(self) -> None:
        try:
            self.cache.close()
        finally:
            close_client = getattr(self.client, "close", None)
            if callable(close_client):
                close_client()

    def find_best_location(
        self,
        accession: str,
        attr_str: str,
        val_str: str,
    ) -> LocationMatch:
        attrs = split_pipe_separated(attr_str)
        vals = split_pipe_separated(val_str)

        if not vals:
            return LocationMatch("NA", None, diagnostics=(LocationDiagnostic.ABSENT_VALUES,))

        valid_matches: list[LocationMatch] = []
        rejected_matches: list[LocationMatch] = []
        unmatched_pairs: list[tuple[str, str]] = []

        for attr, val in zip(attrs, vals, strict=False):
            attr = attr.strip()
            val = val.strip()
            if not val:
                continue

            coord_match = self._try_coordinate(val, attr)
            if coord_match is not None:
                if coord_match.country != "NA":
                    valid_matches.append(coord_match)
                else:
                    rejected_matches.append(coord_match)
                continue

            cc_match = self._try_country_converter(val)
            if cc_match is None:
                unmatched_pairs.append((attr, val))
            elif cc_match.country != "NA":
                valid_matches.append(cc_match)

        # Prefer matches that include a sublocation (coord-decoded city
        # or "Country:City" sublocation) since they carry more information.
        if valid_matches:
            with_subloc = [m for m in valid_matches if m.sublocation]
            selected = self._to_insdc(with_subloc[0] if with_subloc else valid_matches[0])
            return self._preserve_operational_diagnostics(selected, rejected_matches)

        if not unmatched_pairs:
            if rejected_matches:
                return self._preserve_operational_diagnostics(rejected_matches[0], rejected_matches)
            return LocationMatch("NA", None, diagnostics=(LocationDiagnostic.UNRESOLVED_PLACE,))

        context = " ".join(f"{a}={v}" for a, v in unmatched_pairs)
        return self._preserve_operational_diagnostics(
            self._llm_fallback(accession, context), rejected_matches
        )

    @staticmethod
    def _preserve_operational_diagnostics(
        match: LocationMatch,
        rejected_matches: list[LocationMatch],
    ) -> LocationMatch:
        operational = LocationDiagnostic.RECOVERABLE_COORDINATE_FAILURE
        if not any(operational in rejected.diagnostics for rejected in rejected_matches):
            return match
        if operational in match.diagnostics:
            return match
        return LocationMatch(
            match.country,
            match.sublocation,
            match.used_llm,
            (operational, *match.diagnostics),
        )
