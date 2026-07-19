"""
Map location annotations from sample metadata to standardized
country and continent names. Falls back to an LLM for values that country_converter and
reverse_geocode cannot resolve.

See location.md for the documentation.
"""

import json
import logging
import re
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from functools import lru_cache
from pathlib import Path

import country_converter as coco
import openai
import reverse_geocode

from baccurate.llm.client import LLMSettings, load_llm_client
from baccurate.llm.diagnostics import (
    LLMFailureCategory,
    observe_llm_call,
)
from baccurate.llm.request import CanonicalLLMRequest
from baccurate.paths import DEFAULT_GEO_LOC_LIST, DEFAULT_LOC_CACHE_DB
from baccurate.utils.cache import SQLiteKVCache
from baccurate.utils.config import load_config
from baccurate.utils.text import split_pipe_separated

logger = logging.getLogger(__name__)
_LOAD_CONFIGURED_CLIENT = object()

LOCATION_MODEL_PARAMETERS: dict[str, object] = {"temperature": 0, "seed": 100}
# This needs to be bumped by hand whenever parsing/response changes
LOCATION_RESPONSE_SCHEMA_ID = "baccurate.location.country.v1"

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

    ABSENT_CANDIDATES = "absent_candidates"
    UNRESOLVED_PLACE = "unresolved_place"
    MODEL_DISABLED = "model_disabled"
    RECOVERABLE_MODEL_FAILURE = "recoverable_model_failure"
    RECOVERABLE_COORDINATE_FAILURE = "recoverable_coordinate_failure"
    INVALID_MODEL_RESPONSE = "invalid_model_response"
    UNMAPPABLE_RESULT = "unmappable_result"
    COORDINATE_RESOLUTION = "coordinate_resolution"
    DIRECT_RESOLUTION = "direct_resolution"
    CACHE_RESOLUTION = "cache_resolution"
    MODEL_RESOLUTION = "model_resolution"


@dataclass(frozen=True, slots=True)
class LocationMatch:
    """Standardization result for one record."""

    country: str
    continent: str
    sublocation: str | None
    used_llm: bool = False
    diagnostics: tuple[LocationDiagnostic, ...] = ()


@dataclass(frozen=True, slots=True)
class _ModelResponse:
    country: str | None
    diagnostic: LocationDiagnostic | None = None


@dataclass(frozen=True, slots=True)
class LocationOrigin:
    """One source attribute/value pair supporting a location result."""

    attribute: str
    value: str


@dataclass(frozen=True, slots=True)
class LocationOutcome:
    """A standardized location with paired source origins and diagnostics."""

    continent: str
    un_region: str
    country: str
    sublocation: str | None
    origins: tuple[LocationOrigin, ...]
    coordinate_decodes: int = 0
    direct_matches: int = 0
    cache_hits: int = 0
    llm_calls: int = 0
    diagnostics: tuple[LocationDiagnostic, ...] = ()


@dataclass(frozen=True, slots=True)
class LocationRejection:
    """A record with no usable standardized location, plus its diagnostics."""

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
            country TEXT,
            continent TEXT
        )
    """

    def __init__(self, db_path: Path | str = DEFAULT_LOC_CACHE_DB) -> None:
        super().__init__(db_path)

    def get(self, request_fingerprint: str) -> tuple[str, str] | None:
        self.cursor.execute(
            "SELECT country, continent FROM cache WHERE hash_id=?",
            (request_fingerprint,),
        )
        row = self.cursor.fetchone()
        return (row[0], row[1]) if row else None

    def set(self, request_fingerprint: str, country: str, continent: str) -> None:
        self.cursor.execute(
            "INSERT OR REPLACE INTO cache (hash_id, country, continent) VALUES (?, ?, ?)",
            (request_fingerprint, country, continent),
        )
        self.conn.commit()


# --- Main class ---


class LocationStandardizer:
    def __init__(
        self,
        config_path: Path | str,
        *,
        client: openai.OpenAI | None | object = _LOAD_CONFIGURED_CLIENT,
        llm_settings: LLMSettings | None = None,
        result_logger: logging.Logger | None = None,
    ) -> None:
        self.logger = result_logger or logger
        self.config = load_config(config_path)
        self.coordinate_attributes = set(self.config.get("coordinate_attributes", []))
        self.insdc_map = dict(self.config.get("insdc_country_map", {}))
        geo_loc_path = self.config.get("geo_loc_list_path", DEFAULT_GEO_LOC_LIST)
        self.insdc_names = self._load_insdc_names(geo_loc_path)
        self.llm_system_prompt = self.config.get("llm_system_prompt")
        self.llm_user_prompt_template = self.config.get("llm_user_prompt_template")

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

        cache_path = self.config.get("cache_db_path", DEFAULT_LOC_CACHE_DB)
        self.cache = SQLiteCache(cache_path)

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
                "NA",
                match.sublocation,
                match.used_llm,
                (LocationDiagnostic.UNMAPPABLE_RESULT,),
            )
        if mapped == match.country:
            return match
        return LocationMatch(
            mapped,
            match.continent,
            match.sublocation,
            match.used_llm,
            match.diagnostics,
        )

    # --- Per-value matching ---

    def _country_convert(self, loc: str) -> tuple[str, str]:
        """
        Look up a single string via country_converter.

        The input is split on colon/comma/semicolon and each part is tried
        in order, since submitter values often contain trailing extras
        like "France: Paris" or "USA, California".
        """
        raw = re.sub(r"\s+", " ", loc).strip()
        for part in re.split(r"[:;,]", raw):
            token = part.strip()
            if not token:
                continue
            name = _extract_string(self.cc.convert(names=token, to="name_short", not_found="NA"))
            if name and name != "NA":
                continent = _extract_string(
                    self.cc.convert(names=name, to="Continent_7", not_found="NA")
                )
                return name, (continent or "NA")
        return "NA", "NA"

    def _country_to_unregion(self, country: str) -> str:
        """Map a standardized country name to its UN region via country_converter."""
        if not country or country == "NA":
            return "NA"
        converted = self.cc.convert(names=country, to="UNregion", not_found="NA")
        return _extract_string(converted) or "NA"

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
                "NA",
                None,
                diagnostics=(LocationDiagnostic.RECOVERABLE_COORDINATE_FAILURE,),
            )
        if raw_country is None:
            return LocationMatch(
                "NA", "NA", None, diagnostics=(LocationDiagnostic.UNRESOLVED_PLACE,)
            )

        cc_country = _extract_string(
            self.cc.convert(names=raw_country, to="name_short", not_found="NA")
        )
        country = cc_country if cc_country and cc_country != "NA" else raw_country
        continent = (
            _extract_string(self.cc.convert(names=country, to="continent", not_found="NA")) or "NA"
        )

        self.stats["coordinate_decodes"] += 1
        return LocationMatch(
            country,
            continent,
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
            return LocationMatch("NA", "NA", None)

        # Peel off "Country:City" sublocation before lookup.
        sublocation = None
        loc_clean = val
        if ":" in val:
            country_part, sub = val.split(":", 1)
            loc_clean = country_part.strip()
            sublocation = sub.strip() or None

        country, continent = self._country_convert(loc_clean)
        if country == "NA":
            return None
        self.stats["direct_matches"] += 1
        return LocationMatch(
            country,
            continent,
            sublocation,
            diagnostics=(LocationDiagnostic.DIRECT_RESOLUTION,),
        )

    # --- LLM fallback ---

    def _call_llm(
        self,
        accession: str,
        request: CanonicalLLMRequest,
        timeout: int = 30,
    ) -> _ModelResponse:
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
            return _ModelResponse(None, diagnostic=LocationDiagnostic.RECOVERABLE_MODEL_FAILURE)
        except openai.APIError:
            return _ModelResponse(None, diagnostic=LocationDiagnostic.RECOVERABLE_MODEL_FAILURE)
        except Exception as e:
            call.failed(LLMFailureCategory.UNEXPECTED)
            raise RuntimeError(f"Unexpected location model failure: {e}") from e

        if response is None:
            call.failed(LLMFailureCategory.INVALID_MODEL_RESPONSE)
            return _ModelResponse(None, diagnostic=LocationDiagnostic.INVALID_MODEL_RESPONSE)

        try:
            content = response.choices[0].message.content
        except (AttributeError, IndexError):
            call.failed(LLMFailureCategory.INVALID_MODEL_RESPONSE)
            return _ModelResponse(None, diagnostic=LocationDiagnostic.INVALID_MODEL_RESPONSE)

        if not content:
            call.failed(LLMFailureCategory.INVALID_MODEL_RESPONSE)
            return _ModelResponse(None, diagnostic=LocationDiagnostic.INVALID_MODEL_RESPONSE)

        content = content.strip()

        try:
            parsed = json.loads(content)
        except json.JSONDecodeError:
            call.failed(LLMFailureCategory.INVALID_MODEL_RESPONSE)
            return _ModelResponse(None, diagnostic=LocationDiagnostic.INVALID_MODEL_RESPONSE)

        if not isinstance(parsed, dict) or set(parsed) != {"country"}:
            call.failed(LLMFailureCategory.INVALID_MODEL_RESPONSE)
            return _ModelResponse(None, diagnostic=LocationDiagnostic.INVALID_MODEL_RESPONSE)

        country = parsed["country"]
        if not isinstance(country, str) or not country.strip():
            call.failed(LLMFailureCategory.INVALID_MODEL_RESPONSE)
            return _ModelResponse(None, diagnostic=LocationDiagnostic.INVALID_MODEL_RESPONSE)

        call.accepted()
        return _ModelResponse(country.strip())

    def _llm_fallback(self, accession: str, context_string: str) -> LocationMatch:
        user_prompt = self.llm_user_prompt_template.format(attr_val_pairs=context_string)
        messages = []
        if self.llm_system_prompt:
            messages.append({"role": "system", "content": self.llm_system_prompt})
        messages.append({"role": "user", "content": user_prompt})
        request = CanonicalLLMRequest(
            model=self.llm_model or "",
            messages=tuple(messages),
            parameters=LOCATION_MODEL_PARAMETERS,
            response_schema_id=LOCATION_RESPONSE_SCHEMA_ID,
        )
        cached = self.cache.get(request.fingerprint)
        if cached is not None:
            self.stats["cache_hits"] += 1
            diagnostic = (
                LocationDiagnostic.CACHE_RESOLUTION
                if cached[0] != "NA"
                else LocationDiagnostic.UNRESOLVED_PLACE
            )
            return LocationMatch(*cached, None, True, (diagnostic,))

        if self.client is None:
            return LocationMatch("NA", "NA", None, True, (LocationDiagnostic.MODEL_DISABLED,))

        response = self._call_llm(accession, request)
        self.stats["llm_calls"] += 1

        if response.diagnostic is not None:
            return LocationMatch("NA", "NA", None, True, (response.diagnostic,))

        llm_country = response.country
        if llm_country is None:
            raise AssertionError("Successful model response has no country")

        if llm_country == "NA":
            self.cache.set(request.fingerprint, "NA", "NA")
            return LocationMatch("NA", "NA", None, True, (LocationDiagnostic.UNMAPPABLE_RESULT,))

        # Standardize the LLM's country through cc
        cc_country = _extract_string(
            self.cc.convert(names=llm_country, to="name_short", not_found="NA")
        )
        if cc_country and cc_country != "NA":
            cc_continent = _extract_string(
                self.cc.convert(names=cc_country, to="continent", not_found="NA")
            )
            result = (cc_country, cc_continent or "NA")
        else:
            result = (llm_country, "NA")

        self.cache.set(request.fingerprint, *result)
        return self._to_insdc(
            LocationMatch(
                *result,
                None,
                True,
                (LocationDiagnostic.MODEL_RESOLUTION,),
            )
        )

    # --- Per-record dispatch ---

    def standardize(self, record: Mapping[str, str]) -> LocationOutcome | LocationRejection:
        """Standardize one extracted record without performing persistence."""
        accession = record.get("accession", "")
        attributes = tuple(split_pipe_separated(record.get("loc_attr_orig", "")))
        values = tuple(split_pipe_separated(record.get("loc_val_orig", "")))
        if len(attributes) != len(values):
            raise ValueError(
                f"Malformed location candidates for {accession}: "
                f"loc_attr_orig={len(attributes)}, loc_val_orig={len(values)}; counts must match"
            )
        before = self.stats.copy()
        match = self.find_best_location(
            accession,
            record.get("loc_attr_orig", ""),
            record.get("loc_val_orig", ""),
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
        return LocationOutcome(
            continent=match.continent,
            un_region=self._country_to_unregion(match.country),
            country=match.country,
            sublocation=match.sublocation,
            origins=tuple(
                LocationOrigin(attribute, value)
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
            return LocationMatch(
                "NA", "NA", None, diagnostics=(LocationDiagnostic.ABSENT_CANDIDATES,)
            )

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
            return LocationMatch(
                "NA", "NA", None, diagnostics=(LocationDiagnostic.UNRESOLVED_PLACE,)
            )

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
            match.continent,
            match.sublocation,
            match.used_llm,
            (operational, *match.diagnostics),
        )
