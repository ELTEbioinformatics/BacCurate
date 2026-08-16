"""
Standardize submitted geographic-location values to INSDC geographical locations.

For each submitted value, the first step that resolves a country wins:
  (1) parse coordinates and reverse-geocode
  (2) exact match against the INSDC location list
  (3) `country_converter` on the first segment.

Values that none of these resolve are checked against reviewed mappings in `config/location.yaml`.
Still-unresolved values go to the review worklist (`location_review_worklist.tsv`).
"""

import logging
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from functools import lru_cache
from pathlib import Path
from types import MappingProxyType

import country_converter as coco
import reverse_geocode

from baccurate.adapters.policy_yaml import PolicyConfigurationError, load_policy_mapping
from baccurate.paths import DEFAULT_GEO_LOC_LIST
from baccurate.standardization._attribute_value_text import split_pipe_separated
from baccurate.standardization.supporting_attribute_value_pair import SupportingAttributeValuePair

logger = logging.getLogger(__name__)


def normalize_submitted_location_value(value: str) -> str:
    """
    Normalize a submitted value to its reviewed lookup key.

    Trims whitespace, collapses runs to a single space, and lowercases. Punctuation,
    accents, and word order are kept so that reviewed matching stays exact.
    """
    return re.sub(r"\s+", " ", value).strip().lower()


# --- Policy ---


@dataclass(frozen=True, slots=True)
class LocationPolicy:
    """Validated geographic-location standardization policy."""

    schema_version: int
    coordinate_attributes: tuple[str, ...]
    insdc_country_map: Mapping[str, str]
    reviewed_mappings: Mapping[str, str]
    reviewed_unmapped: frozenset[str]
    geo_loc_list_path: Path

    @classmethod
    def load(cls, path: Path | str) -> "LocationPolicy":
        """Load YAML mappings."""
        source = Path(path)
        return _parse_location_policy(load_policy_mapping(source), source)


def _location_policy_error(source: Path, key: str, message: str) -> PolicyConfigurationError:
    return PolicyConfigurationError(f"{source}: {key}: {message}")


def _require_string_mapping(
    config: Mapping[object, object],
    key: str,
    source: Path,
) -> dict[str, str]:
    raw = config.get(key)
    if not isinstance(raw, Mapping):
        raise _location_policy_error(source, key, "must be a mapping")
    mapping: dict[str, str] = {}
    for submitted, standardized in raw.items():
        if not isinstance(submitted, str) or not submitted.strip():
            raise _location_policy_error(source, key, "keys must be non-empty strings")
        if not isinstance(standardized, str) or not standardized.strip():
            raise _location_policy_error(
                source,
                f"{key}.{submitted}",
                "must be a non-empty string",
            )
        mapping[submitted] = standardized
    return mapping


def _parse_location_policy(
    config: Mapping[object, object],
    source: Path,
) -> LocationPolicy:
    allowed = {
        "schema_version",
        "coordinate_attributes",
        "insdc_country_map",
        "reviewed_mappings",
        "reviewed_unmapped",
        "geo_loc_list_path",
    }
    unknown = set(config) - allowed
    if unknown:
        key = sorted(str(value) for value in unknown)[0]
        raise _location_policy_error(source, f"top level.{key}", "unknown policy key")

    schema_version = config.get("schema_version")
    if type(schema_version) is not int:
        raise _location_policy_error(source, "schema_version", "must be integer version 2")
    if schema_version != 2:
        raise _location_policy_error(
            source,
            "schema_version",
            f"unsupported schema version {schema_version}; supported schema version is 2; "
            "migrate this location policy before retrying",
        )

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

    insdc_map = _require_string_mapping(config, "insdc_country_map", source)
    reviewed_mappings = _require_string_mapping(config, "reviewed_mappings", source)

    reviewed_unmapped = config.get("reviewed_unmapped")
    if not isinstance(reviewed_unmapped, list):
        raise _location_policy_error(source, "reviewed_unmapped", "must be a list of strings")
    for index, value in enumerate(reviewed_unmapped):
        if not isinstance(value, str) or not value.strip():
            raise _location_policy_error(
                source,
                f"reviewed_unmapped.{index}",
                "must be a non-empty string",
            )

    geo_loc_list_path = config.get("geo_loc_list_path", str(DEFAULT_GEO_LOC_LIST))
    if not isinstance(geo_loc_list_path, str) or not geo_loc_list_path.strip():
        raise _location_policy_error(source, "geo_loc_list_path", "must be a non-empty string")
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

    return LocationPolicy(
        schema_version=2,
        coordinate_attributes=tuple(coordinate_attributes),
        insdc_country_map=MappingProxyType(insdc_map),
        reviewed_mappings=MappingProxyType(reviewed_mappings),
        reviewed_unmapped=frozenset(reviewed_unmapped),
        geo_loc_list_path=geo_loc_path,
    )


# --- Coordinate parsing ---

# Characters used as degree, minute, and second symbols in submitted values.
# U+FFFD and the Mac Roman "∞" (these are actually present in the dataset) are
# encoding-corruption artifacts. Their original symbol cannot be recovered, so
# both are accepted as valid symbols here.
_MARK_CHARACTERS = "'’′\"”″?_°º˚∞�"  # noqa: RUF001 - these symbols are the submitted characters
_MARK = f"[{re.escape(_MARK_CHARACTERS)}]"

_NUMBER = r"\d+(?:\.\d*)?"

# Separators between the latitude and the longitude component.
_SEPARATOR = r"[\s,;/_]"

# A coordinate must match as a whole value; only these characters may wrap it.
# U+FFFC is another encoding-corruption artifact, like U+FFFD above.
_COORDINATE_PADDING = " \t\r\n()￼"


def _hemisphere_component(name: str, hemispheres: str) -> str:
    """Build a regex for one coordinate component: degrees [minutes [seconds]] hemisphere."""
    return (
        rf"(?P<{name}_sign>[+-]?)\s*(?P<{name}_d>{_NUMBER})"
        rf"(?:\s*{_MARK}\s*(?P<{name}_m>{_NUMBER})(?:\s*{_MARK}\s*(?P<{name}_s>{_NUMBER}))?)?"
        rf"(?:\s*{_MARK})?\s*(?P<{name}_h>[{hemispheres}])"
    )


def _decimal_component(name: str) -> str:
    """Build a regex for one coordinate component: signed decimal degrees."""
    return rf"(?P<{name}_sign>[+-]?)\s*(?P<{name}_d>{_NUMBER})(?:\s*{_MARK})?"


# Degrees/minutes/seconds with a hemisphere letter (N/S/E/W). The letter is mandatory.
# Without it, "6°12'52\"_106°50'42\"" would need guessed signs (e.g. Jakarta at 6°S).
_HEMISPHERE_COORDINATE = re.compile(
    _hemisphere_component("lat", "NS") + rf"{_SEPARATOR}+" + _hemisphere_component("lon", "EW"),
    re.IGNORECASE,
)
# Signed decimal pair. Its separator is mandatory, otherwise a bare run of digits would split
# into a latitude and a longitude.
_DECIMAL_COORDINATE = re.compile(
    _decimal_component("lat") + rf"{_SEPARATOR}+" + _decimal_component("lon")
)


def _parse_coordinate(value: str) -> tuple[float, float] | None:
    """
    Read a whole value as a `(latitude, longitude)` pair, or return `None`.

    Only the two accepted forms are read, and only when they match the whole value, so that
    postal addresses and other number pairs embedded in place names are not mistaken for
    coordinates.
    """
    stripped = value.strip(_COORDINATE_PADDING).lstrip(_MARK_CHARACTERS)
    match = _HEMISPHERE_COORDINATE.fullmatch(stripped) or _DECIMAL_COORDINATE.fullmatch(stripped)
    if match is None:
        return None
    parts = match.groupdict()
    degrees: list[float] = []
    for name in ("lat", "lon"):
        magnitude = float(parts[f"{name}_d"])
        magnitude += float(parts.get(f"{name}_m") or 0) / 60
        magnitude += float(parts.get(f"{name}_s") or 0) / 3600
        hemisphere = (parts.get(f"{name}_h") or "").upper()
        negative = parts[f"{name}_sign"] == "-" or hemisphere in ("S", "W")
        degrees.append(-magnitude if negative else magnitude)
    latitude, longitude = degrees
    if not (-90 <= latitude <= 90 and -180 <= longitude <= 180):
        return None
    return latitude, longitude


# --- Helpers ---


def _split_sublocation(value: str) -> tuple[str, str | None]:
    """
    Split a value into its head (country) and its sublocation (state, city etc.).
    """
    head, _, tail = value.partition(":")
    return head, tail.strip() or None


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


# --- Data structures ---


class LocationDiagnostic(StrEnum):
    """Diagnostic labels for location standardization."""

    ABSENT_VALUES = "absent_values"
    UNRESOLVED_PLACE = "unresolved_place"
    RECOVERABLE_COORDINATE_FAILURE = "recoverable_coordinate_failure"
    UNMAPPABLE_RESULT = "unmappable_result"
    COUNTRY_CONFLICT = "country_conflict"
    REVIEWED_UNMAPPED = "reviewed_unmapped"
    REVIEWED_MAPPING_CONFLICT = "reviewed_mapping_conflict"


class LocationResolutionRoute(StrEnum):
    """The path that produced a record's standardized location.

    `INSDC_TERM` and `COUNTRY_CONVERSION` stay apart because they reach the country in
    different ways. `INSDC_TERM` is membership of the INSDC list. `COUNTRY_CONVERSION`
    is `country_converter` on the first segment of the value.
    """

    COORDINATE = "coordinate"
    INSDC_TERM = "insdc_term"
    COUNTRY_CONVERSION = "country_conversion"
    REVIEWED_MAPPING = "reviewed_mapping"


@dataclass(frozen=True, slots=True)
class LocationMatch:
    """Result of matching one submitted attribute-value pair."""

    country: str
    sublocation: str | None  # region below country level (state, city, etc.)
    diagnostics: tuple[LocationDiagnostic, ...] = ()
    route: LocationResolutionRoute | None = None


@dataclass(frozen=True, slots=True)
class UnresolvedLocationInput:
    """
    One submitted pair that produced no usable INSDC geographical location.

    A standardized record can still contribute unresolved inputs to the review worklist.
    Values already covered by a reviewed section are excluded.
    """

    attribute: str
    value: str


@dataclass(frozen=True, slots=True)
class LocationOutcome:
    """A standardized geographic location with supporting attribute-value pairs and diagnostics."""

    un_region: str
    country: str
    sublocation: str | None
    supporting_pairs: tuple[SupportingAttributeValuePair, ...]
    route: LocationResolutionRoute
    unresolved_inputs: tuple[UnresolvedLocationInput, ...] = ()
    coordinate_decodes: int = 0
    insdc_term_matches: int = 0
    country_conversion_matches: int = 0
    reviewed_mapping_matches: int = 0
    diagnostics: tuple[LocationDiagnostic, ...] = ()


@dataclass(frozen=True, slots=True)
class LocationRejection:
    """An extracted metadata record with no usable location, plus diagnostics."""

    unresolved_inputs: tuple[UnresolvedLocationInput, ...] = ()
    coordinate_decodes: int = 0
    insdc_term_matches: int = 0
    country_conversion_matches: int = 0
    reviewed_mapping_matches: int = 0
    diagnostics: tuple[LocationDiagnostic, ...] = ()


@dataclass(frozen=True, slots=True)
class _RecordResolution:
    """Internal record-level result before it becomes an outcome or a rejection."""

    country: str | None
    sublocation: str | None
    supporting_pairs: tuple[SupportingAttributeValuePair, ...]
    unresolved_inputs: tuple[UnresolvedLocationInput, ...]
    diagnostics: tuple[LocationDiagnostic, ...]
    route: LocationResolutionRoute | None = None


# --- Main class ---


class LocationStandardizer:
    def __init__(
        self,
        policy: LocationPolicy,
        *,
        result_logger: logging.Logger | None = None,
    ) -> None:
        self.logger = result_logger or logger
        self.policy = policy
        self.coordinate_attributes = set(policy.coordinate_attributes)
        self.insdc_map = dict(policy.insdc_country_map)
        self.insdc_names = self._load_insdc_names(policy.geo_loc_list_path)
        self.insdc_by_normalized_name = {
            normalize_submitted_location_value(name): name for name in self.insdc_names
        }
        self.reviewed_mappings = dict(policy.reviewed_mappings)
        self.reviewed_unmapped = frozenset(policy.reviewed_unmapped)

        self.cc = coco.CountryConverter()
        logging.getLogger("country_converter").setLevel(logging.CRITICAL)

        # Cache on this instance so the cache stays small
        # and is freed with the standardizer
        self._country_converter_lookup = lru_cache(maxsize=4096)(self._country_converter_lookup)
        self._country_to_unregion = lru_cache(maxsize=4096)(self._country_to_unregion)
        self._decode_coordinate = lru_cache(maxsize=4096)(self._decode_coordinate)

        self.stats = {
            "coordinate_decodes": 0,
            "insdc_term_matches": 0,
            "country_conversion_matches": 0,
            "reviewed_mapping_matches": 0,
        }

    @staticmethod
    def _load_insdc_names(path: Path | str) -> set[str]:
        """Load the INSDC geo_loc_name list."""
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
                (LocationDiagnostic.UNMAPPABLE_RESULT,),
            )
        if mapped == match.country:
            return match
        return LocationMatch(mapped, match.sublocation, match.diagnostics, match.route)

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

    def _country_to_unregion(self, country: str) -> str:
        """Map a standardized country name to its UN region via country_converter."""
        if not country or country == "NA":
            return "NA"
        return self._country_converter_lookup(country, "UNregion") or "NA"

    def _decode_coordinate(
        self,
        coordinates: tuple[float, float],
    ) -> tuple[str | None, str | None]:
        """Decode a (latitude, longitude) pair to (raw_country, city) via reverse_geocode."""
        info = reverse_geocode.get(coordinates)
        return info.get("country"), info.get("city")

    def _try_coordinate(self, value: str) -> LocationMatch | None:
        """
        Decode a value that parses as a coordinate pair.
        """
        coordinates = _parse_coordinate(value)
        if coordinates is None:
            return None

        try:
            raw_country, city = self._decode_coordinate(coordinates)
        except Exception:
            return LocationMatch(
                "NA",
                None,
                (LocationDiagnostic.RECOVERABLE_COORDINATE_FAILURE,),
            )
        if raw_country is None:
            return LocationMatch("NA", None)

        country = self._country_converter_lookup(raw_country, "name_short") or raw_country

        self.stats["coordinate_decodes"] += 1
        return LocationMatch(country, city, route=LocationResolutionRoute.COORDINATE)

    def _try_insdc_term(self, value: str) -> LocationMatch | None:
        """
        Match a value that is an INSDC geographical location verbatim, or before its first colon.

        These values are already controlled terms, so they skip `country_converter`.
        Substring matching would turn "Indian Ocean" into "India" and
        "Gaza Strip" into "State of Palestine".
        """
        head, tail = _split_sublocation(value)
        for key, sublocation in ((value, None), (head, tail)):
            term = self.insdc_by_normalized_name.get(normalize_submitted_location_value(key))
            if term is not None:
                self.stats["insdc_term_matches"] += 1
                return LocationMatch(term, sublocation, route=LocationResolutionRoute.INSDC_TERM)
        return None

    def _try_country_converter(self, value: str) -> LocationMatch | None:
        """
        Run `country_converter` on the first segment of a value.

        Only the text before the first `:`, `,`, or `;` is used. Later segments
        are ignored because country_converter treats two-letter tokens as ISO codes,
        which turns e.g. "Morehead, KY" into "Cayman Islands".

        Returns None on failure so the caller can try the reviewed fallback.
        """
        sublocation = _split_sublocation(value)[1]
        # country_converter matches by regex, so a run of whitespace inside the key would miss.
        first_segment = re.sub(r"\s+", " ", re.split(r"[:;,]", value, maxsplit=1)[0]).strip()
        if not first_segment:
            return None
        country = self._country_converter_lookup(first_segment, "name_short")
        if country is None:
            return None
        self.stats["country_conversion_matches"] += 1
        return LocationMatch(country, sublocation, route=LocationResolutionRoute.COUNTRY_CONVERSION)

    def _match_value(self, pair: SupportingAttributeValuePair) -> LocationMatch | None:
        """Try each step on one submitted pair, the first that resolves a country wins."""
        match = self._try_coordinate(pair.value)
        if match is not None:
            return match
        if pair.attribute in self.coordinate_attributes and not any(
            character.isalpha() for character in pair.value
        ):
            return None
        return self._try_insdc_term(pair.value) or self._try_country_converter(pair.value)

    # --- Per-record dispatch ---

    def standardize(
        self, extracted_record: Mapping[str, str]
    ) -> LocationOutcome | LocationRejection:
        """Standardize one extracted metadata record and return the result without saving it."""
        accession = extracted_record.get("accession", "")
        attributes = tuple(split_pipe_separated(extracted_record.get("loc_attr_orig", "")))
        values = tuple(split_pipe_separated(extracted_record.get("loc_val_orig", "")))
        if len(attributes) != len(values):
            raise ValueError(
                f"Malformed location attribute-value pairs for {accession}: "
                f"loc_attr_orig={len(attributes)}, loc_val_orig={len(values)}; counts must match"
            )
        before = self.stats.copy()
        resolution = self._resolve_record(attributes, values)
        published = {
            "unresolved_inputs": resolution.unresolved_inputs,
            "coordinate_decodes": self.stats["coordinate_decodes"] - before["coordinate_decodes"],
            "insdc_term_matches": (self.stats["insdc_term_matches"] - before["insdc_term_matches"]),
            "country_conversion_matches": (
                self.stats["country_conversion_matches"] - before["country_conversion_matches"]
            ),
            "reviewed_mapping_matches": (
                self.stats["reviewed_mapping_matches"] - before["reviewed_mapping_matches"]
            ),
            "diagnostics": resolution.diagnostics,
        }
        if resolution.country is None:
            return LocationRejection(**published)
        if resolution.route is None:
            raise ValueError(f"Standardized location without a resolution route for {accession}")
        return LocationOutcome(
            un_region=self._country_to_unregion(resolution.country),
            country=resolution.country,
            sublocation=resolution.sublocation,
            supporting_pairs=resolution.supporting_pairs,
            route=resolution.route,
            **published,
        )

    def _resolve_record(
        self,
        attributes: Sequence[str],
        values: Sequence[str],
    ) -> _RecordResolution:
        """
        Resolve one BioSample record's geographic-location evidence.

        Deterministic resolution runs first with its sublocation preference. The reviewed
        fallback is tried only when no step produced a usable INSDC location
        (including when the selected match ended as `unmappable_result`), and it tries
        every submitted value as a lookup key.
        """
        submitted_pairs = tuple(
            SupportingAttributeValuePair(attribute, value)
            for attribute, value in zip(attributes, values, strict=True)
        )
        pairs = [
            SupportingAttributeValuePair(attribute.strip(), value.strip())
            for attribute, value in zip(attributes, values, strict=True)
            if value.strip()
        ]
        if not pairs:
            return _RecordResolution(None, None, (), (), (LocationDiagnostic.ABSENT_VALUES,))

        # Only matches that map to a valid INSDC location are considered.
        # This prevents an unmappable coordinate from being selected over
        # a valid country on the same record.
        insdc_matches: list[LocationMatch] = []
        unusable_pairs: list[SupportingAttributeValuePair] = []
        coordinate_failure = False
        unmappable_result = False
        for pair in pairs:
            match = self._match_value(pair)
            if match is not None and (
                LocationDiagnostic.RECOVERABLE_COORDINATE_FAILURE in match.diagnostics
            ):
                coordinate_failure = True
            if match is None or match.country == "NA":
                unusable_pairs.append(pair)
                continue
            crosswalked = self._to_insdc(match)
            if crosswalked.country == "NA":
                unmappable_result = True
                unusable_pairs.append(pair)
                continue
            insdc_matches.append(crosswalked)

        if insdc_matches:
            # Prefer matches with a sublocation, since they carry more detail.
            with_sublocation = [match for match in insdc_matches if match.sublocation]
            selected = with_sublocation[0] if with_sublocation else insdc_matches[0]
            unresolved = self._unresolved_inputs(unusable_pairs)
            return _RecordResolution(
                country=selected.country,
                sublocation=selected.sublocation,
                supporting_pairs=submitted_pairs,
                unresolved_inputs=unresolved,
                diagnostics=self._record_diagnostics(
                    coordinate_failure=coordinate_failure,
                    unmappable_result=False,
                    country_conflict=len({match.country for match in insdc_matches}) > 1,
                    reviewed_fallback=(),
                    unresolved=unresolved,
                ),
                route=selected.route,
            )

        return self._reviewed_fallback(
            pairs,
            submitted_pairs=submitted_pairs,
            unusable_pairs=unusable_pairs,
            coordinate_failure=coordinate_failure,
            unmappable_result=unmappable_result,
        )

    def _reviewed_fallback(
        self,
        pairs: Sequence[SupportingAttributeValuePair],
        *,
        submitted_pairs: tuple[SupportingAttributeValuePair, ...],
        unusable_pairs: Sequence[SupportingAttributeValuePair],
        coordinate_failure: bool,
        unmappable_result: bool,
    ) -> _RecordResolution:
        """Apply the reviewed geographic-location policy to every submitted value."""
        mapped: list[tuple[SupportingAttributeValuePair, str]] = []
        reviewed_unmapped_present = False
        for pair in pairs:
            key = normalize_submitted_location_value(pair.value)
            target = self.reviewed_mappings.get(key)
            if target is not None:
                self.stats["reviewed_mapping_matches"] += 1
                mapped.append((pair, target))
            elif key in self.reviewed_unmapped:
                reviewed_unmapped_present = True

        unresolved = self._unresolved_inputs(unusable_pairs)

        def resolution(
            country: str | None,
            *diagnostics: LocationDiagnostic,
        ) -> _RecordResolution:
            return _RecordResolution(
                country=country,
                # Reviewed mappings resolve to a country or water body only,
                # so no sublocation.
                sublocation=None,
                supporting_pairs=submitted_pairs,
                unresolved_inputs=unresolved,
                diagnostics=self._record_diagnostics(
                    coordinate_failure=coordinate_failure,
                    unmappable_result=unmappable_result,
                    reviewed_fallback=diagnostics,
                    unresolved=unresolved,
                ),
                route=(LocationResolutionRoute.REVIEWED_MAPPING if country is not None else None),
            )

        if mapped:
            targets = {target for _, target in mapped}
            if len(targets) > 1:
                return resolution(None, LocationDiagnostic.REVIEWED_MAPPING_CONFLICT)
            return resolution(targets.pop())
        if reviewed_unmapped_present:
            return resolution(None, LocationDiagnostic.REVIEWED_UNMAPPED)
        return resolution(None)

    def _unresolved_inputs(
        self,
        unusable_pairs: Sequence[SupportingAttributeValuePair],
    ) -> tuple[UnresolvedLocationInput, ...]:
        """Filter unusable pairs down to those not already covered by a reviewed section."""
        return tuple(
            UnresolvedLocationInput(pair.attribute, pair.value)
            for pair in unusable_pairs
            if (key := normalize_submitted_location_value(pair.value)) not in self.reviewed_mappings
            and key not in self.reviewed_unmapped
        )

    @staticmethod
    def _record_diagnostics(
        *,
        coordinate_failure: bool,
        unmappable_result: bool,
        country_conflict: bool = False,
        reviewed_fallback: Sequence[LocationDiagnostic],
        unresolved: Sequence[UnresolvedLocationInput],
    ) -> tuple[LocationDiagnostic, ...]:
        """
        Order one record's diagnostics: operational, then selection, then review signals.

        Both `unresolved_place` and `country_conflict` can appear on a successfully
        standardized record.
        """
        return (
            *((LocationDiagnostic.RECOVERABLE_COORDINATE_FAILURE,) if coordinate_failure else ()),
            *((LocationDiagnostic.UNMAPPABLE_RESULT,) if unmappable_result else ()),
            *((LocationDiagnostic.COUNTRY_CONFLICT,) if country_conflict else ()),
            *reviewed_fallback,
            *((LocationDiagnostic.UNRESOLVED_PLACE,) if unresolved else ()),
        )
