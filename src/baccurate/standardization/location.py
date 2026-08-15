"""
Standardize submitted geographic-location values to INSDC geographical locations.

Two routes: ``reverse_geocode`` for coordinates and ``country_converter`` for place names.
When neither resolves a record, the reviewed mappings loaded from `config/location.yaml˙are
tried by whole-key match on normalized values. Values that remain unresolved go to the review
worklist (`location_review_worklist.tsv`).
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
        """Strictly load geographic-location policy from one YAML source."""
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


# --- Data structures ---


class LocationDiagnostic(StrEnum):
    """location-resolution vocabulary used by build diagnostics."""

    ABSENT_VALUES = "absent_values"
    UNRESOLVED_PLACE = "unresolved_place"
    RECOVERABLE_COORDINATE_FAILURE = "recoverable_coordinate_failure"
    UNMAPPABLE_RESULT = "unmappable_result"
    COORDINATE_RESOLUTION = "coordinate_resolution"
    DIRECT_RESOLUTION = "direct_resolution"
    REVIEWED_MAPPING_RESOLUTION = "reviewed_mapping_resolution"
    REVIEWED_UNMAPPED = "reviewed_unmapped"
    REVIEWED_MAPPING_CONFLICT = "reviewed_mapping_conflict"


@dataclass(frozen=True, slots=True)
class LocationMatch:
    """Result of matching one submitted attribute-value pair."""

    country: str
    sublocation: str | None
    diagnostics: tuple[LocationDiagnostic, ...] = ()


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
    unresolved_inputs: tuple[UnresolvedLocationInput, ...] = ()
    coordinate_decodes: int = 0
    direct_matches: int = 0
    reviewed_mapping_matches: int = 0
    diagnostics: tuple[LocationDiagnostic, ...] = ()


@dataclass(frozen=True, slots=True)
class LocationRejection:
    """An extracted metadata record with no usable location, plus diagnostics."""

    unresolved_inputs: tuple[UnresolvedLocationInput, ...] = ()
    coordinate_decodes: int = 0
    direct_matches: int = 0
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
        self.reviewed_mappings = dict(policy.reviewed_mappings)
        self.reviewed_unmapped = frozenset(policy.reviewed_unmapped)

        self.cc = coco.CountryConverter()
        logging.getLogger("country_converter").setLevel(logging.CRITICAL)

        # Cache on this instance so the cache stays small
        # and is freed with the standardizer
        self._country_convert = lru_cache(maxsize=4096)(self._country_convert)
        self._country_to_unregion = lru_cache(maxsize=4096)(self._country_to_unregion)
        self.decode_coordinates = lru_cache(maxsize=4096)(self.decode_coordinates)

        self.stats = {
            "coordinate_decodes": 0,
            "direct_matches": 0,
            "reviewed_mapping_matches": 0,
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
                (LocationDiagnostic.UNMAPPABLE_RESULT,),
            )
        if mapped == match.country:
            return match
        return LocationMatch(mapped, match.sublocation, match.diagnostics)

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
                (LocationDiagnostic.RECOVERABLE_COORDINATE_FAILURE,),
            )
        if raw_country is None:
            return LocationMatch("NA", None)

        country = self._country_converter_lookup(raw_country, "name_short") or raw_country

        self.stats["coordinate_decodes"] += 1
        return LocationMatch(country, city, (LocationDiagnostic.COORDINATE_RESOLUTION,))

    def _try_country_converter(self, val: str) -> LocationMatch | None:
        """
        Run country_converter on a non-coordinate value.

        Returns None on failure so the caller can try the reviewed fallback.
        """
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
        return LocationMatch(country, sublocation, (LocationDiagnostic.DIRECT_RESOLUTION,))

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
        resolution = self._resolve_record(attributes, values)
        published = {
            "unresolved_inputs": resolution.unresolved_inputs,
            "coordinate_decodes": self.stats["coordinate_decodes"] - before["coordinate_decodes"],
            "direct_matches": self.stats["direct_matches"] - before["direct_matches"],
            "reviewed_mapping_matches": (
                self.stats["reviewed_mapping_matches"] - before["reviewed_mapping_matches"]
            ),
            "diagnostics": resolution.diagnostics,
        }
        if resolution.country is None:
            return LocationRejection(**published)
        # loc_UNregion carries the UN region derived from the standardized country.
        return LocationOutcome(
            un_region=self._country_to_unregion(resolution.country),
            country=resolution.country,
            sublocation=resolution.sublocation,
            supporting_pairs=resolution.supporting_pairs,
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
        fallback is tried only when no deterministic route produced a usable INSDC location
        (including when the selected match ended as ``unmappable_result``), and it tries
        every submitted value as a lookup key.
        """
        submitted_pairs = tuple(
            SupportingAttributeValuePair(attribute, value)
            for attribute, value in zip(attributes, values, strict=True)
        )
        pairs = [
            (attribute.strip(), value.strip())
            for attribute, value in zip(attributes, values, strict=True)
            if value.strip()
        ]
        if not pairs:
            return _RecordResolution(None, None, (), (), (LocationDiagnostic.ABSENT_VALUES,))

        # A pair is unusable when neither route matched it to an INSDC location. A pair can
        # be both selectable and unusable: a coordinate that decodes to a non-INSDC country
        # still wins the sublocation preference below.
        selectable: list[LocationMatch] = []
        unusable_pairs: list[tuple[str, str]] = []
        coordinate_failure = False
        for attribute, value in pairs:
            match = self._try_coordinate(value, attribute)
            if match is None:
                match = self._try_country_converter(value)
            if match is not None and (
                LocationDiagnostic.RECOVERABLE_COORDINATE_FAILURE in match.diagnostics
            ):
                coordinate_failure = True
            if match is None or match.country == "NA":
                unusable_pairs.append((attribute, value))
                continue
            selectable.append(match)
            if self._to_insdc(match).country == "NA":
                unusable_pairs.append((attribute, value))

        unmappable_result = False
        if selectable:
            # Prefer matches that include a sublocation (coord-decoded city
            # or "Country:City" sublocation) since they carry more information.
            with_sublocation = [match for match in selectable if match.sublocation]
            selected = self._to_insdc(with_sublocation[0] if with_sublocation else selectable[0])
            if selected.country != "NA":
                unresolved = self._unresolved_inputs(unusable_pairs)
                return _RecordResolution(
                    country=selected.country,
                    sublocation=selected.sublocation,
                    # Deterministic outcomes keep every submitted pair (narrowing to the
                    # winning pair belongs to the decision-provenance workstream).
                    supporting_pairs=submitted_pairs,
                    unresolved_inputs=unresolved,
                    diagnostics=self._record_diagnostics(
                        coordinate_failure=coordinate_failure,
                        unmappable_result=False,
                        resolution=selected.diagnostics,
                        unresolved=unresolved,
                    ),
                )
            unmappable_result = True

        return self._reviewed_fallback(
            pairs,
            unusable_pairs=unusable_pairs,
            coordinate_failure=coordinate_failure,
            unmappable_result=unmappable_result,
        )

    def _reviewed_fallback(
        self,
        pairs: Sequence[tuple[str, str]],
        *,
        unusable_pairs: Sequence[tuple[str, str]],
        coordinate_failure: bool,
        unmappable_result: bool,
    ) -> _RecordResolution:
        """Apply the reviewed geographic-location policy to every submitted value."""
        mapped: list[tuple[tuple[str, str], str]] = []
        reviewed_unmapped_present = False
        for attribute, value in pairs:
            key = normalize_submitted_location_value(value)
            target = self.reviewed_mappings.get(key)
            if target is not None:
                self.stats["reviewed_mapping_matches"] += 1
                mapped.append(((attribute, value), target))
            elif key in self.reviewed_unmapped:
                reviewed_unmapped_present = True

        unresolved = self._unresolved_inputs(unusable_pairs)

        def resolution(
            country: str | None,
            supporting_pairs: tuple[SupportingAttributeValuePair, ...],
            *diagnostics: LocationDiagnostic,
        ) -> _RecordResolution:
            return _RecordResolution(
                country=country,
                # Reviewed mappings resolve to a country or water body only,
                # so no sublocation.
                sublocation=None,
                supporting_pairs=supporting_pairs,
                unresolved_inputs=unresolved,
                diagnostics=self._record_diagnostics(
                    coordinate_failure=coordinate_failure,
                    unmappable_result=unmappable_result,
                    resolution=diagnostics,
                    unresolved=unresolved,
                ),
            )

        if mapped:
            targets = {target for _, target in mapped}
            if len(targets) > 1:
                return resolution(None, (), LocationDiagnostic.REVIEWED_MAPPING_CONFLICT)
            # Unlike deterministic outcomes (which keep every pair), reviewed-mapping
            # outcomes keep only the pairs that matched.
            return resolution(
                targets.pop(),
                tuple(
                    SupportingAttributeValuePair(attribute, value)
                    for (attribute, value), _ in mapped
                ),
                LocationDiagnostic.REVIEWED_MAPPING_RESOLUTION,
            )
        if reviewed_unmapped_present:
            return resolution(None, (), LocationDiagnostic.REVIEWED_UNMAPPED)
        return resolution(None, ())

    def _unresolved_inputs(
        self,
        unusable_pairs: Sequence[tuple[str, str]],
    ) -> tuple[UnresolvedLocationInput, ...]:
        """Filter unusable pairs down to those not already covered by a reviewed section."""
        return tuple(
            UnresolvedLocationInput(attribute, value)
            for attribute, value in unusable_pairs
            if (key := normalize_submitted_location_value(value)) not in self.reviewed_mappings
            and key not in self.reviewed_unmapped
        )

    @staticmethod
    def _record_diagnostics(
        *,
        coordinate_failure: bool,
        unmappable_result: bool,
        resolution: Sequence[LocationDiagnostic],
        unresolved: Sequence[UnresolvedLocationInput],
    ) -> tuple[LocationDiagnostic, ...]:
        """
        Order one record's diagnostics: operational, then selection, then review signals.

        ``unresolved_place`` means at least one value went to the review worklist, so it
        can appear even on a standardized record.
        """
        return (
            *((LocationDiagnostic.RECOVERABLE_COORDINATE_FAILURE,) if coordinate_failure else ()),
            *((LocationDiagnostic.UNMAPPABLE_RESULT,) if unmappable_result else ()),
            *resolution,
            *((LocationDiagnostic.UNRESOLVED_PLACE,) if unresolved else ()),
        )
