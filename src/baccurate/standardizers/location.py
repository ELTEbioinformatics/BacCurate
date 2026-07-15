"""
Map location annotations from sample metadata to standardized
country and continent names. Falls back to an LLM for values that country_converter and
reverse_geocode cannot resolve.

See location.md for the documentation.
"""

import csv
import json
import logging
import re
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import country_converter as coco
import openai
import reverse_geocode

from baccurate.paths import DEFAULT_GEO_LOC_LIST, DEFAULT_LOC_CACHE_DB, LOC_OUTPUT
from baccurate.utils.args import create_arg_parser
from baccurate.utils.cache import SQLiteKVCache
from baccurate.utils.config import load_config
from baccurate.utils.llm import load_llm_client
from baccurate.utils.logging import setup_standardizer_logging
from baccurate.utils.progress import count_tsv_rows, make_inner_bar
from baccurate.utils.text import split_pipe_separated

logger = logging.getLogger(__name__)

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


@dataclass(frozen=True, slots=True)
class LocationMatch:
    """Standardization result for one record."""

    country: str
    continent: str
    sublocation: str | None
    used_llm: bool = False


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

    def get(self, context_string: str) -> tuple[str, str] | None:
        self.cursor.execute(
            "SELECT country, continent FROM cache WHERE hash_id=?",
            (self._sha256(context_string),),
        )
        row = self.cursor.fetchone()
        return (row[0], row[1]) if row else None

    def set(self, context_string: str, country: str, continent: str) -> None:
        self.cursor.execute(
            "INSERT OR REPLACE INTO cache (hash_id, country, continent) VALUES (?, ?, ?)",
            (self._sha256(context_string), country, continent),
        )
        self.conn.commit()


# --- Main class ---


class LocationStandardizer:
    def __init__(self, config_path: Path | str) -> None:
        self.config = load_config(config_path)
        self.coordinate_attributes = set(self.config.get("coordinate_attributes", []))
        self.wait_s = int(self.config.get("wait_s", 5))
        self.insdc_map = dict(self.config.get("insdc_country_map", {}))
        geo_loc_path = self.config.get("geo_loc_list_path", DEFAULT_GEO_LOC_LIST)
        self.insdc_names = self._load_insdc_names(geo_loc_path)
        self.llm_system_prompt = self.config.get("llm_system_prompt")
        self.llm_user_prompt_template = self.config.get("llm_user_prompt_template")

        self.client, self.llm_model = load_llm_client()

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
            logger.warning(
                "Country %r (from %r) not in INSDC vocabulary - setting NA",
                mapped, match.country,
            )
            return LocationMatch("NA", "NA", match.sublocation, match.used_llm)
        if mapped == match.country:
            return match
        return LocationMatch(mapped, match.continent, match.sublocation, match.used_llm)

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
        try:
            info = reverse_geocode.get((lat, lon))
            return info.get("country"), info.get("city")
        except Exception as e:
            logger.error("Reverse geocoding failed for (%s, %s): %s", lat, lon, e)
            return None, None

    def _try_coordinate(self, val: str, attr: str) -> LocationMatch | None:
        """Decode and standardize if the value or attribute looks like a coordinate."""
        if not (_is_coordinate(val) or attr in self.coordinate_attributes):
            return None

        raw_country, city = self.decode_coordinates(val)
        if raw_country is None:
            logger.debug("Coordinate decode failed for %r", val)
            return LocationMatch("NA", "NA", None)

        cc_country = _extract_string(
            self.cc.convert(names=raw_country, to="name_short", not_found="NA")
        )
        country = cc_country if cc_country and cc_country != "NA" else raw_country
        continent = (
            _extract_string(self.cc.convert(names=country, to="continent", not_found="NA")) or "NA"
        )

        self.stats["coordinate_decodes"] += 1
        return LocationMatch(country, continent, city)

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
        return LocationMatch(country, continent, sublocation)

    # --- LLM fallback ---

    def _call_llm(
        self,
        system_prompt: str | None,
        user_prompt: str,
        timeout: int = 30,
    ) -> dict | list | None:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})

        try:
            response = self.client.chat.completions.create(
                model=self.llm_model,
                messages=messages,
                temperature=0,
                timeout=timeout,
                seed=100,
            )
        except openai.APITimeoutError as e:
            logger.warning("LLM API request timed out: %s", e)
            return None
        except openai.APIError as e:
            logger.warning("LLM API returned an APIError: %s", e)
            return None
        except Exception as e:
            logger.error("Unexpected error during LLM API request: %s", e)
            return None

        if response is None:
            logger.warning("LLM API call returned a None response object.")
            return None

        time.sleep(self.wait_s)

        try:
            content = response.choices[0].message.content
        except (AttributeError, IndexError) as e:
            logger.error("Could not extract content from response: %s", e)
            return None

        if not content:
            logger.warning("LLM returned empty content.")
            return None

        content = content.strip()
        logger.debug("LLM response: %s", content)

        try:
            return json.loads(content)
        except json.JSONDecodeError:
            pass

        match = re.search(r"(\{.*?})|(\[.*?])", content, re.DOTALL)
        if match is None:
            logger.warning("No JSON object or array in LLM response: %r", content)
            return None

        json_str = match.group(1) or match.group(2)
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            logger.warning("Failed to parse extracted JSON: %r", json_str)
            return None

    def _llm_fallback(self, context_string: str) -> tuple[str, str]:
        cached = self.cache.get(context_string)
        if cached is not None:
            self.stats["cache_hits"] += 1
            logger.debug("LLM cache hit for %r", context_string)
            return cached

        if self.client is None:
            return "NA", "NA"

        logger.debug("LLM call for context %r", context_string)
        user_prompt = self.llm_user_prompt_template.format(attr_val_pairs=context_string)
        parsed = self._call_llm(self.llm_system_prompt, user_prompt)
        self.stats["llm_calls"] += 1

        if parsed is None:
            self.cache.set(context_string, "NA", "NA")
            return "NA", "NA"

        llm_country = _extract_string(parsed.get("country")) or "NA"
        llm_continent = _extract_string(parsed.get("continent")) or "NA"

        # Standardize the LLM's country through cc
        cc_country = _extract_string(
            self.cc.convert(names=llm_country, to="name_short", not_found="NA")
        )
        if cc_country and cc_country != "NA":
            cc_continent = _extract_string(
                self.cc.convert(names=cc_country, to="continent", not_found="NA")
            )
            result = (cc_country, cc_continent or llm_continent or "NA")
        else:
            logger.debug(
                "country_converter failed on LLM response %r - using LLM values directly",
                llm_country,
            )
            result = (llm_country, llm_continent)

        self.cache.set(context_string, *result)
        return result

    # --- Per-record dispatch ---

    def find_best_location(
        self,
        accession: str,
        attr_str: str,
        val_str: str,
    ) -> LocationMatch:
        attrs = split_pipe_separated(attr_str)
        vals = split_pipe_separated(val_str)

        if not vals:
            return LocationMatch("NA", "NA", None)

        valid_matches: list[LocationMatch] = []
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
            return self._to_insdc(with_subloc[0] if with_subloc else valid_matches[0])

        if not unmatched_pairs:
            return LocationMatch("NA", "NA", None)

        context = " ".join(f"{a}={v}" for a, v in unmatched_pairs)
        country, continent = self._llm_fallback(context)
        return self._to_insdc(LocationMatch(country, continent, None, used_llm=True))

    # --- File processing ---

    def process_file(
        self,
        input_path: Path,
        output_path: Path,
        pathogen: str | None = None,
        disable_progress: bool = False,
    ) -> None:
        output_header = [
            "accession",
            "loc_attr_orig",
            "loc_val_orig",
            "loc_continent",
            "loc_UNregion",
            "loc_country",
            "loc_other",
        ]

        total = count_tsv_rows(input_path)
        bar_desc = f"loc [{pathogen}]" if pathogen else "loc"
        records_processed = 0

        try:
            with (
                input_path.open("r", encoding="utf-8", newline="") as infile,
                output_path.open("w", encoding="utf-8", newline="") as outfile,
                make_inner_bar(total, bar_desc, disable=disable_progress) as bar,
            ):
                reader = csv.DictReader(infile, delimiter="\t")
                writer = csv.writer(outfile, delimiter="\t")
                writer.writerow(output_header)

                for row in reader:
                    if pathogen and row.get("pathogen") != pathogen:
                        bar.update(1)
                        continue

                    accession = row.get("accession", "")
                    attr_str = (row.get("loc_attr_orig") or "").strip()
                    val_str = (row.get("loc_val_orig") or "").strip()

                    match = self.find_best_location(accession, attr_str, val_str)

                    if match.country == "NA":
                        bar.update(1)
                        continue

                    writer.writerow(
                        [
                            accession,
                            attr_str,
                            val_str,
                            match.continent,
                            self._country_to_unregion(match.country),
                            match.country,
                            match.sublocation or "NA",
                        ]
                    )
                    records_processed += 1
                    bar.update(1)

            logger.info(
                "Processed %d records: %d coordinate decodes, %d direct matches, "
                "%d cache hits, %d LLM calls.",
                records_processed,
                self.stats["coordinate_decodes"],
                self.stats["direct_matches"],
                self.stats["cache_hits"],
                self.stats["llm_calls"],
            )
        finally:
            self.cache.close()


def main(
    input_path: Path,
    output_path: Path,
    config_path: Path,
    log_level: str = "INFO",
    pathogen: str | None = None,
    disable_progress: bool = False,
) -> None:
    setup_standardizer_logging(logger, output_path, "loc_standardized", log_level)

    standardizer = LocationStandardizer(config_path)
    standardizer.process_file(
        input_path, output_path, pathogen=pathogen, disable_progress=disable_progress
    )


if __name__ == "__main__":
    parser = create_arg_parser(
        description="Standardizes location values from a TSV file.",
        default_config_path="config/location.yaml",
    )
    args = parser.parse_args()

    in_path = Path(args.input_file)
    out_path = Path(args.output_dir) / LOC_OUTPUT
    cfg_path = Path(args.config)

    main(in_path, out_path, cfg_path, args.log_level)
