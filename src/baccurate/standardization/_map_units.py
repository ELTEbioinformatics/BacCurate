"""Resolve a coordinate to the Natural Earth map unit that contains it."""

import json
from functools import cache

from shapely import STRtree, points
from shapely.geometry import shape

from baccurate.paths import DEFAULT_MAP_UNIT_BOUNDARIES

# Map unit with no ISO code, such as "Somaliland".
_NO_ISO_CODE = "-99"


@cache
def _map_unit_index() -> tuple[STRtree, tuple[str, ...]]:
    """
    Load the map unit boundaries into a spatial index, with their ISO codes in a parallel
    tuple.
    """
    with DEFAULT_MAP_UNIT_BOUNDARIES.open("r", encoding="utf-8") as handle:
        features = json.load(handle)["features"]
    boundaries = [shape(feature["geometry"]) for feature in features]
    codes = tuple(feature["properties"]["ISO_A2_EH"] for feature in features)
    return STRtree(boundaries), codes


def containing_map_unit_code(coordinate: tuple[float, float]) -> str | None:
    """
    Return the ISO alpha-2 code of the map unit that contains a (latitude, longitude) pair.

    The result is None in two conditions:
      (1) no map unit contains the point, which is most often water
      (2) the map unit that contains the point carries no code.
    """
    latitude, longitude = coordinate
    index, codes = _map_unit_index()
    matched = index.query(points(longitude, latitude), predicate="intersects")
    if matched.size == 0:
        return None
    code = codes[min(matched)]
    return None if code == _NO_ISO_CODE else code
