"""Offline US geocoding + great-circle distance.

Used by the Job Search radius filter to keep jobs within N miles of the typed
location, even across state lines. City -> (lat, lon) lookups come from the
bundled ``geonamescache`` dataset (no network), so this module is import-safe in
tests and offline.

Locations the dataset can't resolve (bare states, "United States", "Remote",
small towns not in the dataset) return ``None`` from :func:`geocode`; callers
fall back to text-based in-state matching for those.
"""

from __future__ import annotations

import math

# Abbreviation <-> full name for all US states + DC. Canonical home for these
# maps; ``job_scraper`` imports them so location parsing stays in one place.
_STATE_ABBR = {
    "AL": "Alabama", "AK": "Alaska", "AZ": "Arizona", "AR": "Arkansas",
    "CA": "California", "CO": "Colorado", "CT": "Connecticut", "DE": "Delaware",
    "DC": "District of Columbia", "FL": "Florida", "GA": "Georgia", "HI": "Hawaii",
    "ID": "Idaho", "IL": "Illinois", "IN": "Indiana", "IA": "Iowa", "KS": "Kansas",
    "KY": "Kentucky", "LA": "Louisiana", "ME": "Maine", "MD": "Maryland",
    "MA": "Massachusetts", "MI": "Michigan", "MN": "Minnesota", "MS": "Mississippi",
    "MO": "Missouri", "MT": "Montana", "NE": "Nebraska", "NV": "Nevada",
    "NH": "New Hampshire", "NJ": "New Jersey", "NM": "New Mexico", "NY": "New York",
    "NC": "North Carolina", "ND": "North Dakota", "OH": "Ohio", "OK": "Oklahoma",
    "OR": "Oregon", "PA": "Pennsylvania", "RI": "Rhode Island", "SC": "South Carolina",
    "SD": "South Dakota", "TN": "Tennessee", "TX": "Texas", "UT": "Utah",
    "VT": "Vermont", "VA": "Virginia", "WA": "Washington", "WV": "West Virginia",
    "WI": "Wisconsin", "WY": "Wyoming",
}
_STATE_NAME = {name.lower(): abbr for abbr, name in _STATE_ABBR.items()}

# Comma parts that name the country rather than a city/state.
_COUNTRY_TOKENS = {"us", "usa", "u.s.", "u.s.a.", "united states", "united states of america"}

# Lazily-built {(city_lower, state_abbr): (lat, lon)}; see _city_index().
_INDEX: dict[tuple[str, str], tuple[float, float]] | None = None

EARTH_RADIUS_MILES = 3958.7613


def _city_index() -> dict[tuple[str, str], tuple[float, float]]:
    """Build (and cache) a {(city, state) -> (lat, lon)} map of US cities.

    Each city is indexed by its primary name *and* its alternate names (so
    "New York, NY" resolves to the "New York City" record). When two records
    share a (name, state) key, the more-populous one wins.
    """
    global _INDEX
    if _INDEX is not None:
        return _INDEX

    import geonamescache

    index: dict[tuple[str, str], tuple[float, float]] = {}
    pop_at_key: dict[tuple[str, str], int] = {}
    cities = geonamescache.GeonamesCache().get_cities()
    for c in cities.values():
        if c.get("countrycode") != "US":
            continue
        state = c.get("admin1code")
        if state not in _STATE_ABBR:
            continue
        try:
            coord = (float(c["latitude"]), float(c["longitude"]))
        except (KeyError, TypeError, ValueError):
            continue
        pop = c.get("population") or 0
        names = [c["name"], *c.get("alternatenames", [])]
        for nm in names:
            key = (nm.strip().lower(), state)
            if not key[0]:
                continue
            # Keep the highest-population city for any colliding (name, state).
            if key not in pop_at_key or pop > pop_at_key[key]:
                index[key] = coord
                pop_at_key[key] = pop

    _INDEX = index
    return _INDEX


def _split_city_state(location: str) -> tuple[str, str] | None:
    """Parse ``"City, ST"`` / ``"City, State, US"`` into (city_lower, state_abbr).

    Returns ``None`` when no (city, state) pair can be identified — e.g. a bare
    state ("MA"), a country ("United States"), "Remote", or empty text.
    """
    if not location:
        return None
    parts = [p.strip() for p in location.split(",") if p.strip()]
    # Need at least "City, State" — a lone token is a bare city or state we can't
    # pin to coordinates (and a city like "New York" is itself a state name, so
    # we must take the first part as the city, not scan it for a state).
    if len(parts) < 2:
        return None

    city = parts[0].lower()
    if city in _COUNTRY_TOKENS:
        return None

    state: str | None = None
    for part in parts[1:]:
        pl = part.lower()
        if len(part) == 2 and part.upper() in _STATE_ABBR:
            state = part.upper()
            break
        if pl in _STATE_NAME:
            state = _STATE_NAME[pl]
            break

    if state is None:
        return None
    return city, state


def geocode(location: str) -> tuple[float, float] | None:
    """Return (lat, lon) for a ``"City, ST"`` location, or ``None`` if unknown."""
    parsed = _split_city_state(location or "")
    if parsed is None:
        return None
    return _city_index().get(parsed)


def haversine_miles(a: tuple[float, float], b: tuple[float, float]) -> float:
    """Great-circle distance in miles between two (lat, lon) points."""
    lat1, lon1 = math.radians(a[0]), math.radians(a[1])
    lat2, lon2 = math.radians(b[0]), math.radians(b[1])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    h = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 2 * EARTH_RADIUS_MILES * math.asin(math.sqrt(h))
