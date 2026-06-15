"""Unit tests for offline geocoding + great-circle distance (src/geo.py).

Uses the bundled geonamescache dataset — no network, no DB, no API key.
"""

import pytest

from src.geo import geocode, haversine_miles, _split_city_state


# ── _split_city_state ────────────────────────────────────────────────────────
@pytest.mark.parametrize(
    "text, expected",
    [
        ("New York, NY", ("new york", "NY")),
        ("Boston, MA, US", ("boston", "MA")),
        ("Stamford, Connecticut", ("stamford", "CT")),
        ("MA", None),                 # bare state -> no city
        ("Massachusetts", None),      # bare state name -> no city
        ("United States", None),
        ("Remote", None),
        ("", None),
    ],
)
def test_split_city_state(text, expected):
    assert _split_city_state(text) == expected


# ── geocode ──────────────────────────────────────────────────────────────────
def test_geocode_known_city():
    pt = geocode("New York, NY")
    assert pt is not None
    lat, lon = pt
    assert 40 < lat < 41 and -75 < lon < -73


@pytest.mark.parametrize("text", ["United States", "Remote", "MA", ""])
def test_geocode_ungeocodable_returns_none(text):
    assert geocode(text) is None


# ── haversine_miles ──────────────────────────────────────────────────────────
def test_haversine_zero_for_same_point():
    assert haversine_miles((40.0, -74.0), (40.0, -74.0)) == pytest.approx(0, abs=1e-6)


def test_haversine_nyc_to_newark_is_small():
    d = haversine_miles(geocode("New York, NY"), geocode("Newark, NJ"))
    assert 5 < d < 12


def test_haversine_nyc_to_boston_is_about_190mi():
    d = haversine_miles(geocode("New York, NY"), geocode("Boston, MA"))
    assert 180 < d < 200
