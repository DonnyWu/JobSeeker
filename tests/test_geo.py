"""Unit tests for offline geocoding + great-circle distance (src/geo.py).

Uses the bundled geonamescache dataset — no network, no DB, no API key.
"""

import pytest

from src.geo import (
    city_suggestions,
    format_locations,
    geocode,
    haversine_miles,
    parse_locations,
    _split_city_state,
)


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


# ── parse_locations / format_locations ───────────────────────────────────────
def test_parse_locations_splits_on_newlines():
    assert parse_locations("New York, NY\nBoston, MA\nSan Francisco, CA") == [
        "New York, NY",
        "Boston, MA",
        "San Francisco, CA",
    ]


def test_parse_locations_keeps_a_legacy_single_value_whole():
    """The pre-multi-location format was one location in one cell.

    Splitting on commas would tear "Boston, MA" into "Boston" and "MA" — two
    bogus chips — for every database saved before multi-location search existed.
    Newline separation is what keeps that value intact.
    """
    assert parse_locations("Boston, MA") == ["Boston, MA"]


def test_parse_locations_accepts_a_list():
    assert parse_locations(["New York, NY", "Boston, MA"]) == [
        "New York, NY",
        "Boston, MA",
    ]


@pytest.mark.parametrize("value", [None, "", "   ", "\n\n", []])
def test_parse_locations_empty(value):
    assert parse_locations(value) == []


def test_parse_locations_strips_and_drops_blanks():
    assert parse_locations("  Boston, MA  \n\n  NYC \n") == ["Boston, MA", "NYC"]


def test_parse_locations_collapses_case_insensitive_duplicates():
    """Keeps the first spelling the user typed."""
    assert parse_locations("Boston, MA\nboston, ma\nNYC") == ["Boston, MA", "NYC"]


def test_format_locations_round_trips():
    locs = ["New York, NY", "Boston, MA", "San Francisco, CA"]
    assert parse_locations(format_locations(locs)) == locs


def test_format_locations_of_nothing_is_empty_string():
    assert format_locations([]) == ""


def test_geocode_is_cached():
    """The radius filter geocodes every row once per searched location, so the
    same text gets looked up repeatedly within a single search."""
    geocode.cache_clear()
    first = geocode("Boston, MA")
    assert geocode("Boston, MA") is first
    assert geocode.cache_info().hits >= 1


# ── city_suggestions (the Location box's type-ahead) ─────────────────────────
def test_suggestions_are_city_comma_state_labels():
    assert "Boston, MA" in city_suggestions()


def test_suggestions_include_smaller_metro_towns():
    """Not just the big metros — commuter towns are where people actually search."""
    assert "Braintree, MA" in city_suggestions()


def test_suggestions_are_ordered_by_population():
    """So typing "bo" leads with Boston rather than an alphabetical accident."""
    cities = city_suggestions()
    assert cities.index("Boston, MA") < cities.index("Braintree, MA")
    matches = [c for c in cities if c.lower().startswith("bo")]
    assert matches[0] == "Boston, MA"


def test_suggestions_disambiguate_same_named_cities():
    """Quincy exists in both MA and IL; the state suffix keeps them apart."""
    cities = city_suggestions()
    assert "Quincy, MA" in cities and "Quincy, IL" in cities


def test_suggestions_have_no_duplicates():
    cities = city_suggestions()
    assert len(cities) == len({c.lower() for c in cities})


def test_every_suggestion_geocodes():
    """A suggestion that can't resolve to coordinates would silently downgrade the
    radius filter to in-state text matching for anyone who picked it."""
    unresolvable = [c for c in city_suggestions() if geocode(c) is None]
    assert unresolvable == []


def test_suggestions_are_cached_and_immutable():
    assert city_suggestions() is city_suggestions()
    assert isinstance(city_suggestions(), tuple)
