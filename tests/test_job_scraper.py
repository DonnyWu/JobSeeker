"""Regression tests for Job Search location filtering (src/job_scraper.py).

These pin the behavior behind two bugs that caused out-of-state jobs (e.g. Maine /
Texas) to show up when searching Massachusetts:

1. State-abbreviation matching must be whole-token ("MA" must not match "Miami"),
   and the full-name path ("Massachusetts") must behave identically to "MA".
2. "Remote" must be decided from the location *text* (Remote/Anywhere/WFH), NOT
   jobspy's over-eager ``is_remote`` flag, which had let on-site out-of-state roles
   leak through.

Pure/in-memory: no network, no DB, no GROQ_API_KEY required.
"""

import pandas as pd
import pytest

from src.job_scraper import (
    _build_targets,
    _row_matches,
    _filter_by_location,
    scrape_jobs,
)


# ── _build_targets ───────────────────────────────────────────────────────────
def test_build_targets_abbrev():
    assert _build_targets("MA") == ({"massachusetts"}, {"ma"})


def test_build_targets_full_name_matches_abbrev():
    # Regression: typing the full state name must yield the same targets as the
    # abbreviation (the abbrev was previously added uppercase and never matched).
    assert _build_targets("Massachusetts") == _build_targets("MA")


def test_build_targets_city_and_state():
    names, abbrevs = _build_targets("Boston, MA")
    assert "boston" in names and "massachusetts" in names
    assert abbrevs == {"ma"}


def test_build_targets_empty():
    assert _build_targets("") == (set(), set())


# ── _row_matches (targets from "MA") ─────────────────────────────────────────
_MA_NAMES, _MA_ABBREVS = _build_targets("MA")


@pytest.mark.parametrize(
    "row_loc, expected",
    [
        ("Boston, MA, US", True),
        ("Cambridge, Massachusetts", True),
        ("Portland, ME", False),       # Maine — the original leak
        ("Miami, FL", False),          # must NOT match via the "ma" substring
        ("Maryland", False),           # token mismatch, not Massachusetts
        ("", False),
    ],
)
def test_row_matches(row_loc, expected):
    assert _row_matches(row_loc, _MA_NAMES, _MA_ABBREVS) is expected


# ── _filter_by_location ──────────────────────────────────────────────────────
# (location, is_remote_flag, should_be_kept) for a search of "MA".
_ROWS = [
    ("Boston, MA", False, True),
    ("Cambridge, Massachusetts", False, True),
    ("Worcester, MA", False, True),
    ("Remote", False, True),
    ("Remote - ME", True, True),          # genuinely remote by text -> kept
    ("Anywhere, USA", True, True),
    ("Work from home", False, True),
    ("WFH", False, True),
    ("Portland, ME", False, False),       # Maine on-site -> dropped
    ("Portland, ME", True, False),        # Maine + jobspy false-flag -> dropped
    ("Bangor, Maine", True, False),
    ("Austin, TX", True, False),          # TX false-flag -> dropped
    ("Miami, FL", False, False),
    ("New York, NY", False, False),
    ("United States", False, False),
    ("", False, False),
]


def _frame(rows):
    return pd.DataFrame(
        {
            "location": [r[0] for r in rows],
            "is_remote": [r[1] for r in rows],
            "title": ["t"] * len(rows),
            "company": ["c"] * len(rows),
        }
    )


@pytest.mark.parametrize("location", ["MA", "Massachusetts"])
def test_filter_by_location_ma(location):
    out = _filter_by_location(_frame(_ROWS), location)
    kept = set(zip(out["location"], out["is_remote"]))
    for loc, isr, should_keep in _ROWS:
        assert ((loc, isr) in kept) is should_keep, f"{loc!r} is_remote={isr}"


def test_filter_ignores_is_remote_flag():
    # Two identical out-of-state rows differing only by the jobspy is_remote flag
    # must BOTH be dropped — the flag must not influence the decision.
    df = _frame([("Portland, ME", False, None), ("Portland, ME", True, None)])
    assert _filter_by_location(df, "MA").empty


def test_filter_boston_ma_includes_boston_excludes_out_of_state():
    # Searching "Boston, MA" must surface the Boston job and drop out-of-state
    # roles (Maine/Texas), even when jobspy flags the out-of-state one remote.
    df = _frame(
        [
            ("Boston, MA", False, None),
            ("Portland, ME", False, None),
            ("Austin, TX", True, None),
        ]
    )
    locs = set(_filter_by_location(df, "Boston, MA")["location"])
    assert "Boston, MA" in locs
    assert "Portland, ME" not in locs
    assert "Austin, TX" not in locs


def test_filter_empty_location_returns_all():
    df = _frame(_ROWS)
    assert len(_filter_by_location(df, "")) == len(df)


def test_filter_missing_location_column_returns_unchanged():
    df = pd.DataFrame({"title": ["t"], "company": ["c"]})
    assert _filter_by_location(df, "MA").equals(df)


# ── scrape_jobs (jobspy monkeypatched — no network) ──────────────────────────
def _fake_jobs():
    return pd.DataFrame(
        {
            "title": ["A", "B", "C"],
            "company": ["x", "y", "z"],
            "location": ["Boston, MA", "Austin, TX", "Remote"],
            "is_remote": [False, True, True],   # not in `keep` -> dropped from output
            "job_type": ["fulltime"] * 3,       # not in `keep` -> dropped from output
            "description": ["d"] * 3,
            "site": ["indeed"] * 3,
            "job_url": ["u"] * 3,
        }
    )


def test_scrape_jobs_filters_when_not_remote(monkeypatch):
    monkeypatch.setattr("jobspy.scrape_jobs", lambda **kw: _fake_jobs())
    out = scrape_jobs("engineer", "MA", is_remote=False)
    assert set(out["location"]) == {"Boston, MA", "Remote"}   # Austin, TX dropped
    # Non-kept columns are trimmed; index is reset.
    assert "job_type" not in out.columns and "is_remote" not in out.columns
    assert "title" in out.columns
    assert list(out.index) == list(range(len(out)))


def test_scrape_jobs_skips_filter_when_remote_only(monkeypatch):
    monkeypatch.setattr("jobspy.scrape_jobs", lambda **kw: _fake_jobs())
    out = scrape_jobs("engineer", "MA", is_remote=True)
    # Filter is bypassed for Remote-only searches: the out-of-state row passes.
    assert "Austin, TX" in set(out["location"])


@pytest.mark.parametrize("ret", [None, pd.DataFrame()])
def test_scrape_jobs_empty_or_none(monkeypatch, ret):
    monkeypatch.setattr("jobspy.scrape_jobs", lambda **kw: ret)
    assert scrape_jobs("engineer", "MA").empty
