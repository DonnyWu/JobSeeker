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
    _SITES,
    _dedupe_across_boards,
    _filter_by_location,
    _filter_by_radius,
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


# ── _filter_by_radius (true geocoded mileage, crosses state lines) ───────────
# Real distances from New York, NY: Newark NJ ~9 mi, Stamford CT ~34 mi,
# Boston MA ~190 mi, Portland ME ~280 mi.
def _radius_frame():
    rows = [
        "New York, NY", "Newark, NJ", "Stamford, CT, US", "Boston, MA",
        "Portland, ME", "Remote", "United States", "Austin, TX",
    ]
    return pd.DataFrame(
        {"location": rows, "title": ["t"] * len(rows), "company": ["c"] * len(rows)}
    )


def test_radius_100mi_crosses_state_lines():
    # 100 mi from NYC pulls in NJ + CT, but not far states (MA/ME/TX).
    kept = set(_filter_by_radius(_radius_frame(), "New York, NY", 100)["location"])
    assert {"New York, NY", "Newark, NJ", "Stamford, CT, US"} <= kept
    assert not ({"Boston, MA", "Portland, ME", "Austin, TX"} & kept)


def test_radius_25mi_tightens():
    # 25 mi keeps nearby Newark (~9 mi) but drops Stamford (~34 mi).
    kept = set(_filter_by_radius(_radius_frame(), "New York, NY", 25)["location"])
    assert "Newark, NJ" in kept
    assert "Stamford, CT, US" not in kept


def test_radius_keeps_remote_regardless():
    assert "Remote" in set(_filter_by_radius(_radius_frame(), "New York, NY", 0)["location"])


def test_radius_drops_ungeocodable_out_of_area():
    # "United States" can't be geocoded and isn't in NY -> dropped (anti-leak).
    assert "United States" not in set(
        _filter_by_radius(_radius_frame(), "New York, NY", 100)["location"]
    )


def test_radius_bare_state_origin_defers_to_text_filter():
    # A radius around a whole state is meaningless; behave exactly like the text
    # filter (which is what pins the Maine/Texas anti-leak behavior).
    df = _frame(_ROWS)
    out_radius = _filter_by_radius(df, "MA", 50)
    out_text = _filter_by_location(df, "MA")
    assert list(out_radius["location"]) == list(out_text["location"])


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


def _boards(mapping, raises=()):
    """Build a jobspy stand-in that answers per board.

    scrape_jobs calls jobspy once per board (so one blocked board can't sink the
    search), so a fake that ignored ``site_name`` would hand back the same rows
    five times over and look like four duplicates. ``mapping`` is {site: frame};
    any site in ``raises`` throws, standing in for Google's RetryError.
    """
    def fake(**kw):
        site = kw.get("site_name", [None])[0]
        if site in raises:
            raise RuntimeError(f"{site} is blocked")
        return mapping.get(site, pd.DataFrame())

    return fake


def test_scrape_jobs_filters_when_not_remote(monkeypatch):
    monkeypatch.setattr("jobspy.scrape_jobs", _boards({"indeed": _fake_jobs()}))
    out = scrape_jobs("engineer", "MA", is_remote=False)
    assert set(out["location"]) == {"Boston, MA", "Remote"}   # Austin, TX dropped
    # Non-kept columns are trimmed; index is reset.
    assert "job_type" not in out.columns and "is_remote" not in out.columns
    assert "title" in out.columns
    assert list(out.index) == list(range(len(out)))


def test_scrape_jobs_forwards_distance_to_jobspy(monkeypatch):
    captured = {}

    def fake(**kw):
        captured.update(kw)
        return _fake_jobs() if kw.get("site_name") == ["indeed"] else pd.DataFrame()

    monkeypatch.setattr("jobspy.scrape_jobs", fake)
    scrape_jobs("engineer", "MA", distance_miles=30)
    assert captured.get("distance") == 30


def test_scrape_jobs_skips_filter_when_remote_only(monkeypatch):
    monkeypatch.setattr("jobspy.scrape_jobs", _boards({"indeed": _fake_jobs()}))
    out = scrape_jobs("engineer", "MA", is_remote=True)
    # Filter is bypassed for Remote-only searches: the out-of-state row passes.
    assert "Austin, TX" in set(out["location"])


@pytest.mark.parametrize("ret", [None, pd.DataFrame()])
def test_scrape_jobs_empty_or_none(monkeypatch, ret):
    monkeypatch.setattr("jobspy.scrape_jobs", lambda **kw: ret)
    assert scrape_jobs("engineer", "MA").empty


# ── One board failing must not sink the whole search ─────────────────────────
def test_scrape_jobs_keeps_other_boards_when_one_raises(monkeypatch):
    """Regression: jobspy collects its own boards with a bare future.result(), so
    Google raising RetryError once rate-limited used to discard every other
    board's results too. Each board is now scraped in isolation."""
    monkeypatch.setattr(
        "jobspy.scrape_jobs",
        _boards({"indeed": _fake_jobs()}, raises=("google", "linkedin")),
    )
    out = scrape_jobs("engineer", "MA", is_remote=True)
    assert len(out) == 3                       # the Indeed rows still came through
    assert set(out.attrs["boards_failed"]) == {"google", "linkedin"}


def test_scrape_jobs_reports_every_board_that_failed(monkeypatch):
    monkeypatch.setattr(
        "jobspy.scrape_jobs", _boards({}, raises=tuple(_SITES))
    )
    out = scrape_jobs("engineer", "MA")
    assert out.empty
    assert set(out.attrs["boards_failed"]) == set(_SITES)


def test_scrape_jobs_reports_no_failures_when_all_boards_answer(monkeypatch):
    monkeypatch.setattr("jobspy.scrape_jobs", _boards({"indeed": _fake_jobs()}))
    out = scrape_jobs("engineer", "MA", is_remote=True)
    assert out.attrs["boards_failed"] == []


# ── _dedupe_across_boards ────────────────────────────────────────────────────
def _multi_board_jobs():
    """One role carried by three boards, plus an unrelated job.

    LinkedIn comes first and carries *no* description, which is what really
    happens: the scraper leaves ``linkedin_fetch_description`` off, so its rows
    arrive without one. The Indeed copy is the richest.
    """
    return pd.DataFrame(
        {
            "title": ["Engineer", "Engineer", "Engineer", "Designer"],
            "company": ["Acme", "ACME", "acme  ", "Acme"],
            "location": ["Boston, MA", "Boston, MA, US", "Boston", "Boston, MA"],
            "site": ["linkedin", "indeed", "google", "indeed"],
            "description": [None, "the full job description", "short", "d"],
            "min_amount": [None, 100000, None, None],
            "job_url": ["u1", "u2", "u3", "u4"],
        }
    )


def test_dedupe_collapses_same_role_across_boards():
    out, merged = _dedupe_across_boards(_multi_board_jobs())
    assert merged == 2
    # The three Acme/Engineer/Boston rows became one; the Designer role survives.
    assert len(out) == 2
    assert set(out["title"]) == {"Engineer", "Designer"}


def test_dedupe_keeps_the_copy_that_has_a_description():
    out, _ = _dedupe_across_boards(_multi_board_jobs())
    eng = out[out["title"] == "Engineer"].iloc[0]
    # Not the LinkedIn row that came first with no description.
    assert eng["site"] == "indeed"
    assert eng["description"] == "the full job description"
    assert eng["min_amount"] == 100000


def test_dedupe_ignores_case_whitespace_and_location_suffix():
    # "Acme"/"ACME"/"acme  " and "Boston, MA"/"Boston, MA, US"/"Boston" are one job.
    out, _ = _dedupe_across_boards(_multi_board_jobs())
    assert len(out[out["title"] == "Engineer"]) == 1


def test_dedupe_is_a_noop_on_a_unique_frame():
    unique = _multi_board_jobs().iloc[[1, 3]].reset_index(drop=True)
    out, merged = _dedupe_across_boards(unique)
    assert merged == 0
    assert len(out) == len(unique)


def test_dedupe_does_not_merge_different_companies_or_cities():
    df = pd.DataFrame(
        {
            "title": ["Engineer", "Engineer", "Engineer"],
            "company": ["Acme", "Globex", "Acme"],
            "location": ["Boston, MA", "Boston, MA", "Austin, TX"],
            "site": ["indeed"] * 3,
            "description": ["a", "b", "c"],
        }
    )
    out, merged = _dedupe_across_boards(df)
    assert merged == 0 and len(out) == 3


def test_dedupe_handles_empty_frame():
    out, merged = _dedupe_across_boards(pd.DataFrame())
    assert merged == 0 and out.empty


def test_dedupe_survives_a_frame_with_no_description_column():
    df = pd.DataFrame(
        {
            "title": ["Engineer", "Engineer"],
            "company": ["Acme", "Acme"],
            "location": ["Boston, MA", "Boston, MA"],
            "site": ["linkedin", "indeed"],
        }
    )
    out, merged = _dedupe_across_boards(df)
    assert merged == 1 and len(out) == 1


# ── scrape_jobs: result depth + dedupe wiring ────────────────────────────────
def test_scrape_jobs_requests_150_per_board_by_default(monkeypatch):
    captured = {}

    def fake(**kw):
        captured.update(kw)
        return pd.DataFrame()

    monkeypatch.setattr("jobspy.scrape_jobs", fake)
    scrape_jobs("engineer", "MA")
    assert captured.get("results_wanted") == 150


def test_scrape_jobs_asks_every_board(monkeypatch):
    asked = []

    def fake(**kw):
        asked.append(kw.get("site_name")[0])
        return pd.DataFrame()

    monkeypatch.setattr("jobspy.scrape_jobs", fake)
    scrape_jobs("engineer", "MA")
    assert sorted(asked) == sorted(_SITES)


def test_scrape_jobs_dedupes_across_boards_and_reports_the_count(monkeypatch):
    def _row(site, description):
        return pd.DataFrame(
            {
                "title": ["Engineer"],
                "company": ["Acme"],
                "location": ["Boston, MA"],
                "site": [site],
                "description": [description],
                "job_url": [f"https://example.test/{site}"],
            }
        )

    # The same posting carried by two boards; only Indeed has the description.
    monkeypatch.setattr(
        "jobspy.scrape_jobs",
        _boards({"linkedin": _row("linkedin", None), "indeed": _row("indeed", "full jd")}),
    )
    out = scrape_jobs("engineer", "MA")
    assert len(out) == 1
    assert out.attrs["duplicates_merged"] == 1
    assert out.iloc[0]["description"] == "full jd"   # richest copy survived


def test_scrape_jobs_reports_zero_merges_when_all_unique(monkeypatch):
    monkeypatch.setattr("jobspy.scrape_jobs", _boards({"indeed": _fake_jobs()}))
    out = scrape_jobs("engineer", "MA", is_remote=True)
    assert out.attrs["duplicates_merged"] == 0
