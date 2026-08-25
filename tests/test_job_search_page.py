"""Pagination tests for the Job Search page (pages/3_Job_Search.py).

Driven through Streamlit's AppTest harness so these exercise the real page, not a
reimplementation of its logic. Results are pre-seeded into session state, so no
network scrape and no Groq call happens: the page only scrapes under the Search
button, which is never clicked here.

Each test runs against a throwaway SQLite DB (monkeypatched ENGINE on a tmp file)
so the developer's real data/jobseeker.db is never touched.
"""

import pandas as pd
import pytest
from sqlalchemy import create_engine
from streamlit.testing.v1 import AppTest

import src.profile_manager as pm

PAGE = "pages/3_Job_Search.py"


@pytest.fixture
def db(tmp_path, monkeypatch):
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    monkeypatch.setattr(pm, "ENGINE", engine)
    pm.init_db()
    return engine


def _results(n: int, score: int = 90) -> pd.DataFrame:
    """n distinct scored jobs, all above the default min-score slider."""
    return pd.DataFrame(
        {
            "title": [f"Engineer {i}" for i in range(n)],
            "company": [f"Company {i}" for i in range(n)],
            "location": ["Boston, MA"] * n,
            "site": ["indeed"] * n,
            "description": ["a job description"] * n,
            "job_url": [f"https://example.test/{i}" for i in range(n)],
            "match_score": [score] * n,
            "match_reason": ["reason"] * n,
            "jd_flags": [[] for _ in range(n)],
        }
    )


def _run(df: pd.DataFrame, **state) -> AppTest:
    at = AppTest.from_file(PAGE, default_timeout=60)
    at.session_state["results_df"] = df
    at.session_state["scored"] = True
    for k, v in state.items():
        at.session_state[k] = v
    at.run()
    assert not at.exception, at.exception
    return at


def _cards(at: AppTest) -> int:
    """One 'Auto-Apply' button is rendered per job card."""
    return len([b for b in at.button if b.key and b.key.startswith("apply_")])


def test_first_page_shows_exactly_ten_of_twenty_five(db):
    at = _run(_results(25))
    assert _cards(at) == 10


def test_shorter_than_one_page_renders_everything_without_pager(db):
    at = _run(_results(7))
    assert _cards(at) == 7
    # No pager is drawn when the whole result set fits on one page.
    assert not [b for b in at.button if b.key and b.key.startswith("page_")]


def test_pager_is_rendered_above_and_below_when_paged(db):
    at = _run(_results(25))
    keys = {b.key for b in at.button if b.key and b.key.startswith("page_")}
    assert keys == {"page_prev_top", "page_next_top", "page_prev_bottom", "page_next_bottom"}


def test_next_advances_a_page(db):
    at = _run(_results(25))
    at.button(key="page_next_top").click().run()
    assert at.session_state["results_page"] == 2
    assert _cards(at) == 10


def test_last_page_holds_the_remainder(db):
    at = _run(_results(25), results_page=3)
    assert at.session_state["results_page"] == 3
    assert _cards(at) == 5


def test_prev_disabled_on_first_page_and_next_on_last(db):
    at = _run(_results(25))
    assert at.button(key="page_prev_top").disabled
    assert not at.button(key="page_next_top").disabled

    at = _run(_results(25), results_page=3)
    assert not at.button(key="page_prev_top").disabled
    assert at.button(key="page_next_top").disabled


def test_page_is_clamped_when_the_list_shrinks(db):
    # Landing past the end (e.g. after raising the min-score slider) must clamp to
    # the last real page rather than rendering an empty list.
    at = _run(_results(25), results_page=99)
    assert at.session_state["results_page"] == 3
    assert _cards(at) == 5


def test_pages_stay_full_when_hiding_applied(db):
    """The applied filter must run before the page slice.

    It used to `continue` inside the render loop, which — once slicing existed —
    would leave short pages. Marking 5 of 25 applied must give 10 cards, not 8.
    """
    jobs = _results(25)
    for _, row in jobs.head(5).iterrows():
        pm.mark_job_applied(
            {
                "title": row["title"],
                "company": row["company"],
                "location": row["location"],
                "url": row["job_url"],
            }
        )
    assert len(pm.get_applied_keys()) == 5

    at = AppTest.from_file(PAGE, default_timeout=60)
    at.session_state["results_df"] = jobs
    at.session_state["scored"] = True
    at.run()
    assert not at.exception, at.exception
    at.radio(key=at.radio[0].key).set_value("Hide applied").run()

    assert _cards(at) == 10


def test_duplicate_caption_shows_when_boards_were_merged(db):
    at = _run(_results(12), duplicates_merged=4)
    assert any("4 duplicate posting(s) merged" in c.value for c in at.caption)


def test_no_duplicate_caption_when_nothing_was_merged(db):
    at = _run(_results(12), duplicates_merged=0)
    assert not any("duplicate posting" in c.value for c in at.caption)


def test_blocked_board_is_named(db):
    at = _run(_results(12), boards_failed=["google"])
    assert any("No results from google" in c.value for c in at.caption)


def test_no_board_warning_when_every_board_answered(db):
    at = _run(_results(12), boards_failed=[])
    assert not any("No results from" in c.value for c in at.caption)


# ── A search that finds nothing must not leave the last one on screen ────────
def _click_search(at: AppTest, query: str = "engineer") -> AppTest:
    at.text_input(key="search_query").set_value(query)
    next(b for b in at.button if b.label == "Search").click().run()
    assert not at.exception, at.exception
    return at


def _empty_scrape(monkeypatch, **attrs):
    """Stand in for a search that comes back with nothing."""
    empty = pd.DataFrame()
    empty.attrs.update({"duplicates_merged": 0, "boards_failed": [], **attrs})
    monkeypatch.setattr("src.job_scraper.scrape_jobs", lambda *a, **k: empty)


def test_empty_search_clears_the_previous_results(db, monkeypatch):
    """Regression: the empty branch used to skip updating results_df, so the old
    search's job cards kept rendering underneath the new search's "No jobs found"
    warning."""
    at = _run(_results(25))
    assert _cards(at) == 10          # the previous search is on screen

    _empty_scrape(monkeypatch)
    _click_search(at)

    assert _cards(at) == 0
    assert any("No jobs found" in w.value for w in at.warning)


def test_empty_search_does_not_leave_stale_captions(db, monkeypatch):
    """The caption writes happen before the empty check, so they describe the new
    search. Rendering them over the old search's cards claimed, for example, that
    google was blocked on a search whose results predate that failure."""
    at = _run(_results(25), duplicates_merged=20, boards_failed=[])
    assert any("20 duplicate posting(s) merged" in c.value for c in at.caption)

    _empty_scrape(monkeypatch, boards_failed=["google"])
    _click_search(at)

    assert not any("duplicate posting" in c.value for c in at.caption)
    assert not any("No results from" in c.value for c in at.caption)


def test_empty_search_resets_the_page_number(db, monkeypatch):
    """A stale list could otherwise resume on whatever page you were last on."""
    at = _run(_results(25), results_page=3)
    assert at.session_state["results_page"] == 3

    _empty_scrape(monkeypatch)
    _click_search(at)

    assert at.session_state["results_page"] == 1
    assert at.session_state["results_df"].empty
