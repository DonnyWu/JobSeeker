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


# ── Location chips ───────────────────────────────────────────────────────────
def _prefs(location: str) -> dict:
    return {
        "query": "engineer",
        "location": location,
        "time_filter": "Last 24 hours",
        "is_remote": 0,
        "min_score": 50,
        "distance": 50,
    }


def _fresh(db_=None) -> AppTest:
    at = AppTest.from_file(PAGE, default_timeout=60)
    at.run()
    assert not at.exception, at.exception
    return at


def test_saved_locations_come_back_as_chips(db):
    pm.save_search_prefs(_prefs("New York, NY\nBoston, MA"))
    at = _fresh()
    assert at.multiselect(key="search_locations").value == ["New York, NY", "Boston, MA"]


def test_a_legacy_single_location_loads_as_one_chip(db):
    """Databases saved before multi-location search hold one location in the cell.

    "Boston, MA" must come back as a single chip — splitting on the comma would
    turn every pre-existing save into two bogus chips, "Boston" and "MA".
    """
    pm.save_search_prefs(_prefs("Boston, MA"))
    at = _fresh()
    assert at.multiselect(key="search_locations").value == ["Boston, MA"]


def test_no_saved_locations_means_no_chips(db):
    assert _fresh().multiselect(key="search_locations").value == []


def test_clear_all_empties_the_chips_and_persists(db):
    pm.save_search_prefs(_prefs("New York, NY\nBoston, MA"))
    at = _fresh()
    at.button(key="clear_locations").click().run()
    assert not at.exception, at.exception
    assert at.multiselect(key="search_locations").value == []
    assert pm.get_search_prefs()["location"] == ""


def test_clear_all_is_disabled_when_there_is_nothing_to_clear(db):
    assert _fresh().button(key="clear_locations").disabled


def test_the_box_suggests_cities_as_you_type(db):
    """The box is a type-ahead: it filters a list of US cities by what you type.

    Without a populated option list there is nothing to suggest, which is how it
    behaved when the options were just the chips already chosen.
    """
    at = _fresh()
    options = list(at.multiselect(key="search_locations").proto.options)
    assert len(options) > 1000
    assert "Boston, MA" in options
    assert "Braintree, MA" in options       # small metro town, still offered


def test_chips_lead_the_option_list(db):
    """Streamlit drops a selected value that is not among the options, so the
    chips have to come first — including free-typed ones no city list contains."""
    pm.save_search_prefs(_prefs("Greater Boston Area"))
    at = _fresh()
    options = list(at.multiselect(key="search_locations").proto.options)
    assert options[0] == "Greater Boston Area"
    assert at.multiselect(key="search_locations").value == ["Greater Boston Area"]


def test_a_region_no_city_list_contains_can_still_be_typed(db):
    """accept_new_options is what keeps this a text box rather than a dropdown."""
    pm.save_search_prefs(_prefs("Boston, MA"))
    at = _fresh()
    assert at.multiselect(key="search_locations").proto.accept_new_options

    at.multiselect(key="search_locations").set_value(
        ["Boston, MA", "Greater Boston Area"]
    ).run()
    assert not at.exception, at.exception
    assert pm.get_search_prefs()["location"] == "Boston, MA\nGreater Boston Area"


def test_picking_a_suggestion_persists_it(db):
    pm.save_search_prefs(_prefs("Boston, MA"))
    at = _fresh()
    at.multiselect(key="search_locations").set_value(
        ["Boston, MA", "San Francisco, CA"]
    ).run()
    assert not at.exception, at.exception
    assert pm.get_search_prefs()["location"] == "Boston, MA\nSan Francisco, CA"


def test_the_chip_box_prompts_for_another_region(db):
    """Matches the design: the empty input reads "Enter another region"."""
    assert _fresh().multiselect(key="search_locations").proto.placeholder == (
        "Enter another region"
    )


def test_removing_one_chip_leaves_the_others(db):
    pm.save_search_prefs(_prefs("New York, NY\nBoston, MA\nSan Francisco, CA"))
    at = _fresh()
    at.multiselect(key="search_locations").set_value(
        ["New York, NY", "San Francisco, CA"]
    ).run()
    assert not at.exception, at.exception
    assert pm.get_search_prefs()["location"] == "New York, NY\nSan Francisco, CA"


# ── Scoring a batch at a time ────────────────────────────────────────────────
def _fake_scoring(monkeypatch):
    """Stand in for the résumé lookup and Groq scoring — no DB row, no API key.

    rank_jobs really does sort what it is handed, and the append-at-the-bottom
    guarantee depends on that, so the stand-in sorts too.
    """
    monkeypatch.setattr(pm, "get_latest_resume", lambda: {"skills": ["python"]})
    monkeypatch.setattr(
        "src.job_matcher.rank_jobs",
        lambda df, resume: df.sort_values("match_score", ascending=False),
    )


def _named(titles, scores) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "title": titles,
            "company": [f"Company {t}" for t in titles],
            "location": ["Boston, MA"] * len(titles),
            "site": ["indeed"] * len(titles),
            "description": ["a job description"] * len(titles),
            "job_url": [f"https://example.test/{t}" for t in titles],
            "match_score": scores,
            "match_reason": ["reason"] * len(titles),
            "jd_flags": [[] for _ in titles],
        }
    )


def test_more_jobs_button_hidden_when_everything_is_scored(db):
    at = _run(_results(12))
    assert at.session_state["pending_df"].empty
    assert not [b for b in at.button if b.key == "more_jobs"]


def test_more_jobs_button_offers_the_waiting_jobs(db):
    at = _run(_results(12), pending_df=_results(5))
    assert [b for b in at.button if b.key == "more_jobs"]
    assert any("5 more job(s) found but not scored yet" in c.value for c in at.caption)


def test_more_jobs_appends_at_the_bottom_without_reordering(db, monkeypatch):
    """The chosen behaviour: a new batch lands below what is already on screen,
    even when it contains a better match. Nothing the user is reading moves."""
    _fake_scoring(monkeypatch)
    at = _run(_named(["A", "B"], [90, 80]), pending_df=_named(["C", "D"], [95, 70]))
    at.button(key="more_jobs").click().run()
    assert not at.exception, at.exception

    # C scores 95 — higher than A — but still lands underneath it.
    assert list(at.session_state["results_df"]["title"]) == ["A", "B", "C", "D"]
    assert at.session_state["pending_df"].empty


def test_more_jobs_scores_a_hundred_at_a_time(db, monkeypatch):
    _fake_scoring(monkeypatch)
    at = _run(_results(1), pending_df=_results(250))
    at.button(key="more_jobs").click().run()
    assert not at.exception, at.exception
    assert len(at.session_state["results_df"]) == 101
    assert len(at.session_state["pending_df"]) == 150


def test_a_search_scores_only_the_first_hundred(db, monkeypatch):
    """A three-city search can turn up hundreds of postings; scoring every one up
    front is the bill (and the wait) this cap exists to avoid."""
    _fake_scoring(monkeypatch)
    monkeypatch.setattr("src.job_scraper.scrape_jobs", lambda *a, **k: _results(250))
    at = _fresh()
    _click_search(at)
    assert len(at.session_state["results_df"]) == 100
    assert len(at.session_state["pending_df"]) == 150
    assert any("scored the first 100" in msg.value for msg in at.success)


def test_empty_search_clears_the_pending_pool(db, monkeypatch):
    """Otherwise More Jobs would offer up the *previous* search's leftovers."""
    at = _run(_results(12), pending_df=_results(30))
    _empty_scrape(monkeypatch)
    _click_search(at)
    assert at.session_state["pending_df"].empty
    assert at.session_state["results_df"].empty
    assert not [b for b in at.button if b.key == "more_jobs"]


def test_later_batches_are_scored_not_silently_dropped(db, monkeypatch):
    """Regression: chunks after the first arrived with an index starting at 100,
    and rank_jobs assigns its score columns from a bare pd.Series (index 0..n-1).
    Pandas aligns on the index, so every score landed on a row that wasn't in the
    frame — the batch scored all-NaN, failed the `>= min_score` test, and
    vanished. You paid to score 100 jobs and the list didn't grow.
    """
    scored_lengths = []

    def fake_rank(df, resume):
        # Mimic the real assignment that made the index matter.
        scored_lengths.append(len(df))
        df = df.copy()
        df["match_score"] = pd.to_numeric(pd.Series([90] * len(df)), errors="coerce")
        return df

    monkeypatch.setattr(pm, "get_latest_resume", lambda: {"skills": ["python"]})
    monkeypatch.setattr("src.job_matcher.rank_jobs", fake_rank)

    at = _run(_results(1), pending_df=_results(250))
    at.button(key="more_jobs").click().run()      # batch 1 -> rows 0-99
    assert not at.exception, at.exception
    at.button(key="more_jobs").click().run()      # batch 2 -> the regression
    assert not at.exception, at.exception

    results = at.session_state["results_df"]
    assert scored_lengths == [100, 100]
    assert len(results) == 201
    assert int(results["match_score"].isna().sum()) == 0, (
        "later batch scored NaN — every one of those jobs would be hidden"
    )


def test_every_batch_is_visible_at_the_default_min_score(db, monkeypatch):
    """The end-to-end symptom: the visible job count must actually grow."""
    _fake_scoring(monkeypatch)
    at = _run(_results(1), pending_df=_results(250, score=90))
    at.button(key="more_jobs").click().run()
    first = len(at.session_state["results_df"])
    at.button(key="more_jobs").click().run()
    assert not at.exception, at.exception
    assert len(at.session_state["results_df"]) == first + 100


def test_jobs_the_scorer_failed_on_are_reported_not_just_dropped(db):
    """rank_jobs raises only when every batch fails, so a partial failure leaves
    rows with no score. Those fail the `>= min_score` test and disappear — the
    user must be told why, not left thinking the search found less than it did.
    """
    jobs = _results(10)
    jobs.loc[3:, "match_score"] = pd.NA          # 7 of 10 failed to score
    at = _run(jobs)
    assert any("7 job(s) couldn't be scored" in w.value for w in at.warning)


def test_no_scoring_warning_when_everything_scored(db):
    at = _run(_results(10))
    assert not any("couldn't be scored" in w.value for w in at.warning)


def test_unscored_jobs_are_not_counted_as_low_scoring(db):
    """They are two different problems with two different fixes — lowering the
    slider will never bring back a job that was never scored."""
    jobs = _results(10, score=90)
    jobs.loc[8:, "match_score"] = pd.NA          # 2 unscored, 8 scoring 90
    at = _run(jobs, search_min_score=50)
    assert not any("lower-scoring job(s) hidden" in c.value for c in at.caption)
    assert any("2 job(s) couldn't be scored" in w.value for w in at.warning)


# ── Scoring errors must name the real cause ──────────────────────────────────
def _scoring_raises(monkeypatch, exc: Exception):
    monkeypatch.setattr(pm, "get_latest_resume", lambda: {"skills": ["python"]})
    def boom(df, resume):
        raise exc
    monkeypatch.setattr("src.job_matcher.rank_jobs", boom)


def test_a_403_blames_the_network_not_the_key(db, monkeypatch):
    """Regression: every scoring failure used to say "check your GROQ_API_KEY",
    which sent you re-checking a key that was fine. Groq answers a bad key with
    401; a 403 is it blocking the IP — a VPN or proxy."""
    _scoring_raises(monkeypatch, Exception(
        "Error code: 403 - Access denied. Please check your network settings."
    ))
    at = _run(_results(1), pending_df=_results(5))
    at.button(key="more_jobs").click().run()
    assert not at.exception, at.exception
    text = " ".join(e.value for e in at.error)
    assert "VPN" in text
    assert "GROQ_API_KEY" not in text


def test_a_401_still_points_at_the_key(db, monkeypatch):
    _scoring_raises(monkeypatch, Exception("Error code: 401 - Invalid API Key"))
    at = _run(_results(1), pending_df=_results(5))
    at.button(key="more_jobs").click().run()
    assert not at.exception, at.exception
    assert any("GROQ_API_KEY" in e.value for e in at.error)


def test_an_unrecognised_failure_keeps_the_general_advice(db, monkeypatch):
    _scoring_raises(monkeypatch, Exception("connection reset by peer"))
    at = _run(_results(1), pending_df=_results(5))
    at.button(key="more_jobs").click().run()
    assert not at.exception, at.exception
    assert any("GROQ_API_KEY" in e.value for e in at.error)


def test_a_scoring_error_survives_the_more_jobs_rerun(db, monkeypatch):
    """Regression: "More Jobs" calls st.rerun() right after scoring, which throws
    away anything already drawn — so an st.error() raised during scoring vanished
    before it was ever shown. The failure has to outlive the rerun."""
    _scoring_raises(monkeypatch, Exception("Error code: 403 - Access denied."))
    at = _run(_results(1), pending_df=_results(5))
    at.button(key="more_jobs").click().run()
    assert not at.exception, at.exception
    assert any("VPN" in e.value for e in at.error)


def test_the_scoring_error_clears_once_scoring_works(db, monkeypatch):
    """A stale error would otherwise sit above a perfectly good result set."""
    # More than one chunk, so a second "More Jobs" click is still available.
    _scoring_raises(monkeypatch, Exception("Error code: 403 - Access denied."))
    at = _run(_results(1), pending_df=_results(250))
    at.button(key="more_jobs").click().run()
    assert any("VPN" in e.value for e in at.error)

    _fake_scoring(monkeypatch)                    # network comes back
    at.button(key="more_jobs").click().run()
    assert not at.exception, at.exception
    assert not any("VPN" in e.value for e in at.error)


# ── A failed later batch must not unrank the batches that worked ─────────────
def test_a_failed_later_batch_does_not_unrank_the_earlier_ones(db, monkeypatch):
    """Regression: `scored` was a single session flag written on every scoring
    attempt, so a "More Jobs" batch that failed flipped it to False and the whole
    combined result set — including the batch that had scored perfectly well —
    was rendered as unranked. That silently dropped the min-score filter, so
    low-scoring jobs the user had filtered out reappeared.
    """
    # 10 already-scored jobs: 4 at 90, 6 at 10. At min score 50 only the 4 show.
    first = _named([f"good{i}" for i in range(4)], [90] * 4)
    weak = _named([f"weak{i}" for i in range(6)], [10] * 6)
    already = pd.concat([first, weak], ignore_index=True)

    _scoring_raises(monkeypatch, Exception("Error code: 403 - Access denied."))
    at = _run(already, pending_df=_results(250), search_min_score=50)
    assert _cards(at) == 4                       # the filter is doing its job

    at.button(key="more_jobs").click().run()     # this batch fails entirely
    assert not at.exception, at.exception

    # Still ranked: the subheader must not fall back to "(unranked)" ...
    heads = " ".join(h.value for h in at.subheader)
    assert "unranked" not in heads
    assert "scoring ≥ 50/100" in heads
    # ... the min-score filter must still hold back the six weak jobs ...
    assert _cards(at) == 4
    # ... and the failed batch must be reported rather than silently dropped.
    assert any("100 job(s) couldn't be scored" in w.value for w in at.warning)


def test_everything_unscored_still_renders_unranked(db, monkeypatch):
    """The flip side: with no score anywhere, the min-score filter would hide
    every job, so the list has to fall back to showing them unranked."""
    jobs = _results(10)
    jobs["match_score"] = pd.NA
    at = _run(jobs, search_min_score=50)
    assert any("unranked" in h.value for h in at.subheader)
    assert _cards(at) == 10


def test_a_later_success_after_a_failure_ranks_everything(db, monkeypatch):
    """A batch failing must not permanently mark the results unranked either."""
    jobs = _results(5)
    jobs["match_score"] = pd.NA                  # first attempt failed
    _fake_scoring(monkeypatch)
    at = _run(jobs, pending_df=_results(10, score=90), search_min_score=50)
    assert any("unranked" in h.value for h in at.subheader)

    at.button(key="more_jobs").click().run()     # this one succeeds
    assert not at.exception, at.exception
    heads = " ".join(h.value for h in at.subheader)
    assert "unranked" not in heads
    assert _cards(at) == 10                      # the 5 unscored stay held back
    assert any("5 job(s) couldn't be scored" in w.value for w in at.warning)
