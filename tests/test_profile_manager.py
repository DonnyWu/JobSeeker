"""Tests for applied-job tracking in src/profile_manager.py.

Each test runs against a throwaway SQLite DB (monkeypatched ENGINE on a tmp file)
so the real data/jobseeker.db is never touched. No network, no GROQ.
"""

import pandas as pd
import pytest
from sqlalchemy import create_engine, text

import src.profile_manager as pm


@pytest.fixture
def db(tmp_path, monkeypatch):
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    monkeypatch.setattr(pm, "ENGINE", engine)
    pm.init_db()
    return engine


def _count(engine, key):
    with engine.connect() as conn:
        return conn.execute(
            text("SELECT COUNT(*) FROM saved_jobs WHERE job_key=:k"), {"k": key}
        ).scalar()


def _status(engine, key):
    with engine.connect() as conn:
        return conn.execute(
            text("SELECT status FROM saved_jobs WHERE job_key=:k"), {"k": key}
        ).scalar()


# ── job_signature ────────────────────────────────────────────────────────────
def test_signature_normalizes_case_space_and_location_suffix():
    # Case/whitespace differences and a trailing ", US" must not change the key.
    a = pm.job_signature("Acme Corp ", "  Software   Engineer ", "Boston, MA")
    b = pm.job_signature("acme corp", "software engineer", "Boston, MA, US")
    assert a == b


def test_signature_differs_by_city():
    assert pm.job_signature("Acme", "SWE", "Boston, MA") != pm.job_signature(
        "Acme", "SWE", "Cambridge, MA"
    )


# ── mark / get_applied_keys / unmark ─────────────────────────────────────────
def test_mark_then_reappearance_matches_then_unmark(db):
    pm.mark_job_applied(
        {"company": "Acme", "title": "SWE", "location": "Boston, MA", "url": "u1"}
    )
    key = pm.job_signature("Acme", "SWE", "Boston, MA")
    assert key in pm.get_applied_keys()

    # Same role coming back from another board: different URL + location suffix,
    # jobspy-style keys (job_url/site/date_posted) and a NaN score — still matches.
    reappearance = {
        "company": "Acme",
        "title": "SWE",
        "location": "Boston, MA, US",
        "job_url": "u2",
        "site": "indeed",
        "date_posted": "2026-06-01",
        "match_score": pd.NA,
    }
    assert pm.job_signature(
        reappearance["company"], reappearance["title"], reappearance["location"]
    ) in pm.get_applied_keys()

    pm.unmark_job_applied(key)
    assert key not in pm.get_applied_keys()


def test_mark_accepts_jobspy_style_row(db):
    # row.to_dict()-style payload (no url/source/posted_at) must not raise and
    # must be retrievable.
    pm.mark_job_applied(
        {"company": "Foo", "title": "Data Eng", "location": "Remote",
         "job_url": "x", "site": "google", "date_posted": "2026-06-02", "match_score": 87.4}
    )
    assert pm.job_signature("Foo", "Data Eng", "Remote") in pm.get_applied_keys()


# ── save_job upsert semantics ────────────────────────────────────────────────
def test_save_job_upserts_single_row(db):
    job = {"company": "Acme", "title": "SWE", "location": "Boston, MA", "url": "u"}
    pm.save_job(job)
    pm.save_job(job)
    assert _count(db, pm.job_signature("Acme", "SWE", "Boston, MA")) == 1


def test_save_job_does_not_downgrade_applied(db):
    job = {"company": "Acme", "title": "SWE", "location": "Boston, MA", "url": "u"}
    pm.mark_job_applied(job)
    pm.save_job(job)  # a later Auto-Apply must NOT revert it to 'saved'
    key = pm.job_signature("Acme", "SWE", "Boston, MA")
    assert _status(db, key) == "applied"
    assert key in pm.get_applied_keys()


# ── search_prefs (cached Job Search inputs) ──────────────────────────────────
def test_search_prefs_empty_by_default(db):
    assert pm.get_search_prefs() == {}


def test_save_then_get_search_prefs(db):
    prefs = {
        "query": "Software Engineer",
        "location": "Boston, MA",
        "time_filter": "Past Week",
        "is_remote": 1,
        "min_score": 70,
    }
    pm.save_search_prefs(prefs)
    got = pm.get_search_prefs()
    assert got["query"] == "Software Engineer"
    assert got["location"] == "Boston, MA"
    assert got["time_filter"] == "Past Week"
    assert got["is_remote"] == 1  # bool stored as 0/1
    assert got["min_score"] == 70


def test_save_search_prefs_upserts_single_row(db):
    pm.save_search_prefs(
        {"query": "SWE", "location": "NYC", "time_filter": "Past 24 hours",
         "is_remote": 0, "min_score": 50}
    )
    pm.save_search_prefs(
        {"query": "Data Engineer", "location": "Remote", "time_filter": "Past Month",
         "is_remote": 1, "min_score": 0}
    )
    with db.connect() as conn:
        assert conn.execute(text("SELECT COUNT(*) FROM search_prefs")).scalar() == 1
    got = pm.get_search_prefs()
    assert got["query"] == "Data Engineer"
    assert got["min_score"] == 0  # the second save's values win
