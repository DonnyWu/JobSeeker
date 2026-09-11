"""Tests for saved/applied job tracking in src/profile_manager.py.

Each test runs against a throwaway SQLite DB (monkeypatched ENGINE on a tmp file)
so the real data/jobseeker.db is never touched. No network, no GROQ.
"""

import pandas as pd
import pytest
from sqlalchemy import create_engine, text

import src.application_status as status
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
    pm.save_job(job)  # a later Save must NOT revert it to 'saved'
    key = pm.job_signature("Acme", "SWE", "Boston, MA")
    assert _status(db, key) == "applied"
    assert key in pm.get_applied_keys()


# ── applied-job status tracking (Job History) ────────────────────────────────
def test_get_applied_jobs_returns_only_applied_with_fields(db):
    pm.mark_job_applied({"company": "Acme", "title": "SWE", "location": "Boston, MA", "url": "u"})
    pm.save_job({"company": "Beta", "title": "PM", "location": "NYC", "url": "v"})  # saved only

    applied = pm.get_applied_jobs()
    assert len(applied) == 1
    rec = applied[0]
    assert rec["company"] == "Acme"
    # A freshly applied job carries an explicit 'pending', not a NULL, so a future
    # export has a value in every cell.
    assert rec["outcome"] == status.PENDING
    assert "interview_stage" in rec
    assert rec["applied_at"]  # stamped when marked applied


def test_update_application_outcome_sets_status_and_stage(db):
    job = {"company": "Acme", "title": "SWE", "location": "Boston, MA", "url": "u"}
    pm.mark_job_applied(job)
    key = pm.job_signature("Acme", "SWE", "Boston, MA")

    pm.update_application_outcome(key, status.IN_PROCESS, "Round 2")
    rec = pm.get_applied_jobs()[0]
    assert rec["outcome"] == status.IN_PROCESS
    assert rec["interview_stage"] == "Round 2"
    assert rec["status_updated_at"]  # stamped on every status change


def test_update_status_without_stage_preserves_existing_stage(db):
    job = {"company": "Acme", "title": "SWE", "location": "Boston, MA", "url": "u"}
    pm.mark_job_applied(job)
    key = pm.job_signature("Acme", "SWE", "Boston, MA")

    pm.update_application_outcome(key, status.IN_PROCESS, "Onsite")
    pm.update_application_outcome(key, status.REJECTED)  # no stage passed
    rec = pm.get_applied_jobs()[0]
    assert rec["outcome"] == status.REJECTED
    assert rec["interview_stage"] == "Onsite"  # untouched


def test_applied_at_stamped_and_not_cleared_by_later_save(db):
    job = {"company": "Acme", "title": "SWE", "location": "Boston, MA", "url": "u"}
    pm.mark_job_applied(job)
    first = pm.get_applied_jobs()[0]["applied_at"]
    assert first

    pm.save_job(job)  # a later Save must not clear the application date
    assert pm.get_applied_jobs()[0]["applied_at"] == first


def test_later_save_does_not_reset_status_to_pending(db):
    """Re-saving a job you've moved along must not knock it back to Pending.

    ``save_job`` runs on every Save click in Job Search, including for a role you
    already applied to and are interviewing for. The pending stamp is guarded on
    NULL for exactly this case.
    """
    job = {"company": "Acme", "title": "SWE", "location": "Boston, MA", "url": "u"}
    pm.mark_job_applied(job)
    key = pm.job_signature("Acme", "SWE", "Boston, MA")
    pm.update_application_outcome(key, status.ACCEPTED)

    pm.save_job(job)
    assert pm.get_applied_jobs()[0]["outcome"] == status.ACCEPTED


# ── unmarking clears the application it undoes ───────────────────────────────
def test_unmark_clears_the_post_application_fields(db):
    job = {"company": "Acme", "title": "SWE", "location": "Boston, MA", "url": "u"}
    pm.mark_job_applied(job)
    key = pm.job_signature("Acme", "SWE", "Boston, MA")
    pm.update_application_outcome(key, status.IN_PROCESS, "Round 2")

    pm.unmark_job_applied(key)

    with db.connect() as conn:
        row = conn.execute(
            text(
                "SELECT status, outcome, interview_stage, applied_at, status_updated_at "
                "FROM saved_jobs WHERE job_key=:k"
            ),
            {"k": key},
        ).fetchone()
    assert row[0] == "saved"
    assert row[1] is None  # outcome
    assert row[2] is None  # interview_stage
    assert row[3] is None  # applied_at
    assert row[4]  # the reset is itself a status change, so it is stamped


def test_reapplying_after_unmark_starts_at_pending(db):
    """The bug this guards against.

    An unmarked job kept its outcome, and ``_upsert_job`` only stamps 'pending' when
    outcome is empty — so a job unmarked while Accepted came back Accepted, with its
    original application date, as though nothing had happened.
    """
    job = {"company": "Acme", "title": "SWE", "location": "Boston, MA", "url": "u"}
    pm.mark_job_applied(job)
    key = pm.job_signature("Acme", "SWE", "Boston, MA")
    pm.update_application_outcome(key, status.ACCEPTED)
    first_applied_at = pm.get_applied_jobs()[0]["applied_at"]

    pm.unmark_job_applied(key)
    pm.mark_job_applied(job)

    rec = pm.get_applied_jobs()[0]
    assert rec["outcome"] == status.PENDING
    assert rec["interview_stage"] is None
    assert rec["applied_at"]  # a new application gets a new date
    assert rec["applied_at"] != first_applied_at


def test_unmarked_job_returns_to_saved_without_a_status(db):
    job = {"company": "Acme", "title": "SWE", "location": "Boston, MA", "url": "u"}
    pm.mark_job_applied(job)
    key = pm.job_signature("Acme", "SWE", "Boston, MA")
    pm.update_application_outcome(key, status.REJECTED)

    pm.unmark_job_applied(key)

    assert pm.get_applied_jobs() == []
    saved = pm.get_saved_jobs()
    assert len(saved) == 1
    # Whatever the Saved Jobs tab shows, it must not be a leftover "Rejected".
    assert saved[0]["outcome"] is None


# ── saved-jobs list (Job History → Saved Jobs tab) ───────────────────────────
def test_get_saved_jobs_excludes_applied(db):
    pm.save_job({"company": "Beta", "title": "PM", "location": "NYC", "url": "v"})
    pm.mark_job_applied({"company": "Acme", "title": "SWE", "location": "Boston, MA", "url": "u"})

    saved = pm.get_saved_jobs()
    assert [r["company"] for r in saved] == ["Beta"]


def test_marking_a_saved_job_applied_moves_it_between_lists(db):
    pm.save_job({"company": "Beta", "title": "PM", "location": "NYC", "url": "v"})
    job = pm.get_saved_jobs()[0]

    pm.mark_job_applied(job)

    assert pm.get_saved_jobs() == []
    applied = pm.get_applied_jobs()
    assert len(applied) == 1
    assert applied[0]["company"] == "Beta"
    assert applied[0]["outcome"] == status.PENDING


def test_delete_saved_job_removes_the_row(db):
    pm.save_job({"company": "Beta", "title": "PM", "location": "NYC", "url": "v"})
    key = pm.job_signature("Beta", "PM", "NYC")

    pm.delete_saved_job(key)
    assert pm.get_saved_jobs() == []
    assert _count(db, key) == 0


def test_delete_saved_job_refuses_to_touch_an_applied_job(db):
    """Remove sits next to Mark-as-applied; it must not be able to erase history."""
    job = {"company": "Acme", "title": "SWE", "location": "Boston, MA", "url": "u"}
    pm.mark_job_applied(job)
    key = pm.job_signature("Acme", "SWE", "Boston, MA")

    pm.delete_saved_job(key)
    assert _count(db, key) == 1
    assert key in pm.get_applied_keys()


# ── status vocabulary ────────────────────────────────────────────────────────
@pytest.mark.parametrize(
    "stored,expected",
    [
        (None, status.PENDING),          # every row applied for before this feature
        ("", status.PENDING),
        ("interview", status.IN_PROCESS),  # retired vocabulary, commit 2ef18fe
        ("offer", status.IN_PROCESS),
        ("DECLINED", status.REJECTED),     # case is not part of the value
        ("accepted", status.ACCEPTED),
        ("something odd", status.PENDING),  # never blow up the list over one cell
    ],
)
def test_normalize_status(stored, expected):
    assert status.normalize(stored) == expected


# ── export seam ──────────────────────────────────────────────────────────────
def test_to_application_record_shape_and_normalized_status(db):
    pm.mark_job_applied(
        {"company": "Acme", "title": "SWE", "location": "Boston, MA", "url": "u",
         "site": "indeed", "match_score": 82}
    )
    key = pm.job_signature("Acme", "SWE", "Boston, MA")
    pm.update_application_outcome(key, status.IN_PROCESS, "Round 2")

    rec = pm.to_application_record(pm.get_applied_jobs()[0])
    assert set(rec) == {
        "company", "title", "location", "url", "source", "match_score",
        "status", "stage_note", "applied_at", "status_updated_at",
    }
    assert rec["company"] == "Acme"
    assert rec["match_score"] == 82
    assert rec["status"] == status.IN_PROCESS
    assert rec["stage_note"] == "Round 2"


def test_to_application_record_exports_a_legacy_row_as_a_real_status(db):
    """A row written by the old version must not export as a blank cell."""
    pm.mark_job_applied({"company": "Acme", "title": "SWE", "location": "Boston, MA", "url": "u"})
    with db.connect() as conn:
        conn.execute(text("UPDATE saved_jobs SET outcome='interview'"))
        conn.commit()

    rec = pm.to_application_record(pm.get_applied_jobs()[0])
    assert rec["status"] == status.IN_PROCESS


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
