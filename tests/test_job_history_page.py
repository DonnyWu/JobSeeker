"""Tests for the Job History page (pages/4_Job_History.py).

Driven through Streamlit's AppTest harness so these exercise the real page rather
than a reimplementation of its logic. Nothing here scrapes or calls Groq: jobs are
written straight to the database, which is all the page reads.

Each test runs against a throwaway SQLite DB (conftest's autouse fixture points
profile_manager.ENGINE at a tmp file) so the real data/jobseeker.db is untouched.
"""

from sqlalchemy import text
from streamlit.testing.v1 import AppTest

import src.application_status as status
import src.profile_manager as pm

PAGE = "pages/4_Job_History.py"


def _run() -> AppTest:
    at = AppTest.from_file(PAGE, default_timeout=30)
    at.run()
    assert not at.exception, at.exception
    return at


def _button(at: AppTest, key: str):
    """The one button with this key, or None. Keys carry the row id, so a caller
    that doesn't know the id passes a prefix-matched key it built itself."""
    return next((b for b in at.button if b.key == key), None)


def _applied_job(**over):
    job = {"company": "Acme", "title": "SWE", "location": "Boston, MA", "url": "u",
           "site": "indeed", "match_score": 82}
    job.update(over)
    pm.mark_job_applied(job)
    return pm.get_applied_jobs()[0]


def _text(at: AppTest) -> str:
    """Everything the page rendered, flattened, for 'is this on the page' checks."""
    return " ".join(
        [m.value for m in at.markdown] + [c.value for c in at.caption]
        + [i.value for i in at.info]
    )


# ── empty states ─────────────────────────────────────────────────────────────
def test_renders_with_an_empty_database():
    at = _run()
    body = _text(at)
    assert "No applied jobs yet" in body
    assert "Nothing saved yet" in body


# ── Applied tab ──────────────────────────────────────────────────────────────
def test_applied_job_shows_as_pending_by_default():
    _applied_job()
    at = _run()
    body = _text(at)
    assert "SWE" in body and "Acme" in body
    assert "Pending" in body


def test_legacy_outcome_renders_as_in_process(_isolated_db):
    """A row written by the previous version must still read correctly."""
    _applied_job()
    with _isolated_db.connect() as conn:
        conn.execute(text("UPDATE saved_jobs SET outcome='interview'"))
        conn.commit()

    assert "In-process" in _text(_run())


def test_clicking_a_status_persists_it():
    job = _applied_job()
    at = _run()

    _button(at, f"status_{status.IN_PROCESS}_{job['id']}").click().run()

    assert pm.get_applied_jobs()[0]["outcome"] == status.IN_PROCESS


def test_every_status_is_reachable():
    job = _applied_job()
    for value in (status.IN_PROCESS, status.ACCEPTED, status.REJECTED, status.PENDING):
        at = _run()
        _button(at, f"status_{value}_{job['id']}").click().run()
        assert pm.get_applied_jobs()[0]["outcome"] == value


def test_current_status_button_is_disabled():
    job = _applied_job()
    at = _run()
    # Fresh application: Pending is the state, so its button is the pressed-out one.
    assert _button(at, f"status_{status.PENDING}_{job['id']}").disabled
    assert not _button(at, f"status_{status.ACCEPTED}_{job['id']}").disabled


def test_stage_note_appears_only_when_in_process():
    job = _applied_job()
    at = _run()
    assert not at.text_input  # nothing to note while Pending

    _button(at, f"status_{status.IN_PROCESS}_{job['id']}").click().run()
    at = _run()
    assert [t.key for t in at.text_input] == [f"stage_{job['id']}"]


def test_stage_note_saves():
    job = _applied_job()
    pm.update_application_outcome(pm.job_signature("Acme", "SWE", "Boston, MA"),
                                  status.IN_PROCESS)
    at = _run()

    at.text_input(key=f"stage_{job['id']}").set_value("Round 2").run()
    _button(at, f"savestage_{job['id']}").click().run()

    rec = pm.get_applied_jobs()[0]
    assert rec["interview_stage"] == "Round 2"
    assert rec["outcome"] == status.IN_PROCESS  # saving a note doesn't change status


def test_status_counts_are_shown():
    _applied_job()
    _applied_job(company="Beta", title="PM", location="NYC")
    pm.update_application_outcome(pm.job_signature("Beta", "PM", "NYC"), status.ACCEPTED)

    body = _text(_run())
    assert "1 pending" in body and "1 accepted" in body


# ── Saved Jobs tab ───────────────────────────────────────────────────────────
def test_saved_job_is_listed_with_its_score_and_reason():
    pm.save_job({"company": "Beta", "title": "PM", "location": "NYC", "url": "v",
                 "site": "indeed", "match_score": 77, "match_reason": "strong overlap"})
    body = _text(_run())
    assert "PM" in body and "Beta" in body
    assert "77" in body and "strong overlap" in body


def test_saved_job_does_not_appear_on_the_applied_tab():
    pm.save_job({"company": "Beta", "title": "PM", "location": "NYC", "url": "v"})
    assert "No applied jobs yet" in _text(_run())


def test_marking_a_saved_job_applied_moves_it():
    pm.save_job({"company": "Beta", "title": "PM", "location": "NYC", "url": "v"})
    jid = pm.get_saved_jobs()[0]["id"]
    at = _run()

    _button(at, f"saved_apply_{jid}").click().run()

    assert pm.get_saved_jobs() == []
    applied = pm.get_applied_jobs()
    assert len(applied) == 1 and applied[0]["outcome"] == status.PENDING


def test_removing_a_saved_job_drops_it():
    pm.save_job({"company": "Beta", "title": "PM", "location": "NYC", "url": "v"})
    jid = pm.get_saved_jobs()[0]["id"]
    at = _run()

    _button(at, f"saved_remove_{jid}").click().run()

    assert pm.get_saved_jobs() == []
    assert "Nothing saved yet" in _text(_run())
