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


def _stage_inputs(at: AppTest) -> list:
    """The stage-note boxes. The page's search box is a text input too."""
    return [t.key for t in at.text_input if t.key and t.key.startswith("stage_")]


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

    at.segmented_control(key=f"status_{job['id']}").set_value(status.IN_PROCESS).run()

    assert pm.get_applied_jobs()[0]["outcome"] == status.IN_PROCESS


def test_every_status_is_reachable():
    job = _applied_job()
    for value in (status.IN_PROCESS, status.ACCEPTED, status.REJECTED, status.PENDING):
        at = _run()
        at.segmented_control(key=f"status_{job['id']}").set_value(value).run()
        assert pm.get_applied_jobs()[0]["outcome"] == value


def test_current_status_is_selected():
    job = _applied_job()
    pm.update_application_outcome(pm.job_signature("Acme", "SWE", "Boston, MA"),
                                  status.ACCEPTED)
    at = _run()
    control = at.segmented_control(key=f"status_{job['id']}")
    # The current status shows as the selected option, not as a greyed-out button.
    assert control.value == status.ACCEPTED
    assert list(control.options) == ["Pending", "In-process", "Accepted", "Rejected"]


def test_stage_note_appears_only_when_in_process():
    job = _applied_job()
    at = _run()
    assert not _stage_inputs(at)  # nothing to note while Pending

    at.segmented_control(key=f"status_{job['id']}").set_value(status.IN_PROCESS).run()
    at = _run()
    assert _stage_inputs(at) == [f"stage_{job['id']}"]


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

    metrics = {m.label: m.value for m in _run().metric}
    assert metrics == {"Applications": "2", "Pending": "1", "In-process": "0",
                       "Accepted": "1", "Rejected": "0"}


# ── Saved tab ───────────────────────────────────────────────────────────
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


# ── Filters, tabs and toasts ─────────────────────────────────────────────────
def _three_applications():
    _applied_job()
    _applied_job(company="Beta", title="PM", location="NYC")
    _applied_job(company="Gamma", title="Data Engineer", location="Austin, TX")
    pm.update_application_outcome(pm.job_signature("Beta", "PM", "NYC"), status.IN_PROCESS)


def _rows(at: AppTest) -> list:
    """The status controls on screen, one per application row."""
    return [c.key for c in at.segmented_control]


def test_tabs_carry_their_counts():
    _applied_job()
    pm.save_job({"company": "Beta", "title": "PM", "location": "NYC", "url": "v"})
    assert [t.label for t in _run().tabs] == ["Applied · 1", "Saved · 1"]


def test_status_filter_narrows_the_list():
    _three_applications()
    at = _run()
    assert len(_rows(at)) == 3
    assert list(at.pills(key="history_status_filter").options) == [
        "All 3", "Pending 2", "In-process 1", "Accepted 0", "Rejected 0"]

    at.pills(key="history_status_filter").set_value(status.IN_PROCESS).run()
    assert len(_rows(at)) == 1
    assert "**PM** · Beta" in _text(at)


def test_search_box_narrows_the_list():
    _three_applications()
    at = _run()
    at.text_input(key="history_search").set_value("austin").run()
    assert len(_rows(at)) == 1
    assert "**Data Engineer** · Gamma" in _text(at)


def test_filters_that_match_nothing_say_so():
    _three_applications()
    at = _run()
    at.pills(key="history_status_filter").set_value(status.REJECTED).run()
    assert not _rows(at)
    assert "No applications match these filters." in _text(at)


def test_every_row_shows_its_status_as_text():
    _three_applications()
    body = _text(_run())
    assert body.count(":blue-badge[:material/schedule: Pending]") == 2
    assert body.count(":orange-badge[:material/forum: In-process]") == 1


def test_removing_a_saved_job_queues_a_toast():
    pm.save_job({"company": "Beta", "title": "PM", "location": "NYC", "url": "v"})
    jid = pm.get_saved_jobs()[0]["id"]
    at = _run()
    _button(at, f"saved_remove_{jid}").click().run()
    # The router shows it on the next run (see tests/test_app_shell.py).
    assert at.session_state["_ui_flash"] == [("Removed PM", ":material/delete:")]


def test_empty_states_link_to_job_search():
    at = _run()
    links = [e for e in at.get("page_link")]
    assert len(links) == 2
    assert all(l.proto.page == "search" for l in links)
