"""Tests for the Home page (pages/0_Home.py).

Driven through AppTest against conftest's throwaway database, so the numbers on
the page come from rows these tests write and nothing else.
"""

from streamlit.testing.v1 import AppTest

import src.application_status as status
import src.profile_manager as pm

PAGE = "pages/0_Home.py"


def _run() -> AppTest:
    at = AppTest.from_file(PAGE, default_timeout=30)
    at.run()
    assert not at.exception, at.exception
    return at


def _metrics(at: AppTest) -> dict:
    return {m.label: m.value for m in at.metric}


def _text(at: AppTest) -> str:
    return " ".join([m.value for m in at.markdown] + [c.value for c in at.caption])


def _apply(company, title, location="Boston, MA"):
    pm.mark_job_applied({"company": company, "title": title, "location": location,
                         "url": "u", "site": "indeed"})


def _full_profile():
    pm.save_profile({"name": "Jane Q Doe", "email": "jane@example.test", "phone": "",
                     "current_company": "", "linkedin": "", "portfolio": "",
                     "github": ""})
    pm.save_resume("jane.pdf", "raw", {"summary": "s", "skills": ["Python"]})


def test_empty_database_shows_the_checklist_and_zero_kpis():
    at = _run()
    assert at.title[0].value == "Welcome to JobSeeker"
    assert any(h.value == "Get set up" for h in at.subheader)
    assert "0 of 3 done" in at.get("progress")[0].proto.text
    assert _metrics(at) == {"Applications": "0", "In-process": "0", "Accepted": "0",
                            "Saved": "0"}


def test_full_data_greets_by_first_name_and_hides_the_checklist():
    _full_profile()
    _apply("Acme", "SWE")
    _apply("Beta", "PM", "NYC")
    _apply("Gamma", "Data Engineer", "Austin, TX")
    pm.update_application_outcome(pm.job_signature("Beta", "PM", "NYC"), status.IN_PROCESS)
    pm.update_application_outcome(
        pm.job_signature("Gamma", "Data Engineer", "Austin, TX"), status.ACCEPTED
    )
    pm.save_job({"company": "Delta", "title": "QA", "location": "LA", "url": "v"})

    at = _run()

    assert at.title[0].value == "Welcome back, Jane"
    assert not any(h.value == "Get set up" for h in at.subheader)
    assert _metrics(at) == {"Applications": "3", "In-process": "1", "Accepted": "1",
                            "Saved": "1"}
    body = _text(at)
    for title in ("SWE", "PM", "Data Engineer"):
        assert f"**{title}**" in body
    assert "Delta" not in body           # saved, not applied: not a recent application
    assert ":orange-badge[:material/forum: In-process]" in body
    assert "jane.pdf" in body


def test_recent_list_stops_at_five():
    for i in range(7):
        _apply(f"Company {i}", f"Role {i}")
    body = _text(_run())
    assert sum(f"**Role {i}**" in body for i in range(7)) == 5


def test_checklist_counts_partial_progress():
    pm.save_resume("cv.pdf", "raw", {"summary": "s"})
    at = _run()
    assert "1 of 3 done" in at.get("progress")[0].proto.text


def test_a_saved_search_counts_as_the_first_search():
    _full_profile()
    pm.save_search_prefs({"query": "engineer", "location": "", "time_filter": "",
                          "is_remote": 0, "min_score": 50, "distance": 50})
    at = _run()
    assert not any(h.value == "Get set up" for h in at.subheader)
