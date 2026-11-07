"""Tests for the Profile page (pages/1_Profile.py).

Driven through AppTest against conftest's throwaway database.
"""

from streamlit.testing.v1 import AppTest

import src.profile_manager as pm

PAGE = "pages/1_Profile.py"

_SAVED = {"name": "Jane Doe", "email": "jane@example.test", "phone": "555-0100",
          "current_company": "Acme", "linkedin": "https://linkedin.com/in/jane",
          "portfolio": "", "github": ""}


def _run() -> AppTest:
    at = AppTest.from_file(PAGE, default_timeout=30)
    at.run()
    assert not at.exception, at.exception
    return at


def test_fields_are_prefilled_from_the_saved_profile():
    pm.save_profile(_SAVED)
    at = _run()
    assert at.text_input(key="profile_name").value == "Jane Doe"
    assert at.text_input(key="profile_email").value == "jane@example.test"
    assert at.text_input(key="profile_portfolio").value == ""


def test_every_field_has_a_placeholder():
    at = _run()
    keys = {t.key for t in at.text_input}
    assert keys == {f"profile_{f}" for f in
                    ("name", "email", "phone", "current_company", "linkedin", "github",
                     "portfolio")}
    assert all(t.placeholder for t in at.text_input)


def test_submitting_saves_the_profile():
    at = _run()
    at.text_input(key="profile_name").set_value("  Sam Lee  ")
    at.text_input(key="profile_email").set_value("sam@example.test")
    at.button(key="profile_save").click().run()
    assert not at.exception, at.exception

    saved = pm.get_profile()
    assert saved["name"] == "Sam Lee"           # stripped
    assert saved["email"] == "sam@example.test"
    assert saved["github"] == ""
    # The router shows the toast on the next run (see tests/test_app_shell.py).
    assert at.session_state["_ui_flash"] == [("Profile saved", ":material/check:")]


def test_the_copy_panel_shows_saved_values():
    pm.save_profile(_SAVED)
    at = _run()
    assert [c.value for c in at.code] == [
        "Jane Doe", "jane@example.test", "555-0100", "Acme", "https://linkedin.com/in/jane"]
    assert "5 of 7 filled" in at.get("progress")[0].proto.text


def test_an_empty_profile_explains_the_copy_panel():
    at = _run()
    assert not at.code
    assert "0 of 7 filled" in at.get("progress")[0].proto.text


def test_no_auto_fill_claim():
    """Auto-fill was retired. The old page said "used to auto-fill job
    applications", which this would catch."""
    at = _run()
    text = " ".join([m.value for m in at.markdown] + [c.value for c in at.caption])
    assert "copy" in text.lower()           # the page's subtitle and panel did render
    assert "auto-fill" not in text.lower()
