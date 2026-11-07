"""Tests for the router (app.py): it renders, and it routes.

Every other page test runs its page on its own. These run the real entry point,
so they catch what only shows up with the router in front: the theme CSS, the
logo, the navigation list, and the toast queue.

A note on how a page is picked: the browser asks for a page by its URL path,
which Streamlit turns into a page hash. ``AppTest.switch_page(path)`` is not that.
It hashes the *file name* ("Job_Search"), which AppTest resolves by running that
file directly, without the router in front. So the routing tests below set the
hash the way a URL does (``calc_hash("search")``), and ``switch_page`` is only used
to check a page file still runs standalone.
"""

from streamlit.testing.v1 import AppTest
from streamlit.util import calc_hash

ENTRY = "app.py"


def _app() -> AppTest:
    return AppTest.from_file(ENTRY, default_timeout=60)


def _open(url_path: str) -> AppTest:
    """Run the app as if the browser had asked for /<url_path>."""
    at = _app()
    at._page_hash = calc_hash(url_path)
    at.run()
    assert not at.exception, at.exception
    return at


def test_the_app_renders_home_by_default():
    at = _app().run()
    assert not at.exception, at.exception
    assert at.title[0].value.startswith("Welcome")


def test_job_search_opens_from_its_url():
    at = _open("search")
    assert at.title[0].value == "Job Search"
    assert [b for b in at.button if b.label == "Search"]


def test_my_jobs_opens_from_its_url():
    at = _open("my-jobs")
    assert at.title[0].value == "My Jobs"
    assert at.tabs  # Applied | Saved


def test_switch_page_to_a_page_file_still_works():
    """Runs the file directly (see the note at the top); it must not need the router."""
    at = _app().run()
    at.switch_page("pages/3_Job_Search.py").run()
    assert not at.exception, at.exception
    assert [b for b in at.button if b.label == "Search"]


def test_an_unknown_address_falls_back_to_home():
    """An old bookmark such as /Apply (the retired page) must land on Home, not crash.

    /Job_History is the same case in the browser, where it shows Streamlit's "page
    not found" notice and then Home. It can't be reproduced here because AppTest
    maps that name straight to the file in pages/ (see the note at the top).
    """
    at = _open("Apply")
    assert at.title[0].value.startswith("Welcome")


def test_a_queued_toast_is_shown_on_the_next_run():
    at = _app()
    at.session_state["_ui_flash"] = [("Profile saved", None)]
    at.run()
    assert not at.exception, at.exception
    assert [t.value for t in at.toast] == ["Profile saved"]
    assert "_ui_flash" not in at.session_state  # shown once, then cleared
