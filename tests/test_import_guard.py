"""Tests for src/import_guard.py — the screen a page shows when it can't load its code.

The situation being reproduced is a ``src`` module still sitting in memory as an
old copy that lacks something a newer page imports. That's staged by putting a
bare stand-in module into ``sys.modules`` under the real module's name.
"""

import sys
import types

import pytest
from streamlit.testing.v1 import AppTest

# A minimal page that imports from src the way the real pages do. It leans on
# src.jobkey because that module never touches the database: the Reload test
# really does load a fresh copy of whatever it imports, and a fresh
# src.profile_manager would point at the real data/jobseeker.db.
_PAGE = """
import streamlit as st
from src.import_guard import friendly_import_errors

with friendly_import_errors():
    from src.jobkey import job_signature

st.write("page loaded")
"""


def _is_src(name: str) -> bool:
    return name == "src" or name.startswith("src.")


@pytest.fixture
def make_stale():
    """Return a function that swaps a ``src`` module for an empty old copy.

    Every ``src`` entry in sys.modules is put back exactly as it was afterwards.
    That matters beyond tidiness: conftest isolates the database by patching the
    *current* src.profile_manager module object, so a fresh copy left behind by the
    Reload button would send every later test to the real database.
    """
    saved = {n: m for n, m in sys.modules.items() if _is_src(n)}

    def _make_stale(name: str):
        sys.modules[name] = types.ModuleType(name)

    yield _make_stale

    for name in [n for n in sys.modules if _is_src(n)]:
        del sys.modules[name]
    sys.modules.update(saved)


def _text(at: AppTest) -> str:
    return " ".join(m.value for m in at.markdown)


def test_the_reported_error_shows_steps_instead_of_a_traceback(make_stale):
    """The exact crash from the Job History page: an old profile_manager that
    predates delete_saved_job."""
    make_stale("src.profile_manager")

    at = AppTest.from_file("pages/4_Job_History.py", default_timeout=30).run()

    assert at.title[0].value == "🔄 This page needs a refresh"
    assert [b.label for b in at.button] == ["🔄 Reload the app"]
    assert not at.tabs  # the page itself stopped rather than half-rendering
    # The traceback is still there for whoever fixes a real bug, just folded away.
    assert len(at.exception) == 1
    assert len(at.expander[0].exception) == 1
    assert "delete_saved_job" in at.exception[0].message


def test_reload_button_loads_the_new_code(make_stale):
    make_stale("src.jobkey")
    at = AppTest.from_string(_PAGE, default_timeout=30).run()
    assert "page loaded" not in _text(at)

    at.button[0].click().run()

    assert not at.exception
    assert "page loaded" in _text(at)
    assert hasattr(sys.modules["src.jobkey"], "job_signature")


def test_only_import_errors_are_caught():
    """Anything else is a different problem and keeps Streamlit's normal display."""
    at = AppTest.from_string(
        "from src.import_guard import friendly_import_errors\n"
        "with friendly_import_errors():\n"
        "    raise ValueError('boom')\n",
        default_timeout=30,
    ).run()

    assert not at.title
    assert "boom" in at.exception[0].message
