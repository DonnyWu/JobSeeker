"""Turns a page's "cannot import name" crash into steps a person can follow.

Why the crash happens at all: Streamlit re-reads a page file every time it runs,
but the ``src`` modules that page imports are loaded once and kept in memory. So
when JobSeeker's files change while the server is running — a ``git pull``, a
branch switch — the page is new while the ``src`` code it relies on can still be
the old copy. The moment the new page asks for something the old copy doesn't
have yet, Python stops with a traceback like::

    ImportError: cannot import name 'delete_saved_job' from 'src.profile_manager'

Streamlit is meant to prevent this by dropping the old copies when it sees a file
change, but it only watches while a browser tab is connected. Change the files
with the tab closed and nothing notices.

The fix is what Streamlit would have done: forget the in-memory ``src`` modules so
the next run loads them from disk. :func:`friendly_import_errors` explains that in
plain words and puts it behind a button, with a restart and a reinstall as the
fallbacks for the rarer causes (a new dependency, or a genuine bug).
"""

import importlib
import sys
from contextlib import contextmanager

import streamlit as st


def _forget_loaded_code():
    """Drop every in-memory ``src`` module so the next run loads the files on disk.

    Runs as the Reload button's ``on_click`` callback, which Streamlit calls on the
    script thread at the start of the next run, before any page code — the same
    point at which it flushes its own file-watcher evictions.
    """
    for name in list(sys.modules):
        if name == "src" or name.startswith("src."):
            sys.modules.pop(name, None)
    # Python remembers each folder's file listing. Without this, a module the
    # update *added* can still look like it doesn't exist.
    importlib.invalidate_caches()


def _explain(err: ImportError):
    st.title("🔄 This page needs a refresh")
    st.write(
        "JobSeeker couldn't load everything this page needs. This usually means the "
        "app was updated while it was running, and part of it is using an older "
        "version."
    )
    st.info("Your profile, resume and saved jobs are safe. Nothing has been lost.")
    st.write("Try these steps in order. Most of the time, step 1 is all you need.")

    st.subheader("1. Reload the app")
    st.write(
        "JobSeeker keeps a copy of its code in memory while it runs. This button "
        "throws away the old copy and loads the latest one, without stopping the app."
    )
    st.button("🔄 Reload the app", type="primary", on_click=_forget_loaded_code)

    st.subheader("2. Restart JobSeeker")
    st.write(
        "This shuts JobSeeker down completely and starts it fresh, so every part of "
        "it loads the newest version. Go to the window where JobSeeker is running and "
        "press **Ctrl + C** to stop it. Then start it again:"
    )
    st.code("uv run streamlit run app.py", language=None)

    st.subheader("3. Install what the update needs")
    st.write(
        "Sometimes an update needs new add-ons that aren't on your computer yet. This "
        "installs them. Stop JobSeeker, run this from the JobSeeker folder, then "
        "start it again:"
    )
    st.code("uv pip install -r requirements.txt", language=None)

    st.caption(
        "Still not working? Then it's a bug in JobSeeker, not something you did. "
        "The details below will help whoever fixes it."
    )
    with st.expander("Technical details"):
        st.exception(err)


@contextmanager
def friendly_import_errors():
    """Wrap a page's ``src`` imports so a failure shows steps, not a traceback.

    Usage, at the top of a page::

        with friendly_import_errors():
            from src.profile_manager import get_saved_jobs

    Only ``ImportError`` is caught (``ModuleNotFoundError`` is a kind of it). Any
    other exception still reaches Streamlit's normal error display.
    """
    try:
        yield
    except ImportError as err:
        _explain(err)
        st.stop()
        # Only reached outside a real Streamlit run, where st.stop() does nothing.
        # Re-raising beats carrying on with names that were never imported.
        raise
