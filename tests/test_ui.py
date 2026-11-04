"""Tests for src/ui.py — the formatters and badges every page shares.

These are plain functions, so they are tested directly rather than through
AppTest. The CSS and page chrome are exercised by the page tests and by eye.
"""

import datetime as dt

import pytest
from streamlit.testing.v1 import AppTest

from src import ui
from src.application_status import ACCEPTED, IN_PROCESS, PENDING, REJECTED

TODAY = dt.date(2026, 9, 10)


# ── plural ───────────────────────────────────────────────────────────────────
@pytest.mark.parametrize(
    "n, expected", [(0, "0 jobs"), (1, "1 job"), (2, "2 jobs"), (1200, "1,200 jobs")]
)
def test_plural(n, expected):
    assert ui.plural(n, "job") == expected


def test_plural_takes_an_irregular_form():
    assert ui.plural(2, "match", "matches") == "2 matches"
    assert ui.plural(1, "match", "matches") == "1 match"


# ── md escaping ──────────────────────────────────────────────────────────────
def test_md_escapes_a_dollar_so_salaries_do_not_become_latex():
    assert ui.md("$120k–$150k") == r"\$120k–\$150k"


def test_md_escapes_emphasis_and_brackets():
    # "]" would close a badge early; "*" and "_" would italicize half a title.
    assert ui.md("C++ *Senior* [Remote]_x") == r"C++ \*Senior\* \[Remote\]\_x"


def test_md_escapes_html_and_heading_markers():
    assert ui.md("<b>#1</b>") == r"\<b\>\#1\</b\>"


@pytest.mark.parametrize("blank", [None, float("nan")])
def test_md_treats_missing_as_empty(blank):
    assert ui.md(blank) == ""


def test_md_leaves_ordinary_text_alone():
    assert ui.md("Acme Corp, Boston (MA)") == "Acme Corp, Boston (MA)"


# ── site / date labels ───────────────────────────────────────────────────────
@pytest.mark.parametrize(
    "raw, expected",
    [
        ("zip_recruiter", "ZipRecruiter"),
        ("indeed", "Indeed"),
        ("LINKEDIN", "LinkedIn"),
        ("some_new_board", "Some New Board"),
        ("", ""),
        (None, ""),
    ],
)
def test_site_label(raw, expected):
    assert ui.site_label(raw) == expected


@pytest.mark.parametrize(
    "posted, expected",
    [
        (dt.date(2026, 9, 10), "Today"),
        (dt.date(2026, 9, 9), "Yesterday"),
        ("2026-09-07", "3 days ago"),
        (dt.datetime(2026, 9, 6, 8, 30), "4 days ago"),
        ("2026-09-02", "Sep 2"),
        ("2025-12-30", "Dec 30, 2025"),
        ("", ""),
        (None, ""),
        ("not a date", ""),
    ],
)
def test_posted_label(posted, expected):
    assert ui.posted_label(posted, today=TODAY) == expected


def test_posted_label_reads_a_pandas_timestamp():
    pd = pytest.importorskip("pandas")
    assert ui.posted_label(pd.Timestamp("2026-09-09"), today=TODAY) == "Yesterday"
    assert ui.posted_label(pd.NaT, today=TODAY) == ""


def test_short_date():
    assert ui.short_date("2026-09-06T14:02:11.123", today=TODAY) == "Sep 6"
    assert ui.short_date("2026-09-06", with_year=True, today=TODAY) == "Sep 6, 2026"
    assert ui.short_date("2024-01-15", today=TODAY) == "Jan 15, 2024"
    assert ui.short_date(None) == ""


# ── score badges ─────────────────────────────────────────────────────────────
@pytest.mark.parametrize(
    "score, color",
    [(100, "green"), (70, "green"), (69, "orange"), (40, "orange"), (39, "red"),
     (0, "red"), (None, "gray"), (float("nan"), "gray"), ("n/a", "gray")],
)
def test_score_color_thresholds(score, color):
    assert ui.score_color(score) == color


def test_score_badge():
    assert ui.score_badge(87.4) == ":green-badge[Match 87]"
    assert ui.score_badge(55) == ":orange-badge[Match 55]"
    assert ui.score_badge(None) == ":gray-badge[Unscored]"


# ── status badges ────────────────────────────────────────────────────────────
@pytest.mark.parametrize(
    "value, expected",
    [
        (PENDING, ":blue-badge[:material/schedule: Pending]"),
        (IN_PROCESS, ":orange-badge[:material/forum: In-process]"),
        (ACCEPTED, ":green-badge[:material/check_circle: Accepted]"),
        (REJECTED, ":red-badge[:material/cancel: Rejected]"),
        (None, ":blue-badge[:material/schedule: Pending]"),          # never set
        ("interview", ":orange-badge[:material/forum: In-process]"),  # legacy value
    ],
)
def test_status_badge(value, expected):
    assert ui.status_badge(value) == expected


def test_every_status_has_a_style():
    from src.application_status import APPLICATION_STATUSES

    assert set(ui.STATUS_STYLE) == set(APPLICATION_STATUSES)


# ── CSS ──────────────────────────────────────────────────────────────────────
def test_css_paints_every_status_bar_from_the_display_table():
    from src.application_status import DISPLAY

    css = ui._css()
    for value, (_label, _emoji, color) in DISPLAY.items():
        assert f"st-key-histcard-{value}-" in css
        assert color in css


def test_css_keeps_the_trap_ring():
    assert "st-key-jobcard-trap-" in ui._css()


# ── navigation ───────────────────────────────────────────────────────────────
def _pages_script():
    import streamlit as st

    from src import ui as _ui

    pages = _ui.app_pages()
    for name, page in pages.items():
        st.write(f"{name}|{page.title}|{page.url_path}")


def test_app_pages_have_the_expected_labels_and_addresses():
    at = AppTest.from_function(_pages_script, default_timeout=30).run()
    assert not at.exception, at.exception
    rows = [m.value for m in at.markdown]
    assert rows == [
        "home|Home|",
        "search|Job Search|search",
        "my_jobs|My Jobs|my-jobs",
        "resume|Resume|resume",
        "profile|Profile|profile",
    ]


def test_app_pages_point_at_real_files():
    for name in ("0_Home.py", "1_Profile.py", "2_Resume.py", "3_Job_Search.py",
                 "4_Job_History.py"):
        assert (ui.ROOT / "pages" / name).is_file()
    assert (ui.ROOT / "static" / "fonts" / "InterVariable.woff2").is_file()
    assert (ui.ROOT / "static" / "fonts" / "LICENSE.txt").is_file()
