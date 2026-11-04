"""Shared look and feel: navigation, page chrome, CSS, and small formatters.

Every page imports this, and so does the router in ``app.py``. It deliberately
imports nothing that touches the database, so it is safe to load from anywhere
(including the import-guard screen, which has to render when the database layer
itself is the thing that failed to load).

Two rules the CSS below follows, so that a Streamlit upgrade can only ever make
the app look plainer, never break it:

  * It hooks only onto ``st-key-*`` classes (which Streamlit adds to keyed
    containers) and ``data-testid`` attributes — never onto generated class names.
  * Nothing is conveyed by CSS alone. A card's coloured status bar always sits next
    to a text badge saying the same thing, and a trapped posting's red ring sits
    above an ``st.error`` banner.

Tests run each page on its own, without the router, so they never see this CSS.
That is fine for the same reason: the CSS is decoration on top of a page that is
complete without it.
"""

from __future__ import annotations

import datetime as _dt
import math
import re
from pathlib import Path

import streamlit as st

from src.application_status import (
    ACCEPTED,
    DISPLAY,
    IN_PROCESS,
    PENDING,
    REJECTED,
    normalize,
)

ROOT = Path(__file__).resolve().parent.parent
LOGO = str(ROOT / "static" / "logo.svg")
LOGO_MARK = str(ROOT / "static" / "logo-mark.svg")


# ── Navigation ────────────────────────────────────────────────────────────────
def app_pages() -> dict[str, "st.Page"]:
    """Every page in the app, keyed by a short name, in navigation order.

    The router hands these to ``st.navigation``, and pages pass them to
    ``st.page_link`` for their call-to-action links. Two details matter:

      * Paths are absolute. A page run on its own (as every page test does) has
        no router, and a relative path would resolve against the wrong folder.
      * Linking with these objects, never a path string, is what keeps those
        standalone runs working: ``st.page_link("pages/...")`` looks the path up
        in the router's page list, finds no router, and raises
        ``StreamlitPageNotFoundError``. An ``st.Page`` carries its own address.

    Must be called inside a script run. ``st.Page()`` built at import time looks
    fine but half-initializes, so this is a function rather than a constant.
    """
    pages = ROOT / "pages"
    return {
        "home": st.Page(
            pages / "0_Home.py", title="Home", icon=":material/home:", default=True
        ),
        "search": st.Page(
            pages / "3_Job_Search.py",
            title="Job Search",
            icon=":material/search:",
            url_path="search",
        ),
        # The file keeps its old name so tests and history stay put; only the
        # label and the address changed.
        "my_jobs": st.Page(
            pages / "4_Job_History.py",
            title="My Jobs",
            icon=":material/work_history:",
            url_path="my-jobs",
        ),
        "resume": st.Page(
            pages / "2_Resume.py",
            title="Resume",
            icon=":material/description:",
            url_path="resume",
        ),
        "profile": st.Page(
            pages / "1_Profile.py",
            title="Profile",
            icon=":material/person:",
            url_path="profile",
        ),
    }


# ── Styles ────────────────────────────────────────────────────────────────────
_HOVER = "0 4px 14px rgba(16, 24, 40, 0.08)"
_APPLIED_BAR = "#22C55E"
_TRAP_RED = "#ff4b4b"


def _card(prefix: str) -> str:
    return f"div[class*='st-key-{prefix}']"


def _accent(prefix: str, shadow: str) -> str:
    """An accent drawn with box-shadow, plus the same accent under the hover lift.

    Both rules need the accent: box-shadow is a single property, so a hover rule
    that only set the lift would wipe the coloured bar off the card being pointed at.
    """
    sel = _card(prefix)
    return (
        f"{sel}{{box-shadow:{shadow};}}"
        f"{sel}:hover{{box-shadow:{shadow},{_HOVER};}}"
    )


def _css() -> str:
    cards = ",".join(
        _card(p) for p in ("jobcard-", "histcard-", "savedcard-", "card-")
    )
    rules = [
        # One consistent page width. Wide layout, capped so lines stay readable.
        "[data-testid='stMainBlockContainer']"
        "{max-width:1240px;padding-top:2.25rem;padding-bottom:4rem;}",
        # Cards: a quiet lift on hover and a slightly tighter inner rhythm.
        f"{cards}{{transition:box-shadow .15s ease;gap:0.6rem;}}",
        f"{','.join(s + ':hover' for s in cards.split(','))}{{box-shadow:{_HOVER};}}",
        # The match-score tile: a big number over a small caption. Its colour
        # comes from theme-aware markdown (":green[87]"), so none is set here.
        # Streamlit gives markdown a negative bottom margin, which pulls the
        # caption up over a number this size, so it is zeroed here.
        "div[class*='st-key-score-'] [data-testid='stMarkdownContainer'],"
        "div[class*='st-key-score-'] [data-testid='stCaptionContainer']"
        "{margin-bottom:0;}",
        "div[class*='st-key-score-'] [data-testid='stMarkdownContainer'] p"
        "{font-size:1.9rem;font-weight:700;line-height:1.15;margin:0;"
        "font-variant-numeric:tabular-nums;}",
        "div[class*='st-key-score-'] [data-testid='stCaptionContainer'] p"
        "{font-size:0.75rem;font-weight:500;line-height:1.2;margin:0;}",
        # A hand-drawn widget label (Job Search's "Locations", which has a Clear
        # button beside it): same size as Streamlit's own labels, and a compact
        # button so the row is no taller than a normal label.
        "div[class*='st-key-loclabel'] [data-testid='stMarkdownContainer'] p"
        "{font-size:0.875rem;margin:0;}",
        "div[class*='st-key-loclabel'] [data-testid='stMarkdownContainer']"
        "{margin-bottom:0;}",
        "div[class*='st-key-loclabel'] [data-testid='stBaseButton-tertiary']"
        "{min-height:0;height:1.4rem;padding:0 0.25rem;}",
        # A native label sits ~0.2rem above its box; the column's block gap would
        # put this one a full gap above it, out of line with its neighbours.
        "div[class*='st-key-loclabel']{margin-bottom:-0.85rem;}",
        # KPI rows (a container keyed "kpis" holding st.columns of metrics) wrap
        # two-by-two on a phone instead of stacking four tall cards.
        "@media (max-width: 640px){"
        "div[class*='st-key-kpis'] [data-testid='stHorizontalBlock']"
        "{flex-wrap:wrap;gap:0.5rem;}"
        "div[class*='st-key-kpis'] [data-testid='stColumn']"
        "{flex:1 1 calc(50% - 0.5rem) !important;min-width:calc(50% - 0.5rem) !important;}"
        "}",
        # An applied job: a green bar down the left edge, next to its Applied badge.
        _accent("jobcard-applied-", f"inset 4px 0 0 {_APPLIED_BAR}"),
    ]
    # Applications: a bar in the status colour, generated from the same table the
    # badges read, so the two cannot drift apart.
    rules += [
        _accent(f"histcard-{value}-", f"inset 4px 0 0 {color}")
        for value, (_label, _emoji, color) in DISPLAY.items()
    ]
    # A trapped posting gets a red ring. The box-shadow does the real work: it
    # paints the ring on the keyed element itself, so the outline survives
    # whatever nesting Streamlit puts the actual border on. The border-color
    # rules are a bonus that recolors the real border if it happens to sit within
    # reach — over-applying a colour is harmless, since an element with no border
    # renders nothing from it. Deliberately last, so it beats every rule above.
    trap = _card("jobcard-trap-")
    rules += [
        _accent("jobcard-trap-", f"0 0 0 2px {_TRAP_RED}"),
        f"{trap},{trap} > div,{trap} .stVerticalBlock"
        f"{{border-color:{_TRAP_RED} !important;}}",
    ]
    return "<style>" + "".join(rules) + "</style>"


def inject_styles() -> None:
    """Add the app's CSS. Called once per run by the router."""
    st.html(_css())


# ── Page chrome ───────────────────────────────────────────────────────────────
def page_header(title: str, subtitle: str | None = None, aside=None) -> None:
    """The title block every page opens with.

    ``aside`` is an optional callable that draws something on the right, level
    with the title — the Job Search page uses it for "Matching against <file>".
    """
    if aside is None:
        st.title(title, anchor=False)
        if subtitle:
            st.caption(subtitle)
        return
    left, right = st.columns([3, 2], vertical_alignment="bottom")
    with left:
        st.title(title, anchor=False)
        if subtitle:
            st.caption(subtitle)
    with right:
        aside()


def empty_state(
    title: str,
    body: str,
    icon: str,
    link=None,
    link_label: str | None = None,
    key: str | None = None,
) -> None:
    """A friendly placeholder for a list with nothing in it yet.

    ``icon`` is a Material icon name (``"travel_explore"``); ``link`` is an
    ``st.Page`` from :func:`app_pages`, never a path string.
    """
    slug = key or re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")
    with st.container(border=True, key=f"card-empty-{slug}"):
        st.markdown(f"#### :gray[:material/{icon}:] {title}")
        st.caption(body)
        if link is not None:
            st.page_link(
                link, label=link_label or link.title, icon=":material/arrow_forward:"
            )


# ── Toasts that survive a rerun ───────────────────────────────────────────────
_FLASH_KEY = "_ui_flash"


def flash(message: str, icon: str | None = None) -> None:
    """Queue a toast for the next run.

    A toast shown right before ``st.rerun()`` is thrown away with the rest of the
    run, so the message is parked in session state and the router shows it at the
    start of the run that follows.
    """
    st.session_state.setdefault(_FLASH_KEY, []).append((message, icon))


def show_flash() -> None:
    """Show (and clear) every queued toast. Called once per run by the router."""
    for message, icon in st.session_state.pop(_FLASH_KEY, []):
        st.toast(message, icon=icon)


# ── Formatting ────────────────────────────────────────────────────────────────
def plural(n: int, word: str, plural_word: str | None = None) -> str:
    """``plural(1, "job")`` -> ``"1 job"``; ``plural(3, "job")`` -> ``"3 jobs"``."""
    return f"{n:,} {word if n == 1 else (plural_word or word + 's')}"


# Characters that mean something to Streamlit's markdown. "$" starts LaTeX (a
# salary like "$120k–$150k" turns into italic maths), "]" closes a badge or
# coloured span early, and "<" ">" "#" "|" "~" "*" "_" "`" each start some other
# construct. Scraped titles and company names contain all of them.
_MD_SPECIALS = re.compile(r"([\\`*_\[\]$~|#<>])")


def md(text) -> str:
    """Escape scraped or user-typed text so markdown shows it literally."""
    if text is None:
        return ""
    try:
        if isinstance(text, float) and math.isnan(text):
            return ""
    except TypeError:
        pass
    return _MD_SPECIALS.sub(r"\\\1", str(text))


_SITE_NAMES = {
    "indeed": "Indeed",
    "linkedin": "LinkedIn",
    "glassdoor": "Glassdoor",
    "zip_recruiter": "ZipRecruiter",
    "ziprecruiter": "ZipRecruiter",
    "google": "Google",
    "bayt": "Bayt",
    "naukri": "Naukri",
    "bdjobs": "BDJobs",
}


def site_label(site) -> str:
    """A job board's display name: ``"zip_recruiter"`` -> ``"ZipRecruiter"``."""
    raw = str(site or "").strip()
    if not raw or raw.lower() == "nan":
        return ""
    return _SITE_NAMES.get(raw.lower(), raw.replace("_", " ").title())


def _to_date(value) -> _dt.date | None:
    """Coerce a date, datetime, pandas Timestamp or ISO string to a date."""
    if value is None:
        return None
    try:
        if value != value:  # NaN, and pandas' NaT — which passes as a datetime
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(value, _dt.datetime):
        return value.date()
    if isinstance(value, _dt.date):
        return value
    to_pydatetime = getattr(value, "to_pydatetime", None)  # pandas Timestamp
    if callable(to_pydatetime):
        try:
            return to_pydatetime().date()
        except (TypeError, ValueError):
            return None
    text = str(value).strip()
    if not text or text.lower() in ("nan", "nat", "none"):
        return None
    try:
        return _dt.date.fromisoformat(text[:10])
    except ValueError:
        return None


def _month_day(d: _dt.date, today: _dt.date, with_year: bool) -> str:
    # Built by hand because "%-d" (no leading zero) doesn't exist on Windows.
    label = f"{d:%b} {d.day}"
    if with_year or d.year != today.year:
        label += f", {d.year}"
    return label


def posted_label(value, today: _dt.date | None = None) -> str:
    """How long ago a job was posted: "Today", "Yesterday", "3 days ago", "Sep 6"."""
    d = _to_date(value)
    if d is None:
        return ""
    today = today or _dt.date.today()
    days = (today - d).days
    if days == 0:
        return "Today"
    if days == 1:
        return "Yesterday"
    if 1 < days < 7:
        return f"{days} days ago"
    return _month_day(d, today, with_year=False)


def short_date(value, with_year: bool = False, today: _dt.date | None = None) -> str:
    """``"2026-09-06T14:02:11"`` -> ``"Sep 6"`` (the year is added when it differs
    from this one, or always with ``with_year=True``). Blank for no date."""
    d = _to_date(value)
    if d is None:
        return ""
    return _month_day(d, today or _dt.date.today(), with_year=with_year)


# ── Badges ────────────────────────────────────────────────────────────────────
def _as_score(score) -> int | None:
    if score is None:
        return None
    try:
        f = float(score)
    except (TypeError, ValueError):
        return None
    if math.isnan(f):
        return None
    return int(round(f))


def score_color(score) -> str:
    """Green at 70+, orange from 40, red below; gray when there is no score."""
    s = _as_score(score)
    if s is None:
        return "gray"
    if s >= 70:
        return "green"
    if s >= 40:
        return "orange"
    return "red"


def score_badge(score) -> str:
    """``":green-badge[Match 87]"``, or a gray "Unscored" badge."""
    s = _as_score(score)
    if s is None:
        return ":gray-badge[Unscored]"
    return f":{score_color(s)}-badge[Match {s}]"


#: status -> (badge colour, Material icon). The hex accent lives in DISPLAY.
STATUS_STYLE = {
    PENDING: ("blue", "schedule"),
    IN_PROCESS: ("orange", "forum"),
    ACCEPTED: ("green", "check_circle"),
    REJECTED: ("red", "cancel"),
}


def status_badge(value) -> str:
    """``":blue-badge[:material/schedule: Pending]"`` for an application status."""
    v = normalize(value)
    color, icon = STATUS_STYLE[v]
    return f":{color}-badge[:material/{icon}: {DISPLAY[v][0]}]"
