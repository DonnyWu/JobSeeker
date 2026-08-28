import math
import os
import re

import streamlit as st
import pandas as pd
from groq import RateLimitError

from src.geo import city_suggestions, format_locations, parse_locations
from src.job_scraper import scrape_jobs, HOURS_OLD_MAP
from src.job_matcher import rank_jobs, generate_why_interested
from src.jd_shield import shield_frame
from src.company_finder import find_company_job_url
from src.company_insights import company_summary
from src.profile_manager import (
    get_latest_resume,
    save_job,
    job_signature,
    get_applied_keys,
    mark_job_applied,
    unmark_job_applied,
    get_search_prefs,
    save_search_prefs,
)

st.set_page_config(page_title="Job Search — JobSeeker", page_icon="🔍", layout="wide")
st.title("🔍 Job Search")

# Job cards are tall (score, skills, company dropdown, actions), so a search that
# now returns a few hundred rows has to be paged rather than dumped in one list.
_PAGE_SIZE = 10

# How many jobs the AI scores per batch. A multi-city search can turn up several
# hundred postings and every one of them costs a Groq call, so only the first
# batch is scored up front — the rest wait behind the "More Jobs" button at the
# end of the list. Downloading jobs is cheap; scoring them is what you pay for.
# Jobs scored per click. Was 100, which on the free tier is ~17 batches paced at
# 8,000 tokens/minute — twelve minutes before anything appears, if it survived the
# rate limit at all.
#
# 24 is four batches: results in a couple of minutes, and the rest of the haul is
# still there behind "More Jobs" for anyone who wants it. Nothing is discarded —
# this only decides how much you pay for before deciding whether to continue.
_SCORE_CHUNK = int(os.environ.get("SCORE_CHUNK", "24"))


# ── Helpers for the "More about the company" dropdown ────────────────────────────
def _clean(val) -> str:
    """Return a stripped string, treating None/NaN as empty."""
    if val is None:
        return ""
    try:
        if pd.isna(val):
            return ""
    except (TypeError, ValueError):
        pass
    return str(val).strip()


def _as_list_cell(val) -> list:
    """Read a list-valued results column (matched/missing skills, knockouts),
    treating a missing column or NaN as an empty list."""
    return val if isinstance(val, list) else []


def _job_keys(df: pd.DataFrame) -> pd.Series:
    """The job_key for every row, computed once and reused.

    Streamlit re-executes this script top to bottom on every widget interaction,
    so anything here runs again on each pagination click, filter toggle and
    expander. job_signature() was being called twice per visible job — once in the
    applied filter's per-row apply(), once more in the card loop — and the filter's
    copy walked the *entire* result set, which grows with every "More Jobs".

    Cached on the frame so the work happens once per result set rather than once
    per interaction. Returns a Series so callers can use vectorised .isin().
    """
    if "job_key" in df.columns:
        return df["job_key"]
    return pd.Series(
        [
            job_signature(c, t, l)
            for c, t, l in zip(
                df.get("company", pd.Series([""] * len(df))).fillna(""),
                df.get("title", pd.Series([""] * len(df))).fillna(""),
                df.get("location", pd.Series([""] * len(df))).fillna(""),
            )
        ],
        index=df.index,
    )


def _pct(val) -> str:
    """Format a 0-100 sub-score as 'NN%', or '—' when it's missing/NaN."""
    if val is None:
        return "—"
    try:
        if pd.isna(val):
            return "—"
        return f"{int(round(float(val)))}%"
    except (TypeError, ValueError):
        return "—"


def _reason_with_gaps(row) -> str:
    """Match reason with the top missing keywords appended, so the skills gap
    survives into the saved/applied record (which only persists match_reason)."""
    reason = _clean(row.get("match_reason"))
    missing = _as_list_cell(row.get("missing_skills"))
    if missing:
        gap = "Missing keywords: " + ", ".join(missing[:5])
        reason = f"{reason} — {gap}" if reason else gap
    return reason


def _salary(row) -> str:
    lo, hi = _clean(row.get("min_amount")), _clean(row.get("max_amount"))
    if not lo and not hi:
        return ""
    cur = _clean(row.get("currency")) or "$"
    interval = _clean(row.get("interval"))

    def fmt(x):
        try:
            return f"{float(x):,.0f}"
        except (TypeError, ValueError):
            return x

    rng = f"{cur}{fmt(lo or hi)}"
    if hi and hi != lo:
        rng = f"{cur}{fmt(lo)} – {cur}{fmt(hi)}"
    return rng + (f" / {interval}" if interval else "")


def _render_company_section(idx, row, resume: dict, has_resume: bool):
    company = _clean(row.get("company"))
    title = _clean(row.get("title"))
    k = _clean(row.get("job_url")) or f"{company}-{title}-{idx}"

    with st.expander(f"ℹ️ More about {company or 'this company'}"):
        # ── Company facts (free, from the scrape) ──
        facts = []
        if _clean(row.get("company_industry")):
            facts.append(f"**Industry:** {_clean(row.get('company_industry'))}")
        if _clean(row.get("company_num_employees")):
            facts.append(f"**Size:** {_clean(row.get('company_num_employees'))}")
        if _clean(row.get("company_revenue")):
            facts.append(f"**Revenue:** {_clean(row.get('company_revenue'))}")
        if _clean(row.get("job_level")):
            facts.append(f"**Level:** {_clean(row.get('job_level'))}")
        if _salary(row):
            facts.append(f"**Pay:** {_salary(row)}")
        if facts:
            st.markdown(" · ".join(facts))
        company_site = _clean(row.get("company_url_direct")) or _clean(row.get("company_url"))
        if company_site:
            st.markdown(f"[Company page ↗]({company_site})")
        if _clean(row.get("company_description")):
            st.caption(_clean(row.get("company_description"))[:600])

        # ── AI company summary (role-tailored pros/cons + average salary) ──
        st.markdown(f"**What employees say about working as a _{title or 'this role'}_**")
        # Keyed on what the answer actually depends on — company and role — not on
        # the posting it was requested from. company_summary() never sees the URL,
        # so keying on `k` meant the same role at the same company, listed in three
        # cities, paid for three identical web searches. Multi-location search
        # surfaces exactly that shape by design.
        sum_key = f"sum_{company.lower()}|{title.lower()}"
        if st.button("Generate company summary", key=f"summary_{idx}"):
            with st.spinner("Researching the company…"):
                st.session_state[sum_key] = company_summary(company, title)

        if sum_key in st.session_state:
            res = st.session_state[sum_key]
            if res.get("summary"):
                st.markdown(res["summary"])
                if res.get("source") == "web":
                    st.caption("Based on a live web search of employee reviews & salary data.")
                else:
                    st.caption(
                        "⚠️ General AI summary — live web data was unavailable, so details "
                        "(especially salary) may be outdated."
                    )
            else:
                st.caption(
                    "Couldn't generate a summary right now (check GROQ_API_KEY) — try again."
                )

        # ── "Why do you want to work here?" ──
        st.markdown("**Why do you want to work here?**")
        why_key = f"why_{k}"
        flags_key = f"whyflags_{k}"
        echo_key = f"whyecho_{k}"
        if st.button(
            "Generate answer",
            key=f"whybtn_{idx}",
            disabled=not has_resume,
            help=None if has_resume else "Upload a résumé on the Resume page first.",
        ):
            with st.spinner("Writing a tailored answer…"):
                try:
                    answer, flags, echoed = generate_why_interested(
                        resume, row.to_dict()
                    )
                except Exception as e:
                    answer, flags, echoed = f"(Generation failed: {e})", [], []
                st.session_state[why_key] = answer
                st.session_state[flags_key] = flags
                st.session_state[echo_key] = echoed
            st.session_state.pop(f"whytext_{idx}", None)  # let text_area reseed
        if not has_resume:
            st.caption("Upload a résumé on the Resume page to enable this.")
        if why_key in st.session_state:
            echoed = st.session_state.get(echo_key) or []
            if echoed:
                # Not a suspicion — a confirmed hit. The posting asked for these
                # words and the model produced them, so this text is watermarked:
                # send it and the employer can grep their inbox for it. Louder than
                # the flag warning below, because there is nothing probabilistic
                # left to hedge about.
                st.error(
                    "🚨 **Do not send this as written.** The posting asked for "
                    + ", ".join(f"“{w}”" for w in echoed)
                    + (" and that word is" if len(echoed) == 1 else " and those words are")
                    + " in the answer below. That is a tracking marker — it lets the "
                    "employer identify this text as AI-written. Remove it, or "
                    "regenerate."
                )
            if st.session_state.get(flags_key):
                # This answer is the one thing that leaves the app and lands in
                # front of an employer, so say it plainly before it's copied.
                st.warning(
                    "🪤 This posting contains hidden instructions aimed at AI ("
                    + "; ".join(st.session_state[flags_key])
                    + "). They were stripped before this answer was written — but "
                    "read it over before you send it anywhere."
                )
            st.text_area(
                "Edit / copy your answer",
                value=st.session_state[why_key],
                key=f"whytext_{idx}",
                height=140,
            )

# ── Search inputs (prefilled from the last saved search) ─────────────────────────
prefs = get_search_prefs()
_tf_options = list(HOURS_OLD_MAP.keys())
_saved_tf = prefs.get("time_filter")
_ms = prefs.get("min_score")
_dist = prefs.get("distance")

# Saved prefs seed the chips on first load only; after that session state is the
# source of truth. parse_locations also carries pre-multi-location saves forward:
# a stored "Boston, MA" comes back as one chip, not as "Boston" plus "MA".
st.session_state.setdefault(
    "search_locations", parse_locations(prefs.get("location", ""))
)


def _persist_search_prefs():
    """Save the current search-bar state the moment any filter changes, so it
    survives a browser refresh without waiting for the next Search click."""
    save_search_prefs(
        {
            "query": st.session_state.get("search_query", ""),
            "location": format_locations(st.session_state.get("search_locations", [])),
            "time_filter": st.session_state.get("search_time_filter", ""),
            "is_remote": int(st.session_state.get("search_is_remote", False)),
            "min_score": int(st.session_state.get("search_min_score", 50)),
            "distance": int(st.session_state.get("search_distance", 50)),
        }
    )


def _clear_locations():
    """Empty the location chips.

    Has to run as an on_click callback: Streamlit refuses assignment to a
    widget-backed session key once that widget has been drawn, and callbacks run
    before the script re-executes. Same reason `_go()` below is a callback.
    """
    st.session_state["search_locations"] = []
    _persist_search_prefs()


def _location_options() -> list[str]:
    """Options for the Locations box: current chips first, then the city type-ahead.

    The chips have to lead the list because Streamlit drops a selected value that
    isn't among the options — including free-typed ones like "Remote" that will
    never appear in the city list. Everything after them is the type-ahead pool
    the box filters as you type.
    """
    chosen = st.session_state.get("search_locations", [])
    picked = {c.strip().lower() for c in chosen}
    return list(chosen) + [c for c in city_suggestions() if c.lower() not in picked]


col1, col2, col3 = st.columns([3, 3, 3])
with col1:
    query = st.text_input(
        "Job title / keywords", value=prefs.get("query", ""), placeholder="Software Engineer",
        key="search_query", on_change=_persist_search_prefs,
    )
    is_remote = st.checkbox(
        "Remote only", value=bool(prefs.get("is_remote")),
        key="search_is_remote", on_change=_persist_search_prefs,
    )
    applied_view = st.radio(
        "Applied jobs",
        ["Show applied", "Hide applied"],
        horizontal=True,
        help="Show keeps them in the list with an Applied highlight; Hide removes them.",
    )
    hide_applied = applied_view == "Hide applied"
with col2:
    # Chips with an ✕ each, over a free-text box that suggests US cities as you
    # type. accept_new_options keeps it a text box rather than a dropdown, so a
    # region the dataset has never heard of can still be typed in. Seeded with
    # setdefault above rather than `default=`, which Streamlit rejects alongside a
    # session key it has already written.
    _loc_col, _clear_col = st.columns([7, 3], vertical_alignment="bottom")
    with _loc_col:
        locations = st.multiselect(
            "Locations",
            options=_location_options(),
            accept_new_options=True,
            placeholder="Enter another region",
            key="search_locations",
            on_change=_persist_search_prefs,
            help="Start typing to pick a US city, or enter any region yourself. "
                 "Search several at once — a job counts if it sits near any one "
                 "of them. Leave empty to search nationwide.",
        )
    with _clear_col:
        st.button(
            "Clear All", key="clear_locations", on_click=_clear_locations,
            disabled=not locations, use_container_width=True,
        )
    distance = st.slider(
        "Search radius (miles)", 0, 100, _dist if _dist is not None else 50, step=5,
        help="How far from *each* location to include jobs. Crosses state lines by "
             "true distance (e.g. 100 mi from New York reaches NJ/CT). Ignored for "
             "Remote-only searches.",
        key="search_distance", on_change=_persist_search_prefs,
    )
with col3:
    time_filter = st.selectbox(
        "Posted within",
        _tf_options,
        index=_tf_options.index(_saved_tf) if _saved_tf in _tf_options else 1,
        key="search_time_filter", on_change=_persist_search_prefs,
    )
    min_score = st.slider(
        "Minimum match score", 0, 100, _ms if _ms is not None else 50, step=5,
        help="Only show jobs scoring at least this. Adjust to re-filter without re-searching.",
        key="search_min_score", on_change=_persist_search_prefs,
    )

_bcol1, _bcol2 = st.columns([8, 2])
with _bcol2:
    search_clicked = st.button("Search", type="primary", use_container_width=True)

# ── Session-state storage ──────────────────────────────────────────────────────
if "results_df" not in st.session_state:
    st.session_state.results_df = pd.DataFrame()
if "results_page" not in st.session_state:
    st.session_state.results_page = 1
if "duplicates_merged" not in st.session_state:
    st.session_state.duplicates_merged = 0
if "boards_failed" not in st.session_state:
    st.session_state.boards_failed = []
# Downloaded but not yet scored. Everything a search finds lands here first;
# _score_next_chunk moves it into results_df a batch at a time.
if "pending_df" not in st.session_state:
    st.session_state.pending_df = pd.DataFrame()


def _has_scores(df: pd.DataFrame) -> bool:
    """Whether any row in the results carries a real match score.

    Read off the rows themselves rather than tracked in a flag. A single global
    "did scoring work" flag records only whichever batch ran last, so one failed
    "More Jobs" batch used to retroactively mark the entire result set unranked —
    throwing away the min-score filter and the "couldn't be scored" warning for
    jobs that had scored perfectly well in an earlier batch.
    """
    return "match_score" in df.columns and bool(df["match_score"].notna().any())


def _scoring_error_message(err: Exception) -> str:
    """Explain a scoring failure by what actually went wrong.

    Blaming the API key for everything sends you to re-check a key that is fine.
    Groq answers a bad key with 401 "Invalid API Key"; a 403 "check your network
    settings" is it refusing the *IP* — which is what a VPN or datacenter exit
    node looks like from its side, and the same thing that makes Cloudflare-backed
    job boards (ZipRecruiter, Google) return nothing on the same search.
    """
    text = str(err)
    # 429 first, and by type rather than by string: it is the failure users
    # actually hit, and it used to fall through to the generic message below —
    # which told them to go re-check a key that was never the problem.
    # 413 "Request too large" carries code rate_limit_exceeded but is NOT a
    # RateLimitError in the SDK — there is no 413 class, so it arrives as a bare
    # APIStatusError and used to fall through to the generic "check your key".
    # It is the same condition as a 429 from the user's side.
    if (
        isinstance(err, RateLimitError)
        or "429" in text
        or "413" in text
        or "rate_limit_exceeded" in text
        or "request too large" in text.lower()
    ):
        retry = None
        try:
            retry = err.response.headers.get("retry-after")
        except AttributeError:
            pass
        when = f"about {retry} seconds" if retry else "a minute"
        return (
            "Scoring hit Groq's rate limit (429) — jobs are shown unranked. "
            f"Nothing is wrong with your API key. Wait {when} and press "
            "More Jobs, or search again; already-scored jobs come back from the "
            "cache for free, so a retry only pays for what's left. "
            f"({err})"
        )
    if "403" in text:
        return (
            "Scoring failed — jobs are shown unranked. Groq refused the "
            "connection (403), which means it is blocking this network rather "
            "than rejecting your key: a VPN or proxy will do it. Turn the VPN "
            f"off (or move its exit to the US) and search again. ({err})"
        )
    if "401" in text or "invalid api key" in text.lower():
        return (
            "Scoring failed — jobs are shown unranked. Groq rejected the API key "
            f"itself (401). Check GROQ_API_KEY in your .env and restart. ({err})"
        )
    return (
        "Scoring failed — jobs are shown unranked. "
        f"Check that your GROQ_API_KEY is valid and restart the app. ({err})"
    )


_SENIOR_WORDS = ("director", "vp", "vice president", "principal", "head of",
                 "chief", "staff", "distinguished", "president")
_JUNIOR_WORDS = ("intern", "internship", "apprentice", "trainee")


def _promise(df: pd.DataFrame, query: str, resume: dict) -> pd.Series:
    """A cheap local guess at which jobs are worth grading first.

    Deliberately an *ordering*, never a filter. Filtering on signals this crude
    would quietly drop a job with an unusual title that the model would have
    scored well — the exact failure the user can't see and can't correct for.
    Ordering costs nothing if it's wrong: the job is still in the list, still one
    click from being scored.

    Higher is sooner. Uses only what's already on the frame, so it's free.
    """
    titles = df.get("title", pd.Series([""] * len(df), index=df.index)).fillna("").str.lower()
    terms = {w for w in re.findall(r"[a-z]+", (query or "").lower()) if len(w) > 2}

    # Word overlap with what was actually searched for.
    overlap = titles.apply(
        lambda t: len(terms & set(re.findall(r"[a-z]+", t))) if terms else 0
    )

    try:
        years = float(resume.get("total_years_experience") or 0)
    except (TypeError, ValueError):
        years = 0.0

    # Levels far from the candidate's are scored last, not removed. _blended_score
    # already caps these via _KNOCKOUT_CAP once the model sees them; this only
    # decides what gets seen first.
    def _level_penalty(t: str) -> int:
        if years and years < 8 and any(w in t for w in _SENIOR_WORDS):
            return -2
        if years and years > 2 and any(w in t for w in _JUNIOR_WORDS):
            return -2
        return 0

    return overlap + titles.apply(_level_penalty)


def _scoring_progress():
    """A callback that turns scoring progress into a visible progress bar.

    On the free tier a chunk is paced to fit 8,000 tokens/minute, so it takes
    minutes. A spinner that never changes during that is indistinguishable from
    one that has hung, which is the difference between "this is working" and
    "this is broken" from the user's side.
    """
    bar = st.progress(0.0, text="Scoring jobs…")

    def _update(done: int, total: int):
        frac = done / total if total else 1.0
        bar.progress(min(1.0, frac), text=f"Scoring jobs… batch {done} of {total}")

    return _update


def _score_next_chunk() -> int:
    """Score the next _SCORE_CHUNK downloaded jobs, appending them to the results.

    Scoring is the expensive half of a search — every job is a Groq call — so a
    scrape parks its whole haul in `pending_df` and we pay for it a batch at a
    time. `rank_jobs` sorts whatever it is handed, so each batch arrives
    best-first while the jobs already on screen keep their positions: nothing the
    user is currently reading moves when a new batch lands underneath it.

    Returns how many jobs were scored.
    """
    pending = st.session_state.pending_df
    if pending.empty:
        return 0

    # Both slices are re-indexed from 0. rank_jobs writes its score columns with
    # bare `pd.Series([...])` values, which carry an index of 0..n-1, and pandas
    # aligns an assigned Series on the index — so handing it a slice starting at
    # row 100 lands every score on a row that isn't in the frame and silently
    # scores the whole batch NaN. NaN then fails the `>= min_score` test below,
    # so the batch you just paid to score would vanish from the results entirely.
    st.session_state.pop("scoring_error", None)  # this attempt speaks for itself
    resume = get_latest_resume()

    # Spend the chunk on the most promising jobs first. Nothing is dropped — the
    # rest stay in pending_df and "More Jobs" grades them — but if the first
    # chunk answers the question, the remaining tokens are never spent.
    if resume and not pending.empty:
        order = _promise(
            pending, st.session_state.get("search_query", ""), resume
        ).sort_values(ascending=False, kind="stable")
        pending = pending.loc[order.index]

    chunk = pending.iloc[:_SCORE_CHUNK].copy().reset_index(drop=True)
    st.session_state.pending_df = pending.iloc[_SCORE_CHUNK:].reset_index(drop=True)

    if resume:
        try:
            chunk = rank_jobs(chunk, resume, on_progress=_scoring_progress())
        except Exception as e:
            # Recorded rather than rendered here: the "More Jobs" path calls
            # st.rerun() straight after scoring, which throws away anything
            # already drawn — so an st.error() at this point would vanish before
            # the user ever saw it. The results section renders it instead.
            st.session_state.scoring_error = _scoring_error_message(e)
            chunk["match_score"] = pd.NA
            chunk["match_reason"] = "scoring failed"
    else:
        chunk["match_score"] = pd.NA
        chunk["match_reason"] = "Upload a resume on the Resume page to get AI match scores"

    # Where the new rows begin, so the pager can jump to them below — without it,
    # clicking "More Jobs" from page 1 looks like it did nothing at all.
    st.session_state.jump_to_index = len(st.session_state.results_df)
    # ignore_index keeps results_df on a stable 0..N-1 index. Rows already on
    # screen must not be renumbered: the card widget keys (apply_/find_/why_) and
    # the `df.at[idx, "careers_url"]` write-back are all keyed on that index.
    st.session_state.results_df = pd.concat(
        [st.session_state.results_df, chunk], ignore_index=True
    )
    return len(chunk)

if search_clicked:
    if not query:
        st.warning("Please enter a job title or keywords.")
    else:
        with st.spinner("Scraping job boards…"):
            raw = scrape_jobs(
                query, locations, time_filter, is_remote=is_remote, distance_miles=distance
            )

        # Read the dedupe count straight off the scraper's result: DataFrame.attrs
        # doesn't reliably survive the reshaping that shield_frame and rank_jobs do,
        # so capture it here rather than trying to read it further down.
        st.session_state.duplicates_merged = int(raw.attrs.get("duplicates_merged", 0))
        st.session_state.boards_failed = list(raw.attrs.get("boards_failed", []))

        if raw.empty:
            st.warning("No jobs found. Try broader search terms or a longer time window.")
            # Drop the previous search's results, or the page keeps rendering them
            # under this search's "No jobs found" warning — and under captions
            # (duplicates merged / blocked boards) that were already overwritten
            # above and now describe a different search entirely. Clearing the frame
            # is enough to silence those captions too: they render inside the
            # `if not df.empty` block below. pending_df has to go with it, or
            # "More Jobs" would offer up the *previous* search's leftovers.
            st.session_state.results_df = pd.DataFrame()
            st.session_state.pending_df = pd.DataFrame()
            st.session_state.results_page = 1
            st.session_state.pop("scoring_error", None)
        else:
            # Shield at the scrape boundary, before anything branches on whether we
            # can score. A posting is trapped or not regardless of whether the user
            # has uploaded a résumé, so the jd_flags column — and the red outline it
            # drives below — has to exist on every path. It used to be created
            # inside rank_jobs, which meant the warning silently vanished for anyone
            # browsing without a résumé, or whenever scoring errored out.
            #
            # The whole haul is shielded here, not just the batch we are about to
            # score: shield_frame is local regex work with no API call behind it, so
            # deferring the rest would save nothing.
            raw = shield_frame(raw)
            # Stamp the identity key once, here, while the frame is assembled.
            # Every later use — the applied filter, the card loop, the score
            # cache — reads the column instead of recomputing it, which matters
            # because Streamlit re-runs this whole script on every click.
            #
            # Must come *after* shield_frame: that overwrites title and company
            # with their sanitised values, and the key has to be derived from the
            # same strings everything else sees.
            raw["job_key"] = _job_keys(raw)

            if not get_latest_resume():
                st.warning(
                    "No resume found — showing all jobs unranked. "
                    "Upload a resume to enable scoring."
                )

            st.session_state.pending_df = raw
            st.session_state.results_df = pd.DataFrame()
            st.session_state.results_page = 1  # a fresh result set starts at page 1
            with st.spinner("Filtering jobs based on résumé"):
                shown = _score_next_chunk()
            # A new search always starts at the top, so drop the jump the scorer
            # just recorded for "More Jobs".
            st.session_state.pop("jump_to_index", None)

            if _has_scores(st.session_state.results_df):
                if len(raw) > shown:
                    st.success(
                        f"Found {len(raw)} jobs and scored the first {shown} — "
                        "use More Jobs at the end of the list for the rest."
                    )
                else:
                    st.success(f"Found and scored {len(raw)} jobs.")

# ── Results table ──────────────────────────────────────────────────────────────
if st.session_state.get("scoring_error"):
    st.error(st.session_state["scoring_error"])

df = st.session_state.results_df
# Derived per-render from the rows, never from a flag a later batch can flip.
scored = _has_scores(df)

if not df.empty:
    resume = get_latest_resume()
    has_resume = bool(resume)
    applied_keys = get_applied_keys()

    # Every filter has to be applied before the list is sliced into pages —
    # filtering afterwards (as the applied-check used to, with a `continue` inside
    # the render loop) leaves short pages, e.g. 7 cards on a page of 10.
    #
    # Filter into `base`, never back into `df`: the card actions below write to
    # `df` and push it to st.session_state.results_df, so narrowing `df` here would
    # persist the filtered frame and permanently drop applied jobs from the results.
    base = df
    if hide_applied:
        # Vectorised against the job_key column rather than a per-row apply().
        # Streamlit re-runs this whole script on every widget interaction, so the
        # old df.apply(axis=1) — a Python-level loop doing two regex substitutions
        # per row — ran again on every pagination click and filter toggle, over
        # the entire result set, which grows with each "More Jobs".
        base = df[~_job_keys(df).isin(applied_keys)]

    if scored:
        # Jobs the scorer never managed to grade. rank_jobs only raises when
        # *every* batch fails, so a partial failure — Groq rate-limiting a few of
        # the twenty batches a 100-job chunk fires off — leaves rows with no score
        # at all. NaN fails the `>= min_score` test below, so without saying so
        # here those jobs would drop out of the results with no explanation and
        # look like the search simply found less than it did.
        unscored = int(base["match_score"].isna().sum())
        view = base[base["match_score"] >= min_score]
        hidden = len(base) - len(view) - unscored
        st.subheader(f"Results — {len(view)} job(s) scoring ≥ {min_score}/100")
        if unscored:
            st.warning(
                f"⚠️ {unscored} job(s) couldn't be scored and aren't shown — the AI "
                "scorer failed on them (usually a Groq rate limit when a large batch "
                "is scored at once). Searching again, or waiting a minute, usually "
                "picks them back up."
            )
        if view.empty and not unscored:
            st.info(
                f"No jobs scored ≥ {min_score}/100. "
                "Lower the minimum match score or broaden your search."
            )
        elif hidden > 0:
            st.caption(f"{hidden} lower-scoring job(s) hidden. Lower the slider to see them.")
    else:
        view = base
        st.subheader(f"Results — {len(view)} job(s) (unranked)")

    # Without this the count silently dropping (450 scraped -> 200 shown) reads as
    # lost results rather than the same role being merged across boards.
    if st.session_state.duplicates_merged:
        st.caption(
            f"{st.session_state.duplicates_merged} duplicate posting(s) merged — "
            "the same job listed on more than one board is shown once."
        )

    # A blocked board used to be invisible: you'd just get fewer jobs and assume
    # that was the whole market. Naming it explains a thin result set.
    if st.session_state.boards_failed:
        st.caption(
            "⚠️ No results from "
            + ", ".join(sorted(st.session_state.boards_failed))
            + " — that board blocked or rate-limited this search. "
            "The other boards are unaffected; searching again may pick it back up."
        )

    # ── Jump to a freshly scored batch ────────────────────────────────────────
    # New jobs are appended to the end of the list, so from page 1 a "More Jobs"
    # click would look like it had done nothing. Move to the page holding the
    # first new row. This only navigates — no job changes position.
    _target = st.session_state.pop("jump_to_index", None)
    if _target is not None:
        _new = [pos for pos, i in enumerate(view.index) if i >= _target]
        if _new:  # empty when the whole new batch fell below the min-score slider
            st.session_state.results_page = _new[0] // _PAGE_SIZE + 1

    # ── Pagination ────────────────────────────────────────────────────────────
    total = len(view)
    total_pages = max(1, math.ceil(total / _PAGE_SIZE))
    # Clamp every run: raising the min-score slider shrinks the list, and the page
    # you were on can fall off the end.
    page = min(max(int(st.session_state.results_page), 1), total_pages)
    st.session_state.results_page = page

    start = (page - 1) * _PAGE_SIZE
    page_view = view.iloc[start : start + _PAGE_SIZE]

    def _go(delta: int):
        """Step the page. Runs as a button callback so the new page is in session
        state before the script re-renders, rather than a rerun behind."""
        st.session_state.results_page = min(
            max(st.session_state.results_page + delta, 1), total_pages
        )

    def _pager(position: str):
        """Prev / status / Next. Rendered above *and* below the cards so you never
        have to scroll back up; `position` just keeps the widget keys unique."""
        prev_col, status_col, next_col = st.columns([2, 6, 2])
        with prev_col:
            st.button(
                "← Previous", key=f"page_prev_{position}", disabled=page <= 1,
                on_click=_go, args=(-1,), use_container_width=True,
            )
        with status_col:
            if total:
                st.markdown(
                    f"<div style='text-align:center;padding-top:0.5rem'>Page {page} of "
                    f"{total_pages} — showing {start + 1}–{start + len(page_view)} "
                    f"of {total}</div>",
                    unsafe_allow_html=True,
                )
        with next_col:
            st.button(
                "Next →", key=f"page_next_{position}", disabled=page >= total_pages,
                on_click=_go, args=(1,), use_container_width=True,
            )

    if total > _PAGE_SIZE:
        _pager("top")

    # A keyed container carries an ``st-key-<key>`` CSS class, so varying the card
    # key by flag state is all it takes to outline a trapped posting in red.
    #
    # The box-shadow does the real work: it paints a ring on the keyed element
    # itself, so the outline survives whatever nesting Streamlit puts the actual
    # border on. The border-color rules are a bonus that recolors the real border
    # if it happens to sit within reach — over-applying a colour is harmless,
    # since an element with no border renders nothing from it.
    st.html(
        "<style>"
        "div[class*='st-key-jobcard-trap-']"
        "{box-shadow:0 0 0 2px #ff4b4b;border-radius:0.5rem;}"
        "div[class*='st-key-jobcard-trap-'],"
        "div[class*='st-key-jobcard-trap-'] > div,"
        "div[class*='st-key-jobcard-trap-'] .stVerticalBlock"
        "{border-color:#ff4b4b !important;}"
        "</style>"
    )

    # Iterate the page slice, but keep .iterrows() so `idx` stays the original
    # DataFrame index: the card container keys and the per-job session keys
    # (whytext_/why_/sum_) are all built from it, and a per-page counter would
    # collide across pages and show one job's generated text on another.
    for idx, row in page_view.iterrows():
        # Read the stamped column; only recompute for a frame that predates it
        # (an older session's results_df restored from state).
        key = row.get("job_key") or job_signature(
            row.get("company", ""), row.get("title", ""), row.get("location", "")
        )
        is_applied = key in applied_keys

        raw_score = row.get("match_score")
        has_score = raw_score is not None and not pd.isna(raw_score)
        if has_score:
            score = int(round(float(raw_score)))
            score_color = "green" if score >= 70 else "orange" if score >= 40 else "red"
            score_display = f":{score_color}[{score}/100]"
        else:
            score = 0
            score_display = ":gray[—]"

        jd_flags = _as_list_cell(row.get("jd_flags"))

        with st.container(
            border=True, key=f"jobcard-trap-{idx}" if jd_flags else f"jobcard-{idx}"
        ):
            if jd_flags:
                # Deliberately redundant with the red outline: if a future Streamlit
                # release changes the border markup, the outline silently stops
                # working — and the failure mode must be "no outline", not
                # "no warning".
                st.error(
                    "🪤 Hidden instructions aimed at AI found in this posting: "
                    + "; ".join(jd_flags)
                )
            if is_applied:
                # Special highlighter: keep the job visible but clearly flagged.
                st.success("✅ Applied — you've already applied to this role.")
            c1, c2, c3 = st.columns([5, 2, 3])
            with c1:
                st.markdown(f"**{row.get('title', '')}** — {row.get('company', '')}")
                st.caption(
                    f"{row.get('location', '')} · {row.get('site', '')} · {row.get('date_posted', '')}"
                )
                reason = row.get("match_reason", "")
                if reason:
                    st.caption(f"_{reason}_")
                cov = row.get("ats_coverage")
                if cov is not None and not pd.isna(cov):
                    st.caption(f"📊 ATS keyword coverage: **{int(round(float(cov)))}%**")
                missing = _as_list_cell(row.get("missing_skills"))
                if missing:
                    st.caption(
                        ":orange[**Missing keywords:** " + ", ".join(missing[:5]) + "]"
                    )
                for ko in _as_list_cell(row.get("knockouts")):
                    st.caption(f":red[⚠ Knockout: {ko}]")
            with c2:
                st.markdown(f"**Match:** {score_display}")
            with c3:
                job_url = row.get("job_url") or row.get("url", "")

                # Find on company site (careers/apply page — kept separate from the
                # jobspy "company_url" column, which points at the company's board page)
                if st.button("Find on Company Site", key=f"find_{idx}"):
                    with st.spinner("Searching…"):
                        careers_url = find_company_job_url(
                            row.get("company", ""), row.get("title", ""), job_url
                        )
                    df.at[idx, "careers_url"] = careers_url
                    st.session_state.results_df = df
                    st.rerun()

                company_url = _clean(row.get("careers_url"))
                display_url = company_url if company_url else job_url

                if display_url:
                    st.markdown(f"[Open posting]({display_url})")

                # Save & go to Apply page
                if st.button("Auto-Apply", key=f"apply_{idx}"):
                    apply_url = company_url if company_url else job_url
                    job_record = {
                        "title": row.get("title", ""),
                        "company": row.get("company", ""),
                        "location": row.get("location", ""),
                        "url": apply_url,
                        "company_url": company_url,
                        "match_score": score,
                        "match_reason": _reason_with_gaps(row),
                        "source": row.get("site", ""),
                        "posted_at": str(row.get("date_posted", "")),
                        "status": "saved",
                    }
                    save_job(job_record)
                    st.session_state["apply_job"] = job_record
                    st.switch_page("pages/4_Apply.py")

                # Mark / unmark as applied (persists across searches)
                if is_applied:
                    if st.button("↩︎ Unmark applied", key=f"unmark_{idx}"):
                        unmark_job_applied(key)
                        st.rerun()
                else:
                    if st.button("✓ Mark as applied", key=f"mark_{idx}"):
                        rec = row.to_dict()
                        rec["match_reason"] = _reason_with_gaps(row)
                        mark_job_applied(rec)
                        st.rerun()

            # ── Match breakdown (sub-scores + matched keywords) ──
            matched = _as_list_cell(row.get("matched_skills"))
            tf, sf, ef = row.get("title_fit"), row.get("seniority_fit"), row.get("education_fit")
            have_breakdown = matched or any(
                v is not None and not pd.isna(v) for v in (tf, sf, ef)
            )
            if has_score and have_breakdown:
                with st.expander("Match breakdown"):
                    st.markdown(
                        f"**Title fit:** {_pct(tf)} · **Seniority fit:** {_pct(sf)} · "
                        f"**Education fit:** {_pct(ef)}"
                    )
                    if matched:
                        st.markdown("**Matched keywords:** " + ", ".join(matched[:10]))
                    miss_full = _as_list_cell(row.get("missing_skills"))
                    if miss_full:
                        st.markdown(
                            ":orange[**Missing keywords:** " + ", ".join(miss_full[:10]) + "]"
                        )

            _render_company_section(idx, row, resume, has_resume)

    if total > _PAGE_SIZE:
        _pager("bottom")

    # ── More Jobs ─────────────────────────────────────────────────────────────
    # The jobs behind this button are already downloaded and sitting in memory, so
    # clicking it only pays for the AI scoring — it never goes back out to the job
    # boards, which means no extra wait on them and nothing for one to block.
    _pending = st.session_state.pending_df
    if not _pending.empty:
        st.caption(
            f"{len(_pending)} more job(s) found but not scored yet — "
            "they'll be added to the end of this list."
        )
        if st.button("More Jobs", type="primary", key="more_jobs"):
            with st.spinner("Scoring more jobs…"):
                _score_next_chunk()
            # The results list renders above this button, so by the time we get
            # here it has already drawn the old, shorter list. Rerun so the jobs
            # we just scored actually appear.
            st.rerun()
