import math

import streamlit as st
import pandas as pd

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
        sum_key = f"sum_{k}"
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


def _persist_search_prefs():
    """Save the current search-bar state the moment any filter changes, so it
    survives a browser refresh without waiting for the next Search click."""
    save_search_prefs(
        {
            "query": st.session_state.get("search_query", ""),
            "location": st.session_state.get("search_location", ""),
            "time_filter": st.session_state.get("search_time_filter", ""),
            "is_remote": int(st.session_state.get("search_is_remote", False)),
            "min_score": int(st.session_state.get("search_min_score", 50)),
            "distance": int(st.session_state.get("search_distance", 50)),
        }
    )


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
    location = st.text_input(
        "Location", value=prefs.get("location", ""), placeholder="New York, NY",
        key="search_location", on_change=_persist_search_prefs,
    )
    distance = st.slider(
        "Search radius (miles)", 0, 100, _dist if _dist is not None else 50, step=5,
        help="How far from the location to include jobs. Crosses state lines by "
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
if "scored" not in st.session_state:
    st.session_state.scored = False
if "results_page" not in st.session_state:
    st.session_state.results_page = 1
if "duplicates_merged" not in st.session_state:
    st.session_state.duplicates_merged = 0
if "boards_failed" not in st.session_state:
    st.session_state.boards_failed = []

if search_clicked:
    if not query:
        st.warning("Please enter a job title or keywords.")
    else:
        with st.spinner("Scraping job boards…"):
            raw = scrape_jobs(
                query, location, time_filter, is_remote=is_remote, distance_miles=distance
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
            # `if not df.empty` block below.
            st.session_state.results_df = pd.DataFrame()
            st.session_state.results_page = 1
            st.session_state.scored = False
        else:
            # Shield at the scrape boundary, before anything branches on whether we
            # can score. A posting is trapped or not regardless of whether the user
            # has uploaded a résumé, so the jd_flags column — and the red outline it
            # drives below — has to exist on every path. It used to be created
            # inside rank_jobs, which meant the warning silently vanished for anyone
            # browsing without a résumé, or whenever scoring errored out.
            raw = shield_frame(raw)

            resume = get_latest_resume()
            if resume:
                try:
                    with st.spinner("Filtering jobs based on résumé"):
                        raw = rank_jobs(raw, resume)
                    st.session_state.scored = True
                    st.success(f"Found and scored {len(raw)} jobs.")
                except Exception as e:
                    st.error(
                        f"Scoring failed — jobs are shown unranked. "
                        f"Check that your GROQ_API_KEY is valid and restart the app. ({e})"
                    )
                    raw["match_score"] = pd.NA
                    raw["match_reason"] = "scoring failed"
                    st.session_state.scored = False
            else:
                raw["match_score"] = pd.NA
                raw["match_reason"] = "Upload a resume on the Resume page to get AI match scores"
                st.session_state.scored = False
                st.warning("No resume found — showing all jobs unranked. Upload a resume to enable scoring.")

            st.session_state.results_df = raw
            st.session_state.results_page = 1  # a fresh result set starts at page 1

# ── Results table ──────────────────────────────────────────────────────────────
df = st.session_state.results_df
scored = st.session_state.scored

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
        base = df[
            ~df.apply(
                lambda r: job_signature(
                    r.get("company", ""), r.get("title", ""), r.get("location", "")
                )
                in applied_keys,
                axis=1,
            )
        ]

    if scored:
        view = base[base["match_score"] >= min_score]
        hidden = len(base) - len(view)
        st.subheader(f"Results — {len(view)} job(s) scoring ≥ {min_score}/100")
        if view.empty:
            st.info(
                f"No jobs scored ≥ {min_score}/100. "
                "Lower the minimum match score or broaden your search."
            )
        elif hidden:
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
        key = job_signature(
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
