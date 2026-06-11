import streamlit as st
import pandas as pd

from src.job_scraper import scrape_jobs, HOURS_OLD_MAP
from src.job_matcher import rank_jobs, generate_why_interested
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
        if st.button(
            "Generate answer",
            key=f"whybtn_{idx}",
            disabled=not has_resume,
            help=None if has_resume else "Upload a résumé on the Resume page first.",
        ):
            with st.spinner("Writing a tailored answer…"):
                try:
                    st.session_state[why_key] = generate_why_interested(resume, row.to_dict())
                except Exception as e:
                    st.session_state[why_key] = f"(Generation failed: {e})"
            st.session_state.pop(f"whytext_{idx}", None)  # let text_area reseed
        if not has_resume:
            st.caption("Upload a résumé on the Resume page to enable this.")
        if why_key in st.session_state:
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
        }
    )


col1, col2, col3 = st.columns([3, 3, 2])
with col1:
    query = st.text_input(
        "Job title / keywords", value=prefs.get("query", ""), placeholder="Software Engineer",
        key="search_query", on_change=_persist_search_prefs,
    )
with col2:
    location = st.text_input(
        "Location", value=prefs.get("location", ""), placeholder="New York, NY",
        key="search_location", on_change=_persist_search_prefs,
    )
with col3:
    time_filter = st.selectbox(
        "Posted within",
        _tf_options,
        index=_tf_options.index(_saved_tf) if _saved_tf in _tf_options else 1,
        key="search_time_filter", on_change=_persist_search_prefs,
    )

col4, col5, col6 = st.columns([2, 4, 4])
with col4:
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
with col5:
    min_score = st.slider(
        "Minimum match score", 0, 100, _ms if _ms is not None else 50, step=5,
        help="Only show jobs scoring at least this. Adjust to re-filter without re-searching.",
        key="search_min_score", on_change=_persist_search_prefs,
    )
with col6:
    st.write("")  # spacer to align the button with the inputs
    search_clicked = st.button("Search", type="primary")

# ── Session-state storage ──────────────────────────────────────────────────────
if "results_df" not in st.session_state:
    st.session_state.results_df = pd.DataFrame()
if "scored" not in st.session_state:
    st.session_state.scored = False

if search_clicked:
    if not query:
        st.warning("Please enter a job title or keywords.")
    else:
        with st.spinner("Scraping job boards…"):
            raw = scrape_jobs(query, location, time_filter, is_remote=is_remote)

        if raw.empty:
            st.warning("No jobs found. Try broader search terms or a longer time window.")
        else:
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

# ── Results table ──────────────────────────────────────────────────────────────
df = st.session_state.results_df
scored = st.session_state.scored

if not df.empty:
    if scored:
        view = df[df["match_score"] >= min_score]
        hidden = len(df) - len(view)
        st.subheader(f"Results — {len(view)} job(s) scoring ≥ {min_score}/100")
        if view.empty:
            st.info(
                f"No jobs scored ≥ {min_score}/100. "
                "Lower the minimum match score or broaden your search."
            )
        elif hidden:
            st.caption(f"{hidden} lower-scoring job(s) hidden. Lower the slider to see them.")
    else:
        view = df
        st.subheader(f"Results — {len(view)} job(s) (unranked)")

    resume = get_latest_resume()
    has_resume = bool(resume)
    applied_keys = get_applied_keys()

    for idx, row in view.iterrows():
        key = job_signature(
            row.get("company", ""), row.get("title", ""), row.get("location", "")
        )
        is_applied = key in applied_keys
        if is_applied and hide_applied:
            continue

        raw_score = row.get("match_score")
        has_score = raw_score is not None and not pd.isna(raw_score)
        if has_score:
            score = int(round(float(raw_score)))
            score_color = "green" if score >= 70 else "orange" if score >= 40 else "red"
            score_display = f":{score_color}[{score}/100]"
        else:
            score = 0
            score_display = ":gray[—]"

        with st.container(border=True):
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
                        "match_reason": row.get("match_reason", ""),
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
                        mark_job_applied(row.to_dict())
                        st.rerun()

            _render_company_section(idx, row, resume, has_resume)
