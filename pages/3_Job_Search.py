import streamlit as st
import pandas as pd

from src.job_scraper import scrape_jobs, HOURS_OLD_MAP
from src.job_matcher import rank_jobs
from src.company_finder import find_company_job_url
from src.profile_manager import get_latest_resume, save_job

st.set_page_config(page_title="Job Search — JobSeeker", page_icon="🔍", layout="wide")
st.title("🔍 Job Search")

# ── Search inputs ──────────────────────────────────────────────────────────────
col1, col2, col3 = st.columns([3, 3, 2])
with col1:
    query = st.text_input("Job title / keywords", placeholder="Software Engineer")
with col2:
    location = st.text_input("Location", placeholder="New York, NY")
with col3:
    time_filter = st.selectbox("Posted within", list(HOURS_OLD_MAP.keys()), index=1)

col4, col5 = st.columns([2, 8])
with col4:
    is_remote = st.checkbox("Remote only")
with col5:
    search_clicked = st.button("Search", type="primary")

# ── Session-state storage ──────────────────────────────────────────────────────
if "results_df" not in st.session_state:
    st.session_state.results_df = pd.DataFrame()

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
                with st.spinner(f"Ranking {len(raw)} jobs with Claude…"):
                    ranked = rank_jobs(raw, resume)
            else:
                ranked = raw
                ranked["match_score"] = 0
                ranked["match_reason"] = "Upload a resume to get AI match scores"

            st.session_state.results_df = ranked
            st.success(f"Found {len(ranked)} jobs.")

# ── Results table ──────────────────────────────────────────────────────────────
df = st.session_state.results_df
if not df.empty:
    st.subheader("Results")
    for idx, row in df.iterrows():
        score = row.get("match_score", 0)
        score_color = "green" if score >= 70 else "orange" if score >= 40 else "red"

        with st.container(border=True):
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
                st.markdown(f"**Match:** :{score_color}[{score}/100]")
            with c3:
                job_url = row.get("job_url") or row.get("url", "")

                # Find on company site
                if st.button("Find on Company Site", key=f"find_{idx}"):
                    with st.spinner("Searching…"):
                        company_url = find_company_job_url(
                            row.get("company", ""), row.get("title", ""), job_url
                        )
                    df.at[idx, "company_url"] = company_url
                    st.session_state.results_df = df
                    st.rerun()

                company_url = row.get("company_url", "")
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
                        "match_score": int(score),
                        "match_reason": row.get("match_reason", ""),
                        "source": row.get("site", ""),
                        "posted_at": str(row.get("date_posted", "")),
                        "status": "saved",
                    }
                    save_job(job_record)
                    st.session_state["apply_job"] = job_record
                    st.switch_page("pages/4_Apply.py")
