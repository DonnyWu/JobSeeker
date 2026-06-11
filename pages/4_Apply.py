import threading

import streamlit as st

from src.profile_manager import (
    get_profile,
    mark_job_applied,
    get_applied_keys,
    job_signature,
)
from src.autofill import run_autofill

st.set_page_config(page_title="Apply — JobSeeker", page_icon="✅", layout="wide")
st.title("✅ Apply")

job = st.session_state.get("apply_job")
profile = get_profile()

if not job:
    st.info("No job selected. Go to **Job Search** and click **Auto-Apply** on a listing.")
    st.stop()

if not profile:
    st.warning("Profile is empty. Please fill in your profile first.")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Job Details")
    st.markdown(f"**Title:** {job.get('title', '—')}")
    st.markdown(f"**Company:** {job.get('company', '—')}")
    st.markdown(f"**Location:** {job.get('location', '—')}")
    st.markdown(f"**Match Score:** {job.get('match_score', '—')}/100")
    st.markdown(f"**Reason:** _{job.get('match_reason', '—')}_")
    url = job.get("url", "")
    if url:
        st.markdown(f"**URL:** [{url}]({url})")

with col2:
    st.subheader("Your Profile")
    for key, label in [
        ("name", "Name"),
        ("email", "Email"),
        ("phone", "Phone"),
        ("current_company", "Current Company"),
        ("linkedin", "LinkedIn"),
        ("portfolio", "Portfolio"),
        ("github", "GitHub"),
    ]:
        val = profile.get(key, "")
        st.markdown(f"**{label}:** {val or '—'}")

st.divider()

st.markdown(
    "Clicking **Launch Auto-Fill** opens a headed Chromium window, fills detected form fields "
    "with your profile, then **pauses** so you can review before clicking Submit."
)

if st.button("Launch Auto-Fill", type="primary"):
    if not url:
        st.error("No application URL found for this job.")
    elif not profile:
        st.error("Please save your profile before auto-filling.")
    else:
        st.info("Opening Chromium… Switch to the browser window that appears.")
        # Run in a separate thread so Streamlit doesn't block
        t = threading.Thread(target=run_autofill, args=(url, profile), daemon=True)
        t.start()

st.divider()

# ── Confirm you've applied (marks the job so it's flagged in future searches) ──
job_key = job_signature(job.get("company", ""), job.get("title", ""), job.get("location", ""))
already_applied = job_key in get_applied_keys()

if already_applied:
    st.success("✅ You've marked this job as applied. It'll show an **Applied** badge in Job Search.")
else:
    st.markdown("Once you've submitted the application, confirm it here so it's flagged next time:")
    if st.button("✅ I've applied — mark it", type="primary"):
        mark_job_applied(job)
        st.success("Marked as applied! It'll show an **Applied** badge in Job Search.")
        st.rerun()
