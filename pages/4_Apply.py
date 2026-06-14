import threading

import streamlit as st

from src.profile_manager import (
    get_profile,
    mark_job_applied,
    get_applied_keys,
    get_applied_jobs,
    update_application_outcome,
    job_signature,
)
from src.autofill import run_autofill

st.set_page_config(page_title="Apply — JobSeeker", page_icon="✅", layout="wide")
st.title("✅ Apply")

tab_apply, tab_applied = st.tabs(["Apply", "Applied"])

# ──────────────────────────────────────────────────────────────────────────────
# Tab 1 — Apply to the job currently selected from Job Search
# ──────────────────────────────────────────────────────────────────────────────
with tab_apply:
    job = st.session_state.get("apply_job")
    profile = get_profile()

    if not job:
        st.info("No job selected. Go to **Job Search** and click **Auto-Apply** on a listing.")
    else:
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
            st.success(
                "✅ You've marked this job as applied. Track its outcome on the **Applied** tab."
            )
        else:
            st.markdown("Once you've submitted the application, confirm it here so it's flagged next time:")
            if st.button("✅ I've applied — mark it", type="primary"):
                mark_job_applied(job)
                st.success("Marked as applied! Track its outcome on the **Applied** tab.")
                st.rerun()

# ──────────────────────────────────────────────────────────────────────────────
# Tab 2 — Applied: every job you've applied to, with outcome tracking
# ──────────────────────────────────────────────────────────────────────────────
with tab_applied:
    st.subheader("Jobs you've applied to")
    st.caption("Record what happened after applying — used for your job-search analytics.")

    applied = get_applied_jobs()

    if not applied:
        st.info(
            "No applied jobs yet. Mark a job on the **Apply** tab (or use **Auto-Apply** "
            "in Job Search) and it'll show up here."
        )
    else:
        # outcome value → (badge label, streamlit renderer)
        _OUTCOME_BADGES = {
            "interview": ("🎯 Interview", st.success),
            "offer": ("💼 Offer", st.success),
            "accepted": ("✅ Accepted", st.success),
            "declined": ("❌ Declined", st.error),
        }

        for job in applied:
            key = job.get("job_key") or job_signature(
                job.get("company", ""), job.get("title", ""), job.get("location", "")
            )
            jid = job.get("id")
            outcome = (job.get("outcome") or "").strip()

            with st.container(border=True):
                c1, c2 = st.columns([3, 2])
                with c1:
                    st.markdown(f"**{job.get('title', '—')}** — {job.get('company', '—')}")
                    st.caption(
                        f"{job.get('location') or '—'} · "
                        f"{job.get('source') or '—'} · "
                        f"Match {job.get('match_score', '—')}"
                    )
                    applied_at = (job.get("applied_at") or "")[:10]
                    if applied_at:
                        st.caption(f"Applied: {applied_at}")
                    url = job.get("url", "")
                    if url:
                        st.markdown(f"[Open posting]({url})")
                with c2:
                    if outcome in _OUTCOME_BADGES:
                        label, render = _OUTCOME_BADGES[outcome]
                        stage = (job.get("interview_stage") or "").strip()
                        if outcome == "interview" and stage:
                            label = f"{label} — {stage}"
                        render(label)
                    else:
                        st.info("Applied — no update yet")

                # Outcome buttons
                b1, b2, b3, b4 = st.columns(4)
                if b1.button("🎯 Interview", key=f"oc_interview_{jid}"):
                    update_application_outcome(key, "interview")
                    st.rerun()
                if b2.button("💼 Offer", key=f"oc_offer_{jid}"):
                    update_application_outcome(key, "offer")
                    st.rerun()
                if b3.button("✅ Accepted", key=f"oc_accepted_{jid}"):
                    update_application_outcome(key, "accepted")
                    st.rerun()
                if b4.button("❌ Declined", key=f"oc_declined_{jid}"):
                    update_application_outcome(key, "declined")
                    st.rerun()

                # When you're interviewing, capture which stage/round you're on.
                if outcome == "interview":
                    sc1, sc2 = st.columns([4, 1])
                    stage_input = sc1.text_input(
                        "Current interview stage",
                        value=job.get("interview_stage") or "",
                        key=f"stage_{jid}",
                        placeholder="e.g. Round 2, Onsite, Final round",
                        label_visibility="collapsed",
                    )
                    if sc2.button("Save", key=f"savestage_{jid}"):
                        update_application_outcome(key, "interview", stage_input)
                        st.rerun()
