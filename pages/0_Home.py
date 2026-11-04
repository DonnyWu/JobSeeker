"""Home: where you are in the job hunt, and the next thing worth doing.

Reads only — every number here comes from the same tables the other pages write,
through functions that already existed for them.
"""

import streamlit as st

from src.import_guard import friendly_import_errors

with friendly_import_errors():
    from src import ui
    from src.application_status import ACCEPTED, IN_PROCESS, normalize
    from src.profile_manager import (
        get_applied_jobs,
        get_latest_resume,
        get_profile,
        get_saved_jobs,
        get_search_prefs,
    )

pages = ui.app_pages()

profile = get_profile()
resume = get_latest_resume()
applied = get_applied_jobs()
saved = get_saved_jobs()
prefs = get_search_prefs()

first_name = ((profile.get("name") or "").strip().split() or [""])[0]
ui.page_header(
    f"Welcome back, {first_name}" if first_name else "Welcome to JobSeeker",
    "Find roles that fit your resume and keep every application on track.",
)

# ── Get set up (only while something is still missing) ──────────────────────
steps = [
    (
        bool(resume),
        "Upload your resume",
        "Every match score is worked out against it.",
        pages["resume"],
        "Upload resume",
    ),
    (
        bool((profile.get("name") or "").strip() and (profile.get("email") or "").strip()),
        "Add your contact details",
        "Your name and email, ready to paste into applications.",
        pages["profile"],
        "Open Profile",
    ),
    (
        bool((prefs.get("query") or "").strip() or applied or saved),
        "Run your first search",
        "Five job boards at once, ranked against your resume.",
        pages["search"],
        "Start a search",
    ),
]
done = sum(1 for ok, *_ in steps if ok)

if done < len(steps):
    with st.container(border=True, key="card-setup"):
        st.subheader("Get set up", anchor=False)
        st.progress(done / len(steps), text=f"{done} of {len(steps)} done")
        for ok, title, why, page, cta in steps:
            icon_col, text_col, link_col = st.columns(
                [0.5, 7, 2.5], vertical_alignment="center"
            )
            icon_col.markdown(
                ":green[:material/check_circle:]"
                if ok
                else ":gray[:material/radio_button_unchecked:]"
            )
            with text_col:
                st.markdown(f":gray[{title}]" if ok else f"**{title}**")
                st.caption(why)
            if not ok:
                with link_col:
                    st.page_link(page, label=cta, icon=":material/arrow_forward:")

# ── Where things stand ───────────────────────────────────────────────────────
statuses = [normalize(j.get("outcome")) for j in applied]
with st.container(key="kpis"):
    k1, k2, k3, k4 = st.columns(4)
k1.metric("Applications", len(applied), border=True)
k2.metric("In-process", statuses.count(IN_PROCESS), border=True)
k3.metric("Accepted", statuses.count(ACCEPTED), border=True)
k4.metric("Saved", len(saved), border=True)

left, right = st.columns([3, 2], gap="medium")

with left:
    with st.container(border=True, key="card-recent"):
        st.subheader("Recent applications", anchor=False)
        if not applied:
            st.caption(
                "Nothing yet. Mark a job as applied from Job Search and it shows up here."
            )
        for job in applied[:5]:
            when = ui.short_date(job.get("applied_at"))
            st.markdown(
                f"**{ui.md(job.get('title') or '—')}** · {ui.md(job.get('company') or '—')}  \n"
                f"{ui.status_badge(job.get('outcome'))}"
                + (f" &nbsp;:gray[Applied {when}]" if when else "")
            )
        if applied:
            st.page_link(
                pages["my_jobs"],
                label="View all in My Jobs",
                icon=":material/arrow_forward:",
            )

with right:
    with st.container(border=True, key="card-actions"):
        st.subheader("Quick actions", anchor=False)
        st.page_link(pages["search"], label="Start a search", icon=":material/search:")
        if saved:
            st.page_link(
                pages["my_jobs"],
                label=f"Review {ui.plural(len(saved), 'saved job')}",
                icon=":material/bookmark:",
            )

    with st.container(border=True, key="card-resume"):
        st.markdown("**Your resume**")
        if resume:
            st.markdown(f":material/description: {ui.md(resume.get('file_name') or 'Resume')}")
            uploaded = ui.short_date(resume.get("uploaded_at"), with_year=True)
            if uploaded:
                st.caption(f"Uploaded {uploaded}")
            st.page_link(pages["resume"], label="Update", icon=":material/upload:")
        else:
            st.caption("No resume yet, so jobs can't be scored.")
            st.page_link(pages["resume"], label="Upload resume", icon=":material/upload:")
