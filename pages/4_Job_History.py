import streamlit as st

from src.import_guard import friendly_import_errors

with friendly_import_errors():
    from src.application_status import (
        APPLICATION_STATUSES,
        DISPLAY,
        IN_PROCESS,
        badge,
        normalize,
    )
    from src.profile_manager import (
        delete_saved_job,
        get_applied_jobs,
        get_saved_jobs,
        job_signature,
        mark_job_applied,
        update_application_outcome,
    )

st.set_page_config(page_title="Job History — JobSeeker", page_icon="📋", layout="wide")
st.title("📋 Job History")

# A keyed container carries an ``st-key-<key>`` CSS class, so building the card key
# out of the job's status is all it takes to colour-code the list. Same trick the
# Job Search page uses to outline a trapped posting in red — see the long comment
# above the equivalent block in pages/3_Job_Search.py for why the box-shadow does
# the real work rather than the border.
st.html(
    "<style>"
    + "".join(
        f"div[class*='st-key-histcard-{value}-']"
        f"{{box-shadow:0 0 0 2px {color};border-radius:0.5rem;}}"
        f"div[class*='st-key-histcard-{value}-'],"
        f"div[class*='st-key-histcard-{value}-'] > div,"
        f"div[class*='st-key-histcard-{value}-'] .stVerticalBlock"
        f"{{border-color:{color} !important;}}"
        for value, (_label, _emoji, color) in DISPLAY.items()
    )
    + "</style>"
)


def _job_key(job: dict) -> str:
    """The row's stored key, recomputed only for rows saved before it existed."""
    return job.get("job_key") or job_signature(
        job.get("company", ""), job.get("title", ""), job.get("location", "")
    )


def _headline(job: dict):
    """Title, company and the one-line context both tabs show above the buttons."""
    st.markdown(f"**{job.get('title') or '—'}** — {job.get('company') or '—'}")
    st.caption(
        f"{job.get('location') or '—'} · "
        f"{job.get('source') or '—'} · "
        f"Match {job.get('match_score') if job.get('match_score') is not None else '—'}"
    )
    reason = job.get("match_reason") or ""
    if reason:
        st.caption(f"_{reason}_")
    url = job.get("url") or ""
    if url:
        st.markdown(f"[Open posting]({url})")


tab_applied, tab_saved = st.tabs(["Applied", "Saved Jobs"])

# ──────────────────────────────────────────────────────────────────────────────
# Tab 1 — Applied: every job you've applied to, and where each one stands
# ──────────────────────────────────────────────────────────────────────────────
with tab_applied:
    applied = get_applied_jobs()

    if not applied:
        st.info(
            "No applied jobs yet. Mark a job as applied from **Job Search**, or from "
            "the **Saved Jobs** tab, and it'll show up here."
        )
    else:
        # The point of recording a status at all: the shape of the search at a glance.
        counts = {value: 0 for value in APPLICATION_STATUSES}
        for job in applied:
            counts[normalize(job.get("outcome"))] += 1
        st.caption(
            f"{len(applied)} application(s) — "
            + " · ".join(f"{counts[v]} {DISPLAY[v][0].lower()}" for v in APPLICATION_STATUSES)
        )

        for job in applied:
            key = _job_key(job)
            jid = job.get("id")
            status = normalize(job.get("outcome"))

            with st.container(border=True, key=f"histcard-{status}-{jid}"):
                c1, c2 = st.columns([3, 2])
                with c1:
                    _headline(job)
                    applied_at = (job.get("applied_at") or "")[:10]
                    if applied_at:
                        st.caption(f"Applied: {applied_at}")
                with c2:
                    # Deliberately redundant with the coloured ring. If a future
                    # Streamlit release changes the container markup the outline
                    # silently stops working, and the failure mode has to be "no
                    # colour", not "no idea where this application stands".
                    st.markdown(f"### {badge(status)}")
                    stage = (job.get("interview_stage") or "").strip()
                    if status == IN_PROCESS and stage:
                        st.caption(stage)

                cols = st.columns(len(APPLICATION_STATUSES))
                for col, value in zip(cols, APPLICATION_STATUSES):
                    label, emoji, _color = DISPLAY[value]
                    # The current status is shown pressed-out rather than hidden, so
                    # the four options stay in the same place on every card.
                    if col.button(
                        f"{emoji} {label}",
                        key=f"status_{value}_{jid}",
                        disabled=(value == status),
                        use_container_width=True,
                    ):
                        update_application_outcome(key, value)
                        st.rerun()

                # While you're interviewing, record which round you're on.
                if status == IN_PROCESS:
                    sc1, sc2 = st.columns([4, 1])
                    stage_input = sc1.text_input(
                        "Current stage",
                        value=job.get("interview_stage") or "",
                        key=f"stage_{jid}",
                        placeholder="e.g. Round 2, Onsite, Final round",
                        label_visibility="collapsed",
                    )
                    if sc2.button("Save note", key=f"savestage_{jid}", use_container_width=True):
                        update_application_outcome(key, IN_PROCESS, stage_input)
                        st.rerun()

# ──────────────────────────────────────────────────────────────────────────────
# Tab 2 — Saved Jobs: found and kept, not applied to yet
# ──────────────────────────────────────────────────────────────────────────────
with tab_saved:
    saved = get_saved_jobs()

    if not saved:
        st.info(
            "Nothing saved yet. Hit **💾 Save** on a listing in **Job Search** and it'll "
            "be waiting here."
        )
    else:
        st.caption(
            f"{len(saved)} saved job(s). Marking one as applied moves it to the "
            "**Applied** tab as Pending."
        )

        for job in saved:
            key = _job_key(job)
            jid = job.get("id")

            with st.container(border=True, key=f"savedcard-{jid}"):
                c1, c2 = st.columns([3, 2])
                with c1:
                    _headline(job)
                with c2:
                    if st.button(
                        "✓ Mark as applied",
                        key=f"saved_apply_{jid}",
                        type="primary",
                        use_container_width=True,
                    ):
                        mark_job_applied(job)
                        st.rerun()
                    if st.button(
                        "🗑 Remove",
                        key=f"saved_remove_{jid}",
                        use_container_width=True,
                    ):
                        delete_saved_job(key)
                        st.rerun()
