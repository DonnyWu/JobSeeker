"""My Jobs: every application and where it stands, plus jobs saved for later.

The nav label is "My Jobs"; the file keeps its old name so tests and history
don't move. Everything here reads and writes through src.profile_manager.
"""

import streamlit as st

from src.import_guard import friendly_import_errors

with friendly_import_errors():
    from src import ui
    from src.application_status import (
        APPLICATION_STATUSES,
        IN_PROCESS,
        label,
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

pages = ui.app_pages()

ui.page_header("My Jobs", "Track every application and keep promising roles for later.")

# A row's status is shown three ways, deliberately: the coloured bar down its
# left edge (CSS in src/ui.py, keyed on histcard-<status>-<id>), the badge
# beside the title, and the selected option in its status control. If a future
# Streamlit release breaks the CSS, the bar disappears and the other two remain.


def _job_key(job: dict) -> str:
    """The row's stored key, recomputed only for rows saved before it existed."""
    return job.get("job_key") or job_signature(
        job.get("company", ""), job.get("title", ""), job.get("location", "")
    )


def _link(url: str) -> str:
    """A markdown link to a posting. ")" and spaces would end the link early."""
    safe = url.replace(" ", "%20").replace(")", "%29")
    return f"[Open :material/open_in_new:]({safe})"


def _meta(job: dict, *extra: str) -> str:
    """The grey line under a title: where, which board, the score, then extras."""
    parts = [
        ui.md(job.get("location")),
        ui.site_label(job.get("source")),
        f"Match {job['match_score']}" if job.get("match_score") is not None else "",
        *extra,
    ]
    url = job.get("url") or ""
    if url:
        parts.append(_link(url))
    return " · ".join(p for p in parts if p)


def _title_line(job: dict) -> str:
    return f"**{ui.md(job.get('title') or '—')}** · {ui.md(job.get('company') or '—')}"


def _why(job: dict):
    """The match reason, one click away rather than a full-width row."""
    reason = (job.get("match_reason") or "").strip()
    if reason:
        with st.popover("Why it matched", type="tertiary", icon=":material/info:"):
            st.markdown(ui.md(reason))


def _set_status(job_key: str, widget_key: str):
    """Write the status the moment a different one is picked."""
    update_application_outcome(job_key, st.session_state[widget_key])


applied = get_applied_jobs()
saved = get_saved_jobs()

counts = {value: 0 for value in APPLICATION_STATUSES}
for job in applied:
    counts[normalize(job.get("outcome"))] += 1

# The point of recording a status at all: the shape of the search at a glance.
with st.container(key="kpis"):
    cols = st.columns(len(APPLICATION_STATUSES) + 1)
cols[0].metric("Applications", len(applied), border=True)
for col, value in zip(cols[1:], APPLICATION_STATUSES):
    col.metric(label(value), counts[value], border=True)

tab_applied, tab_saved = st.tabs([f"Applied · {len(applied)}", f"Saved · {len(saved)}"])

# ──────────────────────────────────────────────────────────────────────────────
# Applied: every job you've applied to, and where each one stands
# ──────────────────────────────────────────────────────────────────────────────
with tab_applied:
    if not applied:
        ui.empty_state(
            "No applied jobs yet",
            "Mark a job as applied from Job Search, or from the Saved tab, and it "
            "will show up here as Pending.",
            "work_history",
            link=pages["search"],
            link_label="Find jobs",
            key="applied",
        )
    else:
        filter_col, search_col = st.columns([3, 2], vertical_alignment="bottom")
        with filter_col:
            # Keyed, so its selection survives the counts in the labels changing.
            chosen = st.pills(
                "Status",
                ["all", *APPLICATION_STATUSES],
                format_func=lambda v: (
                    f"All {len(applied)}" if v == "all" else f"{label(v)} {counts[v]}"
                ),
                default="all",
                required=True,
                key="history_status_filter",
                label_visibility="collapsed",
            )
        with search_col:
            needle = st.text_input(
                "Search applications",
                key="history_search",
                placeholder="Search title, company or location",
                icon=":material/search:",
                label_visibility="collapsed",
            ).strip().lower()

        rows = [
            job
            for job in applied
            if (chosen in (None, "all") or normalize(job.get("outcome")) == chosen)
            and (
                not needle
                or needle in " ".join(
                    str(job.get(f) or "") for f in ("title", "company", "location")
                ).lower()
            )
        ]
        if not rows:
            st.info("No applications match these filters.")

        for job in rows:
            key = _job_key(job)
            jid = job.get("id")
            status = normalize(job.get("outcome"))
            when = ui.short_date(job.get("applied_at"))

            with st.container(border=True, key=f"histcard-{status}-{jid}"):
                head, badge = st.columns([5, 1.4], vertical_alignment="center")
                head.markdown(_title_line(job))
                badge.markdown(ui.status_badge(status), text_alignment="right")
                st.caption(_meta(job, f"Applied {when}" if when else ""))

                status_widget = f"status_{jid}"
                with st.container(horizontal=True, vertical_alignment="center"):
                    st.segmented_control(
                        "Status",
                        APPLICATION_STATUSES,
                        format_func=label,
                        default=status,
                        required=True,
                        key=status_widget,
                        on_change=_set_status,
                        args=(key, status_widget),
                        label_visibility="collapsed",
                    )
                    _why(job)

                # While you're interviewing, record which round you're on.
                if status == IN_PROCESS:
                    stage_col, save_col = st.columns([4, 1], vertical_alignment="bottom")
                    stage_input = stage_col.text_input(
                        "Current stage",
                        value=job.get("interview_stage") or "",
                        key=f"stage_{jid}",
                        placeholder="e.g. Round 2, Onsite, Final round",
                        label_visibility="collapsed",
                    )
                    if save_col.button(
                        "Save note", key=f"savestage_{jid}", width="stretch",
                        icon=":material/save:",
                    ):
                        update_application_outcome(key, IN_PROCESS, stage_input)
                        ui.flash("Stage note saved", icon=":material/check:")
                        st.rerun()

# ──────────────────────────────────────────────────────────────────────────────
# Saved: found and kept, not applied to yet
# ──────────────────────────────────────────────────────────────────────────────
with tab_saved:
    if not saved:
        ui.empty_state(
            "Nothing saved yet",
            "Press Save on a listing in Job Search and it will be waiting here.",
            "bookmark",
            link=pages["search"],
            link_label="Find jobs",
            key="saved",
        )
    else:
        st.caption(
            f"{ui.plural(len(saved), 'saved job')}. Marking one as applied moves it to "
            "the Applied tab as Pending."
        )
        for job in saved:
            key = _job_key(job)
            jid = job.get("id")
            title = job.get("title") or "this job"

            with st.container(border=True, key=f"savedcard-{jid}"):
                st.markdown(_title_line(job))
                st.caption(_meta(job))
                with st.container(horizontal=True, gap="small"):
                    if st.button(
                        "Mark as applied", key=f"saved_apply_{jid}", type="primary",
                        icon=":material/check_circle:",
                    ):
                        mark_job_applied(job)
                        ui.flash(f"Moved {title} to Applied", icon=":material/check:")
                        st.rerun()
                    if st.button(
                        "Remove", key=f"saved_remove_{jid}", type="tertiary",
                        icon=":material/delete:",
                    ):
                        delete_saved_job(key)
                        ui.flash(f"Removed {title}", icon=":material/delete:")
                        st.rerun()
                    _why(job)
