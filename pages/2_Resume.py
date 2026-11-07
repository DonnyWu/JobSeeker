import logging

import streamlit as st

from src.import_guard import friendly_import_errors

with friendly_import_errors():
    from src import ui
    from src.profile_manager import get_latest_resume, save_resume
    from src.resume_parser import extract_text, parse_resume

log = logging.getLogger(__name__)


def _parse_upload(uploaded) -> str | None:
    """Parse and save an uploaded résumé. Returns an error message, or None on success.

    The file is read from memory and never written to disk. It used to be saved
    as uploads/<uploaded.name>, but that name is whatever the browser sends —
    Streamlit only checks that it ends in .pdf/.docx — so a name with ../ steps,
    an absolute path, or a //server/share path wrote the file outside uploads/
    (or made Windows connect to another machine). Nothing ever read that copy
    back, so there was nothing worth keeping.

    Failures show a one-line message; the details go to the terminal rather than
    the page, where a traceback exposes file paths and Groq's organization ID.
    """
    file_bytes = uploaded.read()
    try:
        raw_text = extract_text(file_bytes, uploaded.name)
    except Exception:
        log.exception("Could not extract text from the uploaded résumé")
        return "Couldn't read that file. Please upload a valid PDF or DOCX."
    try:
        parsed = parse_resume(raw_text)
    except RuntimeError as e:
        # _get_client's "GROQ_API_KEY is not set" — our own message, and the real fix.
        return str(e)
    except Exception:
        log.exception("Résumé parsing failed")
        return (
            "Couldn't parse the resume — the request to Groq failed. "
            "The details are in the terminal where the app is running."
        )
    save_resume(uploaded.name, raw_text, parsed)
    return None


_UPLOAD_KEY = "resume_upload"
_TRUST_NOTE = (
    ":material/lock: Read once in memory; the file itself is never saved to disk."
)


def _uploader():
    # No "doc": python-docx only reads .docx, so an old Word .doc crashed the page.
    # Keyed, so the same widget (and its file) is found whether it is drawn in the
    # upload card or inside the "Replace resume" expander.
    st.file_uploader("Upload PDF or DOCX", type=["pdf", "docx"], key=_UPLOAD_KEY)
    st.caption(_TRUST_NOTE)


ui.page_header("Resume", "JobSeeker scores every job against this resume.")

# ── Parse a new upload, once ──────────────────────────────────────────────────
# Done before anything is drawn, so the page below already shows the result.
# An uploaded file stays in the widget across reruns, and every click anywhere on
# the page is a rerun: without the file_id check each one repeated the Groq call
# and inserted another copy of the same resume.
notice = None
uploaded = st.session_state.get(_UPLOAD_KEY)
if uploaded is not None and st.session_state.get("parsed_upload_id") != uploaded.file_id:
    with st.spinner("Reading your resume…"):
        error = _parse_upload(uploaded)
    # Recorded on failure too: retrying the same file can't change the outcome,
    # and a missing API key would otherwise be re-reported on every click.
    st.session_state["parsed_upload_id"] = uploaded.file_id
    notice = ("error", error) if error else ("success", "Resume parsed and saved!")

resume = get_latest_resume()

if notice:
    kind, message = notice
    (st.error if kind == "error" else st.success)(message)

# ── No resume yet: the upload card is the whole page ──────────────────────────
if not resume:
    with st.container(border=True, key="card-upload"):
        st.subheader("Upload your resume", anchor=False)
        st.markdown(
            "JobSeeker reads your skills, roles, education and years of experience, "
            "then scores every job against them."
        )
        _uploader()
    st.stop()

# ── File card ─────────────────────────────────────────────────────────────────
with st.container(border=True, key="card-resume-file"):
    name_col, replace_col = st.columns([3, 2], vertical_alignment="top")
    with name_col:
        st.markdown(f":material/description: **{ui.md(resume.get('file_name') or 'Resume')}**")
        uploaded_on = ui.short_date(resume.get("uploaded_at"), with_year=True)
        if uploaded_on:
            st.caption(f"Uploaded {uploaded_on}")
    with replace_col:
        with st.expander("Replace resume", icon=":material/upload:"):
            _uploader()

# ── At a glance ───────────────────────────────────────────────────────────────
skills = resume.get("skills") or []
experience = resume.get("experience") or []
education = resume.get("education") or []
years = resume.get("total_years_experience")
try:
    years_label = (
        f"{float(years):g} {'yr' if float(years) == 1 else 'yrs'}" if years is not None else "—"
    )
except (TypeError, ValueError):
    years_label = "—"

with st.container(key="kpis"):
    m1, m2, m3, m4 = st.columns(4)
m1.metric("Experience", years_label, border=True)
m2.metric("Skills", len(skills), border=True)
m3.metric("Roles", len(experience), border=True)
m4.metric("Education", len(education), border=True)

# ── What was extracted ────────────────────────────────────────────────────────
left, right = st.columns([3, 2], gap="medium")

with left:
    with st.container(border=True, key="card-summary"):
        st.subheader("Summary", anchor=False)
        st.markdown(ui.md(resume.get("summary")) or ":gray[No summary was found.]")

    with st.container(border=True, key="card-experience"):
        st.subheader("Experience", anchor=False)
        if not experience:
            st.caption("No roles were found.")
        for exp in experience:
            title = ui.md(exp.get("title")) or "—"
            company = ui.md(exp.get("company"))
            st.markdown(f"**{title}**" + (f" · {company}" if company else ""))
            if exp.get("duration"):
                st.caption(ui.md(exp.get("duration")))
            bullets = exp.get("bullets") or []
            if bullets:
                st.markdown("\n".join(f"- {ui.md(b)}" for b in bullets))

with right:
    with st.container(border=True, key="card-skills"):
        st.subheader("Skills", anchor=False)
        if skills:
            st.markdown(" ".join(f":gray-badge[{ui.md(s)}]" for s in skills))
        else:
            st.caption("No skills were found.")

    with st.container(border=True, key="card-education"):
        st.subheader("Education", anchor=False)
        if not education:
            st.caption("No education was found.")
        for edu in education:
            st.markdown(f"**{ui.md(edu.get('degree')) or '—'}**")
            line = " · ".join(
                ui.md(v) for v in (edu.get("institution"), edu.get("year")) if v
            )
            if line:
                st.caption(line)
