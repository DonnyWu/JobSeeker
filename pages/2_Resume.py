import logging

import streamlit as st

from src.import_guard import friendly_import_errors

with friendly_import_errors():
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
            "Couldn't parse the résumé — the request to Groq failed. "
            "The details are in the terminal where the app is running."
        )
    save_resume(uploaded.name, raw_text, parsed)
    return None


st.set_page_config(page_title="Resume — JobSeeker", page_icon="📄", layout="centered")
st.title("📄 Resume")
st.markdown("Upload your resume to extract your skills and experience.")

# No "doc": python-docx only reads .docx, so an old Word .doc crashed the page.
uploaded = st.file_uploader("Upload PDF or DOCX", type=["pdf", "docx"])

if uploaded:
    with st.spinner("Extracting and parsing resume…"):
        error = _parse_upload(uploaded)
    if error:
        st.error(error)
    else:
        st.success("Resume parsed and saved!")

# Always show the latest parsed resume
resume = get_latest_resume()
if resume:
    st.subheader(f"Latest resume: {resume.get('file_name', '')}")

    with st.expander("Professional Summary"):
        st.write(resume.get("summary", "—"))

    with st.expander("Skills"):
        skills = resume.get("skills", [])
        if skills:
            st.write(", ".join(skills))
        else:
            st.write("—")

    with st.expander("Work Experience"):
        for exp in resume.get("experience", []):
            st.markdown(
                f"**{exp.get('title', '')}** @ {exp.get('company', '')}  \n"
                f"_{exp.get('duration', '')}_"
            )
            for bullet in exp.get("bullets", []):
                st.markdown(f"- {bullet}")

    with st.expander("Education"):
        for edu in resume.get("education", []):
            st.markdown(
                f"**{edu.get('degree', '')}** — {edu.get('institution', '')}  \n"
                f"_{edu.get('year', '')}_"
            )
else:
    st.info("No resume uploaded yet.")
