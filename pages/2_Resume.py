import os

import streamlit as st

from src.profile_manager import get_latest_resume, save_resume
from src.resume_parser import extract_text, parse_resume

UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "..", "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

st.set_page_config(page_title="Resume — JobSeeker", page_icon="📄", layout="centered")
st.title("📄 Resume")
st.markdown("Upload your resume to extract your skills and experience.")

uploaded = st.file_uploader("Upload PDF or DOCX", type=["pdf", "docx", "doc"])

if uploaded:
    file_bytes = uploaded.read()

    # Save to uploads/
    dest = os.path.join(UPLOAD_DIR, uploaded.name)
    with open(dest, "wb") as f:
        f.write(file_bytes)

    with st.spinner("Extracting and parsing resume…"):
        raw_text = extract_text(file_bytes, uploaded.name)
        parsed = parse_resume(raw_text)
        save_resume(uploaded.name, raw_text, parsed)

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
