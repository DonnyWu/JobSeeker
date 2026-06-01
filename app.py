import os

from dotenv import load_dotenv
import streamlit as st

load_dotenv()

from src.profile_manager import init_db

init_db()

st.set_page_config(
    page_title="JobSeeker",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("💼 JobSeeker")
st.markdown(
    "AI-powered job matching and auto-apply. Use the sidebar to navigate between pages."
)

st.info(
    "**Get started:**\n"
    "1. **Profile** — enter your contact info\n"
    "2. **Resume** — upload your PDF or DOCX resume\n"
    "3. **Job Search** — search and rank matching jobs\n"
    "4. **Apply** — auto-fill applications with Playwright"
)
