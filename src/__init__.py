import os

from dotenv import load_dotenv

# Load .env once for the whole process, regardless of which page is the entry point.
# Streamlit only runs the active page's script, so a sub-page opened directly (e.g.
# Job Search) would otherwise never trigger app.py's load_dotenv() and GROQ_API_KEY
# would be missing. Every page imports from `src`, so loading it here covers them all.
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))
