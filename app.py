"""JobSeeker's entry point: `uv run streamlit run app.py`.

This file is the router. It sets up what every page shares (the database, the
theme's CSS, the logo, the top navigation bar) and then runs whichever page the
URL points at. The pages themselves live in pages/ and are listed, in order, by
``src.ui.app_pages()``.

Because the app calls ``st.navigation``, Streamlit no longer builds the menu from
whatever happens to be in pages/. A file in there that ``app_pages()`` doesn't
list is simply not part of the app.
"""

from dotenv import load_dotenv
import streamlit as st

load_dotenv()

from src.import_guard import friendly_import_errors

with friendly_import_errors():
    from src.profile_manager import init_db
    from src import ui

init_db()

# The favicon has to be set *before* st.navigation. If the navigation message
# reaches the browser first, Streamlit uses the current page's nav icon as the
# favicon, and for a ":material/...:" icon that means fetching an SVG from
# fonts.gstatic.com — the one request that would leave this machine.
st.set_page_config(page_icon=ui.LOGO_MARK, layout="wide")

pages = ui.app_pages()
pg = st.navigation(list(pages.values()), position="top")

st.set_page_config(page_title=f"{pg.title} · JobSeeker")
st.logo(ui.LOGO, size="large")
ui.inject_styles()
ui.show_flash()

pg.run()
