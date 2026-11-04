"""Profile: your contact details, kept ready to paste into job applications.

Nothing in the app fills forms for you any more (auto-fill was retired, see
archive/README.md), so the page's job is the copy panel on the right.
"""

import streamlit as st

from src.import_guard import friendly_import_errors

with friendly_import_errors():
    from src import ui
    from src.profile_manager import get_profile, save_profile

# (field, label, placeholder), in the order they are shown.
_CONTACT = [
    ("name", "Full name", "Jordan Lee"),
    ("email", "Email", "jordan@example.com"),
    ("phone", "Phone", "(555) 555-0123"),
    ("current_company", "Current company", "Acme Inc."),
]
_LINKS = [
    ("linkedin", "LinkedIn", "https://www.linkedin.com/in/you"),
    ("github", "GitHub", "https://github.com/you"),
    ("portfolio", "Portfolio or website", "https://you.dev"),
]
_FIELDS = _CONTACT + _LINKS

ui.page_header("Profile", "Your contact details, ready to paste into job applications.")

profile = get_profile()

form_col, copy_col = st.columns([3, 2], gap="medium")

with form_col:
    with st.form("profile_form", border=True):
        values = {}

        def _inputs(fields):
            # Two per row; a lone last field takes the left half.
            for i in range(0, len(fields), 2):
                cols = st.columns(2)
                for col, (field, label, placeholder) in zip(cols, fields[i : i + 2]):
                    values[field] = col.text_input(
                        label,
                        value=profile.get(field) or "",
                        placeholder=placeholder,
                        key=f"profile_{field}",
                    )

        st.subheader("Contact", anchor=False)
        _inputs(_CONTACT)
        st.subheader("Links", anchor=False)
        _inputs(_LINKS)

        submitted = st.form_submit_button(
            "Save changes", type="primary", key="profile_save", icon=":material/save:"
        )

if submitted:
    save_profile({field: (values[field] or "").strip() for field, _, _ in _FIELDS})
    ui.flash("Profile saved", icon=":material/check:")
    st.rerun()

with copy_col:
    with st.container(border=True, key="card-copy"):
        st.subheader("Copy-ready", anchor=False)
        filled = [(label, (profile.get(field) or "").strip()) for field, label, _ in _FIELDS]
        filled = [(label, value) for label, value in filled if value]
        if filled:
            st.caption("Hover a value and use its copy button.")
            for label, value in filled:
                st.caption(label)
                st.code(value, language=None)
        else:
            st.caption("Save your details and they'll appear here, one click to copy.")
        st.progress(
            len(filled) / len(_FIELDS),
            text=f"{len(filled)} of {len(_FIELDS)} filled",
        )
