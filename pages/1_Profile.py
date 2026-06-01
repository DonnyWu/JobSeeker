import streamlit as st
from src.profile_manager import get_profile, save_profile

st.set_page_config(page_title="Profile — JobSeeker", page_icon="👤", layout="centered")
st.title("👤 Profile")
st.markdown("Your contact information is used to auto-fill job applications.")

profile = get_profile()

with st.form("profile_form"):
    name = st.text_input("Full Name", value=profile.get("name", ""))
    email = st.text_input("Email", value=profile.get("email", ""))
    phone = st.text_input("Phone", value=profile.get("phone", ""))
    current_company = st.text_input("Current Company", value=profile.get("current_company", ""))
    linkedin = st.text_input("LinkedIn URL", value=profile.get("linkedin", ""))
    portfolio = st.text_input("Portfolio / Website URL", value=profile.get("portfolio", ""))
    github = st.text_input("GitHub URL", value=profile.get("github", ""))

    submitted = st.form_submit_button("Save Profile")

if submitted:
    save_profile(
        {
            "name": name,
            "email": email,
            "phone": phone,
            "current_company": current_company,
            "linkedin": linkedin,
            "portfolio": portfolio,
            "github": github,
        }
    )
    st.success("Profile saved!")
