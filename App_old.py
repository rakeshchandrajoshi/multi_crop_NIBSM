import streamlit as st
from ui import render_mainpage
from ui_about import render_about



# --- Sidebar UI ---
st.sidebar.markdown("## 🌿 KrishakSakha 🌿")
st.sidebar.markdown("Empowering Farmers with AI-driven Crop Health Insights")

if "page" not in st.session_state:
    st.session_state.page = "Home"

if st.sidebar.button("🏠      Home       ", use_container_width=True):
    st.session_state.page = "Home"
if st.sidebar.button("📘 About the Project", use_container_width=True):
    st.session_state.page = "About"

# --- Page Routing ---
if st.session_state.page == "Home":
    render_mainpage()
elif st.session_state.page == "About":
    render_about()
