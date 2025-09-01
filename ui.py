import streamlit as st
from ui_mustard_aphid import render_mustard_aphid
from ui_mustard_powder import render_mustard_powder
from ui_wheat_white_ear import render_wheat_white_ear  


# === Apply Full Width Layout Globally ===
st.set_page_config(
    page_title="CropsDiagnosis",
    layout="wide",
    page_icon="🌿"
)

# === MAIN FUNCTION ===
def render_mainpage():
    # === CSS Styling (optional - for button polish) ===
    st.markdown("""
        <style>
        .stButton > button {
            padding: 0.6rem 1.2rem;
            font-weight: bold;
            font-size: 1rem;
            border-radius: 8px;
        }
        </style>
    """, unsafe_allow_html=True)

    # === LOGOS (side by side, full width) ===
    logo_col1, logo_col2 = st.columns(2)
    with logo_col1:
        st.image("./assets/logo2.png", width=250, caption="Amity Centre for Artificial Intelligence\nAmity University Noida")
    with logo_col2:
        st.image("assets/nibsm_logo.jpg", width=250, caption="ICAR - National Institute of Biotic Stress Management\nChhattisgarh")

    st.markdown("---")

    # === Session State Initialization ===
    if "selected_module" not in st.session_state:
        st.session_state.selected_module = "Mustard_Aphid"

    # === HORIZONTAL BUTTONS: Aphid / Powder ===
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("🌿 Mustard Aphid", use_container_width=True):
            st.session_state.selected_module = "Mustard_Aphid"
            st.session_state.uploaded_file = None
            st.session_state.predicted_class = None

    with col2:
        if st.button("🌾 Mustard Powdery", use_container_width=True):
            st.session_state.selected_module = "Mustard_Powdery"
            st.session_state.uploaded_file = None
            st.session_state.predicted_class = None

    with col3:
        if st.button("🌾 Wheat White Ear", use_container_width=True):
            st.session_state.selected_module = "Wheat_White_ear"
            st.session_state.uploaded_file = None
            st.session_state.predicted_class = None

    st.markdown("---")

    # === LOAD SELECTED MODULE ===
    if st.session_state.selected_module == "Mustard_Aphid":
        render_mustard_aphid()
    elif st.session_state.selected_module == "Mustard_Powdery":
        render_mustard_powder()
    elif st.session_state.selected_module == "Wheat_White_ear":
        render_wheat_white_ear()
