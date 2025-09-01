import streamlit as st

def render_about():
    st.title("🤝 Collaborative Project")

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
    st.markdown("### Project Overview")
    st.markdown("""
This project is a **collaborative initiative** between:

- 🧠 **Amity Centre for Artificial Intelligence (ACAI)**, Amity University Noida  
- 🌾 **ICAR - National Institute of Biotic Stress Management (NIBSM)**, Chhattisgarh

The tool aids in mustard crop health analysis by identifying **aphid severity stages** using deep learning and attention mechanisms.
""")

    st.markdown("### 👨‍💻 Contributors")
    st.markdown("""
- **Dr. Rakesh Chandra Joshi**, **Prof. M. K. Dutta**, **Mr. Suman Kumar**  
  *(Amity Centre for Artificial Intelligence (ACAI) - Amity University Noida)*

- **Dr. Pankaj Sharma**, **Dr. Arkaprava Roy**, **Dr. R.K. Murali Baskaran**, **Mr. Yogesh Kumar Chelak**  
  *(ICAR - NIBSM)*
""")
