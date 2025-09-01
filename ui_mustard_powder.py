import streamlit as st
from PIL import Image
import torch
from io import BytesIO
import base64

from Model_logic_mustard_powder import (
    load_model,
    preprocess_image,
    class_names,
    remedies,
    original_labels,
    amp_autocast
)

# --- Helper: Display Fixed Size Image ---
def display_image_fixed_size(image, width, height):
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    img_bytes = buffered.getvalue()
    img_base64 = base64.b64encode(img_bytes).decode()

    st.markdown(f"""
        <div style="text-align: center;">
            <img src="data:image/png;base64,{img_base64}" width="{width}" height="{height}" 
                 style="border-radius: 10px; border: 1px solid #ddd;" />
        </div>
    """, unsafe_allow_html=True)

# --- Helper: Shared Prediction Logic ---
def predict_and_display(image):
    model = load_model()
    input_tensor = preprocess_image(image)

    with st.spinner("🔍 Analyzing image..."):
        with amp_autocast():
            output = model(input_tensor)
            pred_idx = output.argmax(1).item()

    predicted_class = class_names[pred_idx]
    remedy = remedies[original_labels[pred_idx]]

    st.success(f"### 🧪 Diagnosis: `{predicted_class}`")
    st.info(f"💡 **Recommendation**: {remedy}")

# --- Main UI Function ---
def render_mustard_powder():
    st.markdown("""
        <h1 style='text-align: center; font-size: 32px;'>
            🍃 <span style='font-family:Georgia;'><em>PowderyScan</em></span>: Mustard Powdery Infestation Analyzer
        </h1>
    """, unsafe_allow_html=True)

    st.markdown("""
        <p style='text-align: center; font-size: 18px;'>
            An AI-powered tool for detecting <strong>powdery mildew infestation stages</strong> or confirming a 
            <strong>healthy mustard plant</strong>.
        </p>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 📸 Upload a Sample Image")

    with st.expander("ℹ️ Sample Image Upload Guidelines", expanded=False):
        st.markdown("""
        - Make sure the plant is clearly visible (leaves & stems).
        - Use images in **JPG** or **PNG** format.
        - Keep file size under **200MB** for smooth upload.
        """)

    # --- File Upload Section ---
    uploaded_file = st.file_uploader("📤 Drag and drop or browse files", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        display_image_fixed_size(image, width=800, height=300)
        st.markdown("---")

        if st.button("🔍 Predict Diagnosis (Uploaded Image)", use_container_width=True):
            predict_and_display(image)

    # --- Sample Dropdown Selection ---
    st.markdown("### 🎯 Or select from sample images")
    sample_images = {
        "Sample 1": "Sample/mustard_powdery/IMG_20250224_105252.jpg",
        "Sample 2": "Sample/mustard_powdery/DSC_0027.jpg",
        "Sample 3": "Sample/mustard_powdery/DSC_0064.JPG",
        "Sample 4": "Sample/mustard_powdery/DSC_0068.JPG",
        "Sample 5": "Sample/mustard_powdery/DSC_0187.JPG",
    }

    sample_selection = st.selectbox("Choose a sample:", list(sample_images.keys()))
    selected_sample = sample_images.get(sample_selection)

    if selected_sample:
        image = Image.open(selected_sample).convert("RGB")
        display_image_fixed_size(image, width=800, height=300)
        st.markdown("---")

        if st.button("🔍 Predict Diagnosis (Sample Image)", use_container_width=True):
            predict_and_display(image)
