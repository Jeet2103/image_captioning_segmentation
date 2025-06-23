# app.py

import streamlit as st
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os
from captioning.generate_caption import generate_caption
from segmentation.segment import segment_image
from app.utils import display_masks

# Suppress PyTorch debug messages
os.environ['PYTORCH_JIT_LOG_LEVEL'] = 'NONE'

# === STREAMLIT CONFIGURATION ===
st.set_page_config(
    page_title="🧠 Image Captioning & Segmentation",
    layout="wide",
    page_icon="🧠"
)

# === HEADER ===
st.markdown(
    """
    <h1 style='text-align: center; color: #4F8BF9;'>🧠 Integrated Image Captioning & Segmentation</h1>
    <p style='text-align: center; font-size: 18px;'>Upload an image to automatically generate a caption and perform both instance and semantic segmentation.</p>
    <hr>
    """,
    unsafe_allow_html=True
)

# === IMAGE UPLOAD SECTION ===
st.sidebar.header("📤 Upload Your Image")
uploaded_file = st.sidebar.file_uploader(
    label="Select an image file",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    # Load and preprocess image
    image = Image.open(uploaded_file).convert("RGB")
    resized_image = image.resize((384, 384))  # Uniform size for display

    # === CAPTION GENERATION ===
    with st.spinner("🔄 Generating caption..."):
        caption = generate_caption(image)

    st.markdown("### 📝 Generated Caption")
    st.success(caption)

    # === SEGMENTATION PROCESSING ===
    with st.spinner("🔍 Running instance and semantic segmentation..."):
        resized_img, masks, boxes, labels, semantic_mask = segment_image(image)

    # Prepare semantic segmentation visualization
    semantic_fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(np.array(resized_img.resize((384, 384))))
    ax.imshow(semantic_mask, alpha=0.6, cmap="jet")
    ax.axis("off")
    plt.tight_layout()

    # === THREE COLUMN DISPLAY ===
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("#### 📷 Original Image")
        st.image(resized_image, use_column_width=True)

    with col2:
        st.markdown("#### 📌 Instance Segmentation")
        display_masks(resized_img.resize((384, 384)), masks, boxes, labels)

    with col3:
        st.markdown("#### 🌈 Semantic Segmentation")
        st.pyplot(semantic_fig)

    # Completion message
    st.success("✅ All processing completed successfully!")

else:
    # Default instruction
    st.info("👈 Please upload an image from the sidebar to get started.")
