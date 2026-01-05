# app.py
import streamlit as st

st.set_page_config(
    page_title="Unified XAI Interface",
    layout="wide"
)

st.title("Unified Explainable AI Interface")
st.write(
    """
    This application provides a unified interface for **image and audio classification**
    with **explainability methods** (Grad-CAM, LIME, SHAP).

    👉 Use the **sidebar** to navigate between:
    - Inference
    - XAI Comparison
    """
)

st.info("Select a page from the left sidebar to start.")
