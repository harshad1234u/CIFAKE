import streamlit as st
import os

st.set_page_config(
    page_title="CIFAKE | AI Image Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load CSS
def load_css():
    css_path = os.path.join(os.path.dirname(__file__), 'assets', 'style.css')
    if os.path.exists(css_path):
        with open(css_path) as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

load_css()

# --- Home Page Content ---
st.title("🛡️ CIFAKE")
st.subheader("Image Classification and Explainable Identification of AI-Generated Synthetic Images")

st.markdown("---")

col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("""
    ### Welcome to CIFAKE
    This application utilizes Deep Learning and Transfer Learning (MobileNetV2 & EfficientNetV2) to accurately distinguish between real photographs and AI-generated synthetic images.
    
    In an era where AI-generated content is becoming increasingly indistinguishable from reality, **CIFAKE** provides a reliable, explainable tool for verification.
    
    #### Key Features:
    * **High Accuracy:** Validated at 93.32% accuracy on the CIFAKE dataset, and highly accurate on StyleGAN generated Deepfakes.
    * **Dual Detection Engines:** Choose between the standard CIFAKE (MobileNetV2) model or the advanced StyleGAN (EfficientNetV2) model.
    * **Explainable AI (XAI):** Uses Grad-CAM to visualize exactly *why* the model made its decision.
    * **Fast Inference:** Optimized for rapid image processing.
    """)
    
with col2:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-label">Validation Accuracy</div>
        <div class="metric-value">93.32%+</div>
    </div>
    
    <div class="metric-card">
        <div class="metric-label">Model Architectures</div>
        <div class="metric-value">MobileNetV2 & EfficientNetV2</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")
st.info("👈 Please select a page from the sidebar to begin.")
