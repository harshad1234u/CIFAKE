import streamlit as st
import os

st.set_page_config(page_title="Analytics | CIFAKE", page_icon="📊", layout="wide")

# Load CSS
css_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'assets', 'style.css')
if os.path.exists(css_path):
    with open(css_path) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

st.title("📊 Model Analytics")
st.markdown("Review the performance metrics of the classification models.")

# Model selection for analytics
model_option = st.sidebar.selectbox(
    "Select Model to View Analytics",
    ["CIFAKE (MobileNetV2)", "StyleGAN (EfficientNetV2)"]
)

st.markdown(f"### Performance Metrics for {model_option}")
col1, col2, col3 = st.columns(3)

if "CIFAKE" in model_option:
    with col1:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">Validation Accuracy</div>
            <div class="metric-value">93.32%</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">Validation Loss</div>
            <div class="metric-value">0.1690</div>
        </div>
        """, unsafe_allow_html=True)
        
    with col3:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">Model Architecture</div>
            <div class="metric-value">MobileNetV2</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### Dataset Overview")
    st.markdown("""
    * **Dataset Source:** CIFAKE (CIFAR-10 size standard objects)
    * **Total Training Images:** 100,000 (50,000 REAL, 50,000 FAKE)
    * **Total Testing Images:** 20,000 (10,000 REAL, 10,000 FAKE)
    * **Resolution:** 32x32 pixels (upscaled to 224x224 for MobileNetV2)
    * **Classes:** REAL, AI GENERATED (Stable Diffusion v1.4)
    """)
else:
    with col1:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">Validation Accuracy</div>
            <div class="metric-value">~98.5%</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">Validation Loss</div>
            <div class="metric-value">~0.0450</div>
        </div>
        """, unsafe_allow_html=True)
        
    with col3:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">Model Architecture</div>
            <div class="metric-value">EfficientNetV2B0</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### Dataset Overview")
    st.markdown("""
    * **Dataset Source:** StyleGAN-StyleGAN2 Deepfake Face Images
    * **Total Training/Validation/Testing Images:** 140,000
    * **Resolution:** 256x256 pixels (downscaled to 224x224 for EfficientNetV2)
    * **Classes:** REAL, AI GENERATED (NVIDIA StyleGAN/StyleGAN2 photorealistic faces)
    """)

st.info("Note: Detailed training history graphs are not available for this session. The final validation metrics accurately represent the model's performance on the validation/test sets.")

