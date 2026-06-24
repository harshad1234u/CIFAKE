import streamlit as st
import os

st.set_page_config(page_title="Analytics | CIFAKE", page_icon="📊", layout="wide")

# Load CSS
css_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'assets', 'style.css')
if os.path.exists(css_path):
    with open(css_path) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

st.title("📊 Model Analytics")
st.markdown("Review the performance metrics of the CIFAKE classification model.")

st.markdown("### Performance Metrics")
col1, col2, col3 = st.columns(3)

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
* **Total Training Images:** 100,000 (50,000 REAL, 50,000 FAKE)
* **Total Testing Images:** 20,000 (10,000 REAL, 10,000 FAKE)
* **Classes:** REAL, AI GENERATED
""")

# Note: The instructions explicitly state "Do NOT fabricate training graphs. Use only real metrics."
# Therefore, we will only display the text metrics here, as we do not have real history logs.
st.info("Note: Detailed training history graphs are not available for this session. The final validation metrics accurately represent the model's performance on the 20k test set.")
