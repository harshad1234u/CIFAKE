import streamlit as st
import os
import sys
import datetime
from PIL import Image

# Add project root to path to import model.predict
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.append(project_root)

import importlib
import model.predict
importlib.reload(model.predict)
from model.predict import predict_image

st.set_page_config(page_title="Detection | CIFAKE", page_icon="🔍", layout="wide")

# Load CSS
css_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'assets', 'style.css')
if os.path.exists(css_path):
    with open(css_path) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# Initialize Session State for history
if 'history' not in st.session_state:
    st.session_state.history = []

st.title("🔍 Detection Engine")
st.markdown("Upload an image to determine if it is **REAL** or **AI GENERATED**.")

# Model Selection Sidebar
model_option = st.sidebar.selectbox(
    "Select Detection Model",
    ["CIFAKE (MobileNetV2)", "StyleGAN (EfficientNetV2)"],
    help="CIFAKE (MobileNetV2): Best for general objects & synthetic artwork.\nStyleGAN (EfficientNetV2): Best for photorealistic faces."
)

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png", "webp"])

if uploaded_file is not None:
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Uploaded Image")
        try:
            image = Image.open(uploaded_file)
            st.image(image, use_column_width=True)
        except Exception as e:
            st.error("Unsupported file type or corrupted image.")
            st.stop()
            
    with col2:
        st.markdown("### Results")
        predict_btn = st.button("Run Prediction 🚀", use_container_width=True)
        
        if predict_btn:
            with st.spinner("Analyzing image..."):
                try:
                    result = predict_image(image, model_name=model_option)
                    
                    label = result['label']
                    confidence = result['confidence'] * 100
                    probability = result['probability']
                    
                    st.markdown('<div class="custom-card">', unsafe_allow_html=True)
                    if label == "REAL":
                        st.markdown(f'<div class="result-real">Prediction: {label}</div>', unsafe_allow_html=True)
                    else:
                        st.markdown(f'<div class="result-fake">Prediction: {label}</div>', unsafe_allow_html=True)
                    
                    st.markdown(f"**Confidence:** {confidence:.2f}%")
                    st.markdown(f"**Raw Probability (REAL):** {probability:.4f}")
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    if confidence < 60:
                        st.warning("Prediction confidence is low. The model is uncertain.")
                        
                    # Save to session history
                    st.session_state.history.append({
                        'filename': uploaded_file.name,
                        'label': label,
                        'confidence': f"{confidence:.2f}%",
                        'time': datetime.datetime.now().strftime("%H:%M:%S")
                    })
                    
                except FileNotFoundError:
                    st.error("Model unavailable. Please ensure the model file is in the correct directory.")
                except Exception as e:
                    st.error(f"Prediction failed: {str(e)}")

# Display History
if st.session_state.history:
    st.markdown("---")
    st.markdown("### 🕒 Recent Predictions")
    history_md = "| Time | Filename | Prediction | Confidence |\n|---|---|---|---|\n"
    for item in reversed(st.session_state.history[-5:]): # Show last 5
        history_md += f"| {item['time']} | {item['filename']} | **{item['label']}** | {item['confidence']} |\n"
    st.markdown(history_md)
