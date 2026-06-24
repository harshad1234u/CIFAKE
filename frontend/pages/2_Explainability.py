import streamlit as st
import os
import sys
from PIL import Image

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.append(project_root)

import importlib
import model.predict
import model.gradcam
importlib.reload(model.predict)
importlib.reload(model.gradcam)
from model.gradcam import generate_gradcam
from model.predict import get_model

st.set_page_config(page_title="Explainability | CIFAKE", page_icon="🧠", layout="wide")

# Load CSS
css_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'assets', 'style.css')
if os.path.exists(css_path):
    with open(css_path) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

st.title("🧠 Explainable AI (Grad-CAM)")
st.markdown("Understand *why* the model made its prediction by visualizing the regions it focused on.")

# Model Selection Sidebar
model_option = st.sidebar.selectbox(
    "Select Model for Explanation",
    ["CIFAKE (MobileNetV2)", "StyleGAN (EfficientNetV2)"],
    help="CIFAKE (MobileNetV2): Best for general objects & synthetic artwork.\nStyleGAN (EfficientNetV2): Best for photorealistic faces."
)

uploaded_file = st.file_uploader("Upload an image for explanation...", type=["jpg", "jpeg", "png", "webp"])

if uploaded_file is not None:
    try:
        # We need a temporary path for cv2 and load_img inside gradcam
        # We'll save it to a temp file
        import tempfile
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tf_file:
            tf_file.write(uploaded_file.getbuffer())
            temp_path = tf_file.name
            
        st.markdown("### Visualization")
        
        with st.spinner("Generating Grad-CAM heatmap..."):
            try:
                model, target_size = get_model(model_option)
                original, heatmap, overlay = generate_gradcam(temp_path, model, target_size)
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.image(original, caption="Original Image", use_column_width=True)
                with col2:
                    st.image(heatmap, caption="Grad-CAM Heatmap", use_column_width=True)
                with col3:
                    st.image(overlay, caption="Overlay", use_column_width=True)
                    
                st.markdown("### Analysis")
                st.info("The red regions in the heatmap indicate the specific patterns, textures, or features that the model found most influential in making its prediction. For AI-generated images, it often focuses on subtle artifacts or inconsistencies in the background and edges.")
                
            except Exception as e:
                st.error("Explanation unavailable. The model could not generate a heatmap for this image.")
                st.error(str(e))
            finally:
                # Cleanup
                if os.path.exists(temp_path):
                    os.remove(temp_path)
    except Exception as e:
        st.error("Error processing the uploaded file.")
