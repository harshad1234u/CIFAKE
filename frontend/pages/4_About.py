import streamlit as st
import os

st.set_page_config(page_title="About | CIFAKE", page_icon="ℹ️", layout="wide")

# Load CSS
css_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'assets', 'style.css')
if os.path.exists(css_path):
    with open(css_path) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

st.title("ℹ️ About the Project")

st.markdown("""
### Problem Statement
With the rapid advancement of generative AI, distinguishing between authentic photographs and AI-generated synthetic images has become increasingly difficult. This project aims to address this challenge by providing a robust, highly accurate image classification system that not only detects fake images but also explains its reasoning.

### Dataset Information
The **CIFAKE** dataset contains 120,000 images, balanced equally between REAL and FAKE classes.
* **Train Set:** 100,000 images (50,000 Real / 50,000 Fake)
* **Test Set:** 20,000 images (10,000 Real / 10,000 Fake)

### Model Architecture
The underlying model uses **MobileNetV2** as the base architecture via Transfer Learning.
MobileNetV2 is highly efficient and designed for performance on mobile and resource-constrained environments, making inference lightning fast without sacrificing accuracy.

### Methodology
1. **Transfer Learning:** The pre-trained MobileNetV2 architecture was frozen and used as a feature extractor.
2. **Fine-tuning:** Custom dense layers were added, and the top layers of the base model were fine-tuned to adapt specifically to the CIFAKE dataset.
3. **Explainability:** Grad-CAM (Gradient-weighted Class Activation Mapping) is integrated into the inference pipeline to provide visual explanations of the model's spatial focus during prediction.

### Technology Stack
* **Frontend:** Streamlit
* **Deep Learning Framework:** TensorFlow / Keras
* **Image Processing:** OpenCV, Pillow, NumPy
* **Explainable AI:** Grad-CAM algorithm implementation

### Future Scope
* Integrating ensemble models for increased accuracy.
* Adding support for video frame analysis.
* Deploying as a REST API for mobile app integration.
""")
