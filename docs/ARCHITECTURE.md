# CIFAKE: System Architecture

## Overview
CIFAKE is a Streamlit-based web application designed to detect AI-generated synthetic images using a pre-trained MobileNetV2 Deep Learning model.

## System Components

### 1. Frontend (Streamlit)
- **`app.py`**: The entry point. Manages global configurations and custom CSS for a professional look.
- **Pages**:
  - `1_Detection.py`: Handles image uploads, displays previews, and triggers the prediction pipeline. Maintains a session history of recent predictions.
  - `2_Explainability.py`: Displays Grad-CAM heatmaps and overlays for the uploaded images.
  - `3_Analytics.py`: Shows model performance metrics and plots.
  - `4_About.py`: Contains information regarding the project's background and architecture.

### 2. Backend / Model Inference
- **`model/predict.py`**: Encapsulates the model loading (singleton pattern/caching) and image preprocessing logic. Returns class label, confidence score, and probabilities.
- **`model/gradcam.py`**: Implements the Gradient-weighted Class Activation Mapping (Grad-CAM) algorithm. Extracts the last convolutional layer's gradients to compute class activation heatmaps.

### 3. Data Flow
1. User uploads an image via the Streamlit frontend.
2. The image is passed to `predict.py` for preprocessing and inference.
3. The prediction results (Label, Confidence) are returned to the frontend and stored in Streamlit Session State.
4. Concurrently or on-demand, the image is passed to `gradcam.py` to generate an explainability heatmap.
5. The frontend displays the original image alongside the heatmap and the overlay, along with an explanation text.

## Constraints
- The system must fail gracefully when encountering empty uploads, invalid file types, or corrupted images.
- All states are managed in-memory per session using Streamlit Session State; no external databases are utilized.
