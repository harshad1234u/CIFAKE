# CIFAKE: Image Classification and Explainable Identification of AI-Generated Synthetic Images

![CIFAKE Banner](https://img.shields.io/badge/AI-Image%20Detection-blue)
![Accuracy](https://img.shields.io/badge/Accuracy-93.32%25-brightgreen)
![Framework](https://img.shields.io/badge/TensorFlow-2.11%2B-orange)
![Frontend](https://img.shields.io/badge/Streamlit-1.20%2B-red)

## Overview
CIFAKE is a complete end-to-end AI application built to detect AI-generated synthetic images. With the rise of advanced generative models, distinguishing reality from AI fiction is increasingly challenging. This project tackles the issue by utilizing a highly optimized Deep Learning model (MobileNetV2) integrated into a modern, professional web interface with built-in Explainable AI (XAI) capabilities.

## Problem Statement
Provide a reliable, explainable tool to verify the authenticity of digital images by distinguishing between real photographs and AI-generated content.

## Dataset
* **Total Images:** 120,000
* **Training Set:** 100,000 images
* **Testing Set:** 20,000 images
* **Classes:** REAL, FAKE (Balanced 50/50)

## Model Architecture
The core model leverages **MobileNetV2** via Transfer Learning. This architecture was chosen for its excellent balance of high accuracy and fast inference speed, making it suitable for real-time web applications.

## Explainable AI (Grad-CAM)
This project doesn't just provide a prediction; it explains *why*. By integrating Gradient-weighted Class Activation Mapping (Grad-CAM), the application generates a heatmap highlighting the exact regions and textures the model focused on when making its decision. 

## Results
* **Validation Accuracy:** 93.32%
* **Validation Loss:** 0.1690

## Installation Guide
1. Clone this repository.
2. Ensure you have Python 3.9+ installed.
3. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Run the Streamlit application:
   ```bash
   streamlit run frontend/app.py
   ```

## Future Scope
* Support for analyzing specific AI artifacts (e.g., GAN fingerprints).
* Expanding the dataset to include newer diffusion model outputs.
* Mobile application integration via a RESTful API backend.
