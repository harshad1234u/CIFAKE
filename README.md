# CIFAKE & Deepfake Image Classification: Explainable Identification of AI-Generated Images

![AI Detection](https://img.shields.io/badge/AI-Image%20Detection-blue)
![CIFAKE Accuracy](https://img.shields.io/badge/CIFAKE%20Accuracy-93.32%25-brightgreen)
![StyleGAN Accuracy](https://img.shields.io/badge/StyleGAN%20Accuracy-98.18%25-darkgreen)
![Framework](https://img.shields.io/badge/TensorFlow-2.11%2B-orange)
![Frontend](https://img.shields.io/badge/Streamlit-1.20%2B-red)

## Overview
This application is a comprehensive, end-to-end AI platform designed to detect synthetic media. As generative models become increasingly sophisticated, verifying the authenticity of digital images is critical. This project addresses this challenge by integrating two specialized deep learning models into an explainable, professional web application with built-in Explainable AI (XAI) capabilities.

## Dual-Model Architecture

The application dynamically supports two state-of-the-art models tailored for different detection domains:

### 1. CIFAKE Detector (MobileNetV2)
* **Best For:** General objects, animals, vehicles, and standard synthetic artwork.
* **Architecture:** MobileNetV2 utilizing transfer learning.
* **Accuracy:** **93.32% validation accuracy** on the CIFAKE dataset.
* **Dataset:** 120,000 images (100,000 training, 20,000 testing).

### 2. StyleGAN Deepfake Face Detector (EfficientNetV2B0)
* **Best For:** Photorealistic human faces and deepfake portraits.
* **Architecture:** EfficientNetV2B0 with a custom dense classification head.
* **Accuracy:** **98.18% validation accuracy** on NVIDIA's StyleGAN dataset.
* **Dataset:** 140,000 real and fake faces (70,000 real from FFHQ, 70,000 synthetic from StyleGAN/StyleGAN2).

---

## Explainable AI (Grad-CAM)
Both models are integrated with an **Explainable AI (XAI)** module using Gradient-weighted Class Activation Mapping (Grad-CAM). Instead of functioning as a "black box," the application overlays a jet-colormap heatmap onto the image, visually highlighting the exact spatial regions, textures, and artifacts (e.g. background warping, edge irregularities) that influenced the model's decision.

---

## Installation & Setup

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/harshad1234u/CIFAKE.git
   cd CIFAKE
   ```

2. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download Model Files:**
   Ensure both model files are placed in the appropriate directory:
   * `best_model.keras` or `model/best_model.keras` (CIFAKE model)
   * `stylegan_detector_best.keras` or `model/stylegan_detector_best.keras` (StyleGAN model)

4. **Run the Application:**
   ```bash
   streamlit run frontend/app.py
   ```

---

## Model Training
* To train the general object detector, refer to `train.py`.
* To train the face deepfake detector using GPU acceleration (e.g., in Google Colab), refer to `train_stylegan.py`. Note that native Windows GPU acceleration requires Python 3.10 and TensorFlow <= 2.10, so running the training pipeline via Google Colab is highly recommended.

