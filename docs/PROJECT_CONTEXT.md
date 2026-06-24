# CIFAKE: Project Context

## Current State
* The model training phase is complete.
* Current model: MobileNetV2 Transfer Learning
* Validation Accuracy: 93.32%
* Validation Loss: 0.1690
* Model File: `cifake_model.keras`
* Retraining the model is explicitly forbidden.

## Project Goal
To transform the trained model into a complete, professional AI project suitable for demonstration, portfolio, and resume showcasing. The application will accept image uploads, detect REAL vs AI-GENERATED images, display prediction confidence, and explain model decisions using Grad-CAM.

## Key Constraints
* **Do Not Build:** User authentication (login, signup), databases (MySQL, MongoDB, Firebase), cloud deployment configurations (Cloud Run, AWS), or additional AI features (Chatbots, RAG, LLMs).
* **Must Have:** Graceful failure on edge cases (empty uploads, invalid files, missing model).

## Development Phases
1. **Phase 1:** Prediction Engine (`predict.py`)
2. **Phase 2:** Explainable AI (`gradcam.py`)
3. **Phase 3:** Professional Frontend (Streamlit)
4. **Phase 4:** Visual Assets
5. **Phase 5:** Documentation (`README.md`)
6. **Phase 6:** Testing
7. **Phase 7:** Project Polish
