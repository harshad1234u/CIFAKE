# CHAPTER 1: INTRODUCTION

**Project Background**
With the rapid proliferation of highly capable generative artificial intelligence models, the digital ecosystem is experiencing an influx of hyper-realistic synthetic media. Distinguishing between authentic, camera-captured photographs and synthetically generated images has become increasingly difficult for human observers. This phenomenon raises critical concerns regarding digital forgery, misinformation, and intellectual property infringement.

**Problem Statement**
There is an urgent requirement for reliable, automated computational tools to verify the authenticity of digital images. Current manual verification methods are subjective and inefficient. Furthermore, many existing deep learning solutions operate as opaque "black boxes," providing predictions without transparent contextual evidence, which diminishes user trust in automated systems.

**Objectives**
1. To develop a robust deep learning classifier capable of distinguishing between real and artificial intelligence-generated images with high accuracy.
2. To implement an Explainable Artificial Intelligence (XAI) module that visually elucidates the neural network's decision-making process.
3. To design and deploy a professional, user-friendly web application for real-time image analysis.

**Scope of the Project**
The project encompasses the training, evaluation, and deployment of a MobileNetV2-based convolutional neural network on the CIFAKE dataset. The scope includes the development of a Streamlit frontend application and the integration of Gradient-weighted Class Activation Mapping (Grad-CAM) for explainability. The application is limited to binary classification (Real vs. AI Generated) on standard two-dimensional image formats.

**Expected Outcomes**
The expected outcome is a fully functional web-based application that accepts image uploads, accurately predicts authenticity, displays statistical confidence metrics, and provides a Grad-CAM heatmap visualization explaining the localization of discriminative features.

# CHAPTER 2: SYSTEM OVERVIEW

**Existing System**
Existing systems for synthetic image detection frequently rely on metadata analysis, which is highly susceptible to manipulation. Alternative commercial deep learning models output simple probability scores without explicit reasoning, preventing users from understanding the specific features that triggered a positive or negative detection.

**Proposed System**
The proposed system bridges this gap by combining an optimized transfer learning model (MobileNetV2) with a Grad-CAM module. This architecture ensures that every prediction is accompanied by a visual heatmap, explicitly highlighting the specific regions, textures, or artifacts that influenced the model's decision.

**System Modules**
1. **Inference Engine Module:** Manages input processing, image normalization, and execution of the TensorFlow model.
2. **Explainability (XAI) Module:** Computes spatial gradients from the final convolutional layer to generate transparent heatmaps.
3. **Frontend Application Module:** A Streamlit-based graphical interface for user interaction and visualization.
4. **Analytics Module:** Computes and displays static performance metrics and dataset distributions.

**Overall Workflow**
The user uploads an image via the web interface. The image is processed by the Inference Engine, normalized, and fed through the MobileNetV2 architecture. Concurrently, the Explainability Module extracts feature maps from the model to generate a Grad-CAM overlay. The final composite results are presented to the user.

# CHAPTER 3: LITERATURE SURVEY

**Research Papers Reviewed**
1. *Deep Residual Learning for Image Recognition* (He et al.) – Provided the foundation for residual connections in deep convolutional networks.
2. *MobileNetV2: Inverted Residuals and Linear Bottlenecks* (Sandler et al.) – Explored for its parameter efficiency and low latency in computer vision tasks.
3. *Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization* (Selvaraju et al.) – Formed the theoretical basis for the project's explainability module.

**Existing Solutions**
Current solutions often utilize massive Vision Transformers or frequency-domain analyzers. While highly accurate, they remain computationally expensive and lack granular, user-facing visual explainability.

**Comparative Analysis**
Compared to heavy Vision Transformers, the MobileNetV2 architecture requires significantly fewer parameters (approximately 3.4 million), making inference rapid on standard hardware. The explicit integration of Grad-CAM provides a layer of transparency frequently absent in commercial APIs.

**Research Gap**
A noticeable gap exists regarding lightweight, web-deployable synthetic image detectors that prioritize both inference speed and explicit visual explainability. The proposed system directly addresses this gap.

# CHAPTER 4: TECHNOLOGIES USED

**Programming Languages**
* Python 3.9+ 
* CSS3

**Frameworks and Libraries**
* **TensorFlow & Keras:** Utilized for model architecture construction, training, inference, and gradient calculations.
* **Streamlit:** Employed for the rapid development of the web frontend.
* **OpenCV:** Used for colormap application and spatial image manipulations in the Grad-CAM module.
* **NumPy:** Used for high-performance numerical array operations.
* **Pillow (PIL):** Utilized for input output operations, image resizing, and format conversion.

**Development Tools**
* Git & GitHub
* Visual Studio Code

# CHAPTER 5: ARCHITECTURE AND DESIGN

**System Architecture Diagram**
The system architecture follows a modular pipeline. Inputs are routed from the frontend to the backend processing units, diverging into raw prediction computation and gradient mapping, before converging back at the display layer.

**Workflow Explanation**
The frontend is decoupled from the model inference logic. The main application file manages routing and session state management. The backend encapsulates prediction modules utilizing a singleton pattern to maintain the model in memory. The Grad-CAM module hooks into the global average pooling layer's inputs to extract gradients without violating nested functional graph constraints.

# CHAPTER 6: IMPLEMENTATION

**Key Features Implemented**
1. **Dynamic Model Loading:** A singleton pattern ensures the model is loaded sequentially only once, reducing memory overhead and prediction latency.
2. **Grad-CAM Integration:** Feature maps are extracted from the nested Keras model to overlay jet-colormapped heatmaps on the original spatial inputs.
3. **Session History:** State management is implemented to persist user predictions across the application session lifecycle.
4. **Robust Error Handling:** The system safely intercepts corrupted files and anomalous resolutions.

# CHAPTER 7: RESULTS AND DISCUSSION

**Output Analysis**
The model effectively computes binary classification indicating authenticity, accompanied by a confidence percentage derived from the final sigmoid activation function. 

**Performance Evaluation**
The model was evaluated on a 20,000-image test set partitioned from the CIFAKE dataset. The evaluation yielded the following metrics:
* **Validation Accuracy:** 93.32%
* **Validation Loss:** 0.1690

Table 7.1: Validation Performance Metrics (Accuracy: 93.32%, Loss: 0.1690)

**Testing Results**
The application successfully processed standard inputs. During out-of-distribution testing, it was observed that inputs significantly diverging from the CIFAR-10 training distribution (e.g., text or complex indoor scenes) yielded deterministic but occasionally incorrect classifications.

# CHAPTER 8: CHALLENGES AND LEARNING OUTCOMES

**Challenges Faced**
1. **Nested Model Gradients:** The strict graph partitioning in Keras 3 complicated the extraction of the nested base model outputs, initially resulting in disconnected graph errors during Grad-CAM computation.
2. **Out-of-Distribution Data:** The model exhibited degraded confidence when classifying objects absent from the core CIFAKE dataset.

**Solutions Implemented**
1. **Architectural Hooking:** The Grad-CAM logic was refactored to interface with the input tensor of the subsequent layer (global average pooling), effectively extracting the required feature map while maintaining computational graph integrity.
2. **Operational Guidelines:** Explicit documentation was added to the application to set analytical expectations regarding the model's operational domain.

**Technical Skills Learned**
The project facilitated the acquisition of skills in advanced TensorFlow graph manipulation, GradientTape operations, Streamlit state management, and optimized inference pipeline design.

# CHAPTER 9: FUTURE ENHANCEMENTS

**Proposed Improvements**
* **Ensemble Modeling:** Integrating the current architecture with an EfficientNet variant is proposed to improve generalization on out-of-distribution media.
* **Dataset Expansion:** Retraining the model on a broader dataset encompassing human faces and complex landscapes would likely reduce false positive rates on everyday photography.

**Additional Features**
* **Batch Processing:** Implementing parallel processing to allow users to verify bulk image datasets simultaneously.

**Scalability Opportunities**
The core inference logic can be abstracted into a REST API endpoint, permitting the model to serve mobile applications at scale via cloud containerization.

# CHAPTER 10: CONCLUSION

**Summary of Work Completed**
The project successfully transitioned a trained MobileNetV2 architecture into a production-ready web application. It features a robust prediction pipeline, an integrated Explainable Artificial Intelligence module using Grad-CAM, and a formal user interface.

**Achievements**
The system maintained a strict 93.32% validation accuracy on the CIFAKE dataset while successfully mitigating complex nested-model gradient extraction errors. It delivers a visually striking and interactive application.

**Project Impact**
The developed application provides an educational and effective tool for identifying synthetic media. By prioritizing explainability, the project demystifies opaque artificial intelligence decisions and fosters critical analytical evaluation of digital authenticity.

**Final Remark**
As generative artificial intelligence capabilities advance, verification tools serve as a necessary mechanism in preserving digital authenticity. The architectural foundations established in this project are highly adaptable for future iterations against emerging synthetic media threats.
