



<div align="center">
  
![image](https://github.com/user-attachments/assets/5f6b0fdf-a89a-46b1-8fdc-768d0c46825c)

**Advanced Medical Image Analysis Platform powered by Deep Learning**

[![Live Demo](https://img.shields.io/badge/Live%20Demo-Visit%20Now-blue?style=for-the-badge)](https://meddetectai-health-hub.lovable.app)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://tensorflow.org)
[![Status](https://img.shields.io/badge/Status-Active-green.svg)]()

</div>



## 🚀 Overview

MedDetectAI is a cutting-edge medical image analysis platform that leverages state-of-the-art deep learning models to assist in the detection and classification of critical medical conditions. Our platform combines advanced computer vision techniques with medical expertise to provide accurate, fast, and reliable diagnostic assistance.

### 🎯 Key Features

- **🧠 Brain Tumor MRI Detection** - Advanced neural network analysis of brain MRI scans
- **🫘 Kidney Disease Detection** - Comprehensive kidney imaging analysis
- **⚡ Real-time Processing** - Fast and efficient image analysis
- **📊 Confidence Scoring** - Detailed probability distributions for each diagnosis
- **🔬 Research-Grade Accuracy** - Models trained on extensive medical datasets

---

## 🏥 Medical Modules

### 1. Brain Tumor MRI Detection

Our brain tumor detection module uses a fine-tuned **Xception architecture** to analyze MRI scans and identify tumor types with high precision.

#### 🎯 Detection Categories
- **Glioma Tumors** - Malignant tumors originating in glial cells
- **Meningioma Tumors** - Typically benign tumors from the meninges
- **Pituitary Tumors** - Hormone-affecting tumors in the pituitary gland
- **No Tumor** - Normal brain tissue classification

#### 🔧 Technical Specifications
- **Base Architecture:** Xception (ImageNet pretrained)
- **Input Resolution:** 299×299 pixels
- **Training Dataset:** Thousands of annotated MRI images
- **Classification Output:** 4 categories with confidence scores

#### 📋 Clinical Information
- **Glioma:** Represents 33% of all brain tumors, 80% of malignant cases
- **Meningioma:** Accounts for 37% of primary brain tumors
- **Pituitary:** Comprises 10-15% of all brain tumors

### 2. Kidney Disease Detection

Our kidney disease detection system utilizes a fine-tuned **InceptionResNetV2 model** for comprehensive renal imaging analysis.

#### 🎯 Detection Categories
- **Kidney Cysts** - Fluid-filled sacs detection
- **Kidney Stones** - Mineral deposit identification
- **Kidney Tumors** - Benign and malignant growth detection
- **Normal Kidney** - Healthy tissue classification

#### 🔧 Technical Specifications
- **Base Architecture:** InceptionResNetV2
- **Input Resolution:** 244×244 pixels
- **Training Dataset:** Comprehensive kidney imaging dataset
- **Classification Output:** 4 categories with confidence scores

---
## 📈 Model Performance

| Model | Accuracy | 
|-------|----------|
| Brain Tumor MRI Detection | 99.54% | 
| Kidney Disease Detection | 99.52% | 

---

## 🔬 Technical Architecture

```mermaid
graph TD
    A[Medical Image Input] --> B[Image Preprocessing]
    B --> B1[Normalization & Resizing]
    B --> B2[Noise Reduction]
    B --> B3[Contrast Enhancement]
    
    B1 --> C[Model Selection]
    B2 --> C
    B3 --> C
    
    C --> D[Brain Tumor Detection<br/>Xception Model]
    C --> E[Kidney Disease Detection<br/>InceptionResNetV2 Model]
    
    D --> F[Feature Extraction]
    E --> F
    
    F --> F1[Convolutional Layers]
    F --> F2[Transfer Learning<br/>ImageNet Pretrained]
    F --> F3[Fine-tuning<br/>Medical Datasets]
    
    F1 --> G[Classification Layer]
    F2 --> G
    F3 --> G
    
    G --> H[Multi-class Prediction]
    H --> I[Confidence Scoring]
    I --> J[Result Interpretation]
    
    J --> K[Brain Tumor Results]
    J --> L[Kidney Disease Results]
    
    K --> K1[Glioma]
    K --> K2[Meningioma]
    K --> K3[Pituitary]
    K --> K4[No Tumor]
    
    L --> L1[Kidney Cysts]
    L --> L2[Kidney Stones]
    L --> L3[Kidney Tumors]
    L --> L4[Normal Kidney]
    
    style A fill:#0366d6,stroke:#0366d6,stroke-width:2px,color:#fff
    style D fill:#28a745,stroke:#28a745,stroke-width:2px,color:#fff
    style E fill:#fd7e14,stroke:#fd7e14,stroke-width:2px,color:#fff
    style J fill:#6f42c1,stroke:#6f42c1,stroke-width:2px,color:#fff
    style K fill:#dc3545,stroke:#dc3545,stroke-width:2px,color:#fff
    style L fill:#20c997,stroke:#20c997,stroke-width:2px,color:#fff
```

### Model Training Pipeline

```mermaid
graph LR
    A[Raw Medical Images] --> B[Data Augmentation]
    B --> B1[Rotation]
    B --> B2[Scaling]
    B --> B3[Flipping]
    B --> B4[Brightness/Contrast]
    
    B1 --> C[Training Split]
    B2 --> C
    B3 --> C
    B4 --> C
    
    C --> D[80% Training]
    C --> E[10% Validation]
    C --> F[10% Testing]
    
    D --> G[Model Training]
    E --> G
    
    G --> H[Adam Optimizer]
    G --> I[Learning Rate Scheduling]
    G --> J[Regularization]
    
    J --> J1[Dropout]
    J --> J2[Batch Normalization]
    J --> J3[Early Stopping]
    
    H --> K[Model Evaluation]
    I --> K
    J1 --> K
    J2 --> K
    J3 --> K
    
    F --> K
    K --> L[Performance Metrics]
    L --> M[K-fold Cross Validation]
    M --> N[Final Model]
    
    style A fill:#0366d6,stroke:#0366d6,stroke-width:2px,color:#fff
    style G fill:#28a745,stroke:#28a745,stroke-width:2px,color:#fff
    style K fill:#fd7e14,stroke:#fd7e14,stroke-width:2px,color:#fff
    style N fill:#6f42c1,stroke:#6f42c1,stroke-width:2px,color:#fff
```

---

## 🛠️ Installation

### Prerequisites
```bash
Python 3.8+
TensorFlow 2.x
NumPy
Pillow
Streamlit (for web interface)
```

### Quick Setup
```bash
# Clone the repository
git clone https://github.com/yourusername/MedDetectAI.git
cd MedDetectAI

# Install dependencies
pip install -r requirements.txt

# Run the application
streamlit run app.py
```

---

## 🚀 Usage

### Web Interface
1. Launch the application using `streamlit run app.py`
2. Select the desired medical module (Brain Tumor or Kidney Disease)
3. Upload your medical image (MRI scan or kidney imaging)
4. View real-time analysis results with confidence scores
5. Download detailed reports for medical review


---

## ⚖️ Medical Disclaimer

> [!CAUTION]
> **🚨 CRITICAL MEDICAL DISCLAIMER**
> 
> This application is designed for **educational and research purposes only**. It is not intended to replace professional medical diagnosis, advice, or treatment.

> [!WARNING]
> **⚠️ IMPORTANT SAFETY NOTICE**
> 
> **Key Points:**
> - Always consult qualified healthcare providers for medical decisions
> - Results may contain false positives and false negatives
> - Not FDA approved for clinical diagnostic use
> - Requires validation by medical professionals
> - Should not be used as the sole basis for medical decisions

---
