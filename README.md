# 🛍️ E-Commerce Product Attribute Prediction from Images
### Meesho Data Challenge 2024 — Team Neural Ninjas
[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.4.1-EE4C2C.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

---

## 📘 Project Overview

This project was developed for the **Meesho Data Challenge 2024**, focusing on predicting **fine-grained product attributes** (e.g., color, pattern, style) directly from e-commerce product images.  
Our final solution achieved a **0.802 private leaderboard score** using a **dual-backbone deep-learning ensemble** combining **CLIP ViT-H/14** and **ConvNext-XXLarge**, with **category-aware MLP classifiers** for robust, multi-attribute prediction.

The system is optimized for both **accuracy and inference efficiency**, capable of classifying an image in **0.05 seconds on an NVIDIA T4 GPU**.

---

## 🚀 Getting Started

### 1️⃣ Environment Setup

- Install Python **3.10+** and set up a virtual environment.
- Install **PyTorch with CUDA** (if available):
  ```bash
  pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu121
  ```
- Ensure you have a CUDA-capable GPU (**minimum 8GB VRAM**).
- Install all dependencies:
  ```bash
  pip install -r requirements.txt
  ```

---

### ⚙️ Configuration

Before running any scripts, configure your training parameters in:
```bash
config.yaml
```

---

## 🧠 Training

### Option 1: Training with Validation
Recommended during development:
```bash
python train_with_val.py
```
This will:
- Split the dataset into **90% training** and **10% validation**
- Display validation metrics after each epoch

### Option 2: Full Dataset Training
For final model training:
```bash
python main.py
```
This will:
- Train on the complete dataset  
- Save model checkpoints periodically  
- Generate training metrics and logs  

---

## 🏠 Model Architecture

### Training Pipeline
<img src="assets/training_pipeline.jpg" alt="Training Pipeline">

### Inference Pipeline
<div style="display: flex; align-items: flex-start;">
  <img src="assets/inference_pipeline.jpg" alt="Inference Pipeline" width="400" style="margin-right: 15px;">
  <p>
    Our final model is a <b>weighted ensemble</b> of two architectures — <b>ViT-H/14-quickgelu</b> and <b>ConvNext-XXLarge</b>. Predictions from both networks are combined using weighted averaging to optimize overall performance.
  </p>
</div>

---

## 🌟 Key Features

| Model | Description | Strengths |
|--------|--------------|------------|
| **ViT-H/14-quickgelu** | Transformer-based model optimized for fine-grained feature extraction, capturing intricate visual patterns. | Ideal for detailed attribute recognition tasks. |
| **ConvNext-XXLarge** | Convolutional model known for its robust classification power and efficiency on large datasets. | Handles diverse product categories effectively. |

---

## 📊 Results

### Leaderboard Comparison

| **Approach Type** | **Model & Technique** | **Public Score** | **Private Score** |
|--------------------|-----------------------|------------------|-------------------|
| **VQA using VLM** | Finetuned Qwen2VL-7B (instruct model using VQA) | 0.551 | 0.583 |
| **Image Similarity (Majority Voting)** | Hashing | 0.337 | 0.342 |
|  | SeResNext + FAISS | 0.670 | 0.669 |
|  | Swin Transformer + FAISS | 0.606 | 0.605 |
|  | Frozen CLIP ViT B/32 | 0.723 | 0.724 |
|  | Frozen CLIP ViT L/14 | 0.777 | 0.778 |
| **Classification (MLP Head)** | CLIP ViT-B/32 | 0.765 | 0.765 |
|  | CLIP ViT-L/14 | 0.770 | 0.771 |
|  | CLIP ViT-L/14 (optimized) | 0.785 | 0.785 |
|  | CLIP ViT-L/14 (with background removal) | 0.782 | 0.779 |
|  | CoCa | 0.797 | 0.794 |
|  | ConvNext-XXLarge | 0.801 | 0.799 |
|  | ViT-H/14-quickgelu | **0.806** | **0.800** |
| **Ensemble (Final)** | ConvNext-XXLarge + ViT-H/14-quickgelu | 🏆 **0.807** | 🏆 **0.802** |

---

## 🧩 Highlights

- **Dual-backbone ensemble** integrating CLIP ViT-H/14 and ConvNext-XXLarge  
- **Category-aware MLP heads** enabling precise multi-attribute classification  
- **Layer normalization and dropout** for improved regularization and generalization  
- **Mixed-precision training** and LR scheduling for stable optimization  
- **0.05s inference per image** on NVIDIA T4 GPU  

---

## 🧮 Technologies Used

- **Frameworks:** PyTorch, OpenCLIP  
- **Models:** CLIP ViT-H/14, ConvNext-XXLarge  
- **Techniques:** Ensemble Learning, Multi-Attribute Classification, MLP, Layer Normalization, Dropout  
- **Environment:** CUDA 12.1, Python 3.10+

---
