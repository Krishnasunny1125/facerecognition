# 🎯 Face Recognition for Autonomous Drone Tracking

> A Vision Transformer (ViT) based face recognition system designed as the perception module of an intelligent **"Follow Me" drone** — capable of identifying a specific individual and tracking them in real time.

---

## 📌 Table of Contents
- [Overview](#overview)
- [Demo](#demo)
- [Architecture](#architecture)
- [Features](#features)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [Configuration](#configuration)
- [Results](#results)
- [Tech Stack](#tech-stack)
- [Future Work](#future-work)

---

## Overview

Traditional face recognition systems rely on CNN-based architectures. This project takes a modern approach using **Vision Transformers (ViT)** with **ArcFace loss** for superior embedding quality — making it robust enough for real-time drone tracking applications.

The system:
1. Trains a ViT model on a custom labeled face dataset
2. Builds an identity gallery from known individuals
3. Detects and recognizes faces in real-time video/images using cosine similarity

This module is designed to integrate directly with a drone's flight controller, feeding identity + position data to the tracking system.

---

## Demo

> *(Add a GIF or screenshot of real-time recognition here)*

```
Input Frame → Face Detection (Haar Cascade) → ViT Embedding → Cosine Similarity → Identity
```

---

## Architecture

```
Input Image (224×224)
      │
      ▼
Vision Transformer (vit_base_patch16_224)
  └── Patch Embedding (16×16 patches)
  └── 12 Transformer Encoder Blocks
  └── [CLS] Token
      │
      ▼
Custom Projection Head → 512-dim Face Embedding
      │
      ▼
ArcFace Loss (s=30, m=0.5) ← Training
      │
      ▼
Cosine Similarity vs Gallery → Identity + Score
```

**Key design choices:**
- **ViT over CNN**: Better long-range feature capture for subtle facial differences
- **ArcFace Loss**: Maximizes inter-class margin in embedding space — state-of-the-art for face recognition
- **Gallery-based inference**: No retraining needed to add new individuals

---

## Features

- ✅ Fine-tuned **Vision Transformer (ViT-Base/16)** with ImageNet pre-trained weights
- ✅ **ArcFace loss** for high-quality, discriminative face embeddings
- ✅ Real-time face detection using **Haar Cascade** (OpenCV)
- ✅ **Cosine similarity** based identity matching with configurable threshold
- ✅ Full training pipeline with validation split
- ✅ Comprehensive evaluation: Confusion matrix, ROC curves, Precision-Recall curves
- ✅ Easy dataset setup — one folder per person
- ✅ GPU/CPU auto-detection

---

## Project Structure

```
facerecognition-main/
├── model code.py          # Full training + inference pipeline
├── requirements.txt       # Python dependencies
└── README.md

# Expected dataset structure (not included):
dataset/
├── person_1/
│   ├── img1.jpg
│   └── img2.jpg
├── person_2/
│   └── ...
```

---

## Getting Started

### Prerequisites
- Python 3.8+
- CUDA-enabled GPU recommended (Kaggle T4 / local GPU)

### Installation

```bash
git clone https://github.com/YOUR_USERNAME/facerecognition.git
cd facerecognition
pip install -r requirements.txt
```

### Dataset Setup

Organize your face images in the following structure:
```
your_dataset/
├── Alice/
│   ├── alice_01.jpg
│   └── alice_02.jpg
├── Bob/
│   └── bob_01.jpg
```

### Configuration

Edit the `CONFIG` dictionary at the top of `model code.py`:

```python
CONFIG = {
    "data_dir":    "/path/to/your/dataset",
    "gallery_dir": "/path/to/your/dataset",
    "model_save_path": "./face_vit_model_best.pth",
    "num_epochs": 25,
    "batch_size": 32,
    "recognition_threshold": 0.60,  # Tune this for accuracy vs recall
    ...
}
```

### Run Training

```bash
python "model code.py"
```

The script will:
1. Load and split your dataset (80/20 train/val)
2. Train the ViT model with ArcFace loss
3. Save the best model checkpoint
4. Generate evaluation plots (confusion matrix, ROC, PR curves)
5. Build the identity gallery and run sample inference

---

## Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `img_size` | 224 | Input image size |
| `embedding_dim` | 512 | Face embedding dimensions |
| `vit_model_name` | `vit_base_patch16_224` | ViT variant from `timm` |
| `learning_rate` | 1e-4 | AdamW learning rate |
| `num_epochs` | 25 | Training epochs |
| `arcface_s` | 30.0 | ArcFace scale |
| `arcface_m` | 0.50 | ArcFace margin (radians) |
| `recognition_threshold` | 0.60 | Cosine similarity cutoff |

---

## Results

> *(Fill in your actual results after training)*

| Metric | Value |
|--------|-------|
| Validation Accuracy | — |
| Precision | — |
| Recall | — |
| F1 Score | — |
| AUC-ROC | — |

Evaluation plots generated automatically:
- `training_history.png` — Loss & accuracy curves
- `confusion_matrix.png` — Per-class confusion matrix
- `evaluation_curves.png` — ROC & PR curves

---

## Tech Stack

| Category | Technology |
|----------|-----------|
| Deep Learning | PyTorch, timm |
| Model Architecture | Vision Transformer (ViT-Base/16) |
| Loss Function | ArcFace (Additive Angular Margin) |
| Face Detection | OpenCV Haar Cascade |
| Evaluation | scikit-learn |
| Visualization | Matplotlib, Seaborn |
| Platform | Kaggle (GPU) / Local |

---

## Future Work

- [ ] Replace Haar Cascade with MTCNN or RetinaFace for better detection
- [ ] Add real-time webcam / video stream support
- [ ] Integrate with drone flight controller (MAVLink / ROS)
- [ ] Experiment with ViT-Large for higher accuracy
- [ ] Add face liveness detection to prevent spoofing
- [ ] Export model to ONNX for edge deployment

---

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.

---

## Author

> *(Add your name, LinkedIn, and portfolio link here)*

Made with ❤️ using Vision Transformers and PyTorch
