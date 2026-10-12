# ♻️ Apple EcoLoop

> **AI-Powered Automated Cosmetic Grading for Apple Devices**  
> EcoLoop leverages computer vision and transfer learning to automatically assess, grade, and evaluate the physical condition of pre-owned and refurbished smartphones.

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 📌 Overview

Assessing the cosmetic condition of pre-owned Apple devices during trade-ins or refurbishment is traditionally manual, subjective, and slow. **EcoLoop** automates this evaluation by fine-tuning a deep residual network (ResNet18) to classify device photos into 4 standardized industry condition grades: **A**, **B**, **C**, and **D**.

For a deep dive into the business and architectural design, see [CONTEXT.md](file:///Users/himanshuthakur/Documents/Appleloop/CONTEXT.md).  
For developer tips, data collection clues, and training instructions, see [AGENT.md](file:///Users/himanshuthakur/Documents/Appleloop/AGENT.md).

---

## ✨ Features

- **Transfer Learning Backbone**: Utilizes a pretrained `ResNet18` with frozen feature extractors for high sample efficiency on small datasets (~100–300 images).
- **Data Augmentation**: Built-in `RandomHorizontalFlip` and `ColorJitter` to handle varying photography conditions, lighting, and angles.
- **Automated Checkpointing**: Tracks validation accuracy per epoch and automatically serializes the best weights to `grading_model.pt`.
- **Built-in Inference Helper**: Simple [`predict(image_path)`](file:///Users/himanshuthakur/Documents/Appleloop/train_grading_model.py#L98-L109) utility to evaluate standalone images and return grade predictions with softmax confidence scores.

---

## 🗂️ Project Structure

```text
Appleloop/
├── AGENT.md                 # Agent developer instructions, data sourcing clues & training guide
├── CONTEXT.md               # System context, business problem, and architecture
├── README.md                # Project documentation and quickstart
├── train_grading_model.py   # Model definition, training loop, and inference helper
└── data/                    # Dataset directory (created before training)
    ├── train/
    │   ├── A/
    │   ├── B/
    │   ├── C/
    │   └── D/
    └── val/
        ├── A/
        ├── B/
        ├── C/
        └── D/
```

---

## 🏷️ The 4-Tier Grading Standard

| Grade | Tier | Visual Indicators |
| :---: | :--- | :--- |
| **A** | **Pristine / Like New** | Flawless screen and housing; zero visible scratches or scuffs. |
| **B** | **Very Good / Light Wear** | Minor hairline micro-scratches not noticeable at arm's length (8+ inches); no cracks. |
| **C** | **Fair / Noticeable Wear** | Visible scratches, minor edge dents, or light housing scuffs; glass fully intact. |
| **D** | **Damaged / Faulty Glass** | Cracked screen, shattered back glass, severe dents, or broken camera lens. |

---

## 🚀 Quick Start

### 1. Clone & Set Up Environment

```bash
git clone https://github.com/himthakur5417-star/Apple_Ecoloop.git
cd Apple_Ecoloop

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install torch torchvision pillow
```

### 2. Prepare the Dataset

Organize your training and validation images inside a `data/` folder following the `ImageFolder` structure:

```text
data/
  train/ (A/, B/, C/, D/)
  val/   (A/, B/, C/, D/)
```

*(See [AGENT.md](file:///Users/himanshuthakur/Documents/Appleloop/AGENT.md) for data sourcing clues from Kaggle, Roboflow, and marketplaces).*

### 3. Train the Model

Run the training pipeline:

```bash
python3 train_grading_model.py
```

The script will train for 8 epochs and save the highest validation accuracy checkpoint to `grading_model.pt`.

### 4. Run Inference

Once trained, you can predict the condition of any phone image in Python:

```python
from train_grading_model import predict

grade, confidence = predict("path/to/phone_sample.jpg", weights_path="grading_model.pt")
print(f"Assigned Grade: {grade} (Confidence: {confidence:.2%})")
```

---

## 🗺️ Roadmap

- [x] **Module A**: ResNet18 transfer learning cosmetic grader.
- [ ] **Hardware Acceleration**: Enable Apple Silicon `mps` acceleration in training and inference.
- [ ] **REST API Service**: FastAPI service with `/grade` endpoint for multipart photo uploads.
- [ ] **Module B**: Dynamic trade-in valuation engine based on device specs + cosmetic grade.
- [ ] **Web Portal**: Interactive UI for customers to upload photos and receive instant valuations.

---

## 📄 License

This project is licensed under the MIT License.
