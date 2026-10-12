# Agent Operating Guide: Apple EcoLoop (Module A)

> **Role & Purpose**: This guide is written for AI agents and developers working on the **Apple EcoLoop** repository. It contains comprehensive instructions on sourcing training data, structuring datasets, fine-tuning the vision model, debugging common training issues, and deploying the model for inference.

---

## 1. How to Get Data, Label, and Prepare the Dataset

Currently, the repository contains the training script [`train_grading_model.py`](file:///Users/himanshuthakur/Documents/Appleloop/train_grading_model.py), which expects training and validation images under a `data/` directory.

### 1.1 Where to Source Phone Grading Data

You need ~100–300 labeled phone images (minimum 25–75 images per class) to train an initial model. Here are the best sources and clues for collecting them:

1. **Public Kaggle Datasets**:
   - **Mobile Phone Defect Dataset**: Contains photos of phones with scratches, cracks, and clean bodies.
   - **Smartphone Screen Damage Dataset**: Images labeled with screen cracks, shattered glass, and spotless screens.
   - **Phone Surface Inspection Datasets**: Industrial quality-control inspection images for consumer electronics.
   - *Search keywords on Kaggle*: `phone damage`, `smartphone defect`, `screen crack`, `refurbished phone`.

2. **Roboflow Universe (Computer Vision Datasets)**:
   - Roboflow contains pre-annotated datasets of used smartphones.
   - *Search keywords on Roboflow*: `phone-crack`, `mobile-damage-grading`, `scratches-detection`, `screen-scratch`.
   - Export format: Folder-based classification or standard JPGs.

3. **E-Commerce & Refurbished Marketplace Scraping**:
   - Platforms like **Back Market**, **Swappa**, **Gazelle**, and **eBay Refurbished** already categorize phones into condition tiers:
     - Back Market: *Fair*, *Good*, *Excellent*, *Premium*.
     - Swappa: *Fair*, *Good*, *Mint*.
     - eBay: *Certified Refurbished*, *Excellent*, *Very Good*, *Good*.
   - Scraping or saving representative listing photos from these categories directly maps to Grades A, B, C, and D.

4. **Synthetic Data & Augmentation**:
   - If real images of severe damage (Grade D) or micro-scratches (Grade B) are scarce, augment clean phone images with overlay masks of scratches or cracks.

---

### 1.2 The 4-Tier Grading Standard (Labeling Guide)

Follow this labeling rubric to classify images into `A`, `B`, `C`, or `D`:

| Grade | Label | Description & Visual Clues | Criteria |
| :---: | :---: | :--- | :--- |
| **A** | **Pristine / Like New** | Device looks flawless right out of the box. | • Screen: Zero scratches, zero dead pixels.<br>• Housing: No scuffs, dents, or paint chipping.<br>• Camera lens: 100% spotless. |
| **B** | **Very Good / Light Wear** | Normal gentle use. No damage visible at arm's length (8+ inches). | • Screen: Micro-scratches visible only under direct angled light.<br>• Housing: Faint hairline scuffs near charging port or edges.<br>• No cracks, no deep gouges. |
| **C** | **Good / Fair Wear** | Noticeable cosmetic wear from daily use without a case. | • Screen: Noticeable scratches that catch a fingernail, but **no cracks**.<br>• Housing: Visible scuffs, minor corner dents, or edge blemishes.<br>• Camera glass intact. |
| **D** | **Damaged / Faulty Glass** | Severe cosmetic flaws, structural damage, or broken glass. | • Screen: Hairline cracks, web fractures, or shattered glass.<br>• Back Glass: Cracked or shattered glass.<br>• Housing: Heavy impact dents or bent frame.<br>• Broken camera lens cover. |

---

### 1.3 Expected Folder Structure

PyTorch's `datasets.ImageFolder` requires the following exact folder hierarchy:

```text
Appleloop/
├── data/
│   ├── train/
│   │   ├── A/       # ~50-100+ images of Grade A phones (.jpg, .jpeg, .png)
│   │   ├── B/       # ~50-100+ images of Grade B phones
│   │   ├── C/       # ~50-100+ images of Grade C phones
│   │   └── D/       # ~50-100+ images of Grade D phones
│   └── val/
│       ├── A/       # ~15-30 images of Grade A phones
│       ├── B/       # ~15-30 images of Grade B phones
│       ├── C/       # ~15-30 images of Grade C phones
│       └── D/       # ~15-30 images of Grade D phones
├── train_grading_model.py
└── grading_model.pt  # Generated after training
```

> **Data Preparation Tips**:
> - Ensure class balance: keep roughly similar numbers of images across `A`, `B`, `C`, and `D`.
> - Include photos from multiple angles (front screen, back glass, frame edges).
> - Remove corrupted or non-RGB images before training.

---

## 2. Environment Setup & Hardware Acceleration

### 2.1 Dependencies Installation

Create a virtual environment and install the required dependencies:

```bash
# Create and activate virtual environment (optional but recommended)
python3 -m venv venv
source venv/bin/activate

# Install PyTorch, torchvision, and imaging libraries
pip install torch torchvision pillow
```

### 2.2 Device Detection & Apple Silicon (MPS) Clue

In [`train_grading_model.py`](file:///Users/himanshuthakur/Documents/Appleloop/train_grading_model.py#L28), line 28 currently reads:
```python
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
```

On Apple Silicon (M1/M2/M3/M4 Macs), you can accelerate training using Apple's Metal Performance Shaders (`mps`). The improved device selection pattern is:

```python
if torch.cuda.is_available():
    DEVICE = "cuda"
elif torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"
```

---

## 3. How to Train the Model

### 3.1 Running the Training Script

Once the `data/` folder is populated with `train` and `val` images:

```bash
python3 train_grading_model.py
```

### 3.2 What Happens During Training

1. **Loads Datasets**: Checks that folders `A`, `B`, `C`, and `D` exist under `data/train` and `data/val`.
2. **Transforms & Augments**:
   - `train`: Resizes to 224×224, random horizontal flip, color jitter (brightness & contrast ±20%), ImageNet normalization.
   - `val`: Resizes to 224×224, ImageNet normalization.
3. **Builds Model**:
   - Downloads pretrained weights for `ResNet18`.
   - Freezes all convolutional backbone layers (`param.requires_grad = False`).
   - Replaces final fully connected head `model.fc` with `nn.Linear(512, 4)`.
4. **Executes 8 Epochs**:
   - Runs forward and backward passes on the trainable classification head.
   - Calculates cross-entropy loss and accuracy on training and validation splits.
5. **Checkpoints Best Model**:
   - Whenever validation accuracy reaches a new high, saves `grading_model.pt`:
     ```python
     torch.save({"state_dict": model.state_dict(), "classes": CLASSES}, "grading_model.pt")
     ```

### 3.3 Key Hyperparameters to Tune

If accuracy is low or training behaves unexpectedly, tune these variables in [`train_grading_model.py`](file:///Users/himanshuthakur/Documents/Appleloop/train_grading_model.py#L23-L27):

| Parameter | Default | When to Adjust |
| :--- | :--- | :--- |
| `EPOCHS` | `8` | Increase to `15-20` if validation loss is still steadily decreasing. |
| `BATCH_SIZE`| `16` | Lower to `8` if training on low-memory CPU/GPU; increase to `32` if you have ample GPU VRAM. |
| `LR` | `1e-4` | If loss oscillates wildly, reduce to `5e-5`. If loss barely changes, try `3e-4`. |
| `Backbone Fine-tuning` | Frozen | For higher accuracy with >500 images, unfreeze `model.layer4` in a second training stage with lower learning rate (`1e-5`). |

---

## 4. How to Test & Run Inference

The file includes a built-in [`predict(image_path)`](file:///Users/himanshuthakur/Documents/Appleloop/train_grading_model.py#L98-L109) function.

### 4.1 Running a Prediction from Python

You can test any image against the trained `grading_model.pt` file:

```python
from train_grading_model import predict

# Predict the grade of an inspection photo
predicted_grade, confidence = predict("path/to/test_iphone.jpg", "grading_model.pt")

print(f"Condition Grade: {predicted_grade}")
print(f"Confidence: {confidence:.2%}")
```

---

## 5. Troubleshooting & Debugging Clues

- **`AssertionError: Expected class folders ['A', 'B', 'C', 'D'], found [...]`**:
  Ensure subdirectories under both `data/train/` and `data/val/` match the exact uppercase names `A`, `B`, `C`, and `D`.
- **`UnidentifiedImageError`**:
  Corrupted image file or hidden system file (such as `.DS_Store` on macOS). Remove non-image files with:
  ```bash
  find data/ -name ".DS_Store" -delete
  ```
- **Validation accuracy stuck around 25%**:
  25% is random guessing for 4 classes. Check if:
  - Dataset is too small (<15 images per class).
  - Learning rate is too high or images are improperly labeled.
  - Background clutter dominates the frame (the model is learning the table background instead of the phone screen).

---

## 6. Next Steps & Agent Roadmap

When expanding this repository, agents should prioritize:

1. **FastAPI Inference Service (`api.py`)**:
   - Provide a `POST /grade` endpoint accepting an uploaded multipart image.
   - Return `{ "grade": "B", "confidence": 0.91, "details": { ... } }`.
2. **Apple Silicon Hardware Acceleration**:
   - Add `mps` fallback in [`train_grading_model.py`](file:///Users/himanshuthakur/Documents/Appleloop/train_grading_model.py).
3. **Module B: Valuation Engine**:
   - Implement dynamic price quote calculation based on: `(Model, Storage, Battery Health, Grade A/B/C/D)`.
4. **Interactive Web Interface**:
   - Modern, sleek drag-and-drop UI for users to upload device photos and receive instant grading and trade-in quotes.
