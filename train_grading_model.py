"""
EcoLoop — Module A trainer
Fine-tunes a ResNet18 to grade phone condition (A/B/C/D) from photos.

Usage:
1. Get data: ~100-300 labeled phone images (Kaggle/Roboflow, or your own),
   organized as:
     data/
       train/A/*.jpg  train/B/*.jpg  train/C/*.jpg  train/D/*.jpg
       val/A/*.jpg    val/B/*.jpg    val/C/*.jpg    val/D/*.jpg
2. Run in Colab (free GPU) or locally: python train_grading_model.py
3. Output: grading_model.pt — load it in a FastAPI backend for inference.

This mirrors Step 3 of the build guide: transfer learning on a pretrained
ResNet18, fine-tuned on a small labeled set, rather than training from scratch.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

CLASSES = ["A", "B", "C", "D"]
DATA_DIR = "data"
EPOCHS = 8
BATCH_SIZE = 16
LR = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

transform = {
    "train": transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]),
    "val": transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]),
}


def build_model():
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    for param in model.parameters():
        param.requires_grad = False  # freeze backbone
    model.fc = nn.Linear(model.fc.in_features, len(CLASSES))  # new head, trainable
    return model.to(DEVICE)


def run_epoch(model, loader, criterion, optimizer=None):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()
    total_loss, correct, n = 0.0, 0, 0
    with torch.set_grad_enabled(is_train):
        for images, labels in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            loss = criterion(outputs, labels)
            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * images.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()
            n += images.size(0)
    return total_loss / n, correct / n


def main():
    train_ds = datasets.ImageFolder(f"{DATA_DIR}/train", transform["train"])
    val_ds = datasets.ImageFolder(f"{DATA_DIR}/val", transform["val"])
    assert train_ds.classes == CLASSES, f"Expected class folders {CLASSES}, found {train_ds.classes}"

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

    model = build_model()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.fc.parameters(), lr=LR)

    best_acc = 0.0
    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc = run_epoch(model, val_loader, criterion)
        print(f"epoch {epoch}/{EPOCHS}  train_loss={train_loss:.3f} train_acc={train_acc:.2%}  "
              f"val_loss={val_loss:.3f} val_acc={val_acc:.2%}")
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({"state_dict": model.state_dict(), "classes": CLASSES}, "grading_model.pt")

    print(f"Best val accuracy: {best_acc:.2%} — saved to grading_model.pt")


def predict(image_path, weights_path="grading_model.pt"):
    """Quick inference helper for a single image, once trained."""
    from PIL import Image
    ckpt = torch.load(weights_path, map_location=DEVICE)
    model = build_model()
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    img = transform["val"](Image.open(image_path).convert("RGB")).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        probs = torch.softmax(model(img), dim=1)[0]
    idx = probs.argmax().item()
    return ckpt["classes"][idx], probs[idx].item()


if __name__ == "__main__":
    main()
