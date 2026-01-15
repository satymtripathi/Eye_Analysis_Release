import os, json, random
from pathlib import Path
from typing import Tuple, List

import numpy as np
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

from tqdm import tqdm

# ================== EDIT ONLY THIS ==================
DATA_ROOT = r"C:\Users\satyam.tripathi\Downloads\Original_Slit-lamp_Images\Original_Slit-lamp_Images"
# Folder names must exist inside DATA_ROOT:
CLASS_FOLDERS = ["good_eye", "bad_eye", "non_eye"]
# ====================================================

# Training config
IMG_SIZE = 512
BATCH_SIZE = 4          # 512 is heavy; increase if GPU + memory allows
EPOCHS = 20
LR = 3e-4
WEIGHT_DECAY = 1e-4
VAL_SIZE = 0.2
SEED = 42
NUM_WORKERS = 2         # on Windows keep small (0/2/4)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

OUT_DIR = os.path.join(DATA_ROOT, "_eye_quality_model")
os.makedirs(OUT_DIR, exist_ok=True)

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def is_image_file(p: Path) -> bool:
    return p.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"]

def collect_files(data_root: str, class_folders: List[str]) -> Tuple[List[str], List[int], List[str]]:
    paths, labels = [], []
    for i, cname in enumerate(class_folders):
        cdir = Path(data_root) / cname
        if not cdir.exists():
            raise FileNotFoundError(f"Missing folder: {cdir}")
        files = [str(p) for p in cdir.rglob("*") if p.is_file() and is_image_file(p)]
        paths.extend(files)
        labels.extend([i] * len(files))
        print(f"{cname}: {len(files)} images")
    return paths, labels, class_folders

class ImageClassificationDataset(Dataset):
    def __init__(self, paths: List[str], labels: List[int], tfm=None):
        self.paths = paths
        self.labels = labels
        self.tfm = tfm

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        p = self.paths[idx]
        y = self.labels[idx]
        img = Image.open(p).convert("RGB")
        if self.tfm:
            img = self.tfm(img)
        return img, y, p

def build_transforms(img_size: int):
    # Train augmentations (keep realistic for slit-lamp)
    train_tfm = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomApply([
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.02)
        ], p=0.6),
        transforms.RandomApply([
            transforms.GaussianBlur(kernel_size=7, sigma=(0.1, 2.0))
        ], p=0.25),
        transforms.ToTensor(),
        # ImageNet normalization for EfficientNet
        transforms.Normalize(mean=(0.485, 0.456, 0.406),
                             std=(0.229, 0.224, 0.225)),
    ])

    val_tfm = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406),
                             std=(0.229, 0.224, 0.225)),
    ])
    return train_tfm, val_tfm

def build_model(num_classes: int):
    # EfficientNet-B0 is a good start; switch to b2/b3 if you want later
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    return model

@torch.no_grad()
def evaluate(model, loader, class_names):
    model.eval()
    all_y, all_pred = [], []
    for x, y, _ in loader:
        x = x.to(DEVICE)
        y = y.to(DEVICE)
        logits = model(x)
        pred = torch.argmax(logits, dim=1)
        all_y.extend(y.cpu().numpy().tolist())
        all_pred.extend(pred.cpu().numpy().tolist())

    report = classification_report(all_y, all_pred, target_names=class_names, output_dict=True, zero_division=0)
    cm = confusion_matrix(all_y, all_pred)
    acc = (np.array(all_y) == np.array(all_pred)).mean()
    return acc, report, cm

def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    running_loss = 0.0
    for x, y, _ in tqdm(loader, desc="train", leave=False):
        x = x.to(DEVICE)
        y = y.to(DEVICE)

        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * x.size(0)
    return running_loss / len(loader.dataset)

def main():
    set_seed(SEED)
    print("Device:", DEVICE)

    paths, labels, class_names = collect_files(DATA_ROOT, CLASS_FOLDERS)

    # Stratified split so each class stays balanced
    tr_paths, va_paths, tr_y, va_y = train_test_split(
        paths, labels, test_size=VAL_SIZE, random_state=SEED, stratify=labels
    )
    print(f"Train: {len(tr_paths)} | Val: {len(va_paths)}")

    train_tfm, val_tfm = build_transforms(IMG_SIZE)

    ds_train = ImageClassificationDataset(tr_paths, tr_y, tfm=train_tfm)
    ds_val   = ImageClassificationDataset(va_paths, va_y, tfm=val_tfm)

    dl_train = DataLoader(ds_train, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=NUM_WORKERS, pin_memory=(DEVICE=="cuda"))
    dl_val   = DataLoader(ds_val, batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=NUM_WORKERS, pin_memory=(DEVICE=="cuda"))

    model = build_model(num_classes=len(class_names)).to(DEVICE)

    # If class imbalance exists, add weights automatically
    counts = np.bincount(np.array(tr_y), minlength=len(class_names))
    weights = (counts.sum() / (counts + 1e-6))
    weights = weights / weights.mean()
    class_weights = torch.tensor(weights, dtype=torch.float32).to(DEVICE)
    print("Train class counts:", dict(zip(class_names, counts.tolist())))
    print("Class weights:", dict(zip(class_names, [float(w) for w in weights])))

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    best_acc = -1.0
    history = []

    for epoch in range(1, EPOCHS + 1):
        train_loss = train_one_epoch(model, dl_train, optimizer, criterion)
        val_acc, report, cm = evaluate(model, dl_val, class_names)

        row = {
            "epoch": epoch,
            "train_loss": float(train_loss),
            "val_acc": float(val_acc),
            "confusion_matrix": cm.tolist(),
            "report": report
        }
        history.append(row)

        print(f"\nEpoch {epoch}/{EPOCHS} | train_loss={train_loss:.4f} | val_acc={val_acc:.4f}")
        print("Confusion matrix (rows=true, cols=pred):")
        print(cm)

        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), os.path.join(OUT_DIR, "best_model.pt"))
            with open(os.path.join(OUT_DIR, "class_names.json"), "w") as f:
                json.dump(class_names, f, indent=2)
            with open(os.path.join(OUT_DIR, "best_metrics.json"), "w") as f:
                json.dump(row, f, indent=2)
            print("✅ Saved best_model.pt")

    with open(os.path.join(OUT_DIR, "train_history.json"), "w") as f:
        json.dump(history, f, indent=2)

    print("\nDONE ✅")
    print("Best val_acc:", best_acc)
    print("Saved to:", OUT_DIR)

if __name__ == "__main__":
    main()
