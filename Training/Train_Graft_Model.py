import os, json, random
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
from PIL import Image, ImageOps, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm


# ================== EDIT ONLY THIS ==================
DATA_ROOT = r"C:\Users\satyam.tripathi\Downloads\Original_Slit-lamp_Images\Original_Slit-lamp_Images\Data_Graft"
POS_DIR = "Graft"       # label 1
NEG_DIR = "No_Graft"    # label 0

# If you have cornea cropper, set this to True and fill crop_cornea() below
USE_CORNEA_CROP = False
# ====================================================

# Training config
IMG_SIZE = 512
BATCH_SIZE = 4          # GPU: 8/16, CPU: 2/4
EPOCHS = 15
LR = 3e-4
WEIGHT_DECAY = 1e-4
VAL_SIZE = 0.2
SEED = 42
NUM_WORKERS = 2         # Windows: 0/2/4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

OUT_DIR = os.path.join(DATA_ROOT, "_graft_model")
os.makedirs(OUT_DIR, exist_ok=True)

SUPPORTED_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


# --------- OPTIONAL: plug your cornea cropper here ----------
def crop_cornea(pil_img: Image.Image) -> Image.Image:
    """
    Replace this with your cornea cropper output.

    Return: PIL image of corneal ROI (recommended: include padding/context)
    """
    # Example fallback (no crop):
    return pil_img
# -----------------------------------------------------------


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def is_image(p: Path) -> bool:
    return p.suffix.lower() in SUPPORTED_EXT


def collect_files(data_root: str) -> Tuple[List[str], List[int]]:
    pos_path = Path(data_root) / POS_DIR
    neg_path = Path(data_root) / NEG_DIR
    if not pos_path.exists(): raise FileNotFoundError(pos_path)
    if not neg_path.exists(): raise FileNotFoundError(neg_path)

    pos_files = [str(p) for p in pos_path.rglob("*") if p.is_file() and is_image(p)]
    neg_files = [str(p) for p in neg_path.rglob("*") if p.is_file() and is_image(p)]

    paths = neg_files + pos_files
    labels = [0] * len(neg_files) + [1] * len(pos_files)

    print(f"No_Graft (0): {len(neg_files)}")
    print(f"Graft    (1): {len(pos_files)}")
    return paths, labels


class GraftDataset(Dataset):
    def __init__(self, paths: List[str], labels: List[int], tfm):
        self.paths = paths
        self.labels = labels
        self.tfm = tfm

    def __len__(self): return len(self.paths)

    def __getitem__(self, idx):
        p = self.paths[idx]
        y = self.labels[idx]

        img = Image.open(p)
        img = ImageOps.exif_transpose(img).convert("RGB")

        if USE_CORNEA_CROP:
            img = crop_cornea(img)

        x = self.tfm(img)
        return x, y, p


def build_tfms(img_size: int):
    train = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomResizedCrop(img_size, scale=(0.75, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomApply([transforms.ColorJitter(0.25, 0.25, 0.10, 0.02)], p=0.6),
        transforms.RandomApply([transforms.GaussianBlur(7, sigma=(0.1, 2.0))], p=0.25),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
        transforms.RandomErasing(p=0.25, scale=(0.02, 0.10), ratio=(0.3, 3.3), value="random"),
    ])

    val = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
    ])
    return train, val


def build_model():
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, 2)
    return model


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    ys, ps = [], []
    for x, y, _ in loader:
        x = x.to(DEVICE)
        y = y.to(DEVICE)
        logits = model(x)
        pred = torch.argmax(logits, dim=1)
        ys += y.cpu().tolist()
        ps += pred.cpu().tolist()

    acc = float((np.array(ys) == np.array(ps)).mean())
    rep = classification_report(ys, ps, target_names=["No_Graft","Graft"], output_dict=True, zero_division=0)
    cm = confusion_matrix(ys, ps)
    return acc, rep, cm


def train_one_epoch(model, loader, optimizer, loss_fn):
    model.train()
    total = 0.0
    for x, y, _ in tqdm(loader, desc="train", leave=False):
        x = x.to(DEVICE)
        y = y.to(DEVICE)

        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = loss_fn(logits, y)
        loss.backward()
        optimizer.step()

        total += loss.item() * x.size(0)
    return total / len(loader.dataset)


def main():
    set_seed(SEED)
    print("Device:", DEVICE)
    print("USE_CORNEA_CROP:", USE_CORNEA_CROP)

    paths, labels = collect_files(DATA_ROOT)

    tr_p, va_p, tr_y, va_y = train_test_split(
        paths, labels, test_size=VAL_SIZE, random_state=SEED, stratify=labels
    )
    print(f"Train: {len(tr_p)} | Val: {len(va_p)}")

    train_t, val_t = build_tfms(IMG_SIZE)

    ds_tr = GraftDataset(tr_p, tr_y, train_t)
    ds_va = GraftDataset(va_p, va_y, val_t)

    dl_tr = DataLoader(ds_tr, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS,
                       pin_memory=(DEVICE=="cuda"))
    dl_va = DataLoader(ds_va, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS,
                       pin_memory=(DEVICE=="cuda"))

    model = build_model().to(DEVICE)

    # class weights to handle imbalance
    counts = np.bincount(np.array(tr_y), minlength=2)  # [No_Graft, Graft]
    weights = counts.sum() / (counts + 1e-6)
    weights = weights / weights.mean()
    class_w = torch.tensor(weights, dtype=torch.float32).to(DEVICE)

    print("Train counts [No_Graft, Graft]:", counts.tolist())
    print("Class weights:", [float(w) for w in weights])

    loss_fn = nn.CrossEntropyLoss(weight=class_w)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    best_acc = -1.0
    best_row = None

    for ep in range(1, EPOCHS + 1):
        tr_loss = train_one_epoch(model, dl_tr, optimizer, loss_fn)
        va_acc, rep, cm = evaluate(model, dl_va)

        print(f"\nEpoch {ep}/{EPOCHS} | train_loss={tr_loss:.4f} | val_acc={va_acc:.4f}")
        print("Confusion matrix (rows=true, cols=pred):")
        print(cm)

        row = {
            "epoch": ep,
            "train_loss": float(tr_loss),
            "val_acc": float(va_acc),
            "confusion_matrix": cm.tolist(),
            "report": rep
        }

        if va_acc > best_acc:
            best_acc = va_acc
            best_row = row
            torch.save(model.state_dict(), os.path.join(OUT_DIR, "best_model.pt"))
            with open(os.path.join(OUT_DIR, "class_names.json"), "w") as f:
                json.dump(["No_Graft","Graft"], f, indent=2)
            with open(os.path.join(OUT_DIR, "best_metrics.json"), "w") as f:
                json.dump(best_row, f, indent=2)
            print("✅ Saved best_model.pt")

    print("\nDONE ✅ Best val_acc:", best_acc)
    print("Saved to:", OUT_DIR)


if __name__ == "__main__":
    main()
