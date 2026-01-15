import os
# --- FIX FOR OMP ERROR ---
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# -------------------------

import json
import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim.lr_scheduler as lr_scheduler
import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp
import glob
import random
from tqdm import tqdm

# ==========================================
# 1. CONFIGURATION
# ==========================================
CONFIG = {
    "TARGET_CLASS": "roi_crop",
    "IMG_SIZE": (768, 768),    # Increased resolution
    "BATCH_SIZE": 2,           # Reduced batch size for higher res
    "LEARNING_RATE": 1e-4,    
    "EPOCHS": 50,             
    "DEVICE": "cuda" if torch.cuda.is_available() else "cpu",
    
    # LOSS PARAMS
    "ALPHA": 0.5,             
    "BETA": 0.5,              
    
    "ENCODER_NAME": "timm-efficientnet-b0",
    "MODEL_NAME": "best_roi_crop_model_Slitlamp.pth" 
}

print(f"Running on device: {CONFIG['DEVICE']}")
if CONFIG['DEVICE'] == 'cuda':
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")

# ==========================================
# 2. DATASET (ROI Specific)
# ==========================================
class ROIDataset(Dataset): 
    
    def __init__(self, data_pairs, transform=None):
        self.data_pairs = data_pairs
        self.transform = transform
        
    def __len__(self):
        return len(self.data_pairs)
    
    def _create_mask_from_json(self, json_path, img_shape):
        """
        Generates a binary mask (0 or 1) containing ONLY the 'crop' region
        by handling the 'rectangle' annotation type.
        """
        mask = np.zeros(img_shape, dtype=np.uint8) # 2D mask (H, W)
        
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
                
            for shape in data.get("shapes", []):
                label = shape.get("label", "").strip().lower()
                shape_type = shape.get("shape_type", "").lower()
                points = np.array(shape.get("points", []), dtype=np.float32)

                # Look for 'crop' label and 'rectangle' shape
                if "crop" == label and shape_type == "rectangle" and len(points) >= 2:
                    
                    # Rectangle points are typically [[x1, y1], [x2, y2]]
                    pt1 = tuple(points[0].astype(int))
                    pt2 = tuple(points[1].astype(int))
                    
                    # Draw a filled rectangle with value 1
                    cv2.rectangle(mask, pt1, pt2, color=1, thickness=-1)
            
        except Exception as e:
            print(f"Error processing {json_path}: {e}")
            
        return mask 

    def __getitem__(self, idx):
        img_path, json_path = self.data_pairs[idx]
        image = cv2.imread(img_path)
        
        if image is None:
             raise RuntimeError(f"Failed to load image at {img_path}")
             
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w, _ = image.shape
        
        mask = self._create_mask_from_json(json_path, (h, w)) 
        
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
            
        if mask.ndim == 2:
            mask = mask.unsqueeze(0)
            
        return image, mask.float()

# ==========================================
# 3. AUGMENTATION
# ==========================================
def get_transforms(phase):
    if phase == 'train':
        return A.Compose([
            A.Resize(CONFIG['IMG_SIZE'][0], CONFIG['IMG_SIZE'][1]),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ])
    else:
        return A.Compose([
            A.Resize(CONFIG['IMG_SIZE'][0], CONFIG['IMG_SIZE'][1]),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ])

# ==========================================
# 4. LOSS FUNCTION
# ==========================================
class TverskyLoss(nn.Module):
    def __init__(self, alpha=0.5, beta=0.5, smooth=1):
        super(TverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth

    def forward(self, inputs, targets):
        inputs = torch.sigmoid(inputs)
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        TP = (inputs * targets).sum()
        FP = ((1 - targets) * inputs).sum()
        FN = (targets * (1 - inputs)).sum()
        
        tversky_index = (TP + self.smooth) / (TP + self.alpha * FP + self.beta * FN + self.smooth)
        return 1 - tversky_index

# ==========================================
# 5. TRAINING PIPELINE
# ==========================================
def train_pipeline(data_path):
    # Data Split
    random.seed(42)
    full_dataset_pairs = []
    
    # Support multiple extensions
    image_paths = sorted(glob.glob(os.path.join(data_path, "*.jpg")) + 
                         glob.glob(os.path.join(data_path, "*.png")))
                         
    for img_path in image_paths:
        base_name = os.path.splitext(img_path)[0]
        json_path = base_name + ".json"
        if os.path.exists(json_path):
            full_dataset_pairs.append((img_path, json_path))
    
    if not full_dataset_pairs:
        print(f"Error: No image/json pairs found in {data_path}")
        return

    random.shuffle(full_dataset_pairs)
    train_size = int(0.8 * len(full_dataset_pairs))
    train_pairs = full_dataset_pairs[:train_size]
    val_pairs = full_dataset_pairs[train_size:]
    
    print(f"Total samples: {len(full_dataset_pairs)}")
    print(f"Training samples: {len(train_pairs)}")
    print(f"Validation samples: {len(val_pairs)}")

    train_dataset = ROIDataset(train_pairs, transform=get_transforms('train'))
    val_dataset = ROIDataset(val_pairs, transform=get_transforms('val'))

    train_loader = DataLoader(train_dataset, batch_size=CONFIG['BATCH_SIZE'], shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0)

    # Model
    model = smp.UnetPlusPlus(
        encoder_name=CONFIG['ENCODER_NAME'], 
        encoder_weights="imagenet",      
        in_channels=3,                     
        classes=1,                  
    )
    model.to(CONFIG['DEVICE'])

    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG['LEARNING_RATE'])
    criterion = TverskyLoss(alpha=CONFIG['ALPHA'], beta=CONFIG['BETA'])

    best_iou = 0.0
    print(f"Starting Training for {CONFIG['TARGET_CLASS']} at {CONFIG['IMG_SIZE']}...")
    
    for epoch in range(CONFIG['EPOCHS']):
        model.train()
        epoch_loss = 0
        
        train_loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{CONFIG['EPOCHS']} (Train)")
        
        for images, masks in train_loop:
            images = images.to(CONFIG['DEVICE'])
            masks = masks.to(CONFIG['DEVICE'])
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            train_loop.set_postfix(loss=loss.item())

        # Validation
        model.eval()
        val_iou_score = 0
        val_loop = tqdm(val_loader, desc=f"Epoch {epoch+1}/{CONFIG['EPOCHS']} (Val)")
        
        with torch.no_grad():
            for images, masks in val_loop:
                images = images.to(CONFIG['DEVICE'])
                masks = masks.to(CONFIG['DEVICE'])
                outputs = model(images)
                
                pred = torch.sigmoid(outputs)
                pred = (pred > 0.5).float()
                intersection = (pred * masks).sum()
                union = pred.sum() + masks.sum() - intersection
                iou = (intersection + 1e-6) / (union + 1e-6)
                val_iou_score += iou.item()
                val_loop.set_postfix(iou=iou.item())

        avg_val_iou = val_iou_score / len(val_loader) if len(val_loader) > 0 else 0
        
        print(f"\nEpoch {epoch+1}/{CONFIG['EPOCHS']} | Avg Loss: {epoch_loss/len(train_loader):.4f} | Avg Val IoU: {avg_val_iou:.4f}")

        if avg_val_iou > best_iou:
            best_iou = avg_val_iou
            torch.save(model.state_dict(), CONFIG['MODEL_NAME'])
            print(f"  >>> Best ROI Model Saved! New Best IoU: {best_iou:.4f}")

    print(f"\nTraining Complete. Final Best IoU: {best_iou:.4f}")

if __name__ == "__main__":
    DATA_FOLDER = r"C:\Users\satyam.tripathi\Downloads\30images\50_Cropping"
    
    if os.path.exists(DATA_FOLDER):
        train_pipeline(DATA_FOLDER)
    else:
        print(f"Error: Folder not found at {DATA_FOLDER}")
