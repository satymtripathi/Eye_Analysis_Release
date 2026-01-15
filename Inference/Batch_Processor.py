import os
import shutil
import json
import numpy as np
import cv2
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image, ImageFile, ImageOps
import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp

ImageFile.LOAD_TRUNCATED_IMAGES = True

# ================== CONFIGURATION ==================
INPUT_DIR = r"C:\Users\satyam.tripathi\Downloads\30images\data_nonslit"
OUTPUT_DIR = "classified"
ACCEPTED_DIR = os.path.join(OUTPUT_DIR, "Accepted")
REJECTED_DIR = os.path.join(OUTPUT_DIR, "Rejected")

ROOT = os.path.dirname(os.path.abspath(__file__))
MODELS_ROOT = os.path.join(ROOT, "..", "Models")

MODEL_DIR = os.path.join(MODELS_ROOT, "Quality_Model")
GRAFT_MODEL_DIR = os.path.join(MODELS_ROOT, "Graft_Model")
STD_MODEL_PATH = os.path.join(MODELS_ROOT, "Segmentation_Models", "model_standard.pth")
SLIT_MODEL_PATH = os.path.join(MODELS_ROOT, "Segmentation_Models", "model_slitlamp.pth")

IMG_SIZE = 512
SEG_IMG_SIZE = 768
GOOD_CLASS_NAME = "good_eye"
GRAFT_CLASS_NAME = "Graft"

GOOD_ACCEPT_THRESH = 0.60
UNCERTAIN_THRESH = 0.55

SUPPORTED_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# ====================================================

def setup_dirs():
    os.makedirs(ACCEPTED_DIR, exist_ok=True)
    os.makedirs(REJECTED_DIR, exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, "Graft"), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, "No_Graft"), exist_ok=True)

def load_quality_model(model_dir):
    class_path = os.path.join(model_dir, "class_names.json")
    ckpt_path = os.path.join(model_dir, "best_model.pt")

    with open(class_path, "r") as f:
        class_names = json.load(f)

    model = models.efficientnet_b0(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, len(class_names))

    state = torch.load(ckpt_path, map_location=DEVICE)
    model.load_state_dict(state)
    model.to(DEVICE).eval()

    tfm = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
    ])

    return model, class_names, tfm

def load_graft_model(model_dir):
    class_path = os.path.join(model_dir, "class_names.json")
    ckpt_path = os.path.join(model_dir, "best_model.pt")
    
    # Fallback class names if json missing
    class_names = ["No_Graft", "Graft"]
    if os.path.exists(class_path):
        with open(class_path, "r") as f:
            class_names = json.load(f)

    model = models.efficientnet_b0(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, len(class_names))

    if os.path.exists(ckpt_path):
        state = torch.load(ckpt_path, map_location=DEVICE)
        model.load_state_dict(state)
    else:
        print(f"Warning: Graft model not found at {ckpt_path}")
        return None, class_names, None

    model.to(DEVICE).eval()
    
    tfm = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
    ])
    return model, class_names, tfm

def load_seg_models():
    models_dict = {}
    
    if os.path.exists(STD_MODEL_PATH):
        try:
            model_std = smp.UnetPlusPlus(encoder_name="timm-efficientnet-b0", in_channels=3, classes=1)
            model_std.load_state_dict(torch.load(STD_MODEL_PATH, map_location=DEVICE))
            model_std.to(DEVICE)
            model_std.eval()
            models_dict['standard'] = model_std
        except Exception as e:
            print(f"Failed to load Standard Model: {e}")
    
    if os.path.exists(SLIT_MODEL_PATH):
        try:
            model_slit = smp.UnetPlusPlus(encoder_name="timm-efficientnet-b0", in_channels=3, classes=1)
            model_slit.load_state_dict(torch.load(SLIT_MODEL_PATH, map_location=DEVICE))
            model_slit.to(DEVICE)
            model_slit.eval()
            models_dict['slitlamp'] = model_slit
        except Exception as e:
            print(f"Failed to load Slitlamp Model: {e}")
            
    return models_dict

def get_seg_transform():
    return A.Compose([
        A.Resize(SEG_IMG_SIZE, SEG_IMG_SIZE),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])

def predict_mask(model, image_rgb):
    transform = get_seg_transform()
    augmented = transform(image=image_rgb)["image"].unsqueeze(0).to(DEVICE)
    
    with torch.no_grad():
        output = model(augmented)
        prob = torch.sigmoid(output)[0, 0].cpu().numpy()
        
    return prob

def crop_image_from_prob(image_rgb, prob_map, padding=0):
    h, w = image_rgb.shape[:2]
    mask = (prob_map > 0.5).astype(np.uint8)
    mask_resized = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
    
    coords = cv2.findNonZero(mask_resized)
    if coords is not None:
        x, y, rect_w, rect_h = cv2.boundingRect(coords)
        x = max(0, x - padding)
        y = max(0, y - padding)
        rect_w = min(w - x, rect_w + 2*padding)
        rect_h = min(h - y, rect_h + 2*padding)
        crop = image_rgb[y:y+rect_h, x:x+rect_w]
        return crop
    return None

def detect_best_model(models_dict, image_rgb):
    if 'standard' not in models_dict or 'slitlamp' not in models_dict:
        return 'standard'
    
    prob_std = predict_mask(models_dict['standard'], image_rgb)
    prob_slit = predict_mask(models_dict['slitlamp'], image_rgb)
    
    score_std = np.mean(prob_std[prob_std > 0.5]) if np.max(prob_std) > 0.5 else 0
    score_slit = np.mean(prob_slit[prob_slit > 0.5]) if np.max(prob_slit) > 0.5 else 0
    
    if score_slit > score_std:
        return 'slitlamp'
    return 'standard'

def process_images():
    print(f"Source: {INPUT_DIR}")
    print(f"Output: {OUTPUT_DIR}")
    
    setup_dirs()
    
    print("Loading models...")
    quality_model, class_names, quality_tfm = load_quality_model(MODEL_DIR)
    graft_model, graft_classes, graft_tfm = load_graft_model(GRAFT_MODEL_DIR)
    seg_models = load_seg_models()
    
    good_idx = class_names.index(GOOD_CLASS_NAME)
    
    files = [f for f in os.listdir(INPUT_DIR) if os.path.splitext(f)[1].lower() in SUPPORTED_EXT]
    total = len(files)
    print(f"Found {total} images.")
    
    for i, fname in enumerate(files):
        fpath = os.path.join(INPUT_DIR, fname)
        print(f"[{i+1}/{total}] Processing: {fname}", end="... ")
        
        try:
            # 1. Load Image
            img = Image.open(fpath)
            img = ImageOps.exif_transpose(img)
            img = img.convert("RGB")
            image_np = np.array(img)
            
            # 2. Pipeline Detection & Cropping
            detected_type = detect_best_model(seg_models, image_np)
            processed_img = img
            
            if detected_type == 'standard':
                if 'standard' in seg_models:
                    prob = predict_mask(seg_models['standard'], image_np)
                    crop_np = crop_image_from_prob(image_np, prob)
                    if crop_np is not None:
                        processed_img = Image.fromarray(crop_np)
                        print(f"[STD-CROP]", end=" ")
                    else:
                        print(f"[STD-FAIL]", end=" ")
                else:
                    print(f"[STD-NO-MODEL]", end=" ")
            else:
                print(f"[SLIT]", end=" ")
                
            # 3. Quality Check
            x = quality_tfm(processed_img).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                probs = torch.softmax(quality_model(x), dim=1).cpu().numpy()[0]
                
            max_idx = int(np.argmax(probs))
            max_conf = float(probs[max_idx])
            pred_class = class_names[max_idx]
            good_conf = float(probs[good_idx])
            
            # 4. Decision
            is_accepted = (max_conf >= UNCERTAIN_THRESH) and \
                          (pred_class == GOOD_CLASS_NAME) and \
                          (good_conf >= GOOD_ACCEPT_THRESH)
            
            final_status = "REJECTED"
            dest_dir = REJECTED_DIR

            if is_accepted:
                # 5. Graft Detection
                if graft_model:
                    gx = graft_tfm(processed_img).unsqueeze(0).to(DEVICE)
                    with torch.no_grad():
                        gprobs = torch.softmax(graft_model(gx), dim=1).cpu().numpy()[0]
                    
                    g_idx = int(np.argmax(gprobs))
                    g_class = graft_classes[g_idx]
                    g_conf = float(gprobs[g_idx])
                    
                    if g_class == "Graft":
                        final_status = f"GRAFT ({g_conf:.2f})"
                        dest_dir = os.path.join(OUTPUT_DIR, "Graft")
                    else:
                        final_status = f"NO_GRAFT ({g_conf:.2f})"
                        dest_dir = os.path.join(OUTPUT_DIR, "No_Graft")
                else:
                    final_status = "ACCEPTED (Graft Model Missing)"
                    dest_dir = ACCEPTED_DIR

            shutil.copy2(fpath, os.path.join(dest_dir, fname))
            
            print(f"-> {final_status} (Good Prob: {good_conf:.4f})")
            
        except Exception as e:
            print(f"ERROR: {e}")

if __name__ == "__main__":
    process_images()
