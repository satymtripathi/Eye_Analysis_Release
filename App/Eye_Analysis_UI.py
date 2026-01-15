import os, json
import numpy as np
from PIL import Image, ImageFile, ImageOps
ImageFile.LOAD_TRUNCATED_IMAGES = True

import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp

import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms, models

SUPPORTED_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

# ================== EDIT ONLY THIS ==================
ROOT = r"C:\Users\satyam.tripathi\Downloads\Original_Slit-lamp_Images\Original_Slit-lamp_Images"
MODEL_DIR = os.path.join(ROOT, "_eye_quality_model")   # your existing 3-class model
IMG_SIZE = 512
GOOD_CLASS_NAME = "good_eye"

# Standard Pipeline Config
STD_MODEL_PATH = os.path.join(ROOT, "Standard_Pipeline", "model_standard.pth")
SLIT_MODEL_PATH = os.path.join(ROOT, "Standard_Pipeline", "model_slitlamp.pth")
SEG_IMG_SIZE = 768

GOOD_ACCEPT_THRESH = 0.60   # accept only if good_eye prob >= this
UNCERTAIN_THRESH = 0.55     # if max prob < this => treat as not correct
# ====================================================

@st.cache_resource
def load_model(model_dir: str):
    class_path = os.path.join(model_dir, "class_names.json")
    ckpt_path = os.path.join(model_dir, "best_model.pt")

    if not os.path.exists(class_path):
        raise FileNotFoundError(f"Missing: {class_path}")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Missing: {ckpt_path}")

    with open(class_path, "r") as f:
        class_names = json.load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = models.efficientnet_b0(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, len(class_names))

    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state)
    model.to(device).eval()

    tfm = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
    ])

    return model, class_names, tfm, device

@st.cache_resource
def load_seg_models(device):
    models_dict = {}
    
    # 1. Standard Model
    if os.path.exists(STD_MODEL_PATH):
        try:
            model_std = smp.UnetPlusPlus(encoder_name="timm-efficientnet-b0", in_channels=3, classes=1)
            model_std.load_state_dict(torch.load(STD_MODEL_PATH, map_location=device))
            model_std.to(device)
            model_std.eval()
            models_dict['standard'] = model_std
        except Exception as e:
            st.error(f"Failed to load Standard Model: {e}")
    
    # 2. Slitlamp Model
    if os.path.exists(SLIT_MODEL_PATH):
        try:
            model_slit = smp.UnetPlusPlus(encoder_name="timm-efficientnet-b0", in_channels=3, classes=1)
            model_slit.load_state_dict(torch.load(SLIT_MODEL_PATH, map_location=device))
            model_slit.to(device)
            model_slit.eval()
            models_dict['slitlamp'] = model_slit
        except Exception as e:
            st.error(f"Failed to load Slitlamp Model: {e}")
            
    return models_dict

def get_seg_transform():
    return A.Compose([
        A.Resize(SEG_IMG_SIZE, SEG_IMG_SIZE),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])

def predict_mask(model, image_rgb, device):
    transform = get_seg_transform()
    augmented = transform(image=image_rgb)["image"].unsqueeze(0).to(device)
    
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

def detect_best_model(models_dict, image_rgb, device):
    if 'standard' not in models_dict or 'slitlamp' not in models_dict:
        return 'standard' # Default to standard if missing
    
    prob_std = predict_mask(models_dict['standard'], image_rgb, device)
    prob_slit = predict_mask(models_dict['slitlamp'], image_rgb, device)
    
    # Metric: Mean confidence of top pixels
    score_std = np.mean(prob_std[prob_std > 0.5]) if np.max(prob_std) > 0.5 else 0
    score_slit = np.mean(prob_slit[prob_slit > 0.5]) if np.max(prob_slit) > 0.5 else 0
    
    if score_slit > score_std:
        return 'slitlamp'
    return 'standard'

def main():
    st.set_page_config(page_title="Eye Gatekeeper", layout="centered")
    st.title("Eye Image Gatekeeper")

    try:
        model, class_names, tfm, device = load_model(MODEL_DIR)
        seg_models = load_seg_models(device)
    except Exception as e:
        st.error(f"Model load failed: {e}")
        st.stop()

    if GOOD_CLASS_NAME not in class_names:
        st.error(f"GOOD_CLASS_NAME='{GOOD_CLASS_NAME}' not found in class_names={class_names}")
        st.stop()

    good_idx = class_names.index(GOOD_CLASS_NAME)

    up = st.file_uploader("Upload image", type=[e.strip(".") for e in SUPPORTED_EXT])

    if up is None:
        st.info("Upload an image to check.")
        return

    img = Image.open(up)
    img = ImageOps.exif_transpose(img)
    img = img.convert("RGB")
    

    # --- STANDARD PIPELINE INTEGRATION ---
    image_np = np.array(img)
    
    # 1. Detect Standard vs Slitlamp
    detected_type = detect_best_model(seg_models, image_np, device)
    st.info(f"Pipeline Detected: **{detected_type.upper()}**")
    
    # 2. If Standard, Select ROI (Crop)
    processed_img = img
    if detected_type == 'standard':
        with st.spinner("Standard Image Detected. Cropping ROI..."):
            if 'standard' in seg_models:
                prob = predict_mask(seg_models['standard'], image_np, device)
                crop_np = crop_image_from_prob(image_np, prob)
                
                if crop_np is not None:
                    processed_img = Image.fromarray(crop_np)
                    st.success("ROI Cropped Successfully ✅")
                    # Show comparison
                    c1, c2 = st.columns(2)
                    c1.image(img, caption="Original", width="stretch")
                    c2.image(processed_img, caption="Cropped ROI", width="stretch")
                else:
                    st.warning("Could not crop ROI. Using original.")
                    st.image(img, caption="Original (ROI Failed)", width="stretch")
            else:
                 st.image(img, caption="Original (Model Missing)", width="stretch")
    else:
        st.write("Using full image (Slit Lamp).")
        st.image(img, caption="Uploaded (Slit Lamp)", width="stretch")

    # --- END STANDARD PIPELINE ---

    x = tfm(processed_img).unsqueeze(0).to(device)
    with torch.no_grad():
        probs = torch.softmax(model(x), dim=1).cpu().numpy()[0]

    max_idx = int(np.argmax(probs))
    max_conf = float(probs[max_idx])
    pred_class = class_names[max_idx]
    good_conf = float(probs[good_idx])
    st.write(f"Good Eye Probability: **{good_conf:.4f}**")

    # Decision:
    # ACCEPT only if predicted good AND good_conf >= threshold AND model is not uncertain
    if (max_conf >= UNCERTAIN_THRESH) and (pred_class == GOOD_CLASS_NAME) and (good_conf >= GOOD_ACCEPT_THRESH):
        st.success(f"This is good eye image ✅")
        st.write(f"Confidence: **{good_conf:.3f}**")
    else:
        st.error("This is not correct eye image ❌")

        # confidence should be max of bad/non (i.e., best non-good confidence)
        probs_non_good = probs.copy()
        probs_non_good[good_idx] = -1.0
        non_good_conf = float(np.max(probs_non_good))

        st.write(f"Confidence: **{non_good_conf:.3f}**")

if __name__ == "__main__":
    main()
