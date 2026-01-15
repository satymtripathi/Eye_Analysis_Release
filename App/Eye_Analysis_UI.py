import streamlit as st
import os
import json
import numpy as np
import cv2
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image, ImageOps, ImageFile
import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp

ImageFile.LOAD_TRUNCATED_IMAGES = True

# ================== CONFIGURATION ==================
ROOT = os.path.dirname(os.path.abspath(__file__))
MODELS_ROOT = os.path.join(ROOT, "..", "Models")

MODEL_DIR = os.path.join(MODELS_ROOT, "Quality_Model")
GRAFT_MODEL_DIR = os.path.join(MODELS_ROOT, "Graft_Model")
STD_MODEL_PATH = os.path.join(MODELS_ROOT, "Segmentation_Models", "model_standard.pth")
SLIT_MODEL_PATH = os.path.join(MODELS_ROOT, "Segmentation_Models", "model_slitlamp.pth")

IMG_SIZE = 512
SEG_IMG_SIZE = 768
GOOD_CLASS_NAME = "good_eye"

GOOD_ACCEPT_THRESH = 0.60
UNCERTAIN_THRESH = 0.55

SUPPORTED_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# ====================================================

# ================== STYLES ==================
def inject_custom_css():
    st.markdown("""
    <style>
        .main {
            background-color: #F8F9FA;
        }
        .block-container {
            padding-top: 3rem;
            max-width: 900px;
        }
        .stButton>button {
            border-radius: 24px;
            padding: 0.5rem 2rem;
            border: 1px solid #E0E0E0;
            background: white;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
            transition: all 0.3s ease;
        }
        .stButton>button:hover {
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
            transform: translateY(-1px);
        }
        .metric-card {
            background: white;
            border-radius: 16px;
            padding: 24px;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
            margin-bottom: 24px;
            border: 1px solid #E5E7EB;
        }
        .metric-label {
            font-size: 0.875rem;
            color: #6B7280;
            font-weight: 500;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }
        .metric-value {
            font-size: 2.25rem;
            font-weight: 700;
            color: #111827;
            margin-top: 0.5rem;
        }
        .status-badge {
            display: inline-flex;
            align-items: center;
            padding: 6px 16px;
            border-radius: 9999px;
            font-size: 0.875rem;
            font-weight: 600;
            margin-top: 1rem;
        }
        .status-good {
            background-color: #DEF7EC;
            color: #03543F;
        }
        .status-bad {
            background-color: #FDE8E8;
            color: #9B1C1C;
        }
        .section-header {
            font-size: 1.5rem;
            font-weight: 600;
            color: #374151;
            margin-bottom: 1.5rem;
            margin-top: 2rem;
            border-bottom: 2px solid #E5E7EB;
            padding-bottom: 0.5rem;
        }
    </style>
    """, unsafe_allow_html=True)

@st.cache_resource
def load_all_models():
    # 1. Quality Model
    class_path = os.path.join(MODEL_DIR, "class_names.json")
    ckpt_path = os.path.join(MODEL_DIR, "best_model.pt")
    
    with open(class_path, "r") as f:
        q_classes = json.load(f)

    q_model = models.efficientnet_b0(weights=None)
    in_features = q_model.classifier[1].in_features
    q_model.classifier[1] = nn.Linear(in_features, len(q_classes))
    q_model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
    q_model.to(DEVICE).eval()
    
    q_tfm = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
    ])

    # 2. Graft Model
    g_class_path = os.path.join(GRAFT_MODEL_DIR, "class_names.json")
    g_ckpt_path = os.path.join(GRAFT_MODEL_DIR, "best_model.pt")
    
    g_classes = ["No_Graft", "Graft"]
    if os.path.exists(g_class_path):
        with open(g_class_path, "r") as f:
            g_classes = json.load(f)

    g_model = None
    g_tfm = None
    if os.path.exists(g_ckpt_path):
        g_model = models.efficientnet_b0(weights=None)
        in_features = g_model.classifier[1].in_features
        g_model.classifier[1] = nn.Linear(in_features, len(g_classes))
        g_model.load_state_dict(torch.load(g_ckpt_path, map_location=DEVICE))
        g_model.to(DEVICE).eval()
        
        g_tfm = transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
        ])

    # 3. Seg Models
    seg_models = {}
    if os.path.exists(STD_MODEL_PATH):
        try:
            m = smp.UnetPlusPlus(encoder_name="timm-efficientnet-b0", in_channels=3, classes=1)
            m.load_state_dict(torch.load(STD_MODEL_PATH, map_location=DEVICE))
            m.to(DEVICE).eval()
            seg_models['standard'] = m
        except: pass
        
    if os.path.exists(SLIT_MODEL_PATH):
        try:
            m = smp.UnetPlusPlus(encoder_name="timm-efficientnet-b0", in_channels=3, classes=1)
            m.load_state_dict(torch.load(SLIT_MODEL_PATH, map_location=DEVICE))
            m.to(DEVICE).eval()
            seg_models['slitlamp'] = m
        except: pass

    return (q_model, q_classes, q_tfm), (g_model, g_classes, g_tfm), seg_models

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
    if score_slit > score_std: return 'slitlamp'
    return 'standard'

def main():
    st.set_page_config(page_title="Eye Analysis Lab", layout="centered", initial_sidebar_state="collapsed")
    inject_custom_css()
    
    # Header
    st.markdown("<h1 style='text-align: center; color: #1F2937; margin-bottom: 2rem;'>Eye Analysis Lab 🔬</h1>", unsafe_allow_html=True)
    
    # Load Models
    (q_model, q_classes, q_tfm), (g_model, g_classes, g_tfm), seg_models = load_all_models()
    good_idx = q_classes.index(GOOD_CLASS_NAME)

    # Upload Section
    cols = st.columns([1, 2, 1])
    with cols[1]:
        up = st.file_uploader("", type=[e.strip(".") for e in SUPPORTED_EXT], help="Upload a slit-lamp or standard eye image")

    if up is not None:
        try:
            img = Image.open(up)
            img = ImageOps.exif_transpose(img)
            img = img.convert("RGB")
            image_np = np.array(img)
            
            # --- PIPELINE START ---
            
            # 1. Preprocessing
            detected_type = detect_best_model(seg_models, image_np)
            processed_img = img
            
            if detected_type == 'standard':
                if 'standard' in seg_models:
                    prob = predict_mask(seg_models['standard'], image_np)
                    crop_np = crop_image_from_prob(image_np, prob)
                    if crop_np is not None:
                        processed_img = Image.fromarray(crop_np)

            # 2. Quality Check
            x = q_tfm(processed_img).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                probs = torch.softmax(q_model(x), dim=1).cpu().numpy()[0]
                
            max_idx = int(np.argmax(probs))
            max_conf = float(probs[max_idx])
            pred_class = q_classes[max_idx]
            good_conf = float(probs[good_idx])
            
            is_accepted = (max_conf >= UNCERTAIN_THRESH) and \
                          (pred_class == GOOD_CLASS_NAME) and \
                          (good_conf >= GOOD_ACCEPT_THRESH)

            # --- DISPLAY RESULTS ---
            
            c1, c2 = st.columns([1, 1.2])
            
            with c1:
                st.markdown(f"<div class='metric-label' style='margin-bottom: 0.5rem'>Input Analysis</div>", unsafe_allow_html=True)
                st.image(processed_img, use_container_width=True, caption=f"Processed: {detected_type.title()}")
                
            with c2:
                # Quality Card
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Quality Assessment</div>
                    <div class="metric-value">{pred_class.replace('_', ' ').title()}</div>
                    <div class="status-badge {'status-good' if is_accepted else 'status-bad'}">
                        {f'Score: {good_conf*100:.1f}%'}
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                if is_accepted:
                    # Graft Analysis
                    if g_model:
                        gx = g_tfm(processed_img).unsqueeze(0).to(DEVICE)
                        with torch.no_grad():
                            gprobs = torch.softmax(g_model(gx), dim=1).cpu().numpy()[0]
                        
                        g_idx = int(np.argmax(gprobs))
                        g_class = g_classes[g_idx]
                        g_conf = float(gprobs[g_idx])
                        
                        is_graft = (g_class == "Graft")
                        
                        st.markdown(f"""
                        <div class="metric-card" style="border-left: 5px solid {'#EF4444' if is_graft else '#10B981'}">
                            <div class="metric-label">Graft Detection</div>
                            <div class="metric-value" style="color: {'#B91C1C' if is_graft else '#047857'}">
                                {g_class.upper()}
                            </div>
                            <div style="font-size: 1.1rem; color: #6B7280; margin-top: 0.5rem">
                                Confidence: <b>{g_conf*100:.1f}%</b>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.warning("Graft Model Unavailable")
                
                else:
                    st.error("Image Quality too low for Graft Analysis.")

        except Exception as e:
            st.error(f"Error: {str(e)}")

if __name__ == "__main__":
    main()
