# Eye Analysis & Graft Detection System

This package provides a comprehensive set of tools for Eye Image Quality Assessment, ROI Cropping, and Graft Detection.

## 📂 Directory Structure

```
Eye_Analysis_Release/
├── App/
│   └── Eye_Analysis_UI.py          # Main Streamlit Application
├── Inference/
│   └── Batch_Processor.py          # Script for batch processing folders
├── Training/
│   ├── Train_Quality_Model.py      # Retrain Quality Gatekeeper Model
│   ├── Train_Graft_Model.py        # Retrain Graft Detection Model
│   └── Segmentation/               # Retrain Segmentation Models
│       ├── Train_Standard_Seg.py
│       └── Train_SlitLamp_Seg.py
├── Models/                         # Contains all trained models
└── requirements.txt                # Python dependencies
```

## 🚀 Installation

1.  **Dependencies**: Install required libraries.
    ```bash
    pip install -r requirements.txt
    ```

## 🖥️ Usage

### 1. Integrated Eye App (GUI)
The easiest way to use the system. It combines Standard/Slitlamp detection, Auto-Cropping, Quality Checking, and Graft Detection.

```bash
cd App
streamlit run Eye_Analysis_UI.py
```

### 2. Batch Processing
Process an entire folder of images automatically.

1.  Open `Inference/Batch_Processor.py`.
2.  Update `INPUT_DIR` to point to your image folder.
3.  Run the script:
    ```bash
    cd Inference
    python Batch_Processor.py
    ```
4.  Results will be saved in `classified/` subfolders (`Accepted`, `Rejected`, `Graft`, `No_Graft`).

## 🧠 Training (Optional)

If you have new data and want to improve the models:

- **Quality Model**: Run `Training/Train_Quality_Model.py`.
- **Graft Model**: Run `Training/Train_Graft_Model.py`.
- **Segmentation**: Use scripts in `Training/Segmentation/`.

All training scripts are configured to use relative paths, but you may need to point them to your raw data folders.

## 📦 Models

- **Standard/Slitlamp**: Detects image type and segment ROI.
- **Quality**: EfficientNet-B0 classifier (Good/Bad/Non-Eye).
- **Graft**: EfficientNet-B0 classifier (Graft/No-Graft).
