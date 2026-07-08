# 🛡️ AI Powered Video Deepfake Detection

A production-ready deepfake detection system using **Swin Transformer** and **MTCNN** face detection, deployed as a **Streamlit** web application.

Upload a video (`.mp4`, `.avi`, `.mov`) or image (`.jpg`, `.jpeg`) and get real-time deepfake classification with confidence scores.

---

## 🏗️ Architecture

```
Upload File → Frame Extraction → MTCNN Face Detection → Frame Sampling (16 frames)
     → Swin Transformer Backbone → Temporal Mean Pooling → Classification Head
     → REAL / FAKE Prediction with Confidence Score
```

**Model Pipeline:**
- **Face Detection**: MTCNN extracts and crops faces from video frames.
- **Backbone**: Swin Transformer (`swin_base_patch4_window7_224`) pretrained on ImageNet.
- **Temporal Aggregation**: Mean pooling across frame-level features.
- **Classifier**: Linear layers (1024 → 512 → 2) with dropout.

---

## 📁 Folder Structure

```
Video-Deepfake-Detection-SwinTransformer/
│
├── app.py                         # Streamlit frontend (entry point)
│
├── backend/                       # Inference pipeline
│   ├── __init__.py
│   ├── config.py                  # Configuration & device setup
│   ├── model.py                   # SwinDeepfakeDetector architecture
│   ├── preprocessing.py           # Frame extraction, face detection, transforms
│   └── inference.py               # Model loading & prediction functions
│
├── training/                      # Training utilities (preserved from Colab)
│   ├── __init__.py
│   ├── dataset.py                 # PreprocessedDeepfakeDataset & DataLoaders
│   ├── trainer.py                 # Training loop, evaluation
│   └── checkpoint.py              # Checkpoint saving & data preprocessing
│
├── models/                        # Trained model weights & split parts
│   ├── Final_model_part_aa        # Split model part aa
│   ├── Final_model_part_ab        # Split model part ab
│   ├── Final_model_part_ac        # Split model part ac
│   ├── Final_model_part_ad        # Split model part ad
│   └── best_model.pth             # ← Reconstructed/placed model file
│
├── requirements.txt               # Python dependencies
├── README.md                      # This file
├── Actual_video_deepfake_full_code.py # Latest Colab export (reference)
└── video_model_full_code.py       # Original Colab export (reference)
```

---

## 🚀 Model Handling & Git Compatibility

GitHub has a 100MB file size limit. Since the trained model checkpoint (`best_model.pth`) is **349 MB**, it has been split into smaller files under `models/`.

### ⚡ Automatic Reconstruction (Self-Healing)
**No manual action is required!** On startup, the application will automatically detect the split parts (`models/Final_model_part_a*`) and reconstruct the `best_model.pth` file for you if it is missing.

### 🛠️ Manual Merging / Splitting (Optional)
If you wish to merge or split the files manually:

#### To merge the split parts back to the original model:
- **Windows (PowerShell)**:
  ```powershell
  Get-Content models/Final_model_part_* | Set-Content models/best_model.pth -Encoding Byte
  ```
- **Linux/Mac**:
  ```bash
  cat models/Final_model_part_* > models/best_model.pth
  ```

#### To split a new model file for GitHub upload:
- **Linux/Mac/Git Bash** (splits into 100MB chunks):
  ```bash
  split -b 100M models/best_model.pth models/Final_model_part_
  ```

---

## 🚀 Installation & Setup

### Prerequisites
- Python 3.9+
- pip

### Step 1: Clone the repository
```bash
git clone https://github.com/your-username/Video-Deepfake-Detection-SwinTransformer.git
cd Video-Deepfake-Detection-SwinTransformer
```

### Step 2: Create a virtual environment
```bash
python -m venv .venv
```

Activate it:
- **Windows (PowerShell)**:
  ```powershell
  Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
  .\.venv\Scripts\Activate.ps1
  ```
- **Linux/Mac**:
  ```bash
  source .venv/bin/activate
  ```

### Step 3: Install dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Run the application
```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`. If `models/best_model.pth` is missing, it will automatically reconstruct it from the split parts on startup.

---

## 🎯 Usage

1. Open the app in your browser.
2. Drag and drop or browse to upload a video file (`.mp4`, `.avi`, `.mov`) or image (`.jpg`, `.jpeg`).
3. Click **"🔍 Analyze for Deepfake"**.
4. View results:
   - **Prediction**: REAL or FAKE (color-coded badge).
   - **Confidence**: Percentage score.
   - **Probability breakdown**: REAL vs FAKE probability bars.
   - **Processing time**: Inference latency.
   - **Frames analyzed**: Number of frames used (16 frames).

---

## 🧠 Model Details

| Parameter | Value |
|---|---|
| Backbone | `swin_base_patch4_window7_224` |
| Input Size | 224 × 224 |
| Frame Count | 16 frames per video |
| Face Detector | MTCNN |
| Classes | REAL (0), FAKE (1) |
| Dropout | 0.3 |

---

## 📝 Notes

- The app automatically uses **GPU** if available, otherwise falls back to **CPU**.
- MTCNN face detection is lazy-loaded on first inference to speed up app startup.
- If no face is detected in a frame, a center-crop fallback is used.
- The model is cached after first load — subsequent predictions are faster.
- Training functions are preserved in the `training/` folder for retraining purposes.

