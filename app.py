# -*- coding: utf-8 -*-
"""
AI Powered Video Deepfake Detection — Streamlit Frontend

This is the main entry point for the Streamlit web application.
Run with: streamlit run app.py
"""

import os
import tempfile
import time
from pathlib import Path

import streamlit as st

from backend.config import CONFIG, DEVICE, LABEL_NAMES
from backend.inference import load_model_for_inference, predict_video, predict_image


# ── Page Configuration ─────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Deepfake Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

    /* Global */
    .stApp {
        font-family: 'Inter', sans-serif;
    }

    /* Main header */
    .main-header {
        text-align: center;
        padding: 2rem 0 1rem 0;
    }
    .main-header h1 {
        font-size: 2.6rem;
        font-weight: 800;
        background: linear-gradient(135deg, #FD5901, #F78104, #FAAB36, #249EA0, #008083);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 0.3rem;
        letter-spacing: -0.5px;
    }
    .main-header p {
        color: #94a3b8;
        font-size: 1.05rem;
        font-weight: 400;
    }

    /* Upload zone */
    .upload-zone {
        border: 2px dashed rgba(247, 129, 4, 0.35);
        border-radius: 16px;
        padding: 2.5rem;
        text-align: center;
        background: linear-gradient(135deg, rgba(253, 89, 1, 0.04), rgba(36, 158, 160, 0.04));
        transition: all 0.3s ease;
        margin: 1rem 0;
    }
    .upload-zone:hover {
        border-color: rgba(247, 129, 4, 0.6);
        background: linear-gradient(135deg, rgba(253, 89, 1, 0.08), rgba(36, 158, 160, 0.08));
    }

    /* Result cards */
    .result-card {
        border-radius: 16px;
        padding: 1.2rem;
        margin: 0.3rem 0;
        backdrop-filter: blur(12px);
    }
    .result-real {
        background: linear-gradient(135deg, rgba(36, 158, 160, 0.1), rgba(0, 128, 131, 0.15));
        border: 1px solid rgba(36, 158, 160, 0.35);
    }
    .result-fake {
        background: linear-gradient(135deg, rgba(253, 89, 1, 0.1), rgba(247, 129, 4, 0.15));
        border: 1px solid rgba(253, 89, 1, 0.35);
    }
    .result-unknown {
        background: linear-gradient(135deg, rgba(250, 171, 54, 0.1), rgba(247, 129, 4, 0.12));
        border: 1px solid rgba(250, 171, 54, 0.35);
    }

    /* Label badges */
    .label-badge {
        display: inline-block;
        padding: 0.4rem 1.3rem;
        border-radius: 50px;
        font-weight: 700;
        font-size: 1.3rem;
        letter-spacing: 1px;
    }
    .badge-real {
        background: linear-gradient(135deg, #249EA0, #008083);
        color: white;
    }
    .badge-fake {
        background: linear-gradient(135deg, #FD5901, #F78104);
        color: white;
    }
    .badge-unknown {
        background: linear-gradient(135deg, #FAAB36, #F78104);
        color: white;
    }

    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, rgba(253, 89, 1, 0.04), rgba(36, 158, 160, 0.06));
        border: 1px solid rgba(247, 129, 4, 0.15);
        border-radius: 12px;
        padding: 1rem;
        text-align: center;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(247, 129, 4, 0.12);
    }
    .metric-value {
        font-size: 1.6rem;
        font-weight: 700;
        color: #F78104;
    }
    .metric-label {
        font-size: 0.8rem;
        color: #94a3b8;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-top: 0.3rem;
    }

    /* Sidebar styling */
    .sidebar-info {
        background: linear-gradient(135deg, rgba(253, 89, 1, 0.05), rgba(36, 158, 160, 0.08));
        border: 1px solid rgba(247, 129, 4, 0.15);
        border-radius: 12px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    .sidebar-info h4 {
        color: #F78104;
        margin-bottom: 0.5rem;
        font-weight: 600;
    }

    /* Probability bar */
    .prob-container {
        margin: 0.6rem 0;
    }
    .prob-bar-bg {
        background: rgba(148, 163, 184, 0.15);
        border-radius: 10px;
        height: 12px;
        overflow: hidden;
        margin-top: 0.3rem;
    }
    .prob-bar-real {
        background: linear-gradient(90deg, #249EA0, #008083);
        height: 100%;
        border-radius: 10px;
        transition: width 0.8s ease;
    }
    .prob-bar-fake {
        background: linear-gradient(90deg, #FD5901, #F78104);
        height: 100%;
        border-radius: 10px;
        transition: width 0.8s ease;
    }

    /* File info card */
    .file-info {
        background: linear-gradient(135deg, rgba(253, 89, 1, 0.04), rgba(36, 158, 160, 0.05));
        border: 1px solid rgba(247, 129, 4, 0.12);
        border-radius: 12px;
        padding: 1rem 1.2rem;
        margin: 0.8rem 0;
    }

    /* Divider */
    .styled-divider {
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(247, 129, 4, 0.3), transparent);
        margin: 1.5rem 0;
    }

    /* Compact metrics grid for side-panel */
    .metrics-grid {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 0.5rem;
        margin: 0.5rem 0;
    }

    /* Hide default Streamlit footer */
    footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ── Model Loading (cached) ────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_detection_model():
    """Load the deepfake detection model (cached across reruns)."""
    model_path = CONFIG["model_path"]
    if not os.path.isfile(model_path):
        from backend.inference import reconstruct_model_from_parts
        if not reconstruct_model_from_parts(model_path):
            return None, model_path
    return load_model_for_inference(model_path), model_path


# ── Helper Functions ───────────────────────────────────────────────────────────
def get_file_size_str(size_bytes: int) -> str:
    """Convert bytes to human-readable file size."""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    else:
        return f"{size_bytes / (1024 * 1024):.1f} MB"


def is_video_file(filename: str) -> bool:
    """Check if a file is a video based on extension."""
    return filename.lower().endswith((".mp4", ".avi", ".mov"))


def is_image_file(filename: str) -> bool:
    """Check if a file is an image based on extension."""
    return filename.lower().endswith((".jpg", ".jpeg", ".png"))


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🛡️ Detection Settings")
    st.markdown('<div class="styled-divider"></div>', unsafe_allow_html=True)

    # Model info
    st.markdown('<div class="sidebar-info">', unsafe_allow_html=True)
    st.markdown("#### 🧠 Model Information")
    st.markdown(f"""
    - **Architecture:** Swin Transformer
    - **Backbone:** `{CONFIG['model_name']}`
    - **Input Size:** {CONFIG['face_size']}×{CONFIG['face_size']}
    - **Classes:** Real / Fake
    - **Device:** `{DEVICE}`
    """)
    st.markdown('</div>', unsafe_allow_html=True)

    # Pipeline info
    st.markdown('<div class="sidebar-info">', unsafe_allow_html=True)
    st.markdown("#### ⚙️ Pipeline Settings")
    st.markdown(f"""
    - **Max Frames:** {CONFIG['max_frames_per_video']}
    - **Sample Count:** {CONFIG['frame_sample_count']}
    - **Face Confidence:** {CONFIG['min_face_confidence']}
    - **Face Detection:** MTCNN
    """)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="styled-divider"></div>', unsafe_allow_html=True)

    # Model status
    model, model_path = load_detection_model()
    if model is not None:
        st.success("✅ Model loaded successfully")
    else:
        st.error(f"❌ Model not found")
        st.caption(f"Expected at: `{model_path}`")

    st.markdown('<div class="styled-divider"></div>', unsafe_allow_html=True)

    st.markdown("""
    <div style="text-align: center; color: #64748b; font-size: 0.8rem; margin-top: 1rem;">
        Built with Swin Transformer & MTCNN<br/>
        Powered by PyTorch & Streamlit
    </div>
    """, unsafe_allow_html=True)


# ── Main Content ───────────────────────────────────────────────────────────────

# Header
st.markdown("""
<div class="main-header">
    <h1>🛡️ AI Powered Video Deepfake Detection</h1>
    <p>Upload a video or image to detect deepfakes using Swin Transformer</p>
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="styled-divider"></div>', unsafe_allow_html=True)

# CSS to style the native file uploader beautifully
st.markdown("""
<style>
    /* Style the file uploader drop zone to look exactly like a custom upload area */
    [data-testid="stFileUploader"] {
        border: 2px dashed rgba(247, 129, 4, 0.45) !important;
        border-radius: 16px !important;
        padding: 2rem !important;
        background: linear-gradient(135deg, rgba(253, 89, 1, 0.04), rgba(36, 158, 160, 0.04)) !important;
        transition: all 0.3s ease !important;
    }
    [data-testid="stFileUploader"] section {
        background-color: transparent !important;
        padding: 0 !important;
        display: flex !important;
        flex-direction: column !important;
        align-items: center !important;
        justify-content: center !important;
        gap: 12px !important;
    }
    [data-testid="stFileUploader"] section > input + div {
        display: flex !important;
        flex-direction: column !important;
        align-items: center !important;
        justify-content: center !important;
        text-align: center !important;
    }
    [data-testid="stFileUploader"]:hover {
        border-color: rgba(247, 129, 4, 0.8) !important;
        background: linear-gradient(135deg, rgba(253, 89, 1, 0.07), rgba(36, 158, 160, 0.07)) !important;
    }
    [data-testid="stFileUploader"] label {
        display: none !important;
    }
    /* Style button inside uploader and center it */
    [data-testid="stFileUploader"] button {
        background-color: #F78104 !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 0.5rem 1.5rem !important;
        font-weight: 600 !important;
        box-shadow: 0 4px 12px rgba(247, 129, 4, 0.2) !important;
        transition: background-color 0.2s ease !important;
        margin: 0 auto !important;
        display: block !important;
    }
    [data-testid="stFileUploader"] button:hover {
        background-color: #FD5901 !important;
    }
</style>
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "Upload a video or image",
    type=["mp4", "avi", "mov", "jpg", "jpeg"],
    label_visibility="collapsed",
    key="file_uploader",
)

if uploaded_file is not None:
    file_name = uploaded_file.name
    file_size = uploaded_file.size
    file_type = "Video" if is_video_file(file_name) else "Image"

    # Reset results if user uploads a new/different file
    if "current_file" not in st.session_state or st.session_state["current_file"] != file_name:
        st.session_state["current_file"] = file_name
        if "result" in st.session_state:
            del st.session_state["result"]

    # File info
    st.markdown(f"""
    <div class="file-info">
        <span style="font-weight: 600; color: #F78104;">📄 {file_name}</span>
        <span style="color: #94a3b8; margin-left: 1rem;">
            {get_file_size_str(file_size)} • {file_type}
        </span>
    </div>
    """, unsafe_allow_html=True)

    # Pre-analysis view (no video display before clicking Analyze to avoid empty space layout)
    if "result" not in st.session_state:
        st.markdown('<div class="styled-divider"></div>', unsafe_allow_html=True)

        # Check model is loaded
        if model is None:
            st.error(
                f"⚠️ **Model file not found.** "
                f"Please place your trained model at:\n\n`{model_path}`"
            )
            st.stop()

        # Analyze button (only show when results aren't generated yet)
        analyze_clicked = st.button(
            "🔍  Analyze for Deepfake",
            use_container_width=True,
            type="primary",
        )
    else:
        analyze_clicked = False

    if analyze_clicked:
        with st.spinner("🔄 Processing... Extracting faces and running inference..."):
            try:
                # Save uploaded file to temp location
                suffix = Path(file_name).suffix
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                    tmp.write(uploaded_file.getbuffer())
                    tmp_path = tmp.name

                # Run inference
                if is_video_file(file_name):
                    result = predict_video(tmp_path, model)
                elif is_image_file(file_name):
                    result = predict_image(tmp_path, model)
                else:
                    st.error("Unsupported file format.")
                    st.stop()

                # Cleanup temp file
                os.unlink(tmp_path)

                # Store result in session state
                st.session_state["result"] = result

            except Exception as e:
                st.error(f"❌ **Inference failed:** {str(e)}")
                st.stop()

    # ── Display Results (side-by-side with preview) ────────────────────
    if "result" in st.session_state:
        result = st.session_state["result"]
        label = result["label"]
        confidence = result["confidence"]
        probs = result["probabilities"]
        num_frames = result["num_frames"]
        latency = result["latency_seconds"]

        # Result styling
        if label == "REAL":
            card_class = "result-real"
            badge_class = "badge-real"
            result_icon = "✅"
        elif label == "FAKE":
            card_class = "result-fake"
            badge_class = "badge-fake"
            result_icon = "🚨"
        else:
            card_class = "result-unknown"
            badge_class = "badge-unknown"
            result_icon = "⚠️"

        # Side-by-side layout: preview (left, smaller) | results (right)
        col_media, col_results = st.columns([2, 3])

        with col_media:
            if is_video_file(file_name):
                st.video(uploaded_file)
            elif is_image_file(file_name):
                st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)

        with col_results:
            # Result card
            st.markdown(f"""
            <div class="result-card {card_class}" style="text-align: center;">
                <p style="font-size: 0.9rem; color: #94a3b8; font-weight: 500; margin-bottom: 0.5rem;">
                    DETECTION RESULT
                </p>
                <span class="label-badge {badge_class}">
                    {result_icon} {label}
                </span>
                <p style="font-size: 1rem; font-weight: 600; margin-top: 0.7rem;">
                    {confidence}% Confidence
                </p>
            </div>
            """, unsafe_allow_html=True)

            # Metrics as 2x2 grid
            st.markdown(f"""
            <div class="metrics-grid">
                <div class="metric-card">
                    <div class="metric-value">{confidence}%</div>
                    <div class="metric-label">Confidence</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{num_frames}</div>
                    <div class="metric-label">Frames Analyzed</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{latency}s</div>
                    <div class="metric-label">Processing Time</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{'GPU' if DEVICE.type == 'cuda' else 'CPU'}</div>
                    <div class="metric-label">Device Used</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Probability bars
            st.markdown(f"""
            <div class="prob-container">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <span style="font-weight: 600; color: #249EA0;">✅ Real</span>
                    <span style="font-weight: 700; color: #249EA0;">{probs['REAL']}%</span>
                </div>
                <div class="prob-bar-bg">
                    <div class="prob-bar-real" style="width: {probs['REAL']}%;"></div>
                </div>
            </div>
            <div class="prob-container">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <span style="font-weight: 600; color: #FD5901;">🚨 Fake</span>
                    <span style="font-weight: 700; color: #FD5901;">{probs['FAKE']}%</span>
                </div>
                <div class="prob-bar-bg">
                    <div class="prob-bar-fake" style="width: {probs['FAKE']}%;"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)


else:
    # No file uploaded — show instructions
    st.markdown("")
    col_inst1, col_inst2, col_inst3 = st.columns(3)

    with col_inst1:
        st.markdown("""
        <div class="metric-card">
            <div style="font-size: 2rem; margin-bottom: 0.5rem;">📤</div>
            <div style="font-weight: 600; margin-bottom: 0.3rem;">Step 1</div>
            <div style="color: #94a3b8; font-size: 0.85rem;">Upload a video or image file</div>
        </div>
        """, unsafe_allow_html=True)

    with col_inst2:
        st.markdown("""
        <div class="metric-card">
            <div style="font-size: 2rem; margin-bottom: 0.5rem;">🔍</div>
            <div style="font-weight: 600; margin-bottom: 0.3rem;">Step 2</div>
            <div style="color: #94a3b8; font-size: 0.85rem;">Click Analyze to start detection</div>
        </div>
        """, unsafe_allow_html=True)

    with col_inst3:
        st.markdown("""
        <div class="metric-card">
            <div style="font-size: 2rem; margin-bottom: 0.5rem;">📊</div>
            <div style="font-weight: 600; margin-bottom: 0.3rem;">Step 3</div>
            <div style="color: #94a3b8; font-size: 0.85rem;">Get results with confidence score</div>
        </div>
        """, unsafe_allow_html=True)
