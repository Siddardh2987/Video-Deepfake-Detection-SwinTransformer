# -*- coding: utf-8 -*-
"""
Configuration for Video Deepfake Detection using Swin Transformer.

Contains all inference-relevant settings, device selection, and path configuration.
Derived from the original Colab CONFIG dict.
"""

import os
import random
from pathlib import Path

import numpy as np
import torch

# ── Project Root ───────────────────────────────────────────────────────────────
# Resolve project root relative to this file: backend/config.py → project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ── Device ─────────────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Reproducibility ────────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if DEVICE.type == "cuda":
    torch.cuda.manual_seed_all(SEED)

# ── Configuration ──────────────────────────────────────────────────────────────
CONFIG = {
    # Paths (relative to project root)
    "model_path": str(PROJECT_ROOT / "models" / "best_model.pth"),

    # Frame extraction
    "max_frames_per_video": 30,     # hard cap before sampling
    "frame_sample_count":   16,     # frames to keep after sampling
    "face_size":            224,    # pixels (H=W fed to model)
    "min_face_confidence":  0.90,   # MTCNN threshold

    # Model
    "model_name":   "swin_base_patch4_window7_224",
    "num_classes":  2,              # REAL=0, FAKE=1
    "pretrained":   True,
    "dropout":      0.3,
}

# ── Label Mapping ──────────────────────────────────────────────────────────────
LABEL_NAMES = {0: "REAL", 1: "FAKE"}

# ── Training-specific CONFIG (used by training/ package) ──────────────────────
TRAINING_CONFIG = {
    # Paths
    "dataset_root":   str(PROJECT_ROOT / "data"),
    "checkpoint_dir": str(PROJECT_ROOT / "models" / "checkpoints"),
    "best_model_path": str(PROJECT_ROOT / "models" / "best_model.pth"),
    "log_path":       str(PROJECT_ROOT / "training_log.json"),

    # Dataset
    "val_split":   0.15,
    "test_split":  0.05,

    # Training
    "batch_size":          2,
    "accumulation_steps":  8,
    "num_epochs":          6,
    "lr":                  5e-5,
    "weight_decay":        0.05,
    "num_workers":         2,
    "save_every":          5,

    # Augmentation
    "use_augment": True,
}
