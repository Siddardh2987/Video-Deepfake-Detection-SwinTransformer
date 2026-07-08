# -*- coding: utf-8 -*-
"""
Inference pipeline for Video Deepfake Detection.

Contains:
  - load_checkpoint(): Load model weights from a .pth file
  - load_model_for_inference(): Build model + load weights + set eval mode
  - predict_video(): End-to-end video inference
  - predict_image(): Image inference using the same multi-frame pipeline

Preserved from the original Colab implementation.
"""

import os
import time
from typing import Dict, Tuple, Optional

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image

from backend.config import CONFIG, DEVICE, LABEL_NAMES
from backend.model import SwinDeepfakeDetector, build_model
from backend.preprocessing import (
    extract_faces_from_video,
    sample_frames,
    get_transforms,
    detect_and_crop_face,
    fallback_center_crop,
)


def load_checkpoint(
    model: nn.Module,
    optimizer: Optional[optim.Optimizer],
    path: str,
) -> Tuple[nn.Module, Optional[optim.Optimizer], int, float]:
    """
    Load a checkpoint.  Returns (model, optimizer, start_epoch, best_val_acc).
    Pass optimizer=None for inference-only loading.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    ckpt = torch.load(path, map_location=DEVICE, weights_only=False)

    # Robust loading supporting varying checkpoint formats
    if isinstance(ckpt, dict):
        if "model_state_dict" in ckpt:
            model.load_state_dict(ckpt["model_state_dict"])
        elif "state_dict" in ckpt:
            model.load_state_dict(ckpt["state_dict"])
        elif "model" in ckpt:
            model.load_state_dict(ckpt["model"])
        else:
            model.load_state_dict(ckpt)
    else:
        model.load_state_dict(ckpt)

    start_epoch = 0
    best_val_acc = 0.0

    if isinstance(ckpt, dict):
        if optimizer is not None and "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch  = ckpt.get("epoch", 0) + 1
        best_val_acc = ckpt.get("val_acc", 0.0)

    print(f"[Checkpoint] Loaded from '{path}'  epoch={start_epoch-1}  val_acc={best_val_acc:.4f}")
    return model, optimizer, start_epoch, best_val_acc


def reconstruct_model_from_parts(model_path: str) -> bool:
    """
    If the reconstructed model file is missing but split parts exist,
    reconstruct the model file automatically.
    """
    from pathlib import Path
    dest_path = Path(model_path)
    if dest_path.exists():
        return True

    models_dir = dest_path.parent
    parts = sorted(list(models_dir.glob("Final_model_part_*")))
    if not parts:
        return False

    print(f"[Inference] Reconstructing {dest_path.name} from {len(parts)} parts...")
    try:
        with open(dest_path, "wb") as dest_file:
            for part in parts:
                with open(part, "rb") as part_file:
                    dest_file.write(part_file.read())
        print(f"[Inference] Reconstruction successful!")
        return True
    except Exception as e:
        print(f"[Inference] Error reconstructing model: {e}")
        if dest_path.exists():
            dest_path.unlink()
        return False


def load_model_for_inference(
    checkpoint_path: str = CONFIG["model_path"]
) -> nn.Module:
    """
    Load model weights for inference.

    Args:
        checkpoint_path : Path to .pth checkpoint.

    Returns:
        model in eval() mode.
    """
    reconstruct_model_from_parts(checkpoint_path)
    model = build_model()
    model, _, _, _ = load_checkpoint(model, None, checkpoint_path)
    model.eval()
    print(f"[Inference] Model loaded from '{checkpoint_path}'")
    return model


@torch.no_grad()
def predict_video(
    video_path: str,
    model: nn.Module,
    frame_sample_count: int = CONFIG["frame_sample_count"],
) -> Dict:
    """
    Run end-to-end inference on a single video.

    Args:
        video_path         : Path to input video file.
        model              : Loaded SwinDeepfakeDetector in eval mode.
        frame_sample_count : Number of frames to sample for prediction.

    Returns:
        dict with keys: label, confidence, probabilities, num_frames, latency_seconds
    """
    transform = get_transforms(split="val")
    start_time = time.time()

    # Step 1: Extract faces
    face_crops = extract_faces_from_video(video_path, use_fallback=True)

    # Step 2: Sample
    face_crops = sample_frames(face_crops, n=frame_sample_count)

    # Step 3: Handle edge case
    if len(face_crops) == 0:
        return {
            "label": "UNKNOWN",
            "confidence": 0.0,
            "probabilities": {"REAL": 50.0, "FAKE": 50.0},
            "num_frames": 0,
            "latency_seconds": round(time.time() - start_time, 3),
        }

    num_frames_analyzed = len(face_crops)

    # Pad if needed
    while len(face_crops) < frame_sample_count:
        face_crops.append(face_crops[-1])

    # Step 4: Transform & batch
    tensors = [transform(f) for f in face_crops]     # each (3, H, W)
    frames  = torch.stack(tensors).unsqueeze(0)       # (1, T, 3, H, W)
    frames  = frames.to(DEVICE)

    # Step 5: Forward pass
    logits = model(frames)                             # (1, 2)
    probs  = torch.softmax(logits, dim=1)[0]           # (2,)
    pred   = probs.argmax().item()
    conf   = probs[pred].item()

    latency = round(time.time() - start_time, 3)

    result = {
        "label":        LABEL_NAMES[pred],
        "confidence":   round(conf * 100, 2),
        "probabilities": {
            "REAL": round(probs[0].item() * 100, 2),
            "FAKE": round(probs[1].item() * 100, 2),
        },
        "num_frames":      num_frames_analyzed,
        "latency_seconds": latency,
    }

    return result


@torch.no_grad()
def predict_image(
    image_path: str,
    model: nn.Module,
    frame_sample_count: int = CONFIG["frame_sample_count"],
) -> Dict:
    """
    Run inference on a single image using the same multi-frame pipeline.

    The image is treated as a single BGR frame → face detection → padded to
    `frame_sample_count` via the same padding logic used for videos.

    No architectural changes — uses the exact same model forward pass.

    Args:
        image_path         : Path to input image file.
        model              : Loaded SwinDeepfakeDetector in eval mode.
        frame_sample_count : Number of frames to pad to for prediction.

    Returns:
        dict with keys: label, confidence, probabilities, num_frames, latency_seconds
    """
    transform = get_transforms(split="val")
    start_time = time.time()

    # Read image as BGR (same as video frames)
    frame_bgr = cv2.imread(image_path)
    if frame_bgr is None:
        return {
            "label": "UNKNOWN",
            "confidence": 0.0,
            "probabilities": {"REAL": 50.0, "FAKE": 50.0},
            "num_frames": 0,
            "latency_seconds": round(time.time() - start_time, 3),
        }

    # Detect face (same as video pipeline)
    face = detect_and_crop_face(frame_bgr)
    if face is None:
        face = fallback_center_crop(frame_bgr)

    face_crops = [face]

    # Pad to frame_sample_count using the same padding logic as predict_video
    while len(face_crops) < frame_sample_count:
        face_crops.append(face_crops[-1])

    # Transform & batch (identical to predict_video)
    tensors = [transform(f) for f in face_crops]     # each (3, H, W)
    frames  = torch.stack(tensors).unsqueeze(0)       # (1, T, 3, H, W)
    frames  = frames.to(DEVICE)

    # Forward pass (identical to predict_video)
    logits = model(frames)                             # (1, 2)
    probs  = torch.softmax(logits, dim=1)[0]           # (2,)
    pred   = probs.argmax().item()
    conf   = probs[pred].item()

    latency = round(time.time() - start_time, 3)

    result = {
        "label":        LABEL_NAMES[pred],
        "confidence":   round(conf * 100, 2),
        "probabilities": {
            "REAL": round(probs[0].item() * 100, 2),
            "FAKE": round(probs[1].item() * 100, 2),
        },
        "num_frames":      1,
        "latency_seconds": latency,
    }

    return result
