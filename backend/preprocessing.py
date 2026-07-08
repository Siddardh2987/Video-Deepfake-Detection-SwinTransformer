# -*- coding: utf-8 -*-
"""
Preprocessing pipeline for Video Deepfake Detection.

Contains:
  - MTCNN face detector (lazy-loaded singleton)
  - extract_frames(): Extract uniformly spaced frames from video
  - detect_and_crop_face(): Detect and crop the primary face from a frame
  - fallback_center_crop(): Center-crop fallback when no face is detected
  - extract_faces_from_video(): Full pipeline: frames → face crops
  - sample_frames(): Reduce face crops to target count
  - get_transforms(): Torchvision transforms for inference

Preserved from the original Colab implementation.
"""

import random
from typing import List, Optional

import cv2
import numpy as np
from PIL import Image
from torchvision import transforms
from facenet_pytorch import MTCNN

from backend.config import CONFIG, DEVICE


# ── MTCNN Singleton (lazy-loaded) ──────────────────────────────────────────────
_mtcnn_instance = None


def get_mtcnn() -> MTCNN:
    """
    Return a lazily-initialized MTCNN face detector.
    Avoids blocking Streamlit startup with model download/init.
    """
    global _mtcnn_instance
    if _mtcnn_instance is None:
        _mtcnn_instance = MTCNN(
            image_size=CONFIG["face_size"],
            margin=20,
            min_face_size=40,
            thresholds=[0.6, 0.7, CONFIG["min_face_confidence"]],
            device=DEVICE,
            keep_all=False,          # keep only the largest / most-confident face
            post_process=False,      # return uint8 tensor, not normalised float
        )
    return _mtcnn_instance


# ── Frame Extraction ───────────────────────────────────────────────────────────

def extract_frames(
    video_path: str,
    max_frames: int = CONFIG["max_frames_per_video"]
) -> List[np.ndarray]:
    """
    Extract up to `max_frames` frames (uniformly spaced) from a video file.

    Args:
        video_path : Path to the video file.
        max_frames : Maximum number of raw frames to extract before sampling.

    Returns:
        List of BGR numpy arrays (H x W x 3).
    """
    frames = []
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"[WARNING] Could not open video: {video_path}")
        return frames

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    total_frames = max(1, total_frames)  # guard divide-by-zero

    # Sample `max_frames` indices spread across the video
    indices = np.linspace(0, total_frames - 1, num=min(max_frames, total_frames), dtype=int)

    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if ret:
            frames.append(frame)

    cap.release()
    return frames


# ── Face Detection & Cropping ──────────────────────────────────────────────────

def detect_and_crop_face(
    frame_bgr: np.ndarray,
    output_size: int = CONFIG["face_size"]
) -> Optional[np.ndarray]:
    """
    Detect the primary face in a BGR frame and return it as an RGB numpy array.

    Returns:
        RGB uint8 array (output_size x output_size x 3), or None if no face found.
    """
    # Convert BGR → RGB PIL image (MTCNN expects RGB)
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    pil_img   = Image.fromarray(frame_rgb)

    try:
        face_tensor = get_mtcnn()(pil_img)           # Tensor [3, H, W] or None
    except Exception:
        return None

    if face_tensor is None:
        return None

    # Convert tensor [3, H, W] → numpy [H, W, 3]
    face_np = face_tensor.permute(1, 2, 0).cpu().numpy().astype(np.uint8)

    # Resize to target size (MTCNN already outputs image_size, but guard anyway)
    if face_np.shape[0] != output_size or face_np.shape[1] != output_size:
        face_np = cv2.resize(face_np, (output_size, output_size))

    return face_np


def fallback_center_crop(
    frame_bgr: np.ndarray,
    output_size: int = CONFIG["face_size"]
) -> np.ndarray:
    """
    When no face is detected, fall back to a center crop of the frame.
    Useful so the video is not silently discarded.
    """
    h, w = frame_bgr.shape[:2]
    side  = min(h, w)
    top   = (h - side) // 2
    left  = (w - side) // 2
    crop  = frame_bgr[top:top+side, left:left+side]
    crop  = cv2.resize(crop, (output_size, output_size))
    return cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)


# ── Video → Face Crops Pipeline ───────────────────────────────────────────────

def extract_faces_from_video(
    video_path: str,
    use_fallback: bool = True
) -> List[np.ndarray]:
    """
    Full pipeline: extract frames → detect faces → return list of RGB face crops.

    Args:
        video_path   : Path to video file.
        use_fallback : If True, use center-crop when no face found in a frame.

    Returns:
        List of RGB numpy arrays, one per accepted frame.
    """
    raw_frames = extract_frames(video_path)

    if len(raw_frames) == 0:
        print(f"[WARNING] No frames extracted from: {video_path}")
        return []

    face_crops = []
    no_face_count = 0

    for i, frame in enumerate(raw_frames):
        face = detect_and_crop_face(frame)
        if face is not None:
            face_crops.append(face)
        else:
            no_face_count += 1
            if use_fallback:
                face_crops.append(fallback_center_crop(frame))

    return face_crops


# ── Frame Sampling ─────────────────────────────────────────────────────────────

def sample_frames(
    face_crops: List[np.ndarray],
    n: int = CONFIG["frame_sample_count"],
    strategy: str = "uniform"
) -> List[np.ndarray]:
    """
    Reduce a list of face crops to exactly `n` samples.

    Strategies:
      "uniform"  — evenly spaced indices (default)
      "random"   — random subset
      "first"    — first N frames

    Args:
        face_crops : List of RGB numpy arrays.
        n          : Target number of frames.
        strategy   : Sampling strategy.

    Returns:
        List of up to `n` face crops.
    """
    total = len(face_crops)
    if total == 0:
        return []
    if total <= n:
        return face_crops  # already few enough

    if strategy == "uniform":
        indices = np.linspace(0, total - 1, num=n, dtype=int)
    elif strategy == "random":
        indices = sorted(random.sample(range(total), n))
    elif strategy == "first":
        indices = list(range(n))
    else:
        raise ValueError(f"Unknown sampling strategy: {strategy}")

    sampled = [face_crops[i] for i in indices]
    return sampled


# ── Transforms ─────────────────────────────────────────────────────────────────

def get_transforms(split: str = "train") -> transforms.Compose:
    """
    Return torchvision transforms for the given split.

    Args:
        split : "train" | "val" | "test"
    """
    mean = [0.485, 0.456, 0.406]   # ImageNet stats (Swin pretrained)
    std  = [0.229, 0.224, 0.225]

    if split == "train" and CONFIG.get("use_augment", False):
        return transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
            transforms.RandomResizedCrop(
                CONFIG["face_size"], scale=(0.85, 1.0), ratio=(0.95, 1.05)
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])
    else:
        return transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((CONFIG["face_size"], CONFIG["face_size"])),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])
