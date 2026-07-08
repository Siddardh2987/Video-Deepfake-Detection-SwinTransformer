# -*- coding: utf-8 -*-
"""
Checkpoint and data preprocessing utilities for training.

Contains:
  - save_checkpoint(): Save model + optimizer state
  - preprocess_and_save_all(): Bulk preprocess videos → .npy face crops

Preserved from the original Colab implementation.
"""

import os
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from backend.config import CONFIG, TRAINING_CONFIG
from backend.preprocessing import extract_faces_from_video, sample_frames


def save_checkpoint(
    model:     nn.Module,
    optimizer: optim.Optimizer,
    epoch:     int,
    val_acc:   float,
    path:      str,
) -> None:
    """Save training checkpoint."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "epoch":                epoch,
        "model_state_dict":     model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "val_acc":              val_acc,
    }, path)
    print(f"  [Checkpoint] Saved → {path}")


def preprocess_and_save_all(
    dataset_root: str = TRAINING_CONFIG["dataset_root"],
    output_root:  str = None,
) -> None:
    """
    Preprocess all videos in the dataset: extract faces and save as .npy files.

    Args:
        dataset_root : Root directory containing Real/ and Fake/ subdirs with videos.
        output_root  : Where to save .npy files. Defaults to dataset_root + '_faces'.
    """
    if output_root is None:
        output_root = dataset_root + "_faces"

    for label in ["Real", "Fake"]:
        src_dir = os.path.join(dataset_root, label)
        out_dir = os.path.join(output_root, label)
        os.makedirs(out_dir, exist_ok=True)

        if not os.path.isdir(src_dir):
            print(f"[Preprocess] WARNING: {src_dir} not found, skipping.")
            continue

        video_files = [
            f for f in os.listdir(src_dir)
            if f.lower().endswith((".mp4", ".avi", ".mov", ".mkv"))
        ]

        print(f"\n[Preprocess] {label.upper()} — {len(video_files)} videos")

        for i, fname in enumerate(video_files):
            video_path = os.path.join(src_dir, fname)
            out_path   = os.path.join(out_dir, fname.rsplit(".", 1)[0] + ".npy")

            if os.path.exists(out_path):
                print(f"  [{i+1}/{len(video_files)}] SKIP (already done): {fname}")
                continue

            faces = extract_faces_from_video(video_path, use_fallback=True)
            faces = sample_frames(faces, n=CONFIG["frame_sample_count"])

            if len(faces) == 0:
                print(f"  [{i+1}/{len(video_files)}] WARNING: no faces → skipping {fname}")
                continue

            # Pad if needed
            while len(faces) < CONFIG["frame_sample_count"]:
                faces.append(faces[-1])

            np.save(out_path, np.array(faces, dtype=np.uint8))
            print(f"  [{i+1}/{len(video_files)}] Done: {fname}")

    print("\n[Preprocess] All videos preprocessed!")
