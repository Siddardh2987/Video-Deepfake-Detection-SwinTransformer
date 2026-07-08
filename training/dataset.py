# -*- coding: utf-8 -*-
"""
Dataset and DataLoader utilities for training.

Contains:
  - PreprocessedDeepfakeDataset: PyTorch Dataset for preprocessed .npy face crops
  - build_dataloaders(): Factory for train/val/test DataLoaders

Preserved from the original Colab implementation.
"""

import os
import random
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from backend.config import CONFIG, TRAINING_CONFIG
from backend.preprocessing import get_transforms


class PreprocessedDeepfakeDataset(Dataset):
    """
    PyTorch Dataset that loads preprocessed .npy face crops.

    Each .npy file contains a numpy array of shape (frame_sample_count, 224, 224, 3)
    representing face crops from a single video.
    """
    LABEL_MAP = {"Real": 0, "Fake": 1}

    def __init__(self, dataset_root, split="train", indices=None):
        self.split     = split
        self.transform = get_transforms(split)
        self.samples   = []

        for class_name, label in self.LABEL_MAP.items():
            class_dir = os.path.join(dataset_root, class_name)
            if not os.path.isdir(class_dir):
                continue
            for fname in sorted(os.listdir(class_dir)):
                if fname.endswith(".npy"):
                    self.samples.append((os.path.join(class_dir, fname), label))

        if indices is not None:
            self.samples = [self.samples[i] for i in indices]

        print(f"[PreprocessedDataset] split={split!r}  videos={len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        npy_path, label = self.samples[idx]
        faces = np.load(npy_path)               # (N, 224, 224, 3)  uint8

        tensors = [self.transform(faces[i]) for i in range(len(faces))]
        return torch.stack(tensors), label       # (N, 3, 224, 224)


def build_dataloaders(
    dataset_root: str = TRAINING_CONFIG["dataset_root"],
    batch_size:   int = TRAINING_CONFIG["batch_size"],
    num_workers:  int = TRAINING_CONFIG["num_workers"],
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Build train / val / test DataLoaders with a stratified(-ish) split.

    Returns:
        train_loader, val_loader, test_loader
    """
    # Build full dataset temporarily to get length
    full_ds = PreprocessedDeepfakeDataset(dataset_root, split="train")
    n       = len(full_ds)

    if n == 0:
        raise RuntimeError(f"No videos found under '{dataset_root}'. "
                           "Check your folder structure (Real/ & Fake/).")

    n_val  = max(1, int(n * TRAINING_CONFIG["val_split"]))
    n_test = max(1, int(n * TRAINING_CONFIG["test_split"]))
    n_train = n - n_val - n_test

    indices = list(range(n))
    random.shuffle(indices)

    train_idx = indices[:n_train]
    val_idx   = indices[n_train:n_train + n_val]
    test_idx  = indices[n_train + n_val:]

    train_ds = PreprocessedDeepfakeDataset(dataset_root, split="train",  indices=train_idx)
    val_ds   = PreprocessedDeepfakeDataset(dataset_root, split="val",    indices=val_idx)
    test_ds  = PreprocessedDeepfakeDataset(dataset_root, split="test",   indices=test_idx)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)

    print(f"[DataLoader] Train={len(train_ds)}  Val={len(val_ds)}  Test={len(test_ds)}")
    return train_loader, val_loader, test_loader
