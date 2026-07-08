# -*- coding: utf-8 -*-
"""
Training loop and evaluation for Video Deepfake Detection.

Contains:
  - train_one_epoch(): Single training epoch
  - evaluate(): Evaluate on any DataLoader
  - train(): Full training loop with checkpointing
  - evaluate_on_test_set(): Load best model and evaluate on test split

Preserved from the original Colab implementation.
"""

import json
import os
import time
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from backend.config import CONFIG, DEVICE, TRAINING_CONFIG
from backend.model import build_model
from backend.inference import load_checkpoint
from training.checkpoint import save_checkpoint
from training.dataset import build_dataloaders


def train_one_epoch(
    model:     nn.Module,
    loader:    DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    epoch:     int,
) -> Tuple[float, float]:
    """
    One full pass over the training set with gradient accumulation.

    Returns:
        avg_loss, accuracy (both float)
    """
    model.train()
    total_loss = 0.0
    correct    = 0
    total      = 0

    accum_steps = TRAINING_CONFIG.get("accumulation_steps", 8)
    optimizer.zero_grad()

    pbar = tqdm(loader, desc=f"Epoch {epoch+1} [Train]", leave=False)
    for batch_idx, (frames, labels) in enumerate(pbar):
        frames = frames.to(DEVICE, non_blocking=True)    # (B, T, 3, H, W)
        labels = labels.to(DEVICE, non_blocking=True)    # (B,)

        logits = model(frames)               # (B, num_classes)
        loss   = criterion(logits, labels)

        loss = loss / accum_steps
        loss.backward()

        if (batch_idx + 1) % accum_steps == 0 or (batch_idx + 1) == len(loader):
            optimizer.step()
            optimizer.zero_grad()

        # Metrics
        preds   = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total   += labels.size(0)
        total_loss += (loss.item() * accum_steps)

        pbar.set_postfix({
            "loss": f"{(total_loss/(batch_idx+1)):.4f}",
            "acc":  f"{(correct/total if total > 0 else 0.0):.4f}",
        })

    avg_loss = total_loss / len(loader)
    accuracy = correct / total if total > 0 else 0.0
    return avg_loss, accuracy


@torch.no_grad()
def evaluate(
    model:     nn.Module,
    loader:    DataLoader,
    criterion: nn.Module,
    split:     str = "Val",
) -> Tuple[float, float]:
    """
    Evaluate on any DataLoader.

    Returns:
        avg_loss, accuracy
    """
    model.eval()
    total_loss = 0.0
    correct    = 0
    total      = 0

    pbar = tqdm(loader, desc=f"  [{split}]", leave=False)
    for frames, labels in pbar:
        frames = frames.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)

        logits = model(frames)
        loss   = criterion(logits, labels)

        preds   = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total   += labels.size(0)
        total_loss += loss.item()

    avg_loss = total_loss / len(loader)
    accuracy = correct / total if total > 0 else 0.0
    return avg_loss, accuracy


def train(
    resume_from: Optional[str] = None,
) -> Dict:
    """
    Full training loop.

    Args:
        resume_from : Path to checkpoint to resume from (or None).

    Returns:
        Dictionary with training history.
    """
    print("\n" + "=" * 70)
    print("  TRAINING START")
    print("=" * 70)

    # ── Build components ──────────────────────────────────────────────────────
    model     = build_model()
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=TRAINING_CONFIG["lr"],
        weight_decay=TRAINING_CONFIG["weight_decay"],
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=TRAINING_CONFIG["num_epochs"], eta_min=1e-6
    )

    train_loader, val_loader, _ = build_dataloaders()

    start_epoch  = 0
    best_val_acc = 0.0
    history      = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    # ── Resume from checkpoint ────────────────────────────────────────────────
    if resume_from and os.path.isfile(resume_from):
        model, optimizer, start_epoch, best_val_acc = load_checkpoint(
            model, optimizer, resume_from
        )

    # ── Ensure checkpoint directory exists ─────────────────────────────────────
    os.makedirs(TRAINING_CONFIG["checkpoint_dir"], exist_ok=True)

    # ── Epoch loop ────────────────────────────────────────────────────────────
    for epoch in range(start_epoch, TRAINING_CONFIG["num_epochs"]):
        t0 = time.time()

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, epoch
        )
        val_loss, val_acc = evaluate(model, val_loader, criterion, split="Val")
        scheduler.step()

        elapsed = time.time() - t0

        print(
            f"Epoch [{epoch+1:03d}/{TRAINING_CONFIG['num_epochs']}]  "
            f"Train Loss={train_loss:.4f}  Train Acc={train_acc:.4f}  "
            f"Val Loss={val_loss:.4f}  Val Acc={val_acc:.4f}  "
            f"LR={scheduler.get_last_lr()[0]:.2e}  "
            f"Time={elapsed:.1f}s"
        )

        # Accumulate history
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        # ── Save periodic checkpoint ──────────────────────────────────────────
        if (epoch + 1) % TRAINING_CONFIG["save_every"] == 0:
            ckpt_path = os.path.join(
                TRAINING_CONFIG["checkpoint_dir"],
                f"checkpoint_epoch{epoch+1:03d}.pth",
            )
            save_checkpoint(model, optimizer, epoch, val_acc, ckpt_path)

        # ── Save best model ───────────────────────────────────────────────────
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_checkpoint(
                model, optimizer, epoch, val_acc, TRAINING_CONFIG["best_model_path"]
            )
            print(f"  [Best] New best val_acc={val_acc:.4f} → saved best_model.pth")

    # Save training log
    with open(TRAINING_CONFIG["log_path"], "w") as f:
        json.dump(history, f, indent=2)
    print(f"\n[Training] Log saved to {TRAINING_CONFIG['log_path']}")
    print(f"[Training] Best Val Acc = {best_val_acc:.4f}")
    print("=" * 70 + "\n")

    return history


def evaluate_on_test_set(
    checkpoint_path: str = TRAINING_CONFIG["best_model_path"],
) -> None:
    """
    Load the best model checkpoint and evaluate on the held-out test split.

    Args:
        checkpoint_path : Path to saved checkpoint.
    """
    print("\n[Evaluation] Loading best model for test-set evaluation...")
    _, _, test_loader = build_dataloaders()

    model = build_model()
    model, _, _, _ = load_checkpoint(model, None, checkpoint_path)

    criterion = nn.CrossEntropyLoss()
    test_loss, test_acc = evaluate(model, test_loader, criterion, split="Test")

    print(f"[Evaluation] Test Loss={test_loss:.4f}  Test Acc={test_acc:.4f}")
