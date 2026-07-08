# -*- coding: utf-8 -*-
"""
Model architecture for Video Deepfake Detection.

Contains:
  - TemporalAggregator: Mean-pools frame-level features into video-level.
  - SwinDeepfakeDetector: Swin Transformer backbone + temporal pooling + classifier.
  - build_model(): Factory function.

Preserved from the original Colab implementation without architectural changes.
"""

import torch
import torch.nn as nn
import timm

from backend.config import CONFIG, DEVICE


class TemporalAggregator(nn.Module):
    """
    Aggregate frame-level features into a video-level representation.

    Strategy: mean-pool across time dimension.
    """
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x : (B, T, feature_dim)
        return x.mean(dim=1)   # → (B, feature_dim)


class SwinDeepfakeDetector(nn.Module):
    """
    Swin Transformer backbone + temporal mean-pooling + classification head.

    For each video:
      1. Each of T frames passes independently through the Swin backbone.
      2. Frame features are mean-pooled → single video embedding.
      3. Dropout + Linear → logits for [REAL, FAKE].
    """

    def __init__(
        self,
        model_name  : str = CONFIG["model_name"],
        num_classes : int = CONFIG["num_classes"],
        pretrained  : bool = CONFIG["pretrained"],
        dropout     : float = CONFIG["dropout"],
    ):
        super().__init__()

        # Load Swin backbone from timm
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,         # remove classification head → returns features
        )

        # Get feature dimension dynamically
        with torch.no_grad():
            dummy = torch.zeros(1, 3, CONFIG["face_size"], CONFIG["face_size"])
            feature_dim = self.backbone(dummy).shape[-1]

        self.aggregator = TemporalAggregator()
        self.dropout    = nn.Dropout(p=dropout)
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.ReLU(),
            nn.Dropout(p=dropout / 2),
            nn.Linear(512, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : (B, T, 3, H, W)  — batch of videos, T frames each

        Returns:
            logits : (B, num_classes)
        """
        B, T, C, H, W = x.shape

        # Merge batch & time → process all frames at once
        x = x.view(B * T, C, H, W)          # (B*T, 3, H, W)
        features = self.backbone(x)           # (B*T, feature_dim)
        features = features.view(B, T, -1)   # (B, T, feature_dim)

        video_emb = self.aggregator(features) # (B, feature_dim)
        video_emb = self.dropout(video_emb)
        logits    = self.classifier(video_emb) # (B, num_classes)

        return logits


def build_model() -> SwinDeepfakeDetector:
    """Create a SwinDeepfakeDetector and move it to the configured device."""
    model = SwinDeepfakeDetector()
    model = model.to(DEVICE)
    return model
