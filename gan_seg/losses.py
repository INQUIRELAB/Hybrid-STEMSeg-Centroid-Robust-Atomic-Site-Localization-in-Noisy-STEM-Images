"""Segmentation losses for sparse binary masks."""

from __future__ import annotations

import torch
import torch.nn as nn


def soft_dice_loss_logits(
    logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-6
) -> torch.Tensor:
    """1 - soft Dice for binary masks; logits (N,1,H,W), target (N,1,H,W) in {0,1}."""
    p = torch.sigmoid(logits)
    dims = (2, 3)
    inter = (p * target).sum(dim=dims)
    denom = p.pow(2).sum(dim=dims) + target.pow(2).sum(dim=dims)
    dice = (2 * inter + eps) / (denom + eps)
    return 1.0 - dice.mean()


class CombinedSegLoss(nn.Module):
    """pos-weighted BCE + weighted soft Dice."""

    def __init__(self, pos_weight: float = 18.0, dice_weight: float = 0.5):
        super().__init__()
        self.dice_weight = dice_weight
        self.register_buffer("_pw", torch.tensor([pos_weight]))
        self.bce = nn.BCEWithLogitsLoss(pos_weight=self._pw.view(()))

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.bce(logits, target) + self.dice_weight * soft_dice_loss_logits(
            logits, target
        )
