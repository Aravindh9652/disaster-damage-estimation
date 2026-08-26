"""ResNet50 damage classifier (xView2-style ordinal 4-class head)."""

from __future__ import annotations

import torch
from torch import nn
from torchvision.models import ResNet50_Weights, resnet50

from labels import CLASSES


def build_resnet50(pretrained: bool = True, freeze_backbone: bool = True) -> nn.Module:
    weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
    model = resnet50(weights=weights)
    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(in_features, 256),
        nn.ReLU(inplace=True),
        nn.Dropout(0.2),
        nn.Linear(256, len(CLASSES)),
    )
    return model


class OrdinalCrossEntropy(nn.Module):
    """Penalize distant Joint Damage Scale mistakes more than adjacent ones."""

    def __init__(self, class_weights: torch.Tensor | None = None) -> None:
        super().__init__()
        self.class_weights = class_weights

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        log_probs = torch.log_softmax(logits, dim=1)
        n_classes = logits.size(1)
        distances = (torch.arange(n_classes, device=logits.device).unsqueeze(0) - target.unsqueeze(1)).abs()
        ordinal_w = 1.0 + distances.float()
        nll = -log_probs.gather(1, target.unsqueeze(1)).squeeze(1)
        if self.class_weights is not None:
            nll = nll * self.class_weights[target]
        penalty = ordinal_w.gather(1, target.unsqueeze(1)).squeeze(1)
        return (nll * penalty).mean()
