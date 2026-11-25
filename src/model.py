"""Convolutional neural network used for ACP/PCP subtyping.

Input  : SERSome tensor of shape (batch, 1, 200, 907)
Output : sigmoid probability of the PCP class (label 1; ACP is label 0)

12,409 trainable parameters, ~48.5 KB in float32.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class SERSomeCNN(nn.Module):
    def __init__(self, kernel: tuple[int, int] = (3, 6), n_wavenumbers: int = 907, n_spectra: int = 200):
        super().__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(1, 8, kernel), nn.BatchNorm2d(8), nn.ReLU())
        self.conv2 = nn.Sequential(nn.Conv2d(8, 16, kernel), nn.BatchNorm2d(16), nn.ReLU())
        self.conv3 = nn.Sequential(nn.Conv2d(16, 32, kernel), nn.BatchNorm2d(32), nn.ReLU())
        self.maxpool = nn.MaxPool2d((3, 6))

        with torch.no_grad():
            dummy = torch.zeros(1, 1, n_spectra, n_wavenumbers)
            n_flat = self._features(dummy).numel()

        self.fc1 = nn.Linear(n_flat, 1)
        self.dropout = nn.Dropout(p=0.6)
        self.sig = nn.Sigmoid()

    def _features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.maxpool(self.conv1(x))
        x = self.maxpool(self.conv2(x))
        x = self.maxpool(self.conv3(x))
        return torch.flatten(x, start_dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.sig(self.dropout(self.fc1(self._features(x))))


def load_model(path: str, map_location: str | torch.device = "cpu") -> SERSomeCNN:
    """Load a checkpoint saved by ``train_cnn.py`` (state_dict based, no pickle)."""
    ckpt = torch.load(path, map_location=map_location, weights_only=True)
    model = SERSomeCNN()
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model
