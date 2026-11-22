"""Sequence models for 15-class hand-gesture recognition.

Input:  (B, T, 85)  see src/datasets/ipn_landmarks.py
Output: (B, 15) logits
"""

from __future__ import annotations

import torch
import torch.nn as nn


class MLPFrame(nn.Module):
    """Per-frame MLP then mean-pool: the no-temporal-order baseline."""

    def __init__(self, in_dim: int = 85, hidden: int = 128, num_classes: int = 15,
                 dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).mean(dim=1)


class BiGRU(nn.Module):
    def __init__(self, in_dim: int = 85, hidden: int = 128, num_classes: int = 15,
                 layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.inp = nn.Linear(in_dim, hidden)
        self.gru = nn.GRU(hidden, hidden, num_layers=layers, batch_first=True,
                          bidirectional=True, dropout=dropout if layers > 1 else 0.0)
        self.head = nn.Linear(hidden * 2, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, _ = self.gru(self.inp(x))
        return self.head(h.mean(dim=1))


class _TCNBlock(nn.Module):
    def __init__(self, channels: int, dilation: int, dropout: float = 0.1):
        super().__init__()
        self.conv = nn.Conv1d(channels, channels, kernel_size=3,
                              padding=dilation, dilation=dilation)
        self.norm = nn.BatchNorm1d(channels)
        self.act = nn.ReLU(inplace=True)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.drop(self.act(self.norm(self.conv(x))))


class TCN(nn.Module):
    def __init__(self, in_dim: int = 85, channels: int = 128, num_classes: int = 15,
                 dilations=(1, 2, 4, 8), dropout: float = 0.1):
        super().__init__()
        self.inp = nn.Conv1d(in_dim, channels, kernel_size=1)
        self.blocks = nn.Sequential(*[_TCNBlock(channels, d, dropout) for d in dilations])
        self.head = nn.Linear(channels, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.inp(x.transpose(1, 2))       # (B, C, T)
        h = self.blocks(h)
        return self.head(h.mean(dim=2))


class GestureTransformer(nn.Module):
    def __init__(self, in_dim: int = 85, d_model: int = 128, num_classes: int = 15,
                 layers: int = 2, heads: int = 4, ff: int = 256,
                 dropout: float = 0.1, max_len: int = 64):
        super().__init__()
        self.inp = nn.Linear(in_dim, d_model)
        self.pos = nn.Parameter(torch.randn(1, max_len, d_model) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=heads, dim_feedforward=ff,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=layers)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.inp(x) + self.pos[:, : x.size(1)]
        h = self.norm(self.encoder(h))
        return self.head(h.mean(dim=1))


MODELS = {
    "mlp": MLPFrame,
    "gru": BiGRU,
    "tcn": TCN,
    "transformer": GestureTransformer,
}


def build_model(name: str, in_dim: int = 85, num_classes: int = 15) -> nn.Module:
    if name not in MODELS:
        raise ValueError(f"unknown model {name!r}; choose from {sorted(MODELS)}")
    return MODELS[name](in_dim=in_dim, num_classes=num_classes)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
