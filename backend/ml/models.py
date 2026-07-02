from __future__ import annotations

import torch
from torch import nn

LABELS = ["normal_vitals", "critical_vitals", "device_error"]


class LSTMClassifier(nn.Module):
    def __init__(self, input_dim: int = 8, hidden_dim: int = 48, num_classes: int = 3):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.head = nn.Sequential(nn.Dropout(0.15), nn.Linear(hidden_dim, num_classes))

    def forward(self, x):
        _, (hidden, _) = self.lstm(x)
        return self.head(hidden[-1])


class AdaptiveFusionGate(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x)
