from __future__ import annotations

import numpy as np
import torch
from torch import nn


def make_windows(X: np.ndarray, y: np.ndarray, intervals, window: int = 32, stride: int = 4):
    xs, ys = [], []
    for start, end in intervals:
        left = int(start) + window - 1
        for idx in range(left, int(end) + 1, stride):
            xs.append(X[idx - window + 1:idx + 1])
            ys.append(y[idx])
    if not xs:
        return np.empty((0, window, X.shape[1]), dtype=np.float32), np.empty((0,), dtype=np.int64)
    return np.stack(xs).astype(np.float32), np.asarray(ys, dtype=np.int64)


def fit_standardizer(train_x: np.ndarray):
    mean = train_x.mean(axis=(0, 1), keepdims=True)
    std = train_x.std(axis=(0, 1), keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)
    return mean, std


def apply_standardizer(x: np.ndarray, mean: np.ndarray, std: np.ndarray):
    return (x - mean) / std


class Cnn1d(nn.Module):
    def __init__(self, n_features: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(n_features, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Linear(32, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.net(x.transpose(1, 2)).squeeze(-1)
        return self.head(z).squeeze(-1)


class GruNet(nn.Module):
    def __init__(self, n_features: int) -> None:
        super().__init__()
        self.gru = nn.GRU(input_size=n_features, hidden_size=32, batch_first=True)
        self.head = nn.Linear(32, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, hidden = self.gru(x)
        return self.head(hidden[-1]).squeeze(-1)


class LstmNet(nn.Module):
    def __init__(self, n_features: int) -> None:
        super().__init__()
        self.lstm = nn.LSTM(input_size=n_features, hidden_size=32, batch_first=True)
        self.head = nn.Linear(32, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, (hidden, _) = self.lstm(x)
        return self.head(hidden[-1]).squeeze(-1)


def build_model(name: str, n_features: int) -> nn.Module:
    if name == 'CNN1D':
        return Cnn1d(n_features)
    if name == 'GRU':
        return GruNet(n_features)
    if name == 'LSTM':
        return LstmNet(n_features)
    raise ValueError(name)
