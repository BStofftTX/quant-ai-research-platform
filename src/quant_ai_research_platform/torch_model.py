"""Optional PyTorch classifier for market-direction research."""

import numpy as np
import pandas as pd
import torch
from torch import nn


class MarketDirectionNetwork(nn.Module):
    """Small neural network for binary market-direction classification."""

    def __init__(self, input_features: int = 3, hidden_units: int = 16):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_features, hidden_units),
            nn.ReLU(),
            nn.Linear(hidden_units, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features).squeeze(-1)


class TorchMarketClassifier:
    """Scikit-learn-compatible interface for a PyTorch classifier."""

    def __init__(
        self,
        hidden_units: int = 16,
        epochs: int = 100,
        learning_rate: float = 0.01,
        seed: int = 42,
        device: str = "cpu",
    ):
        self.hidden_units = hidden_units
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.seed = seed
        self.device = device

    def fit(self, features: pd.DataFrame, target: pd.Series):
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        self.feature_columns_ = features.columns.tolist()

        x = torch.tensor(
            features.to_numpy(dtype=np.float32),
            dtype=torch.float32,
            device=self.device,
        )
        y = torch.tensor(
            target.to_numpy(dtype=np.float32),
            dtype=torch.float32,
            device=self.device,
        )

        self.model_ = MarketDirectionNetwork(
            input_features=x.shape[1],
            hidden_units=self.hidden_units,
        ).to(self.device)

        optimizer = torch.optim.Adam(
            self.model_.parameters(),
            lr=self.learning_rate,
        )
        loss_function = nn.BCEWithLogitsLoss()

        self.model_.train()

        for _ in range(self.epochs):
            optimizer.zero_grad()
            logits = self.model_(x)
            loss = loss_function(logits, y)
            loss.backward()
            optimizer.step()

        self.model_.eval()
        return self

    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        x = torch.tensor(
            features[self.feature_columns_].to_numpy(dtype=np.float32),
            dtype=torch.float32,
            device=self.device,
        )

        with torch.no_grad():
            logits = self.model_(x)
            probabilities = torch.sigmoid(logits).cpu().numpy()

        return np.column_stack([1.0 - probabilities, probabilities])

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        return (self.predict_proba(features)[:, 1] >= 0.5).astype(int)
