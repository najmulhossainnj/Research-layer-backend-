"""
LSTM model plugin (PyTorch).

Wraps a two-layer LSTM for sequence-to-scalar regression.  The plugin
converts a flat feature DataFrame into overlapping windows, trains with
Adam + MSE, and exposes the same `train/predict/save/load` interface as
every other model plugin so the training engine and tuner treat it
identically to an XGBoost or RandomForest model.
"""
import io
import pickle

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from app.plugins.base import BaseModel
from app.plugins.models import model_registry


class _LSTMNet(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, dropout: float):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers=num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :]).squeeze(-1)


def _make_windows(X: np.ndarray, y: np.ndarray, seq_len: int):
    xs, ys = [], []
    for i in range(len(X) - seq_len):
        xs.append(X[i : i + seq_len])
        ys.append(y[i + seq_len])
    return np.array(xs, dtype=np.float32), np.array(ys, dtype=np.float32)


@model_registry.register("dl.lstm")
class LSTMModel(BaseModel):
    def __init__(self, **params):
        super().__init__(**params)
        self._net: _LSTMNet | None = None
        self._seq_len: int = params.get("seq_len", 20)
        self._scaler_mean: np.ndarray | None = None
        self._scaler_std: np.ndarray | None = None

    def train(self, X: pd.DataFrame, y: pd.Series) -> None:
        self._scaler_mean = X.values.mean(axis=0)
        self._scaler_std = X.values.std(axis=0) + 1e-8
        X_scaled = (X.values - self._scaler_mean) / self._scaler_std

        xs, ys = _make_windows(X_scaled, y.values, self._seq_len)
        dataset = TensorDataset(torch.from_numpy(xs), torch.from_numpy(ys))
        loader = DataLoader(dataset, batch_size=self.params.get("batch_size", 64), shuffle=False)

        self._net = _LSTMNet(
            input_size=X.shape[1],
            hidden_size=self.params.get("hidden_size", 64),
            num_layers=self.params.get("num_layers", 2),
            dropout=self.params.get("dropout", 0.1),
        )
        optimizer = torch.optim.Adam(self._net.parameters(), lr=self.params.get("lr", 1e-3))
        loss_fn = nn.MSELoss()

        for _ in range(self.params.get("epochs", 20)):
            for xb, yb in loader:
                optimizer.zero_grad()
                loss_fn(self._net(xb), yb).backward()
                optimizer.step()

    def predict(self, X: pd.DataFrame) -> pd.Series:
        if self._net is None:
            raise RuntimeError("Model not trained or loaded")
        X_scaled = (X.values - self._scaler_mean) / self._scaler_std
        xs, _ = _make_windows(X_scaled, np.zeros(len(X_scaled)), self._seq_len)
        with torch.no_grad():
            preds = self._net(torch.from_numpy(xs)).numpy()
        # Pad the first seq_len rows with NaN so the index aligns with X.
        full = np.full(len(X), np.nan)
        full[self._seq_len :] = preds
        return pd.Series(full, index=X.index)

    def save(self, path: str) -> None:
        payload = {
            "state_dict": self._net.state_dict() if self._net else None,
            "params": self.params,
            "seq_len": self._seq_len,
            "scaler_mean": self._scaler_mean,
            "scaler_std": self._scaler_std,
        }
        with open(path, "wb") as f:
            pickle.dump(payload, f)

    def load(self, path: str) -> None:
        with open(path, "rb") as f:
            payload = pickle.load(f)
        self.params = payload["params"]
        self._seq_len = payload["seq_len"]
        self._scaler_mean = payload["scaler_mean"]
        self._scaler_std = payload["scaler_std"]
        # Rebuild the network from saved params — input_size recovered from scaler shape.
        input_size = len(self._scaler_mean)
        self._net = _LSTMNet(
            input_size=input_size,
            hidden_size=self.params.get("hidden_size", 64),
            num_layers=self.params.get("num_layers", 2),
            dropout=self.params.get("dropout", 0.1),
        )
        self._net.load_state_dict(payload["state_dict"])
        self._net.eval()
