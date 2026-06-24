"""
Plugin base interfaces.

All pluggable research components (features, models, signal generators,
backtest engines) implement these abstract interfaces. New plugins are
registered via the registries in `app/plugins/*/registry.py` and must be
installable without modifying core engine code.
"""
from abc import ABC, abstractmethod
from typing import Any

import pandas as pd


class BaseFeature(ABC):
    """A single, versioned feature transform: raw market/alt data -> a feature series."""

    key: str  # unique plugin identifier, e.g. "technical.rsi"

    def __init__(self, **params: Any):
        self.params = params

    @abstractmethod
    def compute(self, data: pd.DataFrame) -> pd.DataFrame:
        """Return a DataFrame of one or more derived feature columns."""
        raise NotImplementedError


class BaseModel(ABC):
    """A trainable predictive model (statistical, ML, or DL)."""

    key: str  # e.g. "ml.xgboost"

    def __init__(self, **params: Any):
        self.params = params

    @abstractmethod
    def train(self, X: pd.DataFrame, y: pd.Series) -> None:
        raise NotImplementedError

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> pd.Series:
        raise NotImplementedError

    def save(self, path: str) -> None:
        raise NotImplementedError

    def load(self, path: str) -> None:
        raise NotImplementedError


class BaseSignalGenerator(ABC):
    """Converts model predictions (+ optional context features) into trade signals."""

    key: str  # e.g. "signal.threshold"

    def __init__(self, **params: Any):
        self.params = params

    @abstractmethod
    def generate(self, predictions: pd.DataFrame) -> pd.DataFrame:
        """Return a DataFrame with a `signal` column (e.g. BUY/SELL/HOLD or +1/0/-1)."""
        raise NotImplementedError


class BaseBacktestEngine(ABC):
    """Adapter over a concrete backtesting library (vectorbt, Backtrader, Lean...)."""

    key: str  # e.g. "engine.vectorbt"

    def __init__(self, **config: Any):
        self.config = config

    @abstractmethod
    def run(self, prices: pd.DataFrame, signals: pd.DataFrame) -> dict:
        """Execute the backtest and return a result dict with trades, equity_curve, metrics."""
        raise NotImplementedError
