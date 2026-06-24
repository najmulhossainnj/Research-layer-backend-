"""
Example technical feature plugins (RSI, ATR) demonstrating the plugin pattern.
Additional indicators (MACD, Bollinger Bands, VWAP, OBV, ADX) follow the
same shape and can be added here or in their own modules without touching
the feature engine.
"""
import pandas as pd

from app.plugins.base import BaseFeature
from app.plugins.features import feature_registry


@feature_registry.register("technical.rsi")
class RSIFeature(BaseFeature):
    """Relative Strength Index."""

    def compute(self, data: pd.DataFrame) -> pd.DataFrame:
        period = self.params.get("period", 14)
        close = data["close"]
        delta = close.diff()
        gain = delta.clip(lower=0).rolling(period).mean()
        loss = (-delta.clip(upper=0)).rolling(period).mean()
        rs = gain / loss.replace(0, pd.NA)
        rsi = 100 - (100 / (1 + rs))
        return pd.DataFrame({"rsi": rsi})


@feature_registry.register("technical.atr")
class ATRFeature(BaseFeature):
    """Average True Range."""

    def compute(self, data: pd.DataFrame) -> pd.DataFrame:
        period = self.params.get("period", 14)
        high, low, close = data["high"], data["low"], data["close"]
        prev_close = close.shift(1)
        tr = pd.concat(
            [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
        ).max(axis=1)
        atr = tr.rolling(period).mean()
        return pd.DataFrame({"atr": atr})
