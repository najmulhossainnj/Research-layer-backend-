"""
Long-short signal generator.

Converts a continuous prediction score into a signed position size
(+1.0 to -1.0) by clipping and optionally normalising. Supports:

  - Raw pass-through  (size = clip(prediction, -1, 1))
  - Z-score scaling   (size = clip(z-scored prediction, -1, 1))
  - Volatility target (size scaled so expected vol matches a target)

Emits a numeric `signal` column (not discrete BUY/SELL/HOLD) so the
Backtest Engine can size positions continuously rather than binary.
"""
import pandas as pd
import numpy as np

from app.plugins.base import BaseSignalGenerator
from app.plugins.signals import signal_registry


@signal_registry.register("signal.long_short")
class LongShortSignalGenerator(BaseSignalGenerator):
    """
    params:
        prediction_column : str    default "prediction"
        scaling           : str    "clip" | "zscore" | "vol_target"
        vol_target        : float  annualised vol target when scaling="vol_target"
        clip_min          : float  default -1.0
        clip_max          : float  default  1.0
        zscore_window     : int    rolling window for z-score (default 60)
    """

    def generate(self, predictions: pd.DataFrame) -> pd.DataFrame:
        pred_col = self.params.get("prediction_column", "prediction")
        scaling = self.params.get("scaling", "clip")
        clip_min = self.params.get("clip_min", -1.0)
        clip_max = self.params.get("clip_max", 1.0)

        raw = predictions[pred_col].copy()
        out = predictions.copy()

        if scaling == "zscore":
            window = self.params.get("zscore_window", 60)
            mu = raw.rolling(window, min_periods=1).mean()
            sigma = raw.rolling(window, min_periods=1).std().replace(0, np.nan).fillna(1.0)
            scaled = ((raw - mu) / sigma).clip(clip_min, clip_max)

        elif scaling == "vol_target":
            vol_target = self.params.get("vol_target", 0.15)
            window = self.params.get("zscore_window", 60)
            # Annualised rolling vol of predictions as a proxy for signal vol
            rolling_vol = (
                raw.rolling(window, min_periods=5).std() * np.sqrt(252)
            ).replace(0, np.nan).fillna(raw.std() * np.sqrt(252) or 1.0)
            scaled = (raw * vol_target / rolling_vol).clip(clip_min, clip_max)

        else:  # "clip"
            scaled = raw.clip(clip_min, clip_max)

        out["signal"] = scaled  # numeric, not discrete
        out["signal_raw"] = raw
        return out
