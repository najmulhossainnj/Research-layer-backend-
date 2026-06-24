"""
Portfolio signal generator.

Converts cross-sectional prediction scores for multiple symbols into a
weight vector that sums to 1.0 (or ±1.0 for long-short). Used when the
strategy universe has more than one asset and signals need to be expressed
as portfolio weights rather than per-asset BUY/SELL actions.
"""
import pandas as pd
import numpy as np

from app.plugins.base import BaseSignalGenerator
from app.plugins.signals import signal_registry


@signal_registry.register("signal.portfolio")
class PortfolioSignalGenerator(BaseSignalGenerator):
    """
    params:
        prediction_column : str    default "prediction"
        weighting         : str    "softmax" | "rank" | "equal"
        long_only         : bool   default False
        max_weight        : float  per-asset cap, default 1.0
        temperature       : float  softmax temperature, default 1.0
    """

    def generate(self, predictions: pd.DataFrame) -> pd.DataFrame:
        pred_col = self.params.get("prediction_column", "prediction")
        weighting = self.params.get("weighting", "softmax")
        long_only = self.params.get("long_only", False)
        max_weight = self.params.get("max_weight", 1.0)
        temperature = self.params.get("temperature", 1.0)

        scores = predictions[pred_col].copy()
        out = predictions.copy()

        if long_only:
            scores = scores.clip(lower=0)

        if weighting == "softmax":
            exp = np.exp((scores - scores.max()) / temperature)
            weights = exp / exp.sum()

        elif weighting == "rank":
            ranks = scores.rank(method="average")
            weights = ranks / ranks.sum()
            if not long_only:
                weights = weights - weights.mean()

        else:  # equal
            n = (scores > 0).sum() or 1
            weights = pd.Series(0.0, index=scores.index)
            weights[scores > 0] = 1.0 / n

        weights = weights.clip(-max_weight, max_weight)

        out["signal"] = weights
        out["weight"] = weights
        return out
