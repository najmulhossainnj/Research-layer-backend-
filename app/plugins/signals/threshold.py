"""
Threshold-based signal generator.

Implements rules like:
    IF prediction > 0.7 THEN BUY
    IF prediction < -0.7 THEN SELL
Optionally combined with extra conditions (e.g. sentiment > 0.5) via AND/OR,
matching the visual rule-engine JSON produced by the Signal Builder UI.
"""
import pandas as pd

from app.plugins.base import BaseSignalGenerator
from app.plugins.signals import signal_registry


@signal_registry.register("signal.threshold")
class ThresholdSignalGenerator(BaseSignalGenerator):
    def generate(self, predictions: pd.DataFrame) -> pd.DataFrame:
        buy_threshold = self.params.get("buy_threshold", 0.7)
        sell_threshold = self.params.get("sell_threshold", -0.7)
        pred_col = self.params.get("prediction_column", "prediction")

        signal = pd.Series("HOLD", index=predictions.index)
        signal[predictions[pred_col] > buy_threshold] = "BUY"
        signal[predictions[pred_col] < sell_threshold] = "SELL"

        out = predictions.copy()
        out["signal"] = signal
        return out
