"""
Ranking signal generator.

Instead of an absolute threshold ("prediction > 0.7"), this plugin ranks
all predictions in each time step and assigns BUY to the top-N, SELL to
the bottom-N, and HOLD to the rest. Useful for cross-sectional /
universe-level strategies where the absolute scale of predictions is less
meaningful than relative rank.
"""
import pandas as pd

from app.plugins.base import BaseSignalGenerator
from app.plugins.signals import signal_registry


@signal_registry.register("signal.ranking")
class RankingSignalGenerator(BaseSignalGenerator):
    """
    params:
        prediction_column : str   column in the input DataFrame (default: "prediction")
        top_n             : int   number of top-ranked rows to mark BUY
        bottom_n          : int   number of bottom-ranked rows to mark SELL
        position_mode     : str   "long_only" suppresses SELL → HOLD
    """

    def generate(self, predictions: pd.DataFrame) -> pd.DataFrame:
        pred_col = self.params.get("prediction_column", "prediction")
        top_n = self.params.get("top_n", 5)
        bottom_n = self.params.get("bottom_n", 5)
        position_mode = self.params.get("position_mode", "long_short")

        out = predictions.copy()
        signal = pd.Series("HOLD", index=out.index)

        # Rank ascending — highest prediction gets rank == n_rows.
        ranks = out[pred_col].rank(method="first", ascending=True)
        n = len(ranks)

        signal[ranks > (n - top_n)] = "BUY"
        if position_mode != "long_only":
            signal[ranks <= bottom_n] = "SELL"

        out["signal"] = signal
        out["rank"] = ranks
        return out
