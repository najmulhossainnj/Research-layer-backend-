"""
tsfresh automated feature plugin.

Wraps tsfresh's `extract_features` to produce entropy, FFT, trend,
distribution, and autocorrelation features automatically from a price
series — matching the "Automated Features" category in the spec.

Because tsfresh produces hundreds of features, this plugin applies
`MinimalFCParameters` by default (fast, ~60 features) and exposes a
`feature_set` param to select richer sets at the cost of compute time.

params:
    feature_set : str  "minimal" | "efficient" | "comprehensive"
                        (default "minimal")
    column      : str  which price column to extract from (default "close")
    window      : int  rolling window fed into tsfresh (default 50)
"""
import warnings

import numpy as np
import pandas as pd

from app.plugins.base import BaseFeature
from app.plugins.features import feature_registry


@feature_registry.register("automated.tsfresh")
class TSFreshFeature(BaseFeature):

    def compute(self, data: pd.DataFrame) -> pd.DataFrame:
        try:
            from tsfresh import extract_features
            from tsfresh.feature_extraction import (
                MinimalFCParameters,
                EfficientFCParameters,
                ComprehensiveFCParameters,
            )
        except ImportError as exc:
            raise ImportError(
                "tsfresh is required for automated feature extraction. "
                "Install with: pip install tsfresh"
            ) from exc

        feature_set = self.params.get("feature_set", "minimal")
        col = self.params.get("column", "close")
        window = self.params.get("window", 50)

        fc_params = {
            "minimal":       MinimalFCParameters(),
            "efficient":     EfficientFCParameters(),
            "comprehensive": ComprehensiveFCParameters(),
        }.get(feature_set, MinimalFCParameters())

        close = data[col].dropna().values
        n = len(close)

        if n < window:
            return pd.DataFrame(index=data.index)

        rows = []
        for i in range(window, n + 1):
            rows.append({"id": i, "time": range(window), "value": close[i - window : i]})

        ts_df = pd.concat(
            [pd.DataFrame({"id": r["id"], "time": list(r["time"]), "value": list(r["value"])})
             for r in rows],
            ignore_index=True,
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            features = extract_features(
                ts_df,
                column_id="id",
                column_sort="time",
                column_value="value",
                default_fc_parameters=fc_params,
                disable_progressbar=True,
                n_jobs=1,
            )

        # Align back to original index (first `window - 1` rows get NaN)
        features.index = data.index[window - 1 :]
        result = features.reindex(data.index)

        # Drop columns that are all-NaN or constant
        result = result.dropna(axis=1, how="all")
        result = result.loc[:, result.std() > 0]

        return result
