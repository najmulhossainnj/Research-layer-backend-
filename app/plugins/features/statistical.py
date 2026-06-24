"""
Statistical feature plugins.

Implements the statistical features the spec calls for:
  statistical.momentum          — rate of change over a rolling window
  statistical.mean_reversion    — distance from rolling mean in std units
  statistical.z_score           — generic z-score of any series
  statistical.hurst_exponent    — Hurst exponent (trending vs mean-reverting)
  statistical.volatility_regime — rolling realised vol regime label
"""
import numpy as np
import pandas as pd

from app.plugins.base import BaseFeature
from app.plugins.features import feature_registry


@feature_registry.register("statistical.momentum")
class MomentumFeature(BaseFeature):
    """
    Price momentum: (close[t] / close[t - window]) - 1.
    params: window (default 20)
    """
    def compute(self, data: pd.DataFrame) -> pd.DataFrame:
        window = self.params.get("window", 20)
        mom = data["close"].pct_change(window)
        return pd.DataFrame({"momentum": mom}, index=data.index)


@feature_registry.register("statistical.mean_reversion")
class MeanReversionFeature(BaseFeature):
    """
    How far price is from its rolling mean, in rolling std units.
    Negative values = below mean (buy signal for mean-reversion strategies).
    params: window (default 20)
    """
    def compute(self, data: pd.DataFrame) -> pd.DataFrame:
        window = self.params.get("window", 20)
        close = data["close"]
        mu = close.rolling(window).mean()
        sigma = close.rolling(window).std().replace(0, np.nan)
        mean_rev = (close - mu) / sigma
        return pd.DataFrame({"mean_reversion": mean_rev}, index=data.index)


@feature_registry.register("statistical.z_score")
class ZScoreFeature(BaseFeature):
    """
    Rolling z-score of the close price.
    params: window (default 20), col (default "close")
    """
    def compute(self, data: pd.DataFrame) -> pd.DataFrame:
        window = self.params.get("window", 20)
        col = self.params.get("col", "close")
        series = data[col]
        mu = series.rolling(window).mean()
        sigma = series.rolling(window).std().replace(0, np.nan)
        return pd.DataFrame({"z_score": (series - mu) / sigma}, index=data.index)


@feature_registry.register("statistical.hurst_exponent")
class HurstExponentFeature(BaseFeature):
    """
    Hurst exponent estimated via rescaled range (R/S) analysis.
      H < 0.5 → mean-reverting
      H = 0.5 → random walk
      H > 0.5 → trending

    params:
        window   : int   rolling window in bars (default 100)
        min_lags : int   minimum lag for R/S (default 2)
        max_lags : int   maximum lag for R/S (default 20)
    """
    def compute(self, data: pd.DataFrame) -> pd.DataFrame:
        window   = self.params.get("window", 100)
        min_lags = self.params.get("min_lags", 2)
        max_lags = self.params.get("max_lags", 20)

        close = data["close"].values.astype(float)
        n = len(close)
        hurst = np.full(n, np.nan)

        for i in range(window, n + 1):
            hurst[i - 1] = self._hurst_rs(close[i - window : i], min_lags, max_lags)

        return pd.DataFrame({"hurst_exponent": hurst}, index=data.index)

    @staticmethod
    def _hurst_rs(series: np.ndarray, min_lags: int, max_lags: int) -> float:
        try:
            lags = range(min_lags, min(max_lags + 1, len(series) // 2))
            rs_vals = []
            for lag in lags:
                sub = series[:lag]
                mean = sub.mean()
                deviation = np.cumsum(sub - mean)
                r = deviation.max() - deviation.min()
                s = sub.std(ddof=1)
                if s > 0:
                    rs_vals.append((lag, r / s))
            if len(rs_vals) < 2:
                return np.nan
            x = np.log([v[0] for v in rs_vals])
            y = np.log([v[1] for v in rs_vals])
            return float(np.polyfit(x, y, 1)[0])
        except Exception:
            return np.nan


@feature_registry.register("statistical.volatility_regime")
class VolatilityRegimeFeature(BaseFeature):
    """
    Encodes rolling realised vol into a regime label (0=low, 1=mid, 2=high)
    using rolling percentile thresholds.

    params:
        window          : int   bars for realised vol (default 20)
        rank_window     : int   bars for percentile ranking (default 252)
        low_threshold   : float percentile below which vol is "low" (default 33)
        high_threshold  : float percentile above which vol is "high" (default 67)
    """
    def compute(self, data: pd.DataFrame) -> pd.DataFrame:
        window      = self.params.get("window", 20)
        rank_window = self.params.get("rank_window", 252)
        low_pct     = self.params.get("low_threshold", 33)
        high_pct    = self.params.get("high_threshold", 67)

        returns = data["close"].pct_change()
        realised_vol = returns.rolling(window).std() * np.sqrt(252)

        def _regime(series: pd.Series) -> pd.Series:
            low  = np.percentile(series.dropna(), low_pct)
            high = np.percentile(series.dropna(), high_pct)
            return series.apply(
                lambda v: 0 if v <= low else (2 if v >= high else 1)
                if not np.isnan(v) else np.nan
            )

        regime = realised_vol.rolling(rank_window, min_periods=window).apply(
            lambda x: _regime(pd.Series(x)).iloc[-1], raw=False
        )
        return pd.DataFrame(
            {"realised_vol": realised_vol, "volatility_regime": regime},
            index=data.index,
        )
