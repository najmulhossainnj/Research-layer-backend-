"""
Walk-Forward Analysis engine.

Walk-forward analysis is the gold standard for time-series strategy
validation. Unlike in-sample CV (Phase 3), the full pipeline —
feature computation, model training, signal generation, and backtesting
— is re-executed on each fold's *out-of-sample* window, simulating how
the strategy would have been deployed and retrained in live operation.

Three window types are supported:

  Rolling   — fixed-size train window slides forward each fold
              [====train====][test] → [====train====][test] → ...
  Expanding — train window grows from a fixed anchor
              [==train==][test] → [====train====][test] → ...
  Anchored  — train always starts at t=0, like expanding but the
              anchor is explicitly the dataset start

The engine produces one `WalkForwardFold` per fold and an aggregate
`WalkForwardResult` summarizing out-of-sample performance across all
folds, which the Validation Engine then logs to MLflow.
"""
from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal

import numpy as np
import pandas as pd

from app.engines.backtest_engine.metrics import BacktestMetrics, compute_metrics
from app.engines.model_training_engine.cross_validation import CVConfig
from app.engines.model_training_engine.evaluation import evaluate_with_cv


@dataclass
class WalkForwardConfig:
    method: Literal["rolling", "expanding", "anchored"] = "rolling"
    n_splits: int = 5
    # Fraction of total length used for each test fold
    test_size: float = 0.2
    # Minimum train fraction (rolling & expanding only)
    min_train_size: float = 0.3
    # Gap between train end and test start (in bars) to prevent leakage
    gap_bars: int = 0
    # Refit the model on each fold (True) or use the initially-trained model (False)
    refit: bool = True


@dataclass
class WalkForwardFold:
    fold_idx: int
    train_start: int
    train_end: int
    test_start: int
    test_end: int
    # Out-of-sample metrics computed on the test window equity curve
    oos_metrics: dict
    # In-sample CV metrics from training on this fold's train set
    is_metrics: dict


@dataclass
class WalkForwardResult:
    config: WalkForwardConfig
    folds: list[WalkForwardFold] = field(default_factory=list)

    # Aggregate OOS metrics (mean ± std across folds)
    aggregate: dict = field(default_factory=dict)

    # Stability flags
    is_sharpe_stable: bool = False      # std/mean < 0.5
    is_profitable_oos: bool = False     # mean Sharpe > 0
    overfitting_score: float = 0.0      # IS Sharpe / OOS Sharpe ratio (1 = no overfit)

    def to_dict(self) -> dict:
        return {
            "config": {
                "method": self.config.method,
                "n_splits": self.config.n_splits,
                "test_size": self.config.test_size,
                "min_train_size": self.config.min_train_size,
                "gap_bars": self.config.gap_bars,
                "refit": self.config.refit,
            },
            "n_folds": len(self.folds),
            "aggregate": self.aggregate,
            "is_sharpe_stable": self.is_sharpe_stable,
            "is_profitable_oos": self.is_profitable_oos,
            "overfitting_score": self.overfitting_score,
            "folds": [
                {
                    "fold_idx": f.fold_idx,
                    "train_start": f.train_start,
                    "train_end": f.train_end,
                    "test_start": f.test_start,
                    "test_end": f.test_end,
                    "oos_metrics": f.oos_metrics,
                    "is_metrics": f.is_metrics,
                }
                for f in self.folds
            ],
        }


def _make_splits(n: int, cfg: WalkForwardConfig) -> list[tuple[range, range]]:
    """Generate (train_range, test_range) index pairs for each fold."""
    test_len = max(1, int(n * cfg.test_size))
    min_train = max(1, int(n * cfg.min_train_size))
    gap = cfg.gap_bars

    splits = []
    for i in range(cfg.n_splits):
        test_end = n - (cfg.n_splits - 1 - i) * test_len
        test_start = test_end - test_len
        train_end = test_start - gap

        if cfg.method == "rolling":
            train_start = max(0, train_end - min_train)
        else:  # expanding or anchored
            train_start = 0

        if train_end <= train_start or test_start >= test_end:
            continue
        if train_end - train_start < max(5, min_train // 2):
            continue

        splits.append((range(train_start, train_end), range(test_start, test_end)))

    return splits


class WalkForwardEngine:

    def run(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        plugin_key: str,
        params: dict,
        config: WalkForwardConfig | None = None,
        bars_per_year: int = 252,
    ) -> WalkForwardResult:
        """
        Execute walk-forward analysis.

        For each fold: trains the model plugin on the train window,
        predicts on the test window, constructs a synthetic equity curve
        from the prediction sign (long/flat), and computes OOS metrics.

        In a full production deployment the prediction step would run
        through the Signal + Backtest pipeline; here we use the simpler
        sign-based equity proxy so walk-forward can run without market
        data (features + targets are sufficient).
        """
        cfg = config or WalkForwardConfig()
        splits = _make_splits(len(X), cfg)
        if not splits:
            raise ValueError(
                "Could not generate any walk-forward splits with the given config "
                f"(n={len(X)}, n_splits={cfg.n_splits}, test_size={cfg.test_size})"
            )

        from app.plugins.models import model_registry
        plugin_cls = model_registry.get(plugin_key)

        folds: list[WalkForwardFold] = []
        initial_model = None

        for i, (train_range, test_range) in enumerate(splits):
            X_train = X.iloc[train_range]
            y_train = y.iloc[train_range]
            X_test = X.iloc[test_range]
            y_test = y.iloc[test_range]

            # ── Train ────────────────────────────────────────────────
            if cfg.refit or initial_model is None:
                model = plugin_cls(**params)
                model.train(X_train, y_train)
                if not cfg.refit and initial_model is None:
                    initial_model = model
            else:
                model = initial_model

            # ── IS CV metrics ─────────────────────────────────────────
            is_cv = evaluate_with_cv(
                plugin_key, params, X_train, y_train,
                CVConfig(n_splits=3, test_size=0.2, min_train_size=0.4),
            )
            is_metrics = is_cv.summary()

            # ── OOS prediction → equity curve ─────────────────────────
            preds = model.predict(X_test).dropna()
            if len(preds) == 0:
                continue

            # Sign of prediction → daily P&L proxy
            # Return series: y_test * sign(prediction)
            aligned_y = y_test.reindex(preds.index).dropna()
            preds = preds.reindex(aligned_y.index)
            oos_returns = aligned_y * np.sign(preds)

            # Build equity curve from returns
            equity = (1 + oos_returns).cumprod() * 10_000  # notional $10k

            # Dummy trades (we don't have a full engine here)
            trades = pd.DataFrame(
                [{"pnl": float(r * 10_000)} for r in oos_returns],
                columns=["pnl"],
            )

            try:
                oos_bt = compute_metrics(equity, trades, bars_per_year=bars_per_year)
                oos_metrics = oos_bt.to_flat_dict()
                oos_metrics.pop("engine_stats", None)
            except Exception as e:
                oos_metrics = {"error": str(e)}

            folds.append(WalkForwardFold(
                fold_idx=i,
                train_start=train_range.start,
                train_end=train_range.stop,
                test_start=test_range.start,
                test_end=test_range.stop,
                oos_metrics=oos_metrics,
                is_metrics=is_metrics,
            ))

        result = WalkForwardResult(config=cfg, folds=folds)
        result.aggregate = self._aggregate(folds)
        result.is_sharpe_stable = self._check_sharpe_stability(folds)
        result.is_profitable_oos = result.aggregate.get("mean_oos_sharpe", 0.0) > 0
        result.overfitting_score = self._overfitting_score(folds)
        return result

    # ── Aggregation helpers ───────────────────────────────────────────

    @staticmethod
    def _aggregate(folds: list[WalkForwardFold]) -> dict:
        if not folds:
            return {}

        def _fold_vals(key: str) -> list[float]:
            return [f.oos_metrics[key] for f in folds
                    if key in f.oos_metrics and isinstance(f.oos_metrics[key], (int, float))]

        sharpes = _fold_vals("perf_sharpe_ratio")
        returns = _fold_vals("total_return")
        drawdowns = _fold_vals("risk_max_drawdown")

        agg: dict = {}
        if sharpes:
            agg["mean_oos_sharpe"] = float(np.mean(sharpes))
            agg["std_oos_sharpe"] = float(np.std(sharpes))
            agg["min_oos_sharpe"] = float(np.min(sharpes))
        if returns:
            agg["mean_oos_return"] = float(np.mean(returns))
            agg["std_oos_return"] = float(np.std(returns))
        if drawdowns:
            agg["mean_oos_max_drawdown"] = float(np.mean(drawdowns))
            agg["worst_oos_drawdown"] = float(np.max(drawdowns))

        agg["n_profitable_folds"] = sum(1 for r in returns if r > 0)
        agg["n_folds"] = len(folds)
        return agg

    @staticmethod
    def _check_sharpe_stability(folds: list[WalkForwardFold]) -> bool:
        sharpes = [f.oos_metrics.get("perf_sharpe_ratio", 0.0) for f in folds
                   if isinstance(f.oos_metrics.get("perf_sharpe_ratio"), (int, float))]
        if not sharpes or np.mean(sharpes) == 0:
            return False
        cv_ratio = np.std(sharpes) / abs(np.mean(sharpes))
        return bool(cv_ratio < 0.5)

    @staticmethod
    def _overfitting_score(folds: list[WalkForwardFold]) -> float:
        """IS Sharpe / OOS Sharpe. Values >> 1 indicate overfitting."""
        is_sharpes = [f.is_metrics.get("mean_directional_accuracy", 0.5) for f in folds]
        oos_sharpes = [f.oos_metrics.get("perf_sharpe_ratio", 0.0) for f in folds
                       if isinstance(f.oos_metrics.get("perf_sharpe_ratio"), (int, float))]
        if not oos_sharpes or np.mean(oos_sharpes) == 0:
            return float("inf")
        return round(float(np.mean(is_sharpes)) / abs(float(np.mean(oos_sharpes))), 3)
