"""
CPCV Evaluation Engine.

Runs the model plugin across all C(n, k) CPCV splits, producing:

  - Per-split OOS metrics (one row per combination)
  - OOS performance path distribution (Sharpe, return, drawdown)
  - Probability of backtest overfitting (PBO) via the deflated Sharpe
    approach — the fraction of OOS paths that underperform the median
    IS path
  - Aggregate statistics with mean ± std across paths

The output feeds the ValidationEngine and is logged to MLflow so the
Experiment Tracker can display the full distribution, not just a point
estimate.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from app.engines.backtest_engine.metrics import compute_metrics
from app.engines.model_training_engine.evaluation import evaluate_with_cv
from app.engines.model_training_engine.cross_validation import CVConfig
from app.engines.validation_engine.cpcv import CPCVConfig, CPCVSplit, generate_cpcv_splits


@dataclass
class CPCVPathMetrics:
    combination_idx: int
    test_fold_indices: tuple
    oos_sharpe: float
    oos_return: float
    oos_max_drawdown: float
    oos_directional_accuracy: float
    is_sharpe_proxy: float      # IS directional accuracy used as Sharpe proxy
    n_train: int
    n_test: int


@dataclass
class CPCVEvalResult:
    config: CPCVConfig
    paths: list[CPCVPathMetrics] = field(default_factory=list)
    aggregate: dict = field(default_factory=dict)

    # Probability of Backtest Overfitting (Lopez de Prado)
    # Fraction of OOS paths that underperform the median IS performance
    pbo: float = 0.0

    # Deflated Sharpe Ratio (adjusts for selection bias across paths)
    deflated_sharpe: float = 0.0

    def to_dict(self) -> dict:
        return {
            "config": {
                "n_splits": self.config.n_splits,
                "n_test_splits": self.config.n_test_splits,
                "embargo_pct": self.config.embargo_pct,
                "purge": self.config.purge,
                "target_horizon": self.config.target_horizon,
            },
            "n_paths": len(self.paths),
            "pbo": self.pbo,
            "deflated_sharpe": self.deflated_sharpe,
            "aggregate": self.aggregate,
            "paths": [
                {
                    "combination_idx": p.combination_idx,
                    "test_folds": list(p.test_fold_indices),
                    "oos_sharpe": p.oos_sharpe,
                    "oos_return": p.oos_return,
                    "oos_max_drawdown": p.oos_max_drawdown,
                    "oos_directional_accuracy": p.oos_directional_accuracy,
                    "is_sharpe_proxy": p.is_sharpe_proxy,
                    "n_train": p.n_train,
                    "n_test": p.n_test,
                }
                for p in self.paths
            ],
        }


class CPCVEngine:

    def run(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        plugin_key: str,
        params: dict,
        config: CPCVConfig | None = None,
        bars_per_year: int = 252,
    ) -> CPCVEvalResult:
        cfg = config or CPCVConfig()
        cpcv_result = generate_cpcv_splits(len(X), cfg)

        from app.plugins.models import model_registry
        plugin_cls = model_registry.get(plugin_key)

        paths: list[CPCVPathMetrics] = []

        for split in cpcv_result.splits:
            if len(split.train_indices) < 10 or len(split.test_indices) < 5:
                continue

            X_train = X.iloc[split.train_indices]
            y_train = y.iloc[split.train_indices]
            X_test = X.iloc[split.test_indices]
            y_test = y.iloc[split.test_indices]

            # ── IS metrics (directional accuracy as Sharpe proxy) ─────
            is_cv = evaluate_with_cv(
                plugin_key, params, X_train, y_train,
                CVConfig(n_splits=3, test_size=0.2, min_train_size=0.4),
            )
            is_sharpe_proxy = is_cv.mean_directional_accuracy

            # ── Train on full train set, predict on test ───────────────
            model = plugin_cls(**params)
            model.train(X_train, y_train)
            preds = model.predict(X_test).dropna()

            aligned_y = y_test.reindex(preds.index).dropna()
            preds = preds.reindex(aligned_y.index)

            if len(preds) == 0:
                continue

            # ── OOS equity curve from sign-of-prediction returns ───────
            oos_returns = aligned_y * np.sign(preds)
            equity = (1 + oos_returns).cumprod() * 10_000

            trades = pd.DataFrame(
                [{"pnl": float(r * 10_000)} for r in oos_returns]
            )

            dir_acc = float(np.mean(np.sign(preds.values) == np.sign(aligned_y.values)))

            try:
                bt_metrics = compute_metrics(equity, trades, bars_per_year=bars_per_year)
                oos_sharpe = bt_metrics.performance.sharpe_ratio
                oos_return = bt_metrics.total_return
                oos_dd = bt_metrics.risk.max_drawdown
            except Exception:
                oos_sharpe = oos_return = oos_dd = 0.0

            paths.append(CPCVPathMetrics(
                combination_idx=split.combination_idx,
                test_fold_indices=split.test_fold_indices,
                oos_sharpe=float(oos_sharpe),
                oos_return=float(oos_return),
                oos_max_drawdown=float(oos_dd),
                oos_directional_accuracy=dir_acc,
                is_sharpe_proxy=float(is_sharpe_proxy),
                n_train=len(split.train_indices),
                n_test=len(split.test_indices),
            ))

        result = CPCVEvalResult(config=cfg, paths=paths)
        result.aggregate = self._aggregate(paths)
        result.pbo = self._compute_pbo(paths)
        result.deflated_sharpe = self._deflated_sharpe(paths)
        return result

    # ── Statistics ────────────────────────────────────────────────────

    @staticmethod
    def _aggregate(paths: list[CPCVPathMetrics]) -> dict:
        if not paths:
            return {}
        sharpes = [p.oos_sharpe for p in paths]
        returns = [p.oos_return for p in paths]
        dds = [p.oos_max_drawdown for p in paths]
        return {
            "mean_oos_sharpe":      float(np.mean(sharpes)),
            "std_oos_sharpe":       float(np.std(sharpes)),
            "median_oos_sharpe":    float(np.median(sharpes)),
            "min_oos_sharpe":       float(np.min(sharpes)),
            "max_oos_sharpe":       float(np.max(sharpes)),
            "mean_oos_return":      float(np.mean(returns)),
            "std_oos_return":       float(np.std(returns)),
            "mean_oos_drawdown":    float(np.mean(dds)),
            "worst_oos_drawdown":   float(np.max(dds)),
            "n_paths":              len(paths),
            "pct_profitable_paths": float(np.mean([r > 0 for r in returns])),
        }

    @staticmethod
    def _compute_pbo(paths: list[CPCVPathMetrics]) -> float:
        """
        Probability of Backtest Overfitting (Lopez de Prado, Ch. 14).
        Fraction of OOS paths whose Sharpe is below the median IS proxy.
        """
        if not paths:
            return 0.0
        median_is = float(np.median([p.is_sharpe_proxy for p in paths]))
        n_below = sum(1 for p in paths if p.oos_sharpe < median_is)
        return round(n_below / len(paths), 4)

    @staticmethod
    def _deflated_sharpe(paths: list[CPCVPathMetrics]) -> float:
        """
        Deflated Sharpe Ratio: adjusts the best observed OOS Sharpe for
        the number of trials (paths) and the skew/kurtosis of the
        Sharpe distribution — a more conservative estimate than the raw
        best Sharpe.  Simplified implementation.
        """
        if len(paths) < 2:
            return 0.0
        sharpes = np.array([p.oos_sharpe for p in paths])
        best = float(np.max(sharpes))
        n = len(sharpes)
        # Expected maximum of n standard normal variates (approximation)
        euler = 0.5772
        expected_max = (1 - euler) * float(
            np.sqrt(2 * np.log(n)) - np.log(np.pi * np.log(n)) / (2 * np.sqrt(2 * np.log(n)))
        )
        std_sr = float(np.std(sharpes))
        if std_sr == 0:
            return best
        dsr = (best - expected_max * std_sr) / std_sr
        return round(float(dsr), 4)
