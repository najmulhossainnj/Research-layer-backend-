"""
Cross-validated evaluation of a model plugin.

Shared by both the trainer (final CV metrics report) and the hyperparameter
tuner (per-trial objective scoring) so the two always measure performance
the same way.
"""
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

from app.engines.model_training_engine.cross_validation import CVConfig, TimeSeriesCV
from app.plugins.models import model_registry


@dataclass
class FoldMetrics:
    fold: int
    mse: float
    mae: float
    directional_accuracy: float
    n_train: int
    n_test: int


@dataclass
class CVResult:
    folds: list[FoldMetrics] = field(default_factory=list)

    @property
    def mean_mse(self) -> float:
        return float(np.mean([f.mse for f in self.folds])) if self.folds else float("nan")

    @property
    def mean_mae(self) -> float:
        return float(np.mean([f.mae for f in self.folds])) if self.folds else float("nan")

    @property
    def mean_directional_accuracy(self) -> float:
        return (
            float(np.mean([f.directional_accuracy for f in self.folds]))
            if self.folds
            else float("nan")
        )

    def summary(self) -> dict:
        return {
            "mean_mse": self.mean_mse,
            "mean_mae": self.mean_mae,
            "mean_directional_accuracy": self.mean_directional_accuracy,
            "n_folds": len(self.folds),
        }


def evaluate_with_cv(
    plugin_key: str, params: dict, X: pd.DataFrame, y: pd.Series, cv_config: CVConfig
) -> CVResult:
    cv = TimeSeriesCV(cv_config)
    plugin_cls = model_registry.get(plugin_key)

    result = CVResult()
    for i, (train_idx, test_idx) in enumerate(cv.split(len(X))):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        model = plugin_cls(**params)
        model.train(X_train, y_train)
        preds = model.predict(X_test)

        mse = mean_squared_error(y_test, preds)
        mae = mean_absolute_error(y_test, preds)
        directional_accuracy = float(
            np.mean(np.sign(preds) == np.sign(y_test.values))
        )

        result.folds.append(
            FoldMetrics(
                fold=i,
                mse=float(mse),
                mae=float(mae),
                directional_accuracy=directional_accuracy,
                n_train=len(train_idx),
                n_test=len(test_idx),
            )
        )

    return result
