"""
Time-series cross-validation.

Ordinary k-fold CV shuffles rows and leaks future information into training
folds — never appropriate for time series. This module provides simple,
strictly-ordered rolling/expanding window splitters for use during model
training and hyperparameter tuning.

Note: this is intentionally minimal. The full purged/embargoed
Combinatorial Purged Cross-Validation (CPCV) and walk-forward analysis
required for rigorous strategy *validation* (as opposed to in-training
model selection) live in the Validation Engine (Phase 7/8), which also
guards against label leakage from overlapping prediction horizons.
"""
from dataclasses import dataclass
from typing import Iterator

import numpy as np


@dataclass
class CVConfig:
    method: str = "rolling"  # "rolling" | "expanding"
    n_splits: int = 5
    test_size: float = 0.15  # fraction of total length per test fold
    min_train_size: float = 0.2  # fraction of total length for the first train window


class TimeSeriesCV:
    def __init__(self, config: CVConfig):
        self.config = config

    def split(self, n_samples: int) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        cfg = self.config
        test_len = max(1, int(n_samples * cfg.test_size))
        min_train_len = max(1, int(n_samples * cfg.min_train_size))

        # Evenly space `n_splits` test windows across the remaining samples.
        usable = n_samples - min_train_len - test_len
        if usable <= 0:
            raise ValueError(
                "Not enough samples for the requested CV config "
                f"(n_samples={n_samples}, min_train_size={min_train_len}, test_size={test_len})"
            )

        step = max(1, usable // max(1, cfg.n_splits - 1)) if cfg.n_splits > 1 else usable

        for i in range(cfg.n_splits):
            train_end = min_train_len + i * step
            test_start = train_end
            test_end = min(test_start + test_len, n_samples)
            if test_end <= test_start or train_end > n_samples:
                break

            if cfg.method == "expanding":
                train_idx = np.arange(0, train_end)
            else:  # rolling
                train_start = max(0, train_end - min_train_len)
                train_idx = np.arange(train_start, train_end)

            test_idx = np.arange(test_start, test_end)
            yield train_idx, test_idx
