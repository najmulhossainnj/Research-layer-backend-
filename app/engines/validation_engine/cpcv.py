"""
Combinatorial Purged Cross-Validation (CPCV).

Implements the method from Lopez de Prado, "Advances in Financial Machine
Learning" (2018), Chapter 12. CPCV addresses two leakage sources that
ordinary k-fold and even walk-forward CV miss:

  1. Overlapping labels — when the target y[t] is a forward return over
     h bars, the label at bar t overlaps with bars t+1 … t+h.  A train
     sample adjacent to a test sample therefore leaks information about
     the test label.  **Purging** removes any train sample whose label
     window overlaps the test window.

  2. Serial correlation / momentum — even without label overlap, bars
     just outside the test window carry information about the test window
     via autocorrelation.  An **embargo** of `embargo_pct` bars after
     each test fold is also dropped from training.

Additionally, CPCV uses *combinatorial* selection: given n total folds,
it generates all C(n, k) combinations of k folds to use as the test set.
This produces many more distinct test paths than standard walk-forward
(whose test paths number n - k + 1) and provides a distribution of OOS
performance paths rather than a single path — allowing overfitting
detection via the variance of that distribution.

Reference:  https://mlfinlab.readthedocs.io/en/latest/cross_validation/
"""
from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from typing import Iterator

import numpy as np
import pandas as pd


@dataclass
class CPCVConfig:
    n_splits: int = 6          # total number of folds (n)
    n_test_splits: int = 2     # folds used as test per combination (k)
    embargo_pct: float = 0.01  # fraction of total samples embargoed after each test fold
    purge: bool = True         # enable label-overlap purging
    target_horizon: int = 1    # forward-return horizon used to compute label overlap


@dataclass
class CPCVSplit:
    combination_idx: int
    test_fold_indices: tuple[int, ...]   # which of the n folds are test
    train_indices: np.ndarray            # integer row indices into the dataset
    test_indices: np.ndarray


@dataclass
class CPCVResult:
    config: CPCVConfig
    n_combinations: int
    splits: list[CPCVSplit] = field(default_factory=list)


def _fold_boundaries(n_samples: int, n_splits: int) -> list[tuple[int, int]]:
    """Return (start, end) row index pairs for each of the n_splits folds."""
    fold_size = n_samples // n_splits
    boundaries = []
    for i in range(n_splits):
        start = i * fold_size
        end = (i + 1) * fold_size if i < n_splits - 1 else n_samples
        boundaries.append((start, end))
    return boundaries


def _purge_train(
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    target_horizon: int,
) -> np.ndarray:
    """
    Remove training indices whose *label window* overlaps with the test
    window.  Label at bar t covers bars t … t + horizon - 1.  Any train
    bar t_train where t_train + horizon > min(test_idx) is purged.
    """
    if len(test_idx) == 0:
        return train_idx
    test_start = int(test_idx.min())
    # Train samples whose label touches or crosses into the test window
    cutoff = test_start - target_horizon
    return train_idx[train_idx <= cutoff]


def _embargo_train(
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    n_samples: int,
    embargo_pct: float,
) -> np.ndarray:
    """
    Remove training indices that fall in the embargo period immediately
    following the test window (serial correlation / momentum leakage).
    """
    if len(test_idx) == 0 or embargo_pct <= 0:
        return train_idx
    test_end = int(test_idx.max())
    embargo_len = max(1, int(n_samples * embargo_pct))
    embargo_end = min(n_samples - 1, test_end + embargo_len)
    return train_idx[(train_idx < test_idx.min()) | (train_idx > embargo_end)]


def generate_cpcv_splits(
    n_samples: int,
    config: CPCVConfig,
) -> CPCVResult:
    """
    Generate all C(n_splits, n_test_splits) CPCV splits.

    Each split has:
      - test_indices  : rows in the chosen k folds
      - train_indices : all remaining rows, after purging and embargo
    """
    n = config.n_splits
    k = config.n_test_splits

    if k >= n:
        raise ValueError(
            f"n_test_splits ({k}) must be strictly less than n_splits ({n})"
        )

    boundaries = _fold_boundaries(n_samples, n)
    all_combos = list(combinations(range(n), k))
    result = CPCVResult(config=config, n_combinations=len(all_combos))

    for combo_idx, test_fold_ids in enumerate(all_combos):
        # Collect test indices from the chosen k folds
        test_idx = np.concatenate([
            np.arange(boundaries[f][0], boundaries[f][1])
            for f in test_fold_ids
        ])

        # Train indices = all rows not in test folds
        train_fold_ids = [f for f in range(n) if f not in test_fold_ids]
        train_idx = np.concatenate([
            np.arange(boundaries[f][0], boundaries[f][1])
            for f in train_fold_ids
        ])

        # Purge label-overlapping train samples
        if config.purge and config.target_horizon > 0:
            train_idx = _purge_train(train_idx, test_idx, config.target_horizon)

        # Embargo post-test samples from training
        train_idx = _embargo_train(train_idx, test_idx, n_samples, config.embargo_pct)

        result.splits.append(CPCVSplit(
            combination_idx=combo_idx,
            test_fold_indices=test_fold_ids,
            train_indices=train_idx,
            test_indices=test_idx,
        ))

    return result
