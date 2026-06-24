"""
Backtest result container.

Every engine adapter (VectorBT, Backtrader, future Lean) returns a
`BacktestResult` so the rest of the system (result storage, API response,
experiment tracking) is fully engine-agnostic.
"""
from dataclasses import dataclass, field

import pandas as pd

from app.engines.backtest_engine.metrics import BacktestMetrics


@dataclass
class BacktestResult:
    # Normalized metrics computed by the metrics engine
    metrics: BacktestMetrics

    # Portfolio value indexed by bar/date
    equity_curve: pd.Series

    # One row per closed trade
    # Expected columns: entry_date, exit_date, side, entry_price,
    #                   exit_price, size, pnl, return_pct
    trades: pd.DataFrame

    # Drawdown series (fraction from peak), same index as equity_curve
    drawdowns: pd.Series

    # Which engine produced this result
    engine_key: str

    # Raw engine-specific stats dict (for display in the Backtest Lab detail view)
    engine_stats: dict = field(default_factory=dict)
