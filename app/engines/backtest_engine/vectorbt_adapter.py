"""
VectorBT backtest engine adapter (Phase 5 — full implementation).

Replaces the Phase 1 stub. Handles:
  - Discrete BUY/SELL/HOLD signals  (from the rule engine)
  - Numeric ±1 / float signals      (from long-short / portfolio plugins)
  - Per-trade P&L extraction
  - Full metrics via the shared metrics engine
  - Position sizing (fixed size, percent of equity, or signal-driven weight)
"""
import numpy as np
import pandas as pd

from app.engines.backtest_engine.metrics import BacktestMetrics, compute_metrics
from app.engines.backtest_engine.result import BacktestResult
from app.plugins.base import BaseBacktestEngine


class VectorBTAdapter(BaseBacktestEngine):
    key = "engine.vectorbt"

    # config keys:
    #   initial_capital : float  (default 100_000)
    #   commission      : float  fraction per trade (default 0.0005)
    #   slippage        : float  fraction per trade (default 0.0005)
    #   size            : float  fixed number of shares, or None for 100% allocation
    #   size_type       : str    "shares" | "percent" | "signal_weight"
    #   bars_per_year   : int    252 (daily) | 52 (weekly) | 12 (monthly)
    #   risk_free_rate  : float  annualised (default 0.0)

    def run(self, prices: pd.DataFrame, signals: pd.Series) -> BacktestResult:
        import vectorbt as vbt

        close = prices["close"].dropna()
        signals = signals.reindex(close.index).ffill().fillna("HOLD")

        init_cash = float(self.config.get("initial_capital", 100_000))
        fees = float(self.config.get("commission", 0.0005))
        slippage = float(self.config.get("slippage", 0.0005))
        bars_per_year = int(self.config.get("bars_per_year", 252))
        rfr = float(self.config.get("risk_free_rate", 0.0))
        size_type = self.config.get("size_type", "percent")

        # ── Signal → entries/exits ───────────────────────────────────
        if signals.dtype == object or str(signals.dtype) == "object":
            # Discrete: BUY / SELL / HOLD
            entries = signals == "BUY"
            exits = signals == "SELL"
            size = self.config.get("size", None)
            size_kwarg = {"size": size} if size else {}
        else:
            # Numeric: treat sign as direction, magnitude as weight
            numeric = signals.astype(float)
            entries = numeric > 0
            exits = numeric < 0
            if size_type == "signal_weight":
                size = numeric.abs().clip(0, 1)
                size_kwarg = {"size": size, "size_type": "percent"}
            else:
                size_kwarg = {}

        portfolio = vbt.Portfolio.from_signals(
            close,
            entries,
            exits,
            init_cash=init_cash,
            fees=fees,
            slippage=slippage,
            **size_kwarg,
        )

        # ── Extract equity curve ─────────────────────────────────────
        equity = portfolio.value()

        # ── Extract trades ───────────────────────────────────────────
        trades_records = portfolio.trades.records_readable
        if not trades_records.empty:
            trades = trades_records.rename(columns={
                "Entry Timestamp": "entry_date",
                "Exit Timestamp": "exit_date",
                "Entry Price": "entry_price",
                "Exit Price": "exit_price",
                "Size": "size",
                "PnL": "pnl",
                "Return": "return_pct",
                "Direction": "side",
            })
            keep = [c for c in
                    ["entry_date", "exit_date", "side", "entry_price",
                     "exit_price", "size", "pnl", "return_pct"]
                    if c in trades.columns]
            trades = trades[keep].reset_index(drop=True)
        else:
            trades = pd.DataFrame(
                columns=["entry_date", "exit_date", "side",
                         "entry_price", "exit_price", "size", "pnl", "return_pct"]
            )

        # ── Drawdown series ──────────────────────────────────────────
        rolling_max = equity.cummax()
        drawdowns = (equity - rolling_max) / rolling_max

        # ── Bars in market ───────────────────────────────────────────
        position = portfolio.position_now()
        bars_in_market = int((position != 0).sum()) if hasattr(position, "__len__") else len(equity)

        # ── Unified metrics ──────────────────────────────────────────
        bt_metrics = compute_metrics(
            equity_curve=equity,
            trades=trades,
            risk_free_rate=rfr,
            bars_per_year=bars_per_year,
        )
        bt_metrics.bars_in_market = bars_in_market

        # Raw vbt stats for display
        raw_stats = portfolio.stats().to_dict()
        engine_stats = {k: float(v) if hasattr(v, "__float__") else str(v)
                        for k, v in raw_stats.items()}

        return BacktestResult(
            metrics=bt_metrics,
            equity_curve=equity,
            trades=trades,
            drawdowns=drawdowns,
            engine_key=self.key,
            engine_stats=engine_stats,
        )
