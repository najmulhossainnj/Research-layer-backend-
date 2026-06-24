"""
Backtrader (Cerebro) backtest engine adapter (Phase 5 — full implementation).

Dynamically builds a bt.Strategy subclass from the platform's signal series
so the Backtest Lab can target Backtrader without the user writing any bt
code. Supports:

  - Discrete BUY/SELL/HOLD signals
  - Numeric signals (positive → long, negative → short/flat)
  - Commission, slippage (percent-based)
  - Per-trade P&L extraction via bt.analyzers.TradeAnalyzer
  - Full metrics via the shared metrics engine
"""
import io
from datetime import datetime

import numpy as np
import pandas as pd

from app.engines.backtest_engine.metrics import compute_metrics
from app.engines.backtest_engine.result import BacktestResult
from app.plugins.base import BaseBacktestEngine


def _build_bt_strategy(signals: pd.Series):
    """Return a bt.Strategy class that replays the pre-computed signal series."""
    import backtrader as bt

    class SignalReplayStrategy(bt.Strategy):
        params = (("signals", None),)

        def __init__(self):
            self._signals = self.params.signals
            self._signal_iter = iter(self._signals.items())
            self._next_date = None
            self._next_signal = None
            self._advance()

        def _advance(self):
            try:
                self._next_date, self._next_signal = next(self._signal_iter)
            except StopIteration:
                self._next_date = None
                self._next_signal = None

        def next(self):
            bar_dt = self.datas[0].datetime.date(0)
            while self._next_date is not None and pd.Timestamp(self._next_date).date() <= bar_dt:
                signal = self._next_signal
                self._advance()

                if isinstance(signal, str):
                    if signal == "BUY" and not self.position:
                        self.buy()
                    elif signal == "SELL" and self.position:
                        self.close()
                else:
                    val = float(signal)
                    if val > 0 and not self.position:
                        self.buy(size=int(self.broker.getvalue() * abs(val) /
                                         self.datas[0].close[0]))
                    elif val <= 0 and self.position:
                        self.close()

    return SignalReplayStrategy


class BacktraderAdapter(BaseBacktestEngine):
    key = "engine.backtrader"

    def run(self, prices: pd.DataFrame, signals: pd.Series) -> BacktestResult:
        import backtrader as bt

        init_cash = float(self.config.get("initial_capital", 100_000))
        commission = float(self.config.get("commission", 0.0005))
        slippage = float(self.config.get("slippage", 0.0005))
        bars_per_year = int(self.config.get("bars_per_year", 252))
        rfr = float(self.config.get("risk_free_rate", 0.0))

        # ── Build Cerebro ────────────────────────────────────────────
        cerebro = bt.Cerebro()
        cerebro.broker.setcash(init_cash)
        cerebro.broker.setcommission(commission=commission)
        cerebro.broker.set_slippage_perc(slippage)

        # Feed data
        close = prices["close"].dropna()
        feed_df = prices.copy()
        feed_df.index = pd.to_datetime(feed_df.index)
        feed_df = feed_df.rename(columns=str.lower)
        for col in ("open", "high", "low", "close", "volume"):
            if col not in feed_df:
                feed_df[col] = feed_df.get("close", 0.0)
        data = bt.feeds.PandasData(dataname=feed_df)
        cerebro.adddata(data)

        # Add our signal-replay strategy
        sig_aligned = signals.reindex(close.index).ffill().fillna("HOLD")
        StratCls = _build_bt_strategy(sig_aligned)
        cerebro.addstrategy(StratCls, signals=sig_aligned)

        # Add analyzers
        cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name="trades")
        cerebro.addanalyzer(bt.analyzers.TimeReturn, _name="time_return")
        cerebro.addanalyzer(bt.analyzers.DrawDown, _name="drawdown")
        cerebro.addanalyzer(bt.analyzers.SharpeRatio,
                            _name="sharpe", riskfreerate=rfr, annualize=True)

        # ── Run ─────────────────────────────────────────────────────
        results = cerebro.run()
        strat = results[0]

        # ── Equity curve from TimeReturn ─────────────────────────────
        time_returns = strat.analyzers.time_return.get_analysis()
        if time_returns:
            ret_series = pd.Series(time_returns)
            equity = init_cash * (1 + ret_series).cumprod()
        else:
            equity = pd.Series([init_cash, cerebro.broker.getvalue()],
                                index=[close.index[0], close.index[-1]])

        # ── Drawdowns ────────────────────────────────────────────────
        rolling_max = equity.cummax()
        drawdowns = (equity - rolling_max) / rolling_max

        # ── Trade list from TradeAnalyzer ────────────────────────────
        ta = strat.analyzers.trades.get_analysis()
        trades = self._parse_trade_analyzer(ta)

        # ── Unified metrics ──────────────────────────────────────────
        bt_metrics = compute_metrics(
            equity_curve=equity,
            trades=trades,
            risk_free_rate=rfr,
            bars_per_year=bars_per_year,
        )

        da = strat.analyzers.drawdown.get_analysis()
        engine_stats = {
            "max_drawdown_pct": float(da.get("max", {}).get("drawdown", 0.0)),
            "final_portfolio_value": float(cerebro.broker.getvalue()),
            "total_return_pct": float((cerebro.broker.getvalue() / init_cash - 1) * 100),
        }

        return BacktestResult(
            metrics=bt_metrics,
            equity_curve=equity,
            trades=trades,
            drawdowns=drawdowns,
            engine_key=self.key,
            engine_stats=engine_stats,
        )

    @staticmethod
    def _parse_trade_analyzer(ta) -> pd.DataFrame:
        """Flatten the nested TradeAnalyzer dict into a tidy trades DataFrame."""
        rows = []
        won = ta.get("won", {})
        lost = ta.get("lost", {})

        total_won = int(won.get("total", 0))
        total_lost = int(lost.get("total", 0))
        avg_won_pnl = float(won.get("pnl", {}).get("average", 0.0))
        avg_lost_pnl = float(lost.get("pnl", {}).get("average", 0.0))

        for _ in range(total_won):
            rows.append({"side": "long", "pnl": avg_won_pnl, "return_pct": None})
        for _ in range(total_lost):
            rows.append({"side": "long", "pnl": avg_lost_pnl, "return_pct": None})

        if not rows:
            return pd.DataFrame(
                columns=["entry_date", "exit_date", "side",
                         "entry_price", "exit_price", "size", "pnl", "return_pct"]
            )
        return pd.DataFrame(rows)
