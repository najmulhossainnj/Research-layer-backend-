"""
Metrics engine.

Single, engine-agnostic implementation of every metric the spec calls for:

  Performance  — CAGR, Sharpe, Sortino, Calmar
  Risk         — Max Drawdown, VaR (historical), CVaR / Expected Shortfall
  Trading      — Win Rate, Profit Factor, Turnover, Avg Trade, Expectancy

Both the VectorBT and Backtrader adapters call this after their own run so
the Experiment Tracker and Backtest Lab always see the same metric
definitions regardless of which engine was used.
"""
from dataclasses import asdict, dataclass
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class PerformanceMetrics:
    cagr: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float


@dataclass
class RiskMetrics:
    max_drawdown: float          # as a positive fraction, e.g. 0.25 = 25%
    max_drawdown_duration: int   # calendar days at longest drawdown trough
    var_95: float                # 1-day 95% VaR (negative = loss)
    cvar_95: float               # Expected Shortfall at 95%
    var_99: float
    cvar_99: float
    volatility_annualised: float


@dataclass
class TradingMetrics:
    total_trades: int
    win_rate: float              # fraction of trades that are profitable
    profit_factor: float         # gross profit / gross loss
    avg_win: float
    avg_loss: float
    expectancy: float            # expected P&L per trade
    turnover_annualised: float   # average annual portfolio turnover


@dataclass
class BacktestMetrics:
    performance: PerformanceMetrics
    risk: RiskMetrics
    trading: TradingMetrics
    total_return: float
    bars_in_market: int          # bars where position != 0

    def to_flat_dict(self) -> dict:
        """Flat dict for JSON storage in the Backtest.metrics column."""
        d = {}
        d["total_return"] = self.total_return
        d["bars_in_market"] = self.bars_in_market
        for k, v in asdict(self.performance).items():
            d[f"perf_{k}"] = v
        for k, v in asdict(self.risk).items():
            d[f"risk_{k}"] = v
        for k, v in asdict(self.trading).items():
            d[f"trade_{k}"] = v
        return d


def _safe(val: float, default: float = 0.0) -> float:
    if val is None or not np.isfinite(val):
        return default
    return float(val)


def compute_metrics(
    equity_curve: pd.Series,
    trades: pd.DataFrame,
    risk_free_rate: float = 0.0,
    bars_per_year: int = 252,
) -> BacktestMetrics:
    """
    Parameters
    ----------
    equity_curve  : Series of portfolio value indexed by date/bar,
                    starting from initial capital.
    trades        : DataFrame with columns: pnl, return_pct, side
                    (one row per closed trade). Can be empty.
    risk_free_rate: Annualised, as a fraction (e.g. 0.05 = 5%).
    bars_per_year : 252 for daily, 52 for weekly, 12 for monthly.
    """
    ec = equity_curve.dropna()
    if len(ec) < 2:
        raise ValueError("Equity curve must have at least 2 data points")

    # ── Returns ───────────────────────────────────────────────────────
    returns = ec.pct_change().dropna()
    total_return = float((ec.iloc[-1] / ec.iloc[0]) - 1.0)

    # ── Performance ───────────────────────────────────────────────────
    n_bars = len(returns)
    n_years = n_bars / bars_per_year

    cagr = float((ec.iloc[-1] / ec.iloc[0]) ** (1.0 / max(n_years, 1e-6)) - 1.0)

    rf_per_bar = (1 + risk_free_rate) ** (1.0 / bars_per_year) - 1
    excess = returns - rf_per_bar
    vol = float(returns.std()) * np.sqrt(bars_per_year)

    sharpe = _safe(float(excess.mean() / returns.std()) * np.sqrt(bars_per_year))

    downside_returns = returns[returns < rf_per_bar]
    downside_vol = float(downside_returns.std()) * np.sqrt(bars_per_year) if len(downside_returns) > 1 else 1e-9
    sortino = _safe(float(excess.mean()) * bars_per_year / downside_vol)

    # ── Drawdown ──────────────────────────────────────────────────────
    rolling_max = ec.cummax()
    drawdown = (ec - rolling_max) / rolling_max
    max_drawdown = float(abs(drawdown.min()))

    # Drawdown duration: longest run of bars below the previous high
    underwater = drawdown < 0
    dd_duration = 0
    current_run = 0
    for u in underwater:
        if u:
            current_run += 1
            dd_duration = max(dd_duration, current_run)
        else:
            current_run = 0

    calmar = _safe(cagr / max_drawdown if max_drawdown > 0 else 0.0)

    # ── VaR / CVaR ────────────────────────────────────────────────────
    var_95 = float(np.percentile(returns, 5))
    cvar_95 = float(returns[returns <= var_95].mean())
    var_99 = float(np.percentile(returns, 1))
    cvar_99 = float(returns[returns <= var_99].mean())

    # ── Trading metrics ───────────────────────────────────────────────
    if trades is not None and not trades.empty and "pnl" in trades.columns:
        pnl = trades["pnl"].dropna()
        winners = pnl[pnl > 0]
        losers = pnl[pnl < 0]

        total_trades = len(pnl)
        win_rate = _safe(len(winners) / total_trades) if total_trades else 0.0
        gross_profit = float(winners.sum()) if len(winners) else 0.0
        gross_loss = float(abs(losers.sum())) if len(losers) else 1e-9
        profit_factor = _safe(gross_profit / gross_loss)
        avg_win = _safe(float(winners.mean())) if len(winners) else 0.0
        avg_loss = _safe(float(losers.mean())) if len(losers) else 0.0
        expectancy = _safe(win_rate * avg_win + (1 - win_rate) * avg_loss)
    else:
        total_trades = 0
        win_rate = profit_factor = avg_win = avg_loss = expectancy = 0.0

    # Turnover: fraction of portfolio rotated per bar, annualised
    turnover = _safe(float(returns.abs().sum()) / max(n_years, 1e-6))

    # Bars in market (non-zero position)
    bars_in_market = int(n_bars)  # refined by adapters if position series available

    return BacktestMetrics(
        total_return=total_return,
        bars_in_market=bars_in_market,
        performance=PerformanceMetrics(
            cagr=cagr,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            calmar_ratio=calmar,
        ),
        risk=RiskMetrics(
            max_drawdown=max_drawdown,
            max_drawdown_duration=dd_duration,
            var_95=var_95,
            cvar_95=cvar_95,
            var_99=var_99,
            cvar_99=cvar_99,
            volatility_annualised=vol,
        ),
        trading=TradingMetrics(
            total_trades=total_trades,
            win_rate=win_rate,
            profit_factor=profit_factor,
            avg_win=avg_win,
            avg_loss=avg_loss,
            expectancy=expectancy,
            turnover_annualised=turnover,
        ),
    )
