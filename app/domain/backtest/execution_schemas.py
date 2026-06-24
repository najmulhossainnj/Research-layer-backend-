"""
Schemas for backtest execution, results, and comparison.
"""
import uuid
from datetime import datetime
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


# ── Execution request ─────────────────────────────────────────────────────

class BacktestRunConfig(BaseModel):
    """Full execution config embedded in Backtest.config JSONB."""
    symbol: str
    timeframe: str = "1d"
    start_date: datetime
    end_date: datetime
    feature_ids: list[uuid.UUID] = Field(default_factory=list)
    model_id: Optional[uuid.UUID] = None
    signal_logic_id: Optional[uuid.UUID] = None
    signal_plugin_key: Optional[str] = None
    signal_plugin_params: dict = Field(default_factory=dict)
    bars_per_year: int = 252
    risk_free_rate: float = 0.0
    size_type: str = "percent"  # percent | shares | signal_weight


class BacktestExecuteRequest(BaseModel):
    """Passed to POST /backtests/{id}/execute — triggers actual computation."""
    async_mode: bool = False


class BacktestCreateRequest(BaseModel):
    """Combines the CRUD create fields + run config in one request."""
    strategy_id: uuid.UUID
    engine: str = "vectorbt"
    initial_capital: float = 100_000.0
    commission: float = 0.0005
    slippage: float = 0.0005
    config: BacktestRunConfig


# ── Metric sub-schemas (mirrors metrics engine dataclasses) ───────────────

class PerformanceMetricsSchema(BaseModel):
    cagr: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float


class RiskMetricsSchema(BaseModel):
    max_drawdown: float
    max_drawdown_duration: int
    var_95: float
    cvar_95: float
    var_99: float
    cvar_99: float
    volatility_annualised: float


class TradingMetricsSchema(BaseModel):
    total_trades: int
    win_rate: float
    profit_factor: float
    avg_win: float
    avg_loss: float
    expectancy: float
    turnover_annualised: float


class BacktestMetricsSummary(BaseModel):
    total_return: float
    bars_in_market: int
    performance: PerformanceMetricsSchema
    risk: RiskMetricsSchema
    trading: TradingMetricsSchema


# ── Result response ───────────────────────────────────────────────────────

class BacktestResultResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    strategy_id: uuid.UUID
    engine: str
    status: str
    initial_capital: Optional[float]
    commission: Optional[float]
    slippage: Optional[float]
    metrics: dict
    equity_curve_uri: Optional[str]
    trades_uri: Optional[str]
    created_at: datetime
    updated_at: datetime

    # Structured metrics populated when status == "completed"
    structured_metrics: Optional[BacktestMetricsSummary] = None


# ── Comparison ────────────────────────────────────────────────────────────

class BacktestCompareRequest(BaseModel):
    backtest_ids: list[uuid.UUID] = Field(..., min_length=2, max_length=10)


class BacktestCompareRow(BaseModel):
    id: uuid.UUID
    engine: str
    status: str
    metrics: dict


class MetricDiff(BaseModel):
    values: dict       # backtest_id (str) → metric value
    best_id: Optional[str]
    higher_is_better: bool


class BacktestCompareResponse(BaseModel):
    runs: list[BacktestCompareRow]
    metric_diff: dict[str, MetricDiff]
