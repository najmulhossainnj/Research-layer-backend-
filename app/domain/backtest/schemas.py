"""
Pydantic schemas for the Backtest resource.
"""
import uuid
from datetime import datetime
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


class BacktestBase(BaseModel):
    strategy_id: uuid.UUID
    engine: str = "vectorbt"  # vectorbt | backtrader | lean
    initial_capital: Optional[float] = None
    commission: Optional[float] = None
    slippage: Optional[float] = None
    config: dict = Field(default_factory=dict)


class BacktestCreate(BacktestBase):
    pass


class BacktestUpdate(BaseModel):
    status: Optional[str] = None
    metrics: Optional[dict] = None
    trades_uri: Optional[str] = None
    equity_curve_uri: Optional[str] = None


class BacktestRead(BacktestBase):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    status: str
    metrics: dict = Field(default_factory=dict)
    trades_uri: Optional[str] = None
    equity_curve_uri: Optional[str] = None
    created_at: datetime
    updated_at: datetime
