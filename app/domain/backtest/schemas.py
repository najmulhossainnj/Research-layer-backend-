"""
Pydantic schemas for the Backtest resource.
"""
import uuid
from datetime import datetime, timezone
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator


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
    updated_at: Optional[datetime] = None

    @model_validator(mode='after')
    def set_updated_at_default(self):
        if self.updated_at is None:
            self.updated_at = datetime.now(timezone.utc)
        return self
