"""
Backtest domain model.

Stores the configuration and result summary of a single backtest run
against a strategy. Large artifacts (trade lists, equity curves) are
referenced via storage_uri rather than inlined.
"""
import uuid
from typing import Optional

from sqlalchemy import ForeignKey, Numeric, String
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.mixins import TimestampMixin, UUIDPrimaryKeyMixin
from app.db.session import Base


class BacktestEngine:
    VECTORBT = "vectorbt"
    BACKTRADER = "backtrader"
    LEAN = "lean"


class BacktestStatus:
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class Backtest(Base, UUIDPrimaryKeyMixin, TimestampMixin):
    __tablename__ = "backtests"

    strategy_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("strategies.id"), nullable=False
    )

    engine: Mapped[str] = mapped_column(String(50), default=BacktestEngine.VECTORBT)
    status: Mapped[str] = mapped_column(String(20), default=BacktestStatus.PENDING)

    initial_capital: Mapped[Optional[float]] = mapped_column(Numeric(18, 2), nullable=True)
    commission: Mapped[Optional[float]] = mapped_column(Numeric(10, 6), nullable=True)
    slippage: Mapped[Optional[float]] = mapped_column(Numeric(10, 6), nullable=True)

    config: Mapped[dict] = mapped_column(JSONB, default=dict)  # spread, market impact, etc.
    metrics: Mapped[dict] = mapped_column(JSONB, default=dict)  # Sharpe, CAGR, drawdown...

    # Pointers rather than inline blobs
    trades_uri: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)
    equity_curve_uri: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)

    strategy: Mapped["Strategy"] = relationship(back_populates="backtests")
