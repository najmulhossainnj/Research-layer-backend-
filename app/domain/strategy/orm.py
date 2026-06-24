"""
Strategy domain model.

A Strategy is the top-level research artifact: it references a set of
features, a model, and signal logic, and can be promoted to the
Portfolio Construction Layer once validated.
"""
import uuid
from typing import Optional

from sqlalchemy import ARRAY, ForeignKey, String, Text
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.mixins import TimestampMixin, UUIDPrimaryKeyMixin, VersionMixin
from app.db.session import Base


class StrategyStatus:
    DRAFT = "draft"
    BACKTESTED = "backtested"
    VALIDATED = "validated"
    PROMOTED = "promoted"
    ARCHIVED = "archived"


class Strategy(Base, UUIDPrimaryKeyMixin, TimestampMixin, VersionMixin):
    __tablename__ = "strategies"

    name: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    universe: Mapped[list[str]] = mapped_column(ARRAY(String), default=list)
    timeframe: Mapped[str] = mapped_column(String(20), default="1d")

    feature_ids: Mapped[list[uuid.UUID]] = mapped_column(
        ARRAY(UUID(as_uuid=True)), default=list
    )
    model_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True), ForeignKey("models.id"), nullable=True
    )
    signal_logic_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True), nullable=True
    )

    status: Mapped[str] = mapped_column(String(20), default=StrategyStatus.DRAFT)
    pipeline_config: Mapped[dict] = mapped_column(JSONB, default=dict)

    model: Mapped[Optional["MLModel"]] = relationship(back_populates="strategies")
    experiments: Mapped[list["Experiment"]] = relationship(back_populates="strategy")
    backtests: Mapped[list["Backtest"]] = relationship(back_populates="strategy")
