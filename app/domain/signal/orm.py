"""
SignalLogic domain model.

A `SignalLogic` row stores the serialized rule tree produced by the visual
Signal Builder UI. The rule tree is a recursive JSON structure of
conditions and combinators — evaluated at runtime by the rule engine in
`app/engines/signal_engine/rule_engine.py`.

Example rule tree (JSON):
{
  "combinator": "AND",
  "rules": [
    {"field": "prediction", "operator": ">", "value": 0.7},
    {"field": "sentiment",  "operator": ">", "value": 0.5}
  ],
  "action": "BUY"
}

Multiple top-level rule groups (AND/OR) map to different actions (BUY /
SELL / HOLD), evaluated in declaration order — first match wins.
"""
import uuid
from typing import Optional

from sqlalchemy import ForeignKey, JSON, String
from sqlalchemy.orm import Mapped, mapped_column

from app.db.mixins import TimestampMixin, UUIDPrimaryKeyMixin, VersionMixin
from app.db.session import Base


class SignalLogic(Base, UUIDPrimaryKeyMixin, TimestampMixin, VersionMixin):
    __tablename__ = "signal_logic"

    name: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    description: Mapped[Optional[str]] = mapped_column(String(1000), nullable=True)

    # The full rule tree from the Signal Builder, stored as JSON for SQLite compatibility.
    rule_tree: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)

    # Output mode: "discrete" (BUY/SELL/HOLD), "numeric" (+1/0/-1),
    # "score" (raw prediction pass-through)
    output_mode: Mapped[str] = mapped_column(String(20), default="discrete")

    # long_only | long_short | portfolio
    position_mode: Mapped[str] = mapped_column(String(20), default="long_short")

    # SQLite-compatible: store UUID as string
    strategy_id: Mapped[Optional[str]] = mapped_column(
        String(36), ForeignKey("strategies.id"), nullable=True
    )
