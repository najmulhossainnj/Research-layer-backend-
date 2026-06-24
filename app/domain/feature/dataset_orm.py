"""
FeatureDataset domain model.

A `Feature` row is a *definition* (plugin + params). A `FeatureDataset` row
is one *generated instance* of that definition over a specific symbol,
timeframe, and date range — content-addressed by a version hash so the
same definition + inputs always resolves to the same dataset (supporting
reproducibility and historical regeneration).
"""
import uuid
from datetime import datetime
from typing import Optional

from sqlalchemy import ForeignKey, Index, Integer, String
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.db.mixins import TimestampMixin, UUIDPrimaryKeyMixin
from app.db.session import Base


class FeatureDataset(Base, UUIDPrimaryKeyMixin, TimestampMixin):
    __tablename__ = "feature_datasets"
    __table_args__ = (
        Index(
            "ix_feature_datasets_lookup",
            "feature_id",
            "symbol",
            "timeframe",
            "version_hash",
        ),
    )

    feature_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("features.id"), nullable=False
    )

    symbol: Mapped[str] = mapped_column(String(50), nullable=False)
    timeframe: Mapped[str] = mapped_column(String(20), nullable=False)

    start_date: Mapped[datetime] = mapped_column(nullable=False)
    end_date: Mapped[datetime] = mapped_column(nullable=False)

    # Content hash of (plugin_key, params, symbol, timeframe, date range,
    # source data fingerprint). Identical inputs -> identical hash -> reuse.
    version_hash: Mapped[str] = mapped_column(String(64), nullable=False, index=True)

    storage_uri: Mapped[str] = mapped_column(String(500), nullable=False)
    row_count: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    columns: Mapped[list[str]] = mapped_column(JSONB, default=list)

    # Fingerprint of the underlying source data this was computed from,
    # so we can tell when regeneration is needed vs. safe to reuse.
    source_fingerprint: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
