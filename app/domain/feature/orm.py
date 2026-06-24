"""
Feature domain model.

Represents a versioned feature definition (technical, statistical,
automated/tsfresh, or alternative-data) managed by the Feature Store.
"""
from typing import Optional

from sqlalchemy import String, Text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from app.db.mixins import TimestampMixin, UUIDPrimaryKeyMixin, VersionMixin
from app.db.session import Base


class FeatureType:
    TECHNICAL = "technical"
    STATISTICAL = "statistical"
    AUTOMATED = "automated"
    NEWS = "news"
    FUNDAMENTAL = "fundamental"
    MACRO = "macro"


class Feature(Base, UUIDPrimaryKeyMixin, TimestampMixin, VersionMixin):
    __tablename__ = "features"

    name: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    type: Mapped[str] = mapped_column(String(50), nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    parameters: Mapped[dict] = mapped_column(JSONB, default=dict)

    # Plugin identifier resolved through the feature plugin registry
    plugin_key: Mapped[str] = mapped_column(String(100), nullable=False)

    # Pointer to where generated feature data lives in the feature store (S3/MinIO key prefix)
    storage_uri: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)
