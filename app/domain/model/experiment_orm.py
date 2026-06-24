"""
Experiment domain model.

Mirrors / extends an MLflow run with research-platform-specific lineage:
dataset version, feature version, model version, and the strategy it
belongs to. Enables side-by-side comparison in the Experiment Tracker UI.
"""
import uuid
from typing import Optional

from sqlalchemy import ForeignKey, String
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.mixins import TimestampMixin, UUIDPrimaryKeyMixin
from app.db.session import Base


class Experiment(Base, UUIDPrimaryKeyMixin, TimestampMixin):
    __tablename__ = "experiments"

    strategy_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("strategies.id"), nullable=False
    )

    dataset_version: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    feature_version: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    model_version: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    git_commit_hash: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)

    mlflow_run_id: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)

    parameters: Mapped[dict] = mapped_column(JSONB, default=dict)
    metrics: Mapped[dict] = mapped_column(JSONB, default=dict)
    artifacts: Mapped[dict] = mapped_column(JSONB, default=dict)

    strategy: Mapped["Strategy"] = relationship(back_populates="experiments")
