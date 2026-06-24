"""
Model domain model.

Represents a versioned, trained (or trainable) statistical/ML/DL model
configuration. Actual weights/artifacts live in MLflow's artifact store;
this row tracks identity, config, and lineage.
"""
from typing import Optional

from sqlalchemy import String
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.mixins import TimestampMixin, UUIDPrimaryKeyMixin, VersionMixin
from app.db.session import Base


class ModelFamily:
    STATISTICAL = "statistical"  # ARIMA, GARCH, Kalman Filter
    MACHINE_LEARNING = "machine_learning"  # RF, XGBoost, LightGBM, CatBoost
    DEEP_LEARNING = "deep_learning"  # LSTM, GRU, Transformer
    ENSEMBLE = "ensemble"  # Stacking, Voting, Blending


class MLModel(Base, UUIDPrimaryKeyMixin, TimestampMixin, VersionMixin):
    __tablename__ = "models"

    name: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    model_type: Mapped[str] = mapped_column(String(100), nullable=False)  # e.g. "xgboost"
    family: Mapped[str] = mapped_column(String(50), nullable=False)
    parameters: Mapped[dict] = mapped_column(JSONB, default=dict)

    # MLflow linkage
    mlflow_run_id: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    artifact_uri: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)

    metrics: Mapped[dict] = mapped_column(JSONB, default=dict)

    strategies: Mapped[list["Strategy"]] = relationship(back_populates="model")
