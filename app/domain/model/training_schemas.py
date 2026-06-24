"""
Schemas for model training endpoints.
"""
import uuid
from datetime import datetime

from pydantic import BaseModel, Field

from app.engines.model_training_engine.cross_validation import CVConfig


class DatasetSpec(BaseModel):
    """Identifies what to train on: a symbol/timeframe/date range plus the
    feature definitions to assemble into X, and a forward-return target."""

    feature_ids: list[uuid.UUID]
    symbol: str
    timeframe: str = "1d"
    start_date: datetime
    end_date: datetime
    target_horizon: int = Field(1, description="Bars ahead for the forward-return target")


class CVConfigSchema(BaseModel):
    method: str = "rolling"
    n_splits: int = 5
    test_size: float = 0.15
    min_train_size: float = 0.2

    def to_config(self) -> CVConfig:
        return CVConfig(**self.model_dump())


class TrainRequest(BaseModel):
    dataset: DatasetSpec
    cv: CVConfigSchema = Field(default_factory=CVConfigSchema)


class TrainResponse(BaseModel):
    model_id: uuid.UUID
    artifact_uri: str | None
    cv_metrics: dict
    n_train_rows: int
    feature_columns: list[str]


class ParamSpecSchema(BaseModel):
    type: str  # "float" | "int" | "categorical"
    low: float | None = None
    high: float | None = None
    log: bool = False
    choices: list[str] | None = None


class TuneRequest(BaseModel):
    dataset: DatasetSpec
    plugin_key: str
    param_space: dict[str, ParamSpecSchema]
    n_trials: int = 30
    cv: CVConfigSchema = Field(default_factory=CVConfigSchema)
    metric: str = "mean_mse"
    direction: str = "minimize"


class TuneResponse(BaseModel):
    best_params: dict
    best_score: float
    n_trials: int


class AutoMLRequest(BaseModel):
    dataset: DatasetSpec
    candidates: dict[str, dict] = Field(
        description="plugin_key -> fixed params to evaluate, e.g. {'ml.xgboost': {'max_depth': 6}}"
    )
    cv: CVConfigSchema = Field(default_factory=CVConfigSchema)
    metric: str = "mean_mse"


class AutoMLCandidateResponse(BaseModel):
    plugin_key: str
    params: dict
    score: float
    metrics: dict


class AutoMLResponse(BaseModel):
    leaderboard: list[AutoMLCandidateResponse]
