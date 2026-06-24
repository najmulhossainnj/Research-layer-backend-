"""
Pydantic schemas for Experiment CRUD.
"""
import uuid
from datetime import datetime
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


class ExperimentCreate(BaseModel):
    strategy_id: uuid.UUID
    dataset_version: Optional[str] = None
    feature_version: Optional[str] = None
    model_version: Optional[str] = None
    git_commit_hash: Optional[str] = None
    mlflow_run_id: Optional[str] = None
    parameters: dict = Field(default_factory=dict)
    metrics: dict = Field(default_factory=dict)
    artifacts: dict = Field(default_factory=dict)


class ExperimentUpdate(BaseModel):
    mlflow_run_id: Optional[str] = None
    parameters: Optional[dict] = None
    metrics: Optional[dict] = None
    artifacts: Optional[dict] = None


class ExperimentRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    strategy_id: uuid.UUID
    dataset_version: Optional[str] = None
    feature_version: Optional[str] = None
    model_version: Optional[str] = None
    git_commit_hash: Optional[str] = None
    mlflow_run_id: Optional[str] = None
    parameters: dict = Field(default_factory=dict)
    metrics: dict = Field(default_factory=dict)
    artifacts: dict = Field(default_factory=dict)
    created_at: datetime
