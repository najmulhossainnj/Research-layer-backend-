"""
Pydantic schemas for the Model resource.
"""
import uuid
from datetime import datetime
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


class ModelBase(BaseModel):
    name: str = Field(..., max_length=255)
    model_type: str  # e.g. "xgboost", "lstm", "arima"
    family: str  # statistical | machine_learning | deep_learning | ensemble
    parameters: dict = Field(default_factory=dict)


class ModelCreate(ModelBase):
    pass


class ModelUpdate(BaseModel):
    name: Optional[str] = None
    model_type: Optional[str] = None
    family: Optional[str] = None
    parameters: Optional[dict] = None
    mlflow_run_id: Optional[str] = None
    artifact_uri: Optional[str] = None
    metrics: Optional[dict] = None


class ModelRead(ModelBase):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    version: int
    mlflow_run_id: Optional[str] = None
    artifact_uri: Optional[str] = None
    metrics: dict = Field(default_factory=dict)
    created_at: datetime
    updated_at: datetime
