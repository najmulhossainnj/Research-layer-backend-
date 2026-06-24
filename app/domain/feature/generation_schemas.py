"""
Schemas for feature generation endpoints.
"""
import uuid
from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class FeatureGenerateRequest(BaseModel):
    symbol: str
    timeframe: str = "1d"
    start_date: datetime
    end_date: datetime


class FeatureDatasetRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    feature_id: uuid.UUID
    symbol: str
    timeframe: str
    start_date: datetime
    end_date: datetime
    version_hash: str
    storage_uri: str
    row_count: int | None = None
    columns: list[str] = Field(default_factory=list)
    created_at: datetime


class FeatureGenerateResponse(BaseModel):
    dataset: FeatureDatasetRead
    preview: list[dict] = Field(
        default_factory=list, description="First few rows of the generated feature data"
    )
