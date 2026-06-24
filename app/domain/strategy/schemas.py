"""
Pydantic schemas for the Strategy resource.
"""
import uuid
from datetime import datetime
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


class StrategyBase(BaseModel):
    name: str = Field(..., max_length=255)
    description: Optional[str] = None
    universe: list[str] = Field(default_factory=list)
    timeframe: str = "1d"
    feature_ids: list[uuid.UUID] = Field(default_factory=list)
    model_id: Optional[uuid.UUID] = None
    signal_logic_id: Optional[uuid.UUID] = None
    pipeline_config: dict = Field(default_factory=dict)


class StrategyCreate(StrategyBase):
    pass


class StrategyUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    universe: Optional[list[str]] = None
    timeframe: Optional[str] = None
    feature_ids: Optional[list[uuid.UUID]] = None
    model_id: Optional[uuid.UUID] = None
    signal_logic_id: Optional[uuid.UUID] = None
    pipeline_config: Optional[dict] = None
    status: Optional[str] = None


class StrategyRead(StrategyBase):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    status: str
    version: int
    created_at: datetime
    updated_at: datetime
