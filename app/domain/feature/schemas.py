"""
Pydantic schemas for the Feature resource.
"""
import uuid
from datetime import datetime
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


class FeatureBase(BaseModel):
    name: str = Field(..., max_length=255)
    type: str  # technical | statistical | automated | news | fundamental | macro
    description: Optional[str] = None
    parameters: dict = Field(default_factory=dict)
    plugin_key: str


class FeatureCreate(FeatureBase):
    pass


class FeatureUpdate(BaseModel):
    name: Optional[str] = None
    type: Optional[str] = None
    description: Optional[str] = None
    parameters: Optional[dict] = None
    plugin_key: Optional[str] = None
    storage_uri: Optional[str] = None


class FeatureRead(FeatureBase):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    version: int
    storage_uri: Optional[str] = None
    created_at: datetime
    updated_at: datetime
