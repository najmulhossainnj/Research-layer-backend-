"""
Feature CRUD endpoints.
"""
import uuid

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.crud_base import CRUDRepository
from app.db.session import get_db
from app.domain.feature.orm import Feature
from app.domain.feature.schemas import FeatureCreate, FeatureRead, FeatureUpdate

router = APIRouter(prefix="/features", tags=["features"])
repo = CRUDRepository[Feature, FeatureCreate, FeatureUpdate](Feature)


@router.post("", response_model=FeatureRead, status_code=status.HTTP_201_CREATED)
async def create_feature(payload: FeatureCreate, db: AsyncSession = Depends(get_db)):
    return await repo.create(db, payload)


@router.get("", response_model=list[FeatureRead])
async def list_features(skip: int = 0, limit: int = 100, db: AsyncSession = Depends(get_db)):
    return await repo.list(db, skip=skip, limit=limit)


@router.get("/{feature_id}", response_model=FeatureRead)
async def get_feature(feature_id: uuid.UUID, db: AsyncSession = Depends(get_db)):
    feature = await repo.get(db, feature_id)
    if not feature:
        raise HTTPException(status_code=404, detail="Feature not found")
    return feature


@router.patch("/{feature_id}", response_model=FeatureRead)
async def update_feature(
    feature_id: uuid.UUID, payload: FeatureUpdate, db: AsyncSession = Depends(get_db)
):
    feature = await repo.get(db, feature_id)
    if not feature:
        raise HTTPException(status_code=404, detail="Feature not found")
    return await repo.update(db, feature, payload)


@router.delete("/{feature_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_feature(feature_id: uuid.UUID, db: AsyncSession = Depends(get_db)):
    feature = await repo.get(db, feature_id)
    if not feature:
        raise HTTPException(status_code=404, detail="Feature not found")
    await repo.delete(db, feature)
