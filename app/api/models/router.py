"""
Model CRUD endpoints.
"""
import uuid

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.crud_base import CRUDRepository
from app.db.session import get_db
from app.domain.model.orm import MLModel
from app.domain.model.schemas import ModelCreate, ModelRead, ModelUpdate

router = APIRouter(prefix="/models", tags=["models"])
repo = CRUDRepository[MLModel, ModelCreate, ModelUpdate](MLModel)


@router.post("", response_model=ModelRead, status_code=status.HTTP_201_CREATED)
async def create_model(payload: ModelCreate, db: AsyncSession = Depends(get_db)):
    return await repo.create(db, payload)


@router.get("", response_model=list[ModelRead])
async def list_models(skip: int = 0, limit: int = 100, db: AsyncSession = Depends(get_db)):
    return await repo.list(db, skip=skip, limit=limit)


@router.get("/{model_id}", response_model=ModelRead)
async def get_model(model_id: uuid.UUID, db: AsyncSession = Depends(get_db)):
    model = await repo.get(db, model_id)
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")
    return model


@router.patch("/{model_id}", response_model=ModelRead)
async def update_model(
    model_id: uuid.UUID, payload: ModelUpdate, db: AsyncSession = Depends(get_db)
):
    model = await repo.get(db, model_id)
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")
    return await repo.update(db, model, payload)


@router.delete("/{model_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_model(model_id: uuid.UUID, db: AsyncSession = Depends(get_db)):
    model = await repo.get(db, model_id)
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")
    await repo.delete(db, model)
