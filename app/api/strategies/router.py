"""
Strategy CRUD endpoints.
"""
import uuid

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.crud_base import CRUDRepository
from app.db.session import get_db
from app.domain.strategy.orm import Strategy
from app.domain.strategy.schemas import StrategyCreate, StrategyRead, StrategyUpdate

router = APIRouter(prefix="/strategies", tags=["strategies"])
repo = CRUDRepository[Strategy, StrategyCreate, StrategyUpdate](Strategy)


@router.post("", response_model=StrategyRead, status_code=status.HTTP_201_CREATED)
async def create_strategy(payload: StrategyCreate, db: AsyncSession = Depends(get_db)):
    return await repo.create(db, payload)


@router.get("", response_model=list[StrategyRead])
async def list_strategies(
    skip: int = 0, limit: int = 100, db: AsyncSession = Depends(get_db)
):
    return await repo.list(db, skip=skip, limit=limit)


@router.get("/{strategy_id}", response_model=StrategyRead)
async def get_strategy(strategy_id: uuid.UUID, db: AsyncSession = Depends(get_db)):
    strategy = await repo.get(db, strategy_id)
    if not strategy:
        raise HTTPException(status_code=404, detail="Strategy not found")
    return strategy


@router.patch("/{strategy_id}", response_model=StrategyRead)
async def update_strategy(
    strategy_id: uuid.UUID, payload: StrategyUpdate, db: AsyncSession = Depends(get_db)
):
    strategy = await repo.get(db, strategy_id)
    if not strategy:
        raise HTTPException(status_code=404, detail="Strategy not found")
    return await repo.update(db, strategy, payload)


@router.delete("/{strategy_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_strategy(strategy_id: uuid.UUID, db: AsyncSession = Depends(get_db)):
    strategy = await repo.get(db, strategy_id)
    if not strategy:
        raise HTTPException(status_code=404, detail="Strategy not found")
    await repo.delete(db, strategy)
