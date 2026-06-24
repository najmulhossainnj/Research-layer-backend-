"""
Strategy promotion endpoints.

  POST /strategies/{id}/promote  — promote a validated strategy
  POST /strategies/{id}/demote   — roll back a promoted strategy
  GET  /strategies/{id}/status   — current status + promotion readiness
"""
import uuid
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_db
from app.domain.strategy.orm import Strategy, StrategyStatus
from app.engines.promotion.service import PromotionService

router = APIRouter(prefix="/strategies", tags=["strategies"])


class PromoteRequest(BaseModel):
    confidence_override: Optional[float] = None


class DemoteRequest(BaseModel):
    reason: str = ""


@router.post("/{strategy_id}/promote")
async def promote_strategy(
    strategy_id: uuid.UUID,
    payload: PromoteRequest,
    db: AsyncSession = Depends(get_db),
):
    """
    Promote a validated strategy to the Portfolio Construction Layer.
    Publishes a StrategyValidated event to the event bus.
    """
    svc = PromotionService()
    result = await svc.promote(db, strategy_id, payload.confidence_override)
    if not result.success:
        raise HTTPException(status_code=409, detail=result.message)
    return {
        "success":     result.success,
        "message":     result.message,
        "payload_sent": result.payload,
    }


@router.post("/{strategy_id}/demote")
async def demote_strategy(
    strategy_id: uuid.UUID,
    payload: DemoteRequest,
    db: AsyncSession = Depends(get_db),
):
    """Roll back a promoted strategy to validated status."""
    svc = PromotionService()
    return await svc.demote(db, strategy_id, payload.reason)


@router.get("/{strategy_id}/status")
async def get_strategy_status(
    strategy_id: uuid.UUID, db: AsyncSession = Depends(get_db)
):
    """Return current status and promotion readiness checklist."""
    res = await db.execute(select(Strategy).where(Strategy.id == strategy_id))
    strategy = res.scalar_one_or_none()
    if not strategy:
        raise HTTPException(status_code=404, detail="Strategy not found")

    from app.domain.backtest.orm import Backtest, BacktestStatus
    bt_res = await db.execute(
        select(Backtest).where(
            Backtest.strategy_id == strategy_id,
            Backtest.status == BacktestStatus.COMPLETED,
        )
    )
    completed_backtests = list(bt_res.scalars().all())

    checklist = {
        "has_validated_status": strategy.status == StrategyStatus.VALIDATED,
        "has_completed_backtest": len(completed_backtests) > 0,
        "is_promoted": strategy.status == StrategyStatus.PROMOTED,
    }
    promotion_ready = (
        checklist["has_validated_status"] and checklist["has_completed_backtest"]
    )

    return {
        "strategy_id":    str(strategy_id),
        "name":           strategy.name,
        "status":         strategy.status,
        "promotion_ready": promotion_ready,
        "checklist":      checklist,
        "n_backtests":    len(completed_backtests),
    }
