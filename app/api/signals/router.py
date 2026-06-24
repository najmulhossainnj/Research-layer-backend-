"""
Signal endpoints.

  SignalLogic CRUD  — store/retrieve rule trees from the Signal Builder
  Generation        — run a model + rule tree (or plugin) over a date range
  Validation        — dry-run a rule tree against sample data without a model
  Plugins           — list available signal generator plugins
"""
import uuid
from datetime import datetime
from typing import Optional

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.crud_base import CRUDRepository
from app.db.session import get_db
from app.domain.feature.orm import Feature
from app.domain.model.orm import MLModel
from app.domain.model.schemas import ModelCreate, ModelUpdate
from app.domain.signal.orm import SignalLogic
from app.domain.signal.schemas import (
    SignalGenerateRequest,
    SignalGenerateResponse,
    SignalLogicCreate,
    SignalLogicRead,
    SignalLogicUpdate,
    SignalSummary,
)
from app.engines.feature_engine.market_data_client import get_market_data_client
from app.engines.feature_engine.pipeline import FeaturePipeline
from app.engines.signal_engine.pipeline import SignalPipeline
from app.engines.signal_engine.rule_engine import evaluate_rule_tree

router = APIRouter(prefix="/signals", tags=["signals"])

_signal_repo = CRUDRepository[SignalLogic, SignalLogicCreate, SignalLogicUpdate](SignalLogic)
_model_repo = CRUDRepository[MLModel, ModelCreate, ModelUpdate](MLModel)


# ── SignalLogic CRUD ──────────────────────────────────────────────────────

@router.post("", response_model=SignalLogicRead, status_code=status.HTTP_201_CREATED)
async def create_signal_logic(
    payload: SignalLogicCreate, db: AsyncSession = Depends(get_db)
):
    return await _signal_repo.create(db, payload)


@router.get("", response_model=list[SignalLogicRead])
async def list_signal_logic(
    strategy_id: Optional[uuid.UUID] = None,
    skip: int = 0,
    limit: int = 100,
    db: AsyncSession = Depends(get_db),
):
    if strategy_id:
        result = await db.execute(
            select(SignalLogic)
            .where(SignalLogic.strategy_id == strategy_id)
            .offset(skip)
            .limit(limit)
        )
        return list(result.scalars().all())
    return await _signal_repo.list(db, skip=skip, limit=limit)


@router.get("/{signal_id}", response_model=SignalLogicRead)
async def get_signal_logic(signal_id: uuid.UUID, db: AsyncSession = Depends(get_db)):
    sl = await _signal_repo.get(db, signal_id)
    if not sl:
        raise HTTPException(status_code=404, detail="SignalLogic not found")
    return sl


@router.patch("/{signal_id}", response_model=SignalLogicRead)
async def update_signal_logic(
    signal_id: uuid.UUID, payload: SignalLogicUpdate, db: AsyncSession = Depends(get_db)
):
    sl = await _signal_repo.get(db, signal_id)
    if not sl:
        raise HTTPException(status_code=404, detail="SignalLogic not found")
    return await _signal_repo.update(db, sl, payload)


@router.delete("/{signal_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_signal_logic(signal_id: uuid.UUID, db: AsyncSession = Depends(get_db)):
    sl = await _signal_repo.get(db, signal_id)
    if not sl:
        raise HTTPException(status_code=404, detail="SignalLogic not found")
    await _signal_repo.delete(db, sl)


# ── Rule-tree validation (dry-run, no model required) ─────────────────────

@router.post("/validate-rule-tree")
async def validate_rule_tree(payload: dict):
    """
    Dry-run a rule tree against synthetic sample data to catch field/operator
    errors before wiring it to a real model. Returns any validation errors.

    Send: {"rule_tree": [...], "sample_fields": ["prediction", "sentiment"]}
    """
    rule_tree = payload.get("rule_tree", [])
    sample_fields = payload.get("sample_fields", ["prediction"])

    if not rule_tree:
        raise HTTPException(status_code=422, detail="rule_tree must not be empty")

    # Build a single synthetic row at boundary values to exercise every branch
    sample = pd.DataFrame(
        [{field: 0.0 for field in sample_fields}]
    )

    errors = []
    try:
        evaluate_rule_tree(
            df=sample,
            rule_tree=rule_tree if isinstance(rule_tree, list) else [rule_tree],
        )
    except (KeyError, ValueError) as e:
        errors.append(str(e))

    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "rule_group_count": len(rule_tree) if isinstance(rule_tree, list) else 1,
    }


# ── Signal generation ─────────────────────────────────────────────────────

@router.post("/generate", response_model=SignalGenerateResponse)
async def generate_signals(
    payload: SignalGenerateRequest, db: AsyncSession = Depends(get_db)
):
    if not payload.signal_logic_id and not payload.plugin_key:
        raise HTTPException(
            status_code=422,
            detail="Provide either signal_logic_id (rule tree) or plugin_key",
        )
    if payload.signal_logic_id and payload.plugin_key:
        raise HTTPException(
            status_code=422,
            detail="Provide signal_logic_id OR plugin_key, not both",
        )

    # Load model
    model = await _model_repo.get(db, payload.model_id)
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")

    # Load features and build feature DataFrame
    result = await db.execute(
        select(Feature).where(Feature.id.in_(payload.feature_ids))
    )
    features = list(result.scalars().all())
    if len(features) != len(payload.feature_ids):
        raise HTTPException(status_code=404, detail="One or more feature IDs not found")

    ohlcv = await get_market_data_client().get_ohlcv(
        payload.symbol, payload.timeframe, payload.start_date, payload.end_date
    )
    if ohlcv.empty:
        raise HTTPException(status_code=422, detail="No OHLCV data returned for that range")

    feature_pipeline = FeaturePipeline()
    feature_df = await feature_pipeline.run_many(
        db=db,
        features=features,
        market_data=ohlcv,
        symbol=payload.symbol,
        timeframe=payload.timeframe,
        start_date=payload.start_date,
        end_date=payload.end_date,
        join=True,
    )

    signal_pipeline = SignalPipeline()

    if payload.signal_logic_id:
        signal_result = await signal_pipeline.generate_from_rule_tree(
            db=db,
            signal_logic_id=payload.signal_logic_id,
            model=model,
            feature_data=feature_df,
        )
    else:
        signal_result = await signal_pipeline.generate_from_plugin(
            model=model,
            feature_data=feature_df,
            plugin_key=payload.plugin_key,
            plugin_params=payload.plugin_params,
        )

    signals = signal_result.signals
    preds = signal_result.predictions

    # Build summary (handles both discrete and numeric signals)
    discrete = signals.map(
        lambda x: x if isinstance(x, str) else ("BUY" if x > 0 else ("SELL" if x < 0 else "HOLD"))
    )
    buy_count = int((discrete == "BUY").sum())
    sell_count = int((discrete == "SELL").sum())
    hold_count = int((discrete == "HOLD").sum())
    total = len(signals)

    summary = SignalSummary(
        total_bars=total,
        buy_count=buy_count,
        sell_count=sell_count,
        hold_count=hold_count,
        signal_rate=round((buy_count + sell_count) / total, 4) if total else 0.0,
    )

    # Preview: join signal onto predictions for the first 20 rows
    preview_df = preds.copy()
    preview_df["signal"] = signals
    preview = preview_df.head(20).reset_index().to_dict(orient="records")

    return SignalGenerateResponse(
        signal_logic_id=payload.signal_logic_id,
        plugin_key=payload.plugin_key,
        model_id=payload.model_id,
        symbol=payload.symbol,
        metadata=signal_result.metadata,
        summary=summary,
        preview=preview,
    )


@router.post("/generate/async")
async def generate_signals_async(
    payload: SignalGenerateRequest, db: AsyncSession = Depends(get_db)
):
    """Dispatch signal generation as a Celery background task."""
    from app.workers.signal_tasks import generate_signals_task

    task = generate_signals_task.delay(
        model_id=str(payload.model_id),
        feature_ids=[str(fid) for fid in payload.feature_ids],
        symbol=payload.symbol,
        timeframe=payload.timeframe,
        start_date=payload.start_date.isoformat(),
        end_date=payload.end_date.isoformat(),
        signal_logic_id=str(payload.signal_logic_id) if payload.signal_logic_id else None,
        plugin_key=payload.plugin_key,
        plugin_params=payload.plugin_params,
        target_horizon=payload.target_horizon,
    )
    return {"task_id": task.id, "status": "PENDING"}


# ── Plugin listing ─────────────────────────────────────────────────────────

@router.get("/plugins/available")
async def list_signal_plugins():
    from app.plugins.signals import signal_registry
    return {"plugins": signal_registry.list_keys()}
