"""
Validation endpoints.

  POST /validation/walk-forward        — sync walk-forward analysis
  POST /validation/walk-forward/async  — Celery dispatch
  GET  /validation/strategies/{id}     — validation history for a strategy
"""
import uuid
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_db
from app.engines.validation_engine.engine import ValidationConfig, ValidationEngine
from app.engines.validation_engine.walk_forward import WalkForwardConfig

router = APIRouter(prefix="/validation", tags=["validation"])


# ── Request / response schemas ────────────────────────────────────────────

class WalkForwardConfigSchema(BaseModel):
    method: str = "rolling"
    n_splits: int = 5
    test_size: float = 0.2
    min_train_size: float = 0.3
    gap_bars: int = 0
    refit: bool = True

    def to_config(self) -> WalkForwardConfig:
        from app.engines.validation_engine.walk_forward import WalkForwardConfig
        return WalkForwardConfig(**self.model_dump())


class ValidationConfigSchema(BaseModel):
    wf: WalkForwardConfigSchema = Field(default_factory=WalkForwardConfigSchema)
    min_sharpe: float = 0.3
    max_drawdown: float = 0.25
    profitable_fold_ratio: float = 0.5
    max_overfit_ratio: float = 3.0

    def to_config(self) -> ValidationConfig:
        return ValidationConfig(
            wf=self.wf.to_config(),
            min_sharpe=self.min_sharpe,
            max_drawdown=self.max_drawdown,
            profitable_fold_ratio=self.profitable_fold_ratio,
            max_overfit_ratio=self.max_overfit_ratio,
        )


class WalkForwardRequest(BaseModel):
    strategy_id: uuid.UUID
    plugin_key: str
    model_params: dict = Field(default_factory=dict)
    feature_ids: list[uuid.UUID]
    symbol: str
    timeframe: str = "1d"
    start_date: datetime
    end_date: datetime
    target_horizon: int = 1
    bars_per_year: int = 252
    config: ValidationConfigSchema = Field(default_factory=ValidationConfigSchema)


class GateResult(BaseModel):
    passed: bool
    value: float
    threshold: float


class WalkForwardFoldSchema(BaseModel):
    fold_idx: int
    train_start: int
    train_end: int
    test_start: int
    test_end: int
    oos_metrics: dict
    is_metrics: dict


class WalkForwardResponse(BaseModel):
    strategy_id: uuid.UUID
    passed: bool
    gate_results: dict
    aggregate: dict
    folds: list[WalkForwardFoldSchema]
    is_sharpe_stable: bool
    is_profitable_oos: bool
    overfitting_score: float
    mlflow_run_id: Optional[str]


# ── Endpoints ─────────────────────────────────────────────────────────────

@router.post("/walk-forward", response_model=WalkForwardResponse)
async def run_walk_forward(
    payload: WalkForwardRequest, db: AsyncSession = Depends(get_db)
):
    engine = ValidationEngine()
    try:
        result = await engine.validate_strategy(
            db=db,
            strategy_id=payload.strategy_id,
            plugin_key=payload.plugin_key,
            model_params=payload.model_params,
            feature_ids=payload.feature_ids,
            symbol=payload.symbol,
            timeframe=payload.timeframe,
            start_date=payload.start_date,
            end_date=payload.end_date,
            target_horizon=payload.target_horizon,
            config=payload.config.to_config(),
            bars_per_year=payload.bars_per_year,
        )
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    wf = result.walk_forward
    return WalkForwardResponse(
        strategy_id=payload.strategy_id,
        passed=result.passed,
        gate_results=result.gate_results,
        aggregate=wf.aggregate,
        folds=[
            WalkForwardFoldSchema(
                fold_idx=f.fold_idx,
                train_start=f.train_start,
                train_end=f.train_end,
                test_start=f.test_start,
                test_end=f.test_end,
                oos_metrics=f.oos_metrics,
                is_metrics=f.is_metrics,
            )
            for f in wf.folds
        ],
        is_sharpe_stable=wf.is_sharpe_stable,
        is_profitable_oos=wf.is_profitable_oos,
        overfitting_score=wf.overfitting_score,
        mlflow_run_id=result.mlflow_run_id,
    )


@router.post("/walk-forward/async")
async def run_walk_forward_async(payload: WalkForwardRequest):
    """Dispatch walk-forward validation as a Celery task."""
    from app.workers.validation_tasks import walk_forward_task

    task = walk_forward_task.delay(payload.model_dump(mode="json"))
    return {"task_id": task.id, "status": "PENDING"}



class CPCVConfigSchema(BaseModel):
    n_splits: int = 6
    n_test_splits: int = 2
    embargo_pct: float = 0.01
    purge: bool = True


class CPCVRequest(BaseModel):
    strategy_id: uuid.UUID
    plugin_key: str
    model_params: dict = Field(default_factory=dict)
    feature_ids: list[uuid.UUID]
    symbol: str
    timeframe: str = "1d"
    start_date: datetime
    end_date: datetime
    target_horizon: int = 1
    bars_per_year: int = 252
    cpcv: CPCVConfigSchema = Field(default_factory=CPCVConfigSchema)
    max_pbo: float = 0.6
    min_deflated_sharpe: float = 0.1
    min_sharpe: float = 0.2


@router.post("/cpcv")
async def run_cpcv(payload: CPCVRequest, db: AsyncSession = Depends(get_db)):
    """Run CPCV with PBO and Deflated Sharpe gating."""
    from app.engines.validation_engine.cpcv import CPCVConfig
    from app.engines.validation_engine.engine import CPCVValidationEngine

    cfg = CPCVConfig(
        n_splits=payload.cpcv.n_splits,
        n_test_splits=payload.cpcv.n_test_splits,
        embargo_pct=payload.cpcv.embargo_pct,
        purge=payload.cpcv.purge,
        target_horizon=payload.target_horizon,
    )
    engine = CPCVValidationEngine()
    try:
        result = await engine.validate(
            db=db,
            strategy_id=payload.strategy_id,
            plugin_key=payload.plugin_key,
            model_params=payload.model_params,
            feature_ids=payload.feature_ids,
            symbol=payload.symbol,
            timeframe=payload.timeframe,
            start_date=payload.start_date,
            end_date=payload.end_date,
            target_horizon=payload.target_horizon,
            cpcv_config=cfg,
            max_pbo=payload.max_pbo,
            min_deflated_sharpe=payload.min_deflated_sharpe,
            min_sharpe=payload.min_sharpe,
            bars_per_year=payload.bars_per_year,
        )
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    return {
        "strategy_id": str(payload.strategy_id),
        "passed": result.passed,
        "gate_results": result.gate_results,
        "mlflow_run_id": result.mlflow_run_id,
        **result.cpcv.to_dict(),
    }


@router.post("/cpcv/async")
async def run_cpcv_async(payload: CPCVRequest):
    from app.workers.validation_tasks import cpcv_task
    task = cpcv_task.delay(payload.model_dump(mode="json"))
    return {"task_id": task.id, "status": "PENDING"}

@router.get("/strategies/{strategy_id}")
async def get_strategy_validation_history(
    strategy_id: uuid.UUID, db: AsyncSession = Depends(get_db)
):
    """Return all validation experiment rows for a strategy (via Experiment table)."""
    from sqlalchemy import select
    from app.domain.model.experiment_orm import Experiment
    from app.core.mlflow_client import get_run

    result = await db.execute(
        select(Experiment).where(Experiment.strategy_id == strategy_id)
    )
    experiments = list(result.scalars().all())

    history = []
    for exp in experiments:
        entry = {
            "experiment_id": str(exp.id),
            "mlflow_run_id": exp.mlflow_run_id,
            "parameters": exp.parameters,
            "metrics": exp.metrics,
            "created_at": exp.created_at.isoformat(),
        }
        # Enrich with run_type tag from MLflow if available
        if exp.mlflow_run_id:
            run = get_run(exp.mlflow_run_id)
            if run:
                entry["run_type"] = run.data.tags.get("run_type", "unknown")
        history.append(entry)

    return {"strategy_id": str(strategy_id), "history": history}
