"""
Backtest endpoints (Phase 5 — full implementation).

  CRUD        — create/list/get/update/delete Backtest rows
  Execute     — trigger a real backtest run (sync or async)
  Results     — structured metrics, equity curve, trades
  Compare     — side-by-side metric diff across up to 10 runs
  Engines     — list available backtest engines
"""
import io
import uuid

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.crud_base import CRUDRepository
from app.db.session import get_db
from app.domain.backtest.execution_schemas import (
    BacktestCompareRequest,
    BacktestCompareResponse,
    BacktestCompareRow,
    BacktestCreateRequest,
    BacktestExecuteRequest,
    BacktestMetricsSummary,
    BacktestResultResponse,
    MetricDiff,
    PerformanceMetricsSchema,
    RiskMetricsSchema,
    TradingMetricsSchema,
)
from app.domain.backtest.orm import Backtest, BacktestStatus
from app.domain.backtest.schemas import BacktestCreate, BacktestRead, BacktestUpdate
from app.engines.backtest_engine.engine_registry import list_engines
from app.engines.backtest_engine.pipeline import BacktestPipeline
from app.engines.backtest_engine.storage import load_equity_curve, load_trades

router = APIRouter(prefix="/backtests", tags=["backtests"])
_repo = CRUDRepository[Backtest, BacktestCreate, BacktestUpdate](Backtest)

_HIGHER_IS_BETTER = {
    "perf_cagr", "perf_sharpe_ratio", "perf_sortino_ratio", "perf_calmar_ratio",
    "total_return", "trade_win_rate", "trade_profit_factor", "trade_expectancy",
    "bars_in_market",
}


# ── CRUD ──────────────────────────────────────────────────────────────────

@router.post("", response_model=BacktestRead, status_code=status.HTTP_201_CREATED)
async def create_backtest(payload: BacktestCreateRequest, db: AsyncSession = Depends(get_db)):
    create = BacktestCreate(
        strategy_id=payload.strategy_id,
        engine=payload.engine,
        initial_capital=payload.initial_capital,
        commission=payload.commission,
        slippage=payload.slippage,
        config=payload.config.model_dump(mode="json"),
    )
    return await _repo.create(db, create)


@router.get("", response_model=list[BacktestRead])
async def list_backtests(
    strategy_id: uuid.UUID | None = None,
    skip: int = 0,
    limit: int = 100,
    db: AsyncSession = Depends(get_db),
):
    if strategy_id:
        result = await db.execute(
            select(Backtest)
            .where(Backtest.strategy_id == strategy_id)
            .offset(skip)
            .limit(limit)
        )
        return list(result.scalars().all())
    return await _repo.list(db, skip=skip, limit=limit)


@router.get("/{backtest_id}", response_model=BacktestResultResponse)
async def get_backtest(backtest_id: uuid.UUID, db: AsyncSession = Depends(get_db)):
    bt = await _repo.get(db, backtest_id)
    if not bt:
        raise HTTPException(status_code=404, detail="Backtest not found")
    return _enrich_response(bt)


@router.patch("/{backtest_id}", response_model=BacktestRead)
async def update_backtest(
    backtest_id: uuid.UUID, payload: BacktestUpdate, db: AsyncSession = Depends(get_db)
):
    bt = await _repo.get(db, backtest_id)
    if not bt:
        raise HTTPException(status_code=404, detail="Backtest not found")
    return await _repo.update(db, bt, payload)


@router.delete("/{backtest_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_backtest(backtest_id: uuid.UUID, db: AsyncSession = Depends(get_db)):
    bt = await _repo.get(db, backtest_id)
    if not bt:
        raise HTTPException(status_code=404, detail="Backtest not found")
    await _repo.delete(db, bt)


# ── Execution ─────────────────────────────────────────────────────────────

@router.post("/{backtest_id}/execute", response_model=BacktestResultResponse)
async def execute_backtest(
    backtest_id: uuid.UUID,
    payload: BacktestExecuteRequest,
    db: AsyncSession = Depends(get_db),
):
    bt = await _repo.get(db, backtest_id)
    if not bt:
        raise HTTPException(status_code=404, detail="Backtest not found")
    if bt.status == BacktestStatus.RUNNING:
        raise HTTPException(status_code=409, detail="Backtest is already running")

    if payload.async_mode:
        from app.workers.backtest_tasks import execute_backtest_task
        task = execute_backtest_task.delay(str(backtest_id))
        return {"task_id": task.id, "status": "PENDING"}

    pipeline = BacktestPipeline()
    updated_bt, _ = await pipeline.execute(db, bt)
    return _enrich_response(updated_bt)


@router.post("/{backtest_id}/execute/async")
async def execute_backtest_async(
    backtest_id: uuid.UUID, db: AsyncSession = Depends(get_db)
):
    bt = await _repo.get(db, backtest_id)
    if not bt:
        raise HTTPException(status_code=404, detail="Backtest not found")
    if bt.status == BacktestStatus.RUNNING:
        raise HTTPException(status_code=409, detail="Backtest already running")

    from app.workers.backtest_tasks import execute_backtest_task
    task = execute_backtest_task.delay(str(backtest_id))
    return {"task_id": task.id, "status": "PENDING"}


# ── Result downloads ──────────────────────────────────────────────────────

@router.get("/{backtest_id}/equity-curve")
async def get_equity_curve(backtest_id: uuid.UUID, db: AsyncSession = Depends(get_db)):
    bt = await _require_completed(backtest_id, db)
    equity = load_equity_curve(bt)
    df = equity.reset_index()
    df.columns = ["date", "equity"]
    df["date"] = df["date"].astype(str)
    return df.to_dict(orient="records")


@router.get("/{backtest_id}/equity-curve/parquet")
async def download_equity_curve_parquet(
    backtest_id: uuid.UUID, db: AsyncSession = Depends(get_db)
):
    bt = await _require_completed(backtest_id, db)
    equity = load_equity_curve(bt)
    buf = io.BytesIO()
    equity.to_frame(name="equity").to_parquet(buf, index=True)
    buf.seek(0)
    return StreamingResponse(
        buf,
        media_type="application/octet-stream",
        headers={"Content-Disposition": f"attachment; filename=equity_{backtest_id}.parquet"},
    )


@router.get("/{backtest_id}/trades")
async def get_trades(backtest_id: uuid.UUID, db: AsyncSession = Depends(get_db)):
    bt = await _require_completed(backtest_id, db)
    trades = load_trades(bt)
    return trades.to_dict(orient="records")


@router.get("/{backtest_id}/trades/parquet")
async def download_trades_parquet(
    backtest_id: uuid.UUID, db: AsyncSession = Depends(get_db)
):
    bt = await _require_completed(backtest_id, db)
    trades = load_trades(bt)
    buf = io.BytesIO()
    trades.to_parquet(buf, index=False)
    buf.seek(0)
    return StreamingResponse(
        buf,
        media_type="application/octet-stream",
        headers={"Content-Disposition": f"attachment; filename=trades_{backtest_id}.parquet"},
    )


# ── Comparison ────────────────────────────────────────────────────────────

@router.post("/compare", response_model=BacktestCompareResponse)
async def compare_backtests(
    payload: BacktestCompareRequest, db: AsyncSession = Depends(get_db)
):
    result = await db.execute(
        select(Backtest).where(Backtest.id.in_(payload.backtest_ids))
    )
    backtests = list(result.scalars().all())

    missing = set(payload.backtest_ids) - {bt.id for bt in backtests}
    if missing:
        raise HTTPException(status_code=404, detail=f"Backtest(s) not found: {missing}")

    runs = [
        BacktestCompareRow(
            id=bt.id, engine=bt.engine, status=bt.status, metrics=bt.metrics or {}
        )
        for bt in backtests
    ]

    all_keys = sorted({k for r in runs for k in r.metrics if k != "engine_stats"})
    metric_diff: dict[str, MetricDiff] = {}

    for key in all_keys:
        vals = {str(r.id): r.metrics.get(key) for r in runs}
        numeric = {eid: v for eid, v in vals.items() if isinstance(v, (int, float))}
        higher = key in _HIGHER_IS_BETTER
        best_id = (
            max(numeric, key=lambda eid: numeric[eid])
            if higher
            else min(numeric, key=lambda eid: numeric[eid])
        ) if numeric else None

        metric_diff[key] = MetricDiff(
            values=vals, best_id=best_id, higher_is_better=higher
        )

    return BacktestCompareResponse(runs=runs, metric_diff=metric_diff)


# ── Engine listing ────────────────────────────────────────────────────────

@router.get("/engines/available")
async def list_available_engines():
    return {"engines": list_engines()}


# ── Helpers ───────────────────────────────────────────────────────────────

def _enrich_response(bt: Backtest) -> BacktestResultResponse:
    resp = BacktestResultResponse.model_validate(bt)
    if bt.status == BacktestStatus.COMPLETED and bt.metrics:
        m = bt.metrics
        try:
            resp.structured_metrics = BacktestMetricsSummary(
                total_return=m.get("total_return", 0.0),
                bars_in_market=m.get("bars_in_market", 0),
                performance=PerformanceMetricsSchema(
                    cagr=m.get("perf_cagr", 0.0),
                    sharpe_ratio=m.get("perf_sharpe_ratio", 0.0),
                    sortino_ratio=m.get("perf_sortino_ratio", 0.0),
                    calmar_ratio=m.get("perf_calmar_ratio", 0.0),
                ),
                risk=RiskMetricsSchema(
                    max_drawdown=m.get("risk_max_drawdown", 0.0),
                    max_drawdown_duration=m.get("risk_max_drawdown_duration", 0),
                    var_95=m.get("risk_var_95", 0.0),
                    cvar_95=m.get("risk_cvar_95", 0.0),
                    var_99=m.get("risk_var_99", 0.0),
                    cvar_99=m.get("risk_cvar_99", 0.0),
                    volatility_annualised=m.get("risk_volatility_annualised", 0.0),
                ),
                trading=TradingMetricsSchema(
                    total_trades=m.get("trade_total_trades", 0),
                    win_rate=m.get("trade_win_rate", 0.0),
                    profit_factor=m.get("trade_profit_factor", 0.0),
                    avg_win=m.get("trade_avg_win", 0.0),
                    avg_loss=m.get("trade_avg_loss", 0.0),
                    expectancy=m.get("trade_expectancy", 0.0),
                    turnover_annualised=m.get("trade_turnover_annualised", 0.0),
                ),
            )
        except Exception:
            pass
    return resp


async def _require_completed(backtest_id: uuid.UUID, db: AsyncSession) -> Backtest:
    bt = await _repo.get(db, backtest_id)
    if not bt:
        raise HTTPException(status_code=404, detail="Backtest not found")
    if bt.status != BacktestStatus.COMPLETED:
        raise HTTPException(
            status_code=409,
            detail=f"Backtest not completed (status: {bt.status}). Run /execute first.",
        )
    if not bt.equity_curve_uri or not bt.trades_uri:
        raise HTTPException(
            status_code=409, detail="Result artifacts missing — backtest may have failed"
        )
    return bt
