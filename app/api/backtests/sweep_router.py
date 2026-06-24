"""
Parameter sweep endpoint.

Dispatches a grid of backtest configs to the Celery worker pool and
returns a task_id. Poll GET /api/v1/tasks/{task_id} for the ranked
leaderboard once all runs complete.
"""
from fastapi import APIRouter
from pydantic import BaseModel, Field

router = APIRouter(prefix="/backtests/sweep", tags=["backtests"])


class SweepRequest(BaseModel):
    strategy_id: str
    engine: str = "vectorbt"
    base_config: dict = Field(..., description="Base BacktestRunConfig dict")
    param_grid: list[dict] = Field(
        ...,
        description="List of override dicts merged into base_config per run",
        min_length=1,
        max_length=50,
    )
    rank_metric: str = "perf_sharpe_ratio"


@router.post("")
async def run_sweep(payload: SweepRequest):
    """Kick off a parameter sweep. Returns a Celery task_id to poll."""
    from app.workers.sweep_tasks import parameter_sweep_task

    task = parameter_sweep_task.delay(
        strategy_id=payload.strategy_id,
        engine=payload.engine,
        base_config=payload.base_config,
        param_grid=payload.param_grid,
        rank_metric=payload.rank_metric,
    )
    return {
        "task_id": task.id,
        "status": "PENDING",
        "n_configs": len(payload.param_grid),
    }
