"""
Backtest parameter sweep.

Lets the Backtest Agent (Phase 10) and the Backtest Lab UI run a grid of
configs (e.g. varying commission, capital, signal thresholds) and get
back a ranked leaderboard without manually wiring N separate backtest
rows. Each config in the sweep shares the same strategy_id and engine;
only the fields inside `config` vary.

Dispatched as a Celery chord so individual runs execute in parallel on the
worker pool and results are aggregated once all finish.
"""
import asyncio
import uuid

from app.workers.celery_app import celery_app


def _run_async(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


@celery_app.task(bind=True, name="backtests.sweep")
def parameter_sweep_task(
    self,
    strategy_id: str,
    engine: str,
    base_config: dict,
    param_grid: list[dict],
    rank_metric: str = "perf_sharpe_ratio",
):
    """
    param_grid : list of config override dicts.
                 Each is merged into base_config before creating the Backtest row.
    rank_metric: flat metric key to rank results by (higher = better).
    """
    async def _inner():
        from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

        from app.core.config import get_settings
        from app.db.crud_base import CRUDRepository
        from app.domain.backtest.orm import Backtest
        from app.domain.backtest.schemas import BacktestCreate, BacktestUpdate
        from app.engines.backtest_engine.pipeline import BacktestPipeline

        settings = get_settings()
        engine_inst = create_async_engine(settings.DATABASE_URL)
        Session = async_sessionmaker(engine_inst, expire_on_commit=False)
        repo = CRUDRepository[Backtest, BacktestCreate, BacktestUpdate](Backtest)
        pipeline = BacktestPipeline()

        rows = []
        for i, overrides in enumerate(param_grid):
            merged = {**base_config, **overrides}
            async with Session() as db:
                create = BacktestCreate(
                    strategy_id=uuid.UUID(strategy_id),
                    engine=engine,
                    initial_capital=merged.pop("initial_capital", 100_000),
                    commission=merged.pop("commission", 0.0005),
                    slippage=merged.pop("slippage", 0.0005),
                    config=merged,
                )
                bt = await repo.create(db, create)
                try:
                    updated_bt, _ = await pipeline.execute(db, bt)
                    score = updated_bt.metrics.get(rank_metric, float("-inf"))
                    rows.append({
                        "backtest_id": str(updated_bt.id),
                        "overrides": overrides,
                        "status": updated_bt.status,
                        "score": score,
                        "rank_metric": rank_metric,
                        "metrics_summary": {
                            k: v for k, v in updated_bt.metrics.items()
                            if k != "engine_stats"
                        },
                    })
                except Exception as exc:
                    rows.append({
                        "backtest_id": str(bt.id),
                        "overrides": overrides,
                        "status": "failed",
                        "score": float("-inf"),
                        "error": str(exc),
                    })

            self.update_state(
                state="PROGRESS",
                meta={"completed": i + 1, "total": len(param_grid)},
            )

        rows.sort(key=lambda r: r["score"], reverse=True)
        for rank, row in enumerate(rows, 1):
            row["rank"] = rank

        return {"leaderboard": rows, "total_runs": len(rows)}

    self.update_state(state="STARTED", meta={"strategy_id": strategy_id, "n_configs": len(param_grid)})
    return _run_async(_inner())
