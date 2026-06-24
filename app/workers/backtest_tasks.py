"""
Async backtest execution task.

Long backtests (multi-year, large universes, Backtrader with many
indicators) run as Celery background tasks so the API returns immediately.
Poll GET /api/v1/tasks/{task_id} for status, then fetch results from
GET /api/v1/backtests/{id}.
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


@celery_app.task(bind=True, name="backtests.execute")
def execute_backtest_task(self, backtest_id: str):
    async def _inner():
        from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

        from app.core.config import get_settings
        from app.db.crud_base import CRUDRepository
        from app.domain.backtest.orm import Backtest
        from app.domain.backtest.schemas import BacktestCreate, BacktestUpdate
        from app.engines.backtest_engine.pipeline import BacktestPipeline

        settings = get_settings()
        engine = create_async_engine(settings.DATABASE_URL)
        Session = async_sessionmaker(engine, expire_on_commit=False)

        async with Session() as db:
            repo = CRUDRepository[Backtest, BacktestCreate, BacktestUpdate](Backtest)
            bt = await repo.get(db, uuid.UUID(backtest_id))
            if bt is None:
                raise ValueError(f"Backtest {backtest_id} not found")

            pipeline = BacktestPipeline()
            updated_bt, result = await pipeline.execute(db, bt)

            return {
                "backtest_id": backtest_id,
                "status": updated_bt.status,
                "total_return": updated_bt.metrics.get("total_return"),
                "sharpe_ratio": updated_bt.metrics.get("perf_sharpe_ratio"),
                "max_drawdown": updated_bt.metrics.get("risk_max_drawdown"),
            }

    self.update_state(state="STARTED", meta={"backtest_id": backtest_id})
    return _run_async(_inner())
